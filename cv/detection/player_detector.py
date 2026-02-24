"""
YOLO-based human detector wrapper for SAM-3d-body.
Provides bounding boxes from YOLO model to SAM-3d-body.
"""

import sys
from pathlib import Path
from typing import Optional
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


class PlayerDetector:
    """Wrapper to use YOLO model for human detection in SAM-3d-body."""
    
    def __init__(self, model_path: Path, device: Optional[str] = None):
        """Initialize YOLO human detector."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Ultralytics package not installed. Install with `pip install ultralytics`.")
        
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        
        # Try to load YOLO model with error handling for checkpoint format issues
        try:
            self.model = YOLO(str(model_path))
        except (AttributeError, TypeError) as e:
            error_str = str(e)
            if "'collections.OrderedDict' object has no attribute 'float'" in error_str or "'dict' object has no attribute 'float'" in error_str:
                # Checkpoint format issue - try workaround with attempt_load_weights
                print(f"⚠️ YOLO checkpoint format issue - trying workaround...")
                try:
                    from ultralytics.nn.tasks import attempt_load_weights
                    # attempt_load_weights can handle different checkpoint formats
                    self.model, ckpt = attempt_load_weights(str(model_path), device='cpu')
                    if device and device != 'cpu':
                        self.model = self.model.to(device)
                    print(f"   ✅ Loaded YOLO model using workaround")
                except Exception as e2:
                    print(f"⚠️ Workaround failed: {e2}")
                    print(f"   The YOLO model checkpoint format is incompatible with current ultralytics version")
                    print(f"   Trying to downgrade ultralytics...")
                    # Try downgrading ultralytics as a last resort
                    import subprocess
                    import sys
                    try:
                        result = subprocess.run(
                            [sys.executable, "-m", "pip", "install", "ultralytics==8.0.196"],
                            capture_output=True,
                            text=True,
                            timeout=60
                        )
                        if result.returncode == 0:
                            print(f"   ✅ Downgraded ultralytics, please restart runtime and try again")
                            print(f"   Runtime → Restart runtime, then re-run Step 5")
                            raise ImportError("Ultralytics downgraded - restart runtime required")
                        else:
                            raise e
                    except Exception as e3:
                        print(f"   ⚠️ Could not downgrade: {e3}")
                        print(f"   Options:")
                        print(f"   1. Manually run: pip install 'ultralytics<8.3.0' then restart runtime")
                        print(f"   2. Retrain/convert the model to current YOLO format")
                        print(f"   3. Continue without YOLO (will only detect 1 person)")
                        raise e  # Re-raise to let caller handle it
            else:
                raise e
        
        # Find player/person class IDs
        self.person_class_ids = set()
        for idx, class_name in getattr(self.model, "names", {}).items():
            name_lower = str(class_name).lower()
            if name_lower in {"person", "player", "human"}:
                self.person_class_ids.add(idx)
        
        # If no explicit person class, assume class 0 or 1 might be person (common in tennis models)
        if not self.person_class_ids:
            self.person_class_ids = {0, 1}  # Common for tennis player models
        
        print(f"YOLO human detector initialized: {model_path}")
        print(f"  Person class IDs: {self.person_class_ids}")
    
    def run_human_detection(
        self,
        img: np.ndarray,
        det_cat_id: int = 0,
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
        default_to_full_image: bool = False,
    ) -> np.ndarray:
        """
        Detect humans in image and return bounding boxes in format expected by SAM-3d-body.
        
        Returns:
            np.ndarray of shape (N, 4) where each row is [x1, y1, x2, y2]
        """
        # Run YOLO inference with lower confidence to catch more people
        # Use even lower conf for YOLO since we'll filter by bbox_thr later
        yolo_conf = max(0.1, bbox_thr * 0.8)  # Use 80% of bbox_thr for YOLO, min 0.1
        results = self.model.predict(img, verbose=False, conf=yolo_conf)
        
        if not results:
            if default_to_full_image:
                h, w = img.shape[:2]
                return np.array([[0, 0, w, h]], dtype=np.float32)
            return np.array([], dtype=np.float32).reshape(0, 4)
        
        result = results[0]
        if not hasattr(result, "boxes") or result.boxes is None:
            if default_to_full_image:
                h, w = img.shape[:2]
                return np.array([[0, 0, w, h]], dtype=np.float32)
            return np.array([], dtype=np.float32).reshape(0, 4)
        
        # Extract bounding boxes for person/player classes
        boxes = []
        for box in result.boxes:
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            
            # Check if this is a person/player detection
            if cls in self.person_class_ids and conf >= bbox_thr:
                xyxy = box.xyxy.cpu().numpy().astype(np.float32)[0]
                boxes.append(xyxy)
        
        if not boxes:
            if default_to_full_image:
                h, w = img.shape[:2]
                return np.array([[0, 0, w, h]], dtype=np.float32)
            return np.array([], dtype=np.float32).reshape(0, 4)
        
        boxes_array = np.array(boxes, dtype=np.float32)
        
        print(f"[DEBUG YOLO] Found {len(boxes_array)} person detections before NMS (threshold={bbox_thr})")
        
        # Apply NMS if we have multiple boxes
        if len(boxes_array) > 1:
            boxes_before_nms = len(boxes_array)
            boxes_array = self._apply_nms(boxes_array, nms_thr)
            print(f"[DEBUG YOLO] After NMS (threshold={nms_thr}): {len(boxes_array)} detections (removed {boxes_before_nms - len(boxes_array)})")
        
        return boxes_array
    
    def _apply_nms(self, boxes: np.ndarray, iou_threshold: float) -> np.ndarray:
        """Simple NMS implementation."""
        if len(boxes) == 0:
            return boxes
        
        # Calculate areas
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Sort by area (largest first)
        indices = np.argsort(areas)[::-1]
        
        keep = []
        while len(indices) > 0:
            # Take the largest box
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            # Calculate IoU with remaining boxes
            current_box = boxes[current]
            other_boxes = boxes[indices[1:]]
            
            # Intersection
            x1 = np.maximum(current_box[0], other_boxes[:, 0])
            y1 = np.maximum(current_box[1], other_boxes[:, 1])
            x2 = np.minimum(current_box[2], other_boxes[:, 2])
            y2 = np.minimum(current_box[3], other_boxes[:, 3])
            
            intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            union = areas[current] + areas[indices[1:]] - intersection
            iou = intersection / (union + 1e-6)
            
            # Keep boxes with IoU < threshold
            indices = indices[1:][iou < iou_threshold]
        
        return boxes[keep]
