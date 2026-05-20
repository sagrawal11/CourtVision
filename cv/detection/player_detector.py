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
    
    def __init__(self, model_path: Optional[Path] = None, device: Optional[str] = None):
        """Initialize YOLO human detector."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Ultralytics package not installed. Install with `pip install ultralytics`.")
        
        # If no explicit path is given or the file doesn't exist, fallback to standard YOLOv8n.
        # This will auto-download from Ultralytics if not present.
        # We only need it for human bounding boxes (class=0), as the ball is handled by TrackNet.
        model_str = str(model_path) if (model_path and model_path.exists()) else "yolov8n.pt"
        
        try:
            self.model = YOLO(model_str)
        except Exception as e:
            print(f"⚠️ Failed to load YOLO model {model_str}: {e}")
            raise e
            
        if device and device != "cpu":
            self.model.to(device)
        
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
    
    def detect_players(
        self,
        img: np.ndarray,
        bbox_thr: float = 0.35,
        nms_thr:  float = 0.3,
    ) -> list:
        """Detect people in the frame and return them as [(bbox_tuple, conf), …].

        bbox_tuple is (x1, y1, x2, y2) in pixel coords; conf is the YOLO
        confidence in [0, 1]. This is the format the rest of the pipeline
        (e.g. cv/pipeline.py) expects.

        Use this for downstream code that needs confidence values.
        Use run_human_detection() if you only need bare bboxes.
        """
        results = self.model.predict(img, verbose=False, conf=max(0.1, bbox_thr * 0.8))
        if not results:
            return []

        result = results[0]
        if not hasattr(result, "boxes") or result.boxes is None:
            return []

        # Collect (bbox, conf) per person detection above threshold
        candidates: list = []
        for box in result.boxes:
            cls  = int(box.cls.item())
            conf = float(box.conf.item())
            if cls in self.person_class_ids and conf >= bbox_thr:
                xyxy = box.xyxy.cpu().numpy().astype(np.float32)[0]
                candidates.append(((float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])), conf))

        if len(candidates) <= 1:
            return candidates

        # Apply NMS — same algorithm as run_human_detection, but keep confs aligned
        bboxes = np.array([c[0] for c in candidates], dtype=np.float32)
        keep_idx = self._nms_indices(bboxes, nms_thr)
        return [candidates[i] for i in keep_idx]

    def _nms_indices(self, boxes: np.ndarray, iou_threshold: float) -> list:
        """Return the indices of boxes to keep after NMS (preserves caller's order info)."""
        if len(boxes) == 0:
            return []
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        order = np.argsort(areas)[::-1].tolist()
        keep: list = []
        while order:
            i = order.pop(0)
            keep.append(i)
            if not order:
                break
            cur = boxes[i]
            others = boxes[order]
            x1 = np.maximum(cur[0], others[:, 0])
            y1 = np.maximum(cur[1], others[:, 1])
            x2 = np.minimum(cur[2], others[:, 2])
            y2 = np.minimum(cur[3], others[:, 3])
            inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            union = areas[i] + areas[order] - inter
            iou = inter / (union + 1e-6)
            order = [order[k] for k, v in enumerate(iou) if v < iou_threshold]
        return keep

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
        
        # Apply NMS if we have multiple boxes
        if len(boxes_array) > 1:
            boxes_array = self._apply_nms(boxes_array, nms_thr)
        
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
