import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple, Optional
import logging
from scipy.spatial import distance

# Import BallTrackerNet from ball_tracker (same architecture architecture used for court detection)
from .ball_tracker import BallTrackerNet

logger = logging.getLogger('cv.detection.court_detector')

# ---------------------------------------------------------------------------
# Inlined dependencies from upstream yastrebksv/TennisCourtDetector
# ---------------------------------------------------------------------------

class CourtReference:
    """Court reference model for homography transformation"""
    def __init__(self):
        self.baseline_top = ((286, 561), (1379, 561))
        self.baseline_bottom = ((286, 2935), (1379, 2935))
        self.net = ((286, 1748), (1379, 1748))
        self.left_court_line = ((286, 561), (286, 2935))
        self.right_court_line = ((1379, 561), (1379, 2935))
        self.left_inner_line = ((423, 561), (423, 2935))
        self.right_inner_line = ((1242, 561), (1242, 2935))
        self.middle_line = ((832, 1110), (832, 2386))
        self.top_inner_line = ((423, 1110), (1242, 1110))
        self.bottom_inner_line = ((423, 2386), (1242, 2386))

        self.key_points = [*self.baseline_top, *self.baseline_bottom, 
                          *self.left_inner_line, *self.right_inner_line,
                          *self.top_inner_line, *self.bottom_inner_line,
                          *self.middle_line]

        # 12 specific configurations for homography
        self.court_conf = {
            1: [*self.baseline_top, *self.baseline_bottom],
            2: [self.left_inner_line[0], self.right_inner_line[0], self.left_inner_line[1], self.right_inner_line[1]],
            3: [self.left_inner_line[0], self.right_court_line[0], self.left_inner_line[1], self.right_court_line[1]],
            4: [self.left_court_line[0], self.right_inner_line[0], self.left_court_line[1], self.right_inner_line[1]],
            5: [*self.top_inner_line, *self.bottom_inner_line],
            6: [*self.top_inner_line, self.left_inner_line[1], self.right_inner_line[1]],
            7: [self.left_inner_line[0], self.right_inner_line[0], *self.bottom_inner_line],
            8: [self.right_inner_line[0], self.right_court_line[0], self.right_inner_line[1], self.right_court_line[1]],
            9: [self.left_court_line[0], self.left_inner_line[0], self.left_court_line[1], self.left_inner_line[1]],
            10: [self.top_inner_line[0], self.middle_line[0], self.bottom_inner_line[0], self.middle_line[1]],
            11: [self.middle_line[0], self.top_inner_line[1], self.middle_line[1], self.bottom_inner_line[1]],
            12: [*self.bottom_inner_line, self.left_inner_line[1], self.right_inner_line[1]]
        }

court_ref = CourtReference()
refer_kps = np.array(court_ref.key_points, dtype=np.float32).reshape((-1, 1, 2))

# ── Court reference extent constants ──────────────────────────────────────────
# Used by pipeline._build_homography and court_zones.py to normalise raw
# reference-space coordinates to the 0-1 range that the rest of the system uses.
REF_X_MIN = 286    # Left doubles sideline in reference pixel space
REF_X_MAX = 1379   # Right doubles sideline
REF_Y_MIN = 561    # Far (top) baseline
REF_Y_MAX = 2935   # Near (bottom) baseline

court_conf_ind = {}
for i in range(len(court_ref.court_conf)):
    conf = court_ref.court_conf[i+1]
    inds = []
    for j in range(4):
        inds.append(court_ref.key_points.index(conf[j]))
    court_conf_ind[i+1] = inds

def get_trans_matrix(points):
    """Finds best homography matrix using RANSAC over all available detected points"""
    src_pts = []
    dst_pts = []
    
    # Collect all valid point pairs between reference court and detected points
    for i in range(14):
        if points[i][0] is not None and points[i][1] is not None:
            src_pts.append(court_ref.key_points[i])
            dst_pts.append(points[i])
            
    if len(src_pts) >= 4:
        # Use RANSAC to find the best mathematically sound plane that agrees across all points, 
        # discounting any wild false-positive blobs. Reproj threshold of 50px accounts for lens curve
        src_pts = np.float32(src_pts)
        dst_pts = np.float32(dst_pts)
        matrix, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=50.0)
        return matrix
        
    return None

def postprocess(heatmap, scale=2, low_thresh=170, min_radius=2, max_radius=25):
    """Extracts keypoint coordinate from a heatmap using HoughCircles"""
    x_pred, y_pred = None, None
    ret, heatmap_thresh = cv2.threshold(heatmap, low_thresh, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(heatmap_thresh, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=2, minRadius=min_radius, maxRadius=max_radius)
    if circles is not None:
        x_pred = circles[0][0][0] * scale
        y_pred = circles[0][0][1] * scale
    return x_pred, y_pred

# ---------------------------------------------------------------------------
# Main Detector Class
# ---------------------------------------------------------------------------

class CourtDetector:
    """
    Tennis court keypoint detector based on TrackNet architecture.
    Detects 14 key points on the court (corners and line intersections).

    REFERENCE_KEYPOINTS is the list of (x, y) anchor positions in the canonical
    reference pixel space (x: 286–1379, y: 561–2935).  These are the ground
    truths that the RANSAC homography is fitted towards.  pipeline.py imports
    them to build its own video-frame → court-space homography, and then
    normalises the output using REF_X_MIN/MAX and REF_Y_MIN/MAX.
    """
    REFERENCE_KEYPOINTS: list[tuple[int, int]] = court_ref.key_points
    def __init__(self, model_path: Optional[Path] = None, device: Optional[str] = None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        else:
            self.device = device
            
        # Get project root depending on module hierarchy
        project_root = Path(__file__).resolve().parents[2]
        
        if model_path is None:
            model_path = project_root / "models" / "court" / "model_tennis_court_det.pt"
            
        self.model_path = model_path
        self.input_width = 640
        self.input_height = 360
        self.model = None
        self._init_model()
        
    def _init_model(self):
        if not self.model_path.exists():
            logger.error(f"Court detection model not found at {self.model_path}")
            return
            
        try:
            # Reusing BallTrackerNet heavily, but court detection uses 3 input channels (single frame) and 15 output maps
            self.model = BallTrackerNet(in_channels=3, out_channels=15)
            
            torch.serialization.add_safe_globals([np.dtype, np.generic])
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Extract state dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Some checkpoints have 'module.' prefix from DataParallel
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded court detection model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load court model: {e}")
            self.model = None

    def detect_court_in_frame(self, frame: np.ndarray, apply_homography: bool = True, debug: bool = False):
        """Backward compatibility alias for detect()"""
        return self.detect(frame, apply_homography=apply_homography, debug=debug)

    def detect(self, frame: np.ndarray, apply_homography: bool = True, debug: bool = False):
        """
        Detects 14 court keypoints in the frame.
        Applies homography to reconstruct missing points if apply_homography=True.
        """
        if self.model is None:
            return [(None, None)] * 14
            
        orig_height, orig_width = frame.shape[:2]
        
        # 1. Resize and normalize
        img_resized = cv2.resize(frame, (self.input_width, self.input_height))
        inp = (img_resized.astype(np.float32) / 255.)
        inp = torch.tensor(inp).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # 2. Forward pass
        with torch.no_grad():
            from torch.nn.functional import sigmoid
            out = self.model(inp.float())
            if isinstance(out, tuple):
                out = out[0]
            out = out[0]  # Remove batch dim
            
            # Reshape back to spatial dimensions (BallTrackerNet flattens it for its own use case)
            out = out.reshape(15, self.input_height, self.input_width)
            
            pred = sigmoid(out).detach().cpu().numpy()
            
        # 3. Postprocess heatmaps
        points = []
        heatmaps_raw = []
        for kps_num in range(14):
            heatmap = (pred[kps_num] * 255).astype(np.uint8)
            heatmaps_raw.append(heatmap)
            x_pred, y_pred = postprocess(heatmap, low_thresh=170, min_radius=2, max_radius=25)
            
            if x_pred is not None and y_pred is not None:
                # Scale back to original video dims
                scale_x = orig_width / (self.input_width * 2)
                scale_y = orig_height / (self.input_height * 2)
                points.append((int(x_pred * scale_x), int(y_pred * scale_y)))
            else:
                points.append((None, None))
                
        native_points = points.copy()
        transformed_points = []
        
        # 4. Homography to reconstruct missing/noisy points
        if apply_homography:
            try:
                matrix_trans = get_trans_matrix(points)
                if matrix_trans is not None:
                    transformed = cv2.perspectiveTransform(refer_kps, matrix_trans)
                    transformed = [np.squeeze(x) for x in transformed]
                    transformed_points = [tuple(map(int, pt)) for pt in transformed[:14]]
                    # Blend homography and native predictions:
                    # - Native predictions track curved lines better (lens distortion)
                    # - Homography fixes wild false positives (e.g., far points predicted on near court)
                    refined_points = []
                    for orig, trans in zip(points, transformed[:14]):
                        if orig[0] is not None and orig[1] is not None:
                            dist = np.sqrt((orig[0] - trans[0])**2 + (orig[1] - trans[1])**2)
                            if dist < 20:  # 20px tight tolerance - trust RANSAC for most points, only keep native for minor lens corrections
                                refined_points.append(orig)
                            else:
                                logger.debug(f"Rejecting outlier point {orig} vs homography {trans}")
                                refined_points.append(tuple(map(int, trans)))
                        else:
                            refined_points.append(tuple(map(int, trans)))
                    points = refined_points
            except Exception as e:
                logger.debug(f"Homography adjustment failed: {e}")
                
        if debug:
            return {
                "blended": points,
                "native": native_points,
                "homography": transformed_points,
                "heatmaps": heatmaps_raw
            }
        return points
