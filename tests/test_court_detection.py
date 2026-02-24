"""
Test court detection by producing an annotated video using the CourtDetector.

Usage:
    python tests/test_court_detection.py --video path/to/video.mp4
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cv.detection.court_detector import CourtDetector

def annotate_frame(frame: np.ndarray, result) -> np.ndarray:
    annotated = frame.copy()
    
    # Check if result is debug dict or list
    if isinstance(result, dict):
        keypoints = result["blended"]
        native = result["native"]
        homography = result["homography"]
        heatmaps = result["heatmaps"]
    else:
        keypoints = result
        native = []
        homography = []
        heatmaps = []
        
    # 1. Composite heatmaps onto the frame if available
    if heatmaps:
        heatmap_overlay = np.zeros_like(annotated)
        for hm in heatmaps:
            hm_resized = cv2.resize(hm, (frame.shape[1], frame.shape[0]))
            colored = cv2.applyColorMap(hm_resized, cv2.COLORMAP_JET)
            mask = hm_resized > 50
            heatmap_overlay[mask] = colored[mask]
        annotated = cv2.addWeighted(annotated, 0.7, heatmap_overlay, 0.4, 0)

    # 2. Draw error vectors between native and homography points
    if native and homography and len(native) == len(homography):
        for orig, trans in zip(native, homography):
            if orig[0] is not None and trans[0] is not None:
                orig_pt = (int(orig[0]), int(orig[1]))
                trans_pt = (int(trans[0]), int(trans[1]))
                # Draw yellow line for error vector
                cv2.line(annotated, orig_pt, trans_pt, (0, 255, 255), 2)
                # Native point as red
                cv2.circle(annotated, orig_pt, 6, (0, 0, 255), -1)
                # Homography geometric point as blue
                cv2.circle(annotated, trans_pt, 6, (255, 0, 0), -1)

    # 3. Draw final blended points
    if keypoints:
        for i, kp in enumerate(keypoints):
            if kp is not None and kp[0] is not None and kp[1] is not None:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(annotated, (x, y), radius=12, color=(0, 255, 0), thickness=-1)
                cv2.circle(annotated, (x, y), radius=15, color=(255, 255, 255), thickness=3)
                cv2.putText(annotated, str(i), (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
        valid_points = len([k for k in keypoints if k is not None and k[0] is not None])
        cv2.putText(annotated, f"Court Keypoints: {valid_points}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    return annotated

def process_video(video_path: Path):
    OUTPUT_BASE = PROJECT_ROOT / "outputs" / "videos" / "court_model_trials"
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    
    # Optional parameters can be modified directly here to test different configs
    detector = CourtDetector()
    
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"Failed to open {video_path}")
        return
        
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    output_path = OUTPUT_BASE / f"{video_path.stem}_court_detection.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_count = 0
    keypoints = None
    
    print(f"[INFO] Detecting court lines in {video_path}...")
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        
        # Only process on the first frame since the camera is static
        if keypoints is None:
            try:
                # Use debug mode to gather heatmaps and separated native/homography vectors
                keypoints = detector.detect_court_in_frame(frame, debug=True)
            except Exception as e:
                print(f"[ERROR] Detection failed on frame {frame_count}: {e}")
                keypoints = None
            
        annotated = annotate_frame(frame, keypoints)
        writer.write(annotated)
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"[INFO] Processed {frame_count} frames")

    capture.release()
    writer.release()
    print(f"[INFO] Saved {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    args = parser.parse_args()
    process_video(args.video)
