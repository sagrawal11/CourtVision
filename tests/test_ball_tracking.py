"""
Test ball detection by producing an annotated video using the TrackNet detector.

The script loads a test video, runs the TrackNet detector, and saves an
annotated output video for manual review.

Usage:
    python tests/test_ball_tracking.py --video path/to/video.mp4
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_BASE = PROJECT_ROOT / "outputs" / "videos" / "ball_model_trials"

os.environ["TRANSFORMERS_NO_TF_IMPORT"] = "1"

_safe_globals = []
try:
    import numpy as _np

    _safe_globals.append(_np.core.multiarray.scalar)
    _safe_globals.append(_np.dtype)
except Exception:
    pass

try:
    import argparse as _argparse

    _safe_globals.append(_argparse.Namespace)
except Exception:
    pass

if hasattr(torch.serialization, "add_safe_globals") and _safe_globals:
    torch.serialization.add_safe_globals(_safe_globals)


@dataclass
class BallDetectionResult:
    """Standardized ball detection output."""

    center: Tuple[int, int]
    confidence: float
    source: str


class BallDetector:
    """Base interface for ball detectors."""

    name: str

    def detect(self, frame: np.ndarray) -> Optional[BallDetectionResult]:
        raise NotImplementedError

    def warmup(self, frame: np.ndarray) -> None:
        """Optional warmup hook for the detector."""
        _ = self.detect(frame)


class TrackNetPTDetector(BallDetector):
    """TrackNet detector using the PyTorch checkpoint."""

    name = "tracknet_pt"

    def __init__(self) -> None:
        weight_path = MODELS_DIR / "ball" / "pretrained_ball_detection.pt"
        if not weight_path.exists():
            raise FileNotFoundError(f"TrackNet PyTorch weights missing: {weight_path}")

        try:
            from cv.detection.ball_tracker import BallTrackerNet
        except ImportError as exc:
            raise RuntimeError(
                "Could not import BallTrackerNet from cv.detection.ball_tracker."
            ) from exc

        self.torch = torch
        self.frame_buffer: List[np.ndarray] = []
        self.device = "cpu"

        self.model = BallTrackerNet(out_channels=256).to(self.device)
        checkpoint = torch.load(weight_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state") or checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        self.input_width = 640
        self.input_height = 360

    def detect(self, frame: np.ndarray) -> Optional[BallDetectionResult]:
        self.frame_buffer.append(frame)
        if len(self.frame_buffer) < 3:
            return None
        if len(self.frame_buffer) > 3:
            self.frame_buffer.pop(0)

        frame1, frame2, frame3 = self.frame_buffer
        tensor = self._prepare_tensor(frame3, frame2, frame1)

        with self.torch.no_grad():
            output = self.model(tensor, testing=True)
            output = output.argmax(dim=1).detach().cpu().numpy()
            output *= 255

        position = self._extract_position(output[0], frame.shape)
        if position is None:
            return None

        return BallDetectionResult(center=position, confidence=0.8, source=self.name)

    def _prepare_tensor(
        self, frame1: np.ndarray, frame2: np.ndarray, frame3: np.ndarray
    ) -> Any:
        resized = [
            cv2.resize(frame, (self.input_width, self.input_height)) for frame in (frame1, frame2, frame3)
        ]
        imgs = np.concatenate(
            [cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 for f in resized],
            axis=2,
        )
        tensor = self.torch.from_numpy(np.rollaxis(imgs, 2, 0)).unsqueeze(0).float()
        return tensor

    def _extract_position(
        self, output: np.ndarray, original_shape: Tuple[int, int, int]
    ) -> Optional[Tuple[int, int]]:
        heatmap = output.reshape((self.input_height, self.input_width)).astype(np.uint8)
        _, binary = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)

        circles = cv2.HoughCircles(
            binary, cv2.HOUGH_GRADIENT, dp=1, minDist=1,
            param1=50, param2=8, minRadius=2, maxRadius=7,
        )
        if circles is None or len(circles) == 0:
            return None

        x = int(circles[0][0][0])
        y = int(circles[0][0][1])

        orig_h, orig_w = original_shape[:2]
        scale_w = orig_w / self.input_width
        scale_h = orig_h / self.input_height
        x_scaled = int(max(0, min(orig_w - 1, x * scale_w)))
        y_scaled = int(max(0, min(orig_h - 1, y * scale_h)))
        return x_scaled, y_scaled


def annotate_frame(frame: np.ndarray, detection: Optional[BallDetectionResult], label: str) -> np.ndarray:
    annotated = frame.copy()
    cv2.putText(
        annotated,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if detection:
        cx, cy = detection.center
        cv2.circle(annotated, (cx, cy), radius=12, color=(0, 140, 255), thickness=2)
        cv2.putText(
            annotated,
            f"{detection.source} ({detection.confidence:.2f})",
            (cx + 15, cy - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            annotated,
            "No detection",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return annotated


def save_annotated_video(video_path: Path, detector: BallDetector) -> None:
    """Run the detector on the video and save an annotated output."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    video_output_dir = OUTPUT_BASE / video_path.stem
    video_output_dir.mkdir(parents=True, exist_ok=True)

    output_path = video_output_dir / f"{video_path.stem}_{detector.name}.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    # Read all frames (limited to max_args if provided)
    frames: List[np.ndarray] = []
    max_frames = getattr(detector, "max_frames", None)
    
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    capture.release()

    # Warmup
    detector.warmup(frames[0])

    # Process
    for frame in frames:
        detection = detector.detect(frame)
        annotated = annotate_frame(frame, detection, detector.name)
        writer.write(annotated)

    writer.release()
    print(f"[INFO] Saved {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test TrackNet ball detection.")
    parser.add_argument("--video", required=True, type=Path, help="Path to test video.")
    parser.add_argument("--max-frames", type=int, help="Maximum number of frames to process.", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("[INFO] Initializing TrackNet detector...")
    detector = TrackNetPTDetector()
    detector.max_frames = args.max_frames
    
    print(f"[INFO] Running {detector.name} on {args.video}")
    save_annotated_video(args.video, detector)


if __name__ == "__main__":
    main()
