"""
Compare ball-detection models by producing annotated videos for manual review.

The script loads a test video, runs each configured detector individually, and
evaluates every possible ensemble combination. It saves one output video per
strategy so you can watch the overlays and decide which detector (or fusion) is
best suited for production.

Usage:
    python tests/ball_model_comparison.py --video path/to/video.mp4
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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


class YOLOBallDetector(BallDetector):
    """YOLO-based detector using playersnball variants."""

    def __init__(self, model_filename: str, name: str) -> None:
        model_path = MODELS_DIR / "player" / model_filename
        if not model_path.exists():
            raise FileNotFoundError(f"Missing YOLO model: {model_path}")

        self.name = name

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics package not installed. Install with `pip install ultralytics`."
            ) from exc

        self.model = YOLO(str(model_path))
        # Class labels are stored in model.names; prefer dynamic lookup.
        self.ball_class_ids = {
            idx
            for idx, class_name in getattr(self.model, "names", {}).items()
            if str(class_name).lower() in {"ball", "tennis_ball"}
        }
        if not self.ball_class_ids:
            # Fallback to the class index used during training (0 or 1)
            self.ball_class_ids = {0, 1}

    def detect(self, frame: np.ndarray) -> Optional[BallDetectionResult]:
        results = self.model.predict(frame, verbose=False)
        if not results:
            return None

        result = results[0]
        if not hasattr(result, "boxes") or result.boxes is None:
            return None

        best: Optional[BallDetectionResult] = None
        for box in result.boxes:
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            if cls not in self.ball_class_ids:
                continue

            xyxy = box.xyxy.cpu().numpy().astype(int).tolist()[0]
            x1, y1, x2, y2 = xyxy
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            candidate = BallDetectionResult(center=center, confidence=conf, source=self.name)
            if best is None or candidate.confidence > best.confidence:
                best = candidate

        return best


class RFDETRBallDetector(BallDetector):
    """RF-DETR-based detector that reuses the multitask model."""

    name = "rf_detr_ball"

    def __init__(self) -> None:
        model_path = MODELS_DIR / "player" / "playersnball5.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing RF-DETR model: {model_path}")

        rfdetr_root = PROJECT_ROOT / "old" / "external" / "rf-detr"
        if not rfdetr_root.exists():
            raise FileNotFoundError("RF-DETR source not found under old/external/rf-detr.")
        sys.path.insert(0, str(rfdetr_root))

        try:
            import torch
            from rfdetr import RFDETRNano  # type: ignore[import]
        except Exception as exc:
            raise RuntimeError(
                "RF-DETR dependencies missing. Install project requirements and ensure "
                "old/external/rf-detr is intact."
            ) from exc

        self._torch = torch
        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to load RF-DETR checkpoint: {exc}") from exc
        if "args" not in checkpoint or "model" not in checkpoint:
            raise RuntimeError("Invalid RF-DETR checkpoint format.")

        args = checkpoint["args"]
        class_names = getattr(args, "class_names", ["ball", "player"])
        try:
            self.model = RFDETRNano(num_classes=len(class_names), pretrain_weights=None)
            self.model.model.model.load_state_dict(checkpoint["model"], strict=False)
            self.model.model.model.eval()
            try:
                self.model.class_names = class_names
            except Exception:
                pass
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize RF-DETR model: {exc}") from exc

        device = "cpu"
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        try:
            self.model.model.model.to(device)
        except Exception:
            device = "cpu"
            self.model.model.model.to(device)
        self.device = device

        self.class_id_ball = None
        for idx, name in enumerate(class_names):
            if name.lower() == "ball":
                self.class_id_ball = idx
                break

        if self.class_id_ball is None:
            raise RuntimeError("RF-DETR class names do not include 'ball'.")

    def detect(self, frame: np.ndarray) -> Optional[BallDetectionResult]:
        from PIL import Image

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        detections = self.model.predict(pil_img, threshold=0.2)
        if not hasattr(detections, "xyxy") or detections.xyxy is None:
            return None

        best: Optional[BallDetectionResult] = None
        for bbox, conf, cls in zip(detections.xyxy, detections.confidence, detections.class_id):
            if int(cls) != self.class_id_ball:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            candidate = BallDetectionResult(center=center, confidence=float(conf), source=self.name)
            if best is None or candidate.confidence > best.confidence:
                best = candidate

        return best


class TrackNetPTDetector(BallDetector):
    """TrackNet detector using the PyTorch checkpoint."""

    name = "tracknet_pt"

    def __init__(self) -> None:
        weight_path = MODELS_DIR / "ball" / "pretrained_ball_detection.pt"
        if not weight_path.exists():
            raise FileNotFoundError(f"TrackNet PyTorch weights missing: {weight_path}")

        tracknet_root = PROJECT_ROOT / "old" / "external" / "TrackNet"
        if not tracknet_root.exists():
            raise FileNotFoundError("TrackNet sources not found under old/external/TrackNet.")
        sys.path.insert(0, str(tracknet_root))

        try:
            import torch
            from model import BallTrackerNet  # type: ignore[import]
            from general import postprocess  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "TrackNet dependencies missing. Install project requirements and ensure "
                "old/external/TrackNet is available."
            ) from exc

        self.torch = torch  # keep reference
        self.postprocess = postprocess
        self.device = "cpu"
        self.frame_buffer: List[np.ndarray] = []

        self.model = BallTrackerNet(out_channels=256).to(self.device)
        checkpoint = torch.load(weight_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint
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
        rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 for frame in resized]
        combined = np.concatenate(rgb_frames, axis=2)
        tensor = self.torch.from_numpy(combined).permute(2, 0, 1).unsqueeze(0).float()
        return tensor

    def _extract_position(
        self, output: np.ndarray, original_shape: Tuple[int, int, int]
    ) -> Optional[Tuple[int, int]]:
        x, y = self.postprocess(output)
        if x is None or y is None:
            return None

        orig_h, orig_w = original_shape[:2]
        scale_h = orig_h / 720.0  # TrackNet output resolution
        scale_w = orig_w / 1280.0
        x_scaled = int(max(0, min(orig_w - 1, x * scale_w)))
        y_scaled = int(max(0, min(orig_h - 1, y * scale_h)))
        return x_scaled, y_scaled


class TraceTrackNetDetector(BallDetector):
    """TRACE implementation of TrackNet using legacy code."""

    name = "trace_tracknet"

    def __init__(self) -> None:
        trace_root = PROJECT_ROOT / "old" / "external" / "TRACE"
        if not trace_root.exists():
            raise FileNotFoundError("TRACE sources not found under old/external/TRACE.")
        sys.path.insert(0, str(trace_root))

        weights_path = trace_root / "TrackNet" / "Weights.pth"
        if not weights_path.exists():
            raise FileNotFoundError(f"TRACE TrackNet weights missing: {weights_path}")

        try:
            from BallDetection import BallDetector as TraceBallDetector  # type: ignore[import]
        except Exception as exc:
            raise RuntimeError("Failed to import TRACE BallDetector from legacy sources.") from exc

        self.detector = TraceBallDetector(str(weights_path))

    def detect(self, frame: np.ndarray) -> Optional[BallDetectionResult]:
        self.detector.detect_ball(frame)
        if len(self.detector.xy_coordinates) == 0:
            return None
        x, y = self.detector.xy_coordinates[-1]
        if x is None or y is None:
            return None
        return BallDetectionResult(center=(int(x), int(y)), confidence=0.6, source=self.name)


def available_detectors() -> List[BallDetector]:
    detectors: List[BallDetector] = []

    yolo_variants = [
        ("playersnball4.pt", "yolo_playersnball4"),
        ("playersnball5.pt", "yolo_playersnball5"),
    ]
    for weight_name, detector_name in yolo_variants:
        try:
            detectors.append(YOLOBallDetector(weight_name, detector_name))
        except Exception as exc:
            print(f"[WARN] Skipping YOLO variant {detector_name}: {exc}")

    other_detector_classes = [
        RFDETRBallDetector,
        TrackNetPTDetector,
        TraceTrackNetDetector,
    ]
    for detector_cls in other_detector_classes:
        try:
            detectors.append(detector_cls())
        except Exception as exc:
            import traceback

            print(f"[WARN] Skipping {detector_cls.__name__}: {exc}")
            traceback.print_exc()

    return detectors


def enumerate_combinations(detectors: Sequence[BallDetector]) -> Iterable[Tuple[str, Sequence[BallDetector]]]:
    for r in range(1, len(detectors) + 1):
        for combo in itertools.combinations(detectors, r):
            name = "+".join(detector.name for detector in combo)
            yield name, combo


def combine_detections(detections: Sequence[Optional[BallDetectionResult]]) -> Optional[BallDetectionResult]:
    detections = [det for det in detections if det is not None]
    if not detections:
        return None
    return max(detections, key=lambda det: det.confidence)


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


def save_video_variants(video_path: Path, detector_sets: Iterable[Tuple[str, Sequence[BallDetector]]]) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    
    # Create subfolder for this video
    video_output_dir = OUTPUT_BASE / video_path.stem
    video_output_dir.mkdir(parents=True, exist_ok=True)

    frames: List[np.ndarray] = []
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frames.append(frame)
    capture.release()

    for label, detectors in detector_sets:
        if not detectors:
            continue

        output_path = video_output_dir / f"{video_path.stem}_{label}.mp4"
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        for detector in detectors:
            detector.warmup(frames[0])

        for frame in frames:
            detections = [detector.detect(frame) for detector in detectors]
            combined = combine_detections(detections)
            annotated = annotate_frame(frame, combined, label)
            writer.write(annotated)

        writer.release()
        print(f"[INFO] Saved {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ball detection models.")
    parser.add_argument("--video", required=True, type=Path, help="Path to test video.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detectors = available_detectors()
    if not detectors:
        raise RuntimeError("No ball detectors could be initialized.")

    combos = list(enumerate_combinations(detectors))
    print("[INFO] Comparing strategies:")
    for name, _ in combos:
        print(f"  - {name}")

    save_video_variants(args.video, combos)


if __name__ == "__main__":
    main()

