#!/usr/bin/env python3
"""
cv/tools/annotate.py — Interactive video annotation tool for tennis ML training data.

Keyboard controls
-----------------
  Navigation:
    SPACE / W     : play / pause
    A / ←         : back 1 frame
    D / →         : forward 1 frame
    , (comma)     : back 5 frames
    . (period)    : forward 5 frames
    [ (bracket)   : back 30 frames (~1 s)
    ] (bracket)   : forward 30 frames (~1 s)

  Annotation — press key at the exact frame of the event:
    B             : Bounce
    H             : Hit  →  then select stroke type:
                      F = Forehand
                      K = bacKhand
                      S = Serve
                      O = Overhead
                      V = Volley
    P             : Point start (serve contact)
    E             : point End  →  then select outcome:
                      W = Winner
                      U = Unforced error
                      N = Net error
                      F = Forced error

    X             : delete ALL annotations at current frame
    Z             : undo last annotation

  Misc:
    Q / ESC       : save and quit
    Click scrubber: seek to that position

Output CSV columns:
  video_id, frame, event_type, stroke_type, point_end_type, player

  player is auto-inferred during feature extraction (not annotated manually).
  stroke_type is only set for 'hit' events.
  point_end_type is only set for 'point_end' events.

Usage:
    python cv/tools/annotate.py --video tests/match1.mov
    python cv/tools/annotate.py --video tests/match1.mov --output cv/training_data/match1.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ── Display constants ─────────────────────────────────────────────────────────

MAX_DISP_W = 1280
MAX_DISP_H = 720
PANEL_H    = 120   # pixel height of the info / scrubber panel below the video

# ── BGR colour palette ────────────────────────────────────────────────────────

# fmt: off
EVENT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "bounce":               (220, 150,  50),   # steel blue
    "hit_forehand":         ( 50, 200,  50),   # green
    "hit_backhand":         ( 50,  50, 220),   # red
    "hit_serve":            ( 50, 220, 220),   # yellow
    "hit_overhead":         ( 50, 165, 255),   # orange
    "hit_volley":           (220, 220,  50),   # cyan
    "point_start":          ( 50, 255, 100),   # bright green
    "point_end_winner":     ( 50, 215, 255),   # gold
    "point_end_unforced_error": ( 50,  80, 255),  # red-orange
    "point_end_net_error":  (180,  80, 200),   # purple
    "point_end_forced_error":   (150, 100, 255),  # salmon
}
DEFAULT_COLOR: Tuple[int, int, int] = (200, 200, 200)
# fmt: on

# ── Key mappings ──────────────────────────────────────────────────────────────

STROKE_KEYS: Dict[int, str] = {
    ord("f"): "forehand",
    ord("k"): "backhand",
    ord("s"): "serve",
    ord("o"): "overhead",
    ord("v"): "volley",
}

END_KEYS: Dict[int, str] = {
    ord("w"): "winner",
    ord("u"): "unforced_error",
    ord("n"): "net_error",
    ord("f"): "forced_error",
}

# Cross-platform arrow key codes (cv2.waitKeyEx returns full int)
LEFT_ARROWS  = {2424832, 65361, 81}   # macOS, X11, other
RIGHT_ARROWS = {2555904, 65363, 83}

MODE_NORMAL       = "normal"
MODE_AWAIT_STROKE = "await_stroke"
MODE_AWAIT_END    = "await_end"


# ── Data model ────────────────────────────────────────────────────────────────

class Annotation:
    __slots__ = ("frame", "event_type", "stroke_type", "point_end_type", "player")

    def __init__(
        self,
        frame: int,
        event_type: str,
        stroke_type:     Optional[str] = None,
        point_end_type:  Optional[str] = None,
        player:          Optional[str] = None,
    ) -> None:
        self.frame          = frame
        self.event_type     = event_type
        self.stroke_type    = stroke_type
        self.point_end_type = point_end_type
        self.player         = player

    def _color_key(self) -> str:
        if self.event_type == "hit":
            return f"hit_{self.stroke_type or 'forehand'}"
        if self.event_type == "point_end":
            return f"point_end_{self.point_end_type or 'unforced_error'}"
        return self.event_type

    def color(self) -> Tuple[int, int, int]:
        return EVENT_COLORS.get(self._color_key(), DEFAULT_COLOR)

    def label(self) -> str:
        if self.event_type == "hit":
            return f"HIT: {self.stroke_type or '?'}"
        if self.event_type == "bounce":
            return "BOUNCE"
        if self.event_type == "point_start":
            return "POINT START"
        if self.event_type == "point_end":
            return f"POINT END: {(self.point_end_type or '?').replace('_', ' ')}"
        return self.event_type.upper()

    def to_dict(self, video_id: str) -> dict:
        return {
            "video_id":      video_id,
            "frame":         self.frame,
            "event_type":    self.event_type,
            "stroke_type":   self.stroke_type    or "",
            "point_end_type": self.point_end_type or "",
            "player":        self.player         or "",
        }


# ── Main annotator class ─────────────────────────────────────────────────────

class TennisAnnotator:
    WIN_NAME   = "Tennis Annotator"
    CSV_FIELDS = ["video_id", "frame", "event_type", "stroke_type", "point_end_type", "player"]

    def __init__(
        self,
        video_path: str,
        output_csv: str,
    ) -> None:
        self.video_path = video_path
        self.output_csv = Path(output_csv)
        self.video_id   = Path(video_path).stem

        # Open video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        self.total_frames = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.fps          = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.video_w      = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_h      = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Scale so the video display fits within MAX_DISP_W × MAX_DISP_H
        scale        = min(MAX_DISP_W / self.video_w, MAX_DISP_H / self.video_h, 1.0)
        self.disp_w  = int(self.video_w  * scale)
        self.disp_h  = int(self.video_h  * scale)
        self.scale   = scale

        # Annotation state
        self.annotations: List[Annotation] = []
        self._undo_stack: List[Optional[Annotation]] = []  # None = a deletion was undone
        self._load_existing()

        # Playback / UI state
        self.frame_idx   = 0
        self.playing     = False
        self.mode        = MODE_NORMAL
        self._pending_frame: int = 0  # frame captured when H / E was pressed

        # Single-frame cache to avoid redundant seeks while paused
        self._cached_idx:   int               = -1
        self._cached_frame: Optional[np.ndarray] = None

        # Mouse state
        self._scrubber_dragging = False

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load_existing(self) -> None:
        if not self.output_csv.exists():
            return
        with open(self.output_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    self.annotations.append(Annotation(
                        frame          = int(row["frame"]),
                        event_type     = row["event_type"],
                        stroke_type    = row.get("stroke_type")    or None,
                        point_end_type = row.get("point_end_type") or None,
                        player         = row.get("player")         or None,
                    ))
                except (ValueError, KeyError):
                    pass
        self.annotations.sort(key=lambda a: a.frame)
        print(f"Loaded {len(self.annotations)} existing annotations from {self.output_csv}")

    def _save(self) -> None:
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            writer.writeheader()
            for ann in sorted(self.annotations, key=lambda a: a.frame):
                writer.writerow(ann.to_dict(self.video_id))

    # ── Frame access ─────────────────────────────────────────────────────────

    def _get_frame(self, idx: int) -> np.ndarray:
        if self._cached_idx == idx and self._cached_frame is not None:
            return self._cached_frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ret, frame = self.cap.read()
        if not ret or frame is None:
            frame = np.zeros((self.video_h, self.video_w, 3), dtype=np.uint8)
        self._cached_idx   = idx
        self._cached_frame = frame
        return frame

    def _display_frame(self, idx: int) -> np.ndarray:
        frame = self._get_frame(idx)
        if self.scale < 1.0:
            frame = cv2.resize(frame, (self.disp_w, self.disp_h), interpolation=cv2.INTER_AREA)
        return frame.copy()

    # ── Annotation operations ─────────────────────────────────────────────────

    def _add(self, ann: Annotation) -> None:
        # Replace any existing annotation with the same frame + event_type
        self.annotations = [
            a for a in self.annotations
            if not (a.frame == ann.frame and a.event_type == ann.event_type)
        ]
        self.annotations.append(ann)
        self.annotations.sort(key=lambda a: a.frame)
        self._undo_stack.append(ann)
        self._save()

    def _delete_at(self, frame_idx: int) -> None:
        before = len(self.annotations)
        self.annotations = [a for a in self.annotations if a.frame != frame_idx]
        if len(self.annotations) < before:
            self._undo_stack.append(None)
            self._save()

    def _undo(self) -> None:
        if not self._undo_stack:
            return
        last = self._undo_stack.pop()
        if last is not None:
            self.annotations = [
                a for a in self.annotations
                if not (a.frame == last.frame and a.event_type == last.event_type)
            ]
            self._save()

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw_overlay(self, frame: np.ndarray, idx: int) -> None:
        """Annotate the video frame with event markers."""
        # Highlight annotations on the current frame with a coloured banner
        current = [a for a in self.annotations if a.frame == idx]
        if current:
            h, w = frame.shape[:2]
            for i, ann in enumerate(current):
                color = ann.color()
                y0, y1 = i * 28, (i + 1) * 28
                cv2.rectangle(frame, (0, y0), (w, y1), color, -1)
                cv2.putText(frame, ann.label(), (8, y0 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

        # Show prompt when waiting for a second keypress
        if self.mode == MODE_AWAIT_STROKE:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, h - 38), (w, h), (40, 40, 110), -1)
            cv2.putText(
                frame,
                "  HIT TYPE:  [F] Forehand   [K] bacKhand   [S] Serve   [O] Overhead   [V] Volley   |  ESC cancel",
                (6, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 255), 1, cv2.LINE_AA,
            )
        elif self.mode == MODE_AWAIT_END:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, h - 38), (w, h), (110, 40, 40), -1)
            cv2.putText(
                frame,
                "  POINT END:  [W] Winner   [U] Unforced error   [N] Net error   [F] Forced error   |  ESC cancel",
                (6, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 200, 200), 1, cv2.LINE_AA,
            )

    def _make_panel(self, idx: int) -> np.ndarray:
        """Build the info + timeline panel drawn below the video."""
        panel = np.zeros((PANEL_H, self.disp_w, 3), dtype=np.uint8)
        ts      = idx / self.fps
        minutes = int(ts // 60)
        seconds = ts % 60

        # — Status line —
        if self.mode == MODE_AWAIT_STROKE:
            state_str   = "SELECT STROKE TYPE..."
            state_color = (255, 180, 80)
        elif self.mode == MODE_AWAIT_END:
            state_str   = "SELECT POINT END TYPE..."
            state_color = (80, 200, 255)
        elif self.playing:
            state_str   = "PLAYING"
            state_color = (80, 255, 80)
        else:
            state_str   = "PAUSED"
            state_color = (160, 160, 160)

        cv2.putText(
            panel,
            f"  Frame {idx} / {self.total_frames - 1}   {minutes:02d}:{seconds:05.2f}   {state_str}   |   Annotations: {len(self.annotations)}",
            (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, state_color, 1, cv2.LINE_AA,
        )

        # — Quick annotation count summary —
        counts: Dict[str, int] = {}
        for a in self.annotations:
            counts[a.event_type] = counts.get(a.event_type, 0) + 1
        summary = "   ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        cv2.putText(panel, f"  {summary}", (0, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 200, 100), 1, cv2.LINE_AA)

        # — Controls hint —
        cv2.putText(
            panel,
            "  SPC:play/pause   A/D or arrows:±1f   ,/.:±5f   [/]:±30f   B:bounce   H:hit   P:pt.start   E:pt.end   X:del   Z:undo   Q:quit",
            (0, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (100, 100, 100), 1, cv2.LINE_AA,
        )

        # — Timeline scrubber —
        sb_x1, sb_x2 = 4, self.disp_w - 4
        sb_y1, sb_y2 = 70, 90
        cv2.rectangle(panel, (sb_x1, sb_y1), (sb_x2, sb_y2), (50, 50, 50), -1)
        cv2.rectangle(panel, (sb_x1, sb_y1), (sb_x2, sb_y2), (80, 80, 80), 1)

        # Annotation ticks
        span = max(self.total_frames - 1, 1)
        for ann in self.annotations:
            tx = sb_x1 + int(ann.frame / span * (sb_x2 - sb_x1))
            cv2.line(panel, (tx, sb_y1), (tx, sb_y2), ann.color(), 2)

        # Playhead
        cx = sb_x1 + int(idx / span * (sb_x2 - sb_x1))
        cv2.rectangle(panel, (cx - 2, sb_y1 - 4), (cx + 2, sb_y2 + 4), (255, 255, 255), -1)

        cv2.putText(panel, "Timeline (click to seek)", (sb_x1, PANEL_H - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (70, 70, 70), 1, cv2.LINE_AA)
        return panel

    # ── Mouse callback ────────────────────────────────────────────────────────

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        sb_top = self.disp_h + 68   # top of scrubber in combined window coords
        sb_bot = self.disp_h + 96
        sb_x1, sb_x2 = 4, self.disp_w - 4

        if event == cv2.EVENT_LBUTTONDOWN and sb_top <= y <= sb_bot + 10:
            self._scrubber_dragging = True

        if self._scrubber_dragging and event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            frac           = max(0.0, min(1.0, (x - sb_x1) / max(sb_x2 - sb_x1, 1)))
            self.frame_idx = int(frac * (self.total_frames - 1))
            self.playing   = False

        if event == cv2.EVENT_LBUTTONUP:
            self._scrubber_dragging = False

    # ── Key handling ──────────────────────────────────────────────────────────

    def _handle_key(self, key: int) -> bool:
        """Process a key event. Returns False when the user wants to quit."""

        # ESC always cancels the current mode (or quits if already normal)
        if key == 27:
            if self.mode != MODE_NORMAL:
                self.mode = MODE_NORMAL
                return True
            # ESC in normal mode = quit
            self._save()
            return False

        # ── Waiting for stroke type ──────────────────────────────────────────
        if self.mode == MODE_AWAIT_STROKE:
            if key in STROKE_KEYS:
                self._add(Annotation(
                    frame      = self._pending_frame,
                    event_type = "hit",
                    stroke_type = STROKE_KEYS[key],
                ))
                self.mode = MODE_NORMAL
            return True

        # ── Waiting for point-end type ───────────────────────────────────────
        if self.mode == MODE_AWAIT_END:
            if key in END_KEYS:
                self._add(Annotation(
                    frame          = self._pending_frame,
                    event_type     = "point_end",
                    point_end_type = END_KEYS[key],
                ))
                self.mode = MODE_NORMAL
            return True

        # ── Normal mode ──────────────────────────────────────────────────────

        # Quit
        if key == ord("q"):
            self._save()
            return False

        # Navigation
        if key in LEFT_ARROWS or key == ord("a"):
            self.frame_idx = max(0, self.frame_idx - 1)
            self.playing   = False
        elif key in RIGHT_ARROWS or key == ord("d"):
            self.frame_idx = min(self.total_frames - 1, self.frame_idx + 1)
            self.playing   = False
        elif key == ord(","):
            self.frame_idx = max(0, self.frame_idx - 5)
            self.playing   = False
        elif key == ord("."):
            self.frame_idx = min(self.total_frames - 1, self.frame_idx + 5)
            self.playing   = False
        elif key == ord("["):
            self.frame_idx = max(0, self.frame_idx - 30)
            self.playing   = False
        elif key == ord("]"):
            self.frame_idx = min(self.total_frames - 1, self.frame_idx + 30)
            self.playing   = False
        elif key in (ord(" "), ord("w")):
            self.playing = not self.playing

        # Annotation
        elif key == ord("b"):
            self._add(Annotation(frame=self.frame_idx, event_type="bounce"))
        elif key == ord("h"):
            self._pending_frame = self.frame_idx
            self.mode = MODE_AWAIT_STROKE
        elif key == ord("p"):
            self._add(Annotation(frame=self.frame_idx, event_type="point_start"))
        elif key == ord("e"):
            self._pending_frame = self.frame_idx
            self.mode = MODE_AWAIT_END
        elif key in (ord("x"), 127, 255):   # x or Delete
            self._delete_at(self.frame_idx)
        elif key == ord("z"):
            self._undo()

        return True

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        cv2.namedWindow(self.WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WIN_NAME, self.disp_w, self.disp_h + PANEL_H)
        cv2.setMouseCallback(self.WIN_NAME, self._on_mouse)

        print(f"\n{'=' * 62}")
        print(f"  Tennis Annotator  |  {Path(self.video_path).name}")
        print(f"{'=' * 62}")
        print(f"  Frames : {self.total_frames}   FPS : {self.fps:.1f}")
        print(f"  Output : {self.output_csv}")
        print()
        print("  Controls:")
        print("    SPACE/W      : play / pause")
        print("    A / ←        : back 1 frame")
        print("    D / →        : forward 1 frame")
        print("    , / .        : back / forward 5 frames")
        print("    [ / ]        : back / forward 30 frames (~1 s)")
        print("    B            : mark Bounce")
        print("    H → F/K/S/O/V: mark Hit  (Forehand / bacKhand / Serve / Overhead / Volley)")
        print("    P            : mark Point Start")
        print("    E → W/U/N/F  : mark Point End  (Winner / Unforced / Net / Forced error)")
        print("    X            : delete annotation at current frame")
        print("    Z            : undo last annotation")
        print("    Q / ESC      : save and quit")
        print(f"{'=' * 62}\n")

        delay_play = max(1, int(1000 / self.fps))

        while True:
            # Build display
            frame = self._display_frame(self.frame_idx)
            self._draw_overlay(frame, self.frame_idx)
            panel = self._make_panel(self.frame_idx)
            cv2.imshow(self.WIN_NAME, np.vstack([frame, panel]))

            # Wait for key (0 ms when paused = block; delay_play ms when playing)
            key = cv2.waitKeyEx(delay_play if self.playing else 0)

            # Check if window was closed with the X button
            try:
                if cv2.getWindowProperty(self.WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            if key != -1:
                if not self._handle_key(key):
                    break

            if self.playing:
                self.frame_idx += 1
                if self.frame_idx >= self.total_frames:
                    self.frame_idx = self.total_frames - 1
                    self.playing   = False

        self._save()
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"\nSaved {len(self.annotations)} annotations → {self.output_csv}")

        # Print summary
        counts: Dict[str, int] = {}
        for a in self.annotations:
            counts[a.event_type] = counts.get(a.event_type, 0) + 1
        print("\nAnnotation summary:")
        for k, v in sorted(counts.items()):
            print(f"  {k:<20} {v}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate tennis video for ML training data (righty-only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--video",  required=True, help="Path to input video file")
    parser.add_argument(
        "--output", default=None,
        help="Output CSV path (default: cv/training_data/<video_stem>_annotations.csv)",
    )
    args = parser.parse_args()

    if args.output is None:
        stem        = Path(args.video).stem
        args.output = str(PROJECT_ROOT / "cv" / "training_data" / f"{stem}_annotations.csv")

    annotator = TennisAnnotator(
        video_path = args.video,
        output_csv = args.output,
    )
    annotator.run()


if __name__ == "__main__":
    main()
