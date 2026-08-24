"""Vendored subset of WASB — Widely Applicable Strong Baseline for Sports Ball
Detection and Tracking (NTT Communications, BMVC 2023). MIT License.

  Paper: https://arxiv.org/abs/2311.05237
  Code:  https://github.com/nttcom/WASB-SBDT

Only the HRNet model (`hrnet.py`) and affine image utilities (`image_utils.py`)
are vendored — the minimum needed to run the pretrained tennis ball detector.
The runtime wrapper lives in cv/detection/wasb_ball_tracker.py. On real tennis
footage WASB detects ~81% of clearly-visible balls vs ~63% for the previous
TrackNet post-processing, which is the ball-tracking ceiling-raiser the rest of
the event/annotation pipeline depends on.
"""
