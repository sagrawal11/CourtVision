"""cv/tools/check_court_config.py — validate court keypoints' homography.

The acceptance test for the frontend→reference keypoint remap fix. Given a real
court_config (placed in the actual frontend court editor), it builds the
homography BOTH in the raw frontend order and after the remap, so you can see
the fix take a garbage homography to a valid one.

    # Pull a match's court_config from Supabase (needs SUPABASE_URL +
    # SUPABASE_SERVICE_ROLE_KEY — auto-loaded from backend/.env if present):
    python cv/tools/check_court_config.py --match-id <uuid>

    # Or check a saved court_config JSON ({"kp0_x":..,"kp0_y":..,...}, frontend order):
    python cv/tools/check_court_config.py --court-config config.json

    # Or check an editor keypoints.json (already reference order, 14 [x,y] pairs):
    python cv/tools/check_court_config.py --keypoints keypoints.json

Verdict: after the remap, expect valid=True (~11px mean err, ~14 inliers, 14/14
in bounds). If the raw frontend order is already valid and the remap breaks it,
the remap permutation is wrong — do NOT ship it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from cv.eval import homography
from cv.analysis_job import reorder_frontend_to_reference


def _kps_from_config(data: dict):
    """Extract kp0..kp13 (frontend order) from a court_config dict."""
    out = []
    for i in range(14):
        x, y = data.get(f"kp{i}_x"), data.get(f"kp{i}_y")
        out.append((float(x), float(y)) if x is not None and y is not None else None)
    return out


def _load_from_supabase(match_id: str) -> dict:
    # Load backend/.env for credentials if present (never overrides real env).
    env = PROJECT_ROOT / "backend" / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    from supabase import create_client
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    resp = sb.table("court_configs").select("*").eq("match_id", match_id).single().execute()
    if not resp.data:
        raise SystemExit(f"No court_config for match {match_id}")
    return resp.data


def _report(label: str, result: dict) -> None:
    if not result.get("valid") and "reason" in result:
        print(f"  {label:<26} INVALID — {result['reason']}")
        return
    verdict = "VALID ✓" if result["valid"] else "INVALID ✗"
    print(f"  {label:<26} {verdict}  inliers={result['inliers']}/{result['n_points']}  "
          f"mean_err={result['mean_err_px']}px  max={result['max_err_px']}px  in_bounds={result['in_bounds']}/14")


def main():
    ap = argparse.ArgumentParser(description="Validate court-keypoint homography (raw vs remapped)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--match-id", help="pull court_config from Supabase")
    g.add_argument("--court-config", help="court_config JSON (kp{i}_x/_y, frontend order)")
    g.add_argument("--keypoints", help="editor keypoints.json (14 [x,y], reference order)")
    args = ap.parse_args()

    if args.keypoints:
        kps = [tuple(p) if p else None for p in json.loads(Path(args.keypoints).read_text())]
        print("Editor keypoints (already reference order):")
        _report("as-is", homography.check(kps))
        return

    data = _load_from_supabase(args.match_id) if args.match_id else json.loads(Path(args.court_config).read_text())
    frontend_kps = _kps_from_config(data)

    print("Court config homography check:")
    _report("raw (frontend order)", homography.check(frontend_kps))
    _report("remapped (the fix)", homography.check(reorder_frontend_to_reference(frontend_kps)))
    print("\nExpected if the remap is correct: raw INVALID, remapped VALID.")


if __name__ == "__main__":
    main()
