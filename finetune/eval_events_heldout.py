#!/usr/bin/env python3
"""finetune/eval_events_heldout.py — does the ep12 detector lift the EVENT ceiling?

Held-out experiment (same design the bottleneck memory used): train the bounce/hit/
stroke/point CatBoost models on Indoor Match 1 ONLY, pre-label the held-out Indoor
Match 2, and score predicted vs true event frames @±7. Run it twice — once on the
ep12-re-extracted features (current *_features.npz) and once on the wasb_base backup —
so the ONLY variable is the ball detector.

Production cv/models is backed up and restored (train writes into it because prelabel
loads from there). CPU only (~10-15 min).

    python finetune/eval_events_heldout.py
"""
from __future__ import annotations
import sys, os, shutil, tempfile, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cv.eval.labels import parse_events, match_frames, precision_recall_f1

TD = ROOT / "cv/training_data"
M2_STEM = "Indoor Match 2 15.53.25"
M2_VIDEO = str(ROOT / f"tests/{M2_STEM}.mp4")
TRUE = str(TD / f"{M2_STEM}_annotations.csv")
ROI = str(ROOT / "cv/court_rois/Indoor_Match.json")
EXCLUDES = ["Indoor Match 2", "cal_court1", "Outdoor Match 1", "Outdoor Practice"]  # -> train on Indoor Match 1 only
EVENTS = ["hit", "bounce", "point_start", "point_end"]
TOL = 7
PY = sys.executable
MODELS = ROOT / "cv/models"


def train_prelabel_score(data_dir, m2_features, pred_csv):
    if MODELS.exists():
        shutil.rmtree(MODELS)
    MODELS.mkdir(parents=True)
    ex = []
    for e in EXCLUDES:
        ex += ["--exclude", e]
    subprocess.run([PY, str(ROOT / "cv/tools/train_models.py"), "--data-dir", str(data_dir),
                    "--model-dir", str(MODELS)] + ex, check=True, cwd=str(ROOT))
    subprocess.run([PY, str(ROOT / "cv/tools/prelabel.py"), "--video", M2_VIDEO,
                    "--features", str(m2_features), "--output", pred_csv, "--court-roi", ROI],
                   check=True, cwd=str(ROOT))
    pred = parse_events(pred_csv); true = parse_events(TRUE)
    out = {}
    for et in EVENTS:
        pf = [e.frame for e in pred if e.event_type == et]
        tf = [e.frame for e in true if e.event_type == et]
        tp, fp, fn = match_frames(pf, tf, TOL)
        m = precision_recall_f1(tp, fp, fn)
        out[et] = (m["precision"], m["recall"], m["f1"], len(pf), len(tf))
    return out


def main():
    backup = tempfile.mkdtemp()
    if MODELS.exists():
        shutil.copytree(MODELS, Path(backup) / "models")
    try:
        print("=== EP12 features (current) — train Indoor Match 1, prelabel Match 2 ===")
        ep12 = train_prelabel_score(TD, TD / f"{M2_STEM}_features.npz", "/tmp/m2_pred_ep12.csv")

        print("\n=== WASB_BASE features (backup) ===")
        wd = Path(tempfile.mkdtemp())
        for f in TD.glob("*_features.npz"):
            stem = f.name[: -len("_features.npz")]
            base = TD / f"{stem}_features.wasb_base.npz"
            os.symlink((base if base.exists() else f).resolve(), wd / f"{stem}_features.npz")
            ann = TD / f"{stem}_annotations.csv"
            if ann.exists():
                os.symlink(ann.resolve(), wd / f"{stem}_annotations.csv")
        wasb = train_prelabel_score(wd, TD / f"{M2_STEM}_features.wasb_base.npz", "/tmp/m2_pred_wasb.csv")
    finally:
        if (Path(backup) / "models").exists():
            if MODELS.exists():
                shutil.rmtree(MODELS)
            shutil.copytree(Path(backup) / "models", MODELS)
            print(f"\n[restored production cv/models from {backup}]")

    print("\n================ HELD-OUT EVENT SCORES @±7 (Indoor Match 2) ================")
    print(f"{'event':>12} | {'wasb_base P/R/F1':>22} | {'ep12 P/R/F1':>22} | ΔF1")
    for et in EVENTS:
        wp, wr, wf, wnp, wnt = wasb[et]
        ep, er, ef, enp, ent = ep12[et]
        print(f"{et:>12} | {wp:.2f}/{wr:.2f}/{wf:.2f} (n={wnp:>4}) | "
              f"{ep:.2f}/{er:.2f}/{ef:.2f} (n={enp:>4}) | {ef - wf:+.2f}")


if __name__ == "__main__":
    main()
