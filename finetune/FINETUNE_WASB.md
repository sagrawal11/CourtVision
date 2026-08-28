# Fine-tuning WASB with RacketVision (pretrain) → our footage (adapt)

Goal from the SOTA doc (docs/ball-tracking-sota.md, Path 1): WASB is top-tier; its outdoor
collapse is domain shift + clutter, not an architecture ceiling. Fix = fine-tune WASB on our
fixed-camera footage (+ a trajectory-gating layer), using RacketVision as a free pretrain +
permanent eval harness.

## On-disk structure of RacketVision (tennis)
```
<sport>/videos/<match>_<rally>.mp4          # raw clip, 1920x1080 @ 60fps, ~600 frames (one rally per file)
<sport>/all/<match>/csv/<rally>_ball.csv    # ball GT: columns Frame,Visibility,X,Y
                                            #   Frame = 0-indexed video frame; X,Y in ORIGINAL px; (0,0)=invisible
                                            #   SPARSE: ~50 labeled frames/clip (~1 label / 12 frames)
<sport>/all/<match>/racket/<rally>/<frame>.json   # 5-kpt racket pose (not needed for ball)
<sport>/info/{train,val,test}.json          # list of [match, rally] pairs per split
<sport>/info/metainfo.json                  # {"image_shape":[1080,1920,3]}
<sport>/interp_ball/, merged_racket/        # for TrajPred only
```
Tennis split: **train 350 / val 38 / test 43 rallies** (431 clips, 150k frames, ~21.5k ball labels total).
Sizes: full HF dataset 7.54 GB (all 3 sports); **tennis alone 2.0 GB** (videos 1.86 GB). We pulled just the
43 test videos + 43 ball CSVs = **185 MB** for eval.

### Load a clip's frames + labels (recipe)
```python
import cv2, pandas as pd, numpy as np
df = pd.read_csv("tennis/all/match1/csv/000_ball.csv").sort_values("Frame").fillna(0)  # Frame,Visibility,X,Y
cap = cv2.VideoCapture("tennis/videos/match1_000.mp4")
frames = {}; i=0
while True:
    ok, f = cap.read()
    if not ok: break
    frames[i] = f; i += 1          # frame index i aligns with CSV 'Frame'
# For a labeled frame fid: WASB needs [fid-2,fid-1,fid] (clamp>=0); GT = (row.X,row.Y), invisible if (0,0).
```
(See `eval_racketvision.py` for the full, validated loader + metric.)

## Download commands actually used
```bash
mkdir -p /tmp/racketvision_work && cd /tmp/racketvision_work
git clone --depth 1 https://github.com/OrcustD/RacketVision.git      # repo (code + README)

# tennis TEST videos + ball CSVs only (185 MB) — build allow-patterns from info/test.json:
hf download linfeng302/RacketVision --repo-type dataset --local-dir data \
   tennis/videos/match1_000.mp4 tennis/all/match1/csv/000_ball.csv ... (43 pairs)

# For FULL fine-tuning pretrain you'd add the train+val split too:
#   hf download linfeng302/RacketVision --repo-type dataset --local-dir data \
#       --include "tennis/videos/*" "tennis/all/*/csv/*_ball.csv" "tennis/info/*"   # ~2.0 GB
# MS-TrackNetV3 checkpoint (for reference/eval):
hf download linfeng302/RacketVision-Models --repo-type model --local-dir models checkpoints/balltrack_best.pth
```

## Recommended fine-tuning setup for WASB

**Training code: NOT in our vendored `cv/detection/wasb/` (inference-only: hrnet + image_utils).**
Use upstream **`github.com/nttcom/WASB-SBDT`** (MIT) — it has the trainer, dataset loader, online
tracker, and the exact `hrnet` we vendored. Our `wasb_tennis_best.pth.tar` is a WASB-SBDT checkpoint,
so it loads there as the init for fine-tuning.

### Data format our footage + RacketVision must match (WASB-SBDT `SoccerNet/Tennis`-style)
WASB-SBDT expects, per clip: a folder of frames `img_%06d.jpg` (or the video) + a CSV with
per-frame `frame, x, y, visibility` (same 4 fields RacketVision already uses, just column-renamed).
Concretely, to feed **both** RacketVision and our footage to one trainer, emit this per clip:
```
<clip>/frames/000000.jpg ...                # extracted frames (RacketVision: run their extract_frames.py)
<clip>/gt.csv  ->  frame,x,y,visibility      # x,y in ORIGINAL px; visibility 0/1 (0 => x=y=0)
```
Our fixed-cam footage needs the same: extract frames, and produce a `frame,x,y,visibility` CSV.
Densify labels cheaply via the semi-auto loop (SOTA doc §Data-strategy): run WASB at high conf →
keep high-precision hits → fit short ballistic windows + interpolate 1–3 frame gaps → human-correct
only flagged frames (trajectory outliers, low-conf, near-net/bounce). Target ~3–8k corrected fixed-cam frames.

### Two-stage recipe
1. **Pretrain / warm-start on RacketVision tennis train (350 rallies).** WASB is already tennis-pretrained,
   so this is a short domain-refresh at low LR; mainly gives us a clean, published eval harness (this repo).
2. **Fine-tune on our fixed-cam frames.** Init from `wasb_tennis_best.pth.tar` (or the stage-1 ckpt),
   freeze nothing (small model, 6 MB / ~1.5M params), LR ~1e-4 cosine, HRNet input 512x288, 3-frame,
   focal/WBCE heatmap loss (WASB default), Gaussian target sigma ~2.5–3.5. Heavy aug for sun/shadow/low-contrast
   (brightness/contrast/gamma, hue jitter, motion blur). Keep RacketVision frames mixed in ~1:1 to prevent
   forgetting broadcast generalization.
3. **Add the trajectory-gating layer at inference** (the other half of the precision fix): re-enable WASB-SBDT's
   online quadratic-motion tracker OR add a light ballistic-consistency gate over our per-frame detections
   (reject detections that don't fit a short parabola; interpolate ≤3-frame gaps). Our eval showed the raw
   per-frame model loses ~6 precision pts to exactly the clutter FPs this gate removes.

### Rough compute
- Model is tiny (~1.5M params, 512x288, 3 frames). One 3090/4090 (or an A10/L4) is plenty.
- Stage-1 pretrain on RV tennis (350 rallies ≈ 100k frames): ~2–4 h for a few epochs on one modern GPU.
- Stage-2 fine-tune on ~5k of our frames: <1 h. Full loop easily fits an afternoon on a single GPU.
- Apple MPS works for eval (~40 ms/frame per this harness) but use CUDA for training throughput.

### Success criteria (reuse this harness)
Re-run `eval_racketvision.py --model wasb` after each fine-tune to confirm no regression on RacketVision
(recall should stay ≥0.80), and stand up the same metric on a held-out set of OUR fixed-cam frames
(same `frame,x,y,visibility` CSV) to measure the actual outdoor gain. Target from the SOTA doc:
~90% F1 / 80%+ recall outdoors with the trajectory gate cutting clutter FPs.

## MS-TrackNetV3 note (the alternative, Path 2)
Fully working here: their TrackNetV3 model + `balltrack_best.pth` + median-bg concat reproduce R0.880/P0.945.
It beats our per-frame WASB by +8 recall / +7 precision on identical frames and is ~5x more clutter-robust.
It needs a per-match median background and runs ~2x slower than WASB on MPS. If fine-tuned WASB doesn't clear
the recall bar, adopt the RacketVision recipe (multi-sport TrackNetV3): their `source/BallTrack/train.py --cfg
configs/tracknetv3_base.py` trains on all 3 sports jointly (that joint training is what lifts tennis recall).
