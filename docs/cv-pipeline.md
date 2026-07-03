# CV Pipeline

The analytics pipeline lives entirely in `cv/`. It turns a match video plus 14
confirmed court keypoints into structured per-shot data and match statistics.
It produces **data, not video overlays** (a debug overlay renderer exists
separately as `cv/debug_video_job.py`).

Orchestrator: `cv/pipeline.py` → `AnalyticsPipeline`. Entry point in production:
`cv/analysis_job.py` (launched as a subprocess by the backend).

---

## Module map

```
cv/
├── pipeline.py              # AnalyticsPipeline — orchestrates everything
├── analysis_job.py          # prod entry: download video → run pipeline → write Supabase
├── player_selection_job.py  # extract 5 frames + YOLO boxes for the "click your player" UI
├── debug_video_job.py       # render an annotated debug video
├── sqs_worker.py            # EC2 worker: poll SQS → spawn analysis_job.py (production)
├── test_analytics_visually.py # local-only visual debugger for the full pipeline
├── detection/
│   ├── ball_tracker.py      # TrackNet ball detection (3-frame window)
│   ├── player_detector.py   # YOLOv8 person detection
│   └── court_detector.py    # 14-keypoint court model + homography reference
├── analysis/
│   ├── point_detector.py    # BounceDetector, HitDetector, PointStateMachine, PointSegmenter
│   ├── player_identity.py   # PlayerIdentityTracker (stable near/far identity, changeovers)
│   ├── court_zones.py       # 24-zone court classification
│   ├── match_stats.py       # MatchStatsAggregator
│   └── visualizer.py        # debug overlay rendering
├── models/                  # trained CatBoost models (.cbm) + label maps
└── tools/
    ├── extract_features.py  # ball + player tracks → .npz for training
    ├── detect_changeovers.py # detect court-changeover frames from colour signatures
    ├── annotate.py          # local annotation helper
    └── train_models.py      # train the CatBoost models (also exports the runtime feature fns)
```

---

## Processing flow

`AnalyticsPipeline.process()` does:

1. **Homography** — build a matrix from the 14 confirmed keypoints
   (`cv2.findHomography`) mapping video pixels → normalised court coordinates
   (0–1). Done once and reused for the whole video (static camera).
2. **Per-frame detection** — for each frame: TrackNet ball detection, YOLO
   player detection, project to court coords, classify court zone. A
   `PlayerIdentityTracker` keeps a stable near/far identity for the two players
   and flags changeovers.
3. **Post-processing**
   - `PointSegmenter` runs bounce detection → a 5-state point machine
     (`IDLE → SERVING → RALLY → POINT_OVER`, with `CHANGEOVER`) → hit detection →
     speeds and shot types.
   - `MatchStatsAggregator` rolls points up into match stats.
4. **Output** — `AnalysisResult` with per-frame data, a `match_stats` dict, and
   a flattened `shots` list. `analysis_job.py` writes these to `match_data` and
   `shots`.

### What each analysis stage does

| Stage | Method | Notes |
|---|---|---|
| Ball tracking | TrackNet, 3-frame sliding window | robust to motion blur / brief occlusion |
| Player identity | top-2 boxes by area, sorted by feet-Y | filters refs/ball kids/spectators |
| Bounce detection | local Y-maxima in ball trajectory; optional CatBoost | `bounce_model.cbm` used if present |
| Hit detection | directional reversals in ball Y; optional CatBoost | `hit_model.cbm`; assigns hit to nearest player |
| Point segmentation | finite state machine over ball visibility | outputs `error_net` / `error_out` / `in_play` |
| Stroke type | see below | `stroke_model.cbm` |
| Court zones | 24 named zones from normalised coords | serve placement, heatmaps |
| Speeds | court-distance / time × 3.6, +15% arc factor | km/h per shot |

---

## Stroke (forehand / backhand) classification

**Current (built):** `HitDetector` + `stroke_model.cbm` classify each hit into
`forehand / backhand / serve / overhead / volley`. Features today are **ball-xy
and player-centroid trajectory windows** (±6 frames) plus per-player handedness —
see `cv/tools/train_models.py` (`stroke_features`). There is **no pose input** in
the current model. This is fast but limited: perspective distortion and
non-standard footwork reduce accuracy, and the X-vs-player heuristic that seeds
the labels assumes a normal stance.

**Planned (target):** replace the centroid features with **pose features from
SAM-3D-Body**, which is visually far more accurate than YOLOv8-Pose for this use
case. The classifier reads the player's body pose (which arm leads, shoulder
rotation, arm extension) at the moment of contact, so it generalises across
handedness and stance.

> **Decision:** SAM-3D-Body is used because it is more accurate than YOLOv8-Pose.
> Courtvision does **not** sell a separate biomechanics/form product — pose is an
> internal feature feeding stroke classification, not a customer-facing tier.

### Cost control: run SAM-3D-Body only at hit frames

SAM-3D-Body is heavy, so it must **not** run on every frame. The execution model
that keeps cost and processing time reasonable:

1. **Cheap pass (whole video, may be downscaled):** TrackNet + YOLOv8 produce
   ball and player tracks.
2. **Gate to play:** the point start/end + bounce/hit detectors find where points
   happen and where contacts occur. Only ~20–30% of a match is live play.
3. **Pose pass (hit frames only):** run SAM-3D-Body in a small window (±a few
   frames) around each detected hit — roughly 1,000–1,500 contacts per match,
   not ~100k frames.
4. **Stroke classify:** feed pose features into the stroke model.

This keeps the expensive model on the order of a few thousand inferences per
match rather than millions, while still using pose exactly where it matters.

---

## Point gating (the "find points first" architecture)

The intended order of operations for a full match:

```
ball/player tracking (cheap, all frames)
        │
        ▼
point start/end + bounce/hit detection  ──►  list of points + contact frames
        │
        ▼
heavy work (SAM-3D-Body pose) runs ONLY inside detected points, at contacts
        │
        ▼
stroke classification + stats
```

Note the point start/end models themselves consume ball+player features, so the
cheap tracking pass cannot be skipped — but the *expensive* pose pass is fully
gated behind it.

---

## Trained models

CatBoost models in `cv/models/` (see `cv/models/model_info.json`):

| Model | File | Task |
|---|---|---|
| Bounce | `bounce_model.cbm` | is this frame a bounce? |
| Hit | `hit_model.cbm` | is this frame a racket contact? |
| Stroke | `stroke_model.cbm` | forehand/backhand/serve/overhead/volley |
| Point start | `point_start_model.cbm` | does a point start here? |
| Point end | `point_end_model.cbm` | does a point end here? |

The pipeline auto-uses these `.cbm` files when present and falls back to
heuristics otherwise.

### Training workflow

1. Label matches in the [annotation app](../annotation_collaboration/README.md);
   export a CSV per video to `cv/training_data/<video>_annotations.csv`.
2. Extract features: `python cv/tools/extract_features.py --video path/to.mp4`.
3. Train: `python cv/tools/train_models.py` → writes new `.cbm` files.

The annotation app is the data flywheel — more labeled matches directly improve
the CatBoost models (and, once integrated, the pose-based stroke model).

---

## Accuracy roadmap (status)

1. **POI side wiring** — ✅ done. `poi_start_side` comes from the coach's player
   click (majority vote over the 5 sample frames, split at the frame's real
   height midpoint) and is stored on the match before analysis.
2. **Winner vs. in-play** — ✅ done. Dual-bounce detection
   (`PointSegmenter._apply_dual_bounce_winner`): a true winner bounces twice on one
   court half within 2s with no intervening contact.
3. **Net-error precision** — ✅ done. `classify_point_outcome` only calls a
   no-bounce point a net error when the ball was last seen at the net
   (`|court_y-0.5| ≤ 0.12`); otherwise it stays `in_play` rather than assuming the
   tracker's loss was a net error.
4. **Changeover detection** — ✅ done. Committing a side-swap now requires
   `min_persist_samples` valid samples confirming the new side, so a lone flip
   during a timeout/replay gap isn't mistaken for a real changeover.
5. **Stroke accuracy** — 🚧 scaffolded. See the SAM-3D-Body plan above;
   `cv/detection/pose_estimator.py` ships the gated-window cost control and is
   inert until a model is wired in.

> The single biggest remaining accuracy lever is **more labeled data + retraining**
> the CatBoost detectors — the base bounce/hit/point detectors, not these
> post-processing refinements. Measure progress with the evaluation harness below.

## Evaluation & keypoint tooling

- `cv/eval/labels.py` — parse the annotation CSVs (`cv/training_data/*.csv`) into
  ground truth + frame-matching / precision-recall scoring.
- `cv/tools/evaluate_pipeline.py` — score pipeline output vs the labels
  (per-event P/R/F1 + point/winner/error counts); `--max-frames` /
  `--save-detections` for fast, reusable runs.
- `cv/tools/court_keypoint_editor.py` — standalone HTML editor to place the 14
  keypoints for a video (in `CourtReference` order).
- `cv/tools/check_court_config.py` — validate a court config's homography
  (raw vs remapped). **Note:** `court_configs` are stored in the frontend editor's
  keypoint order and `analysis_job.fetch_keypoints` remaps them to the
  `CourtReference` order the homography expects — do not feed raw frontend order
  to `_build_homography`.

---

## Models & dependencies

Pretrained weights live in `models/` (git-ignored, downloaded separately):
`models/ball/`, `models/court/`, `models/player/`, `models/pose/`. The CV
runtime needs `torch`, `torchvision`, `ultralytics`, `opencv-python`,
`catboost`, `numpy`, `scipy`, and `tqdm` (installed separately from the slim
`backend/requirements.txt`). The pipeline auto-selects CUDA → MPS → CPU.
