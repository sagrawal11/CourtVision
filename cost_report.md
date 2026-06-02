# Courtvision — Cost Report

**Scenario:** Full analysis of a 2-hour match video  
**Config:** `frame_skip=1` (every frame), TrackNet + YOLOv8 + SAM-3D-Body  
**Date:** 2026-06-02

---

## Technology Stack & Roles

### CV / ML Models

| Model | Role | Where it runs | VRAM requirement |
|---|---|---|---|
| **TrackNet** | Ball detection (3-frame sliding window) | GPU, every frame | ~500 MB |
| **YOLOv8n** | Player bounding boxes (person detection) | GPU, every frame | ~100 MB |
| **SAM-3D-Body (ViT-H)** | Body pose estimation for stroke classification | GPU, hit frames only | ~3.5 GB |
| **CatBoost × 5** | Bounce / hit / stroke / point-start / point-end classification | CPU, post-processing | Negligible |
| **Court homography** | 14-keypoint pixel→court coordinate mapping | CPU, built once | Negligible |

TrackNet and YOLOv8n run sequentially on every frame in the main pipeline loop
(`cv/pipeline.py`). SAM-3D-Body runs in a separate gated pass over hit frames only
(`±3-frame window around each detected contact`). CatBoost models run on the
aggregated track data after all frames are processed — they are fast and CPU-bound.

---

### Infrastructure Services

| Service | Role | Pricing model |
|---|---|---|
| **Supabase** | Postgres DB + Auth + Storage bucket | Monthly flat + overages |
| **GPU worker** (cloud) | Runs the CV jobs launched by `analysis_job.py` | Per-hour (on-demand) |
| **Vercel** | Frontend hosting (Next.js) | Monthly flat |
| **Railway** (or equivalent) | Backend API hosting (FastAPI / Uvicorn) | Monthly usage-based |

---

## Frame Math

| Variable | Value |
|---|---|
| Match duration | 2 hours = 7,200 seconds |
| Video frame rate | 30 fps |
| **Total frames** | **216,000** |
| Frame skip | 1 (every frame processed) |
| Frames processed | **216,000** |
| Hit contacts per match | ~1,250 (midpoint of 1,000–1,500) |
| SAM window per hit | ±3 frames = 7 frames |
| **SAM-3D-Body inferences** | **~8,750** |

---

## Processing Time Estimate (per 2-hour match)

### Phase 1 — TrackNet + YOLOv8 (216,000 frames)

Running both models sequentially on each frame, the combined throughput on a T4
GPU (AWS g4dn.xlarge) is approximately 25–35 fps end-to-end.

| Hardware | Throughput | Wall time |
|---|---|---|
| MacBook MPS (dev reference) | ~10 fps | ~6 hrs |
| AWS T4 (g4dn.xlarge) | ~25–35 fps | ~100–145 min |
| AWS A10G (g5.xlarge) | ~55–70 fps | ~50–65 min |

Using a conservative **30 fps** for T4 planning: **~120 min**.

### Phase 2 — SAM-3D-Body (8,750 inferences)

SAM ViT-H runs at ~4–7 fps on a T4 GPU (~150–250 ms per frame).

| Hardware | Throughput | Wall time |
|---|---|---|
| AWS T4 (g4dn.xlarge) | ~4–7 fps | ~21–36 min |
| AWS A10G (g5.xlarge) | ~10–15 fps | ~10–15 min |

### Phase 3 — CatBoost + stats aggregation (CPU)

Runs on the aggregated track data after all frames. Typically completes in
**< 2 minutes** regardless of video length — not a cost driver.

### Total wall time (T4)

| Phase | Time |
|---|---|
| TrackNet + YOLO | ~120 min |
| SAM-3D-Body | ~28 min (midpoint) |
| CatBoost + aggregation | ~2 min |
| **Total** | **~150 min (~2.5 hrs)** |

---

## Per-Match Cost Breakdown

### GPU compute

#### AWS g4dn.xlarge — NVIDIA T4, 16 GB VRAM ($0.526/hr on-demand)

| Phase | Time | Cost |
|---|---|---|
| TrackNet + YOLO | ~120 min | $1.05 |
| SAM-3D-Body | ~28 min | $0.25 |
| Overhead / startup | ~5 min | $0.04 |
| **GPU total** | **~153 min** | **$1.34** |

#### AWS g5.xlarge — NVIDIA A10G, 24 GB VRAM ($1.006/hr on-demand)

| Phase | Time | Cost |
|---|---|---|
| TrackNet + YOLO | ~60 min | $1.01 |
| SAM-3D-Body | ~13 min | $0.22 |
| Overhead / startup | ~5 min | $0.08 |
| **GPU total** | **~78 min** | **$1.31** |

**Notable:** The A10G costs almost 2× per hour but processes in roughly half the
time, landing at nearly the same total GPU cost (~$1.31 vs ~$1.34). The A10G
advantage is turnaround speed (78 min vs 153 min), not price.

### Supabase (variable per match)

Supabase Pro ($25/mo) includes 100 GB file storage and **250 GB egress/month**.

| Item | Size | Cost |
|---|---|---|
| Video temporary storage | ~4 GB (deleted after job) | Within monthly plan |
| **Egress to GPU worker** | **~4 GB** | **$0 if < 62 matches/mo** |
| Egress overage (> 62 matches/mo) | 4 GB × $0.09/GB | **$0.36/match** |
| Result writes (shots JSON, match_data) | < 1 MB | Negligible |

The 250 GB monthly egress allowance covers approximately **62 two-hour matches**
per month before overages begin. Early-stage usage will likely stay within this.

### Per-match variable total

| Scenario | GPU (T4) | Egress | Total |
|---|---|---|---|
| Early stage (< 62 matches/mo, egress within plan) | $1.34 | $0.00 | **$1.34** |
| Scale (> 62 matches/mo, egress overage) | $1.34 | $0.36 | **$1.70** |
| Premium GPU — A10G, scale | $1.31 | $0.36 | **$1.67** |

---

## Fixed Monthly Infrastructure Costs

| Service | Plan | Monthly cost | What's included |
|---|---|---|---|
| **Supabase** | Pro | $25.00 | 8 GB DB, 100 GB storage, 250 GB egress, no pausing |
| **Vercel** | Hobby | $0.00 | Sufficient for early stage (100 GB bandwidth) |
| **Vercel** | Pro (at scale) | $20.00 | 1 TB bandwidth, team features |
| **Railway** | Hobby | ~$5–15 | Backend API, usage-based (~$0.000463/vCPU-s) |
| **GPU worker** | On-demand only | $0 when idle | No standing cost — charged per analysis job |

**Early-stage fixed total: ~$30–40/mo**  
**At-scale fixed total: ~$45–60/mo**

The GPU worker has **zero standing cost** because it runs on-demand: the backend
launches it as a subprocess (or in production, triggers a cloud job), it runs
the analysis, and terminates. No GPU instance sits idle between jobs.

---

## Scale Projections

Using $1.34/match (T4, early stage, within egress allowance):

| Active coaches | Matches/week each | Analyses/month | Monthly GPU cost | Monthly total cost |
|---|---|---|---|---|
| 10 | 2 | ~87 | ~$117 | **~$150** |
| 50 | 2 | ~433 | ~$580 | **~$630** |
| 100 | 2 | ~867 | ~$1,162 | **~$1,220** |
| 200 | 2 | ~1,733 | ~$2,322 | **~$2,400** |

At 200 coaches paying **$30/month** subscription: $6,000 revenue vs ~$2,400
in total infra costs — a 60% gross margin before other expenses (labor, domain,
email, etc.).

---

## Cost Levers (ranked by impact)

### 1. `frame_skip=2` — biggest single lever

Halves GPU compute time. Effect on accuracy:
- Ball tracking: minimal — 15fps ball trajectory is still dense enough for bounce/hit detection
- Player tracking: negligible — players move slowly relative to frame rate
- SAM-3D-Body: unaffected — it's gated to hit frames, not the main loop

**Saving:** ~50% of GPU cost → ~$0.67/match instead of $1.34.

### 2. Spot instances

AWS spot pricing on g4dn.xlarge is ~$0.16–0.20/hr vs $0.526/hr on-demand.
The analysis job is already fire-and-forget (backend doesn't block on it), so
spot interruptions are tolerable with a simple retry mechanism.

**Saving:** ~65–70% of GPU cost.

### 3. Co-locate GPU worker with Supabase region

Supabase runs on AWS `us-east-1` by default. Running the GPU worker in the
same region via private networking eliminates the $0.09/GB egress charge entirely.

**Saving:** $0.36/match at scale.

### 4. SAM-3D-Body window size

Narrowing the hit-frame window from ±3 to ±2 frames cuts SAM inferences from
~8,750 to ~6,250 (~29% fewer). Accuracy impact is minimal because the classifier
reads pose at the contact moment, not across the whole window.

**Saving:** ~$0.07/match.

### 5. Video resolution

720p video has ~4× fewer pixels than 1080p and processes 2–3× faster through
both TrackNet and YOLO. If PlaySight recordings are available at 720p, requesting
that resolution at download time is a significant free win.

**Saving:** ~40–50% of GPU time for Phase 1.

---

## Worst-Case / Best-Case Summary

| Scenario | Per-match cost |
|---|---|
| On-demand T4, frame_skip=1, scale egress | **$1.70** |
| On-demand T4, frame_skip=1, within egress allowance | **$1.34** |
| On-demand A10G, frame_skip=1 | **$1.67** |
| On-demand T4, frame_skip=2 | **$0.72** |
| Spot T4, frame_skip=2, co-located (no egress) | **~$0.25** |

The primary scenario from this report (on-demand T4, frame_skip=1): **$1.34–$1.70/match**.

---

## VRAM Requirements

The GPU must hold all active models simultaneously:

| Model | VRAM |
|---|---|
| TrackNet | ~500 MB |
| YOLOv8n | ~100 MB |
| SAM-3D-Body ViT-H | ~3,500 MB |
| Framework overhead | ~500 MB |
| **Total** | **~4.6 GB** |

A **T4 (16 GB)** or **A10G (24 GB)** comfortably fits the full stack. An
RTX 3090 (24 GB) on RunPod (~$0.34/hr) is also viable and cheaper than AWS.
Avoid instances with < 8 GB VRAM (e.g. g4dn.medium) — SAM ViT-H won't fit.

---

*Generated 2026-06-02. GPU on-demand pricing from AWS public pricing page;
Supabase pricing from supabase.com/pricing. All compute estimates are based
on published model benchmarks and the pipeline throughput note in `cv/pipeline.py`.*
