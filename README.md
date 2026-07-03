# Courtvision

Courtvision is a tennis analytics platform for coaches and players. A coach
uploads a match video (a local file or a PlaySight share link), confirms the
court layout, and the computer-vision pipeline returns per-shot data and match
statistics — winners, errors, serve placement, rally length, shot speeds, and
court-zone heatmaps.

> **Status:** Active development. The web app (auth, teams, upload, court
> editor, dashboards) and the CV pipeline are both working end-to-end. Stat
> accuracy is still being improved — see [`docs/cv-pipeline.md`](docs/cv-pipeline.md).

---

## Repository layout

| Path | What it is |
|---|---|
| `frontend/` | Customer-facing **Next.js 16 / React 19** app (port 3000) |
| `backend/` | **FastAPI** API (port 8000) — auth, teams, matches, video lifecycle, PlaySight import |
| `cv/` | The **analytics CV pipeline** — ball/player/court detection + point, hit, stroke, and stats analysis |
| `models/` | Pretrained model weights (ball, court, player, pose) — git-ignored, downloaded separately |
| `supabase/` | Postgres schema + migrations (auth, RLS, tables) |
| `annotation_collaboration/` | Standalone labeling app that produces CV training data (see its own [README](annotation_collaboration/README.md)) |
| `tests/` | Test suite — backend (auth, access control, SSRF, rate limiting, uploads) + CV (labels, homography, outcomes) + pipeline tests |
| `docs/` | Architecture and pipeline documentation |
| `.github/workflows/` | CI — backend pytest + frontend typecheck on push/PR |

---

## Tech stack

**Frontend** — Next.js 16 (App Router), React 19, TypeScript, Tailwind CSS 4,
shadcn/ui, TanStack Query, React Hook Form + Zod, Recharts, Supabase Auth.

**Backend** — FastAPI, Uvicorn, Supabase Python client, OpenCV, `yt-dlp` +
`ffmpeg` (PlaySight HLS download).

**CV / ML** — PyTorch, Ultralytics YOLOv8, TrackNet (ball), CatBoost
(bounce / hit / stroke / point classifiers), homography-based court mapping.

**Infrastructure** — Supabase (Postgres, Auth, Storage). CV jobs currently run
as local subprocesses on the backend host; see
[`docs/deployment.md`](docs/deployment.md) for the production GPU-worker plan.

---

## Quickstart (local development)

### Prerequisites
- Python 3.10+ and Node 18+
- `ffmpeg` on `PATH` (`brew install ffmpeg`) — required for PlaySight import
- A Supabase project (free tier is fine)

### 1. Supabase
1. Create a project at [supabase.com](https://supabase.com).
2. In the SQL Editor, run `supabase/schema.sql` (the complete consolidated
   schema), then `supabase/rls_policies.sql` to apply Row-Level Security.
3. Create the `match-videos` Storage bucket manually in the Storage section.
4. Copy the project URL, anon key, and service-role key for the env files below.

### 2. Backend
```bash
python -m venv tennis_env
source tennis_env/bin/activate
pip install -r backend/requirements.txt
# CV dependencies (torch, ultralytics, catboost) are installed separately — see docs/cv-pipeline.md
cp .env.example backend/.env            # fill in Supabase keys (single combined example file)
./start_backend.sh                      # → http://localhost:8000 (docs at /docs)
```

### 3. Frontend
```bash
cd frontend
cp ../.env.example .env.local           # fill in the NEXT_PUBLIC_* values
npm install
cd .. && ./start_frontend.sh            # → http://localhost:3000
```

### Environment variables

**`backend/.env`**
```
SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=
SUPABASE_ANON_KEY=
ALLOWED_ORIGINS=http://localhost:3000
BACKEND_URL=http://localhost:8000
```

**`frontend/.env.local`**
```
NEXT_PUBLIC_SUPABASE_URL=
NEXT_PUBLIC_SUPABASE_ANON_KEY=
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Running the tests

The fast suite (no torch / model weights needed — same set CI runs) is:
```bash
pip install -r backend/requirements.txt -r backend/requirements-dev.txt
pytest tests/test_backend_*.py tests/test_playsight_*.py tests/test_video_validation.py \
       tests/test_rate_limit.py tests/test_label_parser.py tests/test_homography_check.py \
       tests/test_keypoint_remap.py tests/test_cv_outcome_changeover.py tests/test_pose_scaffold.py \
       tests/test_billing_scaffold.py
```
The CV-heavy tests (`test_dual_bounce.py`, etc.) additionally need torch + model
weights and are excluded from CI.

---

## How it works (high level)

1. **Upload** — the browser uploads the video directly to Supabase Storage via a
   signed URL (it never passes through the backend), or the coach pastes a
   PlaySight link that the backend downloads server-side.
2. **Player selection** — five frames are extracted and run through YOLO so the
   coach can click the target player.
3. **Court setup** — the AI suggests 14 court keypoints; the coach confirms/adjusts
   them in the court editor. This locks the homography for the whole video.
4. **Analysis** — `cv/analysis_job.py` runs the pipeline (ball + player + court →
   points → hits → strokes → stats) and writes results to Supabase.
5. **Results** — the dashboard renders shot maps, court zones, and match stats.

See [`docs/video-pipeline.md`](docs/video-pipeline.md) for the full sequence and
[`docs/cv-pipeline.md`](docs/cv-pipeline.md) for the CV internals.

---

## Business model

Access is gated by **team-based activation keys**: a coach activates an account,
which unlocks team creation and uploads for the whole team. Players who join an
activated coach's team are activated automatically. Billing is currently manual
(keys assigned in the database); a metered/subscription layer is future work.

---

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — system components, data model, env, deploy targets
- [`docs/cv-pipeline.md`](docs/cv-pipeline.md) — CV modules, the SAM-3D-Body stroke plan, known gaps, training
- [`docs/video-pipeline.md`](docs/video-pipeline.md) — upload → analysis sequence and status machine
- [`docs/deployment.md`](docs/deployment.md) — local dev and production hosting
- [`annotation_collaboration/README.md`](annotation_collaboration/README.md) — the training-data labeling app

---

## License

All rights reserved. See [`LICENSE`](LICENSE).
