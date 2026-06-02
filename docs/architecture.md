# Architecture

Courtvision has three runtime pieces and one shared data backbone:

- **Frontend** — Next.js app the coach/player interacts with.
- **Backend** — FastAPI service: auth, business logic, video lifecycle, and
  launching CV jobs.
- **CV pipeline** — Python jobs under `cv/` that do the actual video analysis.
- **Supabase** — Postgres + Auth + Storage, shared by all of the above.

```
┌──────────────────────────────────────────────────────────────┐
│  Browser — Next.js on Vercel (frontend/)                      │
│  landing · auth · teams · upload modal · player select ·      │
│  court editor · match dashboard (shots, zones, stats)         │
└───────┬────────────────────────────────┬──────────────────────┘
        │ REST (JWT)                      │ signed PUT (video) / GET (frames)
        ▼                                 ▼
┌───────────────────────┐       ┌───────────────────────────────┐
│  FastAPI — ECS Fargate │       │  Supabase                     │
│  (backend/, port 8000) │◀─────▶│  Postgres + Auth + RLS        │
│  api/teams             │       │  Storage: match-videos bucket │
│  api/matches           │       └───────────────────────────────┘
│  api/videos            │                     ▲
│  api/stats             │                     │ read video / write results
│  api/activation        │                     │
└──────────┬─────────────┘                     │
           │ sqs:SendMessage                    │
           ▼                                    │
┌──────────────────────┐                        │
│  AWS SQS             │                        │
│  analysis-jobs queue │                        │
└──────────┬───────────┘                        │
           │ sqs:ReceiveMessage                 │
           ▼                                    │
┌──────────────────────────────────────────────┴──────────────┐
│  CV Worker — EC2 g4dn.xlarge (GPU)                           │
│  cv/sqs_worker.py polls queue, spawns:                       │
│    analysis_job.py · player_selection_job.py                 │
│    court_setup_job.py · debug_video_job.py                   │
│  Models loaded from S3 into /app/models/                     │
└──────────────────────────────────────────────────────────────┘
```

In local development, `SQS_QUEUE_URL` is unset and `_dispatch_analysis` falls
back to `subprocess.Popen` on the same machine — no AWS required.

---

## Components

### Frontend (`frontend/`)
Next.js 16 App Router. Talks to Supabase directly for **auth** and for the
**signed-URL video upload**, and to the FastAPI backend for everything else.
Notable routes: `/`, `/dashboard`, `/teams`, `/stats`, `/profile`,
`/matches/[id]`, `/matches/[id]/identify`, `/matches/[id]/court-setup`.

### Backend (`backend/`)
FastAPI app (`backend/main.py`) mounting five routers:

| Router | Responsibility |
|---|---|
| `api/teams` | Team CRUD, membership, team codes |
| `api/matches` | Match records, listing, results |
| `api/videos` | Upload lifecycle, PlaySight import, court keypoints, launching CV jobs |
| `api/stats` | Aggregated stats for the dashboard |
| `api/activation` | Activation-key validation and status |

Auth is enforced via Supabase JWT (`backend/auth.py` → `get_user_id`). The
backend uses the Supabase **service-role** key and re-checks ownership on every
match (`_get_match_or_403`).

### CV jobs (`cv/`)
Launched by the backend with `subprocess.Popen`, each job downloads the video
from Storage, does its work, and writes back to Supabase / posts to the backend:

| Job | Trigger | Output |
|---|---|---|
| `player_selection_job.py` | `POST /api/videos/{id}/generate-player-selection` | 5 annotated frames → `player_selection_frames/{id}.json` |
| `court_setup_job.py` | `POST /api/videos/{id}/confirm-upload` | AI-suggested 14 court keypoints |
| `analysis_job.py` | `PUT /api/videos/{id}/court-keypoints` (after confirm) | `match_data` + `shots` rows, `status=completed` |
| `debug_video_job.py` | `POST /api/videos/{id}/generate-debug-video` | annotated debug video in Storage |

The pipeline that `analysis_job.py` drives lives in `cv/pipeline.py`
(`AnalyticsPipeline`). See [`cv-pipeline.md`](cv-pipeline.md).

---

## Data model (Supabase Postgres)

Defined in `supabase/schema.sql` plus `supabase/migrations/`. Core tables:

| Table | Purpose |
|---|---|
| `users` | Profile extending `auth.users`; `role` (coach/player), `team_id`, `activation_key` |
| `teams` / `team_members` | Teams and membership (one coach key activates the team) |
| `matches` | One row per uploaded match; status + court-setup status + `poi_start_side` + storage key |
| `court_configs` | The 14 confirmed court keypoints per match |
| `player_identifications` | The coach's clicked target-player coordinates |
| `match_data` | Full analysis JSON + stats summary |
| `shots` | Per-shot records (type, position, result) used by the shot map |

Row-Level Security isolates data per user, with an explicit policy letting a
**coach read their team members' matches/stats**.

> **Note:** `schema.sql` is the base schema; several columns used by the current
> backend (`s3_temp_key`, `court_setup_status`, `poi_start_side`, `analysis_error`,
> `debug_video_status`, `debug_video_path`, `court_configs`) are added by the
> migrations in `supabase/migrations/`. Always run the migrations after the base schema.

---

## Key design principles

1. **Video never touches the backend.** Files go browser → Supabase Storage via a
   signed PUT URL (or are fetched server-side for PlaySight). Zero backend
   upload bandwidth.
2. **Videos are temporary.** Only results (keypoints, shots, stats) persist.
   Source videos live under `temp-uploads/` and are deleted after processing.
3. **Court keypoints are user-confirmed.** The model suggests 14 keypoints; the
   coach confirms them in the editor, so the homography is correct regardless of
   camera angle. Keypoints are detected once (static-camera assumption) and reused.
4. **CV runs out-of-process.** Jobs are detached subprocesses, so API requests
   return immediately and the frontend polls `GET /api/videos/{id}/status`.

---

## Deployment targets

| Layer | Local dev | Production |
|---|---|---|
| Frontend | `npm run dev` | Vercel |
| Backend API | Uvicorn | AWS ECS Fargate (behind ALB) |
| Database / Auth / Storage | Supabase free tier | Supabase Pro |
| CV processing | Local subprocess on the backend host | EC2 g4dn.xlarge polling SQS |

Infrastructure-as-code lives in `infra/` (Terraform). See [`deployment.md`](deployment.md) for the full first-time setup.

The one part that does **not** scale as-is: CV jobs run on whatever machine runs
the backend. A managed API host (e.g. Railway) won't have the models or a GPU, so
production needs a separate GPU worker. See [`deployment.md`](deployment.md).
