# Deployment

Courtvision is three deployables (frontend, backend API, CV worker) plus a
managed Supabase project.

## Local development

```bash
# Backend
source tennis_env/bin/activate
pip install -r backend/requirements.txt   # + CV deps (see docs/cv-pipeline.md)
./start_backend.sh        # uvicorn main:app --reload --port 8000

# Frontend
./start_frontend.sh       # next dev on :3000
```

In local dev the backend launches CV jobs as subprocesses on the same machine,
so the models and (ideally) a GPU/MPS device must be present locally.

## Production topology

| Layer | Recommended host | Notes |
|---|---|---|
| Frontend | Vercel | set `NEXT_PUBLIC_*` env vars |
| Backend API | Railway / Render / AWS ECS | needs `ffmpeg` for PlaySight import |
| Database / Auth / Storage | Supabase Pro | run `schema.sql` + `migrations/` |
| CV processing | **dedicated GPU worker** | the API host has no GPU/models |

### The CV worker is the part that needs care

Today the backend runs CV jobs with `subprocess.Popen` on its own host. A
managed API host (Railway/Render) typically has **no GPU and not enough
memory/disk for the models**, so for production the analysis must move to a
separate worker. Options:

- **Serverless GPU** (Modal / RunPod / Replicate / Beam): the API enqueues a job;
  the worker spins up on demand, processes, writes results to Supabase, and
  scales to zero. Best fit for spiky, low-volume workloads — you pay per second
  of actual processing instead of for an idle 24/7 GPU.
- **Always-on GPU instance / queue** (e.g. an AWS GPU instance + a job queue):
  simpler mental model, but you pay for idle time.

Whichever you choose, keep the contract identical to the local jobs: the worker
takes a `match_id`, pulls the video and confirmed keypoints from Supabase, runs
`AnalyticsPipeline`, writes `match_data` + `shots`, deletes the temp video, and
sets `status=completed`.

### Cost levers (see also docs/cv-pipeline.md)

- **Gate to live play** — only ~20–30% of a match is in-play; run the heavy
  models only inside detected points.
- **Run SAM-3D-Body only at hit frames**, never per-frame.
- **Batch GPU inference** rather than one frame at a time.
- **Downscale** before inference (e.g. 720p).

## Environment variables

**Backend**
```
SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=
SUPABASE_ANON_KEY=
ALLOWED_ORIGINS=https://your-frontend-domain
BACKEND_URL=https://your-backend-domain
```

**Frontend**
```
NEXT_PUBLIC_SUPABASE_URL=
NEXT_PUBLIC_SUPABASE_ANON_KEY=
NEXT_PUBLIC_API_URL=https://your-backend-domain
```

## Supabase setup

1. Create the project and a Storage bucket named `match-videos`.
2. Run `supabase/schema.sql`, then every file in `supabase/migrations/`.
3. Copy the URL, anon key, and service-role key into the env files above.
