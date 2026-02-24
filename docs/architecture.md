# Tennis Analytics — System Architecture

## Overview

Tennis Analytics is a full-stack platform that ingests match videos, runs multi-layer computer vision analysis (ball tracking, player tracking, court detection), and presents actionable statistics to coaches and players.

---

## Free-Tier Stack (Current)

Everything runs for $0/month during development.

```
┌──────────────────────────────────────────────────────────────────┐
│  Browser (Next.js on Vercel — FREE)                              │
│  ┌────────────────┐  ┌─────────────────┐  ┌──────────────────┐  │
│  │  Upload Modal  │  │  Court Editor   │  │  Dashboard / UI  │  │
│  │  (file picker) │  │  (drag & drop   │  │  (shots, stats)  │  │
│  └───────┬────────┘  └────────┬────────┘  └──────────────────┘  │
└──────────┼────────────────────┼──────────────────────────────────┘
           │ Signed PUT URL     │ Court keypoints confirmed (PUT)
           ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Supabase (FREE tier)                                           │
│  ┌──────────────────────┐  ┌──────────────────────────────────┐ │
│  │  Storage: match-videos│  │  Postgres (DB) + Auth            │ │
│  │  temp-uploads/        │  │  ┌──────────┐  ┌─────────────┐  │ │
│  │    {match_id}/video   │  │  │ matches  │  │court_configs│  │ │
│  │    {match_id}/frame   │  │  │ users    │  │ (14 kps)    │  │ │
│  └──────────────────────┘  │  │ shots    │  └─────────────┘  │ │
│                              └──────────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────────┘
                           │
           ┌───────────────┴─────────────────┐
           ▼                                 ▼
┌──────────────────────┐         ┌──────────────────────────────┐
│  FastAPI Backend      │         │  Local CV Pipeline (your Mac)│
│  (Render.com — FREE) │         │  cv/court_setup_job.py       │
│  backend/api/         │         │  - Download video            │
│  ├── videos.py  ◄────┼─────────│  - Extract frame 1000        │
│  ├── matches.py │    │         │  - Run CourtDetector         │
│  └── storage.py │    │         │  - Upload frame JPEG         │
│                  │    │         │  - POST keypoints to backend │
└──────────────────┘    │         └──────────────────────────────┘
                         │
                         └── subprocess.Popen() (fire & forget)
```

---

## Key Design Principles

1. **Video never touches our backend server.** Files go directly from the browser to Supabase Storage via a signed PUT URL. Zero backend bandwidth cost.

2. **Videos are temporary.** Only analysis *results* (keypoints, shot coordinates, stats) are stored permanently. Videos live in `temp-uploads/` and are explicitly deleted after processing.

3. **Court keypoints are user-confirmed.** The AI model suggests positions for all 14 keypoints, but the user explicitly confirms them in the court editor before analysis runs. This guarantees accuracy regardless of camera angle.

4. **Local CV processing.** For development, `court_setup_job.py` runs as a local subprocess triggered by the backend. No cloud GPU cost. Swap in AWS Batch when scaling.

5. **Static camera assumption.** Keypoints detected from frame ~1000 are cached for the entire video.

---

## Technology Stack

| Layer | Dev (Free) | Production |
|---|---|---|
| Frontend | Vercel (free) | Vercel |
| Backend API | Render.com (free, sleeps when idle) | Railway / AWS ECS |
| Auth | Supabase Auth | Supabase Auth |
| Database | Supabase (free tier) | Supabase Pro |
| Video Storage | Supabase Storage (free tier, 1 GB) | AWS S3 |
| CV Processing | Local subprocess (your Mac) | AWS Batch (GPU spot) |

---

## Data Flow: Video Upload → Analysis

See [video_pipeline.md](./video_pipeline.md) for the full sequence diagram.

---

## Environment Variables

### Backend (`backend/.env`)
```
SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=
SUPABASE_ANON_KEY=
ALLOWED_ORIGINS=http://localhost:3000
BACKEND_URL=http://localhost:8000
```

### Frontend (`frontend/.env.local`)
```
NEXT_PUBLIC_SUPABASE_URL=
NEXT_PUBLIC_SUPABASE_ANON_KEY=
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## Scaling to Production

When you have real users and revenue, the following swaps eliminate the free-tier constraints:

| Constraint | Free Solution | Paid Solution |
|---|---|---|
| Backend sleeps after 15 min | Render.com free | Railway ($5/mo) or AWS ECS |
| 1 GB storage limit | Supabase Storage | AWS S3 (`courtvision-uploads` bucket) |
| Local CV processing only | subprocess on your Mac | AWS Batch (g4dn.xl spot ~$0.15/video) |
