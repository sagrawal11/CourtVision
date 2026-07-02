# Video Pipeline

End-to-end flow from a coach providing a match video to the dashboard showing
results. All video bytes live in the Supabase Storage bucket `match-videos`
under `temp-uploads/{match_id}/...` and are deleted after processing.

## Two ingest paths

1. **Direct browser upload** — the coach picks a local `.mp4`/`.mov`. The
   backend issues a signed PUT URL (`POST /api/videos/prepare-upload`) and the
   browser uploads straight to Supabase Storage. The video never passes through
   the backend.
2. **PlaySight share link** — the coach pastes a `https://...playsight...` share
   URL (`POST /api/videos/import-playsight`). The backend scrapes the page for
   its HLS playlist, downloads + muxes it with `yt-dlp` + `ffmpeg`, and uploads
   the resulting `.mp4` to the same Storage location.

> **Requirement:** the backend host must have `ffmpeg` on `PATH`. `yt-dlp` uses
> it to stitch HLS segments. See `backend/services/playsight.py`.

Both paths converge once the video is in Storage.

## Sequence

```
Browser                Backend (api/videos)         Supabase            CV job (cv/)
   │                         │                          │                   │
   │ POST prepare-upload ───▶│                          │                   │
   │                         │ insert match (pending) ─▶│                   │
   │                         │ signed PUT URL           │                   │
   │◀── {match_id, url} ─────│                          │                   │
   │                         │                          │                   │
   │ PUT video ─────────────────────────────────────▶ Storage              │
   │                         │                          │                   │
   │ POST generate-player-selection ▶                   │                   │
   │                         │ subprocess ──────────────────────────────▶ player_selection_job
   │                         │                          │◀── 5 frames json ─│
   │ GET player-selection-data ▶ (frames to click)      │                   │
   │                         │                          │                   │
   │                         │                          │                   │
   ~~ coach places/adjusts the 14 keypoints in the Court Editor ~~           │
   │ PUT court-keypoints ───▶│ save confirmed kps,      │                   │
   │   (confirmed 14 x,y)    │ status=processing        │                   │
   │                         │ subprocess ─────────────────────────────▶ analysis_job
   │                         │                          │◀── match_data + shots, status=completed
   │ GET status → completed ─│                          │                   │
   │ (dashboard renders)     │                          │                   │
```

## Status state machine

`matches.status` (the full set allowed by the `matches_status_check` constraint
in `supabase/schema.sql`):
```
pending → generating_frames → player_selection → court_setup → processing → completed
   │                                                                      → failed
   └─ importing (transient — PlaySight server-side download, reverts to pending)
```
The valid values are `pending`, `importing`, `generating_frames`,
`player_selection`, `court_setup`, `processing`, `completed`, `failed`.

`matches.court_setup_status`:
```
pending → ready → confirmed
```
The full `analysis_job` only runs after the keypoints are **confirmed**.

## Endpoints (`backend/api/videos.py`)

| Method + path | Purpose |
|---|---|
| `POST /api/videos/prepare-upload` | create match, return signed upload URL |
| `POST /api/videos/import-playsight` | server-side PlaySight download → Storage |
| `POST /api/videos/{id}/generate-player-selection` | extract 5 frames + YOLO boxes |
| `GET /api/videos/{id}/player-selection-data` | fetch the clickable frames manifest |
| `POST /api/videos/identify-player` | store the coach's player click (`match_id` in body) |
| `POST /api/videos/{id}/confirm-upload` | verify upload, save keypoints + `poi_start_side`, mark `processing` |
| `PUT /api/videos/{id}/court-keypoints` | save confirmed 14 keypoints, launch analysis |
| `GET /api/videos/{id}/status` | poll overall + court-setup status |
| `POST /api/videos/{id}/generate-debug-video` | render annotated debug video |
| `GET /api/videos/{id}/debug-video-url` | signed download URL for the debug video |

## Storage layout

```
match-videos/
├── temp-uploads/{match_id}/{filename}        # source video (deleted after analysis)
└── player_selection_frames/{match_id}.json   # base64 frames + YOLO boxes for player select
```

## Court keypoint index reference

The 14 keypoints follow the `BallTrackerNet` / `CourtReference` ordering used by
`cv/detection/court_detector.py`:

| Index | Location |
|---|---|
| 0 | Far baseline — left doubles sideline |
| 1 | Far baseline — right doubles sideline |
| 2 | Near baseline — left doubles sideline |
| 3 | Near baseline — right doubles sideline |
| 4 | Far baseline — left singles sideline |
| 5 | Near baseline — left singles sideline |
| 6 | Far baseline — right singles sideline |
| 7 | Near baseline — right singles sideline |
| 8 | Service line — left end |
| 9 | Service line — right end |
| 10 | Service line — left (near net) |
| 11 | Service line — right (near net) |
| 12 | Center service line — far end |
| 13 | Center service line — near end |

## Running a job locally

```bash
# Player-selection frames (extract 5 frames + YOLO boxes)
python cv/player_selection_job.py --match-id "{id}"

# Full analysis
python cv/analysis_job.py --match-id "{id}" --frame-skip 2
```
