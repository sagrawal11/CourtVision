# Video Processing Pipeline

End-to-end flow from a user uploading a match video to receiving their analysis results.

## Ingest paths

There are now **two** ways a match video can land in `match-videos/temp-uploads/{match_id}/...`:

1. **Direct browser upload** (original flow) — coach picks a local `.mp4`/`.mov`,
   browser PUTs it straight to Supabase Storage using a signed URL.
2. **PlaySight share link** — coach pastes a `https://my.playsight.com/share?...`
   link. The backend scrapes the share page for its `og:video` HLS playlist,
   downloads it via `yt-dlp` + `ffmpeg`, muxes it into a single `.mp4`, then
   uploads to the same storage location.

Both paths converge after the video is in Supabase Storage — the rest of the
pipeline (player selection → court keypoints → analysis) is identical.

See `backend/services/playsight.py` and the `/api/videos/import-playsight`
endpoint in `backend/api/videos.py` for the PlaySight implementation.

> **System requirement**: the backend host must have `ffmpeg` on `PATH`
> (e.g. `brew install ffmpeg` on macOS, `apt-get install ffmpeg` on Debian).
> `yt-dlp` invokes `ffmpeg` to stitch HLS segments into the final MP4.

---

## Sequence Diagram

```
Browser              Backend API           S3               AWS Batch           Supabase
   │                     │                  │                   │                   │
   │──POST /presigned─── ▶                  │                   │                   │
   │    { filename }      │                  │                   │                   │
   │                      │──create match────────────────────────────────────────── ▶
   │                      │  s3_key, status=pending                                 │
   │                      │──generate presigned PUT URL──▶                          │
   │◀──{ upload_url,      │                  │                   │                   │
   │    match_id, s3_key}─│                  │                   │                   │
   │                      │                  │                   │                   │
   │──PUT video file──────────────────────── ▶                   │                   │
   │   (directly to S3)   │                  │                   │                   │
   │◀──200 OK─────────────────────────────── │                   │                   │
   │                      │                  │                   │                   │
   │──POST /confirm-upload▶                  │                   │                   │
   │    { match_id }       │──update status=court_setup──────────────────────────── ▶
   │                       │──trigger court_setup_job──────────── ▶                  │
   │◀──{ status: ready }───│                  │                   │                   │
   │                       │                  │                   │                   │
   ~~ (court_setup_job runs on Batch) ~~      │                   │                   │
   │                       │                  │◀──download video── │                   │
   │                       │                  │                   │extract frame 1000  │
   │                       │                  │                   │run CourtDetector    │
   │                       │◀──POST /court-keypoints──────────────│                   │
   │                       │   { 14 x,y pairs }│                  │                   │
   │                       │──save to court_configs ──────────────────────────────── ▶
   │                       │──update court_setup_status='ready'───────────────────── ▶
   │                       │                  │                   │                   │
   ~~ (frontend polls status) ~~              │                   │                   │
   │──GET /status──────────▶                  │                   │                   │
   │◀──{ court_setup_status: 'ready',         │                   │                   │
   │    frame_url: ... }───│                  │                   │                   │
   │                       │                  │                   │                   │
   ~~ (User adjusts court keypoints in Court Editor UI) ~~        │                   │
   │──PUT /court-keypoints─▶                  │                   │                   │
   │   { 14 confirmed x,y }│──update court_configs ───────────────────────────────── ▶
   │                        │──update court_setup_status='confirmed'────────────────── ▶
   │                        │──trigger full analysis job────────── ▶                  │
   │◀──{ status: processing }│                │                   │                   │
   │                        │                 │                   │                   │
   ~~ (full analysis job runs on Batch) ~~    │                   │                   │
   │                        │                 │◀──download video── │                   │
   │                        │                 │                   │run cv/pipeline.py  │
   │                        │                 │                   │(ball + player +    │
   │                        │                 │                   │ court using locked │
   │                        │                 │                   │ confirmed kps)     │
   │                        │◀──POST results──────────────────────│                   │
   │                        │──store match_data, shots ────────────────────────────── ▶
   │                        │──delete S3 temp video─────────────── ▶                  │
   │                        │──update status='completed'─────────────────────────────▶│
   │──(redirect to dashboard)│                │                   │                   │
```

---

## PlaySight Import Flow

When the coach pastes a PlaySight link instead of uploading a file:

```
Browser              Backend API           PlaySight CDN     Supabase Storage     Supabase DB
   │                     │                    │                   │                   │
   │──POST /import-playsight ─▶                │                   │                   │
   │    { playsight_url,  │                    │                   │                   │
   │      metadata }      │──insert matches ──────────────────────────────────────────▶│
   │                      │◀──match_id────────────────────────────────────────────────│
   │                      │                    │                   │                   │
   │                      │──GET playsight share page──▶          │                   │
   │                      │◀──HTML w/ og:video─                   │                   │
   │                      │──yt-dlp m3u8──────▶                   │                   │
   │                      │◀──HLS segments + ffmpeg mux────       │                   │
   │                      │  (local .mp4 in temp dir)              │                   │
   │                      │──upload mp4 to temp-uploads/────────▶│                   │
   │                      │──update status='pending'──────────────────────────────────▶│
   │◀──{ match_id,        │                    │                   │                   │
   │   storage_path,      │                    │                   │                   │
   │   size_bytes }───────│                    │                   │                   │
   │                      │                    │                   │                   │
~~ continues identically to direct-upload path: generate-player-selection → court editor → analysis ~~
```

The endpoint blocks until the download + upload completes (the wait shows as a
spinner in `upload-modal.tsx`). For typical 5–10 minute PlaySight clips this is
~10–60 s on a residential connection — short enough that the request stays
within FastAPI's default keepalive without needing a separate background job.

---

## Status State Machine

The `matches.status` column tracks the overall match lifecycle:

```
pending  →  court_setup  →  processing  →  completed
                                       →  failed
```

The `matches.court_setup_status` column tracks the court editor sub-flow:

```
pending  →  ready  →  confirmed
```

Full analysis job only starts when `court_setup_status = 'confirmed'`.

---

## S3 Key Naming Convention

```
temp-uploads/{match_id}/{original_filename}
```

Example: `temp-uploads/550e8400-e29b-41d4-a716-446655440000/match_vs_penn_state.mov`

S3 Lifecycle Policy on prefix `temp-uploads/`:
- **Expiration: 48 hours** — auto-deletes any video not processed in time
- The analysis job also explicitly calls `DeleteObject` on success

---

## Court Keypoint Index Reference

The 14 keypoints correspond to specific court line intersections. Index 0–13 matches the `BallTrackerNet`/`CourtReference` ordering:

| Index | Location |
|---|---|
| 0 | Far baseline — left (doubles sideline) |
| 1 | Far baseline — right (doubles sideline) |
| 2 | Near baseline — left (doubles sideline) |
| 3 | Near baseline — right (doubles sideline) |
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

---

## Court Setup Job Entry Point

```bash
# Run locally for testing
python cv/court_setup_job.py \
  --s3-key "temp-uploads/{match_id}/video.mov" \
  --match-id "{match_id}" \
  --backend-url "http://localhost:8000"
```
