# Video Processing Pipeline

End-to-end flow from a user uploading a match video to receiving their analysis results.

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
