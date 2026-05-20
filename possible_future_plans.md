# Courtvision — CV Pipeline: Current State & Future Plans

A detailed technical research document covering the existing computer vision pipeline, its current capabilities, known limitations, and a prioritised roadmap for improvements.

---

## Table of Contents

1. [Overview of the Two CV Codebases](#1-overview-of-the-two-cv-codebases)
2. [What is Already Built and Working](#2-what-is-already-built-and-working)
   - [Player Delimitation](#21-player-delimitation--cvanalysispoi_trackerpy)
   - [Ball Tracking](#22-ball-tracking--cvdetectionball_trackerpy)
   - [Bounce Detection](#23-bounce-detection--cvanalysispoint_detectorpy)
   - [Point State Machine](#24-point-state-machine--pointstatemachine)
   - [Hit Detection](#25-hit-detection--hitdetector)
   - [Court Zone Classification](#26-court-zone-classification--cvanalysiscourt_zonespy)
   - [Match Statistics Aggregation](#27-match-statistics-aggregation--cvanalysismatch_statspy)
   - [Full Pipeline Orchestration](#28-full-pipeline-orchestration--cvpipelinepy)
3. [The Actual Gaps and Limitations](#3-the-actual-gaps-and-limitations)
   - [Gap 1: Player Click Not Wired to poi_start_side](#31-gap-1-player-click-not-wired-to-poi_start_side)
   - [Gap 2: Winner vs. In-Play — The Hard Problem](#32-gap-2-winner-vs-in-play--the-hard-problem)
   - [Gap 3: Shot Type Classification Is Weak](#33-gap-3-shot-type-classification-is-weak)
   - [Gap 4: Net Error Detection Is Imprecise](#34-gap-4-net-error-detection-is-imprecise)
   - [Gap 5: Side Switching at Changeovers](#35-gap-5-side-switching-at-changeovers)
4. [Research: Approaches to Improve Each Gap](#4-research-approaches-to-improve-each-gap)
   - [Improving Winner Detection](#41-improving-winner-detection)
   - [Improving Shot Type Classification](#42-improving-shot-type-classification)
   - [Improving Bounce Detection](#43-improving-bounce-detection)
   - [Improving Error Attribution](#44-improving-error-attribution)
5. [The Legacy CV System (old/) — What It Has](#5-the-legacy-cv-system-old--what-it-has)
6. [Prioritised Implementation Roadmap](#6-prioritised-implementation-roadmap)
7. [Full Feature Status Table](#7-full-feature-status-table)
8. [Data Flow: End-to-End Pipeline](#8-data-flow-end-to-end-pipeline)
9. [Output Schema: What the Frontend Receives](#9-output-schema-what-the-frontend-receives)

---

## 1. Overview of the Two CV Codebases

There are two CV codebases in this repository. Understanding the distinction is critical:

### `cv/` — The Active, Modern Pipeline (use this one)

This is the purpose-built analytics pipeline that plugs directly into the web app. It is designed to be fast, lightweight, and data-oriented — it produces structured JSON rather than video overlays. It is what `backend/api/videos.py` launches via subprocess.

Key files:
```
cv/
├── pipeline.py                    # Master orchestrator — AnalyticsPipeline class
├── analysis_job.py                # Background job: downloads video, runs pipeline, saves to Supabase
├── player_selection_job.py        # Background job: extracts 5 frames, runs YOLO bounding boxes
├── court_setup_job.py             # Background job: runs court keypoint AI detection
├── debug_video_job.py             # Background job: renders annotated debug video
├── detection/
│   ├── ball_tracker.py            # TrackNet-based ball detection
│   ├── player_detector.py         # YOLOv8 player bounding boxes
│   └── court_detector.py          # 14-keypoint court detection + homography
└── analysis/
    ├── point_detector.py          # BounceDetector, PointStateMachine, HitDetector, PointSegmenter
    ├── poi_tracker.py             # POITracker (who is the target player?), SideSwitchDetector
    ├── court_zones.py             # Full 24-zone court classification system
    ├── match_stats.py             # MatchStatsAggregator, classify_serve_zone
    └── visualizer.py              # Debug overlay rendering
```

### `old/src/` — The Legacy Research System (reference only)

This is the original standalone computer vision research system. It outputs annotated video files rather than structured data. It is much more heavyweight (uses pose estimation, RF-DETR, etc.) and is NOT wired into the web app. Its value is as a reference for algorithms that can be ported into the `cv/` pipeline.

Key legacy files of interest:
- `old/src/analysis/tennis_shot_classifier.py` — ML-based forehand/backhand/serve/smash classifier using pose keypoints
- `old/src/core/tennis_CV.py` — The original 3000+ line monolith; reference for multi-model ensemble
- `old/src/analysis/tennis_data_aggregator.py` — Shows how multiple analysis scripts were combined

---

## 2. What is Already Built and Working

### 2.1 Player Delimitation — `cv/analysis/poi_tracker.py`

**The `POITracker` class** handles the task of assigning identity labels ("poi" = target player, "opp" = opponent) to bounding boxes on every frame.

**Algorithm:**

1. **Filter out non-players**: Accepts all YOLO bounding boxes, sorts them by area (largest first), keeps only the top 2. This eliminates ball boys, line judges, and spectators who appear smaller or at the edges of frame.

2. **Sort by Y-position**: The two remaining boxes are sorted by `feet_y` (the Y coordinate of the bottom edge of the bounding box, which represents where the player's feet are in the image). In image coordinates, Y increases downward:
   - **"far" player** = smaller `feet_y` = higher in the frame = farther from camera
   - **"near" player** = larger `feet_y` = lower in the frame = closer to camera

3. **Assign POI vs OPP**: Based on `poi_side` (which is set at init from `poi_start_side`):
   - If `poi_side == "near"`: near box gets label "poi", far box gets label "opp"
   - If `poi_side == "far"`: far box gets label "poi", near box gets label "opp"

4. **Handle single-player frames**: If only one player is detected, classify their side from Y vs the net midpoint, then assign POI or OPP accordingly.

5. **Smoothing**: Maintains a rolling history of POI Y-positions over the last 15 frames. This prevents flickering if one player briefly disappears.

**The `SideSwitchDetector` class** is responsible for detecting game changeovers — the moments when both players walk to the other end of the court, which means the POI switches from near to far (or vice versa). It fires when:
- Ball has not been visible for ≥ 60 consecutive frames (~2 seconds at 30fps)
- Both players have moved less than 20 pixels in each of the last 45 consecutive frames (~1.5 seconds)

When a changeover is detected, `POITracker.switch_sides()` is called, which flips `poi_side` and clears the smoothing history.

---

### 2.2 Ball Tracking — `cv/detection/ball_tracker.py`

Uses **TrackNet**, a deep learning model specifically designed for tennis ball detection in video. TrackNet operates on a 3-frame sliding window (current frame + 2 previous) and predicts a heatmap of ball probability.

Key characteristics:
- Handles ball disappearance (occluded by player, moving too fast for single-frame detection)
- Returns `(center, confidence, heatmap)` per frame
- Returns `None` if ball is not detected (confidence below threshold)
- Very robust to the motion blur that makes single-frame detectors fail on tennis balls

**Note on Current Models**: The current TrackNet implementations in the community have rapidly evolved. **TrackNetV4 (2024)** introduced a "motion-aware fusion mechanism" with learnable motion attention maps, vastly reducing the false negatives when the ball is occluded or blurred, compared to the older TrackNet variants.

---

### 2.3 Bounce Detection — `cv/analysis/point_detector.py` — `BounceDetector`

**Algorithm:**

A bounce is physically characterised by the ball descending toward the court surface, briefly stopping at the lowest point (highest Y in image coordinates since Y increases downward), then ascending. `BounceDetector` finds these local Y-maxima in the ball trajectory.

Steps:
1. **Gap interpolation**: For frames where ball is `None`, linearly interpolate X and Y positions over gaps of ≤ 10 frames. Longer gaps are left as NaN.

2. **Sliding window analysis**: For each frame `i` with a valid position, compare the Y value to a window of `±3` frames (window=7 by default):
   - `drop` = how much Y increased (descended) before this frame = `ys[i] - min(ys[i-half:i])`
   - `rise` = how much Y will decrease (ascend) after this frame = `ys[i] - min(ys[i+1:i+half+1])`

3. **Bounce criterion**: A bounce is detected if `drop >= 8.0px` AND `rise >= 5.0px` (configurable). These thresholds filter out micro-jitter in the ball tracker output.

4. **Deduplication**: If two potential bounces are within 10 frames of each other, keep only the one with the larger `drop + rise` combined value.

**Optional LSTM mode**: The class supports a pre-trained LSTM bounce predictor model via `BounceDetector(use_lstm=True, model_path=...)`. The LSTM architecture is already defined in the code:
- Input: 16-frame sequence of (x, y) normalised ball positions
- Hidden: 2-layer LSTM with 64 hidden units
- Output: sigmoid probability of bounce at the last frame of the window

If a bounce LSTM checkpoint is ever trained, it can be dropped in without changing any other code.

---

### 2.4 Point State Machine — `PointStateMachine`

Segments the continuous video into discrete tennis points (rallies). It's a formal finite state machine with 5 states:

```
IDLE → SERVING → RALLY → POINT_OVER → back to IDLE
                ↘ CHANGEOVER → back to IDLE
```

**State transitions:**

| From | Condition | To |
|---|---|---|
| IDLE | Ball becomes visible + `MIN_BETWEEN_POINTS` frames since last point | SERVING |
| SERVING | Ball is still visible | RALLY |
| SERVING | Ball disappears for >45 frames | POINT_OVER |
| RALLY | Ball gap > 45 frames | POINT_OVER |
| RALLY | Both players still for >90 frames | CHANGEOVER |
| POINT_OVER | (automatic) | IDLE |
| CHANGEOVER | (automatic) | IDLE |

**Outcome classification** from `_classify_outcome(bounces)`:
- `"error_net"` — no bounce detected at all during the point (ball didn't clear the net or tracker lost it immediately)
- `"error_out"` — the final bounce was out of bounds (homography maps it outside the [0,1]×[0,1] court space)
- `"in_play"` — ball bounced in bounds (further refinement needed to distinguish winner from returned ball)

**Serve side alternation**: The state machine tracks which player is serving and alternates at each `CHANGEOVER`. Starting from `poi_start_side`, it flips between "near" and "far" at each game boundary.

---

### 2.5 Hit Detection — `HitDetector`

Detects racket contacts by finding sharp directional changes in the ball's Y-trajectory.

**Algorithm:**

1. **Smooth the Y-trajectory**: Apply a 5-frame moving average to reduce jitter before looking for directional changes.

2. **Find directional reversals**: For each frame `i`, compute:
   - `prev_diff` = `ys[i] - mean(ys[i-half:i])` — was ball going down (positive) or up (negative)?
   - `next_diff` = `mean(ys[i+1:i+half+1]) - ys[i]` — will ball go up (negative) or down (positive)?
   - A reversal is detected if the sign changes by at least 3 pixels: `(prev_diff > 3 and next_diff < -3)` OR `(prev_diff < -3 and next_diff > 3)`

3. **Assign player**: For each detected hit, compute the distance from the ball to each player's last known center. The hit is assigned to the nearest player ("near" or "far").

4. **Deduplication**: If two potential hits are within 15 frames of each other, skip the second one.

5. **Filter out bounces**: Any potential hit within 8 frames of a confirmed bounce is discarded (bounces also look like directional reversals).

**Speed calculation** (`_calculate_speeds`):
- For each hit, find the next event after it (bounce or another hit)
- Transform both points from video pixels to court coordinates using the homography matrix
- Scale to real-world meters: `x_m = court_x × 10.97m`, `y_m = court_y × 23.77m`
- Add 15% to account for the 3D arc of the ball (ball travels further in 3D than the 2D court projection suggests)
- Compute `speed_kmh = (distance_m × 1.15) / (frames_flown / fps) × 3.6`

**Shot type classification** (`_classify_types`):
- First detected hit of a point → flagged as `"serve"` (regardless of position, one per player maximum)
- Subsequent hits for near player: ball X > player X → `"forehand"`, else `"backhand"`
- Subsequent hits for far player: ball X < player X → `"forehand"` (inverted because the far player faces the opposite direction), else `"backhand"`
- Limitation: assumes all players are right-handed

**Winner/Error assignment** in `PointSegmenter.run()`:
- Last shot of a point where `outcome == "error_net"` or `"error_out"` → `last_shot.is_error = True`, `pt.error_player = last_shot.player`
- Last shot of a point where `outcome == "in_play"` AND the last bounce is in-bounds → `last_shot.is_winner = True`, `pt.outcome = "winner"` (heuristic — see Gap 2)

---

### 2.6 Court Zone Classification — `cv/analysis/court_zones.py`

A complete 24-zone court classification system using normalised court coordinates (0.0–1.0).

**Coordinate system:**
```
(0.0, 0.0) = Far baseline, left doubles sideline
(1.0, 1.0) = Near baseline, right doubles sideline
(0.5, 0.5) = Net center
```

**Key boundaries:**
```
x = 0.000  Left doubles sideline
x = 0.125  Left singles sideline
x = 0.500  Center service line
x = 0.875  Right singles sideline
x = 1.000  Right doubles sideline

y = 0.000  Far baseline
y = 0.231  Far service line
y = 0.500  Net
y = 0.769  Near service line
y = 1.000  Near baseline
```

**24 zones total:**

Far baseline area (6 zones): `far_baseline_AA`, `far_baseline_A`, `far_baseline_B`, `far_baseline_C`, `far_baseline_D`, `far_baseline_DD`

Far service boxes (6 zones): `far_service_left_wide`, `far_service_left_body`, `far_service_left_tee`, `far_service_right_tee`, `far_service_right_body`, `far_service_right_wide`

Near service boxes (6 zones): Same pattern for `near_service_*`

Near baseline area (6 zones): Same pattern for `near_baseline_*`

The `classify(court_x, court_y)` function takes normalised coordinates and returns the matching `CourtZone` object (or `None` if out of bounds).

---

### 2.7 Match Statistics Aggregation — `cv/analysis/match_stats.py`

`MatchStatsAggregator.aggregate(points)` takes the list of `PointRecord` objects from the segmenter and computes the full match statistics object.

**Statistics computed:**

| Stat | Description |
|---|---|
| `total_points` | Number of points/rallies detected |
| `poi_points_won` | Points won by the target player |
| `opp_points_won` | Points won by the opponent |
| `poi_shots` | Total shots hit by target player |
| `poi_winners` | Winners hit by target player |
| `poi_errors` | Errors made by target player |
| `poi_in_play` | Shots that were in-play (neither winner nor error) |
| `poi_winner_pct` | Winners / (Winners + Errors) × 100 |
| `poi_serves_total` | Total serves by target player |
| `poi_first_serves_in` | First serves that landed in |
| `poi_serve_1_pct` | First serve percentage |
| `poi_aces` | Aces (serve winner, opponent didn't touch ball) |
| `serve_zones` | Dict of serve placements: `deuce_t`, `deuce_body`, `deuce_wide`, `ad_t`, `ad_body`, `ad_wide` |
| `rally_lengths` | List of rally lengths per point (in bounces) |
| `avg_rally_length` | Mean rally length |
| `poi_serve_speed_avg` | Average serve speed in km/h |
| `poi_forehand_speed_avg` | Average forehand speed in km/h |
| `poi_backhand_speed_avg` | Average backhand speed in km/h |
| `poi_forehands` | Total forehands hit by target player |
| `poi_backhands` | Total backhands hit by target player |

**Serve zone classification** (`classify_serve_zone`): Maps a serve bounce position to one of 6 named zones (T, Body, Wide for each of Deuce and Ad courts). The deuce/ad alternation is inferred from the point number within the game.

---

### 2.8 Full Pipeline Orchestration — `cv/pipeline.py`

The `AnalyticsPipeline` class is the master orchestrator. When called from `analysis_job.py`, it:

1. Opens the video file and reads metadata (fps, width, height, total frames)
2. Loads pre-confirmed court keypoints from Supabase (14 points placed by the user in the court editor)
3. Builds a homography matrix via `cv2.findHomography()` using the 14 confirmed keypoints vs. the reference court coordinate system. This matrix transforms video pixel coordinates to normalised court coordinates for all subsequent analysis.
4. Loops through every frame (or every Nth frame if `frame_skip > 1`):
   - Runs `BallTracker.detect_ball(frame)` — returns `(center, confidence, heatmap)` or None
   - Transforms ball position to court coordinates via homography
   - Classifies ball court zone from normalised coordinates
   - Runs `PlayerDetector.detect_players(frame)` — returns list of `(bbox, confidence)`
   - Transforms each player center to court coordinates
   - Classifies each player's court zone
   - Appends a `FrameResult` to the results list
5. After frame loop, runs post-processing:
   - Extracts `ball_positions` and `near_positions`/`far_positions` lists from `FrameResult` data
   - Creates `PointSegmenter` with fps and `poi_start_side`
   - Calls `segmenter.run(ball_positions, near_positions, far_positions)` which runs the full bounce detection → state machine → hit detection pipeline
   - Creates `MatchStatsAggregator` and calls `aggregate(points)` for statistics
   - Flattens all shot data into a list of dicts for the frontend

**Output: `AnalysisResult`** dataclass containing:
- Video metadata
- Court keypoints used
- Per-frame data (ball + player positions)
- `match_stats` dict (for `match_data` table)
- `shots` list of dicts (for `shots` table)

---

## 3. The Actual Gaps and Limitations

### 3.1 Gap 1: Player Click Not Wired to `poi_start_side`

**The problem**: When the user clicks on themselves in the player selection screen, those click coordinates are saved to the `player_identifications` table. However, the `analysis_job.py` currently has no code to read this and convert it into the `poi_start_side` ("near" or "far") value that `AnalyticsPipeline` needs.

**Current state**: The `poi_start_side` defaults to `"near"` in the pipeline, meaning it assumes the target player always starts at the bottom of the frame. This is wrong 50% of the time.

**What needs to happen**:
1. `player_selection_job.py` extracts 5 frames and runs YOLO, producing a JSON manifest with bounding boxes for each frame. These bounding boxes are stored in Supabase Storage as `player_selection_frames/{match_id}.json`.
2. When the user clicks a player in the frontend, the frontend should send back which bounding box was clicked (or at minimum the click coordinates).
3. `analysis_job.py` (or `confirm-upload`) needs to:
   a. Load the saved player selection manifest
   b. Match the user's click to the closest bounding box (by center distance)
   c. Compute `poi_start_side = "near" if bbox_center_y > frame_height / 2 else "far"`
   d. Store this on the match record before launching analysis

**Impact**: Without this, the POI is identified incorrectly for half of all matches. All shot attribution, winner/error counts, and serve statistics will be assigned to the wrong player.

---

### 3.2 Gap 2: Winner vs. In-Play — The Hard Problem

**The problem**: The current `PointStateMachine._classify_outcome()` can reliably detect `error_net` and `error_out`, but cannot reliably distinguish between a **winner** and a **returned ball** where the ball tracker temporarily lost the ball.

**Current heuristic** (in `PointSegmenter.run()`):
```python
if pt.outcome == "in_play" and pt.bounces:
    last_bounce = pt.bounces[-1]
    if last_bounce.is_in_bounds:
        last_shot.is_winner = True
        pt.outcome = "winner"
```

This marks any point where the ball bounced in-bounds as a winner — but the ball tracker frequently loses the ball after a fast shot, making it look like the ball wasn't returned when it actually was.

**False positive scenarios** (incorrectly classified as winner):
- Fast passing shot that bounces in-bounds, opponent returns it, but ball tracker loses the ball momentarily during the return
- Ball goes behind a player, tracker loses it, player retrieves and returns it
- Cross-court shot at high speed that bounces in the alley area (correctly detected as in-bounds)

**False negative scenarios** (winner missed):
- Ace where the ball bounces in-bounds but the receiver never touches it (correct!) — but is this always correctly detected?
- Drop shot where ball bounces very close to the net

---

### 3.3 Gap 3: Shot Type Classification Is Weak

**The problem**: `HitDetector._classify_types()` assigns forehand/backhand based purely on the ball's X position relative to the player's X position at the moment of contact. This is an oversimplification.

**Known failure modes**:
1. **Left-handed players**: The forehand/backhand assignment is completely inverted for left-handers
2. **Camera perspective distortion**: The far player's apparent position is compressed horizontally due to perspective, making the X-comparison less reliable
3. **Wide shots**: A player running far to their right to hit a shot may appear to have the ball to their left in the image even though they're hitting a forehand
4. **Volleys**: Players at the net are close to the center service line, making X-comparison unreliable
5. **Overheads**: These have a completely different body position that X-comparison doesn't capture

**What the legacy system has**: `old/src/analysis/tennis_shot_classifier.py` contains a full ML-based shot classifier using YOLOv8-Pose keypoints:
- 17 body keypoints per player
- Features: wrist X/Y relative to body center (normalised by player width), arm extension (elbow-to-wrist distance), arm angle (arctangent of wrist-elbow vector)
- Classifies: `FOREHAND`, `BACKHAND`, `OVERHEAD_SMASH`, `SERVE`
- Works regardless of handedness (because it uses relative wrist position, not absolute)
- Has both rule-based (fast) and ML-trained (accurate) modes
- The ML model is a Random Forest or XGBoost trained on manually annotated frames

**2024 State of the Art**: Current research (2024) heavily favors integrating **YOLOv8-Pose directly with an LSTM or GRU** for swing categorization (distinguishing not just forehand/backhand, but stroke vs. slice vs. volley). YOLOv8-Pose is extremely fast (e.g. ~5.4 ms latency per frame on a mid-range GPU), meaning it is well within reasonable bounds for batch processing.

---

### 3.4 Gap 4: Net Error Detection Is Imprecise

**The problem**: `"error_net"` is currently classified whenever a rally ends without any detected bounce. But ball tracker loss (not a net error) also produces a rally with no detected bounce.

**Sources of false `error_net` classification**:
- Ball tracker fails to detect the ball at all during a fast volley exchange (no bounces, not a net error)
- Ball tracker loses the ball behind the player's body during a baseline exchange
- Ball goes out wide before bouncing, tracker loses it (should be `error_out` but gets classified as `error_net` if bounce isn't detected)

**What would help**: The court keypoints include the net posts (keypoints 12 and 13 in the 14-point model). Using the homography, the Y-coordinate of the net in the video frame can be computed. If the ball's last detected position is near this net Y value AND the ball has been moving downward (into the net), confidence in `error_net` goes up significantly.

---

### 3.5 Gap 5: Side Switching at Changeovers

**The problem**: The `SideSwitchDetector` fires based on "ball not visible + both players still for 45 frames." In practice:
- Ball boys collect the ball (ball not visible, players somewhat still)
- Medical timeouts (ball not visible, players very still)
- Rain delays
- TV timeouts in professional matches

These all look like changeovers to the current detector, which can cause the POI assignment to flip at the wrong time, corrupting all subsequent point data.

**What would help**: A more robust changeover detector would combine:
- Ball gap duration (longer = more likely changeover; >90 seconds = almost certainly a changeover)
- Player positions moving to opposite sides of the net (if the court homography is used, near player's court_y should change from >0.5 to <0.5 after a real changeover)
- Duration of stillness (a medical timeout is different from a changeover in length)

---

## 4. Research: Approaches to Improve Each Gap

### 4.1 Improving Winner Detection

The core challenge is distinguishing: ball bounces in-bounds → opponent retrieves it (in-play) vs. ball bounces in-bounds → opponent cannot reach it (winner).

**Approach A — Dual-Bounce Detection (Recommended first step)**

A true winner bounces twice on the same half of the court without an intervening racket contact. The current `BounceDetector` already collects all bounce positions per point. The fix is:

After the point state machine runs, for each point whose outcome is `"in_play"`:
1. Get all bounces for that point
2. Get all hits for that point
3. For each pair of consecutive bounces `(B1, B2)`, check if:
   - Both bounces are in the same half of the court (same `court_y > 0.5` or `< 0.5`)
   - There is no `HitEvent` with `frame_idx` between `B1.frame_idx` and `B2.frame_idx`
   - The two bounces are separated by ≤ 60 frames (~2 seconds)
4. If this condition is met, the shot preceding `B1` is a winner

This requires no new models, just a post-processing step on data already being collected.

**Approach B — Opponent Movement Analysis**

After the last detected hit in a point, observe the opponent's position over the next 20-30 frames:
- If the opponent's center moves toward the bounce location (distance between opponent and bounce decreases), the ball was likely returned (in-play)
- If the opponent doesn't move toward the bounce, it's likely a winner
- A threshold of "opponent must move at least 30 pixels toward the bounce location within 20 frames" would be a reasonable starting point

This works with existing player tracking data and requires no new models. It would need tuning based on how fast players typically move in Playsight videos.

**Approach C — Ball Trajectory Continuity**

After a bounce in-bounds, a returned ball will show a new directional reversal (the ball going back toward the other half of the court) within 10-30 frames. The ball tracker would detect this as a new hit. If no new hit is detected within ~1 second after the bounce, it was likely a winner (the ball just rolled away).

This is essentially what the current `MAX_BALL_GAP = 45` does at the point level, but could be applied at the shot level: "last bounce in-bounds, and ball disappears within 20 frames with no detected return hit → winner."

**Approach D — LSTM Shot Outcome Classifier (Longer Term)**

Train a small LSTM on sequences of:
- Ball trajectory (x, y) for ~2 seconds around the last hit
- Opponent player position during those frames
- Output: one-hot `{winner, error_net, error_out, in_play}`

Requires labelled training data (manually annotated match videos). Would be the most accurate approach but requires ~100+ labelled points to train reliably.

---

### 4.2 Improving Shot Type Classification

The legacy `ShotClassifier` in `old/src/analysis/tennis_shot_classifier.py` uses **pose keypoints** to achieve significantly better accuracy. The approach:

**Features used** (all normalised to player bounding box width to handle different player sizes and camera distances):

| Feature | Description |
|---|---|
| `wrist_relative_x` | Right wrist X - body center X, divided by player width |
| `wrist_relative_y` | Right wrist Y - body center Y, divided by player width |
| `arm_extension` | Elbow-to-wrist distance, divided by player width |
| `arm_angle` | `atan2(wrist_y - elbow_y, wrist_x - elbow_x)` in degrees |
| `feet_x`, `feet_y` | Player foot position |

**Classification rules (rule-based mode)**:
- **Overhead smash**: `wrist_y < -30%` of player height AND `arm_extension > 25%` AND arm angle > 60° or < -60°
- **Serve**: Similar to overhead but with slightly relaxed thresholds
- **Near player forehand**: `wrist_x > 0` (wrist right of body center)
- **Near player backhand**: `wrist_x < -10%` (wrist left of body center by >10%)
- **Far player**: Uses arm angle and position patterns instead (because the perspective inverts the wrist-side relationship)

**To integrate this into the current pipeline**, the steps would be:
1. Add `YOLOv8-Pose` model loading to `cv/detection/player_detector.py` (it already loads YOLOv8 for bounding boxes; pose is the same model with `-pose` suffix)
2. Add a `detect_player_poses(frame)` method that returns the 17 keypoints per player
3. Store keypoints in `PlayerState` dataclass in `pipeline.py`
4. In `HitDetector._classify_types()`, for each `HitEvent`, look up the player's keypoints at that frame and use the pose-based classification instead of the ball-position-based one

**Performance trade-off**: YOLOv8n-pose (nano) adds approximately 30-50% processing time on CPU. YOLOv8x-pose (extra-large) adds 100-200% but gives better accuracy. For a background batch processing job (not real-time), the extra time is acceptable.

**Handling left-handed players**: With pose keypoints, left-handedness can be detected automatically. If the left wrist is consistently more extended than the right wrist at contact, flag the player as left-handed and swap the forehand/backhand logic.

---

### 4.3 Improving Bounce Detection

**Option 1 — Tune existing thresholds**

The current `min_drop_px = 8.0` and `min_rise_px = 5.0` thresholds are reasonable but may need adjustment for different video resolutions, frame rates, or court perspectives. Lower thresholds catch more bounces but also more false positives (ball tracker jitter). Higher thresholds miss slow-moving or distant bounces.

**Option 2 — Add court surface constraint**

A bounce can only happen within the court boundaries. After bounce detection, filter out any bounce candidate whose court coordinates (from homography) are outside [0.0, 1.0] × [0.0, 1.0]. This eliminates false positives from ball tracker artifacts at frame edges.

**Option 3 — Train the LSTM model**

The `BounceDetector` already has full infrastructure for an LSTM-based bounce predictor. The architecture is defined:
- Input: 16 consecutive (x, y) normalised ball positions
- LSTM: 2 layers, 64 hidden units
- Output: probability of bounce at position 16

To train it:
1. Record a set of match videos with manually annotated bounce timestamps
2. Extract 16-frame windows around each bounce (positive examples) and non-bounce frames (negative examples)
3. Train the LSTM on these sequences
4. Save as a `.pt` file and load via `BounceDetector(use_lstm=True, model_path="bounce_lstm.pt")`

The `s-ganguli TrackNetV2` repository on GitHub has a publicly available bounce LSTM model trained on professional match data. This could be used directly with minor interface adaptation.

---

### 4.4 Improving Error Attribution

**Current approach** (in `MatchStatsAggregator`):
```python
poi_hit_last = (rally_len % 2 == (1 if is_poi_serving else 0))
```
This alternates error attribution based on bounce count — crude but surprisingly decent for short rallies.

**CRITICAL BUG IN CURRENT CODE**: In `match_stats.py`, there is actually a "double counting" bug. `MatchStatsAggregator` tallies errors using the bounce-count parity heuristic above, BUT it later iterates through `point.shots` and tallies explicit errors found by `HitDetector` (`if shot.is_error: stats.poi_errors += 1`). This artificially inflates error counts.

**Better approach**: The `HitDetector` already produces `HitEvent` objects with player attribution. In `PointSegmenter.run()`, the last `HitEvent` of each point is already found. The error is already assigned via:
```python
pt.error_player = last_shot.player
```

The `MatchStatsAggregator` should be updated to strictly use `pt.error_player` directly for top-level stats, removing the bounce-counting heuristic completely to fix the double-counting bug.

---

### 4.5 Upgrading to TrackNetV4

**The problem**: Gaps in the ball trajectory often cause false "error_net" classifications (Gap 4) or false "winner" classifications (Gap 2) when the model loses the ball against complex backgrounds or during extremely fast motion.

**The solution**: Upgrading the ball detection model from the current iteration to **TrackNetV4** (released Sept 2024). 
- **Why it helps**: TrackNetV4 natively integrates dynamic motion-aware attention maps. It captures temporal motion differences between frames far better than V2/V3.
- **Impact**: It dramatically lowers false negative (lost ball) rates during occlusion or rapid volleys. Solving ball continuity at the root level fixes downstream bugs much more robustly than building complex recovery heuristics.

---

## 5. The Legacy CV System (`old/`) — What It Has

The `old/src/` codebase is not used by the web app but contains valuable algorithms. Here's what's worth knowing about:

### `old/src/analysis/tennis_shot_classifier.py`

Contains two complete classes:

**`ShotClassifier`** — Real-time shot detection using pose keypoints:
- Ball proximity detection (ball must be within ~80% of player width to trigger shot classification)
- Pose-based feature extraction
- Both rule-based (fast) and ML model (accurate) classification
- Shot persistence: once a shot is classified, maintains the classification for 30 frames (prevents flickering)
- Handles near player vs. far player differently (different arm angle patterns)

**`TennisShotMLTrainer`** — Full ML training pipeline:
- Loads manually annotated CSV with `(video_file, start_frame, end_frame, player_id, shot_type)` columns
- Extracts pose features from each annotated frame range
- Trains Random Forest, Gradient Boosting, SVM, XGBoost classifiers
- Evaluates each model with cross-validation
- Saves best model as `.pkl` file
- Reports feature importance (what the model actually uses)

### `old/src/core/tennis_CV.py`

3000+ line monolith containing:
- Multi-model ensemble detection (YOLOv8 + RF-DETR for player detection)
- Real-time processing loop with visualization
- Court keypoint stability tracking ("soft-lock" system: once a keypoint is detected with high confidence, it's locked and not replaced by worse detections)
- Ball tracking combining TrackNet + YOLO with confidence-weighted fusion
- CSV output with per-frame data: player bboxes, pose keypoints, ball position, ball confidence, court keypoints

The "soft-lock" keypoint system from this file is worth porting to the `cv/detection/court_detector.py` — it significantly improves court detection stability across frames.

---

## 6. Prioritised Implementation Roadmap

### Priority 1 — Wire Up Player Identification (Fixes incorrect POI tracking for 50% of matches)

**What**: Map user's click from player selection screen → bounding box → `poi_start_side`

**Where**: `backend/api/videos.py` (confirm-upload endpoint) and `cv/analysis_job.py`

**How**:
1. In `player_selection_job.py`, the manifest JSON already contains bounding boxes. Ensure the manifest also includes `frame_height`.
2. In the frontend, when the user clicks a bounding box on the player selection screen, send back the bounding box ID (not just click coordinates) along with the confirm-upload request. Alternatively, send the click (x, y) as normalised coordinates (0–1) relative to the frame.
3. In `analysis_job.py`, before launching `AnalyticsPipeline`:
   - Load the `player_identifications` record for this match
   - Load the player selection manifest from Supabase Storage
   - Find the bounding box closest to the user's click
   - Compute: `poi_start_side = "near" if bbox_center_y > frame_height / 2 else "far"`
   - Pass `poi_start_side` to `AnalyticsPipeline(poi_start_side=poi_start_side)`

**Effort**: 1-2 days
**Impact**: Critical — without this, all player-specific stats are potentially wrong

---

### Priority 2 — Dual-Bounce Winner Detection (Improves winner accuracy significantly)

**What**: Detect winners by finding two in-bounds bounces in the same court half without an intervening hit

**Where**: `cv/analysis/point_detector.py` — add a post-processing step in `PointSegmenter.run()`

**How**:
1. After `_points` are produced by the state machine and hits are assigned:
2. For each point with `outcome == "in_play"`:
   a. Sort the point's `bounces` by `frame_idx`
   b. For each consecutive bounce pair `(B_i, B_{i+1})`:
      - Check both bounces are in the same court half (`court_y > 0.5` or `< 0.5`)
      - Check no `HitEvent` exists in `[B_i.frame_idx, B_{i+1}.frame_idx]`
      - Check time gap: `B_{i+1}.frame_idx - B_i.frame_idx <= fps * 2` (≤ 2 seconds)
   c. If condition met, mark the `HitEvent` immediately before `B_i` as `is_winner = True` and set `pt.outcome = "winner"`

**Effort**: 0.5-1 day
**Impact**: High — directly improves the accuracy of the most important stat

---

### Priority 3 — Fix Double-Counting Error Bug in Stats Aggregation

**What**: Remove the bounce-counting error attribution heuristic and strictly use `HitEvent`-based attribution to prevent double-counting errors.

**Where**: `cv/analysis/match_stats.py` — `MatchStatsAggregator.aggregate()`

**How**:
1. In `aggregate()`, for points with `outcome in ("error_out", "error_net")`:
   - Check `point.error_player` (already populated by `PointSegmenter.run()`)
   - If `error_player == poi_start_side`, increment `stats.poi_errors`
   - If `error_player != poi_start_side`, increment `stats.poi_points_won`
2. Remove the bounce-counting heuristic (`poi_hit_last = rally_len % 2 == ...`).
3. Ensure the Phase 6 loop `if shot.is_error: stats.poi_errors += 1` doesn't double-count the same error if you've already tallied it at the point level above. (Suggest removing the Phase 6 top-level stat mutations entirely).

**Effort**: 1-2 hours
**Impact**: High — currently errors may be double-counted. Fixing this directly restores stat accuracy.

---

### Priority 4 — Opponent Movement Winner Detection (Further improves winner accuracy)

**What**: After the last hit of each point, check if the opponent moved toward the bounce location. If not, it's a winner.

**Where**: `cv/analysis/point_detector.py` — add a method to `HitDetector` or `PointSegmenter`

**How**:
1. After `PointSegmenter.run()` produces points and shots, and after dual-bounce detection (Priority 2) runs:
2. For remaining `"in_play"` points with a last hit:
   a. Get the last `HitEvent` for this point
   b. Determine the opponent of the last hitter ("near" if last_shot.player == "far", else "far")
   c. Get the opponent's positions for frames `[last_hit.frame_idx, last_hit.frame_idx + 30]`
   d. Get the last known bounce position in court space
   e. Compute the opponent's distance to the bounce location at `last_hit.frame_idx` and `last_hit.frame_idx + 20`
   f. If distance decreased by less than 30 pixels in those 20 frames, classify as winner

**Effort**: 1-2 days
**Impact**: Medium-high — catches winners that dual-bounce detection misses (e.g., winner where ball only bounces once before going out of reach)

---

### Priority 5 — Pose-Based Shot Type Classification (High quality, high effort)

**What**: Replace ball-position-based shot type with pose-keypoint-based classification from the legacy system

**Where**: `cv/detection/player_detector.py` and `cv/analysis/point_detector.py`

**How**:
1. Add `YOLOv8-Pose` (n or s variant for speed) to `PlayerDetector`:
   - `self.pose_model = YOLO("yolov8n-pose.pt")`
   - Add `detect_poses(frame)` method returning 17 keypoints per player
2. Add `pose_keypoints` field to `PlayerState` dataclass in `pipeline.py`
3. Store pose keypoints in `PlayerState` during frame processing loop
4. Port `ShotClassifier._extract_ml_features()` and `_classify_near_player_shot()` / `_classify_far_player_shot()` from the legacy system into `HitDetector._classify_types()`
5. For each `HitEvent`, look up the striking player's `PlayerState` at that frame and extract pose features
6. Apply the rule-based classification (or load the trained ML model if available)

**Effort**: 3-5 days (plus ~50% increase in processing time per match)
**Impact**: High for shot type accuracy; needed for accurate forehand/backhand stats

---

### Priority 6 — Net Error vs. Tracker Loss Disambiguation

**What**: Use the net position (from court keypoints) to verify whether a "no-bounce" point genuinely ended with a net error, vs. the ball tracker just losing the ball

**Where**: `cv/analysis/point_detector.py` — `PointStateMachine._classify_outcome()`

**How**:
1. Extract the net Y-coordinate in video pixels from the court keypoints (average Y of keypoints 12 and 13, the net posts)
2. For points classified as `"error_net"`, check the ball's last detected position:
   - If the ball's last Y-position is within 20px of the net Y-coordinate AND the ball was moving downward (Y increasing), confirm as `"error_net"` with high confidence
   - If the ball's last Y-position is far from the net (e.g., near the baseline), reclassify as `"tracker_loss"` — a ball tracker failure, not a genuine net error
3. "tracker_loss" points should be excluded from statistics rather than counted as errors

**Effort**: 1-2 days
**Impact**: Medium — reduces false error_net classifications that inflate error counts

---

### Priority 7 — Bounce LSTM Integration (Longer term)

**What**: Replace the heuristic bounce detector with a pre-trained LSTM model

**Where**: `cv/analysis/point_detector.py` — `BounceDetector`

**How**:
1. Obtain a pre-trained bounce LSTM checkpoint (options: train from scratch on annotated data, or adapt `TrackNetV2`'s public bounce model)
2. Save it as `cv/models/bounce_lstm.pt`
3. In `AnalyticsPipeline`, load `BounceDetector(use_lstm=True, model_path="cv/models/bounce_lstm.pt")`
4. The interface is already implemented — no other code changes needed

**Effort**: 2-4 weeks (data collection + training) OR 2-3 days (if adapting existing public model)
**Impact**: High for bounce detection accuracy, cascades to winner/error classification

---

## 7. Full Feature Status Table

| Feature | Module | Status | Gap Description |
|---|---|---|---|
| Ball tracking per frame | `cv/detection/ball_tracker.py` | ✅ Working | Occasionally loses ball on fast shots |
| Player bounding boxes per frame | `cv/detection/player_detector.py` | ✅ Working | May detect ball boys |
| Court keypoint detection (AI) | `cv/detection/court_detector.py` | ✅ Working | User confirmation required for accuracy |
| Homography (video → court coords) | `cv/pipeline.py` | ✅ Working | Requires ≥4 confirmed keypoints |
| Court zone classification | `cv/analysis/court_zones.py` | ✅ Working | Complete 24-zone system |
| Bounce detection | `cv/analysis/point_detector.py` | ✅ Heuristic | LSTM would improve accuracy |
| Rally segmentation (points) | `cv/analysis/point_detector.py` | ✅ Working | Changeover detection sometimes fires early |
| Hit detection (racket contacts) | `cv/analysis/point_detector.py` | ✅ Working | Some false positives near net |
| Ball speed calculation (km/h) | `cv/analysis/point_detector.py` | ✅ Working | 15% arc correction applied |
| Shot type (FH/BH/Serve) | `cv/analysis/point_detector.py` | ⚠️ Weak | Ball-X vs player-X only; no pose |
| POI identification (near/far) | `cv/analysis/poi_tracker.py` | ⚠️ Partial | `poi_start_side` not wired from user click |
| Side switching at changeovers | `cv/analysis/poi_tracker.py` | ⚠️ Partial | Can fire on timeouts/medical breaks |
| Error net classification | `cv/analysis/point_detector.py` | ⚠️ Imprecise | No net Y confirmation |
| Error out classification | `cv/analysis/point_detector.py` | ✅ Working | Relies on homography being accurate |
| Winner detection | `cv/analysis/point_detector.py` | ⚠️ Heuristic | In-bounds bounce = winner (over-simplification) |
| Error attribution (who made it) | `cv/analysis/match_stats.py` | ⚠️ Heuristic | Bounce-count parity, not actual HitEvent data |
| Match stats aggregation | `cv/analysis/match_stats.py` | ✅ Working | Depends on quality of shot data |
| Serve zone classification | `cv/analysis/match_stats.py` | ✅ Working | Complete T/Body/Wide for Deuce/Ad |
| Serve percentage | `cv/analysis/match_stats.py` | ✅ Working | Requires accurate serve bounce detection |
| Rally length tracking | `cv/analysis/match_stats.py` | ✅ Working | Counts bounces per point |
| ML-based forehand/backhand | `old/src/analysis/tennis_shot_classifier.py` | ✅ (legacy only) | Not yet ported to `cv/` pipeline |
| Overhead smash detection | `old/src/analysis/tennis_shot_classifier.py` | ✅ (legacy only) | Not yet ported to `cv/` pipeline |

---

## 8. Data Flow: End-to-End Pipeline

```
User uploads video
        ↓
player_selection_job.py
  → Downloads video from Supabase Storage
  → Extracts 5 evenly-spaced frames
  → Runs YOLOv8 on each frame → player bounding boxes
  → Stores manifest JSON in Supabase Storage: player_selection_frames/{match_id}.json
        ↓
Frontend: User clicks on themselves in player selection UI
  → Sends click coordinates back to backend
        ↓
[GAP] Backend: map click → bounding box → poi_start_side = "near" | "far"
        ↓
court_setup_job.py (optional AI court detection)
  → Runs CourtDetector on first stable frame
  → Produces 14 keypoint suggestions
  → Stores in court_configs table
        ↓
Frontend: Court editor — user confirms/adjusts 14 keypoints
        ↓
PUT /api/videos/{match_id}/court-keypoints
  → Saves confirmed keypoints to court_configs table
  → Triggers analysis_job.py as background subprocess
        ↓
analysis_job.py
  → Downloads video from Supabase Storage to temp file
  → Loads confirmed keypoints from court_configs table
  → Instantiates AnalyticsPipeline(poi_start_side=...)
  → Calls pipeline.process(video_path, court_keypoints, match_id)
        ↓
AnalyticsPipeline.process()
  ├── Builds homography matrix from 14 confirmed keypoints
  ├── Per-frame loop:
  │     ├── BallTracker.detect_ball() → (center, conf) or None
  │     ├── Apply homography → court coords → court zone
  │     ├── PlayerDetector.detect_players() → [(bbox, conf), ...]
  │     └── Apply homography → court coords per player → court zone
  └── Post-processing:
        ├── PointSegmenter.run(ball_positions, near_positions, far_positions)
        │     ├── BounceDetector.detect() → bounce events
        │     ├── PointStateMachine.process() → PointRecord list
        │     ├── HitDetector.detect() → HitEvent list
        │     └── Assign hits to points → winner/error classification
        └── MatchStatsAggregator.aggregate(points) → MatchStats
        ↓
analysis_job.py saves to Supabase:
  → shots table: one row per HitEvent
  → match_data table: full MatchStats dict
  → matches table: status = "completed", analyzed_at = now()
        ↓
Frontend: Fetches shots + stats, displays court diagram + statistics
```

---

## 9. Output Schema: What the Frontend Receives

### Shots Table (one row per detected shot)

```json
{
  "match_id": "uuid",
  "frame": 1240,
  "x": 412.5,
  "y": 680.3,
  "player": "near",
  "speed_kmh": 87.4,
  "shot_type": "forehand",
  "is_winner": false,
  "is_error": false
}
```

The `x` and `y` coordinates are **video pixel coordinates** (not normalised court coordinates). The frontend's court diagram renders shots by applying an inverse transform from video pixels to the SVG court diagram dimensions. This means the frontend needs to know the original video resolution and the court keypoints to correctly position shots on the diagram — or alternatively, the shots could be stored as normalised court coordinates instead, which would be simpler for the frontend. We should be storing the shots as normalised court coordinates instead of video pixel coordinates. This is a priority during the development.

### Match Data Table (one row per match, `json_data` column)

```json
{
  "total_points": 42,
  "poi_points_won": 24,
  "opp_points_won": 18,
  "poi_shots": 187,
  "poi_winners": 31,
  "poi_errors": 22,
  "poi_in_play": 134,
  "poi_winner_pct": 58.5,
  "poi_serves_total": 21,
  "poi_first_serves_in": 15,
  "poi_serve_1_pct": 71.4,
  "poi_aces": 2,
  "serve_zones": {
    "deuce_t": 4,
    "deuce_body": 3,
    "deuce_wide": 5,
    "ad_t": 1,
    "ad_body": 2,
    "ad_wide": 0
  },
  "rally_lengths": [3, 7, 2, 12, 5, 8, ...],
  "avg_rally_length": 5.2,
  "poi_serve_speed_avg": 142.3,
  "poi_forehand_speed_avg": 98.7,
  "poi_backhand_speed_avg": 84.2,
  "poi_forehands": 89,
  "poi_backhands": 61
}
```

---

*Document generated: February 2026*  
*Status: Research & Planning — no implementation has occurred based on this document*
