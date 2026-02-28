-- =============================================================================
-- Migration: Analysis Results
-- Run this in Supabase SQL Editor AFTER migration_video_upload.sql has been applied.
-- =============================================================================

-- 1. Add analysis result columns to matches table
ALTER TABLE public.matches
    ADD COLUMN IF NOT EXISTS poi_start_side TEXT DEFAULT 'near',
        -- 'near' | 'far' — which side of the court the target player starts on
    ADD COLUMN IF NOT EXISTS stats JSONB,
        -- Full MatchStats dict from MatchStatsAggregator.to_dict()
        -- e.g. { "total_points": 24, "poi_winner_pct": 31.2, ... }
    ADD COLUMN IF NOT EXISTS analysis_shots JSONB,
        -- Flattened list of all detected shots for visualization
        -- e.g. [{"frame": 1200, "x": 540.0, "shot_type": "serve", ...}, ...]
    ADD COLUMN IF NOT EXISTS analyzed_at TIMESTAMP WITH TIME ZONE,
        -- Timestamp when the full analysis pipeline completed
    ADD COLUMN IF NOT EXISTS analysis_error TEXT;
        -- Error message if the analysis pipeline failed, null on success

-- 2. Extend the shots table to accommodate the new CV output schema
--    The existing schema used start_pos/end_pos (JSONB, NOT NULL) which
--    doesn't match our per-frame hit detection model. We relax the constraints
--    and add the new columns.
ALTER TABLE public.shots
    ALTER COLUMN start_pos DROP NOT NULL,
    ALTER COLUMN end_pos   DROP NOT NULL,
    ADD COLUMN IF NOT EXISTS speed_kmh  REAL,
        -- Estimated ball speed in km/h (derived from homography + frame timing)
    ADD COLUMN IF NOT EXISTS player     TEXT,
        -- 'near' | 'far' — which player hit the shot
    ADD COLUMN IF NOT EXISTS frame      INTEGER,
        -- Frame number in the original video where the hit was detected
    ADD COLUMN IF NOT EXISTS is_winner  BOOLEAN DEFAULT FALSE,
        -- TRUE if this shot was classified as a winner
    ADD COLUMN IF NOT EXISTS is_error   BOOLEAN DEFAULT FALSE;
        -- TRUE if this shot was classified as an error (net or out)

-- Index so the frontend can quickly fetch all shots for a match ordered by frame
CREATE INDEX IF NOT EXISTS idx_shots_match_frame ON public.shots(match_id, frame);
