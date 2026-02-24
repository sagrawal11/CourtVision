-- =============================================================================
-- Migration: Video Upload Pipeline + Court Keypoints
-- Run this in Supabase SQL Editor AFTER the base schema.sql has been applied.
-- =============================================================================

-- 1. Make playsight_link optional — we're switching to direct video uploads.
--    Existing rows with a value are unaffected; new rows can omit it.
ALTER TABLE public.matches ALTER COLUMN playsight_link DROP NOT NULL;

-- 2. Add video upload tracking columns to matches
ALTER TABLE public.matches
    ADD COLUMN IF NOT EXISTS s3_temp_key TEXT,
        -- S3 object key for the temporary upload, e.g. "temp-uploads/{match_id}/filename.mp4"
        -- Null until the user uploads a video file.
    ADD COLUMN IF NOT EXISTS video_filename TEXT,
        -- Original filename shown in the UI (e.g. "match_vs_penn_state.mov")
    ADD COLUMN IF NOT EXISTS frame_count INTEGER,
        -- Total frame count populated by the court_setup_job after download.
    ADD COLUMN IF NOT EXISTS court_setup_status TEXT
        CHECK (court_setup_status IN ('pending', 'ready', 'confirmed'))
        DEFAULT 'pending',
        -- Tracks the court editor sub-flow:
        --   pending   = video uploaded, court_setup_job not yet finished
        --   ready     = AI keypoints extracted, awaiting user confirmation
        --   confirmed = user confirmed keypoints, full analysis can start
    ADD COLUMN IF NOT EXISTS debug_video_status TEXT
        CHECK (debug_video_status IN ('generating', 'ready'))
        DEFAULT NULL,
        -- Tracks the debug video rendering job:
        --   NULL       = not yet requested
        --   generating = job running
        --   ready      = debug video available for download
    ADD COLUMN IF NOT EXISTS debug_video_path TEXT;
        -- Supabase Storage path for the rendered debug video.
        -- Set by debug_video_job.py after upload, e.g. "debug-videos/{match_id}/debug.mp4"


-- 3. Create court_configs table.
--    Stores the 14 confirmed court keypoints per match.
--    Keypoint indices match the CourtDetector / CourtReference ordering — see docs/video_pipeline.md.
CREATE TABLE IF NOT EXISTS public.court_configs (
    id              UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    match_id        UUID REFERENCES public.matches(id) ON DELETE CASCADE NOT NULL UNIQUE,

    -- Keypoints 0–13 as individual columns for clean querying and indexing.
    -- Each pair is (x, y) in pixels relative to the video frame dimensions.
    kp0_x REAL,  kp0_y REAL,   -- Far baseline, left doubles sideline
    kp1_x REAL,  kp1_y REAL,   -- Far baseline, right doubles sideline
    kp2_x REAL,  kp2_y REAL,   -- Near baseline, left doubles sideline
    kp3_x REAL,  kp3_y REAL,   -- Near baseline, right doubles sideline
    kp4_x REAL,  kp4_y REAL,   -- Far end, left singles sideline
    kp5_x REAL,  kp5_y REAL,   -- Near end, left singles sideline
    kp6_x REAL,  kp6_y REAL,   -- Far end, right singles sideline
    kp7_x REAL,  kp7_y REAL,   -- Near end, right singles sideline
    kp8_x REAL,  kp8_y REAL,   -- Service line, left end (far)
    kp9_x REAL,  kp9_y REAL,   -- Service line, right end (far)
    kp10_x REAL, kp10_y REAL,  -- Service line, left end (near net)
    kp11_x REAL, kp11_y REAL,  -- Service line, right end (near net)
    kp12_x REAL, kp12_y REAL,  -- Center service line, far end
    kp13_x REAL, kp13_y REAL,  -- Center service line, near end

    ai_suggested    BOOLEAN DEFAULT TRUE,
        -- TRUE = keypoints were placed by CourtDetector AI (not yet user-confirmed)
        -- FALSE = user has confirmed/adjusted the positions
    confirmed_at    TIMESTAMP WITH TIME ZONE,
    confirmed_by    UUID REFERENCES public.users(id),
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW()) NOT NULL
);

-- Index for fast lookup by match
CREATE INDEX IF NOT EXISTS idx_court_configs_match_id ON public.court_configs(match_id);
CREATE INDEX IF NOT EXISTS idx_matches_court_setup_status ON public.matches(court_setup_status);

-- Row Level Security
ALTER TABLE public.court_configs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view court configs for their matches"
    ON public.court_configs FOR SELECT
    USING (
        match_id IN (SELECT id FROM public.matches WHERE user_id = auth.uid())
    );

CREATE POLICY "Users can insert court configs for their matches"
    ON public.court_configs FOR INSERT
    WITH CHECK (
        match_id IN (SELECT id FROM public.matches WHERE user_id = auth.uid())
    );

CREATE POLICY "Users can update court configs for their matches"
    ON public.court_configs FOR UPDATE
    USING (
        match_id IN (SELECT id FROM public.matches WHERE user_id = auth.uid())
    );

-- Coaches can view court configs for any team member's matches
CREATE POLICY "Coaches can view team member court configs"
    ON public.court_configs FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM public.users WHERE id = auth.uid() AND role = 'coach'
        )
        AND match_id IN (
            SELECT m.id FROM public.matches m
            JOIN public.team_members tm1 ON tm1.user_id = auth.uid()
            JOIN public.team_members tm2 ON tm2.team_id = tm1.team_id AND tm2.user_id = m.user_id
        )
    );

-- =============================================================================
-- Storage Bucket: match-videos
-- Used for temporary video uploads and court setup frame images.
-- Free tier: 1 GB total storage, 2 GB bandwidth/month.
-- Convention:
--   temp-uploads/{match_id}/{filename}      ← uploaded video
--   temp-uploads/{match_id}/frame_1000.jpg  ← extracted court setup frame
--
-- Run in Supabase Dashboard → Storage → Create bucket OR via SQL below.
-- =============================================================================

-- Create the bucket (idempotent — safe to re-run)
INSERT INTO storage.buckets (id, name, public, file_size_limit)
VALUES ('match-videos', 'match-videos', false, 1073741824)  -- 1 GB file size limit
ON CONFLICT (id) DO NOTHING;

-- Storage RLS: only authenticated users can upload to their own match folder
CREATE POLICY "Users can upload match videos"
    ON storage.objects FOR INSERT
    WITH CHECK (
        bucket_id = 'match-videos'
        AND auth.role() = 'authenticated'
    );

-- Storage RLS: users can read files in their own match folders
-- The service role key (used by backend) bypasses RLS, so processing jobs can always read/write.
CREATE POLICY "Users can read their match files"
    ON storage.objects FOR SELECT
    USING (
        bucket_id = 'match-videos'
        AND auth.role() = 'authenticated'
    );

-- Storage RLS: users can delete their own uploads
CREATE POLICY "Users can delete their match files"
    ON storage.objects FOR DELETE
    USING (
        bucket_id = 'match-videos'
        AND auth.role() = 'authenticated'
    );

