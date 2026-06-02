-- =============================================================================
-- Migration: Add points table for manual point-by-point classification
-- =============================================================================

-- 1. Points table — one row per detected point in a match.
--    CV fills start/end frames and serve_player; coaches fill manual_outcome.
CREATE TABLE IF NOT EXISTS public.points (
    id                  UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    match_id            UUID REFERENCES public.matches(id) ON DELETE CASCADE NOT NULL,
    point_idx           INTEGER NOT NULL,               -- 0-based index within the match
    start_frame         INTEGER,
    end_frame           INTEGER,
    start_timestamp_s   REAL,                           -- seconds into the video
    end_timestamp_s     REAL,
    serve_player        TEXT CHECK (serve_player IN ('near', 'far')),
    rally_length        INTEGER DEFAULT 0,              -- number of shots detected
    manual_outcome      TEXT CHECK (manual_outcome IN (
                            'winner', 'forced_error', 'unforced_error',
                            'ace', 'double_fault'
                        )),
    reviewed_at         TIMESTAMP WITH TIME ZONE,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW()) NOT NULL,
    UNIQUE(match_id, point_idx)
);

CREATE INDEX IF NOT EXISTS idx_points_match_id ON public.points(match_id);

-- 2. Link shots to their point
ALTER TABLE public.shots
    ADD COLUMN IF NOT EXISTS point_idx INTEGER;

-- 3. RLS for points table
ALTER TABLE public.points ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view points for their matches"
    ON public.points FOR SELECT
    USING (
        match_id IN (
            SELECT id FROM public.matches WHERE user_id = auth.uid()
            UNION
            SELECT m.id FROM public.matches m
            JOIN public.team_members tm1 ON tm1.user_id = auth.uid()
            JOIN public.team_members tm2 ON tm2.team_id = tm1.team_id AND tm2.user_id = m.user_id
            JOIN public.users u ON u.id = auth.uid() AND u.role = 'coach'
        )
    );

CREATE POLICY "Users can insert points for their matches"
    ON public.points FOR INSERT
    WITH CHECK (
        match_id IN (SELECT id FROM public.matches WHERE user_id = auth.uid())
    );

CREATE POLICY "Users can update points for their matches"
    ON public.points FOR UPDATE
    USING (
        match_id IN (
            SELECT id FROM public.matches WHERE user_id = auth.uid()
            UNION
            SELECT m.id FROM public.matches m
            JOIN public.team_members tm1 ON tm1.user_id = auth.uid()
            JOIN public.team_members tm2 ON tm2.team_id = tm1.team_id AND tm2.user_id = m.user_id
            JOIN public.users u ON u.id = auth.uid() AND u.role = 'coach'
        )
    );
