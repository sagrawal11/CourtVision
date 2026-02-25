-- Add poi_start_side column to matches table to support selecting the POI visually
ALTER TABLE public.matches
ADD COLUMN poi_start_side text CHECK (poi_start_side IN ('near', 'far'));

-- Update status check constraint to include 'generating_frames' and 'player_selection'
ALTER TABLE public.matches
DROP CONSTRAINT IF EXISTS matches_status_check;

ALTER TABLE public.matches
ADD CONSTRAINT matches_status_check CHECK (status IN ('pending', 'generating_frames', 'player_selection', 'court_setup', 'processing', 'completed', 'failed'));
