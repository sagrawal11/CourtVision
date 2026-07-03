-- ============================================================================
-- Drop the orphaned player_identifications table.
--
-- WHY: this table was only ever written by the POST /api/videos/identify-player
-- endpoint, which the frontend never called. That dead endpoint (and its
-- _compute_poi_side helper + PlayerIdentification model) has been removed from
-- backend/api/videos.py. The live POI flow instead computes poi_start_side in
-- the upload modal and persists it via confirm-upload, so nothing reads or
-- writes player_identifications anymore.
--
-- SAFE TO RUN: no code path references this table after the backend cleanup.
-- Dropping it also removes its RLS state and any dependent policies (there were
-- none — it had RLS enabled with no browser policy). Run in the Supabase SQL
-- editor. This is destructive: any historical rows are deleted.
-- ============================================================================

drop table if exists public.player_identifications;
