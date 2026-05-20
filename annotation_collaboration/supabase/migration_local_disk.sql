-- Run this if you already created tables from the older schema (cloud upload mode).

alter table public.annotation_videos
  alter column storage_path drop not null;

alter table public.annotation_videos
  add column if not exists expected_filename text;

-- Prevent duplicate video_id when two people register the same match name
create unique index if not exists annotation_videos_video_id_key
  on public.annotation_videos (video_id);
