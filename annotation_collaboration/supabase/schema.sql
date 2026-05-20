-- Annotation Collaboration — run in Supabase SQL Editor.
-- Videos stay on annotators' disks; only metadata + labels live in Supabase.

-- ── Tables ───────────────────────────────────────────────────────────────────

create table if not exists public.annotation_videos (
  id uuid primary key default gen_random_uuid(),
  title text not null,
  video_id text not null unique,
  storage_path text,
  expected_filename text,
  fps double precision not null default 30,
  total_frames integer not null default 1,
  duration_sec double precision,
  status text not null default 'ready'
    check (status in ('ready', 'in_progress', 'done')),
  claimed_by text,
  last_frame integer not null default 0,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists idx_annotation_videos_status on public.annotation_videos (status);

create table if not exists public.annotation_events (
  id uuid primary key default gen_random_uuid(),
  video_uuid uuid not null references public.annotation_videos (id) on delete cascade,
  frame integer not null,
  event_type text not null,
  stroke_type text,
  point_end_type text,
  player text not null default '',
  created_at timestamptz not null default now(),
  unique (video_uuid, frame, event_type)
);

create index if not exists idx_annotation_events_video on public.annotation_events (video_uuid);
create index if not exists idx_annotation_events_frame on public.annotation_events (video_uuid, frame);

-- ── updated_at trigger ───────────────────────────────────────────────────────

create or replace function public.annotation_videos_set_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists trg_annotation_videos_updated on public.annotation_videos;
create trigger trg_annotation_videos_updated
  before update on public.annotation_videos
  for each row execute function public.annotation_videos_set_updated_at();

-- ── RLS (open anon — private link is your only access control) ───────────────

alter table public.annotation_videos enable row level security;
alter table public.annotation_events enable row level security;

drop policy if exists "anon_all_videos" on public.annotation_videos;
create policy "anon_all_videos" on public.annotation_videos
  for all to anon using (true) with check (true);

drop policy if exists "anon_all_events" on public.annotation_events;
create policy "anon_all_events" on public.annotation_events
  for all to anon using (true) with check (true);

-- Storage is optional (legacy cloud-upload mode only).
-- Skip bucket setup if you only use local-disk mode.
