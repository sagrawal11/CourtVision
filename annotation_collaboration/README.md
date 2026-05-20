# Annotation Collaboration

Standalone web app for labeling tennis match videos. Exports CSV files compatible with `cv/tools/annotate.py` and `cv/tools/train_models.py`.

**Videos stay on your disk** — only annotations and progress sync through Supabase (tiny data, free tier friendly).

## Workflow

### 1. Download PlaySight videos (your machine)

```bash
source tennis_env/bin/activate   # repo root
python annotation_collaboration/scripts/bulk_download_playsight.py annotation_collaboration/scripts/urls.txt
```

MP4s → `annotation_collaboration/downloads/`. Share those files with your collaborator (Drive, AirDrop, etc.).

### 2. Supabase (one-time)

- Create a project at [supabase.com](https://supabase.com)
- Run `supabase/schema.sql` in the SQL Editor  
  **Already set up?** Also run `supabase/migration_local_disk.sql`
- Copy **Project URL** + **anon** key → `frontend/.env.local`

No storage bucket required for local-disk mode.

### 3. Register videos in the app (metadata only)

```bash
cd annotation_collaboration/frontend
cp .env.example .env.local   # add Supabase URL + anon key
npm install
npm run dev   # http://localhost:3001
```

On the home page, **Register from local MP4s** (reads duration/FPS metadata, does **not** upload files).  
One person can register all 7; or each person registers the videos they will annotate.

**Split work:** agree who does which `video_id` (shown in the list). Click **Start** / pick **in progress** so you do not overlap.

### 4. Annotate

1. Open a video → **Choose local video file** and select the matching MP4 from your disk.
2. Annotate (keyboard shortcuts in **?** cheat sheet). Labels save to Supabase automatically.
3. Close the tab anytime — reopen later, pick the same file again; annotations and `last_frame` resume.

Each session requires choosing the file again (browsers cannot read fixed paths like `/Users/.../match.mp4`).

### 5. Export & train locally

- **Export CSV** per video → save as `cv/training_data/<video_id>_annotations.csv`
- Keep MP4s wherever you already have them for:

```bash
python cv/tools/extract_features.py --video path/to/your.mp4
python cv/tools/train_models.py
```

### 6. Deploy to Vercel (share link with collaborator)

1. **Push this repo to GitHub** (if it is not already there).

2. Go to [vercel.com](https://vercel.com) → **Add New…** → **Project** → import your repo.

3. **Root Directory** (important): click **Edit** and set:
   ```
   annotation_collaboration/frontend
   ```
   Leave Framework Preset as **Next.js**.

4. **Environment variables** — add the same two as `.env.local` (for Production, Preview, and Development):

   | Name | Value |
   |------|--------|
   | `NEXT_PUBLIC_SUPABASE_URL` | `https://YOUR_REF.supabase.co` |
   | `NEXT_PUBLIC_SUPABASE_ANON_KEY` | your `anon` `public` JWT |

   Do not commit `.env.local`; Vercel injects these at build time.

5. Click **Deploy**. First build takes ~1–2 minutes.

6. Copy the URL (e.g. `https://tennis-annotation-xyz.vercel.app`) and send it to your collaborator.

**They use the same Supabase project** — registrations, annotations, and “who claimed which video” are shared. Each person still opens MP4s from their own disk when annotating.

**After changing env vars** in Vercel: **Deployments** → ⋯ on latest → **Redeploy**.

**Optional:** Vercel → Project → **Settings** → **Domains** to use a custom subdomain.

## Collaboration model

| What | Where |
|------|--------|
| MP4 files | Each person's computer |
| Who does which match | `claimed_by` + status on list |
| Annotations | Supabase `annotation_events` |
| Resume frame | Supabase `last_frame` |

## Folder layout

```
annotation_collaboration/
├── README.md
├── scripts/bulk_download_playsight.py
├── downloads/
├── supabase/schema.sql
├── supabase/migration_local_disk.sql
└── frontend/
```

## Security

No login — anyone with the URL can edit. Share the link only with your collaborator.
