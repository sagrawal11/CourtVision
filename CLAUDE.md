# Project Overview

**Courtvision** — a tennis analytics SaaS for coaches and players. A coach
uploads a match video (local file or PlaySight link), confirms the court layout,
and a computer-vision pipeline returns per-shot data and match statistics
(winners, errors, serve placement, rally length, shot speeds, court-zone maps).

Three runtime pieces share one Supabase backbone:
- `frontend/` — Next.js 16 / React 19 / TypeScript app (the product UI).
- `backend/` — FastAPI API: auth, teams, matches, video lifecycle, PlaySight import.
- `cv/` — the Python analytics pipeline (TrackNet ball + YOLO players + court
  homography → points → hits → strokes → stats), run as detached subprocesses.
- `annotation_collaboration/` — separate Next.js app that produces CV training data.

Full docs live in `docs/` — start with `docs/architecture.md` and
`docs/cv-pipeline.md`.

# Core Commands

- **Run backend**: `./start_backend.sh` (Uvicorn on :8000, docs at /docs)
- **Run frontend**: `./start_frontend.sh` (Next dev on :3000)
- **Build (frontend)**: `cd frontend && npm run build`
- **Test (CV/Python)**: `source tennis_env/bin/activate && pytest tests/`
- **Lint (frontend)**: `cd frontend && npm run lint`
- **Install backend deps**: `pip install -r backend/requirements.txt`
  (CV deps — torch, ultralytics, catboost — are installed separately;
  see `docs/cv-pipeline.md`).

> There is no Python build step and no Python linter configured yet. The backend
> is run directly with Uvicorn.

# Repository Map

| Path | What it is |
|---|---|
| `frontend/` | Customer app (App Router pages, shadcn/ui components, Supabase auth) |
| `backend/` | FastAPI routers: `teams`, `matches`, `videos`, `stats`, `activation` |
| `cv/` | Analytics pipeline (`pipeline.py`) + jobs + `detection/` + `analysis/` + `tools/` |
| `cv/models/` | Trained CatBoost models (`.cbm`) for bounce/hit/stroke/point |
| `models/` | Pretrained NN weights (ball/court/player/pose) — **must stay at repo root** |
| `supabase/` | Postgres schema + migrations (auth, RLS, tables) |
| `tests/` | pytest suite (ball tracking, court detection, match stats, PlaySight) |
| `docs/` | Architecture, CV pipeline, video pipeline, deployment docs |

# Architecture Invariants (do not break)

- **CV runs out-of-process.** `backend/api/videos.py` launches `cv/*.py` jobs via
  `subprocess.Popen` with `cwd=PROJECT_ROOT`. `cv/` modules import via
  `from cv.detection ...` with `PROJECT_ROOT` on `sys.path`. Do **not** move `cv/`
  into `backend/` or change these paths without updating the subprocess contract.
- **Model weights load from `PROJECT_ROOT/models/`** (see `cv/detection/*`,
  `cv/analysis/visualizer.py`). Keep `models/` at the repo root.
- **Video never touches the backend.** Browser → Supabase Storage via signed URL
  (or server-side PlaySight download). Source videos are temporary and deleted
  after analysis; only results persist.
- **Court keypoints are user-confirmed.** The 14-point homography is locked from
  the court editor before analysis; analysis only runs when
  `court_setup_status = 'confirmed'`.
- **Supabase RLS isolates data per user**, with an explicit coach-can-read-team
  policy. Never weaken RLS or expose the service-role key to the client.

# Global Guardrails

- **NEVER commit secrets.** `.env`, `.env.local`, and Supabase service-role keys
  stay out of git. The service-role key is backend-only.
- **NEVER commit large binaries.** Model weights (`*.pt`, `*.cbm`, `*.safetensors`)
  and match videos (`*.mp4`/`*.mov`) are git-ignored and must stay that way.
- **Do not bypass type checking or linting.** Fix TypeScript/ESLint errors in the
  frontend rather than suppressing them; don't add blanket `// @ts-ignore` or
  `eslint-disable` to silence real issues.
- **Keep `CLAUDE.md` under 200 lines** and push detailed guidance into
  `.claude/rules/` (path-scoped) or `docs/`.
- **Run the relevant tests before declaring done** — `pytest tests/` for CV
  changes, `npm run build`/`npm run lint` for frontend changes.
- **Match existing conventions.** Don't add narration comments; comments explain
  non-obvious intent only.

# Workflow (Command → Agent → Skill)

This repo uses the `claude-code-best-practice` structure:
- **Commands** (`.claude/commands/`) — slash-command entry points (e.g. `/review`).
- **Agents** (`.claude/agents/`) — specialized personas (e.g. `reviewer`).
- **Skills** (`.claude/skills/`) — reusable procedures (e.g. `run-tests`).
- **Rules** (`.claude/rules/`) — path-scoped coding conventions, auto-loaded.
- **Hooks** (`.claude/hooks/`) — automation around events (e.g. pre-commit checks).
