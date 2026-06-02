---
paths: ["frontend/**", "backend/**", "cv/**"]
---
# Stack Guidelines

This repo spans **Python** (FastAPI backend + `cv/` pipeline) and
**TypeScript/React** (Next.js frontend). Apply the conventions for the area you
are editing.

## Python (`backend/**`, `cv/**`)
- Target **Python 3.10+**. Use type hints and `dataclasses`/`pydantic` models as
  the existing code does (see `cv/pipeline.py`, `backend/api/*.py`).
- The `cv/` pipeline is launched as a **subprocess** by the backend with
  `cwd=PROJECT_ROOT`. Keep `from cv.detection ...` / `from cv.analysis ...`
  absolute imports and the `PROJECT_ROOT` `sys.path` insert intact.
- Load model weights from `PROJECT_ROOT / "models" / ...` (pretrained NN) and
  `PROJECT_ROOT / "cv" / "models" / ...` (trained CatBoost). Don't hardcode
  absolute machine paths.
- Heavy ML work must be **gated to live play / hit frames**, never run per-frame
  on the whole video (see `docs/cv-pipeline.md`).
- Use `logging` for diagnostics in library code; `print()` is acceptable in
  CLI/job entry points and tools (matching current style).

## TypeScript / React (`frontend/**`)
- **Next.js 16 App Router** + React 19. Server Components by default; add
  `"use client"` only when interactivity/hooks are needed.
- Use the existing stack: **shadcn/ui** (Radix) components, **Tailwind CSS 4**,
  **TanStack Query** for server state, **React Hook Form + Zod** for forms.
- Talk to Supabase via the helpers in `frontend/lib/`; call the FastAPI backend
  through `NEXT_PUBLIC_API_URL`. Never embed the Supabase service-role key.
- Fix ESLint/TypeScript errors rather than suppressing them.

## Both
- Never commit secrets or large binaries (`*.pt`, `*.cbm`, `*.mp4`, `*.mov`).
- Respect Supabase RLS; treat the service-role key as backend-only.
