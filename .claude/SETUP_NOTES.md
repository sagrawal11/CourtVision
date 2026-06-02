# Claude Code Setup — Notes & Deviations

This repo is configured **exclusively for Claude Code** using the
`claude-code-best-practice` "Command → Agent → Skill" architecture, scaffolded
from `cursor_claude_best_practices_setup.md`.

This file records exactly where the implementation **deviated** from the
literal boilerplate in that instruction document, and why. The Phase 1 directory
structure and Phase 2 file list were followed exactly; the deviations below are
either (a) additive enrichment the doc explicitly invited via `[Cursor: ...]`
placeholders, or (b) corrections required so a file actually functions in this
repo.

## Claude Code exclusivity

- **No Cursor bridge was added** (no `AGENTS.md`, no `.cursor/rules`, no Cursor
  `hooks.json`). Agent configuration lives only in `.claude/` + `CLAUDE.md`.
- **`.cursor/` was removed and git-ignored.** It previously held two
  auto-generated Cursor agent plans and a stale `SAM3_IMPLEMENTATION.md`
  (Dec 2024; referenced the deleted `old/` tree and a `SAM3/` folder not in the
  repo). All three are recoverable from git history if needed.

## Deviations from the instruction document

### 1. `CLAUDE.md` — expanded beyond the minimal template (additive)
The template defined three sections (Project Overview, Core Commands =
Build/Test/Lint, Global Guardrails). The generated file keeps those and adds
**Repository Map**, **Architecture Invariants**, and **Workflow** sections, and
expands Core Commands (run backend/frontend, install) beyond Build/Test/Lint.
- **Why:** the doc asked to fill in project specifics, and the reference repo's
  own guidance is that a developer should be able to say "run the tests" and have
  it work first try — which needs the extra setup/run/invariant context.
- **Constraint respected:** still 85 lines (well under the 200-line cap).

### 2. `.claude/rules/general-stack.md` — changed `paths` frontmatter (correction)
Template: `paths: ["src/**", "app/**", "lib/**"]`.
Generated: `paths: ["frontend/**", "backend/**", "cv/**"]`.
- **Why:** `src/`, `app/`, and `lib/` do not exist at the repo root (the layout
  is `frontend/` + `backend/` + `cv/`; `src/` was removed during cleanup). The
  literal globs would never match a file, so the rule would never auto-load.

### 3. `.claude/hooks/pre-commit.sh` — changed the rejected patterns (correction)
Template rejected any staged diff containing `console.log` **or** `print(`.
Generated hook rejects `console.log`, `debugger`, `pdb.set_trace()`, and
`breakpoint()`, and **intentionally does not reject plain `print(`**.
- **Why:** this Python codebase uses `print()` legitimately in CLI jobs and
  tools (e.g. `cv/detection/player_detector.py`, `cv/tools/*`). The literal hook
  would block almost every Python commit, making it unusable.
- **Also added:** the hook prints offending lines, and a comment documents that
  it is not auto-wired — enable it with `git config core.hooksPath .claude/hooks`.

### 4. `.claude/agents/reviewer.md` — added repo-specific focus (additive)
Kept the template's Role and **verbatim** Output Format, and inserted a bullet
list of this repo's invariants (CV subprocess contract, model paths, RLS,
no secrets/binaries, gated heavy ML) for the reviewer to watch.

### 5. `.claude/skills/run-tests/SKILL.md` — filled command + extra steps (additive)
Filled the placeholder with `pytest tests/` (as intended) and added a
single-file example, a frontend `npm run build`/`lint` fallback, and extra
gotchas about model-weight/video assets and running from the repo root.

## Followed verbatim (no deviation)
- `.claude/settings.json`
- `.claude/commands/review.md`
- Phase 1 directory structure and Phase 2 file set.
