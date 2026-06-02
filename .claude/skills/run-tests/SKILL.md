# Skill: Run Test Suite
**Trigger:** Use this to verify code changes haven't broken existing functionality.

**Execution Steps:**
1. Activate the project virtualenv and run pytest from the repo root:
   ```bash
   source tennis_env/bin/activate
   pytest tests/
   ```
   The suite covers ball tracking, court detection, dual-bounce detection, match
   stats, and the PlaySight downloader (`tests/test_*.py`).
2. For a single file: `pytest tests/test_match_stats.py -v`.
3. Frontend has no automated test suite yet; verify it with
   `cd frontend && npm run build` (and `npm run lint`).

**Gotchas:**
- Do not skip failing tests; report them immediately.
- Several CV tests need model weights under `models/` and large local match
  videos under `tests/` (git-ignored). If those assets are missing, the test is
  expected to skip/fail on the environment — report that rather than masking it.
- Run from the **repo root** so `from cv...` imports resolve.
