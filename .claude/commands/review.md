# Command: /review
**Description:** Triggers a code review of uncommitted changes.

**Workflow Pipeline:**
1. Check `git status` for modified files.
2. Invoke `.claude/agents/reviewer.md`.
3. Pass changes to the agent and output results.
