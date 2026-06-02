#!/bin/bash
# Pre-commit hook to prevent accidental commits of debug statements.
#
# NOTE: this repo's Python code uses print() legitimately (CLI jobs/tools), so we
# do NOT reject plain print(). We only flag genuine debugging leftovers:
#   - JS/TS:   console.log, debugger
#   - Python:  pdb.set_trace(), breakpoint()
#
# Wire it up with:  git config core.hooksPath .claude/hooks

staged_diff=$(git diff --cached)

if echo "$staged_diff" | grep -nE "console\.log|debugger;?|pdb\.set_trace\(\)|breakpoint\(\)" >/dev/null; then
  echo "REJECTED: Remove debug statements (console.log / debugger / pdb.set_trace / breakpoint) before committing."
  echo "Offending lines:"
  echo "$staged_diff" | grep -nE "console\.log|debugger;?|pdb\.set_trace\(\)|breakpoint\(\)"
  exit 1
fi

exit 0
