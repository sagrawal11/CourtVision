# Cursor AI Instruction: Scaffold Claude Code Architecture

Hello Cursor! Please read this entire document carefully. Your task is to set up the `claude-code-best-practice` architecture in the root of this repository. This architecture transitions Claude Code into a structured "Command → Agent → Skill" workflow.

## Phase 1: Directory Setup
Create the exact following directory structure in the root of the workspace:
```text
.claude/
├── agents/
├── commands/
├── hooks/
├── rules/
├── skills/
│   └── run-tests/
└── settings.json
CLAUDE.md
```

## Phase 2: Boilerplate File Creation
Please create the following files with the provided boilerplate content. Where you see `[Cursor: ...]`, analyze the current workspace and fill in the appropriate details based on the project's tech stack.

### 1. `CLAUDE.md` (Root Directory)
This is the global system prompt. It must be kept under 200 lines.
```markdown
# Project Overview
[Cursor: Briefly describe the project here based on the current workspace]

# Core Commands
- Build: `[Cursor: Insert build command]`
- Test: `[Cursor: Insert test command]`
- Lint: `[Cursor: Insert lint command]`

# Global Guardrails
- [Cursor: Add 2-3 strict rules based on the project's tech stack (e.g., "NEVER bypass type checking errors")]
```

### 2. `.claude/settings.json`
Configure the CLI environment.
```json
{
  "thinkingMode": true,
  "outputStyle": "Explanatory",
  "customCommandsPath": ".claude/commands",
  "agentsPath": ".claude/agents",
  "allowedTools": ["Bash", "Glob", "View", "Edit", "Replace"]
}
```

### 3. `.claude/rules/general-stack.md`
Path-specific context rules.
```markdown
---
paths: ["src/**", "app/**", "lib/**"]
---
# Stack Guidelines
[Cursor: Add 3-4 coding conventions specific to the primary languages/frameworks used in this repo]
```

### 4. `.claude/agents/reviewer.md`
A specialized sub-agent persona.
```markdown
# Role
You are a meticulous Senior Engineer reviewing code. Focus on edge cases, security, and performance.

# Output Format
Output a bulleted list of flaws. If none, output "LGTM." Do not write code.
```

### 5. `.claude/commands/review.md`
The slash command to trigger the reviewer agent.
```markdown
# Command: /review
**Description:** Triggers a code review of uncommitted changes.

**Workflow Pipeline:**
1. Check `git status` for modified files.
2. Invoke `.claude/agents/reviewer.md`.
3. Pass changes to the agent and output results.
```

### 6. `.claude/skills/run-tests/SKILL.md`
A standardized skill documentation file.
```markdown
# Skill: Run Test Suite
**Trigger:** Use this to verify code changes haven't broken existing functionality.

**Execution Steps:**
1. [Cursor: Insert the specific command to run the test suite for this repo]

**Gotchas:**
- Do not skip failing tests; report them immediately.
```

### 7. `.claude/hooks/pre-commit.sh`
A sample bash hook. (Ensure you make this file executable after creating it).
```bash
#!/bin/bash
# Pre-commit hook to prevent accidental commits of debug statements
if git diff --cached | grep -q "console.log\|print("; then
  echo "REJECTED: Remove debug print statements before committing."
  exit 1
fi
exit 0
```

## Phase 3: Finalization
After generating these files and directories, output a message confirming the `claude-code-best-practice` setup is complete. Prompt the user to review the generated files and verify the framework-specific assumptions you made.
