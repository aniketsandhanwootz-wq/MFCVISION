# Repository Instructions

This repository uses a small, speed-focused engineering workflow.

## Shared Working Rules

- Prefer small, local changes over broad refactors.
- Preserve existing behavior outside the requested scope.
- Treat request and response shapes as contracts.
- Keep error handling explicit and avoid silent fallbacks unless already established in the code.
- When reviewing changes, prioritize correctness, regression risk, edge cases, and missing verification.
- When debugging, identify the first violated invariant rather than patching the final crash site.

## Database Safety

- Do not write to, delete from, truncate, drop, alter, migrate, seed, or reset any database unless the user explicitly asks for that exact action.
- Treat database access as read-sensitive by default.
- If validating a change would require a database mutation, explain that limitation instead of attempting it.

## Custom Roles

This repo has matching VS Code agents in `.github/agents/` and Codex skills in `.codex-skills/`.

- `daily-builder`: implement a clear, bounded change with a minimal technically correct diff
- `daily-reviewer`: review diffs for bugs, regressions, security issues, and missing tests
- `daily-debugger`: diagnose root cause from concrete symptoms before changing code

Use these roles only when they reduce effort. For tiny edits, direct manual changes are usually faster.
