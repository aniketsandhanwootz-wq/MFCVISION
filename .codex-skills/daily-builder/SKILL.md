---
name: daily-builder
description: Implement a clear, bounded code or documentation change with a minimal technically correct diff. Use when the task is implementation-focused, the expected outcome is already known, and broad exploration or redesign is not needed.
---

# Daily Builder

## Overview

Use this skill for execution work: feature slices, bug fixes with an already-known cause, small refactors, docs sync, and UI or API updates where the requested result is clear.

## Workflow

1. Read only the files required for the requested change.
2. Infer local invariants from the surrounding code: types, contracts, error handling, and expected state transitions.
3. Make the smallest change that satisfies the request and preserves existing behavior outside the target scope.
4. Re-read the edited path for edge cases, nullability issues, response-shape drift, and unintended side effects.
5. Summarize the change and any obvious verification gaps.

## Technical Standards

- Preserve public API shape unless the user explicitly requests a breaking change.
- Treat types, interfaces, and data contracts as constraints, not suggestions.
- Consider validation, error paths, boundary conditions, and performance-sensitive paths before editing.
- Prefer deterministic logic over clever abstractions.
- Avoid unrelated cleanup, speculative generalization, and wide diff surfaces.

## Do Not Use This Skill When

- the bug is still not understood
- the task is broad architecture design
- the requirements are still ambiguous
- the change would require database mutation for validation

Use `daily-debugger` first when diagnosis is the real task.

## Safety

- Never write to, delete from, reset, migrate, seed, or alter any database unless the user explicitly asks for that exact action.
- If validation would require DB mutation, explain the limitation instead of attempting it.
