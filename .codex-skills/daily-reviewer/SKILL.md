---
name: daily-reviewer
description: Review code changes for correctness, regressions, security issues, risky assumptions, and missing tests. Use when the user asks for a review, wants a diff inspected, or needs a senior-engineer quality check before commit or PR.
---

# Daily Reviewer

## Overview

Use this skill after an implementation change when the main question is not "how to build it" but "what could be wrong with it".

## Workflow

1. Inspect the changed files and the nearest impacted paths.
2. Check correctness first: invariants, type assumptions, control flow, fallback behavior, and response contracts.
3. Check regression risk: edge cases, state transitions, parsing/serialization boundaries, and performance-sensitive paths.
4. Check security-sensitive behavior when relevant: file handling, trust boundaries, injection surfaces, auth assumptions.
5. Report findings first, ordered by severity. If there are no findings, say so explicitly.

## Review Standards

- Prioritize bugs, behavioral regressions, security risks, and missing tests.
- Ignore stylistic nits unless they hide a real correctness or maintenance issue.
- Distinguish confirmed defects from lower-confidence risks.
- Keep findings concrete and actionable.
- If a concern depends on runtime behavior you cannot observe, state it as an assumption or residual risk.

## Output Shape

- Findings first.
- Brief summary second.
- Mention missing verification only when it matters.

## Safety

- Never write to, delete from, reset, migrate, seed, or alter any database.
- If DB behavior is relevant, reason from code inspection and available evidence only.
