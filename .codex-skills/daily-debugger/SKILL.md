---
name: daily-debugger
description: Diagnose a concrete failure from stack traces, logs, failing behavior, or repro steps and identify the first violated invariant. Use when the task is root-cause analysis rather than immediate implementation.
---

# Daily Debugger

## Overview

Use this skill when something is broken and there is real evidence to inspect: an error, a failing path, a bad output, or a reproducible symptom.

## Workflow

1. Start from the observed symptom, not from a guessed fix.
2. Identify the failing boundary: input, contract, state transition, rendering path, or side effect.
3. Trace the code path that transforms the input into the observed failure.
4. Identify the first violated invariant rather than stopping at the final crash site.
5. Separate confirmed cause, likely cause, and missing evidence.
6. Propose the narrowest safe fix and note the likely regression surface.

## What To Look For

- nullability or shape assumptions
- schema drift across module boundaries
- stale derived state or duplicated sources of truth
- incorrect ordering, timing, or async assumptions
- off-by-one and boundary-condition defects
- fallback paths that hide the real failure until later
- resource lifecycle mistakes such as stale handles, invalid state reuse, or missing cleanup

## Output Shape

- symptom
- root cause
- smallest safe fix
- regression risk
- missing evidence if the diagnosis is incomplete

## Safety

- Never mutate a database during debugging.
- Never recommend write/delete/reset/migration SQL unless the user explicitly asks for that exact action in a separate task.
