---
name: layer-readme-section
description: Write or update a layer's section of README.md — overview, per-module architecture, and actual run results pulled from logs. Applies to any pipeline layer (l1_doc_processing, l2_chunking, etc.), not just one. Use when asked to document a layer, write up its run results, or update README after a run.
---

# Layer README Section

Write only the section for the specified layer in README.md. Do not touch any other section.

## Step 0: Identify the layer

Determine which layer from the request, or from context if the user just ran or worked on one. If ambiguous, ask which layer under `src/` before proceeding. Do not guess.

Once identified:
- Layer directory: `src/<layer_name>/`
- Section heading in README.md: `## <layer_name>`

## Step 1: Overview

One paragraph. State what the layer does and its input/output contract — what comes in, where from, what goes out, where to. Read the layer's main orchestrator file (the class that gets called from the module's `main.py`) to confirm the actual contract rather than assuming one. No marketing language, no quality adjectives ("robust," "efficient," "powerful").

## Step 2: Architecture

Read `CLAUDE.md` to get the layer's architecture/file listing. If the layer isn't documented there, stop and state that — do not proceed further, and do not fall back to inferring the structure yourself.

If the layer is documented, list every `.py` file directly under `src/<layer_name>/` and its immediate subdirectories, excluding `__pycache__` and test files. One to two lines per file, describing plainly what that file does — no input/output details, no code-level description.

## Step 3: Results

Locate the log file(s) for this layer under `log/<layer_name>/`.

Rules:
- Every number comes from the logs. Do not estimate, round, or invent throughput, error counts, dedup counts, or timing.
- If a stage-level timing breakdown exists in the log, include it.
- Report whatever the log actually contains for that run — document counts, success/error/skip counts, elapsed time, and any stage breakdown.
- After reporting the numbers, explain what's happening and why — which stage dominates elapsed time, what that stage is computationally doing, why the other stages are negligible by comparison.
- Do not suggest fixes, optimizations, or next steps. Explain the current state only.
- If the run size mentioned in the request isn't in the logs, say so explicitly instead of substituting a different run's numbers.

## Constraints

- No sections beyond Overview / Architecture / Run results — no "Future Work," no "Installation," unless already present elsewhere in README.md and being referenced, not restated.
