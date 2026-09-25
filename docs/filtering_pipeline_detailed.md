# Filtering pipeline: implementation details

## Seed selection

Mining identifies scientific analysis files using repository metadata, source content, and data requirements. `stemgen/orchestrator.py` loads candidate seeds and `stemgen/seed_grouping.py` groups related files. Selection behavior follows the configured filters; metadata alone does not prove scientific validity.

## Real input and adaptation checks

With `--require-real-data`, a seed needs usable real inputs. `stemgen/adapt.py` converts source into an executable program while preserving its scientific logic. Bounded repair handles execution problems; unresolved failures and degenerate outputs are rejected. Actual data previews ground file, column, and shape assumptions.

## Structure and reference checks

`stemgen/taskspec.py` validates workflow structure and decision points. `stemgen/reference_gen.py` executes each source-derived reference and compares repeated runs to detect nondeterministic gold. Target-key fallback and execution provenance are retained for downstream review.

## Packaging checks

`stemgen/package_tasks.py` checks the following:

1. **Input availability:** references to data cannot be satisfied by an empty delivery directory.
2. **Prompt consistency:** named inputs and scripts must exist in delivered files. A model-assisted check can distinguish regex false positives from actual missing inputs.
3. **Answer leakage:** generated reference outputs must not be shipped as agent-visible input answers.
4. **Gold reproduction:** rerun the reference using packaged data in an isolated working directory. The implementation has a legacy path-mirroring fallback; strict-only failures can remain warnings, so passing this fallback does not prove strict-layout reproducibility.
5. **Evaluator self-consistency:** the evaluator must accept the reproduced correct answer. Generation also checks incorrect or malformed answers where configured.
6. **Environment consistency:** inspect dependencies and Dockerfile contents before spending further review calls.
7. **Model QC:** review prompt, gold, reference, evaluator, and environment together. Attempt bounded repairs where eligible, then review again before persisting accepted repairs.

Tasks below the configured threshold are rejected. Unavailable or invalid QC decisions are kept pending. Content-derived IDs and directory locks prevent duplicate concurrent packaging; accepted records are written to the package index.

## Optional difficulty estimation

Difficulty estimation is separate from correctness. It depends on the configured solver and available evaluator integrations. The documented default release command uses `--difficulty off`; optional legacy grading integrations are not bundled.
