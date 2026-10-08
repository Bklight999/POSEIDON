# Production flow

```text
Repository search -> scientific file selection -> workflow grouping
  -> environment setup and verified adaptation -> task derivation
  -> reference execution and evaluator generation -> QC and packaging
```

The pipeline begins with real research source code, then turns verified workflows into reproduction, subworkflow, completion, and correction tasks. Gold answers come from execution with staged inputs. Packaging checks input availability, reference reproducibility, evaluator behavior, and scientific consistency.

A delivered task directory contains:

```text
<task-type>/<difficulty>/<task-id>/
  task.md
  data/
  gold_program.py
  eval_script.py
  dockerfile
  gold_lineage.json
```

Completion and repair tasks also carry the relevant incomplete or buggy code. Content-derived task IDs support deduplication and resumable production. `tasks_with_id.jsonl` indexes accepted packages.

Use the repository README for setup and a small local run. Configure source paths and credentials before larger runs; see [operations](operations.md) for distributed execution considerations.
