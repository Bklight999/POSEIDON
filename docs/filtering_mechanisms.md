# Quality gates and optional difficulty checks

Candidate tasks must have sufficient scientific context, usable delivered inputs, an executed reference, and a reliable evaluator. Workflow specifications also validate decision points and method alternatives.

The main checks are implemented in `stemgen/orchestrator.py`, `stemgen/reference_gen.py`, `stemgen/eval_gen.py`, and `stemgen/package_tasks.py`. See [filtering details](filtering_pipeline_detailed.md) for their current behavior.

Historical designs used separate scientific distractor probes and solver-based difficulty gates. A distractor probe asks whether a plausible wrong scientific method changes the answer. This is a useful design criterion, but the separate historical probe module is not bundled here and should not be assumed to run.

Optional difficulty helpers remain in `stemgen/difficulty_gate.py` and `stemgen/difficulty_pass.py`. Their external grading integrations are not included in the default release. Use `--difficulty off` for the documented generation path. Difficulty labels do not establish correctness or replace reference and evaluator validation.
