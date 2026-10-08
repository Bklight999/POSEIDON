# Source-grounded scientific task generation

The pipeline converts executable research workflows into verifiable scientific-agent tasks. Source code provides the scientific reference; model-generated answers are never accepted without execution. Models identify workflows, propose tasks, adapt interfaces, and write prompts and evaluators.

## Workflow

1. **Mine repositories and files.** Search public research repositories, inspect README content, and select scientific Python files with data inputs and measurable outputs.
2. **Group seeds.** Combine files belonging to one analysis chain. Cache successful groups by repository identity, file content, prompt version, and context budgets so resumed runs retain stable seed names.
3. **Prepare execution.** Discover actual data, establish dependencies, and adapt source into a standalone program. Repairs may adjust imports, paths, and interfaces but must preserve scientific computations. Execute in a temporary working directory and retain the inputs needed by the task.
4. **Structure the workflow.** Extract the scientific objective, ordered steps, target quantities, and decisions whose alternatives materially change the result. Ground column names and schemas in observed data.
5. **Derive tasks.** Generate variants from the verified workflow using the task families below. Preserve required context and available inputs; avoid revealing the answer or prescribing every scientific choice.
6. **Execute references.** Focus the verified program on each target, execute it with the staged data, and check repeated results for determinism. Store the executed program, gold answer, and provenance.
7. **Generate evaluators.** Match output structure, units, tolerances, and accepted representations to the task. Check that correct outputs pass and incorrect or malformed outputs fail.
8. **Package and review.** Validate delivered files, reproduce gold, test evaluator consistency, and perform model QC with bounded repair. Accept only tasks meeting the configured quality threshold; keep unavailable QC decisions pending.

## Task families

| Code identifier | Agent task |
|---|---|
| `original` | Reproduce an end-to-end scientific analysis from its objective and inputs. |
| `subworkflow` | Complete a scientifically meaningful section of a larger workflow. |
| `completion` | Restore an explicitly identified missing scientific step. |
| `repair` | Diagnose and correct an injected scientific mistake. |

Composite answers combine related quantities to test consistency across the analysis chain. Scientific mutations should represent meaningful failure modes such as leakage, inappropriate normalization, incorrect grouping, or missing controls; syntax and path errors are not substitutes for scientific reasoning.

## Contracts and reproducibility

- Use relative data paths in task prompts and programs. Reference only files delivered in the task's `data/` tree.
- Preserve real input data when real-data mode is enabled. Do not substitute fabricated data to make a failing workflow pass.
- Generated reference programs expose an `outputs` dictionary through the runner's output sentinel. Composite tasks use consistent component keys.
- Gold must be produced by the reference execution. Missing target-key fallbacks are recorded in provenance and require careful QC.
- Generated evaluators expose `evaluate(pred, gold)`. Their tolerances should reflect numerical variability without accepting scientifically wrong results.
- Content-derived task IDs support deduplication and resumable packaging. Grouping and sampling use stable fingerprints and seeds.
- Isolated working directories, process-group timeouts, and per-child memory limits contain execution failures. These controls do not replace container or host isolation.

## Implementation map

| Responsibility | Module |
|---|---|
| Seed grouping | `stemgen/seed_grouping.py` |
| Environment and source adaptation | `stemgen/env_setup.py`, `stemgen/adapt.py` |
| Workflow specification | `stemgen/structure.py`, `stemgen/taskspec.py` |
| Task derivation | `stemgen/derive.py` |
| Executed references | `stemgen/reference_gen.py`, `stemgen/sandbox.py` |
| Evaluator generation | `stemgen/eval_gen.py` |
| Scheduling and persistence | `stemgen/orchestrator.py` |
| Validation and packaging | `stemgen/package_tasks.py` |

See [filtering details](filtering_pipeline_detailed.md), [production flow](production_flow.md), and the repository README for runnable entry points. Historical experimental counts and private deployment settings are intentionally excluded from this release.
