# Filtering pipeline

```text
Scientific repository and file selection
  -> usable real inputs, when required
  -> successful source adaptation and execution
  -> task derivation and executed references
  -> delivered-data and prompt consistency checks
  -> reference reproduction
  -> evaluator consistency checks
  -> model QC and bounded repair
  -> accepted task packages
```

Reject unresolved execution failures, unavailable required inputs, invalid gold, evaluator contradictions, and low-quality tasks. A missing QC response is not an acceptance decision: quarantine the task for retry or inspection.

See [filtering details](filtering_pipeline_detailed.md) for implementation responsibilities.
