# Scientific workflow task synthesis

A seed is an executable analysis workflow with identifiable inputs, scientific decisions, and measurable outputs. Several files may form one seed; a repository may contain several independent seeds.

## Design principles

**Executed answers.** The source-derived reference computes the gold answer. The language model interprets and adapts code, but its scientific intuition is not evidence for a numerical result.

**Meaningful decisions.** Questions should require selecting or implementing scientifically justified methods. An alternative is useful only if it changes the requested scientific result, rather than merely changing formatting or implementation style.

**Grounded inputs.** Inspect actual input schemas and retain the required data. Avoid invented columns, inaccessible resources, leaked reference outputs, and instructions referring to files that are not delivered.

**Verifiable output.** Define the estimand, units, answer structure, and tolerance. A scalar, structured object, or composite response is acceptable when its evaluator reliably distinguishes correct and incorrect answers.

## Derivation patterns

- Reproduction asks for the original scientific result.
- Subworkflow tasks target a coherent section of the computation with enough upstream context.
- Intermediate tasks request a scientifically interpretable internal quantity.
- Completion supplies incomplete code and identifies the missing step.
- Correction supplies a scientifically flawed program and symptoms without revealing the fix.
- Missing-step diagnosis silently removes or bypasses a required operation while retaining apparently plausible execution.

Suitable errors include inappropriate train/test splits, ignored grouping, omitted normalization, missing batch adjustment, and invalid statistical assumptions. Errors must preserve the intended scientific task and provide sufficient evidence for diagnosis.

## Candidate validation

Execute the correct reference with delivered inputs, check repeated runs, verify evaluator behavior, then review prompt, data, gold, program, and environment jointly. Repairing wording must not redefine gold. Environment or evaluator repairs require renewed validation. Reject unresolved inconsistencies and keep unavailable review decisions pending.

The implementation and current stages are summarized in [method.md](method.md). Earlier experimental proposals involving separate distractor-probe modules are not part of the bundled default workflow; use the included modules and CLI as the operational reference.
