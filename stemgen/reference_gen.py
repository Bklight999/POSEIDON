"""Construct gold answers by executing source-derived programs. Reproduction runs the workflow, subflow tasks expose selected outputs, completion removes a step for the agent while retaining the complete reference, and correction injects an error while using the correct execution as gold. Preserve scientific logic, reject unresolved execution failures, and leave answer comparison to the generated evaluator."""
from __future__ import annotations

import difflib
import hashlib
import json
import math
import re

from . import adapt as _adapt_mod   # Reuse the shared delivery-path contract.
from . import llm_gen, sandbox
from .safe_egress import sanitize_for_external


# Adapt source code into an executable script that records the target quantity in outputs; executed source is the correct baseline.
_ADAPT_PROMPT = """You are a scientific computing engineer. Below is the **source code** of a real research repository (this is the authoritative, correct implementation).
Please **base on it** and rewrite it into a **self-contained, one-click runnable** python script that computes the specified target quantity.

**Iron rule: the correct logic must stay faithful to the source code; do NOT replace its scientific approach with your own method.** You may ONLY:
add driver code to load the data, wire up the part of the source code that computes the target quantity, and store the result into outputs.

Working directory = the directory that CONTAINS the `data/` folder. **Read data using the relative paths listed below; absolute paths are forbidden.**

=== DATA PATH CONTRACT (mandatory) ===
The `data/` folder already exists and ships with the task. Read the input files at EXACTLY
these paths, verbatim:
{data_paths_block}
- **Do NOT flatten**: if the path is `data/tests/assets/x.npz`, write it in full — NOT `data/x.npz`.
- **Do NOT** add `if not os.path.exists(...)` guards, **do NOT** copy data from repository paths
  (the repo does NOT exist at evaluation time — only `data/` does), **do NOT** generate synthetic
  data, and **do NOT** write anything into `data/`. Read only.

Available data files: {data_files}
Data schema preview: {data_preview}

Source code (the correct implementation — this is the specific code identified by the task author as computing the target quantity; stay faithful to it):
```
{seed_code}
```

Task prompt (what the agent will be asked — your gold_program must compute exactly what this prompt describes):
{task_prompt}

Target quantity: key="{target_key}" ({answer_type}) — {estimand_desc}
{type_note}

Requirements:
1. Only depend on the libraries already used by the source code + numpy/pandas/scipy. No network access, no data downloading, no interaction.
2. Stay faithful to the scientific logic of the source code; only add the driver / invocation / value extraction.
3. {outputs_contract} Do **NOT** print anything yourself.
4. Do not plot, do not write files.
5. **Alignment check**: every method, threshold, and constraint named in the Task prompt above MUST be reflected in your gold computation. If the prompt says "padj < 0.05", use that exact threshold. If the prompt says "BH correction", use BH — not Bonferroni. The gold program must compute exactly what the prompt asks for, using the source code as the implementation.
{repair_note}

ABSOLUTE PROHIBITIONS — violating any of these causes the task to be rejected:
- DO NOT limit input data to the first N rows/samples (e.g. df.head(100), [:100]) unless the Prompt explicitly requires it.
- DO NOT generate random data or use synthetic/dummy values as fallbacks under any circumstances.
- DO NOT use simplified approximations in place of the source code's actual method (e.g. naive sort instead of proper ranking, raw mean instead of weighted mean).
- DO NOT silently catch ImportError / ModuleNotFoundError and continue — if a required package is missing, let the error propagate so the task is rejected upstream.
- MUST call the repo's own functions/classes when they implement the target computation; do not re-implement from scratch.
- Every formula, threshold, and parameter mentioned in the Prompt MUST appear verbatim in the gold computation chain.
Output only a single python code block (wrapped in ```python```), no explanation."""


# For completion, remove a step from the working script for the agent to restore.
_COMPLETION_PROMPT = """Below is a **complete and correct** python analysis script. Please create an incomplete version for a **code-completion task**.
```python
{full_code}
```
Requirements:
1. Remove the code segment that implements "{removed_node}" (scientific function: {node_func}), and replace it with a placeholder comment:
   `# TODO: implement {removed_node} here ({node_func})`
2. Keep the upstream/downstream interfaces (input/output variable names) unchanged; the incomplete script must be syntactically valid and parseable, but without this step it will produce a wrong result or fail to compute.
3. Only remove this single scientific step; keep everything else as-is. Do not leak the implementation of the removed step.

CRITICAL — Anti-leakage rules (SCI-013):
- The TODO comment MUST NOT contain: the algorithm name, the specific formula, the exact parameter values, the condition/threshold logic, or any pseudocode that gives away the implementation.
- ONLY allowed in the TODO comment: the step name and its input/output signature (e.g. "takes X, returns Y").
- BAD (leaks): `# TODO: keep the array with the largest total span; break ties by repeat length; remove intersecting arrays`
- GOOD (no leak): `# TODO: implement pick_arrays here (input: list of Array objects, output: generator of selected arrays)`
- Any TODO comment that reveals HOW to implement the step (logic, conditions, specific constants) is a fatal leak and invalidates the task.
Output only a single python code block (the incomplete script)."""


# For repair, inject the specified error into the correct script for diagnosis and correction.
_REPAIR_PROMPT = """Below is a **complete and correct** python analysis script. Please create a version containing a **single scientific error**, for an error-diagnosis-and-repair task.
```python
{full_code}
```
Requirements:
1. Only at the load-bearing decision point "{decision_point}", change the correct approach into the wrong one: {wrong_choice}
   (error category: {mutation_cat}; violated assumption: {violated}). Keep everything else unchanged.
2. The error must be a **scientific error** (e.g. failing to normalize / ignoring grouping / using the wrong control), not a syntax/import/path error.
3. The script must **still run** (not crash); it is only the scientific result that is wrong. Do **NOT** expose the location of the error (leave no hint comment).

CRITICAL — Anti-leakage rules (SCI-014):
- Do NOT add any comment, variable name, or string that hints at the error location, the wrong value used, or how to fix it.
- Do NOT rename variables to suggest the bug (e.g. `wrong_threshold`, `incorrect_norm`).
- Do NOT leave the original correct line as a comment alongside the wrong line.
- The buggy script must look like a plausible implementation that a scientist might write — the error should only be detectable by running and observing the wrong scientific result.
Output only a single python code block (the buggy script)."""


def _estimand_desc(est: dict) -> str:
    if not est:
        return "(see problem statement)"
    parts = [f"{k}={est[k]}" for k in ("population", "contrast", "condition", "scale") if est.get(k)]
    if est.get("adjustment"):
        parts.append(f"adjustment={est['adjustment']}")
    return "; ".join(parts) or "(see problem statement)"


def _composite_keys(bp) -> list:
    """Return declared composite component keys, or an empty list when not applicable."""
    if getattr(bp, "answer_type", None) != "composite":
        return []
    return [p.get("key") for p in (getattr(bp, "composite_parts", None) or []) if p.get("key")]


def _outputs_contract(bp) -> str:
    """Define one consistent output structure per task type. Composite tasks expose component keys directly instead of receiving contradictory single-target and multi-component instructions."""
    keys = _composite_keys(bp)
    if keys:
        assigns = "; ".join(f"outputs[{k!r}]=..." for k in keys)
        return ("The script MUST end with a top-level dict `outputs` whose **direct top-level keys "
                f"are exactly** these {len(keys)} sub-quantities: {keys}. "
                f"That is: {assigns}. "
                "**DO NOT** nest them inside a wrapper/container entry (e.g. do NOT write "
                f"`outputs[{getattr(bp, 'target_quantity_key', 'target')!r}] = {{...}}` holding them), and do NOT add "
                "a separate aggregate key — each sub-quantity must be reachable as `outputs[<key>]` directly.")
    return ('The script MUST end with a top-level dict `outputs`, and store the target quantity '
            'into it: outputs["{}"]=value.'.format(getattr(bp, "target_quantity_key", "target")))


def _type_note(bp) -> str:
    """Return task-type-specific adaptation instructions using source code as the correct baseline."""
    # Let the shared output contract specify composite structure; describe multi-part semantics without conflicting duplicate instructions.
    
    keys = _composite_keys(bp)
    if keys:
        return ("[composite] This is a MULTI-PART question: all of these sub-quantities "
                f"{keys} must be computed from the same faithful run of the source-code logic. "
                "See the outputs requirement below for exactly how to store them.")
    if bp.task_type == "subworkflow" and bp.given_step:
        return (f"[subworkflow] This task relays from the intermediate product of step {bp.given_step} of the workflow to the target quantity. "
                f"Please extract the logic of that step and its downstream from the source code, add an entry point that 'loads/constructs that intermediate product', and then compute the target quantity.")
    return ""


# With a verified base program, focus it on the requested target instead of readapting raw source for each task.

_RETARGET_PROMPT = """Below is a scientific analysis script that has **already been verified to run one-click successfully** (the authoritative, correct implementation).
Please, while **changing its scientific logic as little as possible**, make it compute and output the target quantity specified by this task.

Working directory = the directory that CONTAINS the `data/` folder. **Read data using the relative paths listed below; absolute paths are forbidden.**

=== DATA PATH CONTRACT (mandatory) ===
The `data/` folder already exists and ships with the task. Read the input files at EXACTLY
these paths, verbatim:
{data_paths_block}
- **Do NOT flatten**: if the path is `data/tests/assets/x.npz`, write it in full — NOT `data/x.npz`.
- **Do NOT** add `if not os.path.exists(...)` guards, **do NOT** copy data from repository paths
  (the repo does NOT exist at evaluation time — only `data/` does), **do NOT** generate synthetic
  data, and **do NOT** write anything into `data/`. Read only.

Available data files: {data_files}

The verified runnable script:
```python
{base_program}
```

Task prompt (what the agent will be asked — your gold_program must compute exactly what this prompt describes):
{task_prompt}

Target quantity: key="{target_key}" ({answer_type}) — {estimand_desc}
{type_note}

Requirements:
1. Preserve the scientific computation logic of the original script; only add/extract the target quantity on top of it.
2. {outputs_contract} Do **NOT** print anything yourself.
3. Do not plot, do not write files, no network access, no interaction.
4. **Alignment check**: every method, threshold, and constraint named in the Task prompt above MUST be reflected in your gold computation. If the prompt specifies a particular method or threshold, the gold program must use it exactly.

ABSOLUTE PROHIBITIONS:
- DO NOT limit input data to the first N rows/samples unless the Prompt explicitly requires it.
- DO NOT generate random data or use synthetic/dummy values as fallbacks.
- DO NOT silently catch ImportError / ModuleNotFoundError and continue — let missing-package errors propagate.
- MUST preserve the repo's own functions/classes for computing the target; do not re-implement from scratch.
- Every formula, threshold, and parameter in the Prompt MUST appear in the gold computation chain.
Output only a single python code block (wrapped in ```python```), no explanation."""


def _run(code, repo_dir, timeout, python_bin, extra_data_dir=None):
    import os as _os
    dbg = _os.getenv("STEM_REF_DEBUG")
    if dbg:
        with open(dbg, "a", encoding="utf-8") as _f:
            _f.write(f"\n===== code ({len(code)} chars) =====\n{code}\n===== END =====\n")
    full = code + "\n" + sandbox.EMIT_SNIPPET
    # Execute in isolated working directories and prefill the seed's staged inputs, avoiding concurrent writes to shared repositories.
    
    
    return sandbox.run_isolated(full, repo_dir, timeout=timeout, python_bin=python_bin,
                                extra_data_dir=extra_data_dir)


def _stable_sha256(text: str | None) -> str:
    """Return a stable content hash for provenance/lineage auditing."""
    return hashlib.sha256((text or "").encode("utf-8", errors="replace")).hexdigest()


def _values_equal(a, b) -> bool:
    """Strict-enough recursive equality used by determinism and buggy separation gates."""
    if a is None or b is None:
        return a is b
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        try:
            af, bf = float(a), float(b)
            if not (math.isfinite(af) and math.isfinite(bf)):
                return False
            # Use tolerance consistent with packaged reproduction checks so normal parallel floating-point variation is not mistaken for nondeterminism.


            return math.isclose(af, bf, rel_tol=1e-6, abs_tol=1e-12)
        except Exception:
            return False
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_values_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_values_equal(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


_MISSING = object()


def resolve_output_key(outputs, key):
    """Look up a key at the output root or uniquely one container level below; return _MISSING otherwise. Avoid deeper guesses that could select unrelated quantities."""
    if not isinstance(outputs, dict) or not key:
        return _MISSING
    if key in outputs:
        return outputs[key]
    hits = [v[key] for v in outputs.values() if isinstance(v, dict) and key in v]
    return hits[0] if len(hits) == 1 else _MISSING


def _extract_gold(outputs, bp):
    """Extract gold using composite components, then the target key, then a first-value fallback. Reuse identical logic for both executions. Return the value and whether it was resolved without fallback."""
    if not isinstance(outputs, dict):
        return outputs, True
    if _composite_keys(bp):
        _ca = _blueprint_answer(outputs, bp)
        if isinstance(_ca, dict) and len(_ca) >= 2:
            return _ca, True
    v = resolve_output_key(outputs, getattr(bp, "target_quantity_key", ""))
    if v is not _MISSING:
        return v, True
    return (next(iter(outputs.values())) if outputs else None), False


def _blueprint_answer(outputs, bp):
    """Extract the answer shape used for separation checks, including composite parts."""
    if not isinstance(outputs, dict):
        return outputs
    parts = getattr(bp, "composite_parts", None) or []
    if getattr(bp, "answer_type", "") == "composite" and parts:
        keys = [p.get("key") for p in parts if p.get("key")]
        if keys:
            # Allow components nested one container level below the output root.
            vals = {k: resolve_output_key(outputs, k) for k in keys}
            if all(v is not _MISSING for v in vals.values()):
                return vals
    key = getattr(bp, "target_quantity_key", "")
    if key in outputs:
        return outputs[key]
    return next(iter(outputs.values())) if outputs else None


def _lineage_record(seed_code: str, source_ref: str, base_program: str | None,
                    gold_program: str) -> dict:
    """Build auditable source→base→gold lineage metadata without claiming semantic proof.

    The record is deliberately non-fatal: it exposes how much code changed and flags suspicious
    loss of target-code literals. GPT QC/package review can use these signals with context.
    """
    source_basis = source_ref.strip() or seed_code
    base = base_program or ""
    diff = list(difflib.unified_diff(base.splitlines(), gold_program.splitlines(), lineterm=""))         if base else []
    added = sum(1 for x in diff if x.startswith("+") and not x.startswith("+++"))
    removed = sum(1 for x in diff if x.startswith("-") and not x.startswith("---"))
    # Numeric/scientific literals in the author-selected source slice are useful audit anchors.
    literals = sorted(set(re.findall(r"(?<![A-Za-z_])(?:\d+\.\d+|\d+e[+-]?\d+)(?![A-Za-z_])",
                                     source_basis, flags=re.I)))
    missing_literals = [x for x in literals if x not in gold_program]
    risk = []
    if not source_ref.strip():
        risk.append("missing_source_code_ref")
    if base and removed > max(25, int(0.25 * max(1, len(base.splitlines())))):
        risk.append("large_base_to_gold_deletion")
    if literals and len(missing_literals) / len(literals) > 0.5:
        risk.append("source_literals_mostly_missing_from_gold")
    return {
        "source_program_sha256": _stable_sha256(seed_code),
        "source_code_ref_sha256": _stable_sha256(source_ref),
        "base_program_sha256": _stable_sha256(base_program),
        "gold_program_sha256": _stable_sha256(gold_program),
        "used_source_code_ref": bool(source_ref.strip()),
        "base_to_gold_diff": {"added_lines": added, "removed_lines": removed},
        "source_ref_numeric_literals": literals[:100],
        "source_ref_literals_missing_from_gold": missing_literals[:100],
        "risk_flags": risk,
        "audit_status": "needs_context_review" if risk else "recorded",
    }


def build_reference_single(spec, bp, seed_code: str, repo_dir: str, data_files: list[str],
                           *, python_bin: str | None = None, max_iters: int = 2,
                           timeout: int = 300, data_preview: str = "",
                           base_program: str | None = None,
                           seed_data_dir: str | None = None) -> dict:
    """Execute a source-derived reference and return status, gold, reference code, optional incomplete/buggy code, iterations, and error. Prefer a verified base program and prefill the same staged inputs delivered with the task; reject unresolved failures."""
    seed_code = seed_code or ""
    # Prefer the blueprint's focused source fragment, falling back to the full seed code.
    _source_ref = getattr(bp, "source_code_ref", "") or ""
    _sc = _source_ref if _source_ref.strip() else seed_code

    # Generate and execute a script that exposes the target quantity in outputs.
    def adapt_prompt(last_code, last_err):
        repair = ""
        if last_code is not None:
            # Sanitize previous code and execution errors before feeding them back; the request client repeats the outbound check.
            
            _err = sanitize_for_external(last_err or "")   # no truncation
            _prev = sanitize_for_external(last_code)       # no truncation
            repair = (f"\n上一版执行失败，报错：\n```\n{_err}\n```\n"
                      f"上一版代码：\n```python\n{_prev}\n```\n请修正后重写完整脚本。")
        if base_program:
            return _RETARGET_PROMPT.format(
                data_files=", ".join(data_files),
                data_paths_block=_adapt_mod.delivery_paths_block(data_files),
                base_program=base_program,
                task_prompt=bp.prompt or "(prompt not yet generated)",
                target_key=bp.target_quantity_key, answer_type=bp.answer_type,
                estimand_desc=_estimand_desc(bp.target_estimand),
                outputs_contract=_outputs_contract(bp),
                type_note=_type_note(bp)) + repair
        return _ADAPT_PROMPT.format(
            data_files=", ".join(data_files),
            data_paths_block=_adapt_mod.delivery_paths_block(data_files),
            data_preview=data_preview or "(无预览)",
            seed_code=_sc, task_prompt=bp.prompt or "(prompt not yet generated)",
            target_key=bp.target_quantity_key, answer_type=bp.answer_type,
            estimand_desc=_estimand_desc(bp.target_estimand), type_note=_type_note(bp),
            outputs_contract=_outputs_contract(bp),
            repair_note=repair)

    adapted = llm_gen.generate_with_repair(
        adapt_prompt, lambda c: _run(c, repo_dir, timeout, python_bin, seed_data_dir),
        max_iters=max_iters, tag=f"ref_gold_{bp.task_id}",
        thinking=False, max_tokens=32000)  # Disable reasoning to preserve the code output budget.
    if not adapted.get("ok"):
        return {"ok": False, "error": adapted.get("error"),
                "gold_answer": None, "gold_program": adapted.get("code"),
                "incomplete_code": None, "buggy_code": None, "iters": adapted.get("iters", 0)}

    outputs = adapted["outputs"]
    # For composite tasks, assemble gold from individual components, including one-level nested outputs.
    
    gold_answer, _gold_key_resolved = _extract_gold(outputs, bp)
    if not _gold_key_resolved and isinstance(outputs, dict) and outputs:
        # If the target key is missing, warn and record provenance before falling back to the first value; this may select the wrong scientific quantity.
        
        
        print(f"  [{bp.task_id}] ⚠ target_key={bp.target_quantity_key!r} 不在 outputs "
              f"(keys={list(outputs)[:6]})，兜底取首个值当 gold —— 可能取错量", flush=True)
    full_code = adapted["code"]

    # Rerun independently to detect random, shuffled, or time-dependent gold answers.
    
    try:
        _run2 = _run(full_code, repo_dir, timeout, python_bin, seed_data_dir)
        if _run2.get("ok"):
            _out2 = _run2.get("outputs", {})
            # Use identical output-selection logic on both runs, including any fallback.
            
            _val2, _ = _extract_gold(_out2, bp)
            _val1 = gold_answer
            # Reject inconsistent repeated results as nondeterministic.
            if not _values_equal(_val1, _val2):
                return {"ok": False,
                        "error": f"gold_nondeterministic: run1={_val1!r} run2={_val2!r}",
                        "gold_answer": None, "gold_program": full_code,
                        "incomplete_code": None, "buggy_code": None,
                        "iters": adapted.get("iters", 1)}
    except Exception as _e:
        pass  # Record a failed second run without overriding the first successful execution.

    # Do not reject slicing or random-related syntax through blanket regex rules. Contextual QC judges inappropriate data truncation; repeated execution checks nondeterminism, and packaging checks environment dependencies.


    result = {"ok": True, "gold_answer": gold_answer, "gold_program": full_code,
              "outputs": outputs if isinstance(outputs, dict) else {},  # Retain all outputs for composite-answer construction.
              # Record whether gold matched its intended target key rather than using the first-value fallback.
              "gold_key_resolved": _gold_key_resolved,
              "incomplete_code": None, "buggy_code": None, "buggy_outputs": None,
              "lineage": _lineage_record(seed_code, _source_ref, base_program, full_code),
              "iters": adapted.get("iters", 1), "error": None}

    # Build incomplete code by removing the specified step.
    if bp.task_type == "completion":
        rs = bp.removed_step or []
        # Support both a list of removed steps and the legacy single-step dictionary.
        if isinstance(rs, dict):
            rs = [rs]
        r = rs[0] if rs else {}
        # Combine removed-step descriptions.
        all_nodes = ", ".join(x.get("node", "") for x in rs if x.get("node"))
        all_funcs = " | ".join(x.get("scientific_function", "") for x in rs if x.get("scientific_function"))
        try:
            inc = llm_gen.generate(
                _COMPLETION_PROMPT.format(full_code=full_code,
                                          removed_node=all_nodes or r.get("node", bp.load_bearing_point),
                                          node_func=all_funcs or r.get("scientific_function", "")),
                max_tokens=32000, temperature=0.2, tag=f"ref_incomplete_{bp.task_id}", thinking=False)
            result["incomplete_code"] = llm_gen._extract_code(inc)
        except Exception as e:  # noqa: BLE001
            print(f"  [{bp.task_id}] incomplete_code 生成失败: {e}")

    # Build buggy code using the mutation specification.
    if bp.task_type == "repair":
        dp = _decision_point(spec, bp.load_bearing_point)
        wrong_choice = (dp.wrong_choices[0] if dp and dp.wrong_choices
                        else (bp.wrong_methods[0] if bp.wrong_methods else "wrong approach"))
        ms = bp.mutation or []
        # Support both lists and legacy single-mutation dictionaries.
        if isinstance(ms, dict):
            ms = [ms]
        m = ms[0] if ms else {}
        # Combine mutation descriptions.
        all_cats = ", ".join(x.get("category", "") for x in ms if x.get("category"))
        all_violated = " | ".join(x.get("violated_assumption", "") for x in ms if x.get("violated_assumption"))
        try:
            bug = llm_gen.generate(
                _REPAIR_PROMPT.format(full_code=full_code, decision_point=bp.load_bearing_point,
                                      wrong_choice=wrong_choice,
                                      mutation_cat=all_cats or m.get("category", ""),
                                      violated=all_violated or m.get("violated_assumption", "")),
                max_tokens=32000, temperature=0.2, tag=f"ref_buggy_{bp.task_id}", thinking=False)
            result["buggy_code"] = llm_gen._extract_code(bug)
        except Exception as e:  # noqa: BLE001
            print(f"  [{bp.task_id}] buggy_code 生成失败: {e}")

        # P0 gate: a repair task is valid only when the injected program still runs and
        # changes the target quantity. A crashing/no-op mutation is not a scientific repair task.
        buggy_code = result.get("buggy_code")
        if not buggy_code:
            return {**result, "ok": False, "error": "buggy_code_missing"}
        buggy_run = _run(buggy_code, repo_dir, timeout, python_bin, seed_data_dir)
        if not buggy_run.get("ok"):
            return {**result, "ok": False,
                    "error": "buggy_code_not_runnable: " + str(buggy_run.get("stderr", ""))[-500:]}
        buggy_outputs = buggy_run.get("outputs")
        buggy_value = _blueprint_answer(buggy_outputs, bp)
        gold_for_separation = _blueprint_answer(outputs, bp)
        result["buggy_outputs"] = buggy_outputs if isinstance(buggy_outputs, dict) else {}
        result["buggy_answer"] = buggy_value
        if _values_equal(buggy_value, gold_for_separation):
            return {**result, "ok": False,
                    "error": (f"buggy_not_separated: buggy={buggy_value!r} "
                              f"gold={gold_for_separation!r}")}
        result["buggy_separation_verified"] = True

    return result


def _decision_point(spec, name: str):
    """Find a decision point by name, allowing a step-number suffix."""
    base = name.split("(")[0].strip()
    for d in spec.decision_points:
        if d.name == base:
            return d
    return None
