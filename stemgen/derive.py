"""Derive task blueprints from a structured workflow. Each blueprint specifies its scientific objective, load-bearing decision, target output, answer type, and method alternatives. Targets must be sensitive to the tested decision so incorrect methods cannot fall within the correct-answer tolerance. The implementation follows the paper's four task families."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any
from . import adapt as _adapt_mod   # Reuse the shared delivery_paths_block path contract.
from . import llm_gen
from .taskspec import TaskSpec

# Map answer types to evaluator types.
ANSWER_TYPES = ["scalar", "count", "set", "ranking", "choice", "composite"]


@dataclass
class TaskBlueprint:
    """Task blueprint produced by derivation and consumed by reference construction."""
    task_id: str
    seed_id: str
    task_type: str                   # original / subworkflow / completion / repair
    prompt: str                      # Scientific task prompt, load-bearing decision point, and target key in executed reference outputs.
    load_bearing_point: str          
    target_quantity_key: str         
    answer_type: str                 # grader type
    valid_methods: list[str]         # Correct and plausible incorrect methods to execute when constructing reference branches.
    wrong_methods: list[str]         
    target_estimand: dict = field(default_factory=dict)  # Precise target estimand, including population, contrast, and scale.
    workflow: list[str] = field(default_factory=list)
    given_step: int = 0              # Intermediate workflow step for subworkflow or completion tasks; zero means not applicable.
    mutation: dict = field(default_factory=dict)      # Repair mutation specification and completion step interface contract.
    removed_step: dict = field(default_factory=dict)  
    data_files: list[str] = field(default_factory=list)  # Input files used by this task.
    composite_parts: list[dict] = field(default_factory=list)  # Component definitions for composite answers.
    source_code_ref: str = ""  # Source fragment supporting the target answer, used for focused reference adaptation.

    def to_dict(self) -> dict:
        return asdict(self)


_PROMPT = """You are an expert at authoring scientific problems. Below is a structured description (TaskSpec) of an analysis workflow, together with the actual source code it is derived from.
Please derive, following the STEM-GitHub method, a set of "full-workflow problems that test scientific judgment".

**IMPORTANT**: The source code below is the ground truth. The TaskSpec is a summary derived from it.
When authoring prompts, refer to the source code to ensure every detail you describe (method names, thresholds, file handling, column names) accurately reflects what the code actually does.
Do NOT invent constraints or behaviors that are not present in the source code.

Source code (the authoritative implementation — use this to verify every claim in your prompts):
```python
{seed_code}
```

TaskSpec:
- Research goal (goal): {goal}
- Full chain of analysis steps (workflow_steps, in real code order, numbered): {workflow_steps}
- Load-bearing decision points (load_bearing, with the corresponding step numbers): {load_bearing}
- Cosmetic decision points (cosmetic): {cosmetic}
- Correct methods (valid_methods): {valid_methods}
- Plausible-but-wrong methods (plausible_wrong_methods): {wrong_methods}
- Visible data files — the EXACT delivery paths the agent receives. Copy them VERBATIM:
{data_paths_block}

Preview of the real data schema (**the target/source sample names for injection-type operators MUST be chosen from the real column names here; do not fabricate them**):
{data_preview}

Iron rules for authoring (must be followed):
1. Every problem must lock in at least 1 load-bearing decision point (getting it wrong -> the answer is wrong).
2. **The target quantity must be sensitive to the locked-in load-bearing decision point**: for example, if you are testing the "normalization" decision, use "count of significant genes" or
   "top-N gene list" as the target quantity (a wrong approach will deviate by orders of magnitude); **do not use the log2FC of a single gene** (insensitive to normalization,
   the wrong approach's answer is also ≈ correct, making the problem meaningless).
3. **NO METHOD/STEP LEAKAGE (critical — this is what makes it test judgment, not execution)**: the problem statement gives ONLY the scientific goal + data + necessary constraints (e.g. threshold definitions). It **MUST NOT** tell the model HOW to solve it — no method names, no tool/library names, no step-by-step recipe, no ordered procedure.
   - ❌ BAD (leaks): "You should train a linear predictor on each embedding", "Build a denoising autoencoder that applies masking noise, then train...", "First normalize by library size, then run DESeq2, then filter by padj", "解析AGP→读FASTA→提取序列→..." (a step list).
   - ✅ GOOD (goal only): "Report the number of genes significantly up-regulated under hypoxia (padj<0.05, log2FC>1)." — states the target + threshold, says nothing about how.
   - Rule of thumb: if a competent scientist could disagree about WHICH method/steps to use, you must NOT name them. Naming a specific method/tool/step order = leakage = the problem tests execution not judgment = REJECT. (Exception: `completion` provides an incomplete program by design; `repair`/`missing_node` show buggy code — but even those must not narrate the *fix* steps.)
4. **The answer to every problem must be derivable/reproducible from the original workflow code** (the downstream reference will actually run to verify), so do not create problems that need external data or cannot be reproduced.
4b. **PROMPT MUST FAITHFULLY DESCRIBE WHAT THE CODE ACTUALLY COMPUTES (critical)**:
   The Gold Program is the authoritative source of truth — it is the original repository code, not a re-implementation.
   The Prompt you write is a *description* of what that code computes. These must match exactly:
   - If the code uses BH correction, do NOT write "Bonferroni" in the Prompt.
   - If the code only processes the first input file, do NOT say "all files" in the Prompt.
   - If the code uses a specific threshold (e.g. padj < 0.05), use that exact threshold in the Prompt.
   - If the code produces output X using method M, the Prompt must ask for output X using method M (or leave method unspecified if multiple valid methods exist).
   - A mismatch between Prompt and Gold Program means the task is unanswerable — it will be rejected in QC.
   - When in doubt: describe ONLY what you can observe in the workflow code; do not invent constraints not present in the code.
5. **DATA FILENAMES — MANDATORY**: when the problem statement mentions input data files, you MUST use the EXACT real filenames listed in "Visible data files" above (verbatim, same name and extension).
   **Do NOT invent, rename, or guess filenames** (e.g. do not write `phenotype.csv` when the real file is `phenotype.txt`, do not write `expression_data/` if no such folder is listed). If the analysis needs a file
   not in that list, do not reference it. The agent receives exactly those files under the current directory; any other filename makes the problem unsolvable.
   **DATA PATHS — copy verbatim, do NOT construct them**: the "Visible data files" list above already
   contains the complete delivery paths (each already starts with `data/`). Reference files by copying
   a line from that list EXACTLY, character for character.
   - Do NOT add another `data/` prefix (writing `data/data/x.csv` when the list says `data/x.csv` is wrong).
   - Do NOT drop or invent sub-directories (if the list says `data/examples/x.csv`, never write `data/x.csv`).
   - Do NOT change the extension or basename (`.fa` vs `.fastq`, `phenotype.csv` vs `phenotype.txt`).
   - If a file you need is not in that list, do not reference it at all.
   This applies to ALL file mentions, including those inside embedded code snippets.
6. **PREFER MULTI-PART (composite) questions for difficulty**: a single scalar answer is easy to guess/luck into. Whenever the workflow naturally yields multiple related quantities, make the problem **composite** — ask for **2-3 related target quantities that must ALL be correct** (e.g. "report (a) the number of up-regulated genes, (b) the number of down-regulated genes, and (c) the log2FC of the top gene"). This forces the model to walk the whole chain, not just hit one endpoint. Aim for **at least half** of the problems you author to be composite (answer_type="composite" with 2-3 composite_parts). Use a single scalar only when the workflow genuinely produces just one meaningful quantity. Each composite part still needs its own target_quantity_key and must be sensitive to the load-bearing decision (rule 2).

Four task types (task_type, please produce a balanced mix):
- original: **Original full problem**. Given the data + scientific goal, have the model perform the full end-to-end analysis from scratch to compute the target quantity.
- subworkflow: **Subworkflow problem**. Given some intermediate result (a quantity tagged with step N in intermediate_outputs below) as a known input,
  have the model continue from that intermediate result to the downstream target quantity, focusing on testing one load-bearing decision in the latter half (hiding the earlier steps).
  The problem statement must state "the <intermediate result> is provided (equivalent to the product of step N of the workflow)".
- completion: **Code completion problem**. **Remove one scientific computation step** from the full analysis (such as normalization / model_fitting / effect_adjustment),
  provide the incomplete program with that step removed (interfaces preserved, syntactically valid), and require the model to reimplement this step based on the upstream/downstream interfaces and scientific goal.
  removed_step must describe the scientific function and the input/output contracts of the removed step. **Do not remove** import / file paths / a single line of ordinary arithmetic / saving a file / column renaming.
  **Explicitly tell the agent which step is missing**; this tests scientific code implementation.
- repair: **Error-diagnosis-and-repair problem**. Based on a mutation mechanism -- inject one controlled scientific error into the **correct workflow** (see the error catalog below),
  the error must be: a real failure mode, substantively affecting the target quantity, leaving observable evidence, having a scientifically defensible fix, and restoring the correct answer after the fix.
  **Do not** inject syntax/import errors, path errors, spelling errors, or errors that crash directly. Provide the mutation spec; the downstream reference will run clean+buggy to verify the answers separate.
Error catalog (repair problems pick one from here; in the first phase only these low-risk errors are allowed):
- duplicate_sample_as_independent: treating repeated records of the same biological individual as independent samples
- wrong_unit_scale: a batch used different units but was not converted
- selected_subset_estimates_population: using only the measured subset to estimate population parameters
- ignored_batch_effect / missing_step: removing an existing necessary step (such as batch correction, normalization, cell-line splitting)
- group_leakage: using ordinary random splitting instead of group-wise splitting
- wrong_direction_or_scale: getting the effect direction/scale/unit backwards

Available intermediate results (for subworkflow / completion to choose from, in the format 'description (step N)'):
{intermediate_outputs}

Every problem must define target_estimand (the precise definition of the scientific target quantity): population (target population) / contrast (control or intervention) /
condition (condition) / scale (mathematical scale, such as log2 fold change / count) / adjustment (adjustment variables); leave empty strings if not applicable.

Output only the following JSON (wrapped in ```json```). **This time produce only the single task_type="{only_type}", exactly {only_n} of them**:
{{
  "blueprints": [
    {{
      "task_id": "short English id",
      "task_type": "original | subworkflow | completion | repair",
      "prompt": "full question text in English (include the relative data file names; subworkflow must state the provided intermediate result and its step; completion must give the incomplete program in the statement; repair must describe the symptom but **not reveal the error location**)",
      "load_bearing_point": "name of the locked-in load-bearing decision point (taken from load_bearing above)",
      "target_quantity_key": "the variable name of this target quantity in the reference script (snake_case, e.g. n_up_genes)",
      "target_estimand": {{"population":"", "contrast":"", "condition":"", "scale":"", "adjustment":[]}},
      "answer_type": "scalar|count|set|ranking|choice|composite",
      "given_step": "for subworkflow/completion problems: the step number (integer) corresponding to the step given as a known condition / being asked / being removed; otherwise fill in 0",
      "valid_methods": ["≥2 snake_case names of correct methods; the reference will compute each one"],
      "wrong_methods": ["≥1 snake_case name of a plausible-but-wrong method"],
      "workflow": ["the analysis steps the model must go through, with [decision] marking decision points"],
      "mutation": {{"category":"an id from the error catalog", "affected_step":"the affected step number or name", "violated_assumption":"the violated assumption", "observable_signal":"the observable evidence", "correction":"the correct fix"}} or {{}} (only needed for repair problems),
      "removed_step": {{"node":"name of the removed step", "scientific_function":"its scientific function", "input_contract":"input contract", "output_contract":"output contract"}} or {{}} (only needed for completion problems),
      "composite_parts": [{{"key":"sub-item name","type":"choice|scalar",...}}] or [],
      "source_code_ref": "the exact lines or function(s) from the source code above that compute the target quantity for this problem — copy the relevant code snippet verbatim. This tells the downstream gold_program generator exactly which part of the source to use, avoiding guesswork. For repair, include both the correct code AND the mutated version."
    }}
  ]
}}"""


_VALID_TYPES = ["original", "subworkflow", "completion", "repair"]


def _parse_blueprint(bp: dict, spec, data_files, lb) -> "TaskBlueprint":
    tt = bp.get("task_type", "original")
    if tt not in _VALID_TYPES:
        tt = "original"
    try:
        given_step = int(bp.get("given_step", 0) or 0)
    except (ValueError, TypeError):
        given_step = 0
    return TaskBlueprint(
        task_id=bp.get("task_id", "task"),
        seed_id=spec.seed_id,
        task_type=tt,
        prompt=bp.get("prompt", ""),
        load_bearing_point=bp.get("load_bearing_point", lb[0] if lb else ""),
        target_quantity_key=bp.get("target_quantity_key", "answer"),
        answer_type=bp.get("answer_type", "scalar"),
        valid_methods=bp.get("valid_methods", []),
        wrong_methods=bp.get("wrong_methods", []),
        target_estimand=bp.get("target_estimand") or {},
        workflow=bp.get("workflow", []),
        given_step=given_step,
        mutation=bp.get("mutation") or {},
        removed_step=bp.get("removed_step") or {},
        data_files=list(data_files),
        composite_parts=bp.get("composite_parts", []),
        source_code_ref=bp.get("source_code_ref", ""),
    )


def derive_tasks(spec: TaskSpec, data_files: list[str],
                 counts_df=None, data_preview: str = "",
                 per_type: int = 2, workers: int = 12,
                 seed_code: str = "") -> tuple[list[TaskBlueprint], dict]:
    """Generate blueprints for the four paper task types using independent requests."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    lb = [f"{d.name}(step {d.step})" if d.step else d.name for d in spec.load_bearing_points()]
    lb_names = {d.name for d in spec.load_bearing_points()}
    cos = [d.name for d in spec.decision_points if d.name not in lb_names]
    fmt = dict(
        goal=spec.goal or "(未提供)",
        workflow_steps=spec.workflow_steps or "(无)",
        load_bearing=lb, cosmetic=cos,
        valid_methods=spec.valid_methods, wrong_methods=spec.plausible_wrong_methods,
        intermediate_outputs=spec.intermediate_outputs or "(无)",
        # Supply exact delivered paths through the shared contract rather than asking the model to infer prefixes, extensions, or subdirectories.
        
        
        data_paths_block=_adapt_mod.delivery_paths_block(data_files),
        data_preview=data_preview or "(无预览)",
        seed_code=seed_code or "(源码未提供)")

    # Generate each task type/index in an independent request. STEM_EXCLUDE_TYPES can disable comma-separated task types.
    
    import os as _os
    _excl = {t.strip() for t in _os.getenv("STEM_EXCLUDE_TYPES", "").split(",") if t.strip()}
    _types = [t for t in _VALID_TYPES if t not in _excl]
    if _excl:
        print(f"  [derive] 排除题型 {sorted(_excl)}，本次只产 {_types}")
    jobs = [(tt, i) for tt in _types for i in range(per_type)]

    def _gen_one(job):
        tt, i = job
        prompt = _PROMPT.format(only_type=tt, only_n=1, **fmt)
        # Reserve enough output tokens for reasoning and a complete blueprint JSON object.
        try:
            obj = llm_gen.generate_json(prompt, max_tokens=32000, tag=f"derive_{tt}_{i}", thinking=True)
        except Exception as e:  # noqa: BLE001 One failed blueprint must not block the others.
            print(f"  [derive] {tt}#{i} 失败({e})，跳过")
            return None
        got = obj.get("blueprints", [])
        if not got:
            return None
        bp = got[0]
        bp["task_type"] = tt
        # Append type and sequence number to avoid concurrent task-ID collisions.
        bp["task_id"] = f"{tt}_{i}_{bp.get('task_id','t')}"
        return _parse_blueprint(bp, spec, data_files, lb)

    blueprints = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_gen_one, j): j for j in jobs}
        for fu in as_completed(futs):
            r = fu.result()
            if r is not None:
                blueprints.append(r)
    print(f"  [derive] 并发产出 {len(blueprints)}/{len(jobs)} 道蓝图")
    return blueprints, {}   # variant_data is always empty.
