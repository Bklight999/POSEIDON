"""Convert seed source and real input metadata into a structured TaskSpec. Validate decision points and method alternatives; callers reject invalid specifications. Use relative filenames and the shared guarded LLM client."""
from __future__ import annotations

from . import llm_gen
from .taskspec import TaskSpec, DecisionPoint, LOAD_BEARING, COSMETIC

_PROMPT = """You are an expert in scientific data analysis. Below is a piece of analysis code from a real scientific research GitHub repository (discipline: {discipline}).
Data files visible in the repository (relative filenames): {data_files}

Real data schema preview (**strictly follow this; column names / sample names must match it exactly, do not fabricate**):
{data_preview}

Please structure the **complete analysis workflow** represented by this code into a JSON.
**First be clear about two things before writing anything else**: (1) the scientific goal of this analysis (goal); (2) the complete end-to-end chain of steps starting from the data (workflow_steps).
**Focus on the analysis chain that can be done end-to-end directly from the data files previewed above** (e.g., differential expression from a count matrix),
and do not treat upstream steps that cannot be reproduced from the existing data (such as sequence alignment / SRA download / strandedness selection) as decision points.
The goal is to later create questions that "test scientific judgment" based on this, so pay special attention to marking **decision points**: which steps require scientific judgment,
where a wrong choice would affect the conclusion (load_bearing), and which are choices among equivalent correct methods (cosmetic).

Output strictly and only a JSON of the following structure (wrapped in ```json```). **Please output the fields in the order below** (goal and workflow first, then details):
{{
  "seed_id": "short English identifier",
  "goal": "the scientific question / research goal this analysis aims to answer, in one sentence (e.g., assess the effect of hypoxia on differential expression in LNCaP/PC3 prostate cancer cell lines)",
  "workflow_steps": ["the complete chain of analysis steps in the true order of the code, **each step beginning with a number** ('1. Read raw_counts.csv ...', '2. ...', ...): from reading data -> preprocessing -> normalization -> modeling -> diagnostics -> computing the target quantity, one sentence per step, including key parameters/functions", ...],
  "inputs": ["description of raw data / metadata", ...],
  "intermediate_outputs": ["intermediate results, **each annotated at the end with which step produced it**, in the format 'description (step N)' (e.g., 'normalized count matrix (step 6)'), to provide intermediate quantities for workflow_slicing question generation", ...],
  "final_outputs": ["final products", ...],
  "decision_points": [
    {{"name":"English name","kind":"load_bearing|cosmetic","step":the corresponding step number in workflow_steps (integer),
      "correct_choices":["correct approach",...],"wrong_choices":["plausible but wrong approach",...],
      "note":"why it is load-bearing / cosmetic"}}
  ],
  "valid_methods": ["at least 2 scientifically correct methods that should give consistent results"],
  "plausible_wrong_methods": ["plausible but wrong methods, to serve as distractors"]
}}

Requirements: decision_points must contain at least 1 load_bearing, and each decision point's step must point to its corresponding step number in workflow_steps;
valid_methods must contain at least 2 (for cross-validation);
wrong_choices/plausible_wrong_methods must be specific (e.g., "compare raw means directly without library-size normalization");
workflow_steps must be **faithful to the true analysis order of the original code, with step numbers** (this is the reference for downstream answer reproduction, do not fabricate steps).

Code:
```
{code}
```"""


def _mk_decision(d: dict) -> DecisionPoint:
    kind = d.get("kind", "").strip()
    if kind not in (LOAD_BEARING, COSMETIC):
        kind = LOAD_BEARING if "load" in kind.lower() else COSMETIC
    try:
        step = int(d.get("step", 0) or 0)
    except (ValueError, TypeError):
        step = 0
    return DecisionPoint(
        name=d.get("name", "unnamed"), kind=kind,
        correct_choices=d.get("correct_choices", []),
        wrong_choices=d.get("wrong_choices", []),
        note=d.get("note", ""), step=step)


def structure_seed(seed_code: str, discipline: str, data_files: list[str],
                   repo: str = "", data_preview: str = "") -> tuple[TaskSpec | None, list[str]]:
    """Return a valid TaskSpec and issue list, or None with validation issues. data_preview grounds the structure in actual input schemas."""
    code = seed_code  # Keep the full source to preserve scientific context.
    prompt = _PROMPT.format(discipline=discipline,
                            data_files=", ".join(data_files) or "(未知，请从代码推断)",
                            data_preview=data_preview or "(无预览)",
                            code=code)
    # Retain reasoning for workflow structuring and reserve enough tokens for both reasoning and complete structured output.
    
    obj = llm_gen.generate_json(prompt, tag="structure", max_tokens=32000, thinking=True)

    spec = TaskSpec(
        seed_id=obj.get("seed_id", "seed"),
        repo=repo,
        goal=obj.get("goal", ""),
        workflow_steps=obj.get("workflow_steps", []),
        inputs=obj.get("inputs", []),
        intermediate_outputs=obj.get("intermediate_outputs", []),
        final_outputs=obj.get("final_outputs", []),
        decision_points=[_mk_decision(d) for d in obj.get("decision_points", [])],
        valid_methods=obj.get("valid_methods", []),
        plausible_wrong_methods=obj.get("plausible_wrong_methods", []),
    )
    problems = spec.validate()
    return (spec if not problems else None), problems
