"""Generate evaluation scripts in two stages: plan metrics, tolerances, and structure from the task and executed gold, then write the evaluator. The source program's executed output is authoritative; evaluate agreement with that gold through the shared LLM client."""
from __future__ import annotations

import re

from . import llm_gen

_PLANNER_PROMPT = """You are an expert in scientific data analysis and evaluation methodology. Design a **grading plan** for the following scientific analysis question,
to be implemented by the coder as a grading function. Grading goal: determine whether "the final answer given by the model" scientifically agrees with "the authoritative gold answer".

Question:
{prompt}

Target quantity: key="{target_key}" (answer type {answer_type})
Authoritative gold answer (from a real run of the github source code): {gold}
Task type: {task_type}

Produce a concise, unambiguous grading strategy that gives **exactly one** approach, covering:
1. Answer type determination (numeric / count / set / ordering / classification / text / composite)
2. Grading metric and tolerance (numeric: give relative/absolute tolerance; count: allow ±; set: use recall/intersection; text/complex: use semantic equivalence)
3. Pass threshold
4. Special considerations (e.g. for repair tasks judge "whether the fixed result aligns with gold"; units/scale; missing-value handling)
Output only the grading plan text, no code."""


_CODER_PROMPT = """You are a Python grading-script engineer. Following the grading plan below, write a grading function.

Grading plan:
{plan}

Question: {prompt}
Target quantity key: "{target_key}" ({answer_type})
Gold answer: {gold}

Conventions (must be followed):
1. Top-level function signature: `def evaluate(pred, gold):`, where pred=model answer, gold=authoritative answer.
   Return (result: bool, message: str).
2. **pred and gold are already normalized structured objects** (dict / list / number / str) when your
   function is called — the runtime deserializes them before calling you. So access them directly as
   objects (e.g. `pred["key"]`, `gold["key"]`); **do NOT call json.loads / ast.literal_eval on them yourself.**
   For a composite (multi-part) answer, both pred and gold are dicts keyed by each sub-quantity.
3. Numeric/count: compare with the tolerance specified in the grading plan; set/list: use an appropriate metric (intersection/recall) rather than byte-for-byte equality;
   text/complex structures that cannot be compared exactly: **use llm_judge(pred, gold, question) as an aid** (this function is already provided by the runtime environment,
   returns bool; just call it directly, no need to implement it yourself).
4. Wrap the body in try/except; on exception return (False, f"Error: {{e}}").
5. Do not import network/heavyweight libraries; use only the standard library + the injected llm_judge.

Tolerance and precision rules:
- **Follow the grading plan's specified tolerance exactly.** Do NOT apply a fixed default tolerance.
- If the Question specifies an exact precision (e.g. "rounded to 4 decimal places", "exact integer count"), use that precision — do NOT use a loose tolerance that would accept incorrect answers.
- If the Question does NOT specify precision, use the tolerance from the grading plan.
- For FLOAT comparisons: use `abs(pred - gold) / max(abs(gold), 1e-10) < rtol` where rtol comes from the grading plan.
- For INTEGER counts: do NOT use `int(pred) == gold` (truncation loophole that lets 42.9 pass for gold=42). Instead: try converting pred to a number first (it may arrive as int, float, or str), then check it equals the integer gold exactly with no fractional part: `float(pred).is_integer() and int(float(pred)) == gold`. If conversion fails, return False.
- For SET/LIST answers: use order-independent comparison (set intersection / recall), not list equality.
- For STRING answers: normalize whitespace and compare case-insensitively unless the Question explicitly requires case sensitivity.

NaN / None / invalid input rules (MUST implement):
- ALWAYS check for None and NaN BEFORE any comparison. If pred (or any required field of pred) is None, NaN, or inf, return (False, "invalid: None/NaN/inf in submission") immediately.
- Use `import math; math.isnan(float(x))` to detect NaN after converting to float. Never let NaN silently propagate into a comparison.
- For composite answers (dict): validate ALL required keys are present and none are NaN/None before scoring any sub-field. Missing or NaN fields count as incorrect.

Composite answer rules:
- For composite (multi-part) answers, ALL sub-fields must be correct for the overall result to be True. Never return True if any required sub-field is missing, wrong, or NaN.
- Do not award partial credit. If field (a) is correct but field (b) is wrong, return (False, "field b incorrect: ...").
- Explicitly check that every key present in the Gold dict also exists in pred.
Output only a Python code block (wrapped in ```python```), containing only the evaluate function (and any necessary small helper functions)."""


# Wrap generated evaluators with uniform coercion: try JSON, then ast.literal_eval for Python representations, then preserve the original value. Normalize both prediction and gold before calling the renamed inner evaluator.


_COERCE_WRAPPER = '''

# === injected by eval_gen: normalize pred/gold supplied as Python dictionary strings===
import json as _json_coerce, ast as _ast_coerce, math as _math_coerce
def _coerce(x):
    if isinstance(x, str):
        s = x.strip()
        for _parse in (_json_coerce.loads, _ast_coerce.literal_eval):
            try:
                return _parse(s)
            except Exception:
                pass
    return x
def _is_invalid(x):
    """检测 NaN/None/inf，防止 F-002 类 NaN 绕过 eval。"""
    if x is None:
        return True
    if isinstance(x, float) and (_math_coerce.isnan(x) or _math_coerce.isinf(x)):
        return True
    if isinstance(x, str) and x.strip().lower() in ("nan", "inf", "-inf", "+inf", "infinity"):
        return True
    return False
def evaluate(pred, gold):
    p, g = _coerce(pred), _coerce(gold)
    if _is_invalid(p):
        return (False, "invalid submission: None/NaN/inf")
    return _evaluate_inner(p, g)
'''

# Match only a top-level evaluate definition, not nested helpers.
_TOP_EVAL_RE = re.compile(r"(?m)^def\s+evaluate\s*\(")


def _inject_coerce_wrapper(code: str) -> str:
    """Rename the top-level evaluator and append a normalization wrapper. Warn and return unchanged code if no matching definition exists. Skip previously injected code to avoid recursive double wrapping."""
    if not code:
        return code
    if "injected by eval_gen" in code or "_evaluate_inner" in code:
        return code  # Skip already wrapped code to keep injection idempotent.
    if not _TOP_EVAL_RE.search(code):
        print("  [eval_gen] 未找到顶层 def evaluate，跳过 wrapper 注入")
        return code
    renamed = _TOP_EVAL_RE.sub("def _evaluate_inner(", code, count=1)
    return renamed.rstrip() + "\n" + _COERCE_WRAPPER


def generate_eval_script(prompt: str, target_key: str, answer_type: str,
                         gold, task_type: str = "original",
                         gold_program: str = "") -> str:
    """Generate an evaluator with planner and coder requests; return an empty string on failure. Inject prediction/gold normalization and use reference code to clarify types, precision, and output structure."""
    gold_str = str(gold)   # no truncation
    gp_section = f"\n\nGold Program (the code that produces the gold answer — use to understand output precision, data types, and structure; do NOT require the agent to follow this exact implementation):\n```python\n{gold_program}\n```" if gold_program else ""
    try:
        plan = llm_gen.generate(
            _PLANNER_PROMPT.format(prompt=prompt, target_key=target_key,
                                   answer_type=answer_type, gold=gold_str, task_type=task_type) + gp_section,
            max_tokens=12000, temperature=0.2, tag="eval_planner", thinking=True,
            model=llm_gen.GW_MODEL_FLASH)
    except Exception as e:  # noqa: BLE001
        print(f"  [eval_gen] planner 失败: {e}"); return ""
    try:
        code = llm_gen.generate(
            _CODER_PROMPT.format(plan=plan, prompt=prompt,
                                 target_key=target_key, answer_type=answer_type, gold=gold_str) + gp_section,
            max_tokens=32000, temperature=0.2, tag="eval_coder", thinking=False,
            model=llm_gen.GW_MODEL_FLASH)
        return _inject_coerce_wrapper(llm_gen._extract_code(code))
    except Exception as e:  # noqa: BLE001
        print(f"  [eval_gen] coder 失败: {e}"); return ""
