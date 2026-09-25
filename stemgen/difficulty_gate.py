"""Optional difficulty gate with injected solve(prompt, task) functions. Of eight first-model trials, five or more correct is too easy, one to four is retained, and zero triggers three stronger-model trials. Retain the latter only if at least one succeeds. Judge each trial with the task grader."""
from __future__ import annotations

from typing import Callable, Any


# Decision thresholds.
GOOD = "good"
TOO_EASY = "too_easy"
TRASH = "trash"


def classify(c_correct: int, g_correct: int | None) -> str:
    """Classify from first-model correct trials and, only when that count is zero, stronger-model correct trials."""
    if c_correct > 4:          # 5-8
        return TOO_EASY
    if c_correct >= 1:         # One to four correct trials, inclusive.
        return GOOD
    # Only run the stronger solver when the first solver has zero correct trials.
    if g_correct is None:
        raise ValueError("c=0 必须提供 gpt-5.5 结果")
    return TRASH if g_correct == 0 else GOOD


def run_trials(prompt: str, task: dict, grader_spec: dict,
               solve_fn: Callable[[str, dict], Any], n: int) -> tuple[int, list[dict]]:
    """Run n solver trials and return the correct count and details. Agent difficulty mode requires the legacy grader, which is not bundled; supply it or adapt this stage to eval_script."""
    try:
        from . import grader
    except ImportError as e:
        raise ImportError(
            "难度分级 agent 模式需要 grader 模块(已随无关代码清理移除)。"
            "如需该模式，请恢复 grader.py 或改用 eval_script 判分。") from e
    correct = 0
    trials = []
    for i in range(n):
        ans = solve_fn(prompt, task)
        ok, detail = grader.grade(ans, grader_spec)
        correct += int(ok)
        trials.append({"trial": i, "answer": ans, "passed": ok, "detail": detail})
    return correct, trials


def grade_difficulty(prompt: str, task: dict, grader_spec: dict,
                     hy3_solve: Callable, gpt_solve: Callable,
                     n_hy3: int = 8, n_gpt: int = 3) -> dict:
    """Run the complete difficulty gate for one task and return its report."""
    c, hy3_trials = run_trials(prompt, task, grader_spec, hy3_solve, n_hy3)
    report = {"hy3_correct": c, "hy3_trials": hy3_trials,
              "gpt_correct": None, "gpt_trials": None}

    if c == 0:
        g, gpt_trials = run_trials(prompt, task, grader_spec, gpt_solve, n_gpt)
        report["gpt_correct"] = g
        report["gpt_trials"] = gpt_trials
        report["verdict"] = classify(c, g)
    else:
        report["verdict"] = classify(c, None)
    return report


def format_report(task_id: str, rep: dict) -> str:
    lines = [f"=== 难度分级: {task_id} ===",
             f"hy3 做对 {rep['hy3_correct']}/8"]
    if rep["gpt_correct"] is not None:
        lines.append(f"gpt-5.5 做对 {rep['gpt_correct']}/3 (c=0 触发兜底)")
    verdict_cn = {GOOD: "好题✓ 保留", TOO_EASY: "太简单 剔除", TRASH: "废题 剔除"}
    lines.append(f"判定: {verdict_cn.get(rep['verdict'], rep['verdict'])}")
    return "\n".join(lines)
