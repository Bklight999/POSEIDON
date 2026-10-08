"""Data structures for workflow specifications and derived graded tasks. TaskSpec represents a seed's scientific workflow; GradedTask adds a concrete prompt, evaluator metadata, executed answer, alternatives, and quality results."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any
import json


# A decision is load-bearing when an incorrect choice materially changes the answer.
LOAD_BEARING = "load_bearing"
COSMETIC = "cosmetic"


@dataclass
class DecisionPoint:
    name: str
    kind: str               # LOAD_BEARING | COSMETIC
    correct_choices: list[str]      # Scientifically valid methods and plausible incorrect alternatives.
    wrong_choices: list[str]        
    note: str = ""
    step: int = 0           # One-based workflow step number; zero means unspecified.

    def __post_init__(self):
        assert self.kind in (LOAD_BEARING, COSMETIC), f"bad kind: {self.kind}"


@dataclass
class TaskSpec:
    """Structured workflow seed with its objective, ordered steps, and associated decisions and methods."""
    seed_id: str
    repo: str
    goal: str = ""          # One-sentence scientific research question or objective.
    # Ordered source-derived workflow steps form the specification's backbone; decision points refer to their numbers, and derivation and reference construction preserve this chain.
    
    
    workflow_steps: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    intermediate_outputs: list[str] = field(default_factory=list)
    final_outputs: list[str] = field(default_factory=list)
    decision_points: list[DecisionPoint] = field(default_factory=list)
    valid_methods: list[str] = field(default_factory=list)
    plausible_wrong_methods: list[str] = field(default_factory=list)

    def load_bearing_points(self) -> list[DecisionPoint]:
        return [d for d in self.decision_points if d.kind == LOAD_BEARING]

    def validate(self) -> list[str]:
        """Return validation issues; an empty list means the specification is valid."""
        problems = []
        if not self.inputs:
            problems.append("inputs 为空：seed 必须有输入数据")
        if not self.workflow_steps:
            problems.append("workflow_steps 为空：没有可复现的分析流程")
        if not self.load_bearing_points():
            problems.append("无承重决策点：题目只能考执行力，不合格")
        if len(self.valid_methods) < 2:
            problems.append("valid_methods < 2：无法做可识别性交叉验证(§6)")
        return problems


@dataclass
class GradedTask:
    """A concrete task derived from a workflow specification."""
    task_id: str
    seed_id: str
    task_type: str                   # original/subworkflow/completion/repair
    prompt: str                      # Scientific prompt, task input files, grading specification, and executed reference answer.
    data_files: list[str]            
    grader_spec: dict                
    gold_answer: Any                 
    target_estimand: dict = field(default_factory=dict)   # Precise estimand and analysis chain with decision annotations.
    workflow: list[str] = field(default_factory=list)     
    load_bearing_point: str = ""     # The load-bearing decision tested by this task.
    distractors: list[dict] = field(default_factory=list) # Executed alternative-method answers and mutation, seed, and reproduction provenance.
    provenance: dict = field(default_factory=dict)        
    # Quality and difficulty results.
    probe_passed: bool | None = None
    difficulty_verdict: str = ""     # good, too_easy, trash, or empty when not tested.

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def save_tasks(tasks: list[GradedTask], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")


def append_task(task, path: str):
    """Append one GradedTask or dictionary as JSONL so downstream consumers can read completed tasks before the entire run finishes."""
    row = task if isinstance(task, dict) else asdict(task)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_tasks(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
