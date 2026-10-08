"""Create and summarize the independent three-reviewer audit used by POSEIDON."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

TASK_TYPES = ("original", "subworkflow", "completion", "repair")

def load_tasks(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows

def stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row.get("task_type", "unknown"), row.get("discipline", "unknown"))].append(row)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    keys = sorted(groups)
    selected = []
    while len(selected) < min(n, len(rows)):
        progressed = False
        for key in keys:
            if groups[key] and len(selected) < n:
                selected.append(groups[key].pop())
                progressed = True
        if not progressed:
            break
    return selected

def summarize(sample: list[dict], reviews: list[dict]) -> dict:
    by_task = defaultdict(list)
    for review in reviews:
        by_task[str(review.get("task_id", ""))].append(bool(review.get("approved")))
    audited = [row for row in sample if str(row.get("id", row.get("task_id", ""))) in by_task]
    accepted = [row for row in audited if len(by_task[str(row.get("id", row.get("task_id", "")))]) >= 3 and all(by_task[str(row.get("id", row.get("task_id", "")))][:3])]
    return {"sampled": len(sample), "audited": len(audited), "accepted": len(accepted),
            "acceptance_rate": len(accepted) / len(audited) if audited else None,
            "all_three_required": True}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks_jsonl", type=Path)
    ap.add_argument("output_json", type=Path)
    ap.add_argument("--reviews", type=Path)
    ap.add_argument("--sample-size", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rows = load_tasks(args.tasks_jsonl)
    sample = stratified_sample(rows, args.sample_size, args.seed)
    result = {"sample": sample, "audit": {"sample_size": len(sample), "seed": args.seed}}
    if args.reviews:
        reviews = load_tasks(args.reviews)
        result["audit"] = summarize(sample, reviews) | {"sample_size": len(sample), "seed": args.seed}
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
