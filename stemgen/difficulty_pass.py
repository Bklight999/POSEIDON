"""Optional asynchronous difficulty pass over an append-only candidate JSONL file. Execute agent trials independently of synthesis, writing retained and rejected tasks separately. Persist the processed-line cursor for resumption. Configure candidate, good, dropped, source-data, polling, and trial-count arguments. If no source directory is supplied, resolve it from seed_id and STEM_STAGE1_REPOS."""
from __future__ import annotations

import argparse
import json
import os
import time

from . import difficulty_gate
from .agent_solver import make_hy3_agent_solver, make_gpt_agent_solver


def _read_lines(path: str) -> list[dict]:
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _resolve_src_dir(row: dict, default_src: str) -> str:
    """Resolve task data from default_src first, otherwise seed_id under STEM_STAGE1_REPOS."""
    if default_src:
        return default_src
    repos = os.getenv("STEM_STAGE1_REPOS", "")
    sid = row.get("seed_id", "")
    if repos and sid:
        cand = os.path.join(repos, sid)
        if os.path.isdir(cand):
            return cand
    return default_src or repos


def grade_one(row: dict, src_data_dir: str, n_hy3: int, n_gpt: int) -> dict:
    """Grade one candidate's difficulty and return a report including the verdict."""
    prompt = row.get("prompt", "")
    grader_spec = row.get("grader_spec") or {}
    data_files = row.get("data_files") or []
    td = {"gold_answer": row.get("gold_answer"),
          "seed_id": row.get("seed_id", ""), "operator": row.get("operator", "")}
    hy3 = make_hy3_agent_solver(src_data_dir, data_files)
    gpt = make_gpt_agent_solver(src_data_dir, data_files)  # Run only when the first model has zero correct trials.
    return difficulty_gate.grade_difficulty(prompt, td, grader_spec, hy3, gpt,
                                            n_hy3=n_hy3, n_gpt=n_gpt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="产题进程 append 的候选题 jsonl")
    ap.add_argument("--good", required=True, help="好题落库 jsonl（评测/watcher 消费）")
    ap.add_argument("--dropped", required=True, help="剔除题 jsonl（含原因）")
    ap.add_argument("--src-data-dir", default="", help="数据源目录；空=按 seed_id 到 STEM_STAGE1_REPOS 找")
    ap.add_argument("--n-hy3", type=int, default=8)
    ap.add_argument("--n-gpt", type=int, default=3)
    ap.add_argument("--poll", type=int, default=30, help="candidates 无新题时轮询间隔(秒)")
    ap.add_argument("--idle-exit", type=int, default=0, help="连续多少轮无新题后退出(0=常驻)")
    args = ap.parse_args()

    cursor_path = args.candidates + ".cursor"
    cursor = 0
    if os.path.isfile(cursor_path):
        try:
            cursor = int(open(cursor_path).read().strip() or "0")
        except ValueError:
            cursor = 0

    print(f"[difficulty_pass] 盯 {args.candidates} (游标={cursor}) "
          f"→ good={args.good} dropped={args.dropped}  n_hy3={args.n_hy3} n_gpt={args.n_gpt}")
    idle = 0
    n_good = sum(1 for _ in open(args.good)) if os.path.isfile(args.good) else 0
    n_drop = sum(1 for _ in open(args.dropped)) if os.path.isfile(args.dropped) else 0

    while True:
        rows = _read_lines(args.candidates)
        if cursor >= len(rows):
            idle += 1
            if args.idle_exit and idle >= args.idle_exit:
                print(f"[difficulty_pass] 连续 {idle} 轮无新题，退出。"
                      f"good={n_good} dropped={n_drop}")
                break
            time.sleep(args.poll)
            continue
        idle = 0
        row = rows[cursor]
        tid = row.get("task_id", f"cand_{cursor}")
        src = _resolve_src_dir(row, args.src_data_dir)
        print(f"\n[difficulty_pass] 分级 #{cursor} {tid} (src={src}) ...")
        try:
            rep = grade_one(row, src, args.n_hy3, args.n_gpt)
            verdict = rep.get("verdict", "trash")
            c = rep.get("hy3_correct"); g = rep.get("gpt_correct")
            row.setdefault("provenance", {})["difficulty_detail"] = {
                "hy3_correct": c, "gpt_correct": g, "verdict": verdict}
            row["difficulty_verdict"] = verdict
            if verdict == difficulty_gate.GOOD:
                with open(args.good, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_good += 1
                print(f"  → 好题✓ (hy3={c}/{args.n_hy3}"
                      + (f", gpt={g}/{args.n_gpt}" if g is not None else "") + f") good={n_good}")
            else:
                row["_dropped_reason"] = f"difficulty={verdict}"
                with open(args.dropped, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_drop += 1
                print(f"  → 剔除({verdict}) (hy3={c}/{args.n_hy3}"
                      + (f", gpt={g}/{args.n_gpt}" if g is not None else "") + f") dropped={n_drop}")
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            row["_dropped_reason"] = f"exception: {e!r}"
            with open(args.dropped, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_drop += 1
        cursor += 1
        with open(cursor_path, "w") as f:
            f.write(str(cursor))


if __name__ == "__main__":
    main()
