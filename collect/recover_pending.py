"""Retry QC for tasks staged in packaged/_qc_pending after API failures. Passing tasks return to the main package; rejected tasks are removed; API failures remain pending. Usage: python collect/recover_pending.py [--dirs DIR ...] [--workers N] [--dry-run]."""
import os, sys, re, json, shutil, threading, argparse, glob
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stemgen.package_tasks import _qc_with_gpt

BASE = os.getenv("STEM_PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VALID_TYPES = {"original", "subworkflow", "intermediate", "completion", "repair", "missing_node_repair"}


def _parse_task(stem_dir):
    md = os.path.join(stem_dir, "task.md")
    content = open(md, encoding="utf-8", errors="ignore").read()
    pm = re.search(r'## Prompt[^\n]*\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
    prompt = pm.group(1).strip() if pm else ""
    gm = re.search(r'## Gold answer\s*\n```[^\n]*\n(.*?)```', content, re.DOTALL)
    if gm:
        raw = gm.group(1).strip()
        try: gold = json.loads(raw)
        except Exception: gold = raw
    else:
        gold = ""
    def _rd(fn):
        p = os.path.join(stem_dir, fn)
        return open(p, encoding="utf-8", errors="ignore").read() if os.path.exists(p) else ""
    return prompt, gold, _rd("gold_program.py"), _rd("eval_script.py"), _rd("dockerfile")


_index_lock = threading.Lock()


def _append_index(packaged_dir, tid, task_type, diff, score):
    """Append an index entry after recovery. Moving the directory alone undercounts recovered tasks. Only persisted ID, path, type, difficulty, and recovery status are available here; downstream consumers must not assume these rows contain full candidate metadata."""
    idx = os.path.join(packaged_dir, "tasks_with_id.jsonl")
    row = {"id": tid, "task_type": task_type, "difficulty": diff,
           "path": f"{task_type}/difficulty_{diff}/{tid}",
           "recovered": True, "qc_score": score}
    try:
        # Serialize index appends across recovery workers; write each record in one operation.
        
        with _index_lock, open(idx, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as e:  # noqa: BLE001 Index failures must not invalidate an already recovered task.
        print(f"  [warn] 索引追加失败 {tid}: {e}")


def recover_one(stem_dir, dry_run=False, min_score=4):
    prompt, gold, gp, es, df = _parse_task(stem_dir)
    result = _qc_with_gpt(prompt, gold, gp, es, df)
    if result.get("skipped"):
        return "still_error", result.get("summary", "")[:60]
    score = result.get("score", -1)
    pending_dir = os.path.dirname(stem_dir)                 # .../packaged/_qc_pending
    packaged_dir = os.path.dirname(pending_dir)             # .../packaged
    tid = os.path.basename(stem_dir)
    if score >= min_score:
        # Move the task back to its task-type and difficulty directory.
        task_type = "original"; diff = result.get("difficulty") or "unknown"
        try:
            head = open(os.path.join(stem_dir, "task.md"), encoding="utf-8", errors="ignore").read(500)
            m = re.search(r'\*\*task_type\*\*:\s*(\S+)', head)
            if m and m.group(1).strip().rstrip("*").lower() in VALID_TYPES:
                task_type = m.group(1).strip().rstrip("*").lower()
        except Exception: pass
        if not dry_run:
            dest = os.path.join(packaged_dir, task_type, f"difficulty_{diff}", tid)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            # Append the QC record.
            try:
                with open(os.path.join(stem_dir, "task.md"), "a", encoding="utf-8") as f:
                    f.write(f"\n## QC (recovered)\n- **score**: {score}\n- **difficulty**: {diff}\n- **final_score**: {score}\n")
            except Exception: pass
            shutil.move(stem_dir, dest)
            _append_index(packaged_dir, tid, task_type, diff, score)
        return "recovered", f"score={score}"
    else:
        if not dry_run:
            rej = os.path.join(packaged_dir, "_rejected", tid)
            os.makedirs(os.path.dirname(rej), exist_ok=True)
            try: shutil.move(stem_dir, rej)
            except OSError: shutil.rmtree(stem_dir, ignore_errors=True)
        return "rejected", f"score={score}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=None,
                    help="packaged 目录列表；默认本机 realdata_b8_off*/packaged")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dirs:
        pending_roots = [os.path.join(d, "_qc_pending") for d in args.dirs]
    else:
        pending_roots = glob.glob(os.path.join(BASE, "tasks_out", "realdata_b8_off*", "packaged", "_qc_pending"))

    all_stems = []
    for pr in pending_roots:
        if not os.path.isdir(pr): continue
        for tid in os.listdir(pr):
            sp = os.path.join(pr, tid)
            if os.path.isdir(sp) and os.path.exists(os.path.join(sp, "task.md")):
                all_stems.append(sp)
    print(f"待回捞: {len(all_stems)} 道（{len(pending_roots)} 个 _qc_pending 目录），workers={args.workers}, dry_run={args.dry_run}\n")

    counters = {}; lock = threading.Lock(); done = [0]
    def process(sp):
        action, reason = recover_one(sp, dry_run=args.dry_run)
        with lock:
            counters[action] = counters.get(action, 0) + 1
            done[0] += 1
            if done[0] % 50 == 0:
                print(f"  [{done[0]}/{len(all_stems)}] recovered={counters.get('recovered',0)} "
                      f"rejected={counters.get('rejected',0)} still_error={counters.get('still_error',0)}")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process, s): s for s in all_stems}
        for fut in as_completed(futs):
            try: fut.result()
            except Exception as e: print(f"  [ERR] {futs[fut]}: {e}")

    print(f"\n=== 完成 ===")
    for k, v in sorted(counters.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
