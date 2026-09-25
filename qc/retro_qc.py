"""Apply retrospective QC to an existing dataset. Delete tasks scoring below 4 or marked UNREPAIRABLE; do not modify prompts or evaluators. Usage: python qc/retro_qc.py [--dirs DIR ...] [--workers N] [--dry-run]."""
import os, sys, json, re, shutil, threading, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Reuse the package_tasks QC functions.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stemgen.package_tasks import _qc_with_gpt, _QC_SENSITIVE_RE

VALID_TYPES = {"original", "subworkflow", "intermediate", "completion", "repair", "missing_node_repair"}

def find_stem_dirs(base_dir):
    """Enumerate task directories in either supported layout."""
    stems = []
    for root, dirs, files in os.walk(base_dir):
        for d in dirs:
            if d.startswith("stem_"):
                full = os.path.join(root, d)
                if os.path.exists(os.path.join(full, "task.md")):
                    stems.append(full)
        # Do not recurse into individual task directories.
        dirs[:] = [d for d in dirs if not d.startswith("stem_")]
    return sorted(stems)


def qc_one(stem_dir, dry_run=False, force=False):
    """Check one task, deleting it if its score is below 4; return action and reason."""
    md_path = os.path.join(stem_dir, "task.md")
    gp_path = os.path.join(stem_dir, "gold_program.py")
    es_path = os.path.join(stem_dir, "eval_script.py")

    content = open(md_path, encoding="utf-8", errors="ignore").read()

    # Skip tasks with a final passing QC score unless --force is set.
    if not force:
        m = re.search(r'\*\*final_score\*\*:\s*(\d+)', content)
        if m and int(m.group(1)) >= 4:
            return "skip_already_passed", f"final_score={m.group(1)}"

    # Extract the task prompt.
    prompt_m = re.search(r'## Prompt[^\n]*\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
    prompt_text = prompt_m.group(1).strip() if prompt_m else ""

    # Prefer the Gold answer section, falling back to the legacy gold_answer field.
    gold = ""
    gold_block_m = re.search(r'## Gold answer\s*\n```[^\n]*\n(.*?)```', content, re.DOTALL)
    if gold_block_m:
        gold_raw = gold_block_m.group(1).strip()
        try:
            gold = json.loads(gold_raw)
        except Exception:
            gold = gold_raw
    else:
        gold_field_m = re.search(r'\*\*gold_answer\*\*:\s*(.+)', content)
        if not gold_field_m:
            gold_field_m = re.search(r'"gold":\s*(\{[^}]+\}|"[^"]*"|\[[^\]]*\]|\S+)', content)
        if gold_field_m:
            gold_raw = gold_field_m.group(1).strip()
            try:
                gold = json.loads(gold_raw)
            except Exception:
                gold = gold_raw

    gold_program = open(gp_path, encoding="utf-8", errors="ignore").read() if os.path.exists(gp_path) else ""
    eval_script  = open(es_path, encoding="utf-8", errors="ignore").read() if os.path.exists(es_path) else ""

    result = _qc_with_gpt(prompt_text, gold, gold_program, eval_script)

    if result.get("skipped"):
        return "skip_qc_error", result.get("summary", "")[:80]

    score = result.get("score", -1)
    summary = result.get("summary", "")[:120]

    if score < 4:
        if not dry_run:
            shutil.rmtree(stem_dir, ignore_errors=True)
        return "deleted", f"score={score}: {summary}"
    else:
        # Write QC results to task.md.
        try:
            diff = result.get("difficulty")
            ms_risk = result.get("multiple_solution_risk", "none")
            qc_section = f"\n## QC (retro)\n- **score**: {score}\n- **summary**: {summary}\n"
            if diff: qc_section += f"- **difficulty**: {diff}\n"
            qc_section += f"- **multiple_solution_risk**: {ms_risk}\n"
            qc_section += f"- **final_score**: {score}\n"
            with open(md_path, "a", encoding="utf-8") as f:
                f.write(qc_section)
        except Exception:
            pass
        return "passed", f"score={score}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=[
        "/path/to/project/tasks_out/20260809",
        "/path/to/project/tasks_out/20260810",
        "/path/to/project/tasks_out/all_merged",
    ])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true", help="只统计，不删除")
    ap.add_argument("--force", action="store_true", help="强制重跑，不跳过已有 QC 的题")
    args = ap.parse_args()

    all_stems = []
    for d in args.dirs:
        stems = find_stem_dirs(d)
        print(f"{d}: {len(stems)} 道")
        all_stems.extend(stems)
    print(f"合计: {len(all_stems)} 道题，workers={args.workers}, dry_run={args.dry_run}\n")

    counters = {"passed": 0, "deleted": 0, "skip_already_passed": 0, "skip_qc_error": 0}
    lock = threading.Lock()
    done = [0]

    def process(stem_dir):
        action, reason = qc_one(stem_dir, dry_run=args.dry_run, force=args.force)
        with lock:
            counters[action] = counters.get(action, 0) + 1
            done[0] += 1
            if done[0] % 100 == 0:
                print(f"  [{done[0]}/{len(all_stems)}] passed={counters['passed']} "
                      f"deleted={counters['deleted']} skip={counters['skip_already_passed']} "
                      f"error={counters['skip_qc_error']}")
            if action == "deleted":
                print(f"  [DEL] {os.path.basename(stem_dir)}: {reason}")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process, s): s for s in all_stems}
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:
                print(f"  [ERR] {futs[fut]}: {e}")

    print(f"\n=== 完成 ===")
    for k, v in sorted(counters.items()):
        print(f"  {k}: {v}")
    print(f"  删除比例: {counters.get('deleted',0)/len(all_stems)*100:.1f}%")


if __name__ == "__main__":
    main()
