#!/usr/bin/env python3
"""Collect undelivered tasks into a dated output directory. Compare all packaged content-hash IDs against every delivered dataset, including subsets and the destination itself. Skip legacy sequential IDs, private staging directories, nested task directories, and tasks without task.md. Preserve IDs, deduplicate across machines, and copy incrementally with 32 workers. This avoids missed shards and makes repeated runs idempotent. Usage: python collect/collect_daily.py [--dry-run] [--dest YYYYMMDD]."""
import concurrent.futures as cf
import json
import os
import re
import shutil
import sys
import threading
import time
from collections import Counter

BASE = os.getenv("STEM_PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASKS_OUT = os.path.join(BASE, "tasks_out")

DEST = os.path.join(TASKS_OUT, time.strftime("%Y%m%d"))
for _i, _a in enumerate(sys.argv):
    if _a == "--dest" and _i + 1 < len(sys.argv):
        p = sys.argv[_i + 1]
        DEST = (os.path.join(TASKS_OUT, p)
                if os.sep not in p and "/" not in p else os.path.abspath(p))

dry = "--dry-run" in sys.argv


# Legacy sequential IDs are invalid and collide across batches. Collect only content-hash IDs so deleted deliveries cannot reintroduce obsolete tasks.


_SEQ_RE = re.compile(r"^stem_\d{4,6}$")


def _scan(root, drop_seq=True):
    """Scan task-type/difficulty directories into an ID-to-metadata mapping; optionally exclude obsolete sequential IDs."""
    out = {}
    if not os.path.isdir(root):
        return out
    for tt in os.listdir(root):
        d = os.path.join(root, tt)
        if not os.path.isdir(d) or tt.startswith("_"):
            continue
        for diff in os.listdir(d):
            dd = os.path.join(d, diff)
            if not os.path.isdir(dd) or not diff.startswith("difficulty_"):
                continue
            for tid in os.listdir(dd):
                if not tid.startswith("stem_"):
                    continue
                if drop_seq and _SEQ_RE.match(tid):
                    continue
                out[tid] = (tt, diff.replace("difficulty_", ""))
    return out


def main():
    # Collect IDs from all delivered datasets.
    delivered = set()
    libs = []
    for name in sorted(os.listdir(TASKS_OUT)):
        if not name.startswith("2026"):
            continue
        got = _scan(os.path.join(TASKS_OUT, name))
        if got:
            delivered |= set(got)
            libs.append((name, len(got)))
    print(f"  已交付库 {len(libs)} 个，唯一 tid {len(delivered)}")

    # Compute the set difference.
    cand = {}
    stat = Counter()
    runs = sorted(d for d in os.listdir(TASKS_OUT) if d.startswith("realdata_"))
    for run in runs:
        pkg = os.path.join(TASKS_OUT, run, "packaged")
        if not os.path.isdir(pkg):
            continue
        for tid, (tt, diff) in _scan(pkg).items():
            if tid in delivered:
                stat["跳过(已交付)"] += 1
                continue
            if tid in cand:
                stat["跳过(跨机重复)"] += 1
                continue
            src = os.path.join(pkg, tt, f"difficulty_{diff}", tid)
            if not os.path.exists(os.path.join(src, "task.md")):
                stat["跳过(无task.md)"] += 1
                continue
            cand[tid] = (src, tt, diff, run)
    print(f"  扫了 {len(runs)} 个 run 目录，待收 {len(cand)} 题")

    # Copy with 32 workers to reduce shared-filesystem latency.
    rows = []
    lock = threading.Lock()
    done = [0]

    def one(item):
        tid, (src, tt, diff, run) = item
        dst = os.path.join(DEST, tt, f"difficulty_{diff}", tid)
        if not dry:
            if os.path.exists(dst):
                return ("已存在跳过", tid, tt, diff, run)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                shutil.copytree(src, dst,
                                ignore=shutil.ignore_patterns("stem_*"))
            except Exception as e:  # noqa: BLE001
                return (f"失败:{e}", tid, tt, diff, run)
        return ("新增", tid, tt, diff, run)

    items = sorted(cand.items())
    with cf.ThreadPoolExecutor(32) as ex:
        for kind, tid, tt, diff, run in ex.map(one, items):
            with lock:
                done[0] += 1
                if kind.startswith("失败"):
                    stat["拷贝失败"] += 1
                    print(f"  [skip] {tid}: {kind[3:]}", flush=True)
                    continue
                if kind == "已存在跳过":
                    stat["已存在跳过"] += 1
                    continue
                batch = run.replace("realdata_", "").split("_off")[0]
                rows.append({"id": tid, "task_type": tt, "difficulty": diff,
                             "path": f"{tt}/difficulty_{diff}/{tid}",
                             "source_machine": run.replace("realdata_", ""),
                             "batch": batch,
                             "orig": os.path.relpath(
                                 cand[tid][0], BASE)})
                stat["新增"] += 1
                stat[f"类型:{tt}"] += 1
                stat[f"难度:d{diff}"] += 1
                stat[f"批次:{batch}"] += 1
                if done[0] % 2000 == 0:
                    print(f"    {done[0]}/{len(items)}", flush=True)

    if not dry and rows:
        os.makedirs(DEST, exist_ok=True)
        with open(os.path.join(DEST, "tasks_with_id.jsonl"), "a",
                  encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'[DRY-RUN] ' if dry else ''}汇总到 {DEST}")
    for k in sorted(stat):
        print(f"  {k}: {stat[k]}")
    print(f"  ── 本次新增 {stat['新增']} 题")


if __name__ == "__main__":
    main()
