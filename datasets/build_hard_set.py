#!/usr/bin/env python3
"""Build a 5,000-task hard subset with target proportions 20% d2, 70% d3, and 10% d4. Accept only content-hash IDs to avoid collisions across legacy datasets. Stratify each difficulty by task type using largest-remainder allocation. Use a fixed random seed and sorted candidates for reproducibility. Defer task.md checks until copying and use 32 copy workers. Usage: python datasets/build_hard_set.py [--dry-run]."""
import concurrent.futures as cf
import json
import os
import random
import re
import shutil
import sys
import threading
from collections import Counter, defaultdict

BASE = os.getenv("STEM_PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASKS_OUT = os.path.join(BASE, "tasks_out")
DEST = os.path.join(TASKS_OUT, "20260825_hard")
QUOTA = {"2": 1000, "3": 3500, "4": 500}     # 20% / 70% / 10% of 5000
SEED = 20260825

# Exclude legacy IDs with four to six decimal digits; valid content hashes have twelve hexadecimal digits.

_SEQ_RE = re.compile(r"^stem_\d{4,6}$")

dry = "--dry-run" in sys.argv


def collect() -> dict:
    """Map each content-hash ID to source path, dataset, task type, and difficulty; keep the first occurrence across datasets."""
    seen = {}
    n_seq = 0
    for lib in sorted(os.listdir(TASKS_OUT)):
        if not lib.startswith("2026") or lib == os.path.basename(DEST):
            continue
        p = os.path.join(TASKS_OUT, lib)
        if not os.path.isdir(p):
            continue
        for tt in sorted(os.listdir(p)):
            d = os.path.join(p, tt)
            if not os.path.isdir(d) or tt.startswith("_"):
                continue
            for df in sorted(os.listdir(d)):
                dd = os.path.join(d, df)
                if not os.path.isdir(dd) or not df.startswith("difficulty_"):
                    continue
                diff = df.replace("difficulty_", "")
                if diff not in QUOTA:
                    continue
                for tid in os.listdir(dd):
                    if not tid.startswith("stem_") or tid in seen:
                        continue
                    if _SEQ_RE.match(tid):
                        n_seq += 1        # Skip invalid legacy sequential IDs.
                        continue
                    # Avoid a per-task stat on the shared filesystem. List directories in bulk and validate task.md during copying.


                    seen[tid] = (os.path.join(dd, tid), lib, tt, diff)
    print(f"  跳过老式顺序号（废题）{n_seq} 道", flush=True)
    return seen


def alloc(counts: dict, quota: int) -> dict:
    """Allocate quota proportionally using largest remainders without exceeding available inventory."""
    total = sum(counts.values())
    exact = {t: quota * n / total for t, n in counts.items()}
    out = {t: min(int(v), counts[t]) for t, v in exact.items()}
    # Allocate remainder slots in descending order, skipping exhausted strata.
    rest = quota - sum(out.values())
    for t, _ in sorted(exact.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True):
        while rest > 0 and out[t] < counts[t]:
            out[t] += 1
            rest -= 1
            break
    # Fill any remaining deficit from strata with spare capacity.
    while rest > 0:
        room = [t for t in out if out[t] < counts[t]]
        if not room:
            break
        for t in sorted(room, key=lambda x: counts[x] - out[x], reverse=True):
            if rest <= 0:
                break
            out[t] += 1
            rest -= 1
    return out


def main():
    print("① 汇总全库唯一题", flush=True)
    seen = collect()
    bydiff = defaultdict(lambda: defaultdict(list))
    for tid, (src, lib, tt, diff) in seen.items():
        bydiff[diff][tt].append(tid)
    for d in sorted(QUOTA):
        n = sum(len(v) for v in bydiff[d].values())
        print(f"  d{d}: 可用 {n}  需 {QUOTA[d]}"
              f"{'  ★不足' if n < QUOTA[d] else ''}", flush=True)

    print("② 分层抽样（每难度内按题型保比）", flush=True)
    rng = random.Random(SEED)
    # If d4 inventory is insufficient, fill the deficit from d3 to preserve the total size. Report the actual difficulty proportions explicitly.
    
    
    quota = dict(QUOTA)
    shortfall = 0
    for d in sorted(quota):
        avail = sum(len(v) for v in bydiff[d].values())
        if avail < quota[d]:
            shortfall += quota[d] - avail
            quota[d] = avail
    if shortfall:
        quota["3"] += shortfall
        print(f"  ⚠ d4 库存不足，缺 {shortfall} 道，从 d3 补齐（d3 → {quota['3']}）", flush=True)
    picked = []
    for d in sorted(quota):
        counts = {t: len(v) for t, v in bydiff[d].items()}
        plan = alloc(counts, quota[d])
        got = 0
        for tt in sorted(plan):
            k = plan[tt]
            if k <= 0:
                continue
            pool = sorted(bydiff[d][tt])          # Sort before sampling for reproducibility.
            picked += [(tid, d, tt) for tid in rng.sample(pool, k)]
            got += k
        print(f"  d{d}: 抽 {got}  " +
              " ".join(f"{t}={plan[t]}" for t in sorted(plan) if plan[t]), flush=True)
    print(f"  合计 {len(picked)}", flush=True)
    assert len(set(t for t, _, _ in picked)) == len(picked), "抽样出现重复 tid"

    if dry:
        print("\n[DRY-RUN] 未拷贝", flush=True)
        return 0

    print("③ 拷贝", flush=True)
    stat = Counter()
    rows = []
    lock = threading.Lock()
    done = [0]

    def one(item):
        tid, d, tt = item
        src, lib = seen[tid][0], seen[tid][1]
        dst = os.path.join(DEST, tt, f"difficulty_{d}", tid)
        if os.path.exists(dst):
            return ("已存在跳过", tid, d, tt, lib)
        if not os.path.exists(os.path.join(src, "task.md")):
            # Validate task.md here; collection deliberately avoids per-task metadata calls.
            return ("无task.md跳过", tid, d, tt, lib)
        try:
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("stem_*"))
        except Exception as e:  # noqa: BLE001
            return (f"失败:{e}", tid, d, tt, lib)
        return ("新增", tid, d, tt, lib)

    # Copy with 32 workers to hide shared-filesystem latency. Globally unique task IDs give each worker an independent destination.
    
    
    os.makedirs(DEST, exist_ok=True)
    with cf.ThreadPoolExecutor(32) as ex:
        for kind, tid, d, tt, lib in ex.map(one, sorted(picked)):
            with lock:
                done[0] += 1
                if kind.startswith("失败"):
                    stat["失败"] += 1
                    print(f"  [skip] {tid}: {kind[3:]}", flush=True)
                    continue
                stat[kind] += 1
                stat[f"难度:d{d}"] += 1
                stat[f"类型:{tt}"] += 1
                rows.append({"id": tid, "task_type": tt, "difficulty": d,
                             "path": f"{tt}/difficulty_{d}/{tid}", "source_lib": lib})
                if done[0] % 500 == 0:
                    print(f"    {done[0]}/{len(picked)}", flush=True)

    with open(os.path.join(DEST, "tasks_with_id.jsonl"), "w", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda x: x["id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n汇总到 {DEST}", flush=True)
    for k in sorted(stat):
        print(f"  {k}: {stat[k]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
