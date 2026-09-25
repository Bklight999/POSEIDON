#!/usr/bin/env python3
"""Build a 20,000-task distillation subset. Keep all available d4 tasks and allocate the remainder 80/20 between d3 and d2. Use repository round-robin sampling with separate difficulty caps to maximize diversity. Prefer prebuilt images, capped at twice the ordinary repository limit. Deduplicate by content-hash ID, preferring image-backed copies and the hard subset. Cache the inventory and copy with 32 workers to limit shared-filesystem contention. Usage: python datasets/build_distill_set.py [--dry-run]."""
import concurrent.futures as cf
import json
import os
import tempfile
import re
import shutil
import sys
import threading
from collections import Counter, defaultdict

BASE = os.getenv("STEM_PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASKS_OUT = os.path.join(BASE, "tasks_out")
DEST = os.path.join(TASKS_OUT, "20260827_distill")
CENSUS = os.path.join(tempfile.gettempdir(), "census.jsonl")          # (tid, lib, task_type, diff, repo, has_image, path)
HARD = "20260825_hard"                # Preferred source datasets.

TOTAL = 20000
D3_RATIO = 0.8                        # Allocate the non-d4 quota as 80% d3 and 20% d2. Apply per-repository caps except for scarce d4 tasks.
CAP = {"3": 12, "2": 3}               
SEED = 20260827

_SEQ_RE = re.compile(r"^stem_\d{4,6}$")   # Exclude invalid legacy sequential IDs.
dry = "--dry-run" in sys.argv


def load_census() -> list:
    """Read and deduplicate the inventory by task ID, preferring images and then the hard subset."""
    if not os.path.exists(CENSUS):
        sys.exit(f"缺普查文件 {CENSUS}，先跑 /path/to/census.py")
    best = {}
    n = 0
    for line in open(CENSUS, encoding="utf-8"):
        tid, lib, tt, diff, repo, img, path = json.loads(line)
        n += 1
        if _SEQ_RE.match(tid):
            continue
        # Prefer existing images, then the hard subset, then other sources.
        score = (1 if img else 0, 1 if lib == HARD else 0)
        if tid not in best or score > best[tid][0]:
            best[tid] = (score, (tid, lib, tt, diff, repo, img, path))
    out = [v[1] for v in best.values()]
    print(f"  普查 {n} 条 → 唯一题 {len(out)}", flush=True)
    return out


def pick(pool: list, quota: int, cap: int | None, label: str) -> list:
    """Select up to quota tasks with a per-repository cap (None means unlimited). First select image-backed tasks under the image cap, then fill in round-robin order using deterministic within-repository sorting."""
    byrepo = defaultdict(list)
    for r in pool:
        byrepo[r[4] or "__NOREPO__"].append(r)
    # Sort each repository by image availability and task ID for reproducibility.
    for v in byrepo.values():
        v.sort(key=lambda r: (0 if r[5] else 1, r[0]))

    picked, used = [], Counter()
    # Select image-backed tasks first, subject to IMG_CAP. Allow up to twice the ordinary cap to reuse images without sacrificing repository diversity.


    img_cap = None if cap is None else cap * 2
    for repo in sorted(byrepo):
        for r in byrepo[repo]:
            if not r[5]:
                break            # The pool is sorted: stop at the first task without an image.
            if len(picked) >= quota:
                break
            if img_cap is not None and used[repo] >= img_cap:
                break
            picked.append(r)
            used[repo] += 1
    n_img = len(picked)

    # Fill remaining slots in round-robin order.
    taken = {r[0] for r in picked}
    repos = sorted(byrepo)
    rnd = 0
    while len(picked) < quota:
        progressed = False
        for repo in repos:
            if len(picked) >= quota:
                break
            # Do not deepen repositories already at or above their cap.
            if cap is not None and used[repo] >= cap:
                continue
            if rnd >= len(byrepo[repo]):
                continue
            r = byrepo[repo][rnd]
            if r[0] in taken:
                continue
            picked.append(r)
            taken.add(r[0])
            used[repo] += 1
            progressed = True
        rnd += 1
        if not progressed:
            break                # The pool or all repository caps are exhausted.

    depth = Counter(used.values())
    print(f"  {label}: 需 {quota:6} 得 {len(picked):6}"
          f"  repo {len(used):5}  带image {n_img:5}"
          f"  最深 {max(used.values()) if used else 0} 道/repo"
          f"{'  ★不足' if len(picked) < quota else ''}", flush=True)
    print(f"      每repo取用道数分布: "
          + " ".join(f"{k}道×{v}" for k, v in sorted(depth.items())[:14]), flush=True)
    return picked


def main():
    print("① 读普查", flush=True)
    uniq = load_census()

    bydiff = defaultdict(list)
    for r in uniq:
        bydiff[r[3]].append(r)
    d4 = bydiff["4"]
    rest = TOTAL - len(d4)
    q3 = int(rest * D3_RATIO)
    quota = {"4": len(d4), "3": q3, "2": rest - q3}
    print(f"  配额: d4 全收 {quota['4']}  d3 {quota['3']}  d2 {quota['2']}"
          f"  合计 {sum(quota.values())}", flush=True)
    for d in ("4", "3", "2"):
        repos = {r[4] for r in bydiff[d] if r[4]}
        img = sum(1 for r in bydiff[d] if r[5])
        print(f"  d{d}: 可用 {len(bydiff[d]):6}  唯一repo {len(repos):5}"
              f"  带image {img:5}", flush=True)

    print("② 抽样（repo 轮转 + 镜像优先）", flush=True)
    picked = []
    for d in ("4", "3", "2"):
        picked += pick(bydiff[d], quota[d], CAP.get(d), f"d{d}")
    assert len({r[0] for r in picked}) == len(picked), "抽样出现重复 tid"

    repos = {r[4] for r in picked if r[4]}
    n_img = sum(1 for r in picked if r[5])
    print(f"\n  合计 {len(picked)} 道  覆盖 **{len(repos)} 个 repo**"
          f"  带镜像 {n_img}（{100*n_img/max(len(picked),1):.0f}%）", flush=True)
    print(f"  来源库: " + "  ".join(
        f"{k}={v}" for k, v in Counter(r[1] for r in picked).most_common()), flush=True)
    print(f"  题型:   " + "  ".join(
        f"{k}={v}" for k, v in Counter(r[2] for r in picked).most_common()), flush=True)
    if len(repos) < 2000:
        print(f"  ★★ repo 数 {len(repos)} < 2000，未达要求", flush=True)

    if dry:
        print("\n[DRY-RUN] 未拷贝", flush=True)
        return 0

    print("③ 拷贝（32 路并行）", flush=True)
    stat = Counter()
    rows = []
    lock = threading.Lock()
    done = [0]

    def one(r):
        tid, lib, tt, diff, repo, img, src = r
        dst = os.path.join(DEST, tt, f"difficulty_{diff}", tid)
        if os.path.exists(dst):
            return ("已存在跳过", r)
        if not os.path.exists(os.path.join(src, "task.md")):
            return ("无task.md跳过", r)
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            # Exclude historical nested tasks and bytecode caches.
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
                "stem_*", "__pycache__"))
        except Exception as e:  # noqa: BLE001
            return (f"失败:{e}", r)
        return ("新增", r)

    os.makedirs(DEST, exist_ok=True)
    with cf.ThreadPoolExecutor(32) as ex:
        for kind, r in ex.map(one, sorted(picked, key=lambda x: x[0])):
            tid, lib, tt, diff, repo, img, src = r
            with lock:
                done[0] += 1
                if kind.startswith("失败"):
                    stat["拷贝失败"] += 1
                    print(f"  [skip] {tid}: {kind[3:]}", flush=True)
                    continue
                if kind != "新增":
                    stat[kind] += 1
                    continue
                stat["新增"] += 1
                stat[f"难度:d{diff}"] += 1
                stat[f"类型:{tt}"] += 1
                if img:
                    stat["带镜像"] += 1
                rows.append({"id": tid, "task_type": tt, "difficulty": diff,
                             "path": f"{tt}/difficulty_{diff}/{tid}",
                             "source_repo": repo, "source_lib": lib,
                             "has_image": img})
                if done[0] % 1000 == 0:
                    print(f"    {done[0]}/{len(picked)}", flush=True)

    with open(os.path.join(DEST, "tasks_with_id.jsonl"), "w", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda x: x["id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n汇总到 {DEST}", flush=True)
    for k in sorted(stat):
        print(f"  {k}: {stat[k]}", flush=True)
    print(f"  唯一 repo: {len({r['source_repo'] for r in rows if r['source_repo']})}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
