#!/usr/bin/env python3
"""Build a fresh production pool from merged and newly mined sources. Exclude the union of repositories with delivered tasks, repositories recorded in grouping-cache contents, and previously dispatched pools. Cache filenames are content hashes, not repository names. Aggregate source directories through symlinks because the orchestrator accepts a single repository root."""
import concurrent.futures as cf
import json
import os
import tempfile
import re
import sys

BASE = os.getenv("STEM_PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WS = os.path.join(BASE, "workspace")
TASKS_OUT = os.path.join(BASE, "tasks_out")
# Read both source pools and deduplicate repository names; do not mutate a snapshot used by active shards.

POOL_SRCS = [os.path.join(WS, "merged_scientific_python_files.jsonl"),
             os.path.join(WS, "bio7_scientific_python_files.jsonl")]
OUT_JSONL = os.path.join(WS, "b19_fresh_scientific_python_files.jsonl")
OUT_REPOS = os.path.join(WS, "download_repos_b19")
# Actual repository locations, including unmerged clone directories.
CLONE_DIRS = [os.path.join(WS, "download_repos_bio_and_medicine"),
              os.path.join(WS, "download_repos_concurrency"),
              os.path.join(WS, "download_repos_bio7")]


# Reuse the inventory cache to avoid competing shared-filesystem scans; dispatched pools separately exclude newly processed repositories.


CACHE_YIELDED = os.path.join(tempfile.gettempdir(), "yielded_repos_b19.txt")


def yielded_repos() -> set:
    """Collect repositories from delivered task metadata, reusing the inventory cache to avoid expensive shared-filesystem scans."""
    if os.path.exists(CACHE_YIELDED):
        with open(CACHE_YIELDED, encoding="utf-8") as fh:
            out = {ln.strip() for ln in fh if ln.strip()}
        if out:
            print(f"  已出题：用缓存 {CACHE_YIELDED} → {len(out)} repo", flush=True)
            return out
    out = set()
    n = 0
    for lib in sorted(os.listdir(TASKS_OUT)):
        if not lib.startswith("2026"):
            continue
        p = os.path.join(TASKS_OUT, lib)
        for tt in os.listdir(p):
            d = os.path.join(p, tt)
            if not os.path.isdir(d) or tt.startswith("_"):
                continue
            for df in os.listdir(d):
                dd = os.path.join(d, df)
                if not os.path.isdir(dd) or not df.startswith("difficulty_"):
                    continue
                for tid in os.listdir(dd):
                    f = os.path.join(dd, tid, "task.md")
                    if not os.path.exists(f):
                        continue
                    n += 1
                    try:
                        with open(f, encoding="utf-8", errors="replace") as fh:
                            for line in fh:
                                m = re.match(r"\s*-\s*\*\*source repo\*\*:\s*(.+)", line)
                                if m:
                                    out.add(m.group(1).strip())
                                    break
                    except OSError:
                        pass
    print(f"  已出题：扫 {n} 题 → {len(out)} repo", flush=True)
    try:
        with open(CACHE_YIELDED, "w", encoding="utf-8") as fh:
            fh.write("\n".join(sorted(out)))
    except OSError:
        pass
    return out


CACHE_SCANNED = os.path.join(tempfile.gettempdir(), "scanned_repos.txt")


def scanned_repos() -> set:
    """Find previously grouped or dispatched repositories. Read each cache file's repo field rather than its MD5 filename; cache the result to avoid repeated small-file scans."""
    out = set()
    cache = os.path.join(TASKS_OUT, "_seed_grouping_cache")
    if os.path.isdir(cache):
        if os.path.exists(CACHE_SCANNED) and "--refresh" not in sys.argv:
            with open(CACHE_SCANNED, encoding="utf-8") as fh:
                out = {ln.strip() for ln in fh if ln.strip()}
            print(f"  归组缓存：用缓存 {CACHE_SCANNED} → {len(out)} repo"
                  f"（--refresh 强制重读）", flush=True)
        else:
            files = os.listdir(cache)

            def _repo_of(fn):
                try:
                    with open(os.path.join(cache, fn), encoding="utf-8") as fh:
                        return json.load(fh).get("repo")
                except Exception:  # noqa: BLE001
                    return None

            with cf.ThreadPoolExecutor(32) as ex:
                for r in ex.map(_repo_of, files):
                    if r:
                        out.add(r)
            print(f"  归组缓存 {len(files)} 文件 → {len(out)} repo"
                  f"（读内容里的 repo 字段，非文件名）", flush=True)
            try:
                with open(CACHE_SCANNED, "w", encoding="utf-8") as fh:
                    fh.write("\n".join(sorted(out)))
            except OSError:
                pass
    # Exclude both previously dispatched pools, including repositories that were scanned without producing tasks.
    
    for tag in ("b17", "b18"):
        p = os.path.join(WS, f"{tag}_fresh_scientific_python_files.jsonl")
        if not os.path.exists(p):
            print(f"  ⚠ {tag} 池文件不存在，未排除：{p}", flush=True)
            continue
        before = len(out)
        with open(p, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    out.add(json.loads(line)["file_path"].split("/")[0])
                except Exception:  # noqa: BLE001
                    pass
        print(f"  {tag} 池 → 累计 {len(out)} repo（新增 {len(out)-before}）", flush=True)
    return out


def main():
    print("① 已出题 repo", flush=True)
    yielded = yielded_repos()
    print("② 已扫过 repo", flush=True)
    scanned = scanned_repos()
    excl = yielded | scanned
    print(f"  排除集合共 {len(excl)} repo", flush=True)

    print("③ 从池源筛 fresh", flush=True)
    # Require a real repository directory so the orchestrator can load source code.
    have = {}
    for cd in CLONE_DIRS:
        if os.path.isdir(cd):
            n = 0
            for d in os.listdir(cd):
                have.setdefault(d, os.path.join(cd, d))
                n += 1
            print(f"  {os.path.basename(cd)}: {n} 目录", flush=True)
    print(f"  磁盘上 repo 目录合计 {len(have)}", flush=True)

    fresh, kept_lines, seen = set(), 0, set()
    out = open(OUT_JSONL, "w", encoding="utf-8")
    for src in POOL_SRCS:
        if not os.path.exists(src):
            print(f"  ⚠ 池源不存在，跳过：{src}", flush=True)
            continue
        s_new, s_lines = 0, 0
        with open(src, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                    r = d["file_path"].split("/")[0]
                except Exception:  # noqa: BLE001
                    continue
                seen.add(r)
                if r in excl or r not in have:
                    continue
                if r not in fresh:
                    s_new += 1
                fresh.add(r)
                out.write(line if line.endswith("\n") else line + "\n")
                kept_lines += 1
                s_lines += 1
        print(f"  {os.path.basename(src)}: +{s_new} fresh repo / {s_lines} 文件",
              flush=True)
    out.close()
    print(f"  池源共 {len(seen)} repo → fresh {len(fresh)} repo / {kept_lines} 文件",
          flush=True)
    print(f"  → {OUT_JSONL}", flush=True)

    print("④ 建软链聚合目录", flush=True)
    os.makedirs(OUT_REPOS, exist_ok=True)
    made = 0
    for r in sorted(fresh):
        dst = os.path.join(OUT_REPOS, r)
        if os.path.lexists(dst):
            continue
        try:
            os.symlink(have[r], dst)
            made += 1
        except OSError as e:
            print(f"  [skip] {r}: {e}", flush=True)
    print(f"  软链 {made} 个（目录共 {len(os.listdir(OUT_REPOS))}）", flush=True)
    print(f"\nPOOL_REPOS = {len(fresh)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
