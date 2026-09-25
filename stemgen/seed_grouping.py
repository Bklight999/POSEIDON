"""Group a repository's scientific files into workflow seeds. The caller filters nonscientific files first. Use a content-based cache to avoid repeated model calls and keep workflow names stable for resume. Fingerprint repository identity, file contents, and grouping inputs; write caches atomically for concurrent access."""
from __future__ import annotations

import hashlib
import json
import os

from . import llm_gen

_PROMPT = """Below are several files from the same GitHub repository that have been judged to be "scientific analysis code".
Determine which **complete analysis workflow** each of them belongs to:
- Different parts of the same analysis chain (e.g. a main script + the utility modules it imports, or a library's main module + a usage example) → **group into the same group**.
- Mutually independent analyses targeting different scientific questions (e.g. "differential expression analysis" vs "a standalone pathway enrichment script") → **assign to different groups**.
- The internal modules of a library usually serve the same analysis purpose → tend to group them together; but multiple **standalone example scripts** shipped with the library,
  if each demonstrates a different complete analysis, may each form their own group.

Basis for judgment: the semantics of the file path (examples/ vs library internals), what the code does, and whether they import/connect to each other.

File list:
{files}

Output only the following JSON (wrapped in ```json```):
{{
  "groups": [
    {{"workflow_name": "a short English name summarizing the analysis this group does",
      "files": ["relative path of a file belonging to this group", ...],
      "primary_file": "the file that best represents the entry point of the complete analysis for this group (e.g. the main script or the most complete example)"}}
  ]
}}
Requirements: every input file must be assigned to exactly one group; primary_file must be one of that group's files."""


def cache_key(repo: str, files: list[dict], repo_dir: str | None = None,
              max_code_chars: int | None = None, char_budget: int = 2_000_000) -> str:
    """Fingerprint every grouping input: repository identity, sorted relative file paths and code hashes, code and total budgets, and the prompt version. Repository names alone are insufficient cache keys."""
    h = hashlib.md5()
    h.update(b"v1|")
    h.update(hashlib.md5(_PROMPT.encode("utf-8", "ignore")).digest())
    h.update(os.path.abspath(repo_dir or repo).encode("utf-8", "ignore"))
    h.update(f"|{max_code_chars}|{char_budget}|".encode())
    for f in sorted(files or [], key=lambda x: x.get("file_path") or ""):
        h.update((f.get("file_path") or "").encode("utf-8", "ignore"))
        h.update(hashlib.md5((f.get("code") or "").encode("utf-8", "ignore")).digest())
    return h.hexdigest()


def load_cached(cache_dir: str | None, key: str) -> list[dict] | None:
    """Read cached grouping; treat missing or corrupt files as cache misses."""
    if not cache_dir or not key:
        return None
    try:
        with open(os.path.join(cache_dir, f"{key}.json"), encoding="utf-8") as f:
            obj = json.load(f)
        groups = obj.get("groups")
        if isinstance(groups, list) and groups and all(
                isinstance(g, dict) and g.get("files") for g in groups):
            return groups
    except (OSError, ValueError):
        pass
    return None


def save_cached(cache_dir: str | None, key: str, repo: str, groups: list[dict]) -> None:
    """Write through a temporary file and atomic replacement so concurrent readers never observe partial JSON."""
    if not cache_dir or not key or not groups:
        return
    try:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"{key}.json")
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"repo": repo, "groups": groups}, f, ensure_ascii=False)
        os.replace(tmp, path)          # Rename atomically.
    except OSError:
        pass                            # Cache-write failures must not interrupt the main pipeline.


def group_repo_files(repo: str, files: list[dict], max_code_chars: int | None = None,
                     char_budget: int = 2_000_000, cache_dir: str | None = None) -> list[dict]:
    """Group scientific file records into workflows. Bound per-file and total code context; reproducibly select files if over budget and keep excluded files separately. Reuse cached successful results. Single files need no model call; on model failure, combine only the submitted files as a temporary fallback."""
    if not files:
        return []

    # Return cached grouping without another model request.
    _ck = cache_key(repo, files, repo_dir=None, max_code_chars=max_code_chars,
                    char_budget=char_budget) if cache_dir else ""
    if cache_dir:
        _hit = load_cached(cache_dir, _ck)
        if _hit is not None:
            return _hit
    # Normalize paths relative to the repository root.
    def rel(fp: str) -> str:
        return fp[len(repo) + 1:] if fp.startswith(repo + "/") else fp

    def _len(f) -> int:
        code = f.get("code") or ""
        return len(code if max_code_chars is None else code[:max_code_chars])

    if len(files) == 1:
        r = rel(files[0]["file_path"])
        _out = [{"workflow_name": repo, "files": [r], "primary_file": r}]
        save_cached(cache_dir, _ck, repo, _out)   # Cache deterministic results that required no model call.
        return _out

    # Select files reproducibly within the total character budget. Keep excluded files as individual groups so no seed is lost.
    
    dropped_groups = []
    to_group = files
    total = sum(_len(f) for f in files)
    if total > char_budget:
        import random as _random
        # Seed sampling with MD5 rather than Python's process-randomized string hash.


        rng = _random.Random(int(hashlib.md5(repo.encode("utf-8", "ignore")).hexdigest()[:8], 16))
        order = list(files)
        rng.shuffle(order)
        picked, used = [], 0
        for f in order:
            L = _len(f)
            if picked and used + L > char_budget:       # Keep files outside the budget as individual groups.
                r = rel(f["file_path"])
                dropped_groups.append({"workflow_name": r, "files": [r], "primary_file": r})
            else:                                        # Include at least one file even if it alone exceeds the budget.
                picked.append(f); used += L
        to_group = picked
        print(f"  [seed_grouping] {repo} 科学文件拼接 {total:,} 字符超预算 {char_budget:,}，"
              f"随机保留 {len(picked)}/{len(files)} 送 LLM，其余 {len(dropped_groups)} 个各自单独成组")
        if len(to_group) == 1:                           # A single selected file needs no model grouping.
            r = rel(to_group[0]["file_path"])
            _out = [{"workflow_name": repo, "files": [r], "primary_file": r}] + dropped_groups
            save_cached(cache_dir, _ck, repo, _out)      # Cache this deterministic result.
            return _out

    listing = []
    for f in to_group:
        r = rel(f["file_path"])
        code = f.get("code") or ""
        if max_code_chars is not None:
            code = code[:max_code_chars]
        listing.append(f"### {r}\n```\n{code}\n```")
    prompt = _PROMPT.format(files="\n\n".join(listing))

    try:
        # Disable reasoning for grouping classification to avoid consuming the JSON response budget.
        obj = llm_gen.generate_json(prompt, max_tokens=8000, tag="seed_grouping", thinking=False)
        groups = obj.get("groups", [])
        # Require nonempty groups and a primary file belonging to each group.
        clean = []
        for g in groups:
            gf = [x for x in g.get("files", []) if x]
            if not gf:
                continue
            pf = g.get("primary_file") or gf[0]
            if pf not in gf:
                pf = gf[0]
            clean.append({"workflow_name": g.get("workflow_name", repo),
                          "files": gf, "primary_file": pf})
        if clean:
            _out = clean + dropped_groups
            # Cache only successful model results, never temporary fallback groupings caused by timeouts or blocked requests.
            
            
            save_cached(cache_dir, _ck, repo, _out)
            return _out
    except Exception as e:  # noqa: BLE001
        print(f"  [seed_grouping] {repo} LLM 归组失败({e})，退化为全合并一组")

    # Fallback: combine model-submitted files into one group while preserving budget-excluded files as individual groups.
    allrel = [rel(f["file_path"]) for f in to_group]
    return [{"workflow_name": repo, "files": allrel, "primary_file": allrel[0]}] + dropped_groups
