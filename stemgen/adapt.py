"""Adapt each seed into an executable standalone program before workflow structuring. Preserve scientific logic while repairing imports, I/O, and paths through a bounded execution-feedback loop. Reject failed seeds early, reuse each successful adaptation across derived tasks, and return the executed outputs and staged data in AdaptResult. Sanitize outbound code and diagnostics."""
from __future__ import annotations

import os
from dataclasses import dataclass

from . import llm_gen, sandbox
from .safe_egress import sanitize_for_external


def delivery_path(rel: str) -> str:
    """Map repository-relative input paths to the delivery data/ tree while preserving original subdirectories, including an existing repository data/ prefix. Centralize this contract so copying and prompts agree."""
    p = (rel or "").strip().lstrip("./").replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p if p.startswith("data/") else f"data/{p}"


def delivery_paths_block(rels, limit: int = 60) -> str:
    """Render the exact delivered input paths, one per line, for prompts to reuse verbatim."""
    seen, lines = set(), []
    for r in rels or []:
        dp = delivery_path(r)
        if dp not in seen:
            seen.add(dp); lines.append(f"  {dp}")
    if not lines:
        return "  (no data files located — the script must not fabricate any)"
    out = lines[:limit]
    if len(lines) > limit:
        out.append(f"  ... (+{len(lines) - limit} more, same `data/<repo-relative-path>` form)")
    return "\n".join(out)


@dataclass
class AdaptResult:
    ok: bool
    program: str = ""             # Executable adapted script, its actual outputs, and the last error for reporting.
    outputs: dict | None = None   
    stderr: str = ""              
    iters: int = 0
    data_dir: str | None = None   # Harvested input-data directory (caller cleans it up) and authoritative repository-relative filenames used consistently by prompts, references, and packaged data.
    data_files: list[str] = None  


# All input data must exist as real files under data/ and be read through relative paths; never embed replacement datasets in source code.


#
# Preserve repository-relative paths under the single delivered data/ directory. Give the model exact paths; forbid flattening, repository fallbacks, synthetic replacements, and writes to data/. The framework stages inputs and the generated program only reads them.


_ADAPT_PROMPT = """You are given a source code file from a real scientific research GitHub repo.
Your task: rewrite it into a **self-contained python script that runs locally in one shot**.
You **must preserve the original scientific logic and functionality**. Do NOT replace any
substantive step with dummy/pass/mock/empty functions.

You may only make these changes to make it run standalone:
- Fix imports (replace unresolvable relative imports with equivalent inline code, or import
  already-installed libraries as needed);
- Fix I/O: read all data with **relative paths under ./data/** (see the DATA RULE below);
  no absolute paths, no network downloads, no interactive input;
- Lift the main-analysis calls that live inside `if __name__ == "__main__"` or functions up
  to module top level, executed in order;
- Add minimal driver code (load data -> call original logic -> compute result). The
  scientific computation must stay faithful to the source.

=== DATA RULE (mandatory — read carefully, this is the #1 cause of rejected tasks) ===
The `data/` folder is ALREADY POPULATED by the framework and ships with the task. Your script
must ONLY READ from it. The input files exist at EXACTLY these paths, relative to the script's
working directory:
{data_paths_block}

Use those paths **VERBATIM**, e.g. `pd.read_csv('data/examples/counts.csv')`.

STRICTLY FORBIDDEN in the data layer (each one causes the task to be rejected):
- **Do NOT flatten the path.** If the path above is `data/tests/assets/x.npz`, you must write
  `data/tests/assets/x.npz` — NOT `data/x.npz`. The sub-directories are real and required.
- **Do NOT add `if not os.path.exists(...)` guards** around data loading.
- **Do NOT copy data from repository paths** (e.g. `shutil.copy('examples/x.csv', 'data/x.csv')`).
  The repo is NOT available at evaluation time — only `data/` is. Such a fallback works while
  generating the task but crashes for the end user.
- **Do NOT generate synthetic/dummy data**, and do NOT write/create anything under `data/`.
- **Never embed the dataset as an in-code variable/string/array** consumed by the analysis.
If a file you need is not in the list above, do NOT invent a path or fabricate the data —
let the script fail loudly instead (the task will be correctly discarded).

Data files currently available under the working directory (relative names): {data_files}
Data schema preview (column names / shapes; **use this, do not invent**):
{data_preview}
Repository directory tree (relative):
{dir_tree}

Source code file:
```
{seed_code}
```

Hard requirements:
1. Only depend on libraries the source already uses + numpy/pandas/scipy; no network, no
   downloads, no interaction, no GUI plotting.
2. Stay faithful to the scientific logic; only change I/O / imports / driver. No dummy/mock/
   TODO/pass placeholders.
3. All input data lives under ./data/ as real files (see DATA RULE); the analysis reads from
   there by relative path.
4. At the very end, define a top-level dict `outputs` holding the **key final results** of this
   analysis (meaningful snake_case keys, e.g. outputs["n_up_genes"]=..., outputs["r2"]=...).
   Do NOT print it yourself.
5. Do not write files outside ./data/ (no new top-level dirs, no output files elsewhere).
Output only one python code block (wrapped in ```python```), no explanation."""


def _dir_tree(repo_dir: str, max_entries: int = 120) -> str:
    """Build a repository-relative inventory with all data files first, then a bounded general directory tree."""
    data_ext = (".csv", ".tsv", ".txt", ".h5ad", ".h5", ".loom", ".mtx", ".npy", ".npz",
                ".vcf", ".vcf.gz", ".fa", ".fasta", ".fastq", ".fq", ".bed", ".gff", ".gff3",
                ".gtf", ".bam", ".sam", ".parquet", ".pkl", ".rds", ".agp", ".mat", ".pdb", ".json")
    skip_name = {"requirements.txt", "readme.txt", "license.txt", "setup.py"}
    data_files, struct_lines, n = [], [], 0
    for root, dirs, files in os.walk(repo_dir):
        if ".git" in dirs:
            dirs.remove(".git")
        rel = os.path.relpath(root, repo_dir)
        # Collect data candidates at all depths without truncating discovery prematurely.
        for fn in files:
            low = fn.lower()
            if low.endswith(data_ext) and low not in skip_name:
                p = os.path.join(root, fn)
                try:
                    if os.path.getsize(p) > 512:   # Exclude empty and placeholder files.
                        data_files.append(os.path.relpath(p, repo_dir))
                except OSError:
                    pass
        # Limit the general directory tree to depth three and a bounded size; it only provides layout context.
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth <= 3 and n < max_entries:
            indent = "  " * depth
            struct_lines.append(f"{indent}{os.path.basename(root) if rel != '.' else '.'}/")
            for fn in sorted(files)[:15]:
                struct_lines.append(f"{indent}  {fn}"); n += 1
                if n >= max_entries:
                    struct_lines.append("  ...(more, see DATA FILES list above)"); break
    out = []
    if data_files:
        out.append("DATA FILES in repo (all, any depth — use these to substitute a missing hard-coded path):")
        out += [f"  {p}" for p in sorted(data_files)[:200]]
        out.append("")
    out.append("Directory structure (truncated):")
    out += struct_lines
    return "\n".join(out)


_LOCATE_PROMPT = """You are an expert at analyzing Python analysis code and its repository to find which
REAL data files (already in the repo) the code needs as input, so a framework can copy them into a
standalone task folder.

Given the code and the repository file tree (relative paths), output which in-repo data files this code
reads as input. Rules:
- A data file is something the analysis READS as input (csv/tsv/txt/json/npz/npy/h5/h5ad/loom/mtx/vcf/
  fasta/fa/fastq/bed/gff/gtf/bam/parquet/pkl/rds/agp/... or example/test data shipped in the repo).
- Give paths RELATIVE TO THE REPO ROOT, exactly as they appear in the file tree (e.g. `examples/x.csv`).
- Only list files that plausibly exist in the tree and are used as INPUT. Do NOT invent paths.
- If the code uses only built-in/downloaded datasets or writes its own outputs (no in-repo input data),
  set dataset_label="No" and dataset_paths=[].

Repository file tree (relative):
{dir_tree}

Code:
```
{code}
```

Output ONLY this JSON (wrapped in ```json```):
{{"dataset_label": "Yes|No", "dataset_paths": ["relative/path/to/data", ...]}}"""


def locate_data_files(code: str, repo_dir: str, dir_tree: str) -> list[str]:
    """Ask the LLM to locate existing repository-relative input files without modifying their contents; return a possibly empty list through the guarded client."""
    prompt = _LOCATE_PROMPT.format(dir_tree=dir_tree, code=code)   # no truncation
    try:
        obj = llm_gen.generate_json(prompt, max_tokens=8000, tag="locate_data", thinking=False)
    except Exception as e:  # noqa: BLE001 Return no located data on failure, including blocked outbound requests.
        print(f"  [locate] 定位数据失败({e})，视为无可定位数据")
        return []
    if str(obj.get("dataset_label", "")).strip().lower() != "yes":
        return []
    paths, seen = [], set()
    for p in obj.get("dataset_paths", []) or []:
        p = str(p).strip().lstrip("/")
        # Keep only existing files inside the repository to prevent hallucinated paths and traversal.
        if p and p not in seen and os.path.isfile(os.path.join(repo_dir, p)):
            seen.add(p); paths.append(p)
    return paths


def copy_located_data(repo_dir: str, dataset_paths: list[str]) -> str | None:
    """Copy located inputs into an independent temporary directory, preserving relative paths. Return that directory if at least one file was copied, otherwise None."""
    import shutil
    import tempfile
    if not dataset_paths:
        return None
    data_dir = tempfile.mkdtemp(prefix="stem_data_")
    n = 0
    for rel in dataset_paths:
        src = os.path.join(repo_dir, rel)
        if not os.path.isfile(src):
            continue
        # Preserve repository-relative subdirectories when copying inputs.
        dst = os.path.join(data_dir, rel)
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            n += 1
        except OSError:
            pass
    if n == 0:
        shutil.rmtree(data_dir, ignore_errors=True)
        return None
    return data_dir


_ZERO_MIN_NUMERIC = int(os.getenv("STEM_DEGEN_ZERO_MIN", "3"))


def _outputs_degenerate(outputs) -> tuple[bool, str]:
    """Detect degenerate executed outputs and return a flag and reason. Accept nonempty categorical, identifier, sequence, and boolean answers. Reject all-nonfinite numeric results; only reject all-zero results when the numeric count reaches the configured threshold, since isolated zeros can be valid."""
    if not outputs or not isinstance(outputs, dict):
        return True, "outputs 空/非dict"
    import math
    numeric_vals: list = []
    nonzero_seen = False
    nan_cnt = 0
    has_content = False        # Track nonempty nonnumeric content such as strings, lists, and booleans.

    def _scan(v):
        nonlocal nonzero_seen, nan_cnt, has_content
        if isinstance(v, bool):
            has_content = True          # Booleans are valid answers; exclude them from numeric all-zero detection.
            return
        if isinstance(v, (int, float)):
            numeric_vals.append(v)
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                nan_cnt += 1
            elif v != 0:
                nonzero_seen = True
            return
        if isinstance(v, str):
            if v.strip():
                has_content = True
            return
        if isinstance(v, (list, tuple, set)):
            if len(v) > 0:
                has_content = True
            for x in v:
                _scan(x)
            return
        if isinstance(v, dict):
            if len(v) > 0:
                has_content = True
            for x in v.values():
                _scan(x)
            return
        if v is not None:
            has_content = True

    for val in outputs.values():
        _scan(val)

    # Accept nonempty categorical, identifier, and sequence answers even without numeric values.
    if not numeric_vals:
        if has_content:
            return False, ""
        return True, "outputs 既无数值量也无任何非空内容"
    # Reject numeric outputs consisting entirely of NaN or infinity.
    if nan_cnt == len(numeric_vals):
        return True, f"outputs 所有数值量均为 nan/inf（{len(numeric_vals)} 个）"
    # Reject all-zero outputs only with enough numeric values; one or two zeros may be legitimate.
    if not nonzero_seen:
        if len(numeric_vals) >= _ZERO_MIN_NUMERIC and not has_content:
            return True, f"outputs 所有数值量全为 0/nan/inf（{len(numeric_vals)} 个）"
        return False, ""
    return False, ""


def adapt_seed(seed_code: str, repo_dir: str, data_files: list[str], *,
               python_bin: str | None = None, data_preview: str = "",
               max_iters: int = 3, timeout: int = 300,
               require_real_data: bool = False) -> AdaptResult:
    """Adapt and execute a seed in python_bin through at most max_iters repair attempts. Use isolated work directories and return the successful program, outputs, and harvested inputs. With require_real_data, forbid synthetic inputs and reject seeds without real repository data. The caller packages and cleans up the harvested directory."""
    import shutil
    import tempfile
    seed_code = seed_code or ""
    _sc = seed_code  # Do not truncate concatenated workflow source code, which may lose scientific logic.
    dir_tree = _dir_tree(repo_dir)

    # Prefer real repository data and allow minimal interface adaptation, but never fabricate data contents.
    
    _NO_SYNTH = (
        "\n\n=== OVERRIDE (mandatory, highest priority) — REAL DATA POLICY ===\n"
        "The analysis MUST run on the repository's REAL data. Follow this 3-tier rule strictly:\n"
        "\n"
        "TIER 1 — data fits directly: if a repo data file matches what the code expects, just read it.\n"
        "\n"
        "TIER 2 — the data file the code hard-codes is MISSING, but the repo has an equivalent real file: "
        "**you MUST actively search the repo (especially `tests/`, `test/`, `example/`, `examples/`, `data/`, "
        "`demo/` directories) for a real data file of the SAME TYPE AND PURPOSE and use it** — do NOT give up "
        "just because the exact hard-coded path is absent. This is the COMMON case: authors hard-code a local/"
        "tutorial path (e.g. `tutorial/paired.csv`) whose data was never committed, but ship a real sample under "
        "`tests/data/` (e.g. `test.csv`). Point the code at that real file. Minimal adaptation changes ONLY the "
        "I/O layer (file path / column names / read format / dtype / column selection), NEVER the data values.\n"
        "  **SUBSTITUTION SAFETY (mandatory — avoid a plausible-but-wrong gold):** only substitute a repo file when "
        "it is the SAME KIND of input the code needs — matching extension AND schema (columns/structure) AND "
        "purpose. E.g. code wants paired antibody heavy/light sequences → `tests/data/test.csv` with heavy/light "
        "columns is a valid substitute. But do NOT swap in a file of a different role (a training set where a test "
        "set is required, a metadata table where an expression matrix is required, a config where data is required) "
        "just because the extension matches — that would run but produce a scientifically wrong answer. If unsure "
        "whether the candidate file has the right schema/purpose, treat it as NOT usable.\n"
        "  After substitution, EVERY value fed into the analysis MUST come verbatim from the real repo file.\n"
        "\n"
        "TIER 3 — no usable real data at all (the hard-coded file is missing AND the repo has no same-type/"
        "same-purpose real data anywhere): only THEN make the script "
        "`raise RuntimeError('NO_REAL_DATA')` to drop this seed. Do this ONLY after you have actually looked "
        "through the repo file tree above for a valid substitute and found none.\n"
        "\n"
        "STRICTLY FORBIDDEN (this is FABRICATION, never do it): generating data content that does not come "
        "from a real file — no np.random / no hand-written arrays or literals / no `f.write('>chr1\\nACGT...')` "
        "/ no `if not exists: create synthetic ...` fallback / no toy DataFrames. Fabricated data makes the "
        "gold meaningless. A dropped seed (Tier 3) is acceptable; fabricated data is NOT.\n"
        "Rule of thumb: changing file name / column names / header / format / dtype = ALLOWED (adaptation); "
        "changing or inventing the data *values* = FORBIDDEN (fabrication).\n"
    )

    def build_prompt(last_code, last_err):
        base = _ADAPT_PROMPT.format(
            data_files=", ".join(data_files) or "(none; infer from code)",
            data_paths_block=delivery_paths_block(data_files),
            data_preview=data_preview or "(no preview)", dir_tree=dir_tree, seed_code=_sc)
        if require_real_data:
            base += _NO_SYNTH
        if last_code is not None:
            # Sanitize error feedback and prior code before repair requests: resolved symlinks can expose private source paths. The LLM client repeats the outbound check.


            _err = sanitize_for_external(last_err or "")   # no truncation
            _prev = sanitize_for_external(last_code)       # no truncation
            base += (f"\n\nThe previous version failed to execute with this error:\n```\n"
                     f"{_err}\n```\n"
                     f"Previous code:\n```python\n{_prev}\n```\n"
                     f"Fix ONLY the error (do not change the scientific logic) and rewrite the "
                     f"full runnable script.")
        return base

    # Retain the successful working directory long enough to harvest staged input files.
    last_workdir = {"path": None}

    def run_fn(code):
        full = code + "\n" + sandbox.EMIT_SNIPPET
        r = sandbox.run_isolated(full, repo_dir, timeout=timeout,
                                 python_bin=python_bin, keep_workdir=True)
        # Use a fresh working directory per attempt and clean up the previous one.
        if last_workdir["path"] and last_workdir["path"] != r.get("workdir"):
            shutil.rmtree(last_workdir["path"], ignore_errors=True)
        last_workdir["path"] = r.get("workdir")
        return r

    # Disable reasoning during adaptation to preserve the code-generation budget; execution feedback supplies the repair signal.
    
    res = llm_gen.generate_with_repair(
        build_prompt, run_fn, max_iters=max_iters, tag="adapt",
        thinking=False, max_tokens=32000)

    wd = last_workdir["path"]
    if res.get("ok"):
        # Reject degenerate outputs early when real data is required, before spending resources on downstream derivation and evaluation.
        
        
        if require_real_data:
            degen, why = _outputs_degenerate(res.get("outputs"))
            if degen:
                if wd:
                    shutil.rmtree(wd, ignore_errors=True)
                print(f"  [②.5 adapt] 跑通但 outputs 退化({why})→ 丢弃 seed（假0/空，非真值）")
                return AdaptResult(ok=False, program=res.get("code", ""),
                                   stderr=f"DEGENERATE_OUTPUTS: {why}",
                                   iters=res.get("iters", 1))
        # Prefer framework-controlled copying from the located repository file list. Fall back to harvesting staged data for compatibility.
        
        
        data_dir = None
        located = locate_data_files(res.get("code", "") or _sc, repo_dir, dir_tree)
        if located:
            data_dir = copy_located_data(repo_dir, located)
            if data_dir:
                print(f"  [②.6 data] 框架按定位清单从 repo 拷 {len(located)} 个真实数据文件")
        if data_dir is None:
            # Fallback: harvest data staged by the script in its working directory.
            src_data = os.path.join(wd, "data") if wd else None
            if src_data and os.path.isdir(src_data):
                data_dir = tempfile.mkdtemp(prefix="stem_data_")
                got = 0
                for name in os.listdir(src_data):
                    sp = os.path.join(src_data, name)
                    if os.path.isfile(sp):
                        try:
                            shutil.copy2(sp, os.path.join(data_dir, name)); got += 1
                        except OSError:
                            pass
                if got == 0:
                    shutil.rmtree(data_dir, ignore_errors=True); data_dir = None
        # When real data is required, reject seeds with no located or harvested inputs.
        if require_real_data and not data_dir:
            if wd:
                shutil.rmtree(wd, ignore_errors=True)
            return AdaptResult(ok=False, program=res.get("code", ""),
                               stderr="NO_REAL_DATA: adapt 跑通但定位/收割不到真实输入数据",
                               iters=res.get("iters", 1))
        if wd:
            shutil.rmtree(wd, ignore_errors=True)
        # Use files actually harvested into data_dir as the authoritative downstream filename list.
        
        real_data_files = []
        if data_dir and os.path.isdir(data_dir):
            for root, _dirs, fns in os.walk(data_dir):
                for fn in fns:
                    real_data_files.append(os.path.relpath(os.path.join(root, fn), data_dir))
        return AdaptResult(ok=True, program=res.get("code", ""),
                           outputs=res.get("outputs"), iters=res.get("iters", 1),
                           data_dir=data_dir, data_files=real_data_files)
    if wd:
        shutil.rmtree(wd, ignore_errors=True)
    # Keep the final exception line and stderr tail; traceback headers alone cannot diagnose dependency, data, logic, or timeout failures.


    _raw = res.get("error") or ""
    _lines = [ln for ln in _raw.splitlines() if ln.strip()]
    _exc_line = _lines[-1] if _lines else ""
    _err = (f"{_exc_line}\n--- stderr tail ---\n{_raw[-800:]}") if _exc_line else _raw[-800:]
    return AdaptResult(ok=False, program=res.get("code", ""),
                       stderr=_err, iters=res.get("iters", max_iters))
