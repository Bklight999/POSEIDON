"""Batch task-synthesis entry point: prepare and adapt executable seeds, structure workflows, derive blueprints, execute references, generate evaluators, optionally grade difficulty, and package validated tasks. Persist candidates incrementally and write outbound audit records. Use --difficulty off for the standalone release because optional agent grading depends on external components."""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import threading
import time

_REPORT_LOCK = threading.Lock()   # Protect concurrent candidate writes and report updates.

# Update the in-memory candidate count under the same lock as file appends; repeated full-file recounting can stall packaging under GIL contention.


_CAND_N = [0]

def _cand_bump(k: int = 1) -> int:
    """Increment and return the candidate count; the caller must hold the report/write lock."""
    _CAND_N[0] += k
    return _CAND_N[0]

def _cand_get() -> int:
    with _REPORT_LOCK:
        return _CAND_N[0]

def _cand_set(v: int) -> None:
    with _REPORT_LOCK:
        _CAND_N[0] = v

from . import structure, derive, reference_gen, difficulty_gate, llm_gen, sandbox, eval_gen
from .taskspec import GradedTask, save_tasks, append_task

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
OUT_BASE = os.path.join(_HERE, "..", "tasks_out")


def _asdict_safe(obj):
    """Convert dataclass records to dictionaries."""
    from dataclasses import asdict, is_dataclass
    return asdict(obj) if is_dataclass(obj) else obj


def _dump_intermediate(out_path, seed_id, stage, data):
    """Persist intermediate stage artifacts beside the candidate file using stage and safe seed identifiers."""
    if not out_path:
        return
    try:
        d = os.path.dirname(out_path)
        base = os.path.basename(out_path)
        prefix = base[:-len("_candidates.jsonl")] if base.endswith("_candidates.jsonl") \
            else base.rsplit(".", 1)[0]
        # Sanitize seed IDs into safe filenames.
        safe_seed = str(seed_id).replace("/", "_").replace("__", "_")[:60]
        fp = os.path.join(d, f"{prefix}_{stage}_{safe_seed}.json")
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  [中间产物] {stage} → {fp}")
    except Exception as e:  # noqa: BLE001 Persisting diagnostics must not interrupt the main pipeline.
        print(f"  [中间产物] {stage} 落盘失败(忽略): {e}")


# --------------------------------------------------------------------------
# Seed configuration: a manual validation seed or all collected repository records.
# --------------------------------------------------------------------------
SEED_CONFIG = {
    "bulk_rnaseq": {
        "repo_dir": os.path.join(_ROOT, "bulk-rnaseq-analysis-master"),
        "code_file": "bulk-rnaseq-analysis-guide.Rmd",  # Real differential-expression workflow source for structural interpretation.
        "discipline": "bulk RNA-seq differential expression",
        "data_files": ["raw_counts.csv", "qc.csv"],
        "counts_file": "raw_counts.csv",  # Metadata for operators that inject variants.
    },
}
STAGE1_JSONL = os.getenv(
    "STEM_STAGE1_JSONL",
    os.path.join(_HERE, "..", "stage1_smoke", "scientific_python_files.jsonl"))
STAGE1_REPOS = os.getenv(
    "STEM_STAGE1_REPOS",
    os.path.join(_HERE, "..", "stage1_smoke", "download_repos"))


# --------------------------------------------------------------------------
# Difficulty-grading solver.
# --------------------------------------------------------------------------
def _make_mock_solver(rate: float, seed: int):
    rng = random.Random(seed)
    def solve(prompt, task):
        gold = task["gold_answer"]
        if rng.random() < rate:
            return gold
        if isinstance(gold, (int, float)):
            return gold * 10 + 999
        if isinstance(gold, list):
            return []
        return "WRONG"
    return solve


def _difficulty(task: GradedTask, mode: str, repo_dir: str, n_hy3: int, n_gpt: int) -> str:
    """Difficulty modes: off skips grading, mock exercises logic with simulated solvers, and agent uses external executable-agent solvers. The latter two require the optional grader integration."""
    if mode == "off":
        return "good"
    td = {"gold_answer": task.gold_answer}
    if mode == "mock":
        # Derive a stable mock success rate from task_id to exercise all grading branches.
        r = (hash(task.task_id) % 100) / 100.0
        hy3 = _make_mock_solver(r, seed=hash(task.task_id) % 1000)
        gpt = _make_mock_solver(0.0 if r < 0.05 else 0.5, seed=hash(task.task_id + "g") % 1000)
        rep = difficulty_gate.grade_difficulty(task.prompt, td, task.grader_spec, hy3, gpt,
                                               n_hy3=n_hy3, n_gpt=n_gpt)
    elif mode == "agent":
        from .agent_solver import make_hy3_agent_solver, make_gpt_agent_solver
        # The executable-agent adapter stages repository data and binds per-task input files.
        
        hy3 = make_hy3_agent_solver(repo_dir, task.data_files)
        gpt = make_gpt_agent_solver(repo_dir, task.data_files)  # Relative prompt paths avoid disclosing source filesystem locations.
        rep = difficulty_gate.grade_difficulty(task.prompt, td, task.grader_spec, hy3, gpt,
                                               n_hy3=n_hy3, n_gpt=n_gpt)
    else:
        raise ValueError(f"未知 difficulty mode: {mode}")
    task.provenance["difficulty_detail"] = {
        "hy3_correct": rep["hy3_correct"], "gpt_correct": rep["gpt_correct"]}
    return rep["verdict"]


# --------------------------------------------------------------------------
# Complete pipeline for a single seed.
# --------------------------------------------------------------------------
def process_seed(seed_id: str, cfg: dict, args, report: list, out_path: str | None = None):
    """Run the complete pipeline for a seed, appending each accepted task immediately when out_path is provided."""
    repo_dir = cfg["repo_dir"]
    print("\n" + "=" * 70 + f"\nSEED: {seed_id}  ({repo_dir})\n" + "=" * 70)
    if not os.path.isdir(repo_dir):
        report.append({"seed": seed_id, "stage": "load", "dropped": "repo 目录不存在"})
        print(f"  ✗ repo 不存在，跳过"); return []

    # When real data is required, reject repositories without qualifying inputs before adaptation or structuring; never substitute synthetic data.
    
    if getattr(args, "require_real_data", False):
        from . import env_setup
        real_data = env_setup.probe_repo_data(repo_dir)
        if not real_data:
            report.append({"seed": seed_id, "stage": "data_gate",
                           "dropped": "no_real_data(repo 无真实数据文件, --require-real-data 丢弃)"})
            print(f"  ✗ 数据可得性门：repo 无真实数据文件，丢弃"); return []
        print(f"  [①.5 data gate] ✓ repo 有 {len(real_data)} 个真实数据文件: {real_data[:5]}")
        # Use the complete discovered real-data inventory as the authoritative list. Do not truncate by alphabetical position, which can hide required inputs from downstream stages.


        cfg = {**cfg, "data_files": list(real_data)}

    # Read all files belonging to the workflow and concatenate them for structuring.
    code_files = cfg.get("code_files") or ([cfg["code_file"]] if cfg.get("code_file") else [])
    parts, missing = [], []
    for cf in code_files:
        p = os.path.join(repo_dir, cf)
        if os.path.exists(p):
            with open(p, encoding="utf-8", errors="ignore") as f:
                parts.append(f"# ===== FILE: {cf} =====\n" + f.read())
        else:
            missing.append(cf)
    if not parts:
        report.append({"seed": seed_id, "stage": "load", "dropped": f"code 文件全缺失 {code_files}"})
        print(f"  ✗ code 文件全缺失，跳过"); return []
    if missing:
        print(f"  (部分文件缺失，跳过: {missing})")
    seed_code = "\n\n".join(parts)

    # Preview real schemas to ground generation in actual columns and sample names.
    preview = sandbox.data_preview(repo_dir, cfg["data_files"])
    print(f"\n[data preview]\n{preview}")

    # Build the shared per-seed environment before adaptation. --no-seed-env uses the configured interpreter but may reject seeds with missing dependencies.
    
    
    seed_python = args.python_bin
    if getattr(args, "seed_env", True):
        print(f"\n[② env] 为 seed {seed_id} 建 conda env（装 repo 依赖）...")
        try:
            from . import env_setup
            seed_python, _envspec = env_setup.build_seed_conda_env(repo_dir)
        except Exception as e:  # noqa: BLE001 Fall back if environment preparation fails.
            print(f"  [② env] 失败({e})，回退 {args.python_bin}")
            seed_python = args.python_bin

    # Adapt and execute the seed before structuring. Reject failures early; reuse the verified base program across reference tasks.


    if getattr(args, "adapt_gate", True):
        print(f"\n[②.5 adapt] 适配源码成自包含可跑脚本并验证（seed 只跑一次）...")
        from . import adapt as _adapt
        ar = _adapt.adapt_seed(seed_code, repo_dir, cfg["data_files"],
                               python_bin=seed_python, data_preview=preview,
                               max_iters=getattr(args, "adapt_iters", 3),
                               timeout=getattr(args, "timeout", 900),
                               require_real_data=getattr(args, "require_real_data", False))
        if not ar.ok:
            report.append({"seed": seed_id, "stage": "adapt",
                           "dropped": "源代码适配后仍跑不通", "error": ar.stderr})
            print(f"  ✗ 源码跑不通（{ar.iters}轮自修复后），丢弃 seed。err={ar.stderr[:200]}")
            return []

        # Reject degraded execution caused by missing tools, accelerators, or models before producing gold answers.
        _DEGRADE_SIGNALS = [
            "ModuleNotFoundError", "ImportError", "CUDA out of memory",
            "model not found", "No module named", "cannot import name",
            "RuntimeError: CUDA", "OSError: [Errno",
        ]
        _stderr_out = (ar.stderr or "") + str(ar.outputs or "")
        _degrade_hit = next((s for s in _DEGRADE_SIGNALS if s in _stderr_out), None)
        if _degrade_hit:
            report.append({"seed": seed_id, "stage": "adapt",
                           "dropped": f"adapt_degraded: {_degrade_hit}", "error": ar.stderr})
            print(f"  ✗ adapt 输出含降级信号({_degrade_hit!r})，丢弃 seed。")
            return []
        base_program = ar.program
        print(f"  ✓ 源码跑通（{ar.iters}轮）。outputs keys={list((ar.outputs or {}).keys())[:8]}")
        _dump_intermediate(out_path, seed_id, "base_program",
                           {"program": base_program, "outputs": ar.outputs, "iters": ar.iters})
        # Persist harvested inputs under the run's seed_data directory for all tasks from the seed.
        
        seed_data_dir = None
        if ar.data_dir and os.path.isdir(ar.data_dir):
            import shutil as _sh
            try:
                base_d = os.path.dirname(out_path) if out_path else OUT_BASE
                safe = str(seed_id).replace("/", "_").replace(" ", "_")[:80]
                seed_data_dir = os.path.join(base_d, "seed_data", safe)
                os.makedirs(seed_data_dir, exist_ok=True)
                # Mirror all subdirectories, not just top-level files.
                for root, _dirs, fns in os.walk(ar.data_dir):
                    rel = os.path.relpath(root, ar.data_dir)
                    tgt = seed_data_dir if rel == "." else os.path.join(seed_data_dir, rel)
                    os.makedirs(tgt, exist_ok=True)
                    for fn in fns:
                        try:
                            _sh.copy2(os.path.join(root, fn), os.path.join(tgt, fn))
                        except OSError:
                            pass
                _sh.rmtree(ar.data_dir, ignore_errors=True)   # Remove temporary inputs after harvesting.
                _nfiles = sum(len(f) for _, _, f in os.walk(seed_data_dir))
                print(f"  [data] 收割 {_nfiles} 个数据文件 → {seed_data_dir}")
            except OSError as e:
                print(f"  [data] 收割 data/ 失败(忽略): {e}")
                seed_data_dir = None
        # Replace guessed filenames with the actual adapted-data inventory so prompts, references, and packaged inputs agree.
        
        
        if ar.data_files:
            cfg = {**cfg, "data_files": list(ar.data_files)}
            preview = sandbox.data_preview(repo_dir, cfg["data_files"])
            print(f"  [data] 权威数据文件（回填题面/gold 对齐）: {ar.data_files[:8]}")
    else:
        base_program = None  # Without the adaptation gate, references fall back to adapting raw source.
        seed_data_dir = None

    # Structure the workflow.
    print("\n[③ structure] 调 deepseek 结构化 TaskSpec ...")
    spec, problems = structure.structure_seed(seed_code, cfg["discipline"],
                                              cfg["data_files"], repo=seed_id,
                                              data_preview=preview)
    if spec is None:
        report.append({"seed": seed_id, "stage": "structure", "dropped": problems})
        print(f"  ✗ TaskSpec 校验不过，丢弃 seed: {problems}"); return []
    spec.seed_id = seed_id
    print(f"  ✓ 承重决策点: {[d.name for d in spec.load_bearing_points()]}")
    _dump_intermediate(out_path, seed_id, "taskspec", _asdict_safe(spec))   # Persist intermediate structure results.

    # Derive the four paper task types from the original workflow without fabricated variant data.
    
    print("\n[④ derive] 调 deepseek 派生任务蓝图 ...")
    blueprints, _ = derive.derive_tasks(spec, cfg["data_files"], data_preview=preview,
                                        per_type=getattr(args, "per_type", 2),
                                        workers=getattr(args, "derive_workers", 12),
                                        seed_code=seed_code)
    print(f"  ✓ 派生 {len(blueprints)} 个蓝图: {[b.task_id for b in blueprints]}")
    _dump_intermediate(out_path, seed_id, "blueprints",
                       [b.to_dict() for b in blueprints])   # Persist intermediate blueprints.

    # Construct task references concurrently by executing the verified base program in the seed environment.
    all_data = sorted({f for b in blueprints for f in b.data_files} | set(cfg["data_files"]))
    tasks = []
    ref_workers = getattr(args, "ref_workers", 4) or 4

    def _one(bp):
        rec = {"seed": seed_id, "task_id": bp.task_id, "task_type": bp.task_type}
        try:
            return _process_blueprint(spec, bp, seed_code, cfg, args, repo_dir, all_data, preview,
                                      seed_id, rec, report, out_path, seed_python, base_program,
                                      seed_data_dir)
        except Exception as e:  # noqa: BLE001 One task failure must not block other tasks.
            import traceback; traceback.print_exc()
            rec["dropped"] = f"exception: {e!r}"
            with _REPORT_LOCK:
                report.append(rec)
            print(f"  [{bp.task_id}] ✗ 异常，丢弃: {e!r}")
            return None

    if ref_workers <= 1:
        results = [_one(bp) for bp in blueprints]
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=ref_workers) as ex:
            futs = [ex.submit(_one, bp) for bp in blueprints]
            results = [f.result() for f in as_completed(futs)]
    tasks = [t for t in results if t is not None]
    return tasks


def _process_blueprint(spec, bp, seed_code, cfg, args, repo_dir, all_data, preview,
                       seed_id, rec, report, out_path, seed_python=None, base_program=None,
                       seed_data_dir=None):
    """Build executed gold from a verified base program, optionally grade difficulty, and assemble a GradedTask. Share staged inputs across the seed's tasks. Return None for rejected tasks; the caller handles exceptions."""
    print(f"\n[⑤ reference] {bp.task_id} ({bp.task_type}) 基于源代码产 gold ...")
    _py = seed_python or args.python_bin
    ref = reference_gen.build_reference_single(
        spec, bp, seed_code, repo_dir, all_data,
        python_bin=_py, data_preview=preview, base_program=base_program,
        timeout=getattr(args, "timeout", 900),
        seed_data_dir=seed_data_dir)
    if not ref.get("ok"):
        rec["dropped"] = "源代码跑不通"
        rec["error"] = (ref.get("error") or "")[:300]
        report.append(rec)
        print(f"  [{bp.task_id}] ✗ 源代码跑不通，丢弃。err={str(ref.get('error'))[:200]}")
        return None
    gold = ref.get("gold_answer")
    if gold is None:
        rec["dropped"] = "源代码未产出 gold"
        report.append(rec); print(f"  [{bp.task_id}] ✗ 无 gold，丢弃"); return None

    # Record answer type and executed gold; composite answers collect per-component outputs and grading metadata.
    
    grader_spec = {"type": bp.answer_type, "gold": gold}
    if bp.answer_type == "composite" and (bp.composite_parts or []):
        outs = ref.get("outputs") or {}
        parts, gold_map = [], {}
        for p in bp.composite_parts:
            k = p.get("key")
            # Look up outputs at the top level and one container level below before treating a component as missing.


            v = reference_gen.resolve_output_key(outs, k) if k else reference_gen._MISSING
            if v is reference_gen._MISSING:
                continue   # Skip genuinely missing components; fall back if fewer than two remain.
            gold_map[k] = v
            parts.append({"key": k, "type": p.get("type", "scalar"),
                          "gold": v, "tol": p.get("tol")})
        if len(parts) >= 2:   # Require at least two components for a composite answer.
            grader_spec = {"type": "composite", "parts": parts, "gold": gold_map}
            gold = gold_map
        else:
            # If component definitions are incomplete, infer the type from the actual gold shape rather than labeling a dictionary as scalar.


            _t = "composite" if isinstance(gold, dict) and len(gold) >= 2 else "scalar"
            grader_spec = {"type": _t, "gold": gold}
            print(f"  [{bp.task_id}] composite 子项不足2个（凑到 {len(parts)} 项），"
                  f"按 gold 实际形状标注为 {_t}")

    # Generate the evaluator through planner and coder stages; generation failure is handled downstream.
    eval_script = ""
    try:
        eval_script = eval_gen.generate_eval_script(
            # Use the effective grader_spec type after any composite fallback, not the original blueprint declaration.
            
            bp.prompt, bp.target_quantity_key, grader_spec.get("type", bp.answer_type),
            gold, bp.task_type,
            gold_program=ref.get("gold_program", ""))
    except Exception as e:  # noqa: BLE001
        print(f"  [{bp.task_id}] eval_script 生成失败(忽略): {e}")

    # Generate a Dockerfile for reproducing the task environment.
    dockerfile = ""
    try:
        from . import env_setup
        # Reuse cached resolved and frozen seed dependencies to match the environment that executed the reference.
        
        
        if getattr(args, "seed_env", True):
            try:
                _, _es = env_setup.build_seed_conda_env(repo_dir)
            except Exception:  # noqa: BLE001 Fall back to dependency probing if environment retrieval fails.
                _es = env_setup.probe_repo_env(repo_dir)
        else:
            _es = env_setup.probe_repo_env(repo_dir)
        # Include actual gold-program imports in Dockerfile dependencies; repository declarations alone may be incomplete.
        
        
        dockerfile, _src = env_setup.gen_dockerfile(
            _es, gold_program=ref.get("gold_program", "") or "")
    except Exception as e:  # noqa: BLE001
        print(f"  [{bp.task_id}] Dockerfile 生成失败(忽略): {e}")

    # Input filenames harvested for this seed.
    _data_names = []
    if seed_data_dir and os.path.isdir(seed_data_dir):
        _data_names = sorted(f for f in os.listdir(seed_data_dir)
                             if os.path.isfile(os.path.join(seed_data_dir, f)))

    task = GradedTask(
        task_id=f"{seed_id}__{bp.task_id}", seed_id=seed_id, task_type=bp.task_type,
        prompt=bp.prompt, data_files=_data_names or bp.data_files, grader_spec=grader_spec,
        gold_answer=gold, target_estimand=bp.target_estimand,
        workflow=bp.workflow, load_bearing_point=bp.load_bearing_point,
        provenance={"answer_source": "github 源代码真跑 (权威唯一 gold)",
                    "given_step": bp.given_step,
                    "mutation": bp.mutation, "removed_step": bp.removed_step,
                    "gold_program": ref.get("gold_program"),
                    "gold_lineage": ref.get("lineage") or {},
                    "buggy_code": ref.get("buggy_code"),
                    "buggy_answer": ref.get("buggy_answer"),
                    "buggy_outputs": ref.get("buggy_outputs") or {},
                    "buggy_separation_verified": bool(ref.get("buggy_separation_verified")),
                    "incomplete_code": ref.get("incomplete_code"),
                    "eval_script": eval_script,
                    "dockerfile": dockerfile,
                    "seed_data_dir": seed_data_dir,   # Copy packaged task data from this directory.
                    # Audit flag: False means the target key was missing and a fallback output was selected.
                    "gold_key_resolved": bool(ref.get("gold_key_resolved", True)),
                    "reference_iters": ref.get("iters")},
    )

    # Optional difficulty grading; skip with --difficulty off.
    verdict = _difficulty(task, args.difficulty, repo_dir, args.n_hy3, args.n_gpt)
    task.difficulty_verdict = verdict
    rec.update({"gold": gold, "difficulty": verdict})
    if verdict != "good":
        rec["dropped"] = f"难度={verdict}"
        report.append(rec)
        print(f"  [{bp.task_id}] gold={gold} 难度={verdict} → 剔除")
        return None
    report.append(rec)
    if out_path:
        with _REPORT_LOCK:
            append_task(task, out_path)   # Append completed candidates under the shared lock and increment the counter in the same critical section.
            _cand_bump()                  
    print(f"  [{bp.task_id}] ✓ 好题 gold={gold}")
    return task


# --------------------------------------------------------------------------
# Load all-mode seeds from collected repository records.
# --------------------------------------------------------------------------
def load_all_seeds(max_seeds: int = 0, shuffle: bool = False,
                   shuffle_seed: int = 42, require_real_data: bool = False,
                   repo_offset: int = 0, repo_limit: int = 0,
                   grouping_cache_dir: str | None = None) -> dict:
    """Load scientific files from JSONL, group them into workflows, and create seed configurations. Apply repository offset and limit for disjoint shards, optionally shuffle reproducibly, stop at max_seeds, and reuse content-based grouping caches to preserve stable IDs during resume."""
    from . import seed_grouping
    if not os.path.exists(STAGE1_JSONL):
        return {}
    # Filter files and group scientific candidates by repository, retaining code for workflow grouping.
    repo_files: dict[str, list[dict]] = {}
    repo_discipline: dict[str, str] = {}
    with open(STAGE1_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue  # Skip malformed records.
            if str(d.get("is_scientific_coding_task", "")).strip().lower() == "no":
                continue  # Exclude non-scientific files such as prompt templates.
            fp = d.get("file_path", "")
            repo = fp.split("/")[0] if "/" in fp else fp
            if not repo:
                continue
            repo_files.setdefault(repo, []).append(d)
            repo_discipline.setdefault(repo, d.get("discipline", ""))

    # Use deterministic repository order or a shuffle with a fixed seed.
    repo_items = list(repo_files.items())
    if shuffle:
        random.Random(shuffle_seed).shuffle(repo_items)
        print(f"  [load_seeds] 已打乱 repo 顺序（shuffle_seed={shuffle_seed}，可复现），从随机 repo 抽 seed")
    # Skip repo_offset repositories as the lower shard bound; pair with a limit for disjoint production shards.
    
    
    if repo_offset > 0:
        repo_items = repo_items[repo_offset:]
        print(f"  [load_seeds] repo_offset={repo_offset}，跳过前 {repo_offset} 个 repo")
    # Set an upper shard bound with repo_limit; offsets alone produce overlapping suffixes.
    
    if repo_limit and repo_limit > 0:
        repo_items = repo_items[:repo_limit]
        print(f"  [load_seeds] repo_limit={repo_limit}，本机仅处理该切片内 repo")

    cfgs = {}
    cfgs_lock = threading.Lock()   # Protect concurrent seed-configuration writes.

    def _group_one(repo_files_item):
        """Group one repository independently and return seed configurations or an empty mapping."""
        repo, files = repo_files_item
        repo_dir = os.path.join(STAGE1_REPOS, repo)
        if not os.path.isdir(repo_dir):
            return {}
        # Check real-data availability.
        if require_real_data:
            from . import env_setup
            if not env_setup.probe_repo_data(repo_dir):
                return {}
        # Collect CSV/TSV inputs with a defensive upper bound large enough to avoid truncating ordinary repository inventories.


        data_files = []
        for root, _, fns in os.walk(repo_dir):
            for fn in fns:
                if fn.endswith((".csv", ".tsv")):
                    data_files.append(os.path.relpath(os.path.join(root, fn), repo_dir))
            if len(data_files) > 500:
                break
        # Reuse cached grouping or call the LLM.
        groups = seed_grouping.group_repo_files(repo, files, cache_dir=grouping_cache_dir)
        result = {}
        for gi, g in enumerate(groups):
            seed_key = repo if len(groups) == 1 else f"{repo}__{g.get('workflow_name', gi)}"
            result[seed_key] = {
                "repo_dir": repo_dir,
                "code_files": g["files"],
                "code_file": g["primary_file"],
                "discipline": repo_discipline.get(repo, ""),
                # Preserve the full discovered input list for downstream generation.
                
                "data_files": list(data_files),
                "counts_file": None,
            }
        return result

    # Group repositories concurrently and cancel unstarted work once max_seeds is reached.
    
    group_workers = min(128, len(repo_items))  # Cap grouping concurrency at 128.
    n_repos = 0
    from concurrent.futures import ThreadPoolExecutor, as_completed as _asc
    with ThreadPoolExecutor(max_workers=group_workers) as ex:
        futs = {ex.submit(_group_one, item): item[0] for item in repo_items}
        for fut in _asc(futs):
            repo = futs[fut]
            try:
                result = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  [load_seeds] {repo} 归组异常({e})，跳过")
                continue
            if not result:
                continue
            n_repos += 1
            with cfgs_lock:
                cfgs.update(result)
                cur = len(cfgs)
            if n_repos % 5 == 0:
                print(f"  [load_seeds] 已归组 {n_repos} repo → {cur} seed ...")
            if max_seeds and max_seeds > 0 and cur >= max_seeds:
                print(f"  [load_seeds] 已归够 {cur} seed（扫了 {n_repos} repo），早停")
                # Cancel pending futures; already running work cannot be interrupted here.
                for f in futs:
                    f.cancel()
                break
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="bulk_rnaseq",
                    help="bulk_rnaseq | all | 逗号分隔的 seed 名")
    ap.add_argument("--difficulty", default="mock", choices=["mock", "agent", "off"])
    ap.add_argument("--n-hy3", type=int, default=8)
    ap.add_argument("--n-gpt", type=int, default=3)
    ap.add_argument("--python-bin", default=None, help="reference 脚本用的 python 解释器（未建 seed env 时的回退）")
    ap.add_argument("--seed-env", dest="seed_env", action="store_true", default=True,
                    help="为每 seed 建独立 conda env 装依赖（默认开，跑源代码需要）")
    ap.add_argument("--no-seed-env", dest="seed_env", action="store_false",
                    help="不建 seed env，用 --python-bin（省时间但缺依赖的源码会被丢）")
    ap.add_argument("--adapt-gate", dest="adapt_gate", action="store_true", default=True,
                    help="structure 前先把源码适配成可跑脚本并验证，跑不通的 seed 直接丢弃（默认开）")
    ap.add_argument("--no-adapt-gate", dest="adapt_gate", action="store_false",
                    help="跳过 adapt gate（reference 退回从 raw 源码适配，慢且丢弃率高）")
    ap.add_argument("--adapt-iters", type=int, default=3,
                    help="Maximum adaptation and debugging attempts per seed (default: 3).")
    ap.add_argument("--timeout", type=int, default=900,
                    help="adapt/reference 单次执行超时秒数（默认900；seed 独立槽不阻塞他人，可放宽到 1000+）")
    ap.add_argument("--require-real-data", dest="require_real_data", action="store_true", default=False,
                    help="数据可得性门：repo 无真实数据文件的 seed 直接丢弃，且 adapt 硬禁造合成数据"
                         "（默认关，保持兼容；开启后只出'真实数据'的题）")
    ap.add_argument("--run-name", default="run")
    ap.add_argument("--package-every", type=int, default=50,
                    help="每累计 N 道好题就增量打包一次到 <out_dir>/packaged/（默认50；0=只在最后打一次）")
    ap.add_argument("--per-type", type=int, default=2, help="每种题型产几道（默认2）")
    ap.add_argument("--derive-workers", type=int, default=12, help="derive 并发数（纯LLM，可高）")
    ap.add_argument("--ref-workers", type=int, default=4, help="reference 并发数（真跑吃资源，适中）")
    ap.add_argument("--auto-recover", dest="auto_recover", action="store_true", default=True,
                    help="打包后自动对 _qc_pending（QC API失败）重跑 QC 回捞（默认开）")
    ap.add_argument("--no-auto-recover", dest="auto_recover", action="store_false",
                    help="关闭自动回捞")
    ap.add_argument("--max-seeds", type=int, default=0, help="最多处理几个 seed（0=全部；用于小试）")
    ap.add_argument("--shuffle-seeds", dest="shuffle_seeds", action="store_true", default=False,
                    help="打乱 repo 遍历顺序（用 --shuffle-seed 固定，可复现）。"
                         "默认顺序遍历总从 jsonl 前排 repo 出题、多样性差；开此项让每批从不同 repo 抽")
    ap.add_argument("--shuffle-seed", type=int, default=42,
                    help="shuffle 的随机种子（默认42；换种子=换一批不同 repo，同种子结果可复现）")
    ap.add_argument("--repo-offset", type=int, default=0, dest="repo_offset",
                    help="跳过前 N 个 repo（多机分片下界）")
    ap.add_argument("--repo-limit", type=int, default=0, dest="repo_limit",
                    help="本机最多处理 M 个 repo，0=不限（多机分片上界）")
    ap.add_argument("--seed-workers", type=int, default=1, help="seed 层并发数（seed 间独立；放量时调大）")
    ap.add_argument("--resume", dest="resume", action="store_true", default=True,
                    help="断点续跑：跳过 candidates/report 里已处理过的 seed（默认开）")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="不续跑，从头处理所有 seed")
    ap.add_argument("--out", default=None,
                    help="candidates jsonl 路径：非空时每确认一道好题立即 append（边产边落盘）。"
                         "默认 <out_dir>/candidates.jsonl")
    ap.add_argument("--no-grouping-cache", action="store_true",
                    help="关闭 seed 归组磁盘缓存（默认开启，缓存在 tasks_out/_seed_grouping_cache/，"
                         "跨批次共享；开启后重启无需重烧 1.5-2h 归组，且 seed_key 稳定使 resume 生效）")
    args = ap.parse_args()

    # Share grouping caches across runs and machines because they depend on repository content, not the batch. --no-grouping-cache disables reuse.
    
    
    _gcache = None if getattr(args, "no_grouping_cache", False) else \
        os.path.abspath(os.path.join(OUT_BASE, "_seed_grouping_cache"))
    if _gcache:
        os.makedirs(_gcache, exist_ok=True)
        try:
            _n_cached = sum(1 for _f in os.listdir(_gcache) if _f.endswith(".json"))
        except OSError:
            _n_cached = 0
        print(f"  [seed_grouping] 归组缓存: {_gcache}（已有 {_n_cached} 个 repo 的结果）")

    # Assemble seed configurations.
    if args.seeds == "all":
        seeds = load_all_seeds(max_seeds=args.max_seeds,
                               shuffle=args.shuffle_seeds, shuffle_seed=args.shuffle_seed,
                               require_real_data=args.require_real_data,
                               repo_offset=getattr(args, "repo_offset", 0),
                               repo_limit=getattr(args, "repo_limit", 0),
                               grouping_cache_dir=_gcache)
    elif args.seeds in ("bulk_rnaseq",) or args.seeds in SEED_CONFIG:
        seeds = {args.seeds: SEED_CONFIG[args.seeds]}
    else:
        names = [s.strip() for s in args.seeds.split(",")]
        allcfg = {**SEED_CONFIG, **load_all_seeds(shuffle=args.shuffle_seeds,
                                                   shuffle_seed=args.shuffle_seed,
                                                   grouping_cache_dir=_gcache)}
        seeds = {n: allcfg[n] for n in names if n in allcfg}
    if not seeds:
        print("没有可用 seed，退出"); return

    # Limit seed count for a small trial.
    if args.max_seeds and args.max_seeds > 0:
        seeds = dict(list(seeds.items())[:args.max_seeds])

    out_dir = os.path.abspath(os.path.join(OUT_BASE, args.run_name))
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.out or os.path.join(out_dir, "candidates.jsonl")

    # Resume from candidates and reports, skipping both successful and explicitly rejected seeds. --no-resume disables this behavior.
    
    done_seeds: set = set()
    if getattr(args, "resume", True):
        for p in (out_path, os.path.join(out_dir, "report.json")):
            if not os.path.exists(p):
                continue
            try:
                if p.endswith(".jsonl"):
                    for line in open(p, encoding="utf-8"):
                        line = line.strip()
                        if line:
                            done_seeds.add(json.loads(line).get("seed_id", ""))
                else:  # Reports also include rejected seeds.
                    for r in json.load(open(p, encoding="utf-8")):
                        if r.get("seed"):
                            done_seeds.add(r["seed"])
            except Exception:  # noqa: BLE001
                pass
        done_seeds.discard("")
        if done_seeds:
            before = len(seeds)
            seeds = {k: v for k, v in seeds.items() if k not in done_seeds}
            print(f"[resume] 已处理 {len(done_seeds)} seed，跳过；剩 {len(seeds)}/{before} 待跑")

    _think = "开(推理模式,慢)" if os.getenv("STEM_GEN_THINKING", "0") == "1" else "关(快)"
    seed_workers = max(1, getattr(args, "seed_workers", 1))
    print(f"[orchestrator] thinking={_think}  seeds={len(seeds)} (并发{seed_workers})  → {out_path}")

    all_tasks, report = [], []

    # Trigger incremental packaging after each candidate batch; serialize packaging rounds and run validation gates.
    
    pkg_every = max(0, getattr(args, "package_every", 50))
    _pkg_lock = threading.Lock()
    _pkg_state = {"last": 0, "running": False, "started_at": 0,
                  "pending": False,   # Queue a pending trigger during packaging; increment the epoch after forced resets so stale rounds cannot unlock newer ones.
                  "epoch": 0}         
    # Allow long packaging rounds enough time to finish. A timeout intended for stuck work must not repeatedly launch overlapping rounds that compete for directory locks.


    _PKG_TIMEOUT = float(os.getenv("STEM_PKG_TIMEOUT", "5400"))

    # Initialize the candidate count once before workers start, using a bulk read; thereafter count appends in memory to avoid repeated line-iteration under GIL contention.


    def _cand_count_init() -> int:
        """Count existing candidate lines once at startup using a bulk byte read to avoid repeated Python iteration under concurrent GIL contention."""
        if not out_path or not os.path.exists(out_path):
            return 0
        try:
            with open(out_path, "rb") as f:
                return f.read().count(b"\n")
        except OSError:
            return 0

    _t_init = time.time()
    _cand_set(_cand_count_init())
    print(f"[候选计数] 初始化 = {_cand_get()} 道，耗时 {time.time()-_t_init:.2f}s"
          f"（此后用内存计数器，不再逐行读 {os.path.basename(out_path or '')}）", flush=True)

    def _pkg_is_stale(epoch):
        """Check whether a packaging epoch is obsolete before accepting new work, allowing invalidated rounds to wind down without competing with their replacements."""
        with _pkg_lock:
            return _pkg_state.get("epoch", 0) != epoch

    def _do_package(reason="", epoch=0, n=None):
        from . import package_tasks
        cand = out_path
        pkg_dir = os.path.join(out_dir, "packaged")
        try:
            # Pass the known count into logging rather than rereading the full candidate file.
            
            
            if n is None:
                n = _cand_get()
            print(f"\n[分批打包] {reason} 当前候选 {n} 题 → 打包到 {pkg_dir} ...", flush=True)
            package_tasks.package(cand, pkg_dir, STAGE1_REPOS, verify=True,
                                  should_abort=lambda: _pkg_is_stale(epoch))
        except Exception as e:  # noqa: BLE001 Packaging failures must not interrupt synthesis.
            print(f"[分批打包] 失败(忽略,继续出题): {e}", flush=True)

    def _maybe_package(cur_n):
        if pkg_every <= 0:
            return
        # Trigger from the synchronized in-memory persisted-candidate count, not delayed main-thread results or full-file recounts.


        n = max(cur_n, _cand_get())
        # Queue triggers while packaging runs and record the actual count without rounding jumps, so newly produced candidates are not skipped.


        while True:
            with _pkg_lock:
                if _pkg_state["running"] and _pkg_state["started_at"] > 0:
                    elapsed = time.time() - _pkg_state["started_at"]
                    if elapsed > _PKG_TIMEOUT:
                        print(f"[分批打包] 检测到打包超时({elapsed:.0f}s)，强制重置 running 锁", flush=True)
                        _pkg_state["running"] = False
                        _pkg_state["epoch"] = _pkg_state.get("epoch", 0) + 1  # Invalidate the old packaging epoch.
                if _pkg_state["running"]:
                    _pkg_state["pending"] = True   # Queue another round instead of dropping the trigger.
                    return
                if n < _pkg_state["last"] + pkg_every and not _pkg_state.get("pending"):
                    return
                _pkg_state["last"] = n            # Record the actual candidate count without rounding.
                _pkg_state["pending"] = False
                _pkg_state["running"] = True
                _pkg_state["started_at"] = time.time()
                _my_epoch = _pkg_state.get("epoch", 0)
            try:
                _do_package(reason=f"累计{n}题", epoch=_my_epoch, n=n)
            finally:
                with _pkg_lock:
                    # Reset state only if this round still owns the current epoch; stale completion must not unlock a newer round.
                    
                    if _pkg_state.get("epoch", 0) == _my_epoch:
                        _pkg_state["running"] = False
                    _again = bool(_pkg_state.get("pending")) and \
                        _pkg_state.get("epoch", 0) == _my_epoch
            if not _again:
                return
            # Immediately process pending work using the latest in-memory candidate count.
            
            n = max(n, _cand_get())

    def _run_seed(item):
        sid, cfg = item
        try:
            return process_seed(sid, cfg, args, report, out_path=out_path)
        except Exception as e:  # noqa: BLE001 One seed failure must not block the others.
            import traceback; traceback.print_exc()
            report.append({"seed": sid, "stage": "exception", "dropped": repr(e)})
            return []

    if seed_workers <= 1:
        for item in seeds.items():
            all_tasks.extend(_run_seed(item))
            _maybe_package(len(all_tasks))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # A background watcher checks packaging triggers periodically, independent of slow seed completion in the main thread.
        
        _bg_stop = threading.Event()
        def _bg_package_watcher():
            while not _bg_stop.is_set():
                _bg_stop.wait(60)
                if _bg_stop.is_set():
                    break
                try:
                    # Use the in-memory count instead of rereading the candidate file.
                    file_n = _cand_get()
                    if file_n > 0:
                        print(f"[bg_pkg_watcher] candidates={file_n}，检查打包触发...", flush=True)
                        _maybe_package(file_n)
                except Exception as e:
                    print(f"[bg_pkg_watcher] 异常: {e}", flush=True)
        _bg_thread = threading.Thread(target=_bg_package_watcher, daemon=True, name="bg_pkg_watcher")
        _bg_thread.start()
        with ThreadPoolExecutor(max_workers=seed_workers) as ex:
            futs = [ex.submit(_run_seed, it) for it in seeds.items()]
            for f in as_completed(futs):
                all_tasks.extend(f.result())
                _maybe_package(len(all_tasks))
        _bg_stop.set()  # Stop the background packaging watcher.

    # Deduplicate and split by seed to avoid cross-split overlap.
    for t in all_tasks:
        rng = random.Random(hash(t.seed_id) & 0xffffffff)
        r = rng.random()
        t.provenance["split"] = "train" if r < 0.8 else ("val" if r < 0.9 else "test")

    tasks_path = os.path.join(out_dir, "tasks.jsonl")
    save_tasks(all_tasks, tasks_path)
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "egress_audit.log"), "w", encoding="utf-8") as f:
        for a in llm_gen.audit_log():
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    # Print the summary.
    print("\n" + "=" * 70 + "\n入库汇总\n" + "=" * 70)
    print(f"好题入库: {len(all_tasks)}  → {tasks_path}")
    dropped = [r for r in report if r.get("dropped")]
    print(f"丢弃: {len(dropped)}")
    for r in dropped:
        print(f"  - {r.get('seed')}/{r.get('task_id','')}: {r.get('dropped')}")
    audits = llm_gen.audit_log()
    bad = [a for a in audits if a["cleaned_has_named_secret"] or a["cleaned_still_sensitive"]]
    print(f"\n脱敏审计: {len(audits)} 次 deepseek 调用，敏感词残留 {len(bad)} 次"
          + ("  ✓全部干净" if not bad else f"  ⚠️{bad}"))

    # Package the final partial batch; this is the only pass when periodic packaging is disabled.
    if getattr(args, "package_every", 50) >= 0 and all_tasks:
        _do_package(reason="最终")

    # Retry QC-pending tasks after packaging to recover transient API failures, moving passing tasks back into the main dataset.
    
    if getattr(args, "auto_recover", True):
        pkg_dir = os.path.join(out_dir, "packaged")
        pending_root = os.path.join(pkg_dir, "_qc_pending")
        if os.path.isdir(pending_root):
            try:
                _pending = [os.path.join(pending_root, t) for t in os.listdir(pending_root)
                            if os.path.isdir(os.path.join(pending_root, t))]
                if _pending:
                    print(f"\n[自动回捞] _qc_pending 有 {len(_pending)} 道，重跑 QC ...")
                    import sys as _sys
                    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    if _root not in _sys.path:
                        _sys.path.insert(0, _root)
                    from collect.recover_pending import recover_one
                    from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _ac
                    _cnt = {}
                    with _TPE(max_workers=16) as _ex:
                        _futs = {_ex.submit(recover_one, sp): sp for sp in _pending}
                        for _f in _ac(_futs):
                            try:
                                _act, _ = _f.result()
                                _cnt[_act] = _cnt.get(_act, 0) + 1
                            except Exception:
                                pass
                    print(f"[自动回捞] 完成: {_cnt}")
            except Exception as _e:  # noqa: BLE001 Recovery failures must not interrupt completion.
                print(f"[自动回捞] 失败(忽略): {_e}")


if __name__ == "__main__":
    main()
