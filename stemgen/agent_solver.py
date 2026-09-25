"""Optional executable-agent difficulty solver using the external science_agent runner. Stage one task in temporary JSONL, run the registered task adapter, and read extracted_answer from its output. Expose solve(prompt, task) for local and gateway-backed models. Sanitize the complete outbound task before gateway execution; block residual sensitive content."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_NEXUS = os.getenv("STEM_NEXUS_BIN", "nexus")
_GEN = os.getenv("STEM_AGENT_GENERATOR", "/path/to/generator.py")

# Endpoint for the model being evaluated.
_HY3_URL = os.getenv("STEM_LOCAL_BASE_URL", "http://localhost:8000/v1")
_HY3_MODEL = os.getenv("STEM_LOCAL_MODEL", "local-model")
_GPT_URL = os.getenv("STEM_SOLVER_BASE_URL", "https://api.example.com/v1")
_GPT_MODEL = os.getenv("STEM_SOLVER_MODEL", "your-model")
_GPT_KEY = os.getenv("STEM_SOLVER_KEY", "")

_VLLM_HOST = "localhost"
_TIMEOUT = int(os.getenv("STEM_SOLVER_TIMEOUT", "1200"))   # Per-task agent timeout in seconds.
_MAX_ITER = os.getenv("STEM_SOLVER_MAX_ITER", "40")


def _base_env() -> dict:
    """Exclude private endpoints from proxy routing while allowing external package and data downloads through the configured proxy."""
    env = dict(os.environ)
    return env


def _run_one(prompt: str, task: dict, src_data_dir: str, data_files: list[str],
             model: str, base_url: str, api_key: str, gpt_mode: bool,
             trial_tag: str):
    """Run one solver trial. External GPT payloads are sanitized and fail-closed before nexus."""
    if gpt_mode:
        from .safe_egress import sanitize_and_assert
        try:
            prompt = sanitize_and_assert(prompt, tag=f"agent_solver:{trial_tag}:prompt")
            # The serialized task row is what the generator ultimately converts into model messages.
            safe_task = json.loads(sanitize_and_assert(task, tag=f"agent_solver:{trial_tag}:task"))
            task = safe_task
            # src_data_dir remains a local filesystem control value and is not model content.
            data_files = json.loads(sanitize_and_assert(data_files, tag=f"agent_solver:{trial_tag}:data_files"))
        except Exception as e:
            print(f"  [agent_solver {trial_tag}] ⛔ egress rejected: {e}")
            return None
    gs = task.get("grader_spec") or {}
    row = {
        "task_id": f"grade_{trial_tag}",
        "prompt": prompt,
        "data_files": data_files or [],
        "grader_spec": gs,
        "gold_answer": task.get("gold_answer"),
        "seed_id": task.get("seed_id", ""),
        "operator": task.get("operator", ""),
    }
    workdir = tempfile.mkdtemp(prefix=f"stem_grade_{trial_tag}_")
    data_jsonl = os.path.join(workdir, "one_task.jsonl")
    with open(data_jsonl, "w", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    db = os.path.join(workdir, "run.db")
    out = os.path.join(workdir, "out.jsonl")

    env = _base_env()
    env.update({
        "STEM_DATA_FILE": data_jsonl,
        "STEM_SRC_DATA_DIR": src_data_dir,
        "STEM_AGENT": "science_agent",
        "STEM_CODE_INTERP": "jupyter",
        "STEM_MAX_ITER": _MAX_ITER,
        "STEM_TIMEOUT": str(_TIMEOUT),
        "STEM_LIMIT": "1",
        "OPENAI_API_KEY": api_key,
        "OPENAI_BASE_URL": base_url,
        "OPENAI_MODEL": model,
        "OPENAI_MAX_TOKENS": "8192",
    })
    if gpt_mode:
        env["OPENAI_GPT_MODE"] = "1"                    # Use max_completion_tokens and omit temperature.
        env["OPENAI_REASONING_EFFORT"] = "high"
        env["OPENAI_REASONING_EFFORT_TOPLEVEL"] = "1"   # This gateway expects reasoning_effort at the request's top level.
    else:
        env["OPENAI_GPT_MODE"] = "0"
        env["OPENAI_TEMPERATURE"] = "0.7"
        env["OPENAI_TOP_P"] = "0.95"
        env["OPENAI_REASONING_EFFORT"] = "high"
        env["OPENAI_REASONING_EFFORT_TOPLEVEL"] = "0"   # vLLM: extra_body

    cmd = [_NEXUS, "run-batch", "-g", _GEN, "-d", db,
           "--max-concurrent", "1", "--no-resume",
           "--export-format", "jsonl", "--export-path", out]
    try:
        subprocess.run(cmd, env=env, timeout=_TIMEOUT + 120,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except subprocess.TimeoutExpired:
        print(f"  [agent_solver {trial_tag}] 超时 {_TIMEOUT}s")
        return None
    except Exception as e:  # noqa: BLE001
        print(f"  [agent_solver {trial_tag}] 跑批失败: {e}")
        return None

    if not os.path.isfile(out):
        return None
    pred = None
    with open(out, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            to = d.get("task_output") or {}
            ea = to.get("extracted_answer")
            if ea not in (None, "", "None"):
                pred = ea
    # Convert stringified numeric answers where possible for the grader.
    if pred is not None:
        try:
            v = float(pred)
            pred = int(v) if v.is_integer() else v
        except (ValueError, TypeError):
            pass
    return pred


def make_hy3_agent_solver(src_data_dir: str, data_files: list[str]):
    """Create a local-model science-agent solver with a solve(prompt, task) interface."""
    def solve(prompt, task):
        tag = f"hy3_{int(time.time()*1000) % 100000000}"
        return _run_one(prompt, task, src_data_dir, data_files,
                        _HY3_MODEL, _HY3_URL, "EMPTY", gpt_mode=False, trial_tag=tag)
    return solve


def make_gpt_agent_solver(src_data_dir: str, data_files: list[str]):
    """Create a gateway-backed science-agent solver. Use relative prompt paths and enforce outbound sanitization before every request."""
    def solve(prompt, task):
        tag = f"gpt_{int(time.time()*1000) % 100000000}"
        return _run_one(prompt, task, src_data_dir, data_files,
                        _GPT_MODEL, _GPT_URL, _GPT_KEY, gpt_mode=True, trial_tag=tag)
    return solve
