"""Controlled subprocess runner for generated reference programs. Execute with relative input paths and parse the outputs dictionary from stdout sentinels. Return failures and diagnostics for the repair loop when execution times out, raises, or omits the sentinel. Outbound callers sanitize diagnostics before model requests."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile

SENTINEL_RE = re.compile(r"###OUTPUTS###\s*(\{.*\})\s*###END###", re.DOTALL)

# Inject a directory-creation prelude so generated scripts tolerate already staged data directories instead of failing on a missing exist_ok=True argument.


_PRELUDE = (
    "import os as _os_stem\n"
    "_orig_makedirs = _os_stem.makedirs\n"
    "def _patched_makedirs(name, mode=0o777, exist_ok=True):\n"
    "    return _orig_makedirs(name, mode, exist_ok=True)\n"
    "_os_stem.makedirs = _patched_makedirs\n"
)


def _wrap(script_str: str) -> str:
    """Prepend directory-creation compatibility code to a script."""
    return _PRELUDE + script_str


# Print top-level outputs with a sentinel; give actionable feedback if the generated script never defines that variable.


EMIT_SNIPPET = (
    "import json as _json\n"
    "if 'outputs' not in dir():\n"
    "    raise RuntimeError('脚本未在顶层定义 outputs 变量：请勿把 outputs 放进 "
    "def/函数/if __name__ 块，必须是模块顶层的 dict；也不要提前 return/exit。')\n"
    "print('###OUTPUTS### ' + _json.dumps(outputs, default=float) + ' ###END###')\n"
)


# Limit each subprocess's virtual address space with configurable RLIMIT_AS to prevent runaway programs from exhausting the entire worker. Zero disables the limit; leave headroom for scientific libraries' virtual allocations.


_CHILD_MEM_GB = float(os.getenv("STEM_CHILD_MEM_GB", "24"))


def _mem_limiter():
    """Return a subprocess hook applying memory limits, or None when unavailable."""
    if _CHILD_MEM_GB <= 0:
        return None
    try:
        import resource
    except ImportError:
        return None

    def _set_limits():
        try:
            nbytes = int(_CHILD_MEM_GB * (1 << 30))
            resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))
        except Exception:  # noqa: BLE001 Continue if this platform cannot apply the memory limit.
            pass
    return _set_limits


def run(script_str: str, cwd: str, *, timeout: int = 300,
        python_bin: str | None = None) -> dict:
    """Execute a script in cwd and return status, parsed outputs, stdout, and stderr for repair feedback."""
    python_bin = python_bin or sys.executable
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=cwd, encoding="utf-8") as f:
        f.write(_wrap(script_str))
        path = f.name
    proc = None
    try:
        # Start a separate process group and kill the whole group on timeout so multiprocessing descendants do not remain as orphan workers.


        proc = subprocess.Popen(
            [python_bin, os.path.basename(path)],
            cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env={**os.environ, "MPLBACKEND": "Agg"},  # Use a noninteractive plotting backend.
            start_new_session=True, preexec_fn=_mem_limiter(),
        )
        stdout_s, stderr_s = proc.communicate(timeout=timeout)

        class _R:  # Preserve the stdout, stderr, and returncode result interface.
            pass
        _r = _R(); _r.stdout, _r.stderr, _r.returncode = stdout_s, stderr_s, proc.returncode
        proc = _r
    except subprocess.TimeoutExpired:
        try:
            import signal
            os.killpg(proc.pid, signal.SIGKILL)   # Terminate the entire process group.
            proc.communicate(timeout=10)
        except Exception:  # noqa: BLE001
            pass
        _cleanup(path)
        return {"ok": False, "outputs": None, "stdout": "",
                "stderr": f"执行超时(>{timeout}s); 已终止整个进程组"}
    except Exception as e:  # noqa
        _cleanup(path)
        return {"ok": False, "outputs": None, "stdout": "", "stderr": repr(e)}
    _cleanup(path)

    stdout, stderr = proc.stdout, proc.stderr
    if proc.returncode != 0:
        return {"ok": False, "outputs": None, "stdout": stdout,
                "stderr": stderr or f"退出码 {proc.returncode}"}
    m = SENTINEL_RE.search(stdout)
    if not m:
        return {"ok": False, "outputs": None, "stdout": stdout,
                "stderr": "脚本跑完但没打印 ###OUTPUTS### sentinel。"
                          "必须在脚本最后 print('###OUTPUTS### '+json.dumps(outputs)+' ###END###')"}
    try:
        outputs = json.loads(m.group(1))
    except json.JSONDecodeError as e:
        return {"ok": False, "outputs": None, "stdout": stdout,
                "stderr": f"sentinel 里的 JSON 解析失败: {e}"}
    return {"ok": True, "outputs": outputs, "stdout": stdout, "stderr": stderr}


def run_isolated(script_str: str, repo_dir: str, *, timeout: int = 300,
                 python_bin: str | None = None, keep_workdir: bool = False,
                 extra_data_dir: str | None = None) -> dict:
    """Run in a temporary working directory with repository links and optional copied seed inputs. Remove it afterward unless keep_workdir is requested; then return its path for caller-controlled harvesting and cleanup. This preserves the same input data across reference generation and delivery."""
    import shutil
    python_bin = python_bin or sys.executable
    workdir = tempfile.mkdtemp(prefix="stem_run_")
    removed = False
    try:
        # Link repository entries into the working directory using resolved source paths so relative inputs and local imports remain available. Relative symlink targets would resolve against the temporary directory and become dangling links.


        repo_abs = os.path.abspath(repo_dir)
        try:
            for name in os.listdir(repo_abs):
                src = os.path.join(repo_abs, name)
                try:
                    os.symlink(src, os.path.join(workdir, name))
                except OSError:
                    pass
        except OSError:
            pass
        # Prefill data/ with the harvested inputs before execution.
        if extra_data_dir and os.path.isdir(extra_data_dir):
            ddir = os.path.join(workdir, "data")
            os.makedirs(ddir, exist_ok=True)
            for fn in os.listdir(extra_data_dir):
                sp = os.path.join(extra_data_dir, fn)
                if os.path.isfile(sp):
                    try:
                        shutil.copy2(sp, os.path.join(ddir, fn))
                    except OSError:
                        pass
        script_path = os.path.join(workdir, f"_run_{os.getpid()}.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(_wrap(script_str))
        proc = None
        try:
            # Use an independent process group so timeout cleanup also reaches child workers and external commands.
            
            proc = subprocess.Popen(
                [python_bin, os.path.basename(script_path)],
                cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env={**os.environ, "MPLBACKEND": "Agg"}, start_new_session=True,
                preexec_fn=_mem_limiter(),   # Apply the subprocess memory limit to protect the main pipeline.
            )
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                import signal
                os.killpg(proc.pid, signal.SIGKILL)
                stdout, stderr = proc.communicate(timeout=10)
            except Exception:
                stdout, stderr = "", ""
            return {"ok": False, "outputs": None, "stdout": stdout or "",
                    "stderr": f"执行超时(>{timeout}s); 已终止整个进程组",
                    "workdir": workdir if keep_workdir else None}
        except Exception as e:  # noqa
            if proc is not None and proc.poll() is None:
                try:
                    import signal
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
            return {"ok": False, "outputs": None, "stdout": "", "stderr": repr(e),
                    "workdir": workdir if keep_workdir else None}
        if proc.returncode != 0:
            return {"ok": False, "outputs": None, "stdout": stdout,
                    "stderr": stderr or f"退出码 {proc.returncode}",
                    "workdir": workdir if keep_workdir else None}
        m = SENTINEL_RE.search(stdout)
        if not m:
            return {"ok": False, "outputs": None, "stdout": stdout,
                    "stderr": "脚本跑完但没打印 ###OUTPUTS### sentinel。"
                              "必须在脚本最后 print('###OUTPUTS### '+json.dumps(outputs)+' ###END###')",
                    "workdir": workdir if keep_workdir else None}
        try:
            outputs = json.loads(m.group(1))
        except json.JSONDecodeError as e:
            return {"ok": False, "outputs": None, "stdout": stdout,
                    "stderr": f"sentinel 里的 JSON 解析失败: {e}",
                    "workdir": workdir if keep_workdir else None}
        return {"ok": True, "outputs": outputs, "stdout": stdout, "stderr": stderr,
                "workdir": workdir if keep_workdir else None}
    finally:
        if not keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)
            removed = True


def _cleanup(path: str):
    try:
        os.unlink(path)
    except OSError:
        pass


def data_preview(repo_dir: str, data_files: list[str], max_files: int = 200) -> str:
    """Preview actual data schemas, shapes, and example indices using relative filenames. Bound file and column counts and estimate rows from sampled bytes to ground generation without reading whole datasets."""
    import csv
    lines = []
    for fname in data_files[:max_files]:
        path = os.path.join(repo_dir, fname)
        if not os.path.exists(path) or not fname.endswith((".csv", ".tsv")):
            continue
        delim = "\t" if fname.endswith(".tsv") else ","
        try:
            # Estimate row counts from sampled line bytes and total file size. Avoid tell() with buffered CSV readers, which reports read-ahead positions rather than consumed rows.


            _size = os.path.getsize(path)
            _sample_bytes, _sample_lines, _raw = 0, 0, []
            with open(path, "rb") as fb:
                for _ in range(20):
                    _ln = fb.readline()
                    if not _ln:
                        break
                    _raw.append(_ln)
                    _sample_bytes += len(_ln); _sample_lines += 1
            if _sample_lines and _sample_bytes:
                _avg = _sample_bytes / _sample_lines
                nrows = max(_sample_lines, round(_size / _avg)) if _size > _sample_bytes else _sample_lines
            else:
                nrows = 0
            _txt = b"".join(_raw[:2]).decode("utf-8", "ignore").splitlines()
            reader = csv.reader(_txt, delimiter=delim)
            header = next(reader, [])
            first = next(reader, [])
            cols = header[1:] if header else []
            # Bound column previews and report omitted columns; extremely wide tables can otherwise generate megabytes of low-value prompt and log text.


            _COL_CAP = 200
            if len(cols) > _COL_CAP:
                cols_txt = f"{cols[:_COL_CAP]}...(+{len(cols) - _COL_CAP} more)"
            else:
                cols_txt = f"{cols}"
            lines.append(f"- {fname}: 约 {nrows} 行 × {len(header)} 列；"
                         f"首列(index)样例={first[0] if first else '?'}；"
                         f"其余列名={cols_txt}")
        except Exception as e:  # noqa
            lines.append(f"- {fname}: (预览失败 {e})")
    return "\n".join(lines) or "(无可预览的表格数据)"
