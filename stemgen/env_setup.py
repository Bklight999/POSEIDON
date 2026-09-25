"""Prepare repository-level execution environments and reproducible build materials shared by derived tasks. Inspect dependency metadata, generate self-contained Dockerfiles, and write build scripts and manifests without invoking Docker. Local interpreter backends support reference validation. Importing this module performs no installation."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict


# --------------------------------------------------------------------------
# Structured environment requirements for one repository.
# --------------------------------------------------------------------------
@dataclass
class EnvSpec:
    repo: str
    python_version: str = "3.11"          # Default when the version cannot be detected.
    pip_requirements: list[str] = field(default_factory=list)   # Dependencies from requirements.txt or pyproject metadata.
    conda_environment_yml: str | None = None   # Repository-relative environment and Dockerfile paths, repository root, and whether R/Bioconductor support is required.
    dockerfile: str | None = None              
    repo_dir: str = ""                         
    needs_r: bool = False                      
    notes: list[str] = field(default_factory=list)
    # Populate validated dependencies from the execution environment so generated Dockerfiles match the environment that produced gold answers.
    
    resolved_deps: list[str] = field(default_factory=list)   # Merged unpinned dependencies and the exact pip-freeze versions.
    frozen_deps: list[str] = field(default_factory=list)     
    # Record the Python version used for successful reference execution and prefer it when generating Dockerfiles.
    
    runtime_python_version: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# Extract project dependencies from pyproject.toml without adding a TOML parser dependency.
_DEP_BLOCK_RE = re.compile(r"dependencies\s*=\s*\[(.*?)\]", re.DOTALL)
_DEP_ITEM_RE = re.compile(r'["\']([^"\']+)["\']')


# Scientific-data extensions for the repository availability gate; exclude code, configuration, and documentation.
_DATA_EXT = (".csv", ".tsv", ".h5ad", ".h5", ".loom", ".mtx", ".vcf", ".vcf.gz",
             ".fasta", ".fa", ".fastq", ".fq", ".bed", ".gff", ".gff3", ".gtf",
             ".bam", ".sam", ".npy", ".npz", ".parquet", ".pkl", ".rds", ".agp",
             ".tab", ".mat", ".nii", ".pdb")
# Names with data-like extensions that are actually configuration or dependency declarations.
_DATA_NAME_BLOCKLIST = {"requirements.txt", "readme.txt", "license.txt", "setup.py",
                        "config.txt", "changelog.txt"}


def probe_repo_data(repo_dir: str, *, min_bytes: int = 1024,
                    max_scan: int = 20000) -> list[str]:
    """Find real scientific data files by extension, size, and name filters, excluding .git. Return repository-relative paths; an empty result means no qualifying data was found."""
    found = []
    if not os.path.isdir(repo_dir):
        return found
    scanned = 0
    for dp, dirs, fns in os.walk(repo_dir):
        if ".git" in dp.split(os.sep):
            dirs[:] = [d for d in dirs if d != ".git"]
            continue
        for fn in fns:
            scanned += 1
            if scanned > max_scan:
                return found
            low = fn.lower()
            if low in _DATA_NAME_BLOCKLIST:
                continue
            if not low.endswith(_DATA_EXT):
                continue
            p = os.path.join(dp, fn)
            try:
                if os.path.getsize(p) > min_bytes:
                    found.append(os.path.relpath(p, repo_dir))
            except OSError:
                pass
    return found


def probe_repo_env(repo_dir: str) -> EnvSpec:
    """Inspect repository dependency declarations into an EnvSpec, retaining defaults and notes on failures."""
    repo = os.path.basename(repo_dir.rstrip("/"))
    spec = EnvSpec(repo=repo)
    spec.repo_dir = os.path.abspath(repo_dir)   # Retain the source root for repository environment metadata.

    def rel_if_exists(fname: str) -> str | None:
        p = os.path.join(repo_dir, fname)
        return fname if os.path.exists(p) else None

    # Inspect the repository's own Dockerfile first.
    spec.dockerfile = rel_if_exists("Dockerfile")

    # 2. conda environment.yml
    spec.conda_environment_yml = rel_if_exists("environment.yml") or rel_if_exists("environment.yaml")
    if spec.conda_environment_yml:
        try:
            with open(os.path.join(repo_dir, spec.conda_environment_yml),
                      encoding="utf-8", errors="ignore") as f:
                txt = f.read()
            if re.search(r"bioconductor|r-base|\br-\w", txt):
                spec.needs_r = True
                spec.notes.append("environment.yml 含 R/bioconductor 依赖")
        except OSError:
            pass

    # Read requirements.txt, distinguishing pip requirements from incompatible conda build specifications.
    req = rel_if_exists("requirements.txt")
    if req:
        try:
            with open(os.path.join(repo_dir, req), encoding="utf-8", errors="ignore") as f:
                raw = [ln.strip() for ln in f]
            lines = []
            for ln in raw:
                if not ln or ln.startswith("#"):
                    continue
                ln = ln.split("#", 1)[0].strip()   # Strip inline comments.
                if ln:
                    lines.append(ln)
            # Do not pass conda-style package=version=build entries to pip.
            conda_style = sum(1 for ln in lines if ln.count("=") >= 2) > len(lines) / 2
            if conda_style:
                spec.notes.append("requirements.txt 是 conda spec 风格，pip 不可直接用")
            else:
                spec.pip_requirements = lines
        except OSError:
            pass

    # Use pyproject dependencies when requirements are unavailable.
    if not spec.pip_requirements:
        pyproj = os.path.join(repo_dir, "pyproject.toml")
        if os.path.exists(pyproj):
            try:
                with open(pyproj, encoding="utf-8", errors="ignore") as f:
                    txt = f.read()
                m = _DEP_BLOCK_RE.search(txt)
                if m:
                    spec.pip_requirements = _DEP_ITEM_RE.findall(m.group(1))
                pm = re.search(r'requires-python\s*=\s*["\']([^"\']+)["\']', txt)
                if pm:
                    vm = re.search(r"(\d+\.\d+)", pm.group(1))
                    if vm:
                        spec.python_version = vm.group(1)
            except OSError:
                pass

    if not any([spec.dockerfile, spec.conda_environment_yml, spec.pip_requirements]):
        spec.notes.append("无任何依赖声明，将用统一 python 基础镜像 + numpy/pandas/scipy 兜底")
    return spec


# --------------------------------------------------------------------------
# Generate a Dockerfile from the environment specification.
# --------------------------------------------------------------------------
_FALLBACK_PIP = ["numpy", "pandas", "scipy", "scikit-learn"]


# Map import names to real PyPI distributions to avoid placeholder packages such as sklearn.

_PIP_NAME_FIX = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "bio": "biopython", "Bio": "biopython",
    "skimage": "scikit-image",
    "PIL": "pillow", "pil": "pillow",
    "yaml": "pyyaml",
    "serial": "pyserial",
    "dateutil": "python-dateutil",
    "attr": "attrs",
    "sklearn-learn": "scikit-learn",
}


def _strip_local_version(dep: str) -> str:
    """Normalize local and prerelease version pins for public package installation. Strip local build suffixes and use lower bounds for development or release-candidate versions that may not exist on PyPI."""
    m = re.match(r"^\s*([A-Za-z0-9_.\-\[\]]+)==([^\s;]+)(.*)$", dep)
    if not m:
        return dep
    name, ver, rest = m.group(1), m.group(2), m.group(3)
    ver = ver.split("+", 1)[0]                      # Strip local-version and prerelease suffixes.
    if re.search(r"(\.dev|rc|a|b)\d*$", ver):       
        base = ver
        while re.search(r"(\.dev|rc|a|b)\d*$", base):   # Suffixes can be combined, such as release-candidate and development tags.
            base = re.sub(r"(\.dev|rc|a|b)\d*$", "", base)
        return f"{name}>={base}{rest}" if base else name + rest
    return f"{name}=={ver}{rest}"


def _fix_pip_names(pip_list: list[str]) -> list[str]:
    """Normalize import aliases to distribution names, preserve version constraints, and deduplicate in order."""
    out, seen = [], set()
    for dep in pip_list or []:
        dep = _strip_local_version(dep)
        m = re.match(r"^\s*([A-Za-z0-9_.\-]+)(.*)$", dep)
        if not m:
            fixed = dep
        else:
            name, rest = m.group(1), m.group(2)
            canon = _PIP_NAME_FIX.get(name) or _PIP_NAME_FIX.get(name.lower())
            fixed = (canon + rest) if canon else dep
        key = fixed.lower()
        if key not in seen:
            seen.add(key); out.append(fixed)
    return out


# Minimum Python versions required by common dependency versions, used to validate the base image.

_PKG_MIN_PY = {
    "pandas":       [("2.0", (3, 9)), ("1.5", (3, 8)), ("1.4", (3, 8)), ("1.0", (3, 7))],
    "numpy":        [("1.25", (3, 9)), ("1.22", (3, 8)), ("1.20", (3, 7))],
    "scipy":        [("1.11", (3, 9)), ("1.9", (3, 8)), ("1.6", (3, 7))],
    "scikit-learn": [("1.3", (3, 8)), ("1.1", (3, 8)), ("1.0", (3, 7))],
    "matplotlib":   [("3.7", (3, 8)), ("3.6", (3, 8))],
}


def _ver_ge(a: str, b: str) -> bool:
    """Compare dotted versions, padding missing components with zeros."""
    def parts(s):
        out = []
        for x in re.split(r"[.\-]", s):
            if x.isdigit():
                out.append(int(x))
            else:
                break
        return out
    pa, pb = parts(a), parts(b)
    n = max(len(pa), len(pb))
    pa += [0] * (n - len(pa)); pb += [0] * (n - len(pb))
    return pa >= pb


def _required_min_python(pip_list: list[str]) -> tuple[int, int]:
    """Return the minimum Python version required by known dependencies, defaulting to 3.7 when unknown."""
    floor = (3, 7)
    for dep in pip_list or []:
        m = re.match(r"^\s*([A-Za-z0-9_.\-]+)\s*==\s*([0-9][0-9.]*)", dep)
        if not m:
            continue
        pkg, ver = m.group(1).lower(), m.group(2)
        table = _PKG_MIN_PY.get(pkg)
        if not table:
            continue
        for thresh, minpy in table:      # Check descending version thresholds and take the first match.
            if _ver_ge(ver, thresh):
                if minpy > floor:
                    floor = minpy
                break
    return floor


def _effective_py_version(spec: "EnvSpec", pip_list: list[str]) -> tuple[str, str]:
    """Select the Dockerfile Python version: prefer the actual reference runtime, then repository metadata, and raise it to dependency minimums. Return version and explanation."""
    notes = []
    # Prefer the Python version from the successful execution environment.
    if spec.runtime_python_version:
        base = spec.runtime_python_version
        notes.append(f"Use Python {base} from the validated reference environment")
    else:
        base = spec.python_version or "3.11"
        notes.append(f"Repository specifies Python {base}")
    # Raise the base version if dependencies require a newer interpreter.
    need_maj, need_min = _required_min_python(pip_list)
    cur = tuple(int(x) for x in re.findall(r"\d+", base)[:2]) or (3, 11)
    if len(cur) < 2:
        cur = (cur[0], 0)
    if (need_maj, need_min) > cur:
        base = f"{need_maj}.{need_min}"
        notes.append(f"Raised Python {cur[0]}.{cur[1]} to dependency minimum {base}")
    return base, "; ".join(notes)


# Map module imports to installable distribution names.
_IMPORT_TO_PIP = {
    "sklearn": "scikit-learn", "cv2": "opencv-python", "PIL": "pillow",
    "Bio": "biopython", "yaml": "pyyaml", "skimage": "scikit-image",
    "bs4": "beautifulsoup4", "mpl_toolkits": "matplotlib", "OpenSSL": "pyOpenSSL",
    "dateutil": "python-dateutil", "pkg_resources": "setuptools",
    "serial": "pyserial", "usb": "pyusb", "Crypto": "pycryptodome",
    "statsmodels.api": "statsmodels", "matplotlib.pyplot": "matplotlib",
}
# Standard-library modules do not need pip installation.
_STDLIB = frozenset("""
os sys re json math cmath time random itertools functools collections pathlib typing datetime
csv io warnings subprocess glob shutil argparse copy pickle string textwrap statistics decimal
fractions hashlib tempfile logging operator unittest ast dataclasses enum abc contextlib inspect
traceback gzip bz2 lzma zipfile tarfile sqlite3 urllib http socket threading multiprocessing queue
heapq bisect array struct binascii base64 uuid platform getpass pprint difflib xml concurrent
ntpath posixpath optparse __future__ importlib signal errno stat locale codecs unicodedata secrets
fnmatch gc fileinput shlex getopt keyword token tokenize weakref types numbers select selectors
asyncio contextvars dbm filecmp ftplib imaplib mimetypes pdb profile pstats pty quopri sched
smtplib ssl stringprep symtable sysconfig tabnanny telnetlib termios tty uu wave webbrowser zlib
builtins site marshal atexit ctypes curses email encodings ensurepip html imp ipaddress json.decoder
""".split())


def extract_pip_imports(code: str) -> list[str]:
    """Extract third-party dependencies from reference-program imports, excluding the standard library and mapping module names to distributions. Prefer AST parsing with a regex fallback; repository declarations alone may omit packages used by generated references."""
    mods: set[str] = set()
    try:
        import ast as _ast
        tree = _ast.parse(code or "")
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Import):
                for a in node.names:
                    mods.add(a.name.split(".")[0])
            elif isinstance(node, _ast.ImportFrom):
                if node.level == 0 and node.module:      # Skip relative imports.
                    mods.add(node.module.split(".")[0])
    except (SyntaxError, ValueError):
        for m in re.finditer(r"^\s*(?:import|from)\s+([A-Za-z_][\w.]*)", code or "", re.M):
            mods.add(m.group(1).split(".")[0])
    out = []
    for m in sorted(mods):
        if not m or m.startswith("_") or m in _STDLIB:
            continue
        out.append(_IMPORT_TO_PIP.get(m, m))
    return list(dict.fromkeys(out))


def gen_dockerfile(spec: EnvSpec, repo_dir_in_image: str = "/path/to/repo",
                   gold_program: str = "") -> tuple[str, str]:
    """Return Dockerfile text and its source category. Prefer validated dependencies merged with actual reference imports, inline conda metadata when needed, and otherwise use a Python fallback. Avoid repository Dockerfiles that depend on absent build contexts or unrelated runtimes."""
    # Merge actual reference-program imports with repository dependencies, preserving repository version pins.
    
    gold_pkgs = extract_pip_imports(gold_program) if gold_program else []

    def _merge(base: list[str]) -> list[str]:
        """Append reference dependencies not already covered by base package names, ignoring version constraints for matching."""
        have = set()
        for d in base or []:
            nm = re.split(r"[<>=!~\[ ]", str(d).strip(), 1)[0].strip().lower()
            if nm:
                have.add(nm.replace("_", "-"))
        out = list(base or [])
        for p in gold_pkgs:
            if p.lower().replace("_", "-") not in have:
                out.append(p)          # Use unpinned installation when no version constraint is available.
        return out

    # Prefer dependencies validated and frozen in the execution environment; fall back to repository declarations.
    
    pip_list = spec.frozen_deps or spec.resolved_deps or spec.pip_requirements
    pip_list = _merge(pip_list)
    if pip_list:
        pip_list = _fix_pip_names(pip_list)   # Normalize import aliases such as sklearn to scikit-learn before installation.
        src_note = ("venv pip freeze (versions from the reference execution environment)" if spec.frozen_deps
                    else "validated venv dependencies (requirements and pipreqs)" if spec.resolved_deps
                    else "repository requirements/pyproject declarations")
        if gold_pkgs:
            src_note += " + actual gold_program imports"
        pkgs = " ".join(f'"{p}"' for p in pip_list)
        eff_py, py_note = _effective_py_version(spec, pip_list)
        ref = (f"# Repository Dockerfile ({spec.repo}/{spec.dockerfile}) is not used: "
               f"it may require the complete repository, system packages, or GPU base image; delivery includes only data/.\n"
               if spec.dockerfile else "")
        df = f"""# Generated from EnvSpec; dependency source: {src_note}
# Python version: {py_note}
{ref}FROM python:{eff_py}-slim
WORKDIR {repo_dir_in_image}
RUN pip install --no-cache-dir {pkgs}
COPY . {repo_dir_in_image}
"""
        return df, "pip"

    # Inline conda environment YAML because the source repository file is not part of the delivered task; a COPY directive would fail.


    if spec.conda_environment_yml:
        body = ""
        try:
            _p = (os.path.join(spec.repo_dir, spec.conda_environment_yml)
                  if spec.repo_dir else "")
            if _p and os.path.isfile(_p):
                with open(_p, encoding="utf-8", errors="ignore") as f:
                    body = f.read()
        except OSError:
            body = ""
        if body.strip():
            # Use a unique heredoc delimiter and preserve indentation-sensitive YAML verbatim.
            df = f"""# Generated from EnvSpec with the repository conda environment.yml inlined
# Source: {spec.repo}/{spec.conda_environment_yml}
FROM continuumio/miniconda3:latest
WORKDIR {repo_dir_in_image}
RUN cat > environment.yml <<'STEM_ENV_YML_EOF'
{body.rstrip()}
STEM_ENV_YML_EOF
RUN conda env update -n base -f environment.yml && conda clean -afy
COPY . {repo_dir_in_image}
# Inputs are in data/; the agent runs in this working directory
"""
            return df, "environment_yml"
        # If the original YAML is unavailable, fall through instead of emitting a broken COPY instruction.

    # Do not reuse an unrelated repository Dockerfile for standard-library-only tasks: it may reference missing build inputs or lack Python. Use the Python fallback image.


    _repo_df_note = ""
    if spec.dockerfile:
        _repo_df_note = (f"# Repository Dockerfile ({spec.repo}/{spec.dockerfile}) is not used: "
                         f"it requires repository or system files absent from the delivered data and task scripts;\n"
                         f"# The reference uses only the standard library and can run in a Python base image.\n")

    # Fallback to a Python base with common numerical packages when there are no explicit dependencies or third-party reference imports.
    
    
    eff_py, py_note = _effective_py_version(spec, _FALLBACK_PIP)
    df = f"""# Fallback: use a standard Python image and common numerical packages when no dependencies are declared
# Python version: {py_note}
{_repo_df_note}FROM python:{eff_py}-slim
WORKDIR {repo_dir_in_image}
RUN pip install --no-cache-dir {' '.join(_FALLBACK_PIP)}
COPY . {repo_dir_in_image}
"""
    return df, "fallback"


# --------------------------------------------------------------------------
# Write per-seed Dockerfile, build script, and manifest without building the image.
# --------------------------------------------------------------------------
def prepare_seed_env(repo_dir: str, out_dir: str, *, image_prefix: str = "stem-seed",
                     registry: str = "") -> dict:
    """Write per-seed build materials to out_dir and return a manifest; build.sh must be run separately on a Docker-enabled host."""
    os.makedirs(out_dir, exist_ok=True)
    spec = probe_repo_env(repo_dir)
    dockerfile, source = gen_dockerfile(spec)
    image_tag = f"{image_prefix}-{spec.repo}".lower().replace("_", "-")
    full_tag = f"{registry.rstrip('/')}/{image_tag}" if registry else image_tag

    # Write environment metadata and the Dockerfile.
    with open(os.path.join(out_dir, "env_spec.json"), "w", encoding="utf-8") as f:
        json.dump(spec.to_dict(), f, ensure_ascii=False, indent=2)
    # Write a self-contained Dockerfile using the output directory as build context; delivered tasks do not contain the original repository.
    
    
    with open(os.path.join(out_dir, "Dockerfile"), "w", encoding="utf-8") as f:
        f.write(dockerfile)

    # Write a build script for a Docker-enabled host.
    ctx = out_dir
    dockerfile_arg = f"-f {os.path.join(out_dir, 'Dockerfile')}"
    build_sh = f"""#!/bin/bash
# Build on a host or CI runner with a Docker daemon
# seed: {spec.repo}  |  Dockerfile source: {source}
set -euo pipefail
docker build {dockerfile_arg} -t {full_tag} {ctx}
{f'docker push {full_tag}' if registry else '# No registry configured; set --registry to push'}
echo "built: {full_tag}"
"""
    build_path = os.path.join(out_dir, "build.sh")
    with open(build_path, "w", encoding="utf-8") as f:
        f.write(build_sh)
    os.chmod(build_path, 0o755)

    manifest = {
        "seed": spec.repo, "image_tag": full_tag, "dockerfile_source": source,
        "needs_r": spec.needs_r, "build_script": build_path,
        "env_spec": spec.to_dict(),
        "run_hint": f"docker run --rm -v <task_dir>:/path/to/task {full_tag} "
                    f"python /path/to/task/reference_workflow.py",
    }
    with open(os.path.join(out_dir, "env_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


# --------------------------------------------------------------------------
# Local execution backend without Docker.
# --------------------------------------------------------------------------
def prepare_local_env(spec: EnvSpec, backend: str = "current",
                      envs_root: str = "") -> str:
    """Return an interpreter path for sandbox execution. current is immediately usable; venv and conda modes report setup commands and the intended path for explicit caller-controlled installation."""
    import sys
    if backend == "current":
        return sys.executable
    # Return environment-setup commands rather than installing packages implicitly during import.
    target = os.path.join(envs_root or os.path.join(tempfile.gettempdir(), "stem_envs"), spec.repo)
    if backend == "venv":
        print(f"[env_setup] venv 方案：请执行\n  python -m venv {target} && "
              f"{target}/bin/pip install {' '.join(spec.pip_requirements or _FALLBACK_PIP)}")
        return os.path.join(target, "bin", "python")
    if backend == "conda":
        yml = spec.conda_environment_yml
        cmd = (f"conda env create -p {target} -f {yml}" if yml
               else f"conda create -p {target} python={spec.python_version} "
                    f"{' '.join(_FALLBACK_PIP)} -y")
        print(f"[env_setup] conda 方案：请执行\n  {cmd}")
        return os.path.join(target, "bin", "python")
    raise ValueError(f"未知 backend: {backend}")


# --------------------------------------------------------------------------
# Build the per-seed environment needed to execute source code.
#
# Use venv with system-site-packages to reuse installed scientific packages and install only missing dependencies. This avoids requiring conda repository access.


# --------------------------------------------------------------------------
import subprocess
import sys as _sys
import threading

_ENV_LOCK = threading.Lock()          # Build each repository environment once under concurrency; cache its interpreter, resolved and frozen dependencies, and runtime version.
_BUILT_ENVS: dict[str, str] = {}      
_BUILT_DEPS: dict[str, tuple] = {}    
def _pick_base_py() -> str:
    """Choose an existing base interpreter, with STEM_BASE_PY taking priority, rather than assuming a machine-specific environment exists."""
    cand = [os.getenv("STEM_BASE_PY", ""),
            os.getenv("STEM_PYTHON_BIN", "python3"),
            os.getenv("STEM_PYTHON_BIN", "python3"),
            _sys.executable]
    for p in cand:
        if p and os.path.exists(p):
            return p
    return _sys.executable


_BASE_PY = _pick_base_py()   # Base interpreter whose site-packages are inherited by each venv.


def _detect_python_version(python_bin: str) -> str:
    """Return the interpreter's major.minor version, or an empty string on failure."""
    try:
        r = subprocess.run(
            [python_bin, "-c", "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            capture_output=True, text=True, timeout=30)
        v = (r.stdout or "").strip()
        return v if re.match(r"^\d+\.\d+$", v) else ""
    except Exception:
        return ""


def _pip_freeze(env_prefix: str, resolved_deps: list[str] | None = None) -> list[str]:
    """Freeze effective dependency versions, including inherited packages. Keep unresolvable packages unpinned; use local pip freeze when the resolved list is empty."""
    py = os.path.join(env_prefix, "bin", "python")
    if not resolved_deps:
        pip = os.path.join(env_prefix, "bin", "pip")
        try:
            r = subprocess.run([pip, "freeze", "--local"],
                               capture_output=True, text=True, timeout=120)
            if r.returncode == 0:
                return [ln.strip() for ln in r.stdout.splitlines()
                        if ln.strip() and "==" in ln and not ln.startswith("-")]
        except Exception:  # noqa: BLE001
            pass
        return []
    # Query effective installed versions, including inherited dependencies, through importlib.metadata.
    import re as _re
    names = []
    for d in resolved_deps:
        nm = _re.split(r"[<>=!~\[ ]", d.strip(), 1)[0].strip()
        if nm:
            names.append(nm)
    if not names:
        return []
    probe = (
        "import json,sys\n"
        "try:\n"
        " from importlib.metadata import version,PackageNotFoundError\n"
        "except Exception:\n"
        " from importlib_metadata import version,PackageNotFoundError\n"
        "out=[]\n"
        f"for n in {names!r}:\n"
        " try: out.append(n+'=='+version(n))\n"
        " except Exception: out.append(n)\n"
        "print(json.dumps(out))\n"
    )
    try:
        r = subprocess.run([py, "-c", probe], capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            import json as _json
            return _json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        pass
    return []


def _pipreqs_cmd(repo_dir: str, out: str) -> list[str]:
    """Locate pipreqs next to the base interpreter or invoke its module, avoiding a dependency on a single machine's executable path."""
    sibling = os.path.join(os.path.dirname(_BASE_PY), "pipreqs")
    base = ([sibling] if os.path.exists(sibling)
            else [_BASE_PY, "-m", "pipreqs.pipreqs"])
    # Use pipreqs --use-local to avoid slow or unavailable PyPI lookups during import discovery.


    return base + [repo_dir, "--savepath", out, "--mode", "no-pin", "--force", "--use-local"]


def _scan_imports_pipreqs(repo_dir: str, worker_tag: str = "") -> list[str]:
    """Discover unpinned third-party imports with pipreqs to supplement incomplete requirements."""
    import tempfile
    out = os.path.join(tempfile.gettempdir(), f"reqs_{worker_tag}_{os.getpid()}.txt")
    try:
        # Clear PYTHONPATH for pipreqs: it recursively scans sys.path entries and can otherwise traverse millions of unrelated files on shared storage.


        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        # Resolve imports locally without network requests.
        pr = subprocess.run(_pipreqs_cmd(repo_dir, out),
                            capture_output=True, timeout=60, env=env)
        if pr.returncode == 0 and os.path.exists(out):
            return [ln.strip() for ln in open(out, encoding="utf-8", errors="ignore")
                    if ln.strip() and not ln.startswith("#")]
    except Exception:  # noqa: BLE001
        pass
    finally:
        try:
            os.remove(out)
        except OSError:
            pass
    return []


def _already_importable(pkg: str) -> bool:
    """Check whether a dependency or a known import alias is available in the base interpreter."""
    import re as _re
    name = _re.split(r"[<>=!~\[ ]", pkg.strip(), 1)[0].strip()
    if not name:
        return True
    mod = name.replace("-", "_")
    # Common distribution-to-import aliases for best-effort availability checks.
    alias = {"scikit_learn": "sklearn", "opencv_python": "cv2", "pillow": "PIL",
             "pyyaml": "yaml", "beautifulsoup4": "bs4", "biopython": "Bio",
             "scikit_image": "skimage", "msgpack_python": "msgpack"}
    mod = alias.get(mod.lower(), mod)
    try:
        r = subprocess.run([_BASE_PY, "-c", f"import {mod}"],
                           capture_output=True, timeout=30)
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def build_seed_conda_env(repo_dir: str, *, timeout_install: int = 900,
                         envs_root: str = os.path.join(tempfile.gettempdir(), "stem_seed_envs")) -> tuple[str, EnvSpec]:
    """Build and cache one venv per repository, installing missing declared and discovered dependencies. The historical function name retains conda, but the backend uses venv with system-site-packages. Fall back to the current interpreter on failure."""
    spec = probe_repo_env(repo_dir)
    repo = spec.repo
    with _ENV_LOCK:
        if repo in _BUILT_ENVS:
            if repo in _BUILT_DEPS:   # On a cache hit, restore dependency metadata into the current environment specification.
                spec.resolved_deps, spec.frozen_deps, spec.runtime_python_version = _BUILT_DEPS[repo]
            return _BUILT_ENVS[repo], spec

    env_prefix = os.path.join(envs_root, repo)
    py = os.path.join(env_prefix, "bin", "python")
    try:
        # Create a venv inheriting the base environment's installed packages.
        if not os.path.exists(py):
            os.makedirs(envs_root, exist_ok=True)
            cr = subprocess.run(
                [_BASE_PY, "-m", "venv", "--system-site-packages", env_prefix],
                capture_output=True, text=True, timeout=120)
            if cr.returncode != 0 or not os.path.exists(py):
                print(f"  [env] {repo} venv 建失败，回退当前解释器: {cr.stderr[:150]}")
                with _ENV_LOCK:
                    _BUILT_ENVS[repo] = _sys.executable
                return _sys.executable, spec

        # Collect dependencies and avoid reinstalling packages already inherited from the base environment.
        deps = list(spec.pip_requirements) + _scan_imports_pipreqs(repo_dir, repo[:20])
        deps = list(dict.fromkeys(d for d in deps if d))     # Deduplicate while preserving order.
        spec.resolved_deps = deps   # Record the complete requirements and discovered-import dependency set for Dockerfile generation.
        need = [d for d in deps if not _already_importable(d)]
        if need:
            import tempfile
            req = os.path.join(tempfile.gettempdir(), f"seedreq_{repo[:20]}_{os.getpid()}.txt")
            with open(req, "w") as f:
                f.write("\n".join(need))
            # Install with the venv's pip and inherited proxy settings; warn and continue on installation failure.
            ir = subprocess.run(
                [os.path.join(env_prefix, "bin", "pip"), "install", "-r", req],
                capture_output=True, text=True, timeout=timeout_install)
            if ir.returncode != 0:
                print(f"  [env] {repo} 部分依赖装失败(继续): {ir.stderr[:150]}")
            try:
                os.remove(req)
            except OSError:
                pass
        # Freeze effective versions, including inherited packages, for reproducible Dockerfiles.
        spec.frozen_deps = _pip_freeze(env_prefix, spec.resolved_deps)
        # Record the interpreter version used for successful reference execution.
        spec.runtime_python_version = _detect_python_version(py)
        print(f"  [env] {repo} venv 就绪: {py} (扫出{len(deps)}依赖, 需装{len(need)}, "
              f"freeze锁定{len(spec.frozen_deps)}, py{spec.runtime_python_version}, 其余继承 nexus)")
        with _ENV_LOCK:
            _BUILT_ENVS[repo] = py
            _BUILT_DEPS[repo] = (spec.resolved_deps, spec.frozen_deps, spec.runtime_python_version)
        return py, spec
    except Exception as e:  # noqa: BLE001
        print(f"  [env] {repo} 建 env 异常，回退当前解释器: {e}")
        with _ENV_LOCK:
            _BUILT_ENVS[repo] = _sys.executable
        return _sys.executable, spec


def remove_seed_conda_env(repo_dir: str, envs_root: str = os.path.join(tempfile.gettempdir(), "stem_seed_envs")):
    """Remove a seed's venv directory to release disk space."""
    import shutil
    repo = os.path.basename(repo_dir.rstrip("/"))
    env_prefix = os.path.join(envs_root, repo)
    if os.path.exists(env_prefix):
        shutil.rmtree(env_prefix, ignore_errors=True)
    with _ENV_LOCK:
        _BUILT_ENVS.pop(repo, None)


if __name__ == "__main__":
    import sys
    rd = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(__file__), "..", "stage1_smoke",
                     "download_repos", "diffexpr")
    print("探测:", json.dumps(probe_repo_env(rd).to_dict(), ensure_ascii=False, indent=2))
