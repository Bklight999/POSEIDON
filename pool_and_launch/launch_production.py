#!/usr/bin/env python3
"""Launch a sharded production run over a fixed fresh repository pool. Configure hosts, pool size, shard size, source root, and per-type quota before use. Smaller shards improve completion coverage; yield does not scale linearly with pool size. Keep concurrent mining outputs separate and preserve source symlinks until the run finishes."""
import concurrent.futures as cf
import os
os.environ.setdefault("STEM_DEVNULL", os.devnull)
import time

import paramiko

WORKDIR = os.getenv("STEM_PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PW = os.getenv("STEM_SSH_PASSWORD", "")
PORT = int(os.getenv("STEM_SSH_PORT", "22"))
PY = os.getenv("STEM_PYTHON_BIN", "python3")

ALL_IPS = [ln.strip() for ln in open(os.path.join(WORKDIR, "docker-ips-32.txt"),
                                     encoding="utf-8")
           if ln.strip() and not ln.lstrip().startswith("#")]
assert ALL_IPS, "Configure at least one worker host"

# Exclude unavailable hosts from shard allocation so their work is not silently lost.
DEAD = set()
IPS = [ip for ip in ALL_IPS if ip not in DEAD]

POOL_REPOS = 4431                      # Set pool size and round shard size upward to cover every repository; retain the configured per-type count.
PER = -(-POOL_REPOS // len(IPS))       
PER_TYPE = 14                          

STAGE1_JSONL = f"{WORKDIR}/workspace/b19_fresh_scientific_python_files.jsonl"
STAGE1_REPOS = f"{WORKDIR}/workspace/download_repos_b19"   # Use the symlink aggregation directory.

PRE = r"""
nohup sh -c "pkill -f 'stemgen.orchestrato[r]'; sleep 3; pkill -9 -f 'stemgen.orchestrato[r]'; pkill -9 -f '_run_[0-9]*\.py'; pkill -9 -f '_verif[y]\.py'" >${STEM_DEVNULL} 2>&1 &
sleep 8
echo "清场: orch=$(pgrep -cf 'stemgen.orchestrato[r]' 2>${STEM_DEVNULL}||echo 0)"
"""

INSTALL_PIPREQS = f"""
export no_proxy='localhost,127.0.0.1'
export NO_PROXY=$no_proxy
export http_proxy="$http_proxy"
export https_proxy=$http_proxy HTTP_PROXY=$http_proxy HTTPS_PROXY=$http_proxy
if [ -x {os.path.dirname(PY)}/pipreqs ]; then
  echo "pipreqs: 已存在"
else
  {PY} -m pip install --no-cache-dir -q pipreqs 2>&1 | tail -2
  [ -x {os.path.dirname(PY)}/pipreqs ] && echo "pipreqs: 装好" || echo "pipreqs: ❌装失败"
fi
"""


def cmd(i: int) -> str:
    repo_off = i * PER
    run = f"realdata_b19_off{repo_off}"
    outd = f"{WORKDIR}/tasks_out/{run}"
    return (
        f"mkdir -p {outd} && cd {WORKDIR} && export PYTHONPATH={WORKDIR} && "
        "export no_proxy='localhost,127.0.0.1' && "
        "export NO_PROXY=$no_proxy && export http_proxy=\"$http_proxy\" && "
        "export https_proxy=$http_proxy && export HTTP_PROXY=$http_proxy && export HTTPS_PROXY=$http_proxy && "
        "export OPENAI_API_KEY='" + os.getenv("STEM_GW_APP_ID", "") + ":" + os.getenv("STEM_GW_APP_KEY", "") + "' && "
        "export OPENAI_BASE_URL='https://api.example.com/v1' && "
        "export STEM_LLM_ENGINE='openai_deepseek-v4-flash' && export STEM_GEN_THINKING=1 && "
        "export STEM_GW_MODE='openai' && "
        "export STEM_GW_LIMIT_FLASH=25 && export STEM_GW_LIMIT_PRO=14 && "
        # Use the fresh repository pool and its aggregated source directory.
        f"export STEM_STAGE1_JSONL='{STAGE1_JSONL}' && "
        f"export STEM_STAGE1_REPOS='{STAGE1_REPOS}' && "
        "export STEM_QC_WORKERS=16 && export STEM_PKG_WORKERS=32 && "
        "export STEM_JUDGE_WORKERS=16 && export STEM_FLASH_WORKERS=16 && "
        "export STEM_QC_MAX_TOKENS=24576 && export STEM_QC_CODE_CHARS=20000 && "
        "export STEM_QC_GW_TIMEOUT=500 && "
        "export STEM_CHILD_MEM_GB=24 && export STEM_PKG_TIMEOUT=21600 && "
        f"nohup {PY} -u -m stemgen.orchestrator "
        "--seeds all --shuffle-seeds --shuffle-seed 9 --require-real-data "
        f"--seed-workers 16 --per-type {PER_TYPE} --derive-workers 16 --ref-workers 4 "
        "--adapt-iters 3 --seed-env --adapt-gate --resume --difficulty off "
        # Pass --difficulty off explicitly; the legacy mock grader is not included.
        f"--python-bin {PY} --repo-offset {repo_off} --repo-limit {PER} --max-seeds 3000 "
        f"--run-name {run} --out {outd}/candidates.jsonl "
        f">> {outd}/nohup_b19.log 2>&1 & echo LAUNCHED"
    )


def go(i: int):
    ip = IPS[i]
    lo, hi = i * PER, min((i + 1) * PER, POOL_REPOS)
    # Use a longer SSH banner timeout and bounded launch concurrency to avoid handshake overload.


    err = None
    for _ in range(3):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(ip, port=PORT, username="root", password=PW,
                      timeout=30, banner_timeout=60, auth_timeout=30)
            _, o, _ = c.exec_command("export STEM_DEVNULL=" + os.devnull + "; " + PRE, timeout=90)
            r1 = " ".join(o.read().decode(errors="replace").split())
            _, o, _ = c.exec_command(INSTALL_PIPREQS, timeout=300)
            r2 = " ".join(o.read().decode(errors="replace").split())[-60:]
            # Launch through a raw channel without waiting on output held open by the background process.
            
            ch = c.get_transport().open_session()
            ch.exec_command(cmd(i))
            time.sleep(3)
            c.close()
            return (i, f"{ip:16} repo[{lo},{hi}) {r1} | {r2} → 已启动")
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:70]}"
            time.sleep(5)
    return (i, f"{ip:16} ERR(重试3次) {err}")


if __name__ == "__main__":
    print(f"=== Batch19: {len(IPS)} 台 × {PER} repo（池 {POOL_REPOS}）=== {time.strftime('%F %T')}")
    print(f"    池={os.path.basename(STAGE1_JSONL)}  仓库目录={os.path.basename(STAGE1_REPOS)}（软链）")
    print(f"    per-type={PER_TYPE}（6 种题型 → 每 seed 最多 {PER_TYPE*6} 道）")
    print(f"    跳过 {sorted(DEAD)}  run-name=realdata_b19_off*")
    # Limit concurrent SSH handshakes rather than connecting to every host at once.
    with cf.ThreadPoolExecutor(10) as ex:
        for _, out in sorted(ex.map(go, range(len(IPS)))):
            print(f"  {out}")
    print(f"=== 全部启动完成 === {time.strftime('%F %T')}")
