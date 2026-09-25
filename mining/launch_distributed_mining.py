#!/usr/bin/env python3
"""Launch repository mining on a configured SSH worker. Supported actions are keywords, search, crawl, status, and stop; the default runs keywords, search, then crawl. Use one writer per output set because concurrent appenders can corrupt records and race deduplication. Configure host, credentials, paths, and network access for the deployment."""
import concurrent.futures as cf
import os
os.environ.setdefault("STEM_DEVNULL", os.devnull)
import subprocess
import sys
import time

import paramiko

WORKDIR = os.getenv("STEM_PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PW = os.getenv("STEM_SSH_PASSWORD", "")
PORT = int(os.getenv("STEM_SSH_PORT", "22"))
PY = os.getenv("STEM_PYTHON_BIN", "python3")
# Read the GitHub token from the environment.
GH_TOKEN = os.getenv("GITHUB_TOKEN", "")

IPS = [ln.strip() for ln in open(os.path.join(WORKDIR, "docker-ips-32.txt"),
                                 encoding="utf-8")
       if ln.strip() and not ln.lstrip().startswith("#")]

# Select an idle worker and update the index when the host inventory changes.


HOST_IDX = int(os.getenv("BIO_HOST_IDX", "0"))
HOST = IPS[HOST_IDX]

LOG = f"{WORKDIR}/workspace/bio_mining.log"

ENVS = (
    f"cd {WORKDIR} && export PYTHONPATH={WORKDIR} && "
    f"export GITHUB_TOKEN='{GH_TOKEN}' && "
    "export no_proxy='localhost,127.0.0.1,"
    "localhost' && export NO_PROXY=$no_proxy && "
    "export http_proxy=\"$http_proxy\" && export https_proxy=$http_proxy && "
    "export HTTP_PROXY=$http_proxy && export HTTPS_PROXY=$https_proxy && "
)

# Use finer star buckets and curated pharmaceutical keywords; keep paging concurrency within the tested rate-limit envelope.

#
# Curated terms improve coverage, and finer buckets reduce per-query truncation. The minimum-star setting controls the quality threshold.


#
# Adaptive buckets probe counts before splitting. A quota gate waits for reset instead of silently losing candidates after short retries.


#
# Search throughput depends on query count and the fraction excluded as already collected.


#       ③ judge_readme ≈ 0.5h
# Estimate crawling time separately from search.
#
# Batch candidates so the configured LLM worker count is actually used.
#       for cand in pex.map(one_combo, combos):  lex.map(judge_readme, cand)
# Sparse per-query batches underfill the executor; accumulate a full batch before classification.


SEARCH = (f"{PY} -u mining/bio_repo_search.py search "
          "--min-stars 10 --max-stars 1000 --star-step 10 --per-combo 50 "
          "--paging-workers 8 --llm-workers 256")
# Crawl with concurrency across repositories and files. Sparse repositories and clone I/O limit effective parallelism; increasing both worker counts has diminishing returns.


CRAWL = (f"{PY} -u mining/bio_repo_search.py crawl "
         "--crawl-workers 32 --file-workers 16 "
         "--max-files-per-repo 60 --clone-timeout 300")
# Rebuild keywords before search; the search stage only reads the existing keyword artifact.


KEYWORDS = f"{PY} -u mining/bio_repo_search.py keywords"


def ssh():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username="root", password=PW,
              timeout=60, banner_timeout=60, auth_timeout=60)
    return c


def run(cmd: str, timeout: int = 120) -> str:
    c = ssh()
    try:
        _, o, e = c.exec_command("export STEM_DEVNULL=" + os.devnull + "; " + cmd, timeout=timeout)
        out = o.read().decode(errors="replace")
        err = e.read().decode(errors="replace")
        return (out + (f"\n[stderr] {err}" if err.strip() else "")).strip()
    finally:
        c.close()


def do_start(steps: list[str]):
    # Do not start duplicate writers to shared output files.
    busy = run("pgrep -af 'bio_repo_search\\.py' | head -3")
    if busy.strip():
        print("✗ worker 上已有 mining/bio_repo_search.py 在跑，先 stop 或等它结束：")
        print("   " + busy.replace("\n", "\n   "))
        return 1

    # Validate the token before launching work.
    chk = run(ENVS + "curl -s -o ${STEM_DEVNULL} -w '%{http_code}' -m 30 "
              "-H \"Authorization: Bearer $GITHUB_TOKEN\" -H 'Content-Type: application/json' "
              "-d '{\"query\":\"{viewer{login}}\"}' https://api.github.com/graphql", timeout=90)
    if chk.strip() != "200":
        print(f"✗ GitHub GraphQL 校验失败 (HTTP {chk.strip()})。401=token 失效，000=代理/网络不通。")
        return 1
    print(f"✓ token 可用，worker={HOST}")

    # Run crawl only after search succeeds, since it consumes the search CSV.
    parts = []
    for s in steps:
        parts.append({"keywords": KEYWORDS, "search": SEARCH, "crawl": CRAWL}[s])
    chain = " && ".join(parts)
    # Detach stdin, stdout, stderr, and the process session. Otherwise a background process keeps SSH pipes open and makes a successful launch appear to time out.


    cmd = (ENVS + f"setsid nohup sh -c '{chain}' < ${{STEM_DEVNULL}} >> {LOG} 2>&1 & "
                  "echo STARTED_$!")
    out = run(cmd, timeout=60)
    print(f"✓ 已后台启动: {' → '.join(steps)}   {out.strip()}")
    print(f"  日志（共享盘，本机可直接看）: {LOG}")
    print(f"  查进度: python3 {os.path.basename(__file__)} status")
    return 0


def do_status():
    print(f"=== 挖掘进度 {time.strftime('%m-%d %H:%M:%S')}  worker={HOST} ===")
    proc = run("pgrep -af 'bio_repo_search\\.py' | grep -v pgrep | head -3")
    print(f"进程: {'运行中' if proc.strip() else '未运行'}")
    if proc.strip():
        for ln in proc.strip().splitlines():
            print(f"   {ln[:110]}")

    # Read artifacts directly from shared storage.
    csv_p = f"{WORKDIR}/workspace/repositories_bio.csv"
    jl_p = f"{WORKDIR}/workspace/bio_scientific_python_files.jsonl"
    repo_d = f"{WORKDIR}/workspace/download_repos_bio_and_medicine"

    def nlines(p):
        try:
            with open(p, encoding="utf-8", errors="ignore") as f:
                return sum(1 for _ in f)
        except OSError:
            return -1

    n_csv = nlines(csv_p)
    print(f"repositories_bio.csv          : {max(n_csv - 1, 0) if n_csv > 0 else 0} 个仓库"
          f"{'' if n_csv > 0 else '（尚无）'}")
    print(f"bio_scientific_python_files    : {max(nlines(jl_p), 0)} 条科学文件")
    try:
        print(f"download_repos_bio_and_medicine/: {len(os.listdir(repo_d))} 个已 clone 仓库")
    except OSError:
        print("download_repos_bio_and_medicine/: 0")

    if os.path.exists(LOG):
        age = (time.time() - os.path.getmtime(LOG)) / 60
        print(f"日志                          : {os.path.getsize(LOG)/1048576:.1f}MB，"
              f"{age:.0f} 分钟前更新")
        # Summarize search verdict counts.
        try:
            txt = open(LOG, encoding="utf-8", errors="ignore").read()[-400000:]
            import re
            for k in ("YES", "NO_非生命科学", "skip_已有", "未判定(限流)"):
                m = re.findall(rf"{re.escape(k)}\s+(\d+)", txt)
                if m:
                    print(f"   {k:14} {m[-1]}")
            tail = [l for l in txt.strip().splitlines() if l.strip()][-3:]
            print("   日志尾部:")
            for l in tail:
                print(f"     {l[:110]}")
        except OSError:
            pass


def do_stop():
    print(run("pkill -f 'bio_repo_search\\.py' ; sleep 2 ; "
              "n=$(pgrep -cf 'bio_repo_search\\.py'); echo \"剩余进程=$n\""))


if __name__ == "__main__":
    # Rebuild keywords by default so new curated terms enter the pool.
    arg = sys.argv[1] if len(sys.argv) > 1 else "keywords search crawl"
    if arg == "status":
        do_status()
    elif arg == "stop":
        do_stop()
    else:
        steps = [s for s in arg.split() if s in ("keywords", "search", "crawl")]
        if not steps:
            print(f"✗ 无效参数 '{arg}'。"
                  "可用: keywords / search / crawl / 'keywords search crawl' / status / stop")
            sys.exit(1)
        sys.exit(do_start(steps))
