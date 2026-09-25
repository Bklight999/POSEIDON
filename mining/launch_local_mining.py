#!/usr/bin/env python3
"""Launch local repository mining with an expanded tool/software keyword axis. Use independent CSV, JSONL, and clone outputs while production consumes a fixed pool snapshot. Register historical CSVs and clone directories separately for deduplication. Validate proxy access, retain README quality filtering, and allow for GraphQL quota waits. Usage: python mining/launch_local_mining.py [status|stop]."""
import os
os.environ.setdefault("STEM_DEVNULL", os.devnull)
import subprocess
import sys
import time

WORKDIR = os.getenv("STEM_PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = os.getenv("STEM_PYTHON_BIN", "python3")
WS = f"{WORKDIR}/workspace"
GH_TOKEN = os.getenv("GITHUB_TOKEN", "")
# Use an independent log for this round so offsets and status parsing cannot refer to a previous run.


LOG = f"{WS}/bio_local8.log"
OFFSET_FILE = f"{WS}/_c8_log_offset.txt"

# Give each concurrent mining round independent output paths through environment overrides.
CSV = f"{WS}/repositories_bio8.csv"
JSONL = f"{WS}/bio8_scientific_python_files.jsonl"
REPOS = f"{WS}/download_repos_bio8"

ENVS = (
    f"cd {WORKDIR} && export PYTHONPATH={WORKDIR} && "
    f"export GITHUB_TOKEN='{GH_TOKEN}' && "
    # Configure the proxy for GitHub access and exclude private model endpoints through no_proxy when required.
    
    "export no_proxy='localhost,127.0.0.1,"
    "localhost' && export NO_PROXY=$no_proxy && "
    "export http_proxy=\"$http_proxy\" && export https_proxy=$http_proxy && "
    "export HTTP_PROXY=$http_proxy && export HTTPS_PROXY=$http_proxy && "
    f"export BIO_REPO_CSV='{CSV}' && export BIO_JSONL_OUT='{JSONL}' && "
    f"export BIO_NEW_REPOS_DIR='{REPOS}' && "
)

KEYWORDS = f"{PY} -u mining/bio_repo_search.py keywords"
# Search across discipline, pharmaceutical, species/disease, and tool keywords without a primary-language restriction.

#   stem-bio-search-language-python-is-the-cap）。
# New keyword axes can discover new repositories in previously searched star ranges. Adaptive buckets save quota; conservative worker counts reduce native TLS and gateway pressure.


SEARCH = (f"{PY} -u mining/bio_repo_search.py search "
          "--min-stars 0 --max-stars 9 --star-step 3 --per-combo 50 "
          "--paging-workers 8 --llm-workers 128")
# Tune clone concurrency to local I/O capacity and simultaneous production traffic on shared storage.


CRAWL = (f"{PY} -u mining/bio_repo_search.py crawl "
         "--crawl-workers 48 --file-workers 16 "
         "--max-files-per-repo 60 --clone-timeout 300")


def running() -> str:
    """Detect an active local mining process. Match a Python interpreter directly executing the script; avoid matching the caller's own shell command or other incidental filename mentions."""
    out = subprocess.run(
        # Match Python processes directly executing the mining script.
        "ps -eo pid,etimes,comm,args | "
        "awk '$3 ~ /^python/ && $0 ~ /bio_repo_search\\.py (keywords|search|crawl|merge|all)/ "
        "{print $1, $2, $3}'",
        shell=True, capture_output=True, text=True).stdout.strip()
    return out


def do_start():
    busy = running()
    if busy:
        print("✗ 本机已有 mining/bio_repo_search.py 在跑（两个进程 append 同一文件会写坏）：")
        print("   " + busy.replace("\n", "\n   "))
        return 1

    # Validate the token and network route before starting a long run.
    chk = subprocess.run(
        ENVS + "curl -s -o ${STEM_DEVNULL} -w '%{http_code}' -m 30 "
        "-H \"Authorization: Bearer $GITHUB_TOKEN\" -H 'Content-Type: application/json' "
        "-d '{\"query\":\"{viewer{login}}\"}' https://api.github.com/graphql",
        shell=True, capture_output=True, text=True, timeout=120).stdout.strip()
    if chk != "200":
        print(f"✗ GitHub GraphQL 校验失败 (HTTP {chk})。000=代理不通，401=token 失效。")
        return 1
    print(f"✓ token + 代理可用（本机直连不通，全靠 star-proxy）")

    # Record the initial byte offset of the append-only log so status reports cover only this run.
    
    os.makedirs(WS, exist_ok=True)
    with open(OFFSET_FILE, "w", encoding="utf-8") as f:
        f.write(str(os.path.getsize(LOG) if os.path.exists(LOG) else 0))

    chain = " && ".join([KEYWORDS, SEARCH, CRAWL])
    # Detach the session and standard streams so the job survives terminal closure.
    cmd = (ENVS + f"setsid nohup sh -c '{chain}' < ${{STEM_DEVNULL}} >> {LOG} 2>&1 & echo STARTED_$!")
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
    time.sleep(5)
    print(f"✓ 已后台启动（本机）: keywords → search → crawl   {out}")
    print(f"  日志: {LOG}")
    print(f"  产物: {os.path.basename(CSV)} / {os.path.basename(JSONL)} / {os.path.basename(REPOS)}/")
    print(f"  查进度: python3 {os.path.basename(__file__)} status")
    now = running()
    print(f"  进程校验: {'✓ 已在跑' if now else '⚠ 没看到进程，看日志'}")
    return 0


def do_status():
    print(f"=== 本机挖矿进度 {time.strftime('%m-%d %H:%M:%S')} ===")
    p = running()
    print(f"进程: {'运行中' if p else '未运行'}")
    for ln in p.splitlines()[:3]:
        f = ln.split(None, 2)
        print(f"   pid={f[0]} 已跑 {int(f[1])/60:.0f}min")

    def n(path):
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                return sum(1 for _ in f)
        except OSError:
            return 0
    print(f"  {os.path.basename(CSV):<38} {max(n(CSV)-1,0):>7} 仓库")
    print(f"  {os.path.basename(JSONL):<38} {n(JSONL):>7} 科学文件")
    try:
        print(f"  {os.path.basename(REPOS)+'/':<38} {len(os.listdir(REPOS)):>7} 已 clone")
    except OSError:
        print(f"  {os.path.basename(REPOS)+'/':<38} {0:>7} 已 clone")

    if not os.path.exists(LOG):
        print("  （日志尚未生成）")
        return
    # Slice logs by byte offset before decoding; slicing decoded text by byte counts mixes old and current run data when multibyte characters are present.


    raw = open(LOG, "rb").read()
    off = 0
    try:
        off = int(open(OFFSET_FILE, encoding="utf-8").read().strip())
    except Exception:  # noqa: BLE001
        pass
    if not (0 <= off <= len(raw)):
        off = 0
    txt = raw[off:].decode("utf-8", "ignore")
    age = (time.time() - os.path.getmtime(LOG)) / 60
    print(f"  日志 {len(raw)/1048576:.1f}MB，{age:.0f} 分钟前更新（本轮 {len(txt)/1024:.0f}KB）")
    import re
    m = re.findall(r"\[(\d+)/(\d+)\]", txt)
    if m:
        print(f"  进度 {m[-1][0]}/{m[-1][1]}")
    for k in ("YES", "NO_非生命科学", "skip_已有", "未判定(限流)",
              "Yes", "No", "未判定", "规则预过滤", "clone失败"):
        v = re.findall(rf"{re.escape(k)}\s+(\d+)", txt)
        if v:
            print(f"    {k:<18} {v[-1]}")
    q = re.findall(r"\[配额\][^\n]*", txt)
    if q:
        print(f"  配额等待 {len(q)} 次，最近: {q[-1][:78]}")
    print("  日志尾部:")
    for ln in [x for x in txt.strip().splitlines() if x.strip()][-4:]:
        print(f"    {ln[:118]}")


def do_stop():
    subprocess.run("pkill -f 'bio_repo_search[.]py'", shell=True)
    time.sleep(2)
    print(f"剩余进程: {len(running().splitlines())}")


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "start"
    if a == "status":
        do_status()
    elif a == "stop":
        do_stop()
    else:
        sys.exit(do_start())
