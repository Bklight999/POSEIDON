#!/usr/bin/env python3
"""Discover life-science and pharmaceutical repositories. Stages: build keywords, search GitHub and classify READMEs, clone and classify Python/notebook files, then merge by file path. Persist discipline labels and reject unrelated domains before synthesis. See USAGE below."""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
import requests   # Use requests instead of urllib's shared opener to avoid concurrent TLS failures.
import concurrent.futures as cf
from collections import Counter

BASE = os.getenv("STEM_PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
from stemgen.safe_egress import sanitize_and_assert  # noqa: E402

WS = os.path.join(BASE, "workspace")
KW_SRC = [os.path.join(BASE, "mining", "keywords", "keywords_bio_pharma.txt")]
KW_OUT = os.path.join(WS, "keywords_bio_pharma.txt")
# Append curated pharmaceutical terms without discipline regex filtering, which would incorrectly remove most of them. Inline counts record search hits.


KW_PHARMA = os.path.join(WS, "keywords_pharma_manual.txt")
# Species and disease terms form an independent search axis. Do not apply discipline-name filters to these ordinary words.


KW_SPECIES = os.path.join(WS, "keywords_species_disease.txt")
# LLM-expanded tool, format, and database names are validated with repository counts between 30 and 5000, excluding invented and overly generic terms.


KW_TOOLS = os.path.join(WS, "keywords_tools_llm.txt")
# Exclude known zero-yield terms to avoid empty keyword/bucket combinations.

KW_ZERO_YIELD = {
    "bioinformation theory and biocybernetics", "biooptics and optical biophysics",
    "cryobiophysics", "immunobiochemistry", "toxicogenetics", "cellular radiobiology",
    "radioimmunology", "radiotoxicology", "ribonucleic acid-omics (transcriptomics)",
    "hydrobiology", "neurobiophysics", "neurobiochemistry", "neuromorphology",
    "ethnobotany", "plant parasitology", "insect toxicology", "insect virology",
    "animal parasitology", "other disciplines of radiobiology",
    "other disciplines of zoology", "other disciplines of microbiology",
}
# Allow independent output paths for concurrent mining rounds. Never append to a pool being consumed by offset-based production shards. Deduplicate against prior outputs separately.


REPO_CSV = os.getenv("BIO_REPO_CSV") or os.path.join(WS, "repositories_bio.csv")
# Keep the clone directory consistent with existing data; using an obsolete directory causes redundant cloning.


NEW_REPOS_DIR = (os.getenv("BIO_NEW_REPOS_DIR")
                 or os.path.join(WS, "download_repos_bio_and_medicine"))
OLD_REPOS_DIR = os.path.join(WS, "download_repos_concurrency")
BIO_JSONL = (os.getenv("BIO_JSONL_OUT")
             or os.path.join(WS, "bio_scientific_python_files.jsonl"))
# Register both prior CSV outputs and clone directories. These are independent deduplication paths and neither substitutes for the other.


EXTRA_HAVE = [os.path.join(WS, "download_repos_bio_and_medicine"),
              # Register the separate source directory while production symlinks still reference it; moving it would break active jobs.


              os.path.join(WS, "download_repos_bio7")]
EXTRA_CSV = [os.path.join(WS, "repositories_bio.csv"),
             os.path.join(WS, "repositories_bio3.csv"),
             os.path.join(WS, "repositories_bio4.csv"),
             os.path.join(WS, "repositories_bio5.csv"),
             # Register prior CSV outputs even when their clone directories have already been merged.


             os.path.join(WS, "repositories_bio6.csv"),
             # Include the latest round's CSV to avoid recollecting its repositories.
             os.path.join(WS, "repositories_bio7.csv")]
OLD_JSONL = os.path.join(WS, "concurrency_scientific_python_files.jsonl")
MERGED_JSONL = os.path.join(WS, "merged_scientific_python_files.jsonl")

# LLM gateway throughput measurements under concurrent classification requests.


#        16        3.2/s       3.9/s
#        64       12.1/s      16.2/s
#       128       18.8/s      30.1/s
#       256         —         47.5/s
#       512         —         59-79/s
#      1024         —        125.7/s
#      2048         —        163.8/s
# The newer gateway sustained higher throughput in this benchmark.
#
# Shared gateway throughput varies with other tenants. Leave concurrency headroom and retry occasional HTTP 504 responses.


from stemgen import llm_gen as _client
_GW = os.getenv("STEM_GW_MODE", "openai")
LLM_APP_ID = _client.GW_APP_ID
LLM_APP_KEY = _client.GW_APP_KEY
LLM_URL = _client.GW_URL
LLM_MODEL = _client.GW_MODEL
LLM_AUTH = _client._auth_header(LLM_MODEL)
LLM_KEY = os.getenv("STEM_GW_KEY", LLM_APP_KEY)

# Disable reasoning for narrow JSON classification. Small output budgets can otherwise be exhausted by reasoning tokens, producing empty or undecidable responses.


#
# The thinking field location is gateway-specific. An incorrect location may be silently ignored and exhaust the output budget.


LLM_THINKING_OFF: dict = {}
_client._apply_thinking(LLM_THINKING_OFF, False)

GH_API = "https://api.github.com/graphql"
GH_TOKEN = os.getenv("GITHUB_TOKEN", "")

# Reuse a requests.Session per worker thread. Sessions are not thread-safe; thread-local connection pools reduce handshake overhead.


_tls = threading.local()


def _llm_post(body: dict, timeout: int = 90):
    """Send an LLM request and return parsed JSON; propagate errors to the classifier retry loop."""
    s = getattr(_tls, "sess", None)
    if s is None:
        s = requests.Session()
        s.mount("http://", requests.adapters.HTTPAdapter(
            pool_connections=4, pool_maxsize=4, max_retries=0))
        _tls.sess = s
    r = s.post(LLM_URL, headers={"Content-Type": "application/json",
                                 "Authorization": LLM_AUTH},
               json=body, timeout=timeout)
    if r.status_code != 200:
        # Propagate occasional HTTP 504 responses for caller retries.
        raise RuntimeError(f"HTTP {r.status_code}")
    return r.json()

_lock = threading.Lock()

# Positive life-science and pharmaceutical keyword filters. Use word boundaries for short roots to avoid unrelated matches such as organic chemistry or fuel cells.


KEEP_RE = re.compile(r"""
    biolog|biochem|biophys|bioinform|biotech|biomed|bioeng|bio-?informat|
    \bgene\b|genom|genet|transcript|proteom|metabolom|epigen|
    protein|peptide|enzym|\brna\b|\bdna\b|nucleic|codon|ribosom|
    \bcell\b|\bcells\b|cellular|cytolog|tissue|\borgans?\b|organism|
    physiolog|anatom|histolog|
    immun|virus|viral|virolog|microbio|bacteri|fungi|parasit|
    medic|clinic|patholog|diagnos|therap|epidemiolog|public\s+health|
    pharma|\bdrug|toxicol|pharmacolog|
    neuro|\bbrain\b|
    ecolog|evolution|phylogen|taxonom|biodivers|zoolog|botan|\bplant\b|agricult|
    cancer|oncolog|tumor|
    \bomics\b|molecular\s+biolog|structural\s+biolog|systems\s+biolog|synthetic\s+biolog
""", re.I | re.X)
# Explicit exclusions override positive matches.
DROP_RE = re.compile(r"""
    acoustic|sonar|seismic|geophys|astron|cosmolog|astrophys|
    plasma|nuclear\s+(physics|reactor)|particle\s+physics|quantum\s+(field|chromo)|
    metallurg|ceramic|polymer\s+physics|semiconductor|photovolt|
    fuel\s+cell|solar\s+cell|galvanic\s+cell|unit\s+cell|
    inorganic|organoelement|organometallic|
    financ|econom|market|
    civil\s+engineer|structural\s+engineer|mechanical\s+engineer|aerospace|
    linguistic|speech|music
""", re.I | re.X)


def step_keywords(args):
    seen, kept, dropped, zero = set(), [], [], []
    for p in KW_SRC:
        if not os.path.exists(p):
            print(f"  [warn] 关键词源不存在: {p}")
            continue
        for ln in open(p, encoding="utf-8", errors="ignore"):
            t = ln.strip()
            if not t:
                continue
            # For bilingual source rows, keep the English side used in GitHub README searches.
            if "|" in t:
                t = t.split("|")[-1].strip()
            # Remove remaining Chinese characters from search expressions.
            t = re.sub(r"[一-鿿]+", "", t).strip(" -—|")
            if not t or t.lower() in seen:
                continue
            seen.add(t.lower())
            # Drop known zero-hit terms before expanding search buckets.
            if t.lower() in KW_ZERO_YIELD:
                zero.append(t)
                continue
            if KEEP_RE.search(t) and not DROP_RE.search(t):
                kept.append(t)
            else:
                dropped.append(t)
    n_bio = len(kept)

    # Append curated pharmaceutical terms without regex filtering; strip inline count comments and full-line comments.
    
    if os.path.exists(KW_PHARMA):
        for ln in open(KW_PHARMA, encoding="utf-8", errors="ignore"):
            t = ln.split("#", 1)[0].strip()
            if not t or t.lower() in seen:
                continue
            seen.add(t.lower())
            kept.append(t)
    else:
        print(f"  [warn] 药学词表不存在，本轮无药学覆盖: {KW_PHARMA}")
    n_pharma = len(kept) - n_bio

    # Append species and disease terms without discipline-name filtering.
    if os.path.exists(KW_SPECIES):
        for ln in open(KW_SPECIES, encoding="utf-8", errors="ignore"):
            t = ln.split("#", 1)[0].strip()
            if not t or t.lower() in seen:
                continue
            seen.add(t.lower())
            kept.append(t)
    else:
        print(f"  [warn] 物种/疾病词表不存在: {KW_SPECIES}")
    n_species = len(kept) - n_bio - n_pharma

    # Tool and software names add an independent search axis when existing keywords are exhausted. Counts were filtered during generation; do not reapply discipline-name regexes.


    if os.path.exists(KW_TOOLS):
        for ln in open(KW_TOOLS, encoding="utf-8", errors="ignore"):
            t = ln.split("#", 1)[0].strip()
            if not t or t.lower() in seen:
                continue
            seen.add(t.lower())
            kept.append(t)
    else:
        print(f"  [warn] 工具/软件词表不存在，本轮无该轴覆盖: {KW_TOOLS}")
    n_tools = len(kept) - n_bio - n_pharma - n_species

    os.makedirs(WS, exist_ok=True)
    with open(KW_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(kept) + "\n")
    print(f"  源关键词 {len(seen)} 条 → 生命科学 {n_bio} 条"
          f" + 药学 {n_pharma} 条 + 物种/疾病 {n_species} 条"
          f" + 工具/软件 {n_tools} 条 = **{len(kept)}** 条")
    print(f"  剔除：非生命科学 {len(dropped)} 条、实测零命中 {len(zero)} 条")
    print(f"  生命科学样例: {kept[:5]}")
    print(f"  药学样例:     {kept[n_bio:n_bio + 5]}")
    print(f"  物种/疾病样例: {kept[n_bio + n_pharma:n_bio + n_pharma + 5]}")
    print(f"  工具/软件样例: {kept[n_bio + n_pharma + n_species:][:5]}")
    print(f"  零命中剔除:   {zero[:5]}")
    print(f"  → {KW_OUT}")


# Request rateLimit alongside search results to wait for reset before exhausting the GraphQL quota.


GQL = """
query($q:String!,$after:String){
  rateLimit{remaining resetAt}
  search(query:$q, type:REPOSITORY, first:50, after:$after){
    pageInfo{hasNextPage endCursor}
    nodes{ ... on Repository {
      nameWithOwner url description stargazerCount
      defaultBranchRef{name}
      object(expression:"HEAD:README.md"){ ... on Blob { text } }
    }}
  }
}"""

# Use remaining quota and resetAt to pause proactively. Short exponential retries cannot bridge an hourly reset and would silently discard candidates.


#
# Compare quota readings at the same time: rolling resets invalidate comparisons across different instants. Reserve enough headroom for concurrent paging requests.


_QUOTA_FLOOR = int(os.getenv("GH_QUOTA_FLOOR", "40"))   # Wait for reset below this quota threshold.

# GH_SEARCH_LANG controls the optional language constraint; empty means unrestricted, and multiword language names are quoted.


#
# Removing the primary-language restriction increases recall: scientific repositories often use notebooks or other languages while still containing Python files.


#     language:"Jupyter Notebook"   1255   +26%
# Primary-language filtering was the limiting constraint in these comparisons; unrelated search constraints did not improve recall comparably.


#
# Per-file scientific Python filtering still applies during crawling. Broader search increases cloning and classification work.


_SEARCH_LANG = os.getenv("GH_SEARCH_LANG", "")
if not _SEARCH_LANG:
    _LANG_Q = ""
elif " " in _SEARCH_LANG:
    _LANG_Q = f' language:"{_SEARCH_LANG}"'
else:
    _LANG_Q = f" language:{_SEARCH_LANG}"
_quota_lock = threading.Lock()
_quota = {"remaining": None, "reset_ts": 0.0, "waits": 0, "slept": 0.0,
          # Count REST quota rechecks and false alarms.
          
          "rechecks": 0, "false_alarms": 0}


def _parse_reset(s: str) -> float:
    """Convert an ISO timestamp to epoch seconds; fall back to a 60-second wait if parsing fails."""
    try:
        return time.mktime(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ")) - time.timezone
    except Exception:  # noqa: BLE001
        return time.time() + 60


def _rest_quota() -> "tuple[int, float] | None":
    """Read GraphQL remaining quota and reset time through the REST rate-limit endpoint. Return None on failure for conservative caller handling."""
    try:
        r = _gh_sess().get("https://api.github.com/rate_limit", timeout=30,
                           headers={"Authorization": f"Bearer {GH_TOKEN}"})
        if r.status_code == 200:
            g = r.json()["resources"]["graphql"]
            return int(g["remaining"]), float(g["reset"])
    except Exception:  # noqa: BLE001
        pass
    return None


def _quota_gate() -> None:
    """Block requests when quota is exhausted. Recheck through REST before sleeping to avoid pausing every worker on stale or misleading concurrent response metadata."""
    while True:
        with _quota_lock:
            rem, rts = _quota["remaining"], _quota["reset_ts"]
            if rem is None or rem > _QUOTA_FLOOR:
                return                      # Allow requests while quota is unknown or sufficient.
        # Recheck before sleeping, outside the lock so the REST request cannot block other threads.
        truth = _rest_quota()
        if truth is not None:
            t_rem, t_rts = truth
            with _quota_lock:
                _quota["remaining"], _quota["reset_ts"] = t_rem, t_rts
                _quota["rechecks"] += 1
                if t_rem > _QUOTA_FLOOR:
                    _quota["false_alarms"] += 1
                    n_fa = _quota["false_alarms"]
            if t_rem > _QUOTA_FLOOR:
                print(f"    [配额] 假警报：响应说 remaining={rem}，"
                      f"REST 实测 {t_rem} → 放行（第 {n_fa} 次假警报）", flush=True)
                return
            rem, rts = t_rem, t_rts
        with _quota_lock:
            wait = max(rts - time.time(), 0) + 5
            if wait <= 5:                   # If reset time has passed, allow a request to refresh quota state.
                _quota["remaining"] = None
                return
            _quota["waits"] += 1
            _quota["slept"] += wait
            n = _quota["waits"]
        print(f"    [配额] remaining={rem} <= {_QUOTA_FLOOR}（REST 已复核），"
              f"等 {wait/60:.1f}min 到 reset"
              f"（第 {n} 次，累计等待 {_quota['slept']/60:.0f}min）", flush=True)
        time.sleep(wait)


def _gh_sess():
    """Return a thread-local requests.Session for GitHub calls. Avoid urllib's global opener: concurrent shared TLS state caused reproducible native crashes in prior runs."""
    s = getattr(_tls, "gh", None)
    if s is None:
        s = requests.Session()
        s.mount("https://", requests.adapters.HTTPAdapter(
            pool_connections=4, pool_maxsize=4, max_retries=0))
        _tls.gh = s
    return s


def _rest_reset_ts() -> float:
    """Obtain the GraphQL reset timestamp through REST because rate-limit error responses omit resetAt. Fall back to a conservative 120-second delay on failure."""
    try:
        r = _gh_sess().get("https://api.github.com/rate_limit", timeout=30,
                           headers={"Authorization": f"Bearer {GH_TOKEN}"})
        if r.status_code == 200:
            return float(r.json()["resources"]["graphql"]["reset"])
    except Exception:  # noqa: BLE001
        pass
    return time.time() + 120


def gh_search(q, after=None, tries=5):
    body = {"query": GQL, "variables": {"q": q, "after": after}}
    for i in range(tries):
        _quota_gate()
        try:
            resp = _gh_sess().post(
                GH_API, json=body, timeout=60,
                headers={"Authorization": f"Bearer {GH_TOKEN}",
                         "Content-Type": "application/json"})
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}")
            out = resp.json()
            rl = ((out.get("data") or {}) or {}).get("rateLimit") or {}
            if rl.get("remaining") is not None:
                with _quota_lock:
                    _quota["remaining"] = rl["remaining"]
                    _quota["reset_ts"] = _parse_reset(rl.get("resetAt", ""))
            if "errors" in out and not out.get("data"):
                # Rate-limit errors may omit data.rateLimit. Explicitly exhaust the local quota state so the gate handles waiting instead of wasting retries.


                if "RATE_LIMIT" in str(out["errors"]).upper():
                    with _quota_lock:
                        _quota["remaining"] = 0
                        if _quota["reset_ts"] <= time.time():
                            _quota["reset_ts"] = _rest_reset_ts()
                    _quota_gate()
                    continue        # Retry after waiting without counting this as a failure.
                raise RuntimeError(str(out["errors"])[:200])
            return out["data"]["search"]
        except Exception as e:  # noqa: BLE001 Back off for secondary rate limits and server errors.
            if i == tries - 1:
                print(f"    [gh fail] {q[:50]} {type(e).__name__}")
                return None
            time.sleep(2 ** i * 3)
    return None


README_SYSTEM = """You decide whether a GitHub repository is a **life-science or pharmaceutical**
paper-companion code repository, for building a scientific coding benchmark.

Answer YES only if BOTH hold:
  1. It is research code accompanying a paper/preprint (or a research tool), AND
  2. Its primary subject is life science / medicine / pharmacology:
     biology, genomics/omics, bioinformatics, cell/molecular biology, neuroscience,
     immunology, microbiology, virology, ecology/evolution, agriculture-biology,
     medicine/clinical/epidemiology, pharmacology/drug discovery/toxicology.

Answer NO for anything else, including: general NLP/LLM/vision/speech/recommenders,
generic ML infrastructure, physics/astronomy/acoustics/materials, chemistry not aimed at
drugs, finance/economics/social science, civil/mechanical engineering, graphics, robotics,
software tooling. Judge SUBJECT MATTER, not methods.

Also answer NO if the repo mainly processes images/video (microscopy, radiology,
satellite) — we only want text/tabular/numeric data.

Output exactly one JSON object, no markdown:
{"verdict":"YES"|"NO","discipline":"<specific subfield, e.g. 'genomics'>","reason":"<=15 words"}"""


def judge_readme(name, desc, readme):
    try:
        user = sanitize_and_assert(
            f"Repository: {name}\nDescription: {desc or '(none)'}\n\n"
            f"README (truncated):\n{(readme or '(no README)')[:8000]}",
            tag="bio_search:user")
        system = sanitize_and_assert(README_SYSTEM, tag="bio_search:system")
    except Exception as e:  # noqa: BLE001
        return None, "", f"egress rejected: {type(e).__name__}"
    body = {"model": LLM_MODEL,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": 200, "temperature": 0.0, "stream": False,
            **LLM_THINKING_OFF}
    for i in range(3):
        try:
            out = _llm_post(body)
            txt = (out["choices"][0]["message"].get("content") or "").strip()
            txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt)
            try:
                obj = json.loads(txt)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", txt, re.S)
                if not m:
                    raise
                obj = json.loads(m.group(0))
            return (str(obj.get("verdict", "")).upper() == "YES",
                    str(obj.get("discipline", ""))[:40], str(obj.get("reason", ""))[:100])
        except Exception:  # noqa: BLE001
            if i == 2:
                # Return None for rate limits or parse failures, not a negative verdict that would silently discard a repository.
                return None, "", "llm fail"
            time.sleep(3 * (i + 1))
    return None, "", "llm fail"


def step_search(args):
    if not GH_TOKEN:
        print("  ✗ 需要 GITHUB_TOKEN 环境变量（GitHub GraphQL 必需）")
        sys.exit(2)
    kws = [l.strip() for l in open(KW_OUT, encoding="utf-8") if l.strip()]
    if args.max_keywords:
        kws = kws[:args.max_keywords]
    # A single-point star interval must produce that interval alone, not an empty range followed by its complement.
    
    
    if args.min_stars == args.max_stars:
        buckets = [f"{args.min_stars}..{args.max_stars}"]
    else:
        buckets = [f"{s}..{s+args.star_step-1}" for s in
                   range(args.min_stars, args.max_stars, args.star_step)] + [f">={args.max_stars+1}"]

    # Probe each keyword's total count first. Use one range when it fits per_combo, and split only larger queries into star buckets to save quota without reducing recall.


    if args.adaptive_buckets:
        print(f"  [按需切桶] 先探 {len(kws)} 个词的全段命中数（成本 {len(kws)} 点配额）…")
        full_bucket = f"{args.min_stars}..{args.max_stars}"

        def probe(kw):
            r = gh_search(f'{kw} in:readme (paper OR cite OR citation OR arxiv OR doi '
                          f'in:readme){_LANG_Q} stars:{full_bucket}')
            # On probe failure, conservatively split into buckets rather than losing recall.
            return kw, (10 ** 9 if r is None else len(r.get("nodes") or []) if not
                        (r.get("pageInfo") or {}).get("hasNextPage") else 10 ** 9)

        with cf.ThreadPoolExecutor(args.paging_workers) as ex:
            probed = list(ex.map(probe, kws))
        small = [k for k, n in probed if n <= args.per_combo]
        big = [k for k, n in probed if n > args.per_combo]
        combos = [(k, full_bucket) for k in small] + [(k, b) for k in big for b in buckets]
        print(f"    小词 {len(small)} 个（全段一次取完）+ 大词 {len(big)} 个 × {len(buckets)} 桶"
              f" = {len(combos)} 组合（固定切桶需 {len(kws)*len(buckets)}）")
    else:
        combos = [(k, b) for k in kws for b in buckets]
    print(f"  组合 {len(combos)}（关键词 {len(kws)} × star桶 {len(buckets)} 上限"
          f" {len(kws)*len(buckets)}）")
    # Skip repositories already present in the current or prior pools.
    have = set()
    for _d in [OLD_REPOS_DIR, NEW_REPOS_DIR] + EXTRA_HAVE:
        if os.path.isdir(_d):
            have |= {d.lower() for d in os.listdir(_d)}
    # Deduplicate CSV short names at the same granularity as clone directory names, and track full URLs separately. Comparing short names to owner/repository strings never matches.


    #     psims(mobiusklein vs RDCEP)、soundscape_IR(schonkopf vs meil-brcas-org)。
    # Short-name deduplication prevents clone-directory collisions; URL deduplication catches repeated hits across queries.
    
    seen_short, seen_url = set(), set()
    # Read both current and historical CSV outputs for deduplication.
    for _p in [REPO_CSV] + EXTRA_CSV:
        if not os.path.exists(_p):
            continue
        with open(_p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                seen_short.add((r.get("name") or "").lower())
                seen_url.add((r.get("url") or "").lower().rstrip("/"))
    print(f"  现池已有 {len(have)} 仓库、本脚本已收 {len(seen_short)}")

    new_f = open(REPO_CSV, "a", encoding="utf-8", newline="")
    w = csv.DictWriter(new_f, fieldnames=["discipline", "name", "url", "description",
                                          "stars", "reason", "source"])
    if os.path.getsize(REPO_CSV) == 0 if os.path.exists(REPO_CSV) else True:
        w.writeheader()
    cnt = Counter()

    def one_combo(item):
        kw, bk = item
        q = (f'{kw} in:readme (paper OR cite OR citation OR arxiv OR doi in:readme)'
             f'{_LANG_Q} stars:{bk}')
        after, got = None, 0
        out = []
        while got < args.per_combo:
            res = gh_search(q, after)
            if not res:
                break
            for nd in res.get("nodes") or []:
                if not nd:
                    continue
                full = nd["nameWithOwner"]
                short = full.split("/")[-1]
                url = (nd["url"] or "").lower().rstrip("/")
                with _lock:
                    # Short names match clone-directory granularity; URLs catch duplicate query hits.
                    if short.lower() in have or short.lower() in seen_short \
                            or url in seen_url:
                        cnt["skip_已有"] += 1
                        continue
                    seen_short.add(short.lower())
                    seen_url.add(url)
                rd = ((nd.get("object") or {}) or {}).get("text") or ""
                out.append((full, short, nd["url"], nd.get("description") or "",
                            nd.get("stargazerCount") or 0, rd))
                got += 1
                if got >= args.per_combo:
                    break
            pi = res.get("pageInfo") or {}
            if not pi.get("hasNextPage"):
                break
            after = pi.get("endCursor")
        return out

    # Preserve the adaptive bucket selection computed above.
    
    n = 0
    # Accumulate candidates into full LLM-worker batches. Processing each sparse query separately leaves the worker pool underutilized; resume deduplication recovers interrupted batches.


    buf: list = []

    def flush_batch(lex):
        """Classify buffered candidates and write CSV rows using the caller's long-lived executor."""
        if not buf:
            return
        verdicts = list(lex.map(lambda c: judge_readme(c[0], c[3], c[5]), buf))
        for (full, short, url, desc, stars, rd), (v, disc, why) in zip(buf, verdicts):
            if v is None:
                cnt["未判定(限流)"] += 1
                continue
            if not v:
                cnt["NO_非生命科学"] += 1
                continue
            cnt["YES"] += 1
            w.writerow({"discipline": disc, "name": short, "url": url,
                        "description": desc[:300], "stars": stars,
                        "reason": why, "source": "bio_search"})
        new_f.flush()
        buf.clear()

    # Reuse one executor throughout the search. Repeated thread-pool teardown destroys thread-local TLS sessions while other workers are active and has caused native TLS crashes.


    lex = cf.ThreadPoolExecutor(args.llm_workers)

    with cf.ThreadPoolExecutor(args.paging_workers) as pex:
        for cand in pex.map(one_combo, combos):
            n += 1
            buf.extend(cand)
            if len(buf) >= args.llm_workers:
                flush_batch(lex)
            if n % 200 == 0:
                print(f"    [{n}/{len(combos)}] YES={cnt['YES']} NO={cnt['NO_非生命科学']} "
                      f"跳过已有={cnt['skip_已有']} 未判定={cnt['未判定(限流)']}"
                      f" 待判={len(buf)}", flush=True)
        flush_batch(lex)   # Classify the final partial batch.
    lex.shutdown(wait=True)
    new_f.close()
    print(f"\n  === 搜索完成 ===")
    for k, v in cnt.most_common():
        print(f"    {k:16s} {v}")
    # Report quota waits explicitly so long pauses are diagnosable.
    if _quota["waits"]:
        print(f"    {'配额等待':16s} {_quota['waits']} 次，累计 {_quota['slept']/60:.0f}min")
    print(f"    {'剩余配额':16s} {_quota['remaining']}")
    print(f"    {'配额等待':16s} {_quota['waits']} 次，累计 {_quota['slept']/60:.0f}min"
          f"（8 线程求和，非墙钟）")
    print(f"    {'闸门复核':16s} {_quota['rechecks']} 次，其中假警报 "
          f"{_quota['false_alarms']} 次（复核挡掉的白等）")
    print(f"  → {REPO_CSV}")


# Clone repositories and extract files.
VERIFY_SYSTEM = """You judge whether a Python file is a COMPLETE scientific analysis task.

Answer Yes only if ALL THREE hold:
 1. Functionality: part of a scientific workflow (feature engineering, ML/DL, computational
    analysis, visualization, model training, numerical computation, statistics, or
    domain-specific analysis/simulation).
 2. Input: the program takes AT LEAST ONE DATASET as input (loaded from a local file or a
    built-in loader). A program that takes no data is NOT a scientific task.
 3. Output: it produces numeric or visual results that can be further evaluated.

Answer No for: pure model definitions, pure train/test scaffolding, pure preprocessing,
utility functions or class definitions only, or files longer than 5000 lines.

Output exactly one JSON object: {"verdict":"Yes"|"No","reason":"<=12 words"}"""

EXCLUDE_DIR = re.compile(r"(^|/)(utils?|tests?|docs?|settings?|scripts|build)(/|$)", re.I)
EXCLUDE_FILE = re.compile(r"(setup|__init__|utils?|requirements|config|settings|manage)\.py$", re.I)


def judge_file(code):
    try:
        user = sanitize_and_assert(f"Python file:\n```python\n{code[:14000]}\n```",
                                   tag="bio_verify:user")
        system = sanitize_and_assert(VERIFY_SYSTEM, tag="bio_verify:system")
    except Exception:  # noqa: BLE001
        return None
    body = {"model": LLM_MODEL,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": 120, "temperature": 0.0, "stream": False,
            **LLM_THINKING_OFF}
    for i in range(3):
        try:
            out = _llm_post(body)
            txt = (out["choices"][0]["message"].get("content") or "").strip()
            txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt)
            m = re.search(r"\{.*\}", txt, re.S)
            obj = json.loads(m.group(0) if m else txt)
            return str(obj.get("verdict", "")).lower().startswith("y")
        except Exception:  # noqa: BLE001
            if i == 2:
                return None
            time.sleep(3 * (i + 1))
    return None


def step_crawl(args):
    if not os.path.exists(REPO_CSV):
        print(f"  ✗ 先跑 search（缺 {REPO_CSV}）"); sys.exit(2)
    rows = list(csv.DictReader(open(REPO_CSV, encoding="utf-8")))
    if args.max_repos:
        rows = rows[:args.max_repos]
    os.makedirs(NEW_REPOS_DIR, exist_ok=True)
    done = set()
    if os.path.exists(BIO_JSONL):
        with open(BIO_JSONL, encoding="utf-8", errors="ignore") as f:
            for ln in f:
                try:
                    done.add(json.loads(ln)["file_path"].split("/")[0])
                except Exception:
                    pass
    todo = [r for r in rows if r["name"] not in done]
    print(f"  csv {len(rows)} 仓库；已抽过 {len(done)}；本次 {len(todo)}")

    cnt = Counter()
    fo = open(BIO_JSONL, "a", encoding="utf-8")

    def do_repo(r):
        name, url, disc = r["name"], r["url"], r.get("discipline", "")
        dst = os.path.join(NEW_REPOS_DIR, name)
        if not os.path.isdir(dst):
            try:
                subprocess.run(["git", "clone", "--depth", "1", "-q", url, dst],
                               timeout=args.clone_timeout, check=True,
                               capture_output=True)
            except Exception:  # noqa: BLE001
                with _lock:
                    cnt["clone失败"] += 1
                return
        out = []
        for root, dirs, files in os.walk(dst):
            dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "node_modules")]
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                rel = os.path.relpath(os.path.join(root, fn), NEW_REPOS_DIR).replace("\\", "/")
                if EXCLUDE_DIR.search(rel) or EXCLUDE_FILE.search(fn):
                    with _lock:
                        cnt["规则预过滤"] += 1
                    continue
                try:
                    code = open(os.path.join(root, fn), encoding="utf-8",
                                errors="ignore").read()
                except OSError:
                    continue
                if not code.strip() or code.count("\n") > 5000:
                    continue
                out.append((rel, code))
        # Classify files concurrently within each repository. Total concurrency is crawl_workers times file_workers; reuse the executor throughout crawling to avoid repeated TLS-session teardown.


        batch = out[:args.max_files_per_repo]
        verdicts = list(fex.map(lambda rc: judge_file(rc[1]), batch)) if batch else []
        for (rel, code), v in zip(batch, verdicts):
            with _lock:
                if v is None:
                    cnt["未判定"] += 1
                elif v:
                    cnt["Yes"] += 1
                    fo.write(json.dumps({
                        "discipline": disc,          # Persist the inferred discipline for downstream filtering.
                        "file_url": f"{url}/{rel.split('/',1)[-1]}",
                        "file_path": rel, "code": code, "response": "",
                        "is_scientific_coding_task": "Yes"}, ensure_ascii=False) + "\n")
                    fo.flush()
                else:
                    cnt["No"] += 1

    # Reuse a shared executor sized as crawl_workers times file_workers; a smaller pool would silently serialize concurrent repository workloads.


    fex = cf.ThreadPoolExecutor(args.crawl_workers * args.file_workers)
    with cf.ThreadPoolExecutor(args.crawl_workers) as ex:
        for i, _ in enumerate(ex.map(do_repo, todo), 1):
            if i % 20 == 0:
                print(f"    [{i}/{len(todo)}] Yes={cnt['Yes']} No={cnt['No']} "
                      f"预过滤={cnt['规则预过滤']} clone失败={cnt['clone失败']}", flush=True)
    fex.shutdown(wait=True)
    fo.close()
    print("\n  === crawl 完成 ===")
    for k, v in cnt.most_common():
        print(f"    {k:12s} {v}")
    print(f"  → {BIO_JSONL}")


# Merge pools.
def step_merge(args):
    """Merge old and new pools by file_path; optionally filter old entries using repository-domain labels."""
    keep_repos = None
    lab = os.path.join(WS, "repo_domain_labels.jsonl")
    if args.filter_old and os.path.exists(lab):
        keep_repos = set()
        c = Counter()
        with open(lab, encoding="utf-8") as f:
            for ln in f:
                try:
                    o = json.loads(ln)
                except Exception:
                    continue
                c[o["verdict"]] += 1
                # Keep UNCLEAR verdicts conservatively: missing README evidence does not imply an out-of-domain repository.
                if o["verdict"] in ("KEEP", "UNCLEAR"):
                    keep_repos.add(o["repo"])
        print(f"  旧池学科标签: KEEP={c['KEEP']} UNCLEAR={c['UNCLEAR']} DROP={c['DROP']}"
              f" → 保留 {len(keep_repos)} 个仓库（KEEP+UNCLEAR）")

    seen = set()
    n_old = n_new = n_dup = n_filtered = 0
    with open(MERGED_JSONL, "w", encoding="utf-8") as fo:
        for src, tag in ((BIO_JSONL, "new"), (OLD_JSONL, "old")):
            if not os.path.exists(src):
                print(f"  [warn] 缺 {src}，跳过")
                continue
            with open(src, encoding="utf-8", errors="ignore") as f:
                for ln in f:
                    try:
                        o = json.loads(ln)
                    except Exception:
                        continue
                    fp = o.get("file_path", "")
                    if not fp:
                        continue
                    if tag == "old" and keep_repos is not None:
                        if fp.split("/")[0] not in keep_repos:
                            n_filtered += 1
                            continue
                    if fp in seen:
                        n_dup += 1
                        continue
                    seen.add(fp)
                    fo.write(json.dumps(o, ensure_ascii=False) + "\n")
                    if tag == "new":
                        n_new += 1
                    else:
                        n_old += 1
    print(f"\n  合并: 新增(bio搜索) {n_new} + 旧池保留 {n_old} = **{n_new+n_old}** 条")
    print(f"        去重丢弃 {n_dup}；旧池按学科过滤掉 {n_filtered}")
    print(f"  → {MERGED_JSONL}")
    print(f"\n  生产时用：export STEM_STAGE1_JSONL={MERGED_JSONL}")
    print(f"           export STEM_STAGE1_REPOS={OLD_REPOS_DIR}"
          f"   # New repositories are in {NEW_REPOS_DIR}; merge or symlink them")


def main():
    ap = argparse.ArgumentParser(description="生物/药学定向仓库搜索 + 合并")
    ap.add_argument("step", choices=["keywords", "search", "crawl", "merge", "all"])
    ap.add_argument("--max-keywords", type=int, default=0, help="只用前 N 个关键词（调试）")
    ap.add_argument("--min-stars", type=int, default=10)
    ap.add_argument("--max-stars", type=int, default=1000)
    # Smaller star buckets reduce truncation by per_combo, at the cost of more queries and quota consumption.


    ap.add_argument("--star-step", type=int, default=10)
    ap.add_argument("--per-combo", type=int, default=50, help="每个(关键词×星桶)最多取几个仓库")
    # Adaptive buckets avoid splitting low-yield keywords unnecessarily; disable only when fixed buckets are needed.
    
    ap.add_argument("--adaptive-buckets", action="store_true", default=True)
    ap.add_argument("--no-adaptive-buckets", dest="adaptive_buckets",
                    action="store_false")
    ap.add_argument("--paging-workers", type=int, default=8, help="GitHub 翻页并发（限流敏感，勿调大）")
    # Also used as the candidate batch size for README classification.
    
    ap.add_argument("--llm-workers", type=int, default=256)
    # Repository-level concurrency matters because many repositories have very few candidate files. Balance it against clone I/O rather than gateway peak throughput alone.


    ap.add_argument("--crawl-workers", type=int, default=32)
    # Per-repository file concurrency multiplies crawl concurrency. Leave headroom for shared gateway load and occasional timeouts.


    ap.add_argument("--file-workers", type=int, default=16,
                    help="每个仓库内 judge_file 的并发（总并发 = crawl_workers × 本值）")
    ap.add_argument("--max-repos", type=int, default=0)
    ap.add_argument("--max-files-per-repo", type=int, default=60)
    ap.add_argument("--clone-timeout", type=int, default=300)
    ap.add_argument("--filter-old", action="store_true",
                    help="merge 时按 repo_domain_labels.jsonl 过滤旧池（只留 KEEP+UNCLEAR）")
    args = ap.parse_args()

    steps = ["keywords", "search", "crawl", "merge"] if args.step == "all" else [args.step]
    for s in steps:
        print(f"\n{'='*70}\n[{s}]\n{'='*70}")
        {"keywords": step_keywords, "search": step_search,
         "crawl": step_crawl, "merge": step_merge}[s](args)


if __name__ == "__main__":
    main()

# USAGE ─────────────────────────────────────────────────────────────────────
# See the mining usage documentation.
