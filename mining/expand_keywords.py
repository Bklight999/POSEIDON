#!/usr/bin/env python3
"""Expand the keyword pool with tool, format, and database names across scientific subfields. Validate each term against GitHub repository counts to remove invented or overly generic terms. Cache generation results. Usage: python mining/expand_keywords.py [--dry-run]."""
import concurrent.futures as cf
import importlib.util
import json
import os
import tempfile
import re
import sys
import threading

import requests

BASE = os.getenv("STEM_PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WS = os.path.join(BASE, "workspace")
OUT = os.path.join(WS, "keywords_tools_llm.txt")
MIN_HITS = 30          # Drop terms with too few hits to justify a search combination.
# An upper bound is equally important: generic infrastructure names create large irrelevant candidate pools.

#   NLP CNN MongoDB PostgreSQL GitHub Actions …
# Keep terms with 30 to 5000 repository hits to balance specificity and recall; broader terms waste classification calls and clone storage.


MAX_HITS = 5000
N_ROUNDS = 12          # Rotate subfields to avoid repeatedly generating the same popular tools.

_dry = "--dry-run" in sys.argv

# Reuse the mining client's gateway configuration and disabled-reasoning setting.
_spec = importlib.util.spec_from_file_location(
    "_brs", os.path.join(BASE, "mining/bio_repo_search.py"))
_brs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_brs)

# Read authentication from configuration rather than manually copying token fragments.

_spec2 = importlib.util.spec_from_file_location(
    "_lb", os.path.join(BASE, "mining/launch_local_mining.py"))
_lb = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_lb)
GH_TOKEN = _lb.GH_TOKEN

FIELDS = [
    "single-cell transcriptomics and spatial omics",
    "structural biology, protein folding and molecular dynamics",
    "cheminformatics, drug discovery and virtual screening",
    "genomics: assembly, alignment, variant calling",
    "medical imaging: MRI, CT, histopathology",
    "neuroscience: EEG, MEG, connectomics",
    "epidemiology, clinical trials and biostatistics",
    "mass spectrometry, proteomics and metabolomics",
    "phylogenetics and evolutionary biology",
    "microbiome and metagenomics",
    "systems biology, metabolic and pathway modelling",
    "electronic health records and clinical NLP",
]

PROMPT = """List GitHub search keywords for finding scientific Python repositories in: {field}

Give me 40 concrete SOFTWARE / TOOL / FILE-FORMAT / DATABASE names that such repos
actually name in their README — for example: scanpy, AnnData, GROMACS, RDKit,
nextflow, Bioconductor, .mzML, UK Biobank, GTEx.

Rules:
- Real, existing names only. No invented tools.
- No generic discipline names ("bioinformatics", "computational biology").
- No single common English words that mean something else ("cell", "match").
- 2..40 characters each.
Return ONLY JSON: {{"keywords": ["...", "..."]}}"""


def ask(field: str) -> list:
    body = {"model": _brs.LLM_MODEL,
            "messages": [{"role": "user",
                          "content": PROMPT.format(field=field)}],
            # Allow enough output tokens for complete JSON lists; long tool names otherwise cause truncation and repeated empty results.


            "temperature": 0.7, "max_tokens": 3000,
            # Explicitly disable reasoning so it cannot consume the JSON output budget.
            
            "thinking": {"type": "disabled"}}
    for _ in range(3):
        try:
            d = _brs._llm_post(body, timeout=180)
            txt = d["choices"][0]["message"]["content"]
            m = re.search(r'\{.*\}', txt, re.DOTALL)
            if not m:
                continue
            kws = json.loads(m.group(0)).get("keywords", [])
            return [str(k).strip() for k in kws if 2 <= len(str(k).strip()) <= 40]
        except Exception:  # noqa: BLE001
            pass
    return []


_gh_tls = threading.local()


def hits(kw: str) -> int:
    """Count GitHub repositories matching a term without language or star constraints; return -1 on failure. Use thread-local requests sessions to isolate TLS connection pools."""
    s = getattr(_gh_tls, "sess", None)
    if s is None:
        s = requests.Session()
        s.mount("https://", requests.adapters.HTTPAdapter(
            pool_connections=2, pool_maxsize=2, max_retries=0))
        _gh_tls.sess = s
    payload = {"query": 'query($q:String!){search(query:$q,type:REPOSITORY,'
                        'first:1){repositoryCount}}',
               "variables": {"q": f'"{kw}"'}}
    for _ in range(3):
        try:
            r = s.post("https://api.github.com/graphql", json=payload,
                       headers={"Authorization": f"bearer {GH_TOKEN}"},
                       timeout=40)
            if r.status_code != 200:
                continue
            return int(r.json()["data"]["search"]["repositoryCount"])
        except Exception:  # noqa: BLE001
            pass
    return -1


def existing() -> set:
    """Load the existing keyword lists to avoid duplicate search combinations."""
    out = set()
    for f in ("keywords_bio_pharma.txt", "keywords_pharma_manual.txt",
              "keywords_species_disease.txt"):
        p = os.path.join(WS, f)
        if not os.path.exists(p):
            continue
        for ln in open(p, encoding="utf-8", errors="ignore"):
            t = ln.split("#", 1)[0].strip()
            if t:
                out.add(t.lower())
    return out


def main():
    have = existing()
    print(f"① 已有关键词 {len(have)}", flush=True)

    # Cache generated keywords to avoid repeating costly, nondeterministic expansion after a later failure.
    
    cache = os.path.join(tempfile.gettempdir(), "expand_kw_cand.json")
    cand = {}
    if os.path.exists(cache) and "--refresh" not in sys.argv:
        try:
            cand = json.load(open(cache, encoding="utf-8"))
            print(f"② LLM 扩词：用缓存 {cache} → {len(cand)} 候选"
                  f"（--refresh 可强制重扩）", flush=True)
        except Exception:  # noqa: BLE001
            cand = {}
    if not cand:
        print(f"② LLM 扩词（{len(FIELDS)} 个子领域并行）", flush=True)
        with cf.ThreadPoolExecutor(len(FIELDS)) as ex:
            for field, kws in zip(FIELDS, ex.map(ask, FIELDS)):
                new = [k for k in kws if k.lower() not in have
                       and k.lower() not in cand]
                for k in new:
                    cand[k.lower()] = k
                print(f"  {field[:44]:46} 得 {len(kws):3} 新 {len(new):3}",
                      flush=True)
        print(f"  候选合计 {len(cand)}（已扣重复）", flush=True)
        try:
            json.dump(cand, open(cache, "w", encoding="utf-8"),
                      ensure_ascii=False)
        except OSError:
            pass

    if _dry:
        print("\n[DRY-RUN] 未查 GitHub、未写文件")
        for k in sorted(cand.values())[:40]:
            print(f"    {k}")
        return 0

    print(f"③ GitHub 实测命中数（筛 {MIN_HITS}..{MAX_HITS}）", flush=True)
    keys = sorted(cand.values())
    kept, thin, fat, failed = [], [], [], []
    # Use eight query workers to balance proxy throughput and GitHub quota.
    with cf.ThreadPoolExecutor(8) as ex:
        for kw, n in zip(keys, ex.map(hits, keys)):
            if n < 0:
                failed.append(kw)
            elif n < MIN_HITS:
                thin.append((kw, n))
            elif n > MAX_HITS:
                fat.append((kw, n))     # Exclude overly generic terms.
            else:
                kept.append((kw, n))
    kept.sort(key=lambda x: -x[1])
    fat.sort(key=lambda x: -x[1])
    print(f"  留 {len(kept)}  太少(<{MIN_HITS}) {len(thin)}  "
          f"太泛(>{MAX_HITS}) {len(fat)}  查询失败 {len(failed)}", flush=True)
    if failed:
        print(f"  失败词（未收录）: {failed[:8]}", flush=True)
    if fat:
        print(f"  剔掉的通用词示例: {', '.join(k for k, _ in fat[:10])}", flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# LLM-expanded tools, software, formats, and databases; inline counts are measured repositoryCount values\n")
        f.write(f"# Generated by expand_keywords.py; accepted hit range: {MIN_HITS}..{MAX_HITS}\n")
        f.write("# The upper bound excludes generic terms; see MAX_HITS in the script\n")
        for kw, n in kept:
            f.write(f"{kw}  # {n}\n")
    print(f"  → {OUT}", flush=True)
    print("\n  命中最多的 20 个（应全是真工具名，若见到通用词说明上限还要压）:",
          flush=True)
    for kw, n in kept[:20]:
        print(f"    {n:7}  {kw}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
