"""Score tasks, repair prompts below threshold without changing gold answers, and rescore. Store passing tasks with repair history in task.md. Usage: python qc/run_qc_split.py [--demo N]."""
import os
import os, re, json, threading, concurrent.futures, urllib.request, time, sys

PKG   = "tasks_out/packaged_final/packaged_v3"
OUT4  = "tasks_out/packaged_final/packaged_score4plus"
OUT3  = "tasks_out/packaged_final/packaged_score3"
WORKERS = 64

# QC scoring model configuration.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stemgen import llm_gen as _client
QC_URL = os.getenv("STEM_QC_URL", "https://api.example.com/v1/chat/completions")
QC_MODEL = os.getenv("STEM_QC_MODEL", "your-qc-model")
QC_AUTH = "Bearer " + os.getenv("STEM_QC_KEY", "")
FLASH_URL = _client.GW_URL
FLASH_MODEL = _client.GW_MODEL_FLASH
FLASH_KEY = os.getenv("STEM_GW_KEY", _client.GW_APP_KEY)

QC_SEM    = threading.Semaphore(WORKERS)
FLASH_SEM = threading.Semaphore(WORKERS)

SENSITIVE_RE = re.compile(r"(?:sk-|ghp_)[A-Za-z0-9_-]{20,}", re.I)

from stemgen.safe_egress import sanitize_and_assert

def _sanitize(x):
    return sanitize_and_assert(x, tag="run_qc_split:field")

# Extend the package QC prompt with prompt-code alignment checks.
_src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stemgen", "package_tasks.py"), encoding="utf-8").read()
_m = re.search(r"_QC_SYSTEM_PROMPT\s*=\s*r?\"\"\"(.*?)\"\"\"", _src, re.DOTALL)
_base_qc = _m.group(1) if _m else ""
# Add the prompt-code alignment dimension.
_CODE_ALIGN_DIM = """
C. Prompt-Code Alignment
- Does the Prompt correctly describe what the Gold Program actually computes?
- Does the Prompt mention files, columns, or variables that do not appear in the Gold Program?
- Does the Gold Program compute something different from what the Prompt requests?
- Are the file paths in the Prompt consistent with what the Gold Program reads (all data files should be under data/)?
"""
# Insert it between the existing scoring dimensions.
QC_SYSTEM = _base_qc.replace(
    "C. Scientific and Logical Validity",
    _CODE_ALIGN_DIM + "\nD. Scientific and Logical Validity"
).replace(
    "category: identifiability|prompt_gold|scientific_validity",
    "category: identifiability|prompt_gold|prompt_code|scientific_validity"
)

FLASH_FIX_SYSTEM = """You are an expert scientific task editor. You will receive:
1. A scientific analysis task PROMPT (shown to an AI agent)
2. The Gold answer (computed from real code — DO NOT change this)
3. A QC evaluation with score < 4 and issues listing what is wrong

Your job: rewrite the PROMPT to fix the identified issues so that the task becomes unambiguous and uniquely gradable.

Rules (STRICT):
- Fix ONLY the prompt text. Never modify the Gold answer.
- Fix ambiguities: replace vague terms ("appropriate", "standard", "significant") with precise operational definitions.
- Fix missing conventions: add threshold values, boundary conditions, rounding rules, tie-breaking rules if they could change the answer.
- Fix path references: all data files must be referenced with the data/ prefix (e.g., data/file.csv).
- Keep the scientific goal identical — do not change what is being asked.
- Keep the prompt concise — do not add unnecessary explanation.
- Return ONLY the rewritten prompt text, nothing else. No preamble, no explanation."""


def _call(url, auth, model, messages, max_tokens, sem, extra_body=None):
    safe_messages = []
    for i, message in enumerate(messages):
        item = dict(message)
        item["content"] = sanitize_and_assert(item.get("content", ""),
                                              tag=f"run_qc_split:message:{i}")
        safe_messages.append(item)
    body = {"model": model, "messages": safe_messages,
            "max_completion_tokens": max_tokens}
    if extra_body:
        body.update(extra_body)
    data = sanitize_and_assert(body, tag="run_qc_split:payload").encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", auth)
    with sem:
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    out = json.loads(resp.read())
                return out["choices"][0]["message"].get("content", "")
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))


def call_gpt_qc(prompt_text, gold, gold_program=""):
    """Score a task and return a QC result dictionary."""
    p = _sanitize(prompt_text or "")
    g_str = json.dumps(gold, ensure_ascii=False) if not isinstance(gold, str) else gold
    g = _sanitize(g_str)
    gp = _sanitize(gold_program or "")
    if SENSITIVE_RE.search(p) or SENSITIVE_RE.search(g) or SENSITIVE_RE.search(gp):
        return {"score": -1, "skipped": True, "summary": "sensitive"}
    gold_prog_section = f"\n\nGold Program (the code that produces the Gold answer — use this to verify prompt-code alignment):\n```python\n{gp[:3000]}\n```" if gp else ""
    user = f"TASK TO EVALUATE:\n\nPrompt:\n{p}\n\nGold:\n{g}{gold_prog_section}"
    try:
        content = _call(QC_URL, QC_AUTH, QC_MODEL,
                        [{"role":"system","content":QC_SYSTEM},
                         {"role":"user","content":user}],
                        1200, QC_SEM,
                        extra_body={"reasoning_effort": "none"})
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
        return json.loads(content)
    except Exception as e:
        return {"score": -1, "skipped": True, "summary": f"error:{str(e)[:60]}"}


def call_flash_fix(prompt_text, gold, qc_result, gold_program=""):
    """Rewrite a prompt from QC feedback and return the repaired text."""
    p = _sanitize(prompt_text or "")
    g_str = json.dumps(gold, ensure_ascii=False) if not isinstance(gold, str) else gold
    g = _sanitize(g_str)
    gp = _sanitize(gold_program or "")
    issues_text = json.dumps(qc_result.get("issues", []), ensure_ascii=False, indent=2)
    summary = qc_result.get("summary", "")
    gold_prog_section = f"\nGold Program (the verified code — use this to understand exactly what the prompt should describe):\n```python\n{gp[:3000]}\n```" if gp else ""
    user = f"""ORIGINAL PROMPT:
{p}

GOLD ANSWER (do not change):
{g}{gold_prog_section}

QC EVALUATION (score={qc_result.get('score','?')}):
Summary: {summary}
Issues:
{issues_text}

Rewrite the prompt to fix all issues listed above. Return ONLY the rewritten prompt."""
    try:
        fixed = _call(FLASH_URL, f"Bearer {FLASH_KEY}", FLASH_MODEL,
                      [{"role":"system","content":FLASH_FIX_SYSTEM},
                       {"role":"user","content":user}],
                      2000, FLASH_SEM)
        return fixed.strip()
    except Exception as e:
        return None


def process_task(tid, demo=False):
    """Return task ID, final score, and details; demo details include original prompt, initial QC, repaired prompt, and final QC."""
    tdir = os.path.join(PKG, tid)
    md_path = os.path.join(tdir, "task.md")
    if not os.path.exists(md_path):
        return tid, None, {}

    content = open(md_path, encoding='utf-8', errors='ignore').read()

    # Skip already scored tasks outside demo mode.
    if not demo and re.search(r'\*\*final_score\*\*:\s*\d+', content):
        m = re.search(r'\*\*final_score\*\*:\s*(\d+)', content)
        return tid, int(m.group(1)), {}

    # Extract prompt and gold answer.
    mp = re.search(r'## Prompt.*?\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
    original_prompt = mp.group(1).strip() if mp else ""
    mg = re.search(r'## Gold answer\n```\n(.*?)\n```', content, re.DOTALL)
    gold_raw = mg.group(1).strip() if mg else ""
    try:
        gold = json.loads(gold_raw)
    except Exception:
        gold = gold_raw

    # Read the reference program for prompt-code alignment checks.
    gp_path = os.path.join(tdir, "gold_program.py")
    gold_program = open(gp_path, encoding='utf-8', errors='ignore').read() if os.path.exists(gp_path) else ""

    # Step 1: score the task and collect feedback.
    qc1 = call_gpt_qc(original_prompt, gold, gold_program)
    score1 = qc1.get("score", -1)

    if qc1.get("skipped"):
        return tid, None, {"qc1": qc1}

    # For a passing score, skip repair and confirm with another QC pass.
    if score1 >= 4:
        qc2 = call_gpt_qc(original_prompt, gold, gold_program)
        final_score = qc2.get("score", score1) if not qc2.get("skipped") else score1
        _write_qc(md_path, content, score1, qc1, None, qc2, final_score)
        return tid, final_score, {"qc1": qc1, "fixed_prompt": None, "qc2": qc2}

    # Step 2: repair prompts with scores below 4.
    fixed_prompt = call_flash_fix(original_prompt, gold, qc1, gold_program)
    if not fixed_prompt or fixed_prompt == original_prompt:
        qc2 = call_gpt_qc(original_prompt, gold, gold_program)
        final_score = qc2.get("score", score1) if not qc2.get("skipped") else score1
        _write_qc(md_path, content, score1, qc1, None, qc2, final_score)
        return tid, final_score, {"qc1": qc1, "fixed_prompt": None, "qc2": qc2}

    # Step 3: rescore the repaired prompt.
    qc2 = call_gpt_qc(fixed_prompt, gold, gold_program)
    score2 = qc2.get("score", -1) if not qc2.get("skipped") else score1
    final_score = score2

    # Update task.md only when the repair improves the score.
    if score2 > score1 and score2 >= 4 and not SENSITIVE_RE.search(fixed_prompt):
        # Replace the prompt section with the repaired text.
        new_content = re.sub(
            r'(## Prompt.*?\n).*?(?=\n##|\Z)',
            lambda m2: m2.group(1) + fixed_prompt,
            content, count=1, flags=re.DOTALL
        )
        open(md_path, 'w', encoding='utf-8').write(new_content)
        content = new_content

    _write_qc(md_path, content, score1, qc1, fixed_prompt, qc2, final_score)

    return tid, final_score, {
        "original_prompt": original_prompt,
        "qc1": qc1,
        "fixed_prompt": fixed_prompt,
        "qc2": qc2,
        "score1": score1,
        "final_score": final_score,
    }


def _write_qc(md_path, content, score1, qc1, fixed_prompt, qc2, final_score):
    """Append QC results to task.md if not already present."""
    if "## QC" in content:
        return
    summary1 = qc1.get("summary","")[:120]
    section = f"\n## QC\n- **score**: {score1}\n- **summary**: {summary1}\n"
    if fixed_prompt and qc2:
        score2 = qc2.get("score", "?")
        summary2 = qc2.get("summary","")[:120]
        section += f"- **fixed**: yes (flash repaired)\n- **score_after_fix**: {score2}\n- **summary_after_fix**: {summary2}\n"
    section += f"- **final_score**: {final_score}\n"
    try:
        open(md_path, 'a', encoding='utf-8').write(section)
    except OSError:
        pass


def main():
    demo_mode = "--demo" in sys.argv
    demo_n = int(sys.argv[sys.argv.index("--demo") + 1]) if demo_mode else 0

    tids = sorted(t for t in os.listdir(PKG) if t.startswith('stem_'))
    if demo_mode:
        tids = tids[:demo_n]
        print(f"=== DEMO 模式：只跑前 {demo_n} 道题 ===\n")
    else:
        print(f"共 {len(tids)} 道题，{WORKERS} 并发，GPT判分+Flash修复+GPT复判 ...")

    os.makedirs(OUT4, exist_ok=True)
    os.makedirs(OUT3, exist_ok=True)

    results = {}
    demo_cases = []  # Collect examples with improved scores.
    done = [0]
    lock = threading.Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(process_task, tid, demo_mode): tid for tid in tids}
        for f in concurrent.futures.as_completed(futs):
            tid, score, info = f.result()
            results[tid] = score
            with lock:
                done[0] += 1
                if not demo_mode and done[0] % 200 == 0:
                    print(f"  {done[0]}/{len(tids)} 完成")
                # Collect successful repair examples in demo mode.
                if info.get("fixed_prompt") and info.get("score1", 0) < 4 \
                        and info.get("final_score", 0) >= 4:
                    demo_cases.append((tid, info))

    # Partition tasks by the final result.
    c4, c3, cskip, c_rescued = 0, 0, 0, 0
    idx_path = f"{PKG}/tasks_with_id.jsonl"
    idx = [json.loads(l) for l in open(idx_path) if l.strip()] if os.path.exists(idx_path) else []
    idx_map = {t.get("id"): t for t in idx}

    rows4, rows3 = [], []
    for tid in sorted(results):
        score = results[tid]
        tdir_abs = os.path.abspath(os.path.join(PKG, tid))
        task = idx_map.get(tid, {"id": tid})
        if score is None:
            cskip += 1; continue
        if score >= 4:
            link = os.path.join(OUT4, tid)
            if not os.path.exists(link): os.symlink(tdir_abs, link)
            rows4.append(task); c4 += 1
        elif score >= 3:
            link = os.path.join(OUT3, tid)
            if not os.path.exists(link): os.symlink(tdir_abs, link)
            rows3.append(task); c3 += 1

    with open(f"{OUT4}/tasks_with_id.jsonl", 'w') as f:
        for r in rows4: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    with open(f"{OUT3}/tasks_with_id.jsonl", 'w') as f:
        for r in rows3: f.write(json.dumps(r, ensure_ascii=False)+"\n")

    print(f"\n=== 完成 ===")
    print(f"score>=4: {c4} 题 → {OUT4}")
    print(f"score=3:  {c3} 题 → {OUT3}")
    print(f"跳过/error: {cskip} 题")

    # Display repair examples.
    if demo_cases or demo_mode:
        print(f"\n{'='*60}")
        print(f"=== 修复成功样例（score 从 <4 提升到 >=4）===")
        for tid, info in (demo_cases or list(results.items()))[:3]:
            if not isinstance(info, dict): continue
            print(f"\n【{tid}】 score: {info.get('score1','?')} → {info.get('final_score','?')}")
            print(f"\n--- GPT 初判意见 ---")
            print(f"Score: {info['qc1'].get('score','?')}")
            print(f"Summary: {info['qc1'].get('summary','')}")
            for iss in info['qc1'].get('issues',[])[:3]:
                print(f"  [{iss.get('severity')}] {iss.get('category')}: {iss.get('explanation','')[:100]}")
            if info.get('fixed_prompt'):
                print(f"\n--- Flash 修复后的 Prompt ---")
                print(info['fixed_prompt'][:600])
                if len(info['fixed_prompt']) > 600: print("...(truncated)")
            if info.get('qc2'):
                print(f"\n--- GPT 复判 ---")
                print(f"Score: {info['qc2'].get('score','?')}")
                print(f"Summary: {info['qc2'].get('summary','')}")
            print()


if __name__ == "__main__":
    main()
