"""Shared LLM generation client. Sanitize and audit every outbound prompt, including requests to private gateways. Expose text generation, structured JSON generation with retries, and an execution-feedback repair loop. Configure endpoint and authentication for the deployment."""
from __future__ import annotations

import os
import json
import re
import time
import urllib.request

from .safe_egress import (SensitiveEgressError, audit, sanitize_and_assert,
                          sanitize_messages)


class SensitiveContentError(RuntimeError):
    """Outbound content could not be sanitized safely. Callers treat this as a failed request or rejected seed; no sensitive payload is sent."""

# Select the gateway protocol with STEM_GW_MODE.
#
# Gateway modes differ in authentication routing and thinking-field placement. Configure an endpoint matching the selected protocol; misplaced thinking parameters may be silently ignored.


#
# Shared gateway load causes large throughput variation; retain concurrency limits and scale them across machines rather than relying on peak benchmarks.


import os as _os_gate
import threading as _threading

_GW_MODE = os.getenv("STEM_GW_MODE", "openai").strip().lower()
GW_URL = os.getenv("STEM_GW_URL", "https://api.example.com/v1/chat/completions")
GW_MODEL = os.getenv("STEM_GW_MODEL", "your-model")
GW_MODEL_FLASH = (os.getenv("STEM_GW_MODEL_FLASH") or GW_MODEL)
GW_MODEL_PRO = (os.getenv("STEM_GW_MODEL_PRO") or GW_MODEL)
GW_APP_ID = os.getenv("STEM_GW_APP_ID", "")
GW_APP_KEY = os.getenv("STEM_GW_APP_KEY", "")

def _auth_header(model: str) -> str:
    return "Bearer " + os.getenv("STEM_GW_KEY", GW_APP_KEY)

def _apply_thinking(body: dict, thinking: bool) -> None:
    if _GW_MODE == "thinking":
        body["thinking"] = {"type": "enabled" if thinking else "disabled"}
    elif _GW_MODE == "extra_body":
        body["extra_body"] = {"thinking": {"type": "enabled" if thinking else "disabled"}}

# Limit concurrent requests by actual model name. If default and fast models are identical, they must share one semaphore and quota.


_PRO_LIMIT = int(_os_gate.getenv("STEM_GW_LIMIT_PRO", "56"))
_FLASH_LIMIT = int(_os_gate.getenv("STEM_GW_LIMIT_FLASH", "100"))
_GATE = {GW_MODEL_FLASH: _threading.Semaphore(_FLASH_LIMIT)}
if GW_MODEL_PRO != GW_MODEL_FLASH:
    _GATE[GW_MODEL_PRO] = _threading.Semaphore(_PRO_LIMIT)

# Accumulate outbound audit records for the orchestrator to flush.
import threading as _threading
_AUDIT_LOG: list[dict] = []
_AUDIT_LOCK = _threading.Lock()


def audit_log() -> list[dict]:
    return _AUDIT_LOG


def _post(messages: list[dict], max_tokens: int, temperature: float,
          timeout: int, thinking: bool | None = None,
          model: str | None = None) -> str:
    model = model or GW_MODEL
    # Defense in depth: even callers inside this module cannot bypass the final egress guard.
    messages = sanitize_messages(messages, tag="llm_gen._post")
    body = {"model": model, "messages": messages, "stream": False,
            "max_tokens": max_tokens, "temperature": temperature}
    # Set thinking through the gateway-specific helper. None follows STEM_GEN_THINKING; callers explicitly enable reasoning for planning and disable it for code generation or extraction to preserve output budgets.


    import os as _os
    if thinking is None:
        thinking = _os.getenv("STEM_GEN_THINKING", "1") == "1"
    _apply_thinking(body, thinking)
    data = json.dumps(body).encode()
    req = urllib.request.Request(GW_URL, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", _auth_header(model))
    # Use a model-specific semaphore for known models; custom models retain unrestricted behavior.
    
    _sem = _GATE.get(model)
    if _sem is not None:
        with _sem:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                out = json.loads(resp.read())
    else:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out = json.loads(resp.read())
    choice = out["choices"][0]
    content = choice["message"].get("content") or ""
    # Raise on empty content so callers can retry with a suitable token budget or reasoning setting.
    if not content.strip():
        u = out.get("usage", {}) or {}
        rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
        raise RuntimeError(
            f"模型返回空 content（finish={choice.get('finish_reason')}, "
            f"reasoning_tokens={rt}, max_tokens={max_tokens}）——"
            f"多半推理耗尽预算，需要更大 max_tokens 或关 thinking。")
    return content


def generate(prompt: str, *, max_tokens: int = 8000, temperature: float = 0.3,
             timeout: int = 900, retries: int = 3, tag: str = "",
             thinking: bool | None = None, model: str | None = None) -> str:
    """Single LLM egress: sanitize, assert clean, audit, then transmit.

    This is fail-closed: unavailable desensitizer or any residual sensitive token rejects the call.
    ``_post`` repeats the guard as defense in depth.
    """
    a = audit(prompt)
    a["tag"] = tag
    try:
        clean_prompt = sanitize_and_assert(prompt, tag=tag or "llm_gen.generate")
        a["rejected"] = False
    except SensitiveEgressError as e:
        a["rejected"] = True
        with _AUDIT_LOCK:
            _AUDIT_LOG.append(a)
        print(f"  [llm_gen egress] ⛔ 拒发 (tag={tag}): {e}")
        raise SensitiveContentError(f"拒发(tag={tag}): {e} —— 未发送任何内容") from e
    with _AUDIT_LOCK:
        _AUDIT_LOG.append(a)

    last_err = None
    for i in range(retries):
        try:
            return _post([{"role": "user", "content": clean_prompt}],
                         max_tokens, temperature, timeout, thinking=thinking,
                         model=model)
        except SensitiveEgressError:
            raise
        except Exception as e:  # noqa
            last_err = e
            print(f"  [llm_gen] 调用失败({i+1}/{retries}): {e}")
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"deepseek 生成失败(tag={tag}): {last_err}")


def _extract_json(text: str) -> dict:
    """Extract JSON from a fenced block or the first balanced object."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    # Fallback to the first balanced JSON object.
    start = text.find("{")
    if start < 0:
        raise ValueError(f"回复里没有 JSON: {text[:200]!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError(f"JSON 括号不平衡: {text[start:start+200]!r}")


def _salvage_json(text: str):
    """Recover truncated JSON by retaining complete objects from an outer array and restoring closing delimiters; return a dictionary or None."""
    start = text.find("{")
    if start < 0:
        return None
    s = text[start:]
    # Locate an outer array field.
    
    m = re.search(r'"[A-Za-z_]+"\s*:\s*\[', s)
    if not m:
        return None
    arr_start = m.end()  # Start of the array contents.
    # Collect complete objects whenever nesting returns to the array level.
    depth = 0
    last_complete = None  # End position immediately after the last complete object's closing brace.
    in_str = False
    esc = False
    for i in range(arr_start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last_complete = i + 1  # An array element has just closed.
    if last_complete is None:
        return None
    candidate = s[:arr_start] + s[arr_start:last_complete] + "]}"
    try:
        return json.loads(candidate)
    except Exception:  # noqa: BLE001
        return None


def generate_json(prompt: str, *, max_tokens: int = 8000, temperature: float = 0.2,
                  retries: int = 3, tag: str = "", thinking: bool | None = None,
                  model: str | None = None) -> dict:
    """Request sanitized JSON, parse it, and feed parse errors back for retries. On final failure, salvage complete elements where possible. Set thinking=False for narrow classification or extraction and select a model explicitly when needed."""
    cur = prompt
    last_err = None
    last_text = ""
    for i in range(retries):
        text = generate(cur, max_tokens=max_tokens, temperature=temperature,
                        tag=f"{tag}#json{i}", thinking=thinking, model=model)
        last_text = text
        try:
            return _extract_json(text)
        except Exception as e:  # noqa
            last_err = e
            cur = (prompt + f"\n\n上次输出无法解析为 JSON（错误：{e}）。"
                   "请只输出一个合法 JSON，用 ```json ``` 包裹，不要任何解释文字。"
                   "若内容较多，请**精简每个元素的文字**确保 JSON 完整闭合，不要被截断。")
    salvaged = _salvage_json(last_text)
    if salvaged is not None:
        print(f"  [generate_json salvage] tag={tag} 截断兜底成功，保住 "
              f"{sum(len(v) for v in salvaged.values() if isinstance(v, list))} 个完整元素")
        return salvaged
    raise RuntimeError(f"generate_json 失败(tag={tag}): {last_err}")


def _extract_code(text: str) -> str:
    """Extract the longest Python code block. Tolerate a missing closing fence or unfenced code, then strip residual standalone fences to avoid syntax errors."""
    if not text:
        return ""
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if blocks:
        code = max(blocks, key=len).strip()
    else:
        # With only an opening fence, take the remaining text; without fences, preserve the text.
        m = re.search(r"```(?:python)?\s*\n?", text)
        code = text[m.end():] if m else text
    # Strip residual standalone code-fence lines.
    lines = code.splitlines()
    while lines and re.fullmatch(r"\s*```(?:python)?\s*", lines[0]):
        lines.pop(0)
    while lines and re.fullmatch(r"\s*```\s*", lines[-1]):
        lines.pop()
    return "\n".join(lines).strip()


def generate_with_repair(build_prompt, run_fn, *, max_iters: int = 3,
                         tag: str = "", thinking: bool | None = None,
                         max_tokens: int = 8000) -> dict:
    """Run a bounded generation/execution repair loop. build_prompt receives prior code and error; run_fn returns execution results. Disable reasoning by default for reference code generation. Return the successful result or a failure dictionary with error and code."""
    last_code, last_err = None, None
    for i in range(max_iters):
        prompt = build_prompt(last_code, last_err)
        text = generate(prompt, max_tokens=max_tokens, temperature=0.2,
                        timeout=900, tag=f"{tag}#repair{i}", thinking=thinking)
        code = _extract_code(text)
        import os as _os
        _dbg = _os.getenv("STEM_REF_DEBUG")
        if _dbg:
            with open(_dbg, "a", encoding="utf-8") as _f:
                _f.write(f"\n##### raw generate text ({len(text)} chars), ``` count="
                         f"{text.count('```')} #####\n{text[:6000]}\n##### END RAW #####\n")
        result = run_fn(code)
        result["code"] = code
        result["iters"] = i + 1
        if result.get("ok"):
            return result
        last_code, last_err = code, result.get("stderr") or result.get("error")
        print(f"  [llm_gen repair] 第{i+1}轮执行失败，拼错误重试")
    return {"ok": False, "error": last_err, "code": last_code, "iters": max_iters}
