"""Local, fail-closed outbound text redaction.

Detect common credential formats, configured secret values, email addresses, private
hostnames and local absolute paths. Deployment-specific names can be supplied as a
JSON string list in STEM_REDACT_TERMS. This is a deterministic guard, not a guarantee
that arbitrary personally identifying prose can be recognized.
"""
from __future__ import annotations
import json
import os
import re
from typing import Any

class SensitiveEgressError(RuntimeError):
    """The payload cannot safely leave the process."""

_HAS_DESENS = True
_PATTERNS = [
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"https?://[^\s/\"']+\.(?:internal|local|corp)(?::\d+)?(?:/[^\s\"']*)?", re.I),
    re.compile(r"(?<![\w:/])/(?!path/to(?:/|\b))(?=[A-Za-z_.])[^\s\"'<>),;\]}]+"),
    re.compile(r"\b[A-Za-z]:\\(?:[^\s\"'<>]+)"),
]

def _terms():
    try:
        terms = json.loads(os.getenv("STEM_REDACT_TERMS", "[]"))
        if not isinstance(terms, list) or any(not isinstance(x, str) or not x for x in terms):
            raise ValueError()
    except Exception:
        raise SensitiveEgressError("STEM_REDACT_TERMS must be a JSON list of nonempty strings") from None
    for key, value in os.environ.items():
        if any(word in key.upper() for word in ("TOKEN", "PASSWORD", "SECRET", "_KEY")) and len(value) >= 8:
            terms.append(value)
    return sorted(set(terms), key=len, reverse=True)

def desensitize(text: str) -> str:
    for term in _terms():
        text = re.sub(re.escape(term), "[REDACTED]", text, flags=re.I)
    for pattern in _PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text

def contains_sensitive(text: str) -> bool:
    return any(term.lower() in text.lower() for term in _terms()) or any(p.search(text) for p in _PATTERNS)

def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)

def has_sensitive(text: str) -> bool:
    if not _HAS_DESENS:
        return True
    try:
        return contains_sensitive(text or "")
    except Exception:
        return True

def sanitize_for_external(text: str) -> str:
    if not _HAS_DESENS:
        raise SensitiveEgressError("outbound redactor unavailable")
    try:
        return desensitize(text)
    except Exception:
        raise SensitiveEgressError("outbound redaction failed") from None

def _sanitize_value(value: Any) -> Any:
    """Redact text before serialization so JSON escaping stays intact."""
    if isinstance(value, str):
        cleaned = sanitize_for_external(value)
        if has_sensitive(cleaned):
            raise SensitiveEgressError("sensitive content remains after sanitization")
        return cleaned
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            clean_key = _sanitize_value(key) if isinstance(key, str) else key
            if clean_key in cleaned:
                raise SensitiveEgressError("redaction would merge distinct object keys")
            cleaned[clean_key] = _sanitize_value(item)
        return cleaned
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_value(item) for item in value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _sanitize_value(str(value))


def sanitize_and_assert(value: Any, *, tag: str = "") -> str:
    cleaned = _as_text(_sanitize_value(value))
    if has_sensitive(cleaned):
        raise SensitiveEgressError("sensitive content remains after sanitization")
    return cleaned


def sanitize_messages(messages: list[dict], *, tag: str = "") -> list[dict]:
    cleaned = _sanitize_value(messages)
    if has_sensitive(_as_text(cleaned)):
        raise SensitiveEgressError("sensitive content remains after sanitization")
    return cleaned

def audit(text: str) -> dict:
    try:
        cleaned = sanitize_and_assert(text)
        return {"ok": True, "cleaned_has_named_secret": False,
                "cleaned_still_sensitive": False, "cleaned_preview": cleaned[:200], "error": ""}
    except Exception:
        return {"ok": False, "cleaned_has_named_secret": True,
                "cleaned_still_sensitive": True, "cleaned_preview": "", "error": "outbound redaction failed"}
