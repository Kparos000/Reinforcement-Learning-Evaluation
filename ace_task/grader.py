"""
Deterministic grader for the ACE RL evaluation task.

Checks:
- JSON schema & types
- Concision: rewrite < 60% of original chars (debug ratio on fail)
- Optional word cap (≤ WORD_CAP)
- Preservation of all FACTS (verbatim or via ALIAS_MAP)
- No BANNED terms
- Numeric fidelity: numbers/percents/currency from ORIGINAL appear in rewrite
- ACE alignment: key_insight mentions preserving quantitative detail to avoid context collapse; delta_update is actionable

Returns:
    (bool, str) -> (pass?, reason)
"""

from __future__ import annotations
import json
import re
import string
from typing import List, Tuple
from .data import ALIAS_MAP

CONCISION_LIMIT = 0.60   # 60% of original characters
WORD_CAP = 16            # set to None to disable
MIN_DELTA_WORDS = 6

INSIGHT_PATTERNS = [
    r"\bcontext collapse\b",
    r"\bpreserv\w*\b.*\b(metric|number|numeric|quant(?:itative)?|percent|unit|figure)s?\b",
    r"\bkeep\b.*\b(numbers?|metrics?|percents?|units?)\b",
    r"\bretain\b.*\b(facts?|numbers?|metrics?)\b",
    r"\bnumeric (fidelity|accuracy|detail)\b",
    r"\bquantitative detail\b",
]

def _norm(s: str) -> str:
    s = s.lower()
    keep = set("%$")
    s = "".join(ch for ch in s if (ch not in string.punctuation) or (ch in keep))
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _word_count(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s))

def _fact_present(rewrite_norm: str, fact: str) -> bool:
    cands = [_norm(fact)] + [_norm(a) for a in ALIAS_MAP.get(fact, [])]
    return any(c in rewrite_norm for c in cands)

def _nums(text: str) -> set[str]:
    pct = r"\d+(?:\.\d+)?%"
    cur = r"\$\d+(?:\.\d+)?[a-zA-Z]?"
    num = r"\b\d+(?:\.\d+)?\b"
    return set(re.findall(f"{pct}|{cur}|{num}", text))

def _has_sentence(text: str) -> bool:
    return bool(re.search(r"[.!?]\s*$", text)) or len(text.split()) >= MIN_DELTA_WORDS

def grade(original: str, facts: List[str], banned: set[str], model_text: str) -> Tuple[bool, str]:
    # 1) JSON
    try:
        obj = json.loads(model_text)
    except Exception as e:
        return False, f"Bad JSON: {e}"

    required = {"rewrite", "preserved_facts", "at_risk_facts", "key_insight", "delta_update"}
    if not isinstance(obj, dict) or set(obj.keys()) != required:
        return False, "Wrong keys (expected exactly: rewrite, preserved_facts, at_risk_facts, key_insight, delta_update)."

    rewrite = obj.get("rewrite")
    if not isinstance(rewrite, str) or not rewrite.strip():
        return False, "rewrite must be a non-empty string."
    if not isinstance(obj["preserved_facts"], list) or not isinstance(obj["at_risk_facts"], list):
        return False, "preserved_facts and at_risk_facts must be lists."
    if not isinstance(obj["key_insight"], str) or not isinstance(obj["delta_update"], str):
        return False, "key_insight and delta_update must be strings."

    # 2) Concision
    ratio = len(rewrite) / max(1, len(original))
    if ratio > CONCISION_LIMIT:
        return False, f"Rewrite not concise enough (>60%). ratio={ratio:.2f} (len(rewrite)={len(rewrite)}, len(original)={len(original)})"
    if WORD_CAP is not None and _word_count(rewrite) > WORD_CAP:
        return False, f"Too many words (> {WORD_CAP}). words={_word_count(rewrite)}"

    # 3) Banned terms
    rew_norm = _norm(rewrite)
    for t in banned:
        if _norm(t) and _norm(t) in rew_norm:
            return False, f"Contains banned term: {t}"

    # 4) Facts
    missing = [f for f in facts if not _fact_present(rew_norm, f)]
    if missing:
        return False, f"Missing facts: {missing}"

    # 5) Numeric fidelity
    if not _nums(original).issubset(_nums(rewrite)):
        lost = sorted(_nums(original) - _nums(rewrite))
        return False, f"Numeric info lost: {lost}"

    # 6) ACE alignment
    ki = obj["key_insight"].strip().lower()
    if not ki or not any(re.search(p, ki) for p in INSIGHT_PATTERNS):
        return False, "key_insight not ACE-aligned (should mention preserving quantitative facts to avoid context collapse)."

    du = obj["delta_update"].strip()
    if len(du.split()) < MIN_DELTA_WORDS or not _has_sentence(du):
        return False, "delta_update not a clear actionable sentence."

    return True, "pass"
