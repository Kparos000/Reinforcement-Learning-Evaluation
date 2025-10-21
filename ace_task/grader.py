"""
Deterministic grader for the ACE RL evaluation task.

Checks:
- JSON schema & types
- Concision: rewrite < 60% of original chars (debug ratio shown on fail)
- Optional word cap (≤ WORD_CAP) for extra brevity pressure
- Preservation of all FACTS (verbatim or via ALIAS_MAP)
- No BANNED terms introduced
- Numeric fidelity: all numbers/percent/currency from ORIGINAL appear in rewrite
- ACE alignment: key_insight mentions avoiding context collapse via preserving quantitative facts; delta_update is a clear, actionable sentence

Returns:
    (bool, str) -> (pass?, reason)
"""

from __future__ import annotations
import json
import re
import string
from typing import List, Tuple
from .data import ALIAS_MAP  # {"fact": ["alias1", "alias2", ...]}

# Tunables (kept minimal and explicit)
CONCISION_LIMIT = 0.70        # rewrite must be < 60% of original characters
WORD_CAP = 16                 # also keep rewrite to ≤ 16 words (set to None to disable)
MIN_DELTA_WORDS = 6           # heuristic: ensure delta_update is substantive

# Simple signals that key_insight addresses “context collapse” via preserving quantitative facts
INSIGHT_PATTERNS = [
    r"\bcontext collapse\b",
    r"\bpreserv(e|ing)\b.*\b(metric|number|numeric|quant(itative)?|percent|unit|figure)s?\b",
    r"\bkeep\b.*\bnumbers?\b",
]

# --------------------------- small helpers ---------------------------

def _norm(s: str) -> str:
    """Lowercase, strip punctuation (keep %), collapse spaces."""
    s = s.lower()
    # remove punctuation except % and $ (keep units/currency visible)
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
    """
    Extract numbers we care about:
      - percentages: 3.2%
      - currency with or without suffix: $5.1B, $10, $7.5m
      - plain numbers (int/float): 2024, 3.2
    """
    pct = r"\d+(?:\.\d+)?%"
    cur = r"\$\d+(?:\.\d+)?[a-zA-Z]?"
    num = r"\b\d+(?:\.\d+)?\b"
    return set(re.findall(f"{pct}|{cur}|{num}", text))

def _has_sentence(text: str) -> bool:
    # One clear sentence heuristic: ends with period or is a short single-sentence clause.
    return bool(re.search(r"[.!?]\s*$", text)) or len(text.split()) >= MIN_DELTA_WORDS

# ------------------------------ grading ------------------------------

def grade(original: str, facts: List[str], banned: set[str], model_text: str) -> Tuple[bool, str]:
    # 1) JSON parse & schema
    try:
        obj = json.loads(model_text)
    except Exception as e:
        return False, f"Bad JSON: {e}"

    required_keys = {"rewrite", "preserved_facts", "at_risk_facts", "key_insight", "delta_update"}
    if not isinstance(obj, dict) or set(obj.keys()) != required_keys:
        return False, "Wrong keys (expected exactly: rewrite, preserved_facts, at_risk_facts, key_insight, delta_update)."

    rewrite = obj.get("rewrite")
    if not isinstance(rewrite, str) or not rewrite.strip():
        return False, "rewrite must be a non-empty string."

    if not isinstance(obj["preserved_facts"], list) or not isinstance(obj["at_risk_facts"], list):
        return False, "preserved_facts and at_risk_facts must be lists."
    if not isinstance(obj["key_insight"], str) or not isinstance(obj["delta_update"], str):
        return False, "key_insight and delta_update must be strings."

    # 2) Concision checks
    ratio = len(rewrite) / max(1, len(original))
    if ratio > CONCISION_LIMIT:
        return False, f"Rewrite not concise enough (>60%). ratio={ratio:.2f} (len(rewrite)={len(rewrite)}, len(original)={len(original)})"

    if WORD_CAP is not None and _word_count(rewrite) > WORD_CAP:
        return False, f"Too many words (> {WORD_CAP}). words={_word_count(rewrite)}"

    # 3) Banned terms (normalized substring)
    rew_norm = _norm(rewrite)
    for t in banned:
        if _norm(t) and _norm(t) in rew_norm:
            return False, f"Contains banned term: {t}"

    # 4) Facts (allow aliases)
    missing = [f for f in facts if not _fact_present(rew_norm, f)]
    if missing:
        return False, f"Missing facts: {missing}"

    # 5) Numeric fidelity (ORIGINAL numbers must appear in rewrite)
    orig_nums = _nums(original)
    rew_nums = _nums(rewrite)
    if not orig_nums.issubset(rew_nums):
        # Show exactly what’s missing to aid debugging
        lost = sorted(orig_nums - rew_nums)
        return False, f"Numeric info lost: {lost}"

    # 6) ACE alignment
    ki = obj["key_insight"].strip().lower()
    if not ki or not any(re.search(p, ki) for p in INSIGHT_PATTERNS):
        return False, "key_insight not ACE-aligned (should mention preserving quantitative facts to avoid context collapse)."

    du = obj["delta_update"].strip()
    if len(du.split()) < MIN_DELTA_WORDS or not _has_sentence(du):
        return False, "delta_update not a clear actionable sentence."

    # If all checks pass
    return True, "pass"
