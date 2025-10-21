"""
Deterministic grader for the task.

- Validates JSON shape exactly matches the prompt.
- Enforces compression bound (≤60% original characters).
- Verifies all facts are present (verbatim or alias).
- Verifies numbers/units are preserved (e.g., 3.2%, 2.1%).
- Blocks banned terms (hallucination traps).
- Requires ACE-style key insight and a one-sentence delta update.

Returns (bool, reason).
"""

import json, re, string
from typing import List, Tuple
from .data import ALIAS_MAP

# Small set of patterns that indicate ACE-aligned reflection
INSIGHT_PATTERNS = [
    r"\bpreserv(e|ing)\b.*\bnumeric\b",      # e.g., "preserving numeric details"
    r"\bavoid\b.*\bcontext collapse\b",      # e.g., "avoid context collapse"
    r"\bdo not\b.*\bcompress away\b",        # e.g., "do not compress away facts"
]

def _norm(s: str) -> str:
    """
    Normalize strings to make matching robust without embeddings:
    - Lowercase
    - Strip punctuation
    - Collapse whitespace
    - Tighten percent spacing
    """
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("% ", "%")
    return s

def _fact_present(rewrite_norm: str, fact: str) -> bool:
    """
    Check if a fact (or any of its aliases) appears in the normalized rewrite.
    """
    cands = [_norm(fact)] + [_norm(a) for a in ALIAS_MAP.get(fact, [])]
    return any(c in rewrite_norm for c in cands)

def _nums(text: str):
    """
    Extract numbers and units we care about:
    - percents (e.g., 3.2%)
    - money with optional suffix (e.g., $9.9B)
    - plain numeric values (fallback)
    """
    return re.findall(r"\d+(?:\.\d+)?%|\$\d+(?:\.\d+)?[a-zA-Z]*|\d+(?:\.\d+)?", text)

def grade(original: str, facts: List[str], banned: set[str], model_text: str) -> Tuple[bool, str]:
    # 1) JSON + schema check
    try:
        obj = json.loads(model_text)
        if set(obj.keys()) != {"rewrite","preserved_facts","at_risk_facts","key_insight","delta_update"}:
            return False, "Wrong keys"
        rewrite = obj["rewrite"]
    except Exception as e:
        return False, f"Bad JSON/schema: {e}"

    # 2) Compression bound
    if len(rewrite) / max(1, len(original)) > 0.60:
        return False, "Rewrite not concise enough (>60%)."

    rew_norm = _norm(rewrite)

    # 3) Banned terms (hallucination guard)
    for t in banned:
        if _norm(t) in rew_norm:
            return False, f"Contains banned term: {t}"

    # 4) Facts preserved (allow aliases)
    missing = [f for f in facts if not _fact_present(rew_norm, f)]
    if missing:
        return False, f"Missing facts: {missing}"

    # 5) Numbers & units preserved exactly
    orig_nums = set(_nums(original))
    rew_nums  = set(_nums(rewrite))
    if not orig_nums.issubset(rew_nums):
        return False, f"Numeric/units lost: {sorted(orig_nums - rew_nums)}"

    # 6) ACE-style reflection (Reflector/Curator micro-outputs)
    ki = obj.get("key_insight", "").lower()
    if not any(re.search(p, ki) for p in INSIGHT_PATTERNS):
        return False, "Key insight not ACE-aligned."

    du = obj.get("delta_update", "").strip()
    if len(du.split()) < 6 or du.count(".") == 0:
        return False, "delta_update must be one clear, actionable sentence."

    return True, "pass"
