"""
Deterministic grader for the ACE RL evaluation task.
Core ACE-style component used by evaluation, Best-of-N, and reward functions.

Checks:
- JSON schema & types
- Concision: rewrite < concision_limit of original chars (debug ratio on fail)
- Optional word cap
- Preservation of all FACTS (verbatim or via ALIAS_MAP)
- No BANNED terms
- Numeric fidelity: numbers/percents/currency from ORIGINAL appear in rewrite
- ACE alignment: key_insight mentions preserving quantitative detail to avoid context collapse; delta_update is actionable

Returns:
    (bool, str) -> (pass?, reason) via grade()
    GradeResult for structured scoring via grade_detailed()
"""

from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

CONCISION_LIMIT = 0.60  # 60% of original characters
WORD_CAP = 16  # set to None to disable
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
    keep = set("%$.")  # Keep %, $, and decimal points
    s = "".join(ch for ch in s if (ch not in string.punctuation) or (ch in keep))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _word_count(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s))


def _fact_present(rewrite_norm: str, fact: str, alias_map: Dict[str, List[str]]) -> bool:
    cands = [_norm(fact)] + [_norm(a) for a in alias_map.get(fact, [])]
    return any(c in rewrite_norm for c in cands)


def _nums(text: str) -> set[str]:
    pct = r"\d+(?:\.\d+)?%"
    cur = r"\$\d+(?:\.\d+)?[a-zA-Z]?"
    num = r"\b\d+(?:\.\d+)?\b"
    return set(re.findall(f"{pct}|{cur}|{num}", text))


def _has_sentence(text: str) -> bool:
    return bool(re.search(r"[.!?]\s*$", text)) or len(text.split()) >= MIN_DELTA_WORDS


@dataclass
class GradeResult:
    """Structured grading result for dense rewards and diagnostics."""

    passed: bool
    reason: str
    facts_total: int
    facts_matched: int
    banned_term_violations: int
    length_violation: bool


def grade_detailed(
    original: str,
    facts: List[str],
    banned: set[str],
    model_text: str,
    alias_map: Optional[Dict[str, List[str]]] = None,
    concision_limit: Optional[float] = None,
    word_cap: Optional[int] = None,
) -> GradeResult:
    """
    Grade a model's output against requirements and return a structured result.
    """
    if alias_map is None:
        alias_map = {}
    if concision_limit is None:
        concision_limit = CONCISION_LIMIT
    if word_cap is None:
        word_cap = WORD_CAP
    facts_total = len(facts)
    banned_hits = 0
    length_violation = False
    facts_matched = 0

    # 1) JSON
    try:
        obj = json.loads(model_text)
    except Exception as e:
        return GradeResult(
            passed=False,
            reason=f"Bad JSON: {e}",
            facts_total=facts_total,
            facts_matched=0,
            banned_term_violations=0,
            length_violation=False,
        )

    required = {"rewrite", "preserved_facts", "at_risk_facts", "key_insight", "delta_update"}
    if not isinstance(obj, dict) or set(obj.keys()) != required:
        return GradeResult(
            passed=False,
            reason="Wrong keys (expected exactly: rewrite, preserved_facts, at_risk_facts, key_insight, delta_update).",
            facts_total=facts_total,
            facts_matched=0,
            banned_term_violations=0,
            length_violation=False,
        )

    rewrite = obj.get("rewrite")
    if not isinstance(rewrite, str) or not rewrite.strip():
        return GradeResult(
            passed=False,
            reason="rewrite must be a non-empty string.",
            facts_total=facts_total,
            facts_matched=0,
            banned_term_violations=0,
            length_violation=False,
        )
    if not isinstance(obj["preserved_facts"], list) or not isinstance(obj["at_risk_facts"], list):
        return GradeResult(
            passed=False,
            reason="preserved_facts and at_risk_facts must be lists.",
            facts_total=facts_total,
            facts_matched=0,
            banned_term_violations=0,
            length_violation=False,
        )
    if not isinstance(obj["key_insight"], str) or not isinstance(obj["delta_update"], str):
        return GradeResult(
            passed=False,
            reason="key_insight and delta_update must be strings.",
            facts_total=facts_total,
            facts_matched=0,
            banned_term_violations=0,
            length_violation=False,
        )

    # 2) Concision
    ratio = len(rewrite) / max(1, len(original))
    if ratio > concision_limit:
        return GradeResult(
            passed=False,
            reason=(
                f"Rewrite not concise enough (>{concision_limit:.0%}). ratio={ratio:.2f} "
                f"(len(rewrite)={len(rewrite)}, len(original)={len(original)})"
            ),
            facts_total=facts_total,
            facts_matched=0,
            banned_term_violations=0,
            length_violation=True,
        )
    if word_cap is not None and _word_count(rewrite) > word_cap:
        return GradeResult(
            passed=False,
            reason=f"Too many words (> {word_cap}). words={_word_count(rewrite)}",
            facts_total=facts_total,
            facts_matched=0,
            banned_term_violations=0,
            length_violation=True,
        )

    # 3) Banned terms
    rew_norm = _norm(rewrite)
    for t in banned:
        if _norm(t) and _norm(t) in rew_norm:
            banned_hits += 1
            return GradeResult(
                passed=False,
                reason=f"Contains banned term: {t}",
                facts_total=facts_total,
                facts_matched=0,
                banned_term_violations=banned_hits,
                length_violation=length_violation,
            )

    # 4) Facts
    missing = [f for f in facts if not _fact_present(rew_norm, f, alias_map)]
    if missing:
        facts_matched = facts_total - len(missing)
        return GradeResult(
            passed=False,
            reason=f"Missing facts: {missing}",
            facts_total=facts_total,
            facts_matched=facts_matched,
            banned_term_violations=banned_hits,
            length_violation=length_violation,
        )
    facts_matched = facts_total

    # 5) Numeric fidelity
    if not _nums(original).issubset(_nums(rewrite)):
        lost = sorted(_nums(original) - _nums(rewrite))
        return GradeResult(
            passed=False,
            reason=f"Numeric info lost: {lost}",
            facts_total=facts_total,
            facts_matched=facts_matched,
            banned_term_violations=banned_hits,
            length_violation=length_violation,
        )

    # 6) ACE alignment
    ki = obj["key_insight"].strip().lower()
    if not ki or not any(re.search(p, ki) for p in INSIGHT_PATTERNS):
        return GradeResult(
            passed=False,
            reason="key_insight not ACE-aligned (should mention preserving quantitative facts to avoid context collapse).",
            facts_total=facts_total,
            facts_matched=facts_matched,
            banned_term_violations=banned_hits,
            length_violation=length_violation,
        )

    du = obj["delta_update"].strip()
    if len(du.split()) < MIN_DELTA_WORDS or not _has_sentence(du):
        return GradeResult(
            passed=False,
            reason="delta_update not a clear actionable sentence.",
            facts_total=facts_total,
            facts_matched=facts_matched,
            banned_term_violations=banned_hits,
            length_violation=length_violation,
        )

    return GradeResult(
        passed=True,
        reason="pass",
        facts_total=facts_total,
        facts_matched=facts_matched,
        banned_term_violations=banned_hits,
        length_violation=length_violation,
    )


def grade(
    original: str,
    facts: List[str],
    banned: set[str],
    model_text: str,
    alias_map: Optional[Dict[str, List[str]]] = None,
    concision_limit: Optional[float] = None,
    word_cap: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    Backwards-compatible grading API: returns (pass?, reason).
    """
    result = grade_detailed(
        original=original,
        facts=facts,
        banned=banned,
        model_text=model_text,
        alias_map=alias_map,
        concision_limit=concision_limit,
        word_cap=word_cap,
    )
    return result.passed, result.reason
