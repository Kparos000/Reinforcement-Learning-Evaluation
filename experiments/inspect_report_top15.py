"""
Quick one-off inspector for the report_top15 scenario.

Runs a single Anthropic call, prints raw/parsed JSON, GradeResult details,
and the dense reward. Use for debugging model behavior on the learnable long-form task.
"""

from __future__ import annotations

import json
import os
from math import floor

from anthropic import Anthropic
from dotenv import load_dotenv

from ace_task.algorithms.rewards import compute_dense_reward
from ace_task.evaluate import build_user_message
from ace_task.grader import GradeResult, grade_detailed
from ace_task.scenarios import get_scenario


def run_inspection() -> None:
    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise EnvironmentError("Missing ANTHROPIC_API_KEY in environment or .env")

    scenario = get_scenario("report_top15")
    concision_limit = getattr(scenario, "concision_limit", None) or 0.70
    word_cap = getattr(scenario, "word_cap", None) or 140
    max_chars = floor(len(scenario.original) * concision_limit)

    prompt = build_user_message(max_chars=max_chars, max_words=word_cap, scenario=scenario)

    client = Anthropic()
    print("Calling model once on report_top15 ...")
    msg = client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=500,
        temperature=0.60,
        top_p=0.9,
        system=(
            "Output ONLY strict JSON (no prose). Keep 'rewrite' within the provided character/word limits. "
            "Preserve all facts and numbers; avoid banned terms and extra claims. "
            "Your key_insight MUST explicitly mention preserving quantitative details to avoid context collapse."
        ),
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(c.text for c in msg.content if c.type == "text").strip()

    print("\nRaw model text:")
    print(text)

    parsed = None
    try:
        parsed = json.loads(text)
        print("\nParsed JSON:")
        print(json.dumps(parsed, indent=2))
    except Exception as e:
        print(f"\nFailed to parse JSON: {e}")

    grade: GradeResult = grade_detailed(
        original=scenario.original,
        facts=scenario.facts,
        banned=scenario.banned,
        model_text=text,
        alias_map=scenario.alias_map,
        concision_limit=concision_limit,
        word_cap=word_cap,
    )
    reward = compute_dense_reward(scenario, text)

    missing = ""
    if not grade.passed and grade.reason.startswith("Missing facts:"):
        missing = grade.reason.replace("Missing facts:", "").strip()

    print("\nGradeResult:")
    print(f"  passed: {grade.passed}")
    print(f"  reason: {grade.reason}")
    print(f"  facts_matched: {grade.facts_matched}/{grade.facts_total}")
    print(f"  banned_term_violations: {grade.banned_term_violations}")
    print(f"  length_violation: {grade.length_violation}")
    print(f"  missing (parsed from reason if any): {missing}")
    print(f"  dense reward: {reward:.3f}")


if __name__ == "__main__":
    run_inspection()
