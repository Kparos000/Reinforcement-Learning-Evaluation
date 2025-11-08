#!/usr/bin/env python3
"""Test if grader accepts a manually crafted legal rewrite."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ace_task.scenarios import get_scenario
from ace_task.grader import grade

# Load legal scenario
scenario = get_scenario("legal")

# Manually crafted rewrite using EXACT approved aliases
test_rewrites = [
    # Test 1: Use exact approved aliases
    "expires 12/31/2025, renewal $50,000, 30-day window, 90-day notice",
    # Test 2: Use full facts
    "terminates on December 31, 2025, renewal fee of $50,000, within 30 days, 90 days written notice",
    # Test 3: Mix of aliases
    "ends Dec 31, 2025, renewal $50,000, 30-day window, 90-day notice",
]

for i, rewrite in enumerate(test_rewrites, 1):
    print(f"\n{'='*80}")
    print(f"TEST {i}: {rewrite}")
    print(f"Length: {len(rewrite)} chars, {len(rewrite.split())} words")
    print('='*80)

    # Build full JSON output
    output = {
        "rewrite": rewrite,
        "preserved_facts": scenario.facts,
        "at_risk_facts": [],
        "key_insight": "preserving quantitative details prevents context collapse in domain-specific analysis",
        "delta_update": "accurate fact preservation maintains semantic fidelity and enables reliable reasoning"
    }

    model_text = json.dumps(output)

    # Grade it
    passed, reason = grade(
        original=scenario.original,
        facts=scenario.facts,
        banned=scenario.banned,
        model_text=model_text,
        alias_map=scenario.alias_map,
        concision_limit=0.60,
        word_cap=16,
    )

    print(f"\n{'✅' if passed else '❌'} Result: {reason}")
