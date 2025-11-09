#!/usr/bin/env python3
"""Debug new scenarios to see why they fail."""

import json
import os
import sys
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).parent))

from ace_task.grader import grade
from ace_task.scenarios import get_scenario

# Test finance scenario
scenario = get_scenario("finance")

# Convert facts to proper JSON format
facts_json = json.dumps(scenario.facts)

# Calculate concision limit
max_chars = int(len(scenario.original) * 0.60)

# Word cap
word_cap = 16

# Create example rewrite (what run_best_of_n.py does)
example_rewrite = ", ".join(scenario.facts)

print("=" * 80)
print("FINANCE SCENARIO DEBUG")
print("=" * 80)
print(f"\nOriginal: {scenario.original}")
print(f"Original length: {len(scenario.original)} chars")
print(f"Facts: {scenario.facts}")
print(f"Max chars allowed: {max_chars}")
print(f"Max words allowed: {word_cap}")
print(f"\nExample rewrite: {example_rewrite}")
print(f"Example length: {len(example_rewrite)} chars")
print(f"Example words: {len(example_rewrite.split())} words")

if len(example_rewrite) > max_chars:
    print(f"\n⚠️  WARNING: Example EXCEEDS char limit! ({len(example_rewrite)} > {max_chars})")

if len(example_rewrite.split()) > word_cap:
    print(f"\n⚠️  WARNING: Example EXCEEDS word limit! ({len(example_rewrite.split())} > {word_cap})")

# Build the prompt
prompt = f"""You are rewriting text for Agentic Context Engineering (ACE).

Original text:
{scenario.original}

Required facts: {facts_json}

Your task: Output ONLY valid JSON with these exact 5 keys:

{{
  "rewrite": "{example_rewrite}",
  "preserved_facts": {facts_json},
  "at_risk_facts": [],
  "key_insight": "preserving quantitative details prevents context collapse in domain-specific analysis",
  "delta_update": "accurate fact preservation maintains semantic fidelity and enables reliable reasoning"
}}

CRITICAL RULES:
- rewrite: Use the EXACT format shown in example above (it includes all required facts)
  Max {max_chars} chars AND max {word_cap} words
- preserved_facts: {facts_json} (always use full fact names, not aliases)
- at_risk_facts: [] (always empty list)
- key_insight: Must mention "preserving quantitative" or "context collapse" (8+ words)
- delta_update: Must be meaningful sentence (8+ words)
- Use DOUBLE QUOTES for all JSON strings
- Output ONLY the JSON object, NO explanations"""

# Call Claude
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
response = client.messages.create(
    model="claude-3-5-haiku-latest",
    max_tokens=1000,
    temperature=1.0,
    messages=[{"role": "user", "content": prompt}]
)

output = response.content[0].text
print(f"\nClaude's response:")
print("-" * 80)
print(output)
print("-" * 80)

# Try grading
try:
    passed, reason = grade(
        original=scenario.original,
        facts=scenario.facts,
        banned=scenario.banned,
        model_text=output,
        alias_map=scenario.alias_map,
        concision_limit=0.60,
        word_cap=16,
    )
    print(f"\n{'✅' if passed else '❌'} Grading result: {reason}")
except Exception as e:
    print(f"\n❌ Grading failed: {e}")
