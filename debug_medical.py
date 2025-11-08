#!/usr/bin/env python3
"""Debug script to see what Claude outputs for medical scenario."""

import json
import os
import sys
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).parent))

from ace_task.scenarios import get_scenario

# Load medical scenario
scenario = get_scenario("medical")

# Convert facts to proper JSON format
facts_json = json.dumps(scenario.facts)

# Calculate concision limit
max_chars = int(len(scenario.original) * 0.60)

# Create example rewrite
example_facts = scenario.facts[:min(3, len(scenario.facts))]
example_rewrite = ", ".join(example_facts)

# Build the prompt
prompt = f"""You are rewriting text for Agentic Context Engineering (ACE).

Original text:
{scenario.original}

Your task: Output ONLY valid JSON with these exact 5 keys:

{{
  "rewrite": "{example_rewrite}",
  "preserved_facts": {facts_json},
  "at_risk_facts": [],
  "key_insight": "preserving quantitative details prevents context collapse in domain-specific analysis",
  "delta_update": "accurate fact preservation maintains semantic fidelity and enables reliable reasoning"
}}

CRITICAL RULES:
- rewrite: Must include ALL these exact phrases: {facts_json}
  Keep it VERY SHORT (under {max_chars} characters - example above is {len(example_rewrite)} chars)
- preserved_facts: {facts_json}
- at_risk_facts: [] (always empty list)
- key_insight: Must mention "preserving quantitative" or "context collapse" (8+ words)
- delta_update: Must be meaningful sentence (8+ words)
- Use DOUBLE QUOTES for all JSON strings
- Output ONLY the JSON object, NO explanations"""

print("=" * 80)
print("MEDICAL SCENARIO DEBUG")
print("=" * 80)
print(f"\nOriginal: {scenario.original}")
print(f"Facts: {scenario.facts}")
print(f"Max chars allowed: {max_chars}")
print(f"Example rewrite length: {len(example_rewrite)} chars")
print(f"\nPrompt being sent to Claude:")
print("-" * 80)
print(prompt)
print("-" * 80)

# Call Claude
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1000,
    temperature=1.0,
    messages=[{"role": "user", "content": prompt}]
)

output = response.content[0].text
print(f"\nClaude's response:")
print("-" * 80)
print(output)
print("-" * 80)

# Try to parse and grade
try:
    parsed = json.loads(output)
    print(f"\n✅ Valid JSON")
    print(f"Keys: {list(parsed.keys())}")
    print(f"Rewrite: {parsed.get('rewrite', 'N/A')}")
    print(f"Rewrite length: {len(parsed.get('rewrite', ''))} chars (max {max_chars})")
except Exception as e:
    print(f"\n❌ JSON parsing failed: {e}")

# Try grading
from ace_task.grader import grade

try:
    passed, reason = grade(
        original=scenario.original,
        facts=scenario.facts,
        banned=scenario.banned,
        model_text=output,
        alias_map=scenario.alias_map,
        concision_limit=0.60,
    )
    print(f"\n{'✅' if passed else '❌'} Grading result: {reason}")
except Exception as e:
    print(f"\n❌ Grading failed: {e}")
