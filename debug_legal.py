#!/usr/bin/env python3
"""Debug script to see what Claude outputs for legal scenario."""

import json
import os
import sys
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).parent))

from ace_task.grader import grade
from ace_task.scenarios import get_scenario

# Load legal scenario
scenario = get_scenario("legal")

# Convert facts to proper JSON format
facts_json = json.dumps(scenario.facts)

# Calculate concision limit
max_chars = int(len(scenario.original) * 0.60)

# Word cap (from config.yaml)
word_cap = 16

# Build alias information
alias_info = ""
if hasattr(scenario, "alias_map") and scenario.alias_map:
    alias_info = "\n\nAccepted shorter versions (use these to save space, but keep exact numeric values like $50,000 not $50K):\n"
    for fact, aliases in scenario.alias_map.items():
        alias_info += f'- "{fact}" can be: {json.dumps(aliases)}\n'

# Build the prompt
prompt = f"""You are rewriting text for Agentic Context Engineering (ACE).

Original text:
{scenario.original}

Required facts: {facts_json}{alias_info}

Your task: Output ONLY valid JSON with these exact 5 keys:

{{
  "rewrite": "your concise rewrite here",
  "preserved_facts": {facts_json},
  "at_risk_facts": [],
  "key_insight": "preserving quantitative details prevents context collapse in domain-specific analysis",
  "delta_update": "accurate fact preservation maintains semantic fidelity and enables reliable reasoning"
}}

CRITICAL RULES:
- rewrite: Must include ALL required facts (use exact wording OR accepted shorter versions)
  Keep VERY concise (max {max_chars} chars AND max {word_cap} words)
  IMPORTANT: Keep numbers EXACTLY as they appear in original (e.g., $50,000 not $50K, 30 not 30d without space)
- preserved_facts: {facts_json} (always use full fact names here, not aliases)
- at_risk_facts: [] (always empty list)
- key_insight: Must mention "preserving quantitative" or "context collapse" (8+ words)
- delta_update: Must be meaningful sentence (8+ words)
- Use DOUBLE QUOTES for all JSON strings
- Output ONLY the JSON object, NO explanations"""

print("=" * 80)
print("LEGAL SCENARIO DEBUG")
print("=" * 80)
print(f"\nOriginal: {scenario.original}")
print(f"Original length: {len(scenario.original)} chars")
print(f"Facts: {scenario.facts}")
print(f"Max chars allowed: {max_chars}")
print(f"Max words allowed: {word_cap}")
print("\nPrompt being sent to Claude:")
print("-" * 80)
print(prompt)
print("-" * 80)

# Call Claude
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
response = client.messages.create(
    model="claude-3-5-haiku-latest",
    max_tokens=1000,
    temperature=1.0,
    messages=[{"role": "user", "content": prompt}],
)

output = response.content[0].text
print("\nClaude's response:")
print("-" * 80)
print(output)
print("-" * 80)

# Try to parse and grade
try:
    parsed = json.loads(output)
    print("\n✅ Valid JSON")
    print(f"Keys: {list(parsed.keys())}")
    print(f"Rewrite: {parsed.get('rewrite', 'N/A')}")
    print(f"Rewrite length: {len(parsed.get('rewrite', ''))} chars (max {max_chars})")
except Exception as e:
    print(f"\n❌ JSON parsing failed: {e}")

# Try grading
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
