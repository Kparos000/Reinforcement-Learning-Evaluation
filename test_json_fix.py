"""Test with proper JSON formatting in prompt."""

import json
import os

import yaml
from anthropic import Anthropic

from ace_task.grader import grade
from ace_task.scenarios import get_scenario

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Set up
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
scenario = get_scenario("economics")

# Convert facts to proper JSON format (with double quotes)
facts_json = json.dumps(scenario.facts)

# NEW PROMPT with proper JSON formatting and SHORT rewrite
prompt = f"""You are evaluating an economics text for Agentic Context Engineering (ACE).

Original text:
{scenario.original}

Your task: Output ONLY valid JSON with these exact 5 keys:

{{
  "rewrite": "GDP grew by 3.2%, inflation was 2.1%, exports increased",
  "preserved_facts": {facts_json},
  "at_risk_facts": [],
  "key_insight": "preserving quantitative details prevents context collapse in economic analysis",
  "delta_update": "supply chain normalization drives export growth and economic recovery"
}}

CRITICAL:
- rewrite: Must include EXACTLY these phrases: "GDP grew by 3.2%", "inflation was 2.1%", "exports increased" - KEEP IT SHORT (under 57 characters)
- preserved_facts: {facts_json}
- at_risk_facts: []
- key_insight: Must contain "preserving quantitative" or "context collapse"
- delta_update: Must be 6+ words
- Use DOUBLE QUOTES for all strings in JSON
- NO explanations, ONLY the JSON object"""

print("=" * 70)
print("TESTING WITH JSON-CORRECTED PROMPT")
print("=" * 70)
print(prompt)
print("\n" + "=" * 70)
print("GENERATING...")
print("=" * 70)

message = client.messages.create(
    model=config["model"]["name"],
    max_tokens=config["model"]["max_tokens"],
    temperature=1.0,
    messages=[{"role": "user", "content": prompt}],
)

output = message.content[0].text

print(f"\nClaude's output:\n{output}\n")

# Grade it
passed, reason = grade(
    original=scenario.original,
    facts=scenario.facts,
    banned=scenario.banned,
    model_text=output,
    alias_map=scenario.alias_map,
    word_cap=config["grader"]["word_cap"],
)

print("=" * 70)
print(f"RESULT: {'✅ PASSED' if passed else '❌ FAILED'}")
print(f"REASON: {reason}")
print("=" * 70)
