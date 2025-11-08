"""Quick test to see if the prompt was updated and what Claude generates."""

import os
from anthropic import Anthropic
from ace_task.scenarios import get_scenario
from ace_task.grader import grade
import yaml

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Set up
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
scenario = get_scenario("economics")

# NEW PROMPT (with ALL 5 required JSON keys!)
prompt = f"""You are evaluating an economics text for Agentic Context Engineering (ACE).

Original text:
{scenario.original}

Required facts: {scenario.facts}
Banned words: {scenario.banned}

Your task: Rewrite this text following these STRICT rules and output ONLY valid JSON:

{{
  "rewrite": "your concise rewrite using EXACT fact wording (max {config['grader']['word_cap']} words)",
  "preserved_facts": {scenario.facts},
  "at_risk_facts": [],
  "key_insight": "preserving quantitative details prevents context collapse",
  "delta_update": "supply chain improvements drive economic recovery and growth"
}}

CRITICAL REQUIREMENTS:
- rewrite: Use EXACT wording from Required facts list, max {config['grader']['word_cap']} words
- preserved_facts: Copy the list exactly: {scenario.facts}
- at_risk_facts: Always use empty list: []
- key_insight: MUST include phrase "preserving quantitative" or "context collapse"
- delta_update: Actionable sentence, minimum 6 words
- NO banned words
- Output ONLY the JSON, no explanation before or after"""

print("=" * 70)
print("TESTING WITH NEW PROMPT")
print("=" * 70)
print(prompt)
print("\n" + "=" * 70)
print("GENERATING...")
print("=" * 70)

message = client.messages.create(
    model=config["model"]["name"],
    max_tokens=config["model"]["max_tokens"],
    temperature=1.0,
    messages=[{"role": "user", "content": prompt}]
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
    word_cap=config['grader']['word_cap']
)

print("=" * 70)
print(f"RESULT: {'✅ PASSED' if passed else '❌ FAILED'}")
print(f"REASON: {reason}")
print("=" * 70)
