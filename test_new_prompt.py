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

# NEW PROMPT (from the fix)
prompt = f"""You are evaluating an economics text for Agentic Context Engineering (ACE).

Original text:
{scenario.original}

Your task: Rewrite this text following these STRICT rules:
1. Include ALL these EXACT facts: {scenario.facts}
2. NEVER use these banned words: {scenario.banned}
3. Keep the rewrite VERY concise (max {config['grader']['word_cap']} words)
4. Output ONLY valid JSON in this exact format (no explanation):

{{
  "rewrite": "your concise rewrite here (max {config['grader']['word_cap']} words)",
  "key_insight": "main insight (6-8 words)",
  "delta_update": "what changed (6-8 words)"
}}

CRITICAL: Use the EXACT fact wording from the list above. Output ONLY the JSON object, nothing else."""

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
