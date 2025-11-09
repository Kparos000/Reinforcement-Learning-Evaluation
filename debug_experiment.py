"""Debug script to see what Claude generates and why it fails."""

import os

from anthropic import Anthropic

from ace_task.grader import grade
from ace_task.scenarios import get_scenario

# Set up client
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Load scenario
scenario = get_scenario("economics")

print("=" * 70)
print("SCENARIO REQUIREMENTS")
print("=" * 70)
print(f"\nOriginal text:\n{scenario.original}\n")
print(f"Required facts: {scenario.facts}")
print(f"Banned words: {scenario.banned}")
print(f"Alias map: {scenario.alias_map}")
print("Word cap: ~16 words")
print()

# Build prompt (same as in run_best_of_n.py)
prompt = f"""You are evaluating an economics text for Agentic Context Engineering (ACE).

Original text:
{scenario.original}

Your task: Rewrite this text following these rules:
1. Include ALL these facts: {scenario.facts}
2. NEVER use these banned words: {scenario.banned}
3. Keep it concise (max ~16 words)
4. Maintain accuracy and clarity

Generate a rewritten version now:"""

print("=" * 70)
print("PROMPT SENT TO CLAUDE")
print("=" * 70)
print(prompt)
print()

# Generate sample
print("=" * 70)
print("GENERATING SAMPLE...")
print("=" * 70)

message = client.messages.create(
    model="claude-3-5-haiku-latest",
    max_tokens=400,
    temperature=1.0,
    messages=[{"role": "user", "content": prompt}],
)

model_output = message.content[0].text

print(f"\nClaude's raw output:\n{model_output}\n")
print(f"Word count: {len(model_output.split())} words")
print(f"Character count: {len(model_output)} characters")
print()

# Grade it
print("=" * 70)
print("GRADING THE OUTPUT")
print("=" * 70)

passed, reason = grade(
    original=scenario.original,
    facts=scenario.facts,
    banned=scenario.banned,
    model_text=model_output,
    alias_map=scenario.alias_map,
)

print(f"\n{'✅ PASSED' if passed else '❌ FAILED'}: {passed}")
print(f"REASON: {reason}")
print()

if not passed:
    print("=" * 70)
    print("MANUAL ANALYSIS - Why did it fail?")
    print("=" * 70)

    # Check for banned words
    model_lower = model_output.lower()
    found_banned = [word for word in scenario.banned if word.lower() in model_lower]
    if found_banned:
        print(f"❌ Contains banned words: {found_banned}")
    else:
        print("✅ No banned words found")

    # Check for facts
    print("\nFact checking:")
    for fact in scenario.facts:
        if fact.lower() in model_lower:
            print(f"  ✅ Found: {fact}")
        else:
            # Check aliases
            aliases = scenario.alias_map.get(fact, [])
            found_alias = any(alias.lower() in model_lower for alias in aliases)
            if found_alias:
                print(f"  ✅ Found via alias: {fact}")
            else:
                print(f"  ❌ MISSING: {fact}")

    # Check JSON format
    if model_output.strip().startswith("{"):
        print("\n✅ Output appears to be JSON format")
    else:
        print("\n❌ Output is NOT JSON format (should be JSON)")

    # Check word count
    word_count = len(model_output.split())
    if word_count <= 16:
        print(f"\n✅ Word count OK: {word_count} words (≤16)")
    else:
        print(f"\n❌ Word count TOO HIGH: {word_count} words (max: 16)")

print("\n" + "=" * 70)
print("DEBUG COMPLETE")
print("=" * 70)
