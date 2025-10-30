"""
Evaluator:
- Loads ANTHROPIC_API_KEY from .env (python-dotenv).
- Builds a user message from prompt.txt + fixtures (ORIGINAL/FACTS/BANNED) and injects a per-run HARD LIMIT (MAX_CHARS).
- Calls Anthropic once per run (no tools) with modest randomness so outputs vary across runs.
- Parses the model's text output and feeds it to the deterministic grader.
- Repeats N times and prints pass-rate.

Run:
  python -m ace_task.evaluate --runs 10 --model claude-3-5-haiku-latest
"""

from __future__ import annotations
import argparse
import json
import os
import random
from math import floor

from dotenv import load_dotenv
from anthropic import Anthropic
from .data import ORIGINAL, FACTS, BANNED
from .grader import grade

load_dotenv()
if not os.getenv("ANTHROPIC_API_KEY"):
    raise EnvironmentError("Missing ANTHROPIC_API_KEY in .env")

PROMPT_PATH = __package__.replace(".", "/") + "/prompt.txt"


def build_user_message(max_chars: int, max_words: int) -> str:
    """Combine the prompt spec with fixtures and numeric caps."""
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        task_text = f.read()

    facts_lines = "\n".join(f"- {f}" for f in FACTS)
    banned_lines = "\n".join(f"- {b}" for b in sorted(BANNED))

    hard_limit = (
        f"\nHARD LIMITS FOR THIS RUN:\n"
        f"- MAX_CHARS for rewrite: {max_chars}\n"
        f"- MAX_WORDS for rewrite: {max_words}\n"
        f"If your rewrite exceeds {max_chars} characters OR {max_words} words, the submission FAILS.\n"
    )

    return (
        f"{task_text}\n"
        f"{hard_limit}\n"
        f"ORIGINAL:\n{ORIGINAL}\n\n"
        f"FACTS:\n{facts_lines}\n\n"
        f"BANNED:\n{banned_lines}\n"
    )


def run_once(client: Anthropic, model: str, max_chars: int, max_words: int) -> str:
    """Send one prompt to the model and return its text response."""
    msg = client.messages.create(
        model=model,
        max_tokens=400,
        temperature=0.45,  # modest variability to avoid 0% / 100% determinism
        top_p=0.9,
        system=(
            "Output ONLY strict JSON (no prose). Keep 'rewrite' within the provided character/word limits. "
            "Preserve all facts and numbers; avoid banned terms and extra claims."
        ),
        messages=[{"role": "user", "content": build_user_message(max_chars, max_words)}],
    )
    return "".join(c.text for c in msg.content if c.type == "text").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10, help="Number of independent attempts")
    ap.add_argument("--model", type=str, default="claude-3-5-haiku-latest")
    args = ap.parse_args()

    max_chars = floor(len(ORIGINAL) * 0.60)  # 60% limit (matches grader)

    client = Anthropic()
    random.seed()

    passes = 0
    for i in range(1, args.runs + 1):
        # small per-run variation to induce phrasing changes, but keep tight
        max_words = random.choice([12, 14, 16])
        print(f"\nRunning evaluation {i}/{args.runs}... (MAX_WORDS={max_words})")
        out = run_once(client, args.model, max_chars, max_words)

        # Debug: show ratio if the output is JSON with a rewrite
        try:
            obj = json.loads(out)
            if isinstance(obj, dict) and "rewrite" in obj:
                rw = obj["rewrite"]
                ratio = len(rw) / max(1, len(ORIGINAL))
                print(
                    f"DEBUG: len(rewrite)={len(rw)}, len(original)={len(ORIGINAL)}, ratio={ratio:.2f}"
                )
        except Exception:
            pass

        ok, msg = grade(ORIGINAL, FACTS, BANNED, out)
        print(f"Run {i}: {'✅ PASS' if ok else '❌ FAIL'} - {msg}")
        passes += int(ok)

    rate = passes / args.runs
    print("\n" + "=" * 60)
    print(f"Passed: {passes}/{args.runs} | Pass rate: {rate:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
