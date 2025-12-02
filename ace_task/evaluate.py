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
from typing import Iterable

from anthropic import Anthropic
from dotenv import load_dotenv

from .grader import grade
from .scenarios import get_scenario

load_dotenv()

PROMPT_PATH = __package__.replace(".", "/") + "/prompt.txt"


def require_api_key() -> str:
    """Ensure ANTHROPIC_API_KEY is present for live evaluations."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing ANTHROPIC_API_KEY in environment or .env file")
    return api_key


def build_user_message(max_chars: int, max_words: int, scenario) -> str:
    """Combine the prompt spec with fixtures and numeric caps."""
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        task_text = f.read()

    facts_lines = "\n".join(f"- {f}" for f in scenario.facts)
    banned_lines = "\n".join(f"- {b}" for b in sorted(scenario.banned))

    hard_limit = (
        f"\nHARD LIMITS FOR THIS RUN:\n"
        f"- MAX_CHARS for rewrite: {max_chars}\n"
        f"- MAX_WORDS for rewrite: {max_words}\n"
        f"If your rewrite exceeds {max_chars} characters OR {max_words} words, the submission FAILS.\n"
    )

    return (
        f"{task_text}\n"
        f"{hard_limit}\n"
        f"ORIGINAL:\n{scenario.original}\n\n"
        f"FACTS:\n{facts_lines}\n\n"
        f"BANNED:\n{banned_lines}\n"
    )


def run_once(
    client: Anthropic,
    model: str,
    max_chars: int,
    max_words: int,
    scenario,
    temperature: float,
) -> str:
    """Send one prompt to the model and return its text response."""
    msg = client.messages.create(
        model=model,
        max_tokens=500,
        temperature=temperature,  # adjustable to tune pass-rate
        top_p=0.9,
        system=(
            "Output ONLY strict JSON (no prose). Keep 'rewrite' within the provided character/word limits. "
            "Preserve all facts and numbers; avoid banned terms and extra claims. "
            "Your key_insight MUST explicitly mention preserving quantitative details to avoid context collapse."
        ),
        messages=[{"role": "user", "content": build_user_message(max_chars, max_words, scenario)}],
    )
    return "".join(c.text for c in msg.content if c.type == "text").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10, help="Number of independent attempts")
    ap.add_argument("--model", type=str, default="claude-3-5-haiku-latest")
    ap.add_argument("--scenario", type=str, default="report_long", help="Scenario to evaluate")
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.60,
        help="Sampling temperature (higher = more diverse, lower = more deterministic)",
    )
    ap.add_argument(
        "--concision", type=float, default=None, help="Override concision limit (0-1). Default uses scenario length x 0.60."
    )
    ap.add_argument(
        "--word-cap",
        type=int,
        default=None,
        help="Override word cap used for prompt and grading. Default uses scenario.word_cap or 16.",
    )
    args = ap.parse_args()

    require_api_key()
    scenario = get_scenario(args.scenario)
    concision_limit = args.concision
    if concision_limit is None:
        concision_limit = getattr(scenario, "concision_limit", None) or 0.60
    max_chars = floor(len(scenario.original) * concision_limit)

    base_word_cap = args.word_cap or getattr(scenario, "word_cap", None) or 16

    client = Anthropic()
    random.seed()

    passes = 0
    for i in range(1, args.runs + 1):
        # small per-run variation to induce phrasing changes, but keep tight
        word_choices: Iterable[int] = (
            [base_word_cap]
            if base_word_cap <= 20
            else [base_word_cap - 10, base_word_cap, min(base_word_cap + 10, base_word_cap + 20)]
        )
        max_words = random.choice(list(word_choices))
        print(f"\nRunning evaluation {i}/{args.runs}... (MAX_WORDS={max_words})")
        out = run_once(
            client=client,
            model=args.model,
            max_chars=max_chars,
            max_words=max_words,
            scenario=scenario,
            temperature=args.temperature,
        )

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

        ok, msg = grade(
            original=scenario.original,
            facts=scenario.facts,
            banned=scenario.banned,
            model_text=out,
            alias_map=scenario.alias_map,
            concision_limit=concision_limit,
            word_cap=max_words,
        )
        print(f"Run {i}: {'PASS' if ok else 'FAIL'} - {msg}")
        passes += int(ok)

    rate = passes / args.runs
    print("\n" + "=" * 60)
    print(f"Passed: {passes}/{args.runs} | Pass rate: {rate:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
