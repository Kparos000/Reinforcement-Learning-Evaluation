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
import statistics
from typing import Dict, Iterable, List

from anthropic import Anthropic
from dotenv import load_dotenv

from .algorithms.rewards import dense_reward_from_grade
from .grader import grade_detailed
from .prompting import PROMPT_PATH as PROMPT_FILE, build_user_message, compute_limits
from .scenarios import get_scenario

load_dotenv()
PROMPT_PATH = str(PROMPT_FILE)


def require_api_key() -> str:
    """Ensure ANTHROPIC_API_KEY is present for live evaluations."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing ANTHROPIC_API_KEY in environment or .env file")
    return api_key


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


def _reward_stats(rewards: List[float]) -> Dict[str, float]:
    if not rewards:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": statistics.mean(rewards),
        "median": statistics.median(rewards),
        "min": min(rewards),
        "max": max(rewards),
    }


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
    max_chars, base_word_cap = compute_limits(
        scenario, concision_limit=concision_limit, word_cap=args.word_cap
    )

    client = Anthropic()
    random.seed()

    passes = 0
    rewards: List[float] = []
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
                ratio = len(rw) / max(1, len(scenario.original))
                print(
                    f"DEBUG: len(rewrite)={len(rw)}, len(original)={len(scenario.original)}, ratio={ratio:.2f}"
                )
        except Exception:
            pass

        grade = grade_detailed(
            original=scenario.original,
            facts=scenario.facts,
            banned=scenario.banned,
            model_text=out,
            alias_map=scenario.alias_map,
            concision_limit=concision_limit,
            word_cap=max_words,
        )
        reward = dense_reward_from_grade(grade)
        rewards.append(reward)
        print(f"Run {i}: {'PASS' if grade.passed else 'FAIL'} - reward={reward:.2f} - {grade.reason}")
        passes += int(grade.passed)

    rate = passes / args.runs
    stats = _reward_stats(rewards)
    print("\n" + "=" * 60)
    print(f"Passed: {passes}/{args.runs} | Pass rate: {rate:.2%}")
    print("Reward stats (dense):")
    print(f"  Mean:   {stats['mean']:.3f}")
    print(f"  Median: {stats['median']:.3f}")
    print(f"  Min:    {stats['min']:.3f}")
    print(f"  Max:    {stats['max']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
