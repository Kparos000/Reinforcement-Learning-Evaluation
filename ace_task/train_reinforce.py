"""
Minimal REINFORCE training loop wired to the ACE task.

This CLI:
- Loads a scenario (ORIGINAL/FACTS/BANNED) and builds the prompt from prompt.txt
- Wraps the deterministic grader as a reward function
- Runs REINFORCE to generate outputs, score them, and update the model
- Logs reward and running pass-rate for quick feedback

Usage:
  python -m ace_task.train_reinforce --model gpt2 --steps 10 --scenario report_long
"""

from __future__ import annotations

import argparse
from typing import List

import torch

from ace_task.algorithms.reinforce import REINFORCEConfig, REINFORCETrainer
from ace_task.algorithms.rewards import create_reward_function
from ace_task.prompting import build_user_message, compute_limits
from ace_task.scenarios import get_scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model on ACE with REINFORCE.")
    parser.add_argument("--model", type=str, default="gpt2", help="HF model name (e.g., gpt2, gpt2-medium)")
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of REINFORCE steps to run (each step = 1 sample + update).",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="report_long",
        help="Scenario to train against (see ace_task.scenarios.get_scenario).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (cpu or cuda).",
    )
    parser.add_argument(
        "--reward-scheme",
        type=str,
        choices=["binary", "dense"],
        default="binary",
        help="Reward style used by the grader.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=5,
        help="How often to print reward/pass-rate metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = get_scenario(args.scenario)

    # Build prompt and grading limits from the canonical task spec
    concision_limit = getattr(scenario, "concision_limit", None) or 0.60
    max_chars, max_words = compute_limits(
        scenario,
        concision_limit=concision_limit,
        word_cap=getattr(scenario, "word_cap", None),
    )
    prompt = build_user_message(max_chars, max_words, scenario)

    grader_config = {
        "alias_map": getattr(scenario, "alias_map", None),
        "concision_limit": concision_limit,
        "word_cap": max_words,
    }
    reward_fn = create_reward_function(
        scenario,
        reward_scheme=args.reward_scheme,
        grader_config=grader_config,
    )

    trainer = REINFORCETrainer(
        model_name=args.model,
        reward_fn=reward_fn,
        config=REINFORCEConfig(max_length=max_chars + max_words),
        device=args.device,
    )

    print(f"\nScenario: {scenario.name} | Domain: {scenario.domain} | Difficulty: {scenario.difficulty}")
    print(f"Original length: {len(scenario.original)} chars | Facts: {len(scenario.facts)} | Banned: {len(scenario.banned)}")
    print(f"Limits: max_chars={max_chars}, max_words={max_words}")
    print(f"Model: {args.model} | Device: {args.device}")
    print(f"Reward scheme: {args.reward_scheme}")
    print("=" * 80)

    rewards: List[float] = []
    pass_count = 0
    for step in range(1, args.steps + 1):
        metrics = trainer.train_step(prompt, verbose=False)
        rewards.append(metrics["reward"])

        # Binary rewards use 1.0 for pass; dense treats >=0.5 as success proxy
        pass_count += int(metrics["reward"] >= 0.5)

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            pass_rate = pass_count / step
            print(
                f"[step {step:04d}/{args.steps}] "
                f"reward={metrics['reward']:.3f} "
                f"avg_reward={metrics['avg_reward']:.3f} "
                f"pass_rate={pass_rate:.1%} "
                f"baseline={metrics['baseline']:.3f} "
                f"tokens={metrics['num_tokens']}"
            )

            snippet = metrics["generated_text"].replace("\n", " ")
            if len(snippet) > 160:
                snippet = snippet[:160] + "..."
            print(f"sample: {snippet}")


if __name__ == "__main__":
    main()
