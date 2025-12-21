"""
Experimental REINFORCE Training Script for Phase 2 Long Scenarios
Not required for core Anthropic evaluation/Best-of-N. Uses local HF models.

This script trains a language model using REINFORCE to compress
long documents (2000+ characters) while preserving all facts.

Usage:
    python experiments/train_reinforce.py --scenario medical_long --epochs 50
    python experiments/train_reinforce.py --scenarios medical_long business_long legal_long --epochs 100
    python experiments/train_reinforce.py --scenario economics --epochs 3 --device cpu --reward-scheme dense --max-length 96
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import yaml

from ace_task.algorithms.reinforce import REINFORCEConfig, REINFORCETrainer
from ace_task.algorithms.rewards import create_reward_function
from ace_task.scenarios import get_scenario


def build_prompt(scenario, max_chars: int, word_cap: int) -> str:
    """
    Build a VERY compact prompt to fit GPT-2's 1024 token limit.

    Key constraint: prompt + generation must stay under 1024 tokens total.
    With 30 facts, we need to be extremely concise.
    """
    # Only list first 10 facts as examples to save space
    facts_sample = scenario.facts[:10]
    facts_json = json.dumps(facts_sample)

    # Ultra-minimal prompt - just sample facts and format
    prompt = f"""Compress text to JSON. Include facts: {facts_json}... (and {len(scenario.facts)-10} more)

Format: {{"rewrite":"compressed text","preserved_facts":[...],"at_risk_facts":[],"key_insight":"preserving details prevents collapse","delta_update":"preservation maintains fidelity"}}

Max {word_cap} words:"""

    return prompt


def train_scenario(
    scenario_name: str,
    model_name: str,
    epochs: int,
    eval_every: int,
    save_dir: str,
    device: str,
    config: dict,
    reward_scheme: str,
    max_length: int,
):
    """Train REINFORCE on a single scenario."""
    print("\n" + "=" * 80)
    print(f"TRAINING: {scenario_name}")
    print("=" * 80)

    # Load scenario
    scenario = get_scenario(scenario_name)

    # Get scenario-specific config
    scenario_config = config["scenarios"].get(scenario_name, {})
    word_cap = scenario_config.get("word_cap", config["grader"]["word_cap"])
    max_chars = int(len(scenario.original) * scenario_config.get("concision_limit", 0.60))

    print(f"\nScenario: {scenario_name}")
    print(f"Original length: {len(scenario.original)} chars")
    print(f"Target: {max_chars} chars, {word_cap} words")
    print(f"Facts to preserve: {len(scenario.facts)}")

    # Create reward function (respects configured reward scheme)
    reward_fn = create_reward_function(scenario, reward_scheme=reward_scheme)

    # Build prompt
    prompt = build_prompt(scenario, max_chars, word_cap)

    print(f"\nPrompt length: {len(prompt)} characters")
    print(f"Model: {model_name}")
    print(f"Device: {device}")

    # Initialize REINFORCE trainer
    reinforce_config = REINFORCEConfig(
        learning_rate=float(config["rl"]["reinforce"]["learning_rate"]),
        gamma=float(config["rl"]["reinforce"]["gamma"]),
        baseline_type=config["rl"]["reinforce"]["baseline"],
        max_length=max_length,
        temperature=1.0,
    )

    trainer = REINFORCETrainer(
        model_name=model_name,
        reward_fn=reward_fn,
        config=reinforce_config,
        device=device,
    )

    # Training loop
    print(f"\n{'=' * 80}")
    print(f"STARTING TRAINING: {epochs} epochs")
    print(f"{'=' * 80}\n")

    training_history = []

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 40)

        # Training step
        metrics = trainer.train_step(prompt, verbose=True)
        training_history.append(metrics)

        # Evaluation
        if epoch % eval_every == 0 or epoch == epochs:
            print(f"\n{'='*40}")
            print(f"EVALUATION at epoch {epoch}")
            print(f"{'='*40}")

            eval_results = trainer.evaluate(prompt, n_samples=5)

            print(f"Mean reward: {eval_results['mean_reward']:.2f}")
            print(f"Max reward: {eval_results['max_reward']:.2f}")
            print(f"Success rate: {eval_results['success_rate']*100:.1f}%")

            # Show best sample
            best_idx = eval_results['rewards'].index(eval_results['max_reward'])
            best_sample = eval_results['samples'][best_idx]
            print(f"\nBest sample (reward={eval_results['max_reward']:.2f}):")
            print(f"{best_sample[:200]}..." if len(best_sample) > 200 else best_sample)

            # Save metrics
            training_history[-1]["eval"] = eval_results

    # Save final model
    save_path = Path(save_dir) / scenario_name / f"epoch_{epochs}"
    save_path.mkdir(parents=True, exist_ok=True)
    trainer.save(str(save_path))

    # Save training history
    history_path = save_path / "training_history.json"
    with open(history_path, "w") as f:
        # Convert non-serializable items
        serializable_history = []
        for entry in training_history:
            serializable_entry = {
                k: v for k, v in entry.items()
                if k not in ["generated_text"]  # Skip large text for JSON
            }
            if "eval" in entry:
                serializable_entry["eval"] = {
                    "mean_reward": entry["eval"]["mean_reward"],
                    "max_reward": entry["eval"]["max_reward"],
                    "success_rate": entry["eval"]["success_rate"],
                }
            serializable_history.append(serializable_entry)

        json.dump({
            "scenario": scenario_name,
            "model": model_name,
            "epochs": epochs,
            "history": serializable_history
        }, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"TRAINING COMPLETE: {scenario_name}")
    print(f"Final avg reward: {trainer.total_reward / trainer.step:.2f}")
    print(f"Model saved to: {save_path}")
    print(f"{'=' * 80}\n")

    return training_history


def main():
    parser = argparse.ArgumentParser(description="Train REINFORCE on long scenarios")
    parser.add_argument(
        "--scenario",
        type=str,
        help="Single scenario to train on (e.g., medical_long)"
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        help="Multiple scenarios to train on"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2-medium",
        help="Model to use (default: gpt2-medium)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=10,
        help="Evaluate every N epochs (default: 10)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cpu or cuda)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/reinforce",
        help="Directory to save models"
    )
    parser.add_argument(
        "--reward-scheme",
        type=str,
        choices=["binary", "dense"],
        default=None,
        help="Reward scheme to use (default: from config.yaml rl.reward_scheme).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=96,
        help="Maximum generation length used by REINFORCEConfig (caps max_new_tokens in generate()).",
    )

    args = parser.parse_args()

    # Load config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Determine scenarios to train
    if args.scenario:
        scenarios = [args.scenario]
    elif args.scenarios:
        scenarios = args.scenarios
    else:
        # Default: all Phase 2 long scenarios
        scenarios = ["medical_long", "business_long", "legal_long"]

    reward_scheme = args.reward_scheme or config["rl"].get("reward_scheme", "binary")

    print("\n" + "=" * 80)
    print("REINFORCE TRAINING - PHASE 2")
    print("=" * 80)
    print(f"Scenarios: {scenarios}")
    print(f"Model: {args.model}")
    print(f"Epochs per scenario: {args.epochs}")
    print(f"Device: {args.device}")
    print(f"Save directory: {args.save_dir}")
    print(f"Reward scheme: {reward_scheme}")
    print(f"Max generation length (max_length): {args.max_length}")
    print("=" * 80)

    # Train each scenario
    all_results = {}
    for scenario_name in scenarios:
        history = train_scenario(
            scenario_name=scenario_name,
            model_name=args.model,
            epochs=args.epochs,
            eval_every=args.eval_every,
            save_dir=args.save_dir,
            device=args.device,
            config=config,
            reward_scheme=reward_scheme,
            max_length=args.max_length,
        )
        all_results[scenario_name] = history

    # Save overall results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(args.save_dir) / f"training_results_{timestamp}.json"

    with open(results_path, "w") as f:
        json.dump({
            "model": args.model,
            "epochs": args.epochs,
            "scenarios": scenarios,
            "device": args.device,
            "timestamp": timestamp,
        }, f, indent=2)

    print("\n" + "=" * 80)
    print("ALL TRAINING COMPLETE!")
    print(f"Results saved to: {results_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
