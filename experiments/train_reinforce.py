"""
REINFORCE Training Script for Phase 2 Long Scenarios

This script trains a language model using REINFORCE to compress
long documents (2000+ characters) while preserving all facts.

Usage:
    python experiments/train_reinforce.py --scenario medical_long --epochs 50
    python experiments/train_reinforce.py --scenarios medical_long business_long legal_long --epochs 100
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import yaml

from ace_task.scenarios import get_scenario
from ace_task.algorithms.rewards import create_reward_function
from ace_task.algorithms.reinforce import REINFORCETrainer, REINFORCEConfig


def build_prompt(scenario, max_chars: int, word_cap: int) -> str:
    """
    Build a compact prompt that fits within GPT-2's 1024 token limit.

    We don't include the full original text since it can be 2500+ chars.
    Instead, we just give the facts to include and constraints.
    """
    facts_json = json.dumps(scenario.facts)

    # Truncate original to first 400 chars for context (optional)
    original_preview = scenario.original[:400] + "..." if len(scenario.original) > 400 else scenario.original

    prompt = f"""Compress this text preserving ALL facts.

Preview: {original_preview}

Facts (must include ALL): {facts_json}

Generate JSON:
{{"rewrite": "compressed text here", "preserved_facts": {facts_json}, "at_risk_facts": [], "key_insight": "preserving quantitative details prevents context collapse", "delta_update": "accurate preservation maintains semantic fidelity"}}

Rules: max {max_chars} chars, {word_cap} words. JSON only:"""

    return prompt


def train_scenario(
    scenario_name: str,
    model_name: str,
    epochs: int,
    eval_every: int,
    save_dir: str,
    device: str,
    config: dict,
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

    # Create reward function
    reward_fn = create_reward_function(scenario, reward_scheme="binary")

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
        max_length=512,
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

    print("\n" + "=" * 80)
    print("REINFORCE TRAINING - PHASE 2")
    print("=" * 80)
    print(f"Scenarios: {scenarios}")
    print(f"Model: {args.model}")
    print(f"Epochs per scenario: {args.epochs}")
    print(f"Device: {args.device}")
    print(f"Save directory: {args.save_dir}")
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
