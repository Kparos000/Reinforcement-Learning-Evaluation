"""
REINFORCE Training with Gemma-2B for Phase 2 (Production)

This is the PRODUCTION implementation using Gemma-2B:
- 4GB model (vs 14GB Llama-2)
- 8,192 token context (vs 1,024 GPT-2)
- Fast training (~2-4 hours total)
- Quality instruction-following

Perfect balance for production UI deployment.

Usage:
    # Single scenario test (~30-45 min)
    python experiments/train_reinforce_gemma.py --scenario medical_long --epochs 50

    # Full training (~2-4 hours)
    python experiments/train_reinforce_gemma.py --scenarios medical_long business_long legal_long --epochs 50
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
    Build a complete prompt for Gemma-2B's 8k context window.

    No need for truncation - we have plenty of space!
    """
    facts_json = json.dumps(scenario.facts, indent=2)

    # Show first 500 chars of original for context
    original_preview = scenario.original[:500] + "..." if len(scenario.original) > 500 else scenario.original

    prompt = f"""<start_of_turn>user
You are an expert at compressing text while preserving all critical facts.

Original Text:
{original_preview}

Required Facts (ALL must be included in compression):
{facts_json}

Task: Generate a JSON object that compresses this text into {word_cap} words maximum, {max_chars} characters maximum, while preserving ALL facts above.

Output Format:
{{
  "rewrite": "compressed text with all facts preserved",
  "preserved_facts": {facts_json},
  "at_risk_facts": [],
  "key_insight": "preserving quantitative details prevents context collapse in specialized domains",
  "delta_update": "accurate fact preservation maintains semantic fidelity for reliable downstream reasoning"
}}

Critical Requirements:
1. Include ALL facts from the list above
2. Keep ALL numeric values exact (e.g., 127.50 not ~127)
3. Maximum {word_cap} words in the rewrite
4. Maximum {max_chars} characters in the rewrite
5. Use compact phrasing but remain clear
6. Output ONLY the JSON object, no explanations

Generate the JSON now:<end_of_turn>
<start_of_turn>model
"""

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
    print(f"TRAINING: {scenario_name} with Gemma-2B")
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
    print("Context window: 8,192 tokens (plenty of room!)")
    print(f"Device: {device}")

    # Initialize REINFORCE trainer with CPU-optimized settings
    reinforce_config = REINFORCEConfig(
        learning_rate=float(config["rl"]["reinforce"]["learning_rate"]),
        gamma=float(config["rl"]["reinforce"]["gamma"]),
        baseline_type=config["rl"]["reinforce"]["baseline"],
        max_length=400,  # Limit generation length (faster on CPU)
        temperature=0.3,  # Low temp = faster, more focused generation
    )

    print("\nInitializing Gemma-2B trainer...")
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
    best_reward = 0.0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 40)

        # Training step
        metrics = trainer.train_step(prompt, verbose=True)
        training_history.append(metrics)

        # Track best
        if metrics['reward'] > best_reward:
            best_reward = metrics['reward']
            print(f"[OK] New best reward: {best_reward:.2f}")

        # Evaluation
        if epoch % eval_every == 0 or epoch == epochs:
            print(f"\n{'='*40}")
            print(f"EVALUATION at epoch {epoch}")
            print(f"{'='*40}")

            eval_results = trainer.evaluate(prompt, n_samples=5)

            print(f"Mean reward: {eval_results['mean_reward']:.2f}")
            print(f"Max reward: {eval_results['max_reward']:.2f}")
            print(f"Success rate: {eval_results['success_rate']*100:.1f}%")
            print(f"Best overall: {best_reward:.2f}")

            # Show best sample
            best_idx = eval_results['rewards'].index(eval_results['max_reward'])
            best_sample = eval_results['samples'][best_idx]
            print(f"\nBest sample (reward={eval_results['max_reward']:.2f}):")
            print(f"{best_sample[:300]}..." if len(best_sample) > 300 else best_sample)

            # Save metrics
            training_history[-1]["eval"] = eval_results

            # Early stopping if consistently successful
            if eval_results['success_rate'] >= 0.8:
                print("\n[OK] Success rate >=80%! Training is effective.")

    # Save final model
    save_path = Path(save_dir) / "gemma-2b" / scenario_name / f"epoch_{epochs}"
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
            "final_success_rate": eval_results['success_rate'],
            "best_reward": best_reward,
            "history": serializable_history
        }, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"TRAINING COMPLETE: {scenario_name}")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Final success rate: {eval_results['success_rate']*100:.1f}%")
    print(f"Model saved to: {save_path}")
    print(f"{'=' * 80}\n")

    return training_history


def main():
    parser = argparse.ArgumentParser(description="Train REINFORCE with Gemma-2B (Production)")
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
        default="google/gemma-2b",
        help="Model to use (default: google/gemma-2b)"
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
    print("REINFORCEMENT LEARNING - PHASE 2 (PRODUCTION)")
    print("Gemma-2B: 4GB model with 8k context")
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
    results_path = Path(args.save_dir) / "gemma-2b" / f"training_summary_{timestamp}.json"

    with open(results_path, "w") as f:
        json.dump({
            "model": args.model,
            "epochs": args.epochs,
            "scenarios": scenarios,
            "device": args.device,
            "timestamp": timestamp,
            "total_scenarios": len(scenarios),
        }, f, indent=2)

    print("\n" + "=" * 80)
    print("ALL TRAINING COMPLETE!")
    print(f"Trained {len(scenarios)} scenarios successfully")
    print(f"Results saved to: {results_path}")
    print("\nModels ready for production UI integration!")
    print("=" * 80)


if __name__ == "__main__":
    main()
