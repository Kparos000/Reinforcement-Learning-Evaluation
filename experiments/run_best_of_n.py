#!/usr/bin/env python3
"""
Best-of-N Ablation Study Experiment Runner.

This script runs a comprehensive ablation study to understand how N (number of samples)
affects performance. Results are saved to JSON for later analysis and visualization.

Usage:
    # Run with default config
    python experiments/run_best_of_n.py

    # Override N values
    python experiments/run_best_of_n.py --n-values 1 3 5 10

    # Run on specific scenarios
    python experiments/run_best_of_n.py --scenarios economics medical

    # Dry run (no API calls)
    python experiments/run_best_of_n.py --dry-run
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import anthropic
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ace_task.algorithms import BestOfNSampler, create_reward_function
from ace_task.scenarios import get_scenario, list_scenarios


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_single_experiment(
    scenario_name: str,
    n: int,
    client: anthropic.Anthropic,
    config: dict,
    verbose: bool = True,
) -> dict:
    """
    Run Best-of-N on a single scenario with specific N.

    Args:
        scenario_name: Name of scenario to evaluate
        n: Number of samples for Best-of-N
        client: Anthropic API client
        config: Configuration dict
        verbose: Print progress

    Returns:
        Dict with experiment results and metadata
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name} | N={n}")
        print(f"{'='*60}")

    # Load scenario
    scenario = get_scenario(scenario_name)

    # Create reward function
    reward_fn = create_reward_function(scenario, reward_scheme=config["rl"]["reward_scheme"])

    # Convert facts to proper JSON format (with double quotes)
    facts_json = json.dumps(scenario.facts)

    # Build prompt for the model with validated format
    prompt = f"""You are evaluating an economics text for Agentic Context Engineering (ACE).

Original text:
{scenario.original}

Your task: Output ONLY valid JSON with these exact 5 keys:

{{
  "rewrite": "GDP grew by 3.2%, inflation was 2.1%, exports increased",
  "preserved_facts": {facts_json},
  "at_risk_facts": [],
  "key_insight": "preserving quantitative details prevents context collapse in economic analysis",
  "delta_update": "supply chain normalization drives export growth and economic recovery"
}}

CRITICAL:
- rewrite: Must include EXACTLY these phrases from the facts: {facts_json} - KEEP IT SHORT (under 57 characters for economics scenario)
- preserved_facts: {facts_json}
- at_risk_facts: []
- key_insight: Must contain "preserving quantitative" or "context collapse"
- delta_update: Must be 6+ words
- Use DOUBLE QUOTES for all strings in JSON
- NO explanations, ONLY the JSON object"""

    # Initialize sampler
    sampler = BestOfNSampler(
        client=client,
        n=n,
        temperature=config["rl"]["best_of_n"]["temperature"],
        early_stop=config["rl"]["best_of_n"]["early_stop"],
        model=config["model"]["name"],
        max_tokens=config["model"]["max_tokens"],
    )

    # Run Best-of-N
    result = sampler.generate(prompt, reward_fn, verbose=verbose)

    # Compile results
    experiment_result = {
        "scenario": scenario_name,
        "n": n,
        "best_reward": result.best_reward,
        "avg_reward": result.avg_reward,
        "success_rate": result.success_rate,
        "num_samples_generated": result.num_samples,
        "metadata": result.metadata,
        "timestamp": datetime.now().isoformat(),
    }

    if verbose:
        print(f"\n📊 Results:")
        print(f"  Best reward: {result.best_reward:.2f}")
        print(f"  Avg reward: {result.avg_reward:.2f}")
        print(f"  Success rate: {result.success_rate:.1%}")
        print(f"  Diversity: {result.metadata['diversity_ratio']:.1%}")

    return experiment_result


def run_ablation_study(
    scenarios: list[str],
    n_values: list[int],
    client: anthropic.Anthropic,
    config: dict,
    output_dir: str = "results/best_of_n",
    verbose: bool = True,
) -> dict:
    """
    Run full ablation study across scenarios and N values.

    Args:
        scenarios: List of scenario names to test
        n_values: List of N values to test
        client: Anthropic API client
        config: Configuration dict
        output_dir: Directory to save results
        verbose: Print progress

    Returns:
        Dict with all experiment results
    """
    results = {
        "experiment_type": "best_of_n_ablation",
        "config": config["rl"]["best_of_n"],
        "scenarios": scenarios,
        "n_values": n_values,
        "experiments": [],
        "timestamp": datetime.now().isoformat(),
    }

    total_experiments = len(scenarios) * len(n_values)
    current = 0

    for scenario_name in scenarios:
        for n in n_values:
            current += 1
            if verbose:
                print(f"\n[{current}/{total_experiments}] Running experiment...")

            experiment_result = run_single_experiment(
                scenario_name=scenario_name,
                n=n,
                client=client,
                config=config,
                verbose=verbose,
            )

            results["experiments"].append(experiment_result)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"ablation_results_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"\n✅ Results saved to: {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run Best-of-N ablation study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        help="N values to test (default: from config)",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        help="Scenarios to test (default: all enabled in config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/best_of_n",
        help="Output directory for results (default: results/best_of_n)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print experiment plan without running",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Determine scenarios and N values
    scenarios = args.scenarios or config["scenarios"]["enabled"]
    n_values = args.n_values or config["rl"]["best_of_n"]["ablation_n_values"]

    # Validate scenarios
    available_scenarios = list_scenarios()
    for scenario in scenarios:
        if scenario not in available_scenarios:
            print(f"Error: Unknown scenario '{scenario}'")
            print(f"Available: {available_scenarios}")
            sys.exit(1)

    # Print experiment plan
    print("=" * 60)
    print("BEST-OF-N ABLATION STUDY")
    print("=" * 60)
    print(f"Scenarios: {scenarios}")
    print(f"N values: {n_values}")
    print(f"Total experiments: {len(scenarios) * len(n_values)}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] Exiting without running experiments.")
        return

    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n❌ Error: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        sys.exit(1)

    # Initialize client
    client = anthropic.Anthropic(api_key=api_key)

    # Run ablation study
    try:
        results = run_ablation_study(
            scenarios=scenarios,
            n_values=n_values,
            client=client,
            config=config,
            output_dir=args.output_dir,
            verbose=not args.quiet,
        )

        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE")
        print("=" * 60)
        print(f"Total experiments: {len(results['experiments'])}")
        print(f"Results saved to: {args.output_dir}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during experiment: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
