#!/usr/bin/env python3
"""
Visualization script for Best-of-N ablation study results.

This script generates publication-quality plots from experiment results:
- N vs Pass Rate (main result)
- N vs Average Reward
- Cost-Performance Trade-off
- Sample Diversity Analysis

Usage:
    python experiments/visualize_results.py results/best_of_n/ablation_results_*.json
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Set publication-quality defaults
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["lines.markersize"] = 8


def load_results(results_file: str) -> dict:
    """Load experiment results from JSON file."""
    with open(results_file) as f:
        return json.load(f)


def aggregate_by_n(experiments: list[dict]) -> dict:
    """
    Aggregate experiment results by N value.

    Returns dict mapping N -> metrics (mean, std, etc.)
    """
    from collections import defaultdict

    by_n = defaultdict(list)

    for exp in experiments:
        n = exp["n"]
        by_n[n].append(
            {
                "best_reward": exp["best_reward"],
                "avg_reward": exp["avg_reward"],
                "success_rate": exp["success_rate"],
            }
        )

    # Compute statistics
    aggregated = {}
    for n, values in by_n.items():
        aggregated[n] = {
            "success_rate_mean": np.mean([v["success_rate"] for v in values]),
            "success_rate_std": np.std([v["success_rate"] for v in values]),
            "avg_reward_mean": np.mean([v["avg_reward"] for v in values]),
            "avg_reward_std": np.std([v["avg_reward"] for v in values]),
            "n_experiments": len(values),
        }

    return aggregated


def plot_n_vs_success_rate(aggregated: dict, output_path: Path):
    """Plot N vs Success Rate (main result)."""
    n_values = sorted(aggregated.keys())
    success_rates = [aggregated[n]["success_rate_mean"] * 100 for n in n_values]
    std_devs = [aggregated[n]["success_rate_std"] * 100 for n in n_values]

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        n_values,
        success_rates,
        yerr=std_devs,
        marker="o",
        capsize=5,
        capthick=2,
        label="Measured",
    )

    # Theoretical curve (if baseline p is known)
    # For now, just show the data
    plt.xlabel("N (Number of Samples)")
    plt.ylabel("Success Rate (%)")
    plt.title("Best-of-N Ablation Study: Impact of Sample Size")
    plt.grid(True, alpha=0.3)
    plt.xticks(n_values)
    plt.ylim(0, 105)

    # Add improvement annotations
    if len(n_values) >= 2:
        baseline = success_rates[0]
        best = success_rates[-1]
        improvement = best - baseline
        plt.annotate(
            f"+{improvement:.1f}% improvement",
            xy=(n_values[-1], best),
            xytext=(n_values[-1] - 1, best - 10),
            arrowprops=dict(arrowstyle="->", color="green", lw=2),
            fontsize=12,
            color="green",
        )

    plt.tight_layout()
    plt.savefig(output_path / "n_vs_success_rate.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_path / "n_vs_success_rate.pdf", bbox_inches="tight")
    print(f"✅ Saved: {output_path / 'n_vs_success_rate.png'}")


def plot_cost_analysis(aggregated: dict, output_path: Path):
    """Plot cost-performance trade-off."""
    n_values = sorted(aggregated.keys())
    success_rates = [aggregated[n]["success_rate_mean"] * 100 for n in n_values]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Success rate
    color = "tab:blue"
    ax1.set_xlabel("N (Number of Samples)")
    ax1.set_ylabel("Success Rate (%)", color=color)
    ax1.plot(n_values, success_rates, marker="o", color=color, label="Success Rate")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)

    # Cost (relative to N=1)
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Relative Cost (×N)", color=color)
    ax2.plot(n_values, n_values, marker="s", color=color, linestyle="--", label="Cost")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Best-of-N: Cost-Performance Trade-off")
    fig.tight_layout()
    plt.savefig(output_path / "cost_analysis.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_path / "cost_analysis.pdf", bbox_inches="tight")
    print(f"✅ Saved: {output_path / 'cost_analysis.png'}")


def plot_efficiency(aggregated: dict, output_path: Path):
    """Plot efficiency (success rate per API call)."""
    n_values = sorted(aggregated.keys())
    success_rates = [aggregated[n]["success_rate_mean"] for n in n_values]
    efficiency = [sr / n for sr, n in zip(success_rates, n_values)]

    plt.figure(figsize=(10, 6))
    plt.plot(n_values, efficiency, marker="o", color="purple")
    plt.xlabel("N (Number of Samples)")
    plt.ylabel("Efficiency (Success Rate / N)")
    plt.title("Best-of-N: Efficiency Analysis")
    plt.grid(True, alpha=0.3)
    plt.xticks(n_values)

    # Mark optimal N (highest efficiency)
    optimal_idx = np.argmax(efficiency)
    optimal_n = n_values[optimal_idx]
    plt.axvline(optimal_n, color="red", linestyle="--", alpha=0.5)
    plt.annotate(
        f"Optimal: N={optimal_n}",
        xy=(optimal_n, efficiency[optimal_idx]),
        xytext=(optimal_n + 1, efficiency[optimal_idx] - 0.02),
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red",
    )

    plt.tight_layout()
    plt.savefig(output_path / "efficiency_analysis.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_path / "efficiency_analysis.pdf", bbox_inches="tight")
    print(f"✅ Saved: {output_path / 'efficiency_analysis.png'}")


def generate_summary_report(results: dict, aggregated: dict, output_path: Path):
    """Generate a text summary report."""
    report = []
    report.append("=" * 60)
    report.append("BEST-OF-N ABLATION STUDY - SUMMARY REPORT")
    report.append("=" * 60)
    report.append("")

    report.append(f"Experiment Date: {results['timestamp']}")
    report.append(f"Scenarios: {', '.join(results['scenarios'])}")
    report.append(f"N Values Tested: {results['n_values']}")
    report.append(f"Total Experiments: {len(results['experiments'])}")
    report.append("")

    report.append("RESULTS BY N:")
    report.append("-" * 60)
    for n in sorted(aggregated.keys()):
        data = aggregated[n]
        report.append(f"\nN = {n}:")
        report.append(f"  Success Rate: {data['success_rate_mean']*100:.1f}% ± {data['success_rate_std']*100:.1f}%")
        report.append(f"  Avg Reward: {data['avg_reward_mean']:.3f} ± {data['avg_reward_std']:.3f}")
        report.append(f"  Experiments: {data['n_experiments']}")

    report.append("")
    report.append("KEY FINDINGS:")
    report.append("-" * 60)

    n_values = sorted(aggregated.keys())
    success_rates = [aggregated[n]["success_rate_mean"] for n in n_values]

    if len(n_values) >= 2:
        baseline = success_rates[0]
        best = success_rates[-1]
        improvement = (best - baseline) * 100
        report.append(f"• Baseline (N={n_values[0]}): {baseline*100:.1f}% success rate")
        report.append(f"• Best (N={n_values[-1]}): {best*100:.1f}% success rate")
        report.append(f"• Absolute improvement: +{improvement:.1f} percentage points")
        report.append(f"• Relative improvement: {(best/baseline - 1)*100:.1f}%")

    report.append("")
    report.append("=" * 60)

    report_text = "\n".join(report)
    report_file = output_path / "summary_report.txt"
    with open(report_file, "w") as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\n✅ Report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Best-of-N ablation results")
    parser.add_argument("results_file", type=str, help="Path to JSON results file")
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for plots (default: same as results file)",
    )

    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.results_file}")
    results = load_results(args.results_file)

    # Determine output directory
    if args.output_dir:
        output_path = Path(args.output_dir)
    else:
        output_path = Path(args.results_file).parent / "plots"

    output_path.mkdir(parents=True, exist_ok=True)

    # Aggregate data
    print("Aggregating experiment data...")
    aggregated = aggregate_by_n(results["experiments"])

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_n_vs_success_rate(aggregated, output_path)
    plot_cost_analysis(aggregated, output_path)
    plot_efficiency(aggregated, output_path)

    # Generate summary report
    generate_summary_report(results, aggregated, output_path)

    print(f"\n✅ All visualizations saved to: {output_path}")


if __name__ == "__main__":
    main()
