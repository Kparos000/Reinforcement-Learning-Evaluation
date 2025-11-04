# Experiments

This directory contains experiment scripts and notebooks for running and analyzing RL algorithms.

## Structure

```
experiments/
├── run_best_of_n.py        # Run Best-of-N ablation study
├── visualize_results.py     # Generate plots from results
├── notebooks/               # Jupyter notebooks for analysis
└── README.md                # This file
```

## Running Experiments

### Best-of-N Ablation Study

Run the full ablation study to test different values of N:

```bash
# Set your API key
export ANTHROPIC_API_KEY='your-key-here'

# Run with default settings (N ∈ {1,3,5,10,20})
python experiments/run_best_of_n.py

# Run on specific scenarios
python experiments/run_best_of_n.py --scenarios economics medical

# Custom N values
python experiments/run_best_of_n.py --n-values 1 5 10

# Dry run (no API calls)
python experiments/run_best_of_n.py --dry-run
```

Results are saved to `results/best_of_n/` as JSON files.

### Visualizing Results

Generate plots from experiment results:

```bash
# Visualize latest results
python experiments/visualize_results.py results/best_of_n/ablation_results_*.json

# Specify output directory
python experiments/visualize_results.py results/best_of_n/ablation_results_*.json --output-dir results/plots
```

Generates:
- `n_vs_success_rate.png` - Main ablation result
- `cost_analysis.png` - Cost-performance trade-off
- `efficiency_analysis.png` - Success rate per API call
- `summary_report.txt` - Text summary of findings

## Expected Results

Based on a ~40% baseline pass rate, we expect:

| N   | Expected Pass Rate | Cost   |
|-----|--------------------|--------|
| 1   | 40%                | 1×     |
| 3   | 78%                | 3×     |
| 5   | 92%                | 5×     |
| 10  | 99.4%              | 10×    |
| 20  | ~100%              | 20×    |

The sweet spot is typically N=5 for portfolio projects (great results, reasonable cost).

## Next Steps

After completing Best-of-N experiments:
1. **Phase 2**: Implement REINFORCE (policy gradient learning)
2. **Phase 3**: Implement PPO (advanced policy optimization)
3. **Final Demo**: Create comprehensive comparison across all three methods
