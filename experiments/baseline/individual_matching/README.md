# Baseline Experiment: Individual Matching vs Merged PTA

## Overview

This experiment establishes a **baseline** by comparing two approaches for scoring agent trajectories:

| Approach | Description |
|----------|-------------|
| **Individual Matching** (baseline) | Match each candidate against **each** training trace individually, then average the scores across all matches. |
| **Merged PTA Matching** (proposed) | Merge all training traces into a single ground-truth PTA, then match each candidate against the merged PTA. |

The goal is to demonstrate that **merging PTAs before matching** produces better pass/fail discrimination than simply averaging individual match scores.

## Why Merging Should Win

Individual matching has key limitations:

1. **Single-path bias**: Each individual trace represents *one* valid solution path. A correct candidate that follows a *different* valid path gets a low score against traces that don't share that path.
2. **Averaging dilutes signal**: Averaging scores across individual matches mixes high-signal matches with low-signal ones, reducing discriminative power.
3. **No solution-space modeling**: Individual traces cannot represent the branching solution space — only merging captures that multiple approaches are valid.

Merging fixes these by creating a unified DAG that accepts *any* valid path, so a correct candidate always finds a high-coverage alignment regardless of which specific solution strategy it used.

## Usage

```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate      # Linux/Mac

# Run the baseline experiment on OpenHands data
python experiments/baseline/individual_matching/run_baseline_experiment.py ^
    C:\path\to\openhands-swebench ^
    --merge-count 3 ^
    --test-pass-count 1 ^
    --test-fail-count 1 ^
    --output-dir C:\path\to\openhands-swebench\__experiment_results ^
    -v
```

### Arguments

| Argument | Description |
|----------|-------------|
| `data_dir` | Directory with trajectory data (positional) |
| `--output-dir` | Output directory (default: `data_dir/__baseline_experiment`) |
| `--merge-count N` | Number of passed trajectories for training (>= 2) |
| `--test-pass-count M` | Number of held-out passed trajectories for testing (>= 1) |
| `--test-fail-count K` | Number of failed trajectories for testing (>= 1) |
| `--seed` | Random seed for reproducibility (default: 42) |
| `--use-llm` | Enable LLM-backed semantic equivalence |
| `-v, --verbose` | Verbose logging |

## Output

- `baseline_experiment_results.json` — Full results with both approaches' metrics
- `baseline_comparison_report.html` — Interactive comparison report with:
  - Side-by-side metric tables (Individual vs Merged)
  - Score distribution histograms for both approaches
  - Per-task breakdown
  - Confusion matrices

## Key Metrics

- **AUROC** — Area Under ROC Curve (discrimination ability)
- **KS-Statistic** — Kolmogorov-Smirnov separation between pass/fail distributions
- **Accuracy / F1 / Precision / Recall** — Classification performance at optimal threshold
