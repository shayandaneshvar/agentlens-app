# Merge Order (Combination & Permutation) Study

## What

A **case-study** experiment that fixes k and investigates two sources of variance:

1. **Combination variance** — does it matter *which* k trajectories you pick?
2. **Permutation variance** — does it matter *in what order* you merge them?

## Why

The merge-count study tells you the optimal k, but two questions remain:
- Are some subsets of trajectories better than others at the same k?
- Is the SDK merge operation order-sensitive (i.e., does `merge([A, B, C])` differ from `merge([C, A, B])`)?

This experiment quantifies both effects and decomposes the total variance into **between-combination** (subset choice) and **within-combination** (ordering) components.

## How

1. Point at a **single task directory** containing pass/fail trajectory folders/zips.
2. Fix a held-out test set (same as other experiments).
3. From the remaining pool, enumerate (or sample) combinations of size k.
4. For each combination, enumerate (or sample) permutations.
5. For each (combination, permutation): merge sequentially via `trace_api.merge()` → score test set.
6. Aggregate:
   - **Between-combination σ** = std of per-combination mean AUROCs
   - **Within-combination σ** = mean of per-combination AUROC stds
7. Output a summary table and an interactive Plotly HTML report.

## Usage

```bash
# Activate virtual environment first
.\.venv\Scripts\Activate.ps1

# Run on a single task directory
python experiments/merge_order_study/run_merge_order_study.py <task_dir> --k 4

# Example with real path
python experiments/merge_order_study/run_merge_order_study.py \
    "C:\path\to\data\python_refactor" \
    --k 4 --test-pass-count 3 --test-fail-count 3

# Exhaustive (all combos & perms)
python experiments/merge_order_study/run_merge_order_study.py <task_dir> \
    --k 3 --max-combinations 0 --max-permutations 0

# Sampled (cap combos and perms for speed)
python experiments/merge_order_study/run_merge_order_study.py <task_dir> \
    --k 4 --max-combinations 10 --max-permutations 6 \
    --output-dir path/to/output
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `task_dir` | *(required)* | Path to a **single task** directory with trajectory folders/zips |
| `--k` | `4` | Fixed merge count |
| `--test-pass-count` | `3` | Passed trajectories held out for testing |
| `--test-fail-count` | `3` | Failed trajectories held out for testing |
| `--max-combinations` | `10` | Max subsets to evaluate (0 = all) |
| `--max-permutations` | `6` | Max orderings per subset (0 = all) |
| `--seed` | `42` | Random seed |
| `--output-dir` | `experiments/results/merge_order_study` | Output directory |

## Input

Unlike the other experiments which take a top-level data directory, this takes a **single task directory**:

```
<task_dir>/
├── task-logs-model1-pass.zip
├── task-logs-model2-pass.zip
├── task-logs-model3-fail.zip
└── ...
```

## Output

```
<output_dir>/
├── merge_order_study_results.json    # Every run's scores, trajectory IDs, combo keys
├── merge_order_study_chart.html      # Interactive Plotly report
└── <task_name>/                      # PTA artifacts
```

### Charts in the HTML report

- **Variance Decomposition** — grouped bar chart: between-combo σ vs within-combo σ (structural & combined)
- **Per-Combination Box Plots** — AUROC distributions across permutations for each combination
- **Summary Table** — all metrics at a glance
