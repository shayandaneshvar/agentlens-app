# Merge Count Study

## What

Studies how the **number of merged trajectories (k)** affects the discriminative power of the resulting PTA.

## Why

When building a ground-truth PTA, you must choose how many passing trajectories to merge. Too few may under-represent valid behaviors; too many may over-generalize. This experiment answers: **"What is the optimal k?"**

## How

1. For each eligible task, **fix the test set once** — the same passed + failed test trajectories are used at every merge count, so the *only* variable is k.
2. Vary k from `--min-merge` to `--max-merge`.
3. At each k, optionally draw `--resamples` random subsets of size k from the training pool, merge each, and score the fixed test set.
4. Aggregate AUROC, accuracy, and score gap (pass mean − fail mean) across tasks and resamples.
5. Output a summary table and an interactive Plotly HTML chart.

## Usage

```bash
# Activate virtual environment first
.\.venv\Scripts\Activate.ps1

# Basic run (no resampling)
python experiments/merge_count_study/run_merge_count_study.py <data_dir> \
    --min-merge 2 --max-merge 7

# With resampling for error bars
python experiments/merge_count_study/run_merge_count_study.py <data_dir> \
    --min-merge 2 --max-merge 7 --resamples 5

# Custom test set and output
python experiments/merge_count_study/run_merge_count_study.py <data_dir> \
    --min-merge 2 --max-merge 7 \
    --test-pass-count 3 --test-fail-count 3 \
    --resamples 3 --output-dir path/to/output
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `data_dir` | *(required)* | Path to dataset directory containing task folders |
| `--min-merge` | `2` | Minimum merge count to test |
| `--max-merge` | `7` | Maximum merge count to test |
| `--test-pass-count` | `3` | Passed trajectories held out for testing (fixed) |
| `--test-fail-count` | `3` | Failed trajectories held out for testing (fixed) |
| `--resamples` | `1` | Random subsets per merge count (1 = no resampling) |
| `--seed` | `42` | Random seed |
| `--output-dir` | `experiments/results/merge_count_study` | Output directory |

## Eligibility

A task needs at least `max_merge + test_pass_count` passed and `test_fail_count` failed trajectories.

## Output

```
<output_dir>/
├── merge_count_study_results.json    # Per-run data + aggregated points
├── merge_count_study_chart.html      # Interactive Plotly report
└── <task_name>/                      # Per-task PTA artifacts
```

### Charts in the HTML report

- **AUROC vs Merge Count** — structural / process / combined with error bars
- **Accuracy vs Merge Count** — same breakdown
- **Score Gap vs Merge Count** — pass mean − fail mean
- **Individual Run Scatter** — every task × resample AUROC as a dot
