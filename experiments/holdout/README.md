# Holdout Validation Experiment

## What

A rigorous **train/test split** evaluation of merged PTAs.  
Instead of building the merged PTA from *all* passed trajectories and then evaluating on the same data (which can overfit), this experiment holds out a portion of passed and failed trajectories exclusively for testing.

## Why

Provides an **honest, unbiased** estimate of how well a merged PTA discriminates between pass and fail trajectories — the gold-standard evaluation for publication-quality results.

## How

1. For each task with enough trajectories, split the data:
   - **Train set**: `merge_count` passed trajectories → merged into a single ground-truth PTA  
   - **Test set**: `test_pass_count` passed + `test_fail_count` failed trajectories (never seen during merging)
2. Score every test trajectory against the merged PTA:
   - **Structural coverage** — subsequence match of tool-call states  
   - **Process coverage** — fraction of required tools present  
   - **Combined** — average of the two  
3. Compute classification metrics (AUROC, KS, F1, accuracy, confusion matrix) across tasks.
4. Generate a detailed HTML report.

## Usage

```bash
# Activate virtual environment first
.\.venv\Scripts\Activate.ps1            # Windows
# source .venv/bin/activate             # Linux/Mac

# Basic run
python experiments/holdout/run_holdout_experiment.py <data_dir> \
    --merge-count 6 --test-pass-count 3 --test-fail-count 3

# With LLM-based semantic comparison
python experiments/holdout/run_holdout_experiment.py <data_dir> \
    --merge-count 6 --test-pass-count 3 --test-fail-count 3 --use-llm

# Custom output directory
python experiments/holdout/run_holdout_experiment.py <data_dir> \
    --merge-count 6 --test-pass-count 3 --test-fail-count 3 \
    --output-dir path/to/output
```

## Key Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `data_dir` | ✅ | — | Path to dataset directory containing task folders |
| `--merge-count` | ✅ | — | Number of passed trajectories used for merging |
| `--test-pass-count` | ✅ | — | Number of passed trajectories held out for testing |
| `--test-fail-count` | ✅ | — | Number of failed trajectories held out for testing |
| `--use-llm` | | `False` | Enable LLM-backed semantic equivalence |
| `--structural-threshold` | | auto | Fixed threshold for structural metric (0.0–1.0) |
| `--process-threshold` | | auto | Fixed threshold for process metric |
| `--combined-threshold` | | auto | Fixed threshold for combined metric |
| `--output-dir` | | `<data_dir>/_holdout_experiment` | Output directory |
| `-v, --verbose` | | `False` | Verbose logging |

## Eligibility

A task is included only if it has at least `merge_count + test_pass_count` passed **and** `test_fail_count` failed trajectories.

## Output

```
<output_dir>/
├── holdout_experiment_results.json   # Full per-trajectory and aggregate metrics
└── holdout_experiment_report.html    # Interactive HTML report with charts
```
