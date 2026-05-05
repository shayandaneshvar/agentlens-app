# PTA Experiments

This directory contains experiments for generating, merging, and evaluating Prefix Tree Acceptors (PTAs) from code-agent execution traces.

## Prerequisites

```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1            # Windows
# source .venv/bin/activate             # Linux/Mac

# Core dependencies
pip install tqdm numpy scikit-learn scipy

# Optional: LLM-based semantic comparison
pip install openai httpx
```

---

## Experiments

### 1. [Holdout Validation](holdout/)

> **Question:** Can a merged PTA reliably discriminate between pass and fail trajectories?

Rigorous train/test split evaluation. Merges a subset of passed trajectories into a ground-truth PTA, then scores held-out passed + failed trajectories to compute AUROC, accuracy, F1, and more.

```bash
python experiments/holdout/run_holdout_experiment.py <data_dir> \
    --merge-count 6 --test-pass-count 3 --test-fail-count 3
```

### 2. [Merge Count Study](merge_count_study/)

> **Question:** How many trajectories should we merge? Where are the diminishing returns?

Fixes the test set and varies the merge count k from `--min-merge` to `--max-merge`. Reports AUROC, accuracy, and score gap at each k with optional resampling for error bars.

```bash
python experiments/merge_count_study/run_merge_count_study.py <data_dir> \
    --min-merge 2 --max-merge 7 --resamples 3
```

### 3. [Merge Order Study](merge_order_study/) *(combination & permutation)*

> **Question:** For a fixed k, does it matter *which* trajectories we pick and *in what order* we merge them?

Case-study experiment on a single task. Enumerates (or samples) subsets of size k and different merge orderings, then decomposes variance into between-combination (subset choice) and within-combination (ordering) components.

```bash
python experiments/merge_order_study/run_merge_order_study.py <task_dir> \
    --k 4 --max-combinations 10 --max-permutations 6
```

---

## Supporting Scripts

| Script | Purpose |
|--------|---------|
| `metrics_utils.py` | Shared utilities for classification metrics (AUROC, KS, F1, etc.) and HTML report generation. Used by all three experiments. |
| `run_full_experiment.py` | Process all tasks at scale — generate individual PTAs and merge per task. |
| `compute_correlation_metrics.py` | Quick correlation analysis on existing experiment outputs (may overfit). |
| `create_visualizations.py` | Generate interactive D3.js PTA graph visualizations. |

---

## Typical Workflow

```bash
# 1. Check trajectory availability
python check_trajectory_requirements.py <data_dir> \
    --merge-count 6 --test-pass-count 3 --test-fail-count 3

# 2. Holdout evaluation (honest metrics)
python experiments/holdout/run_holdout_experiment.py <data_dir> \
    --merge-count 6 --test-pass-count 3 --test-fail-count 3

# 3. Find the optimal merge count
python experiments/merge_count_study/run_merge_count_study.py <data_dir> \
    --min-merge 2 --max-merge 7 --resamples 3

# 4. Deep-dive: subset & ordering effects for a specific task
python experiments/merge_order_study/run_merge_order_study.py <task_dir> \
    --k 4 --max-combinations 10 --max-permutations 6
```

---

## Understanding the Metrics

| Metric | Range | Interpretation |
|--------|-------|----------------|
| **AUROC** | 0–1 | Probability that a random pass trajectory scores higher than a random fail. 0.5 = random, 1.0 = perfect. |
| **KS-Statistic** | 0–1 | Maximum separation between pass/fail cumulative distributions. |
| **Accuracy** | 0–1 | Fraction correctly classified at optimal threshold. |
| **F1** | 0–1 | Harmonic mean of precision and recall. |
| **Score Gap** | any | Pass mean − fail mean. Positive = correct direction. |

### Three Matching Scores

1. **Structural coverage** — subsequence match of tool-call states against merged PTA paths.
2. **Process coverage** — fraction of required tools (tools in *every* ground-truth path) present in the candidate.
3. **Combined** — average of structural and process coverage.

---

## Output Directories (historical runs)

| Directory | Description |
|-----------|-------------|
| `debug_output/` | Debug outputs from initial testing |
| `full_output_v2/` | Full experiment run (v2) |
| `full_run_final/` | Final production run |
| `improved_v3/` | Improved merging algorithm (v3) |
| `improved_order_v8/` | Order-preserving improvements (v8) |
| `treesitter_v5/`, `v6/` | Tree-sitter based parsing |
| `treesitter_llm_v7/` | Tree-sitter + LLM hybrid |
| `multi_lang_v4/` | Multi-language experiments |
