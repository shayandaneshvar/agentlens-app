# Baseline Experiment: Embedding Similarity vs Merged PTA

## Overview

This experiment evaluates an **embedding similarity** baseline for trajectory scoring, comparing it against our proposed **Merged PTA Matching** approach.

| Approach | Description |
|----------|-------------|
| **Embedding Similarity** (baseline) | Convert each state to text, embed using dense (Azure OpenAI `text-embedding-3-large`) or sparse (TF-IDF) vectors, then compute BERTScore-style greedy alignment F1 between candidate and each training trace. Average across all training traces. |
| **Merged PTA Matching** (proposed) | Merge all training traces into a single ground-truth PTA, then match each candidate against the merged PTA via greedy subsequence-coverage. |

The goal is to test whether semantic/lexical similarity of state descriptions can discriminate pass from fail — or whether structural ordering (captured by PTA matching) is essential.

## Why PTA Matching Should Win

Embedding similarity has fundamental limitations:

1. **No ordering information**: BERTScore-style alignment is permutation-invariant — two traces with the same states in different orders get the same score.
2. **Surface overlap**: Failing traces routinely perform the same file reads and tool calls as passing traces. The lexical content of state descriptions is highly similar between pass and fail.
3. **No valid-path modelling**: Embeddings capture *what happened* but not *whether the sequence constitutes a valid solution path*.

## Usage

```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate      # Linux/Mac

# Run with Azure OpenAI embeddings (requires AZURE_OPENAI_* vars in .env)
python experiments/baseline/embedding_similarity/run_embedding_baseline.py ^
    C:\path\to\openhands-swebench ^
    --merge-count 4 ^
    --test-pass-count 2 ^
    --test-fail-count 2 ^
    --embedding-method azure ^
    -v

# Run with TF-IDF (no API key needed)
python experiments/baseline/embedding_similarity/run_embedding_baseline.py ^
    C:\path\to\openhands-swebench ^
    --merge-count 3 ^
    --test-pass-count 1 ^
    --test-fail-count 1 ^
    --embedding-method tfidf ^
    -v
```

### Arguments

| Argument | Description |
|----------|-------------|
| `data_dir` | Directory with trajectory data (positional) |
| `--output-dir` | Output directory (default: `data_dir/__embedding_baseline`) |
| `--merge-count N` | Number of passed trajectories for training (>= 2) |
| `--test-pass-count M` | Number of held-out passed trajectories for testing (>= 1) |
| `--test-fail-count K` | Number of failed trajectories for testing (>= 1) |
| `--embedding-method` | `azure` (text-embedding-3-large), `tfidf`, or `auto` (try azure, fall back to tfidf) |
| `--seed` | Random seed for reproducibility (default: 42) |
| `-v, --verbose` | Verbose logging |

### Embedding Methods

| Method | Dimensions | Requirements | Cost |
|--------|-----------|--------------|------|
| `azure` | 3072 (dense) | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` in `.env` | ~$0.23 per full run |
| `tfidf` | 5000 (sparse) | None (sklearn) | Free |
| `auto` | Tries azure, falls back to tfidf | Optional Azure credentials | — |

### State Representation

Each non-LLM state is converted to text as:
```
tool_used | file_path | operation_type | resulting_state | observation[:300]
```

## Output

- `embedding_baseline_results.json` — Full results with embedding and merged PTA metrics
- `embedding_baseline_report.html` — Interactive report with:
  - Side-by-side metric tables (Embedding vs Merged PTA)
  - Score distribution histograms
  - Per-task breakdown
  - Confusion matrices

## Key Metrics

- **Embedding F1** — BERTScore-style F1 (harmonic mean of precision/recall from greedy cosine alignment)
- **Embedding Precision** — avg best-match cosine sim per candidate state
- **Embedding Recall** — avg best-match cosine sim per reference state
- **Merged Structural** — % of merged PTA states matched in greedy subsequence alignment
- **Merged Combined** — average of structural and process coverage from merged PTA
