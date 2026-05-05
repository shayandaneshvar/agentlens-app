# Baselines

This directory contains baseline experiments that compare alternative trajectory scoring approaches against our proposed **Merged PTA Matching** method.

## Baselines

| Baseline | Directory | Description |
|----------|-----------|-------------|
| **Individual Matching** | [`individual_matching/`](individual_matching/) | Match candidate against each training trace individually and average scores. Tests whether merging adds value over per-trace matching. |
| **Embedding Similarity** | [`embedding_similarity/`](embedding_similarity/) | Embed states as vectors (Azure OpenAI or TF-IDF), compute BERTScore-style F1 alignment. Tests whether semantic similarity can replace structural matching. |

## Comprehensive Comparison

See [../../docs/BASELINES_COMPARISON.md](../../docs/BASELINES_COMPARISON.md) for a detailed comparison of all baselines with results, analysis, and final ranking tables.

## Quick Summary

| Approach | AUROC (Structural) | Key Finding |
|----------|:-----------------:|-------------|
| Individual Matching | 0.460 | Below chance — averaging across diverse strategies destroys the signal |
| TF-IDF Embedding | 0.595 | Barely above chance — lexical similarity can't separate pass/fail |
| Neural Embedding | 0.610 | Similar to TF-IDF — the limitation is paradigmatic, not about embedding quality |
| **Merged PTA (Ours)** | **0.670** | Best discrimination — captures ordering and solution-space branching |
