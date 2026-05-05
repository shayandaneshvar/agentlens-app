# AgentLens-Bench Dataset

A benchmark dataset of 1,815 fully-annotated coding agent trajectories across 47 SWE-bench tasks and 8 frontier models, with quality assessments, waste detection, and ground-truth PTA graphs.

## Quick Start

```python
import pandas as pd

# Load trajectory annotations (40 columns per trajectory)
df = pd.read_parquet("annotations/trajectories.parquet")

# Filter by quality tier
ideal = df[(df["passed"] == True) & (df["quality_tier"] == "ideal")]
lucky = df[(df["passed"] == True) & (df["quality_tier"] == "lucky")]

# Top-25 curated training set (highest quality)
top25 = df[df["passed"] == True].nlargest(25, "quality_score")
```

## Why This Dataset Is Novel

To our knowledge, **no existing coding agent benchmark provides process-level quality annotations**. Current datasets record only whether an agent's patch passes tests—discarding the rich behavioral signal encoded in the trajectory itself. AgentLens-Bench is the first dataset that answers *how well* an agent solved a problem, not just *whether* it did.

### What Makes This Dataset Unique

1. **First process-annotated SWE benchmark.** Every trajectory is scored along 10+ quality dimensions (coverage, coherence, workflow similarity, stage completeness, temporal alignment, etc.) against a ground-truth behavioral model. No prior dataset provides this.

2. **Ground-truth behavioral models (PTAs).** For each of 47 tasks, we release a merged Prefix Tree Acceptor constructed from 5 independently successful trajectories—a canonical representation of *how* the task should be solved. This enables structural comparison that goes far beyond output-matching.

3. **Multi-dimensional waste/inefficiency detection.** Every trajectory has per-step waste annotations across 5 categories (regression loops, blind retries, redundant reads, unnecessary exploration, cyclic patterns), all GT-aware—patterns present in the reference PTA are excluded. No other dataset provides actionable inefficiency attribution.

4. **Quality tier classification.** Each passing trajectory is classified as Ideal (principled), Solid (adequate), or Lucky (correct by chance)—enabling training data curation that eliminates low-quality demonstrations. Existing pipelines (SWE-Gym, R2E-Gym) filter by outcome only.

5. **Divergence-point localization.** For each trajectory, the exact step where behavior diverges from ground truth is recorded, enabling fine-grained failure analysis rather than binary labeling.

6. **Scale and model diversity.** 1,815 trajectories across 8 frontier models (Claude, GPT, Gemini families)—the largest quality-annotated trajectory collection to date.

### Practical Use Cases

- **Training data curation**: Select Top-k trajectories by quality score → eliminate all Lucky passes, raise coherence from 0.576 to 0.816 (Top-25). Directly addresses the quality problem in SWE-Gym/R2E-Gym training pipelines.
- **Process reward model training**: Use the 40-column annotations as supervision signals for training lightweight reward models that assess trajectory quality without running tests.
- **Model selection beyond pass rates**: Compare agent configurations by process quality—models with identical pass rates can differ by 5+ quality points and 34% in wasted steps.
- **Agent debugging**: Identify which waste category dominates a model's failures, with per-tool attribution (e.g., 42% of blind retries come from terminal commands).
- **Benchmark contamination detection**: Trajectories with high structural F1 but low coherence on supposedly novel tasks are contamination signals.

### Comparison with Existing Datasets

| Feature | SWE-bench | SWE-bench Verified | SWE-Gym | R2E-Gym | OpenHands Logs | Graphectory | **AgentLens-Bench** |
|---------|-----------|-------------------|---------|---------|---------------|-------------|---------------------|
| Task instances | 2,294 | 500 | 401 | 5,080 | ~300 | N/A | **47** |
| Agent trajectories | — | — | ~2,400 | ~10K | ~300 | ~100 | **1,815** |
| Models covered | — | — | 1 | 1 | 5 | 1 | **8** |
| Pass/fail labels | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Quality score (0–100) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| Multi-dim quality metrics | ✗ | ✗ | ✗ | ✗ | ✗ | Partial | **✓ (10+ dimensions)** |
| Ground-truth PTA graphs | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ (47 merged PTAs)** |
| Waste/inefficiency detection | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ (5 categories)** |
| Per-step annotations | ✗ | ✗ | ✗ | ✗ | ✗ | Stage labels | **✓ (stage + quality + waste)** |
| Divergence localization | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| Quality tier classification | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ (Ideal/Solid/Lucky)** |
| Training curation support | ✗ | ✗ | Outcome only | Outcome only | ✗ | ✗ | **✓ (quality-guided)** |
| Token waste attribution | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| Full trajectory JSONs | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ | **✓ (PTA format)** |
| Reproducible (no LLM calls) | N/A | N/A | ✗ | ✗ | N/A | ✓ | **✓** |

**Key differentiators:**
- SWE-bench/Verified provide *task definitions* (issues + tests) but no agent trajectories or quality annotations.
- SWE-Gym/R2E-Gym provide trajectories for *training* but filter only by pass/fail—no quality signal to distinguish principled solutions from lucky ones.
- OpenHands Logs provide raw execution traces but no structured quality assessment, waste detection, or ground-truth reference models.
- Graphectory defines process metrics but evaluates trajectories *in isolation* without comparison to a multi-trace ground truth, and does not provide waste detection or divergence localization.
- **AgentLens-Bench is the only dataset that combines full behavioral models (PTAs), multi-dimensional quality annotations, waste detection with GT-aware exclusion, and quality-guided curation support.**

---

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Tasks | 47 (filtered to tasks with ≥5 passing trajectories) |
| Trajectories | 1,815 (1,136 pass / 679 fail) |
| Models | 8 (sonnet-4.5, opus-4.5, opus-4.6, gpt-4o, gpt-4.1, gpt-5.2-codex, gpt-5.3-codex, gemini-2.5-pro) |
| Ground Truth | k=5 merged PTA per task (seed=42) |

### Quality Tier Distribution (Passing)

| Tier | Count | Percentage | Criteria |
|------|-------|-----------|----------|
| Ideal | 229 | 20.2% | quality_score ≥ 70 |
| Solid | 785 | 69.1% | 47 ≤ quality_score < 70 |
| Lucky | 122 | 10.7% | quality_score < 47 |

### Curation Table

| Strategy | k | Ideal% | Lucky% | Mean Coherence | Mean Score |
|----------|---|--------|--------|---------------|------------|
| Random (all passing) | 1,136 | 20.2% | 10.7% | 0.576 | 60.8 |
| Top-50 by score | 50 | 100% | 0% | 0.725 | 81.6 |
| Top-25 by score | 25 | 100% | 0% | 0.816 | 84.6 |

## Directory Structure

```
agentlens-bench/
├── README.md                          # This file
├── LICENSE                            # CC-BY-4.0
├── dataset_summary.json               # Machine-readable summary
├── annotations/
│   ├── trajectories.csv               # 1,815 rows × 40 columns
│   ├── trajectories.parquet           # Same, compressed
│   ├── tasks.csv                      # 47 rows, per-task stats
│   ├── tasks.parquet
│   └── curation_table.json            # Quality-guided curation stats
├── trajectories/                      # Tier 2: Individual PTA JSONs
│   └── {task_id}/
│       └── {trajectory_id}.json       # 1,815 files
├── ground_truth/                      # Tier 2: Merged PTA per task
│   └── {task_id}_merged_pta.json      # 47 files
└── analysis/                          # Tier 3: Full quality reports
    └── {task_id}/
        └── {trajectory_id}_report.json  # 1,815 files
```

## Column Reference (annotations/trajectories)

### Identity & Outcome
| Column | Type | Description |
|--------|------|-------------|
| `task_id` | str | SWE-bench task identifier |
| `model` | str | Agent model name |
| `trajectory_id` | str | Unique trajectory identifier |
| `passed` | bool | Whether the trajectory passed SWE-bench tests |
| `n_states` | int | Number of states in the trajectory |

### Quality Metrics
| Column | Type | Description |
|--------|------|-------------|
| `quality_score` | int | Overall quality score (0–100) |
| `quality_tier` | str | ideal / solid / lucky / partial_fail / off_track |
| `verdict` | str | Human-readable quality verdict |
| `coverage_percent` | float | % of ground-truth states matched |
| `precision_percent` | float | % of candidate states that matched GT |
| `f1_score` | float | Harmonic mean of coverage and precision |
| `coherence_score` | float | Forward-progress coherence (0–1) |
| `temporal_profile_score` | float | Stage timing similarity (0–1) |
| `workflow_similarity` | float | Stage-transition LCS ratio (0–1) |
| `stage_completeness` | float | Fraction of GT stages covered (0–1) |
| `bottleneck_coverage` | float | Minimum per-stage coverage (0–100) |
| `weighted_score` | float | Stage-importance-weighted coverage |

### Per-Stage Coverage
| Column | Type | Description |
|--------|------|-------------|
| `stage_coverage_E` | float | Exploration stage coverage % |
| `stage_coverage_I` | float | Implementation stage coverage % |
| `stage_coverage_V` | float | Verification stage coverage % |
| `stage_coverage_O` | float | Orchestration stage coverage % |

### Waste / Inefficiency Detection
| Column | Type | Description |
|--------|------|-------------|
| `regression_loop_count` | int | E→I→E regression patterns |
| `regression_loop_waste` | int | Wasted steps from regressions |
| `blind_retry_count` | int | 3+ consecutive identical actions |
| `blind_retry_waste` | int | Wasted steps from blind retries |
| `redundant_step_count` | int | Re-reads with no edit between |
| `redundant_step_waste` | int | Wasted steps from redundancy |
| `unnecessary_exploration_count` | int | Post-impl exploration on non-GT files |
| `unnecessary_exploration_waste` | int | Wasted steps from unnecessary exploration |
| `cyclic_pattern_count` | int | Multi-step repeated subsequences |
| `cyclic_pattern_waste` | int | Wasted steps from cyclic patterns |
| `total_wasted_steps` | int | Sum of all waste categories |
| `waste_severity` | float | total_wasted / n_states (0–1) |
| `wasted_input_tokens` | int | Input tokens in wasted steps |
| `wasted_output_tokens` | int | Output tokens in wasted steps |

### Divergence & Alignment
| Column | Type | Description |
|--------|------|-------------|
| `divergence_step` | int | Step where trajectory diverges from GT |
| `divergence_fraction` | float | divergence_step / n_states (0–1) |
| `stage_order_match` | bool | Whether stage ordering matches GT |
| `failure_reasons` | json | Serialized failure reason list |
| `strengths` | json | Serialized strengths list |

## Methodology

1. **Ground Truth Construction**: For each task, 5 passing trajectories (selected with seed=42) are merged into a Program Trace Automaton (PTA) that captures the canonical solution workflow.

2. **Quality Assessment**: Each remaining trajectory is matched against the merged PTA using state equivalence checking. The quality_score (0–100) integrates coverage, coherence, workflow similarity, and stage completeness.

3. **Waste Detection**: Five categories of inefficiency are detected, all GT-aware (patterns present in the merged PTA are excluded):
   - **Regression loops**: E→I→E backtrack patterns
   - **Blind retries**: 3+ consecutive identical (tool, file, stage) actions
   - **Redundant steps**: Re-reading files with no edit between reads
   - **Unnecessary exploration**: Post-implementation exploration on non-GT files
   - **Cyclic patterns**: Multi-step repeated subsequences

## Reproducibility

```bash
# Regenerate the dataset from raw experiment outputs
python experiments/build_dataset.py new_research_experiments/experiment_outputs \
    -o agentlens-bench
```

Configuration: `merge_k=5, seed=42, task_timeout=300s`

## License

This dataset is released under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](LICENSE).
