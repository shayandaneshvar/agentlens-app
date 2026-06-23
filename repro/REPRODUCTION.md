# AgentLens — Reproduction Report

Reproduction of the AgentLens paper (`paper/arXiv-2605.12925v3`) from this repo.

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install pandas numpy scikit-learn scipy pyarrow tqdm
python repro/reproduce_paper.py      # paper tables from shipped annotations
python repro/validate_pipeline.py    # re-run SDK scoring code on shipped PTAs
```

## What is / isn't shippable

The analysis scripts (`experiments/build_dataset.py`, `run_quality_analysis.py`, …)
consume **raw OpenHands traces** (`new_research_experiments/experiment_outputs`),
which are **not in the repo**. What ships is the fully-built **AgentLens-Bench**:
`annotations/trajectories.parquet` (1,815 × 40), 1,815 per-trajectory PTAs,
47 merged ground-truth PTAs, and 1,815 analysis reports — plus the SDK scoring code.

So reproduction is done at two levels.

## Level A — paper tables from annotations (`reproduce_paper.py`)

Every directly-derivable headline number reproduces **exactly**:

| Claim | Paper | Reproduced |
|---|---|---|
| Trajectories / passing / failing | 1815 / 1136 / 679 | ✅ exact |
| Tier distribution (Fig 1) | Ideal 229 (20.2%) · Solid 785 (69.1%) · Lucky 122 (10.7%) | ✅ exact |
| Failing split (§5.3) | Partial-fail 54.9% · Off-track 45.1% | ✅ exact |
| Model comparison (Table 2) | all 8 rows: Pass%, PR-rank, QS, QS-rank, Lucky% | ✅ exact |
| Curation table | Random/Top-50/Top-25 ideal%/lucky%/coh/score | ✅ exact |
| Waste F/P ratios (§5.2) | unnecessary-expl 1.58 · cyclic 1.32 | ✅ exact |

Two blocks differ, both explained:

- **Blind-retry waste/instance (§5.2):** reproduced 6.7 (Lucky) / 1.8 (Ideal),
  ratio 3.7× vs paper 11.4 / 2.7 / 4.2×. The qualitative finding (Lucky wastes
  ~4× more per retry) reproduces; absolute values use a raw-trace waste-window
  accounting not recoverable from the annotation columns.
- **Pass/fail discrimination (Table 3):** the shipped `quality_score` includes a
  `0.10·outcome` **label-leak** term (see `sdk/.../match.py:_compute_quality_score`),
  giving AUROC **0.886**. Removing the outcome term drops AUROC to **0.723**
  (KS p ≪ 0.05), matching the paper's 0.766 regime. The paper's exact 0.766 comes
  from `compute_correlation_metrics.py` on raw traces (not shipped). Per-signal
  AUROCs are approximate because the stored columns aren't the exact Φ signals.

## Level B — re-run the SDK scoring code (`validate_pipeline.py`)

Loads each shipped per-trajectory PTA + the task's merged ground-truth PTA, runs
`match.run` + `quality_assessment` (the actual scoring code), applies
`build_dataset.revised_tier`, and compares to the annotation.

Across all 47 tasks (282-trajectory stratified sample):

- Exact tier match: **70.2%**
- Score Pearson correlation: **0.782**
- Within ±5 points: **50.4%**
- Mean signed diff: **−4.65** (reproduced slightly lower)

**Root cause of the offset (verified):** the shipped `ground_truth/*.json` PTAs
are merges of **all passing** trajectories per task (`metadata.num_traces` ranges
8–44), **not** the **k=5** scoring references the annotations were produced with.
Scoring against the larger all-passing PTA lowers coverage → lower scores.

**Donor exclusion (verified, not assumed).** For all 47 tasks,
`GT num_traces == released_passing + 5` exactly (235 = 5 × 47). The 5 donors per
task — used to build the k=5 scoring reference — are held out of the released
scored set so no trajectory is scored against a reference built from itself
(`build_dataset.py:546`). An exhaustive repo search confirms those 235 donor
trajectory files exist **nowhere** in the release (no other dir, no archive).
So the exact k=5 reference cannot be rebuilt and bit-exact per-trajectory scores
are not recoverable; the pipeline mechanism is validated, exactness is bounded by
what ships.

## Level C — re-score with the paper's k=5 / seed=42 (`reproduce_k5.py`)

End-to-end re-run of the scoring pipeline using the documented hyperparameters:
per task, draw 5 donors (seed=42) from the released passing trajectories, merge
to a k=5 PTA, score the rest, tier via `revised_tier`. (Donors differ from the
paper's — see above — so this validates the methodology, not exact scores.)
Result over 1,513 trajectories (45 tasks; 2 skipped for < 5 released passing):

| Quantity | Paper | k=5 re-score |
|---|---|---|
| Ideal % (passing) | 20.2 | 21.9 |
| Solid % (passing) | 69.1 | 70.7 |
| Lucky % (passing) | 10.7 | 7.4 |
| Model mean-QS (8 models) | 54.7–67.4 | within ~1–4 pts, ranking ≈ preserved |

Tier shape and model QS reproduce closely. The Lucky rate is lower (7.4 vs 10.7)
because the references use different donors over a reduced pool — consistent with
the paper's own ablation that trajectory *selection* drives most PTA variance.

## Bottom line

The paper's **headline empirical claims reproduce exactly** from the released
dataset (tier stratification, the Lucky-Pass finding, model re-ranking, curation,
waste ratios). The **scoring code reproduces the dataset's tiers** at ~70%
agreement / 0.78 correlation. Two caveats surfaced: a `0.10·outcome` term inflates
the shipped `quality_score`'s pass/fail AUROC vs Table 3, and the shipped
ground-truth PTAs are 14-trace (all-passing) merges rather than the documented k=5.
```
