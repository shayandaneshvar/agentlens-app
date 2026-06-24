# `quality_score` embeds the binary pass/fail outcome — should a process-quality metric do this?

## Summary

While reproducing the paper from the released **AgentLens-Bench** dataset, I found that
the shipped `quality_score` column is not a pure *process* signal: 10% of it is the binary
test outcome itself. This inflates pass/fail AUROC computed directly on the column (0.886 vs
~0.72 without the term) and, more importantly, largely determines the Ideal/Solid/Lucky tier
split — the paper's headline finding. I'd like to confirm whether this is intended, and how it
relates to the AUROC = 0.766 reported in Table 3 and the 10.7% Lucky rate.

## Where it comes from

`sdk/src/swe_trace_sdk/match.py`, `_compute_quality_score`:

```python
outcome = 100.0 if passed is True else (0.0 if passed is False else 50.0)
score = (
    0.25 * coverage
    + 0.25 * (coherence * 100.0)
    + 0.18 * (stage_completeness * 100.0)
    + 0.12 * (workflow_similarity * 100.0)
    + 0.10 * f1
    + 0.10 * outcome          # <-- binary label added into the score
)
```

`quality_assessment(...)` calls this with `passed=passed`, and `build_dataset.py` stores the
result as the `quality_score` column and derives `quality_tier` from it.

## Verification (on `annotations/trajectories.parquet`, 1,815 rows)

Recomputing the formula from the stored component columns:

| Recompute | Matches shipped `quality_score` |
|---|---|
| **with** `0.10 * outcome` | **100.0%** (all 1,815 rows, exact) |
| **without** the outcome term | 37.4% (only the failing rows, where `outcome = 0`) |

So the term is definitely present in the released scores.

Effect on mean scores:

| | passing | failing |
|---|---|---|
| process-only (no outcome term) | 50.75 | 41.51 |
| shipped `quality_score` | 60.75 | 41.50 |

The genuine process gap between pass/fail is ~9 points; the shipped gap (~19) is roughly half
injected label.

## Why it matters

### 1. AUROC on the column is inflated

`roc_auc_score(passed, quality_score)` = **0.886**. Removing the outcome term gives **0.723**
(KS p ≪ 0.05). The paper's reported Table 3 AUROC is **0.766**, which is close to the de-leaked
0.723 — suggesting Table 3 used a clean composite, while the *released column* is the inflated
one. Anyone recomputing discrimination from the dataset column gets 0.886, not 0.766.

### 2. The outcome term largely manufactures the tier distribution

Lucky is defined as `score < 47`. Because the `+10` bonus is added to every passing
trajectory, a passing trajectory is Lucky only if its **process** score is below **37**, and a
trajectory only needs a process score of **60** (not 70) to be tiered Ideal. Re-tiering the
1,136 passing trajectories with vs without the term:

| tier | WITH +10 (published) | NO outcome (pure process) |
|---|---|---|
| Ideal | 229 (20.2%) | 28 (**2.5%**) |
| Solid | 785 (69.1%) | 735 (64.7%) |
| Lucky | 122 (10.7%) | 373 (**32.8%**) |

The headline "20.2% Ideal / 10.7% Lucky" is dominated by the outcome term, not by process
quality. Removing it collapses Ideal to 2.5% and raises Lucky to 32.8%.

### 3. The 10.7% Lucky rate is not independently reproducible

There are three different Lucky numbers depending on what you actually compute:

| method | Lucky % | what it is |
|---|---|---|
| read published `quality_score` + threshold | **10.7%** | consistency check — it *is* the paper's column |
| independent k=5 / seed=42 re-run of the pipeline | **7.4%** | genuine end-to-end reproduction (different donors*) |
| strip the outcome term + same thresholds | **32.8%** | weak-process rate the metric would report without the leak |

Only the first matches the paper, and only because it re-reads the paper's own pre-computed,
outcome-contaminated scores. The independent re-run gives 7.4%.

\* The original 5 donors per task are excluded from the release (verified: each GT PTA's
`num_traces == released_passing + 5`, 235 = 5×47 trajectories, absent from the repo), so an
independent k=5 rebuild necessarily uses different donors.

## Fairness caveat (so the critique is airtight)

The 47/70 tier thresholds were calibrated on the pilot set **with** the outcome term in the
score. Applying those same cutoffs to process-only scores shifts everything down ~10 points
mechanically, so the **32.8%** figure overstates the true leak-free Lucky rate — a fair version
would re-calibrate thresholds on process-only scores. The threshold and the outcome term are
entangled and cannot be evaluated independently. The direction and magnitude, however, are
robust: the tier split is highly sensitive to the `+10` term.

---

# Issue 2: the shipped "ground-truth PTA" is not k=5, and is not the PTA used to score the dataset

## Summary

The README and `dataset_summary.json` describe the released ground-truth PTA as a
**k=5 merge (seed=42)**. The shipped files are actually **all-passing merges** (every passing
trajectory for the task), and they are **not** the reference the released `quality_score` /
`quality_tier` values were computed against.

This affects **per-trajectory reproducibility**, not the aggregate. The aggregate tier
distribution *does* reproduce in-distribution: rebuilding fresh k=5 references from the released
passing trajectories and resampling donor subsets (as the paper's merge-count study does) gives
Ideal 20.3 ± 2.1% and Lucky 9.7 ± 1.1% over 6 draws — the paper's 20.2% / 10.7% both fall in
range (`reproduce_k5_resample.py`). What is *not* recoverable is the **exact per-trajectory
score/tier**: the specific k=5 reference each trajectory was scored against is missing, so the
shipped `ground_truth/*.json` (an all-passing merge) yields a systematic offset when used to
re-score (−4.65 mean, ~70% exact tier match).

## What the docs claim

- `agentlens-bench/README.md`: *"For each of 47 tasks, we release a merged Prefix Tree Acceptor
  constructed from **5 independently successful trajectories**"* and *"Ground Truth | **k=5
  merged PTA** per task (seed=42)"*.
- `dataset_summary.json`: `"merge_k": 5`.

## What the files actually contain

Each `ground_truth/<task>_merged_pta.json` records `metadata.num_traces` and a matching
`trace_sources` list:

| task | `num_traces` in shipped GT | released passing |
|---|---|---|
| astropy-13236 | 8 | 3 |
| django-10880 | 32 | 27 |
| django-11066 | 41 | 36 |
| … all 47 … | **8–45** | `num_traces − 5` |

**Zero of the 47** PTAs have `num_traces == 5`. For every task `num_traces == released_passing
+ 5`, i.e. the shipped PTA is merged from the *entire* passing pool (released non-donors **plus**
the 5 held-out donors), not from 5 donors.

## Three different PTAs are implied; only the wrong one ships

| | PTA | shipped? |
|---|---|---|
| **A** | k=5 scoring reference (5 donors, seed=42) — produced the released `quality_score`/tiers (`build_dataset.py:554` scores against the fresh 5-donor merge) | **no** |
| **B** | all-passing merge — copied into `ground_truth/` (`build_dataset.py:519`); `num_traces` = full pool | yes, but mislabeled "k=5" |
| **C** | independent k=5 rebuild from released non-donors (different donors than A) | n/a (reconstructable) |

The 235 donor trajectories (5 × 47) that define **A** are excluded from the release (Issue 1
context: `num_traces == released_passing + 5` for all 47 tasks; an exhaustive repo search finds
no donor trajectory files and no archives).

## Consequence (measured)

- **Aggregate reproduces:** an independent k=5 rebuild (**C**) with donor resampling
  (`reproduce_k5_resample.py`, 6 draws, `PYTHONHASHSEED=0`) gives Ideal 20.3 ± 2.1%, Lucky
  9.7 ± 1.1%, Partial-fail 58.8 ± 2.7%, Off-track 41.2 ± 2.7% — the paper's 20.2 / 10.7 / 54.9 /
  45.1 all fall within range. A *single* draw is noisy (one gave Lucky 7.4%); the paper's
  merge-count study resamples for exactly this reason.
- **Per-trajectory does not:** scoring released trajectories against the shipped GT (**B**) gives
  mean signed error **−4.65**, ~70% exact tier agreement, Pearson 0.78 — because **B** is a
  larger (all-passing) PTA than the missing reference **A**, lowering coverage. Exact
  per-trajectory scores require **A**, which is not in the release.

## Questions

1. Is the `0.10 * outcome` term in `quality_score` intentional? If it is meant for ranking
   (preferring a passing demo over an equivalent failing one in curation), should it be
   excluded from (a) the pass/fail AUROC validation and (b) the Ideal/Solid/Lucky tiering,
   which are meant to measure *process*?
2. Was Table 3's AUROC = 0.766 computed on the released `quality_score` column or on a
   separate outcome-free composite? The numbers suggest the latter.
3. Should the released dataset ship a process-only `quality_score` (or both columns), so the
   tier labels reflect process rather than a process+outcome blend?
4. Is the shipped `ground_truth/*.json` intended to be the k=5 reference (as the README/
   `dataset_summary` state)? Its `num_traces` (8–45) says it is an all-passing merge.
5. Can the actual k=5 scoring references (or the 235 donor trajectories) be released, so the
   per-trajectory `quality_score`/tier values are reproducible from the artifacts?

## Repro

```bash
. .venv/bin/activate
python repro/reproduce_paper.py                 # tables incl. inflated 0.886 vs de-leaked 0.723
PYTHONHASHSEED=0 python repro/reproduce_k5.py    # independent k=5/seed=42 re-run -> 7.4% Lucky
python repro/validate_pipeline.py               # score vs shipped GT -> -4.65 offset, ~70% tier match
# inspect the shipped GT's true trace count (none are 5):
python -c "import json,glob,os; print({os.path.basename(f).replace('_merged_pta.json',''): json.load(open(f))['metadata']['num_traces'] for f in sorted(glob.glob('agentlens-bench/ground_truth/*.json'))[:5]})"
```
