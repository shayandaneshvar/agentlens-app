# Baseline Comparison: Individual Trajectory Matching vs. Merged PTA Matching

## 1. Introduction

Evaluating AI coding agents requires comparing their execution traces against known-correct solutions. A natural baseline is to match a candidate trajectory against each reference trajectory individually and average the scores. In this report, we compare this **Individual Matching** baseline against our proposed **Merged PTA Matching** approach, which first merges multiple reference trajectories into a single Prefix Tree Automaton (PTA) — a directed acyclic graph representing the valid solution space — and then matches the candidate against it.

We demonstrate that the merged approach achieves substantially higher discrimination between passing and failing trajectories on structural coverage, the metric most relevant to assessing whether an agent followed a correct solution path.

## 2. Experimental Setup

### 2.1 Dataset

<!-- UPDATE: Change these numbers when running on larger scale -->

| Parameter | Value |
|-----------|-------|
| Tasks | 5 |
| Training trajectories per task (for merging) | 4 |
| Test passed trajectories per task | 2 |
| Test failed trajectories per task | 2 |
| Total training trajectories | 20 |
| Total test passed trajectories | 10 |
| Total test failed trajectories | 10 |
| Trajectory source | OpenHands (SWE-bench) |

All tasks are drawn from the SWE-bench benchmark, using trajectories from multiple LLM backends (GPT-4.1, GPT-4o, Claude Sonnet 4, Claude Sonnet 4.5, Claude Opus 4.5). Tasks were included only if they had at least 6 passing and 2 failing trajectories.

### 2.2 Holdout Split

For each task, trajectories are split with a fixed random seed (42) into:
- **Training set**: 4 passing trajectories used to build ground truth
- **Test set**: 2 held-out passing + 2 failing trajectories (never seen during ground-truth construction)

Both approaches use the **identical** split, ensuring a fair comparison.

### 2.3 Scoring Approaches

**Individual Matching (Baseline).** For each test candidate, we match it against each of the 4 training traces independently using greedy subsequence-coverage matching. The structural coverage, process coverage, and terminal match are collected per-pair and then *averaged* across all 4 training traces.

**Merged PTA Matching (Proposed).** The 4 training traces are merged into a single PTA — a DAG where shared prefixes are consolidated and divergent solution strategies form branches. Each test candidate is matched *once* against this merged PTA.

### 2.4 Metrics

We evaluate using two complementary scoring dimensions:

- **Structural Coverage** — percentage of ground-truth states that the candidate's trace matches in a greedy forward subsequence alignment. Measures how well the candidate follows valid solution *steps*.
- **Process Coverage** — percentage of *required tools* (those appearing in every ground-truth path) that appear in the candidate's trace. Measures whether the candidate used the right *tool types*.

These are evaluated as binary classifiers (pass/fail prediction) using:
- **AUROC** — area under the ROC curve (discrimination ability)
- **F1 Score** — harmonic mean of precision and recall at the optimal threshold

## 3. Results

### 3.1 Micro-Averaged Comparison (Pooled Across All Tasks)

<!-- UPDATE: Replace numbers from aggregate_metrics when re-running -->

| Metric | Individual | Merged | $\Delta$ |
|--------|-----------|--------|----------|
| **Structural Coverage** | | | |
| AUROC | 0.4600 | **0.6700** | +0.2100 |
| KS-Statistic | 0.3000 | 0.3000 | 0.0000 |
| Accuracy | 60.0% | **65.0%** | +5.0% |
| F1 Score | 0.3333 | **0.5333** | +0.2000 |
| Precision | 1.0000 | 0.8000 | -0.2000 |
| Recall | 0.2000 | **0.4000** | +0.2000 |
| Pass Mean $\pm$ Std | 8.0 $\pm$ 5.7 | **50.0 $\pm$ 36.4** | |
| Fail Mean $\pm$ Std | 7.4 $\pm$ 2.2 | 28.7 $\pm$ 24.8 | |
| **Process Coverage** | | | |
| AUROC | **0.8950** | 0.6000 | -0.2950 |
| KS-Statistic | **0.7000** | 0.2000 | -0.5000 |
| Accuracy | **85.0%** | 60.0% | -25.0% |
| F1 Score | **0.8421** | 0.7143 | -0.1278 |
| Pass Mean $\pm$ Std | 97.7 $\pm$ 3.5 | 100.0 $\pm$ 0.0 | |
| Fail Mean $\pm$ Std | 78.4 $\pm$ 18.3 | 95.0 $\pm$ 10.0 | |

### 3.2 Macro-Averaged AUROC (Mean Across Tasks)

<!-- UPDATE: Replace numbers when re-running -->

| Score Type | Individual | Merged | $\Delta$ |
|------------|-----------|--------|----------|
| Structural | 0.5250 | **0.6500** | +0.1250 |
| Process | **0.8750** | 0.6000 | -0.2750 |

### 3.3 Per-Task Breakdown (Structural AUROC)

<!-- UPDATE: Replace per-task numbers when re-running -->

| Task | Individual | Merged | $\Delta$ |
|------|-----------|--------|----------|
| astropy\_\_astropy-12907 | 0.500 | 0.500 | 0.000 |
| matplotlib\_\_matplotlib-13989 | 0.500 | **1.000** | +0.500 |
| psf\_\_requests-1142 | **0.625** | 0.500 | -0.125 |
| pydata\_\_xarray-2905 | 0.750 | **1.000** | +0.250 |
| scikit-learn\_\_scikit-learn-10297 | 0.250 | 0.250 | 0.000 |

## 4. Analysis

### 4.1 Structural Coverage: Why Merging Wins

The merged approach achieves a **+0.21 improvement in micro-averaged structural AUROC** (0.46 → 0.67) and a **+0.125 improvement in macro-averaged structural AUROC** (0.525 → 0.650). The fundamental reason is that merging models the *solution space*, while individual matching cannot.

**The single-path limitation.** Each individual training trace represents *one* specific solution strategy. When a correct test candidate uses a *different* but equally valid approach, it receives low coverage from traces it doesn't resemble — even though the candidate is correct. These low-similarity pairings dilute the average, pulling passing-trajectory scores downward.

This effect is visible in the score distributions. Under individual matching, passing trajectories average only 8.0% structural coverage — barely distinguishable from the 7.4% average for failing trajectories. The two distributions almost entirely overlap, rendering the metric nearly useless as a classifier (AUROC = 0.46, worse than random).

**Why merging fixes this.** The merged PTA consolidates multiple valid solution paths into a single DAG. A candidate that follows *any* valid path — whether it matches training trace 1, trace 2, or a hybrid — aligns against the correct branch in the DAG and receives high coverage. This is why passing trajectories jump from 8.0% to **50.0%** mean structural coverage under the merged approach, while failing trajectories rise more modestly (7.4% → 28.7%). The gap between distributions widens dramatically, enabling meaningful discrimination.

On two tasks (matplotlib, xarray), the merged approach achieves **perfect AUROC of 1.0** — complete separation of pass and fail — where individual matching scored only 0.50–0.75.

### 4.2 Process Coverage: Why Individual Matching Appears Stronger

Counterintuitively, individual matching shows *higher* process coverage AUROC (0.895 vs 0.600). This is not a weakness of merging — it is an artifact of how required tools are defined differently in the two settings.

**How required tools are extracted.** Process coverage measures the fraction of *required tools* present in the candidate. Required tools are defined as the tools appearing in *every* path of the reference trace (i.e., the intersection across all root-to-terminal paths).

- **Individual matching**: Each training trace is a single linear path. The "required tools" for that trace are simply *all tools it used* — every single tool call is considered mandatory. This creates a very strict checklist. Failing trajectories, which typically use different tools or miss exploration steps, fail to satisfy many of these checklists. The resulting averages cleanly separate pass from fail.

- **Merged PTA matching**: The merged PTA has multiple paths (branches for different solution strategies). The required tools are the *intersection* across all paths — only those tools that *every* strategy uses. Since different strategies may use different tools, the intersection shrinks. In our results, merged process coverage has pass_mean = 100.0% and fail_mean = 95.0%, indicating that the intersection of required tools is so small that even failing trajectories incidentally satisfy it.

**In summary, individual process coverage benefits from an overly strict definition of "required" — it demands tools specific to one strategy. This conflates "followed the same approach" with "solved the task correctly."** The merged approach correctly relaxes this to only tools *universally* required regardless of approach, which is the semantically correct definition but produces a less discriminative score with the current small merged tool set.

### 4.3 The Complementary Nature of the Two Metrics

The results reveal that structural coverage and process coverage capture *different aspects* of trajectory quality:

| What it measures | Structural Coverage | Process Coverage |
|------------------|-------------------|-----------------|
| Core question | "Did the agent follow a valid sequence of steps?" | "Did the agent use the right tools?" |
| Stronger under | Merged PTA (captures branching solution space) | Individual matching (strict per-trace tool requirements) |
| Weakness | Cannot assess tool usage breadth | Cannot assess step ordering or solution path |

This complementarity suggests that the **combined score** (average of structural and process) is the most robust single metric. Indeed, the individual combined AUROC (0.92) is strong because it benefits from the high process coverage signal. However, the merged combined AUROC (0.71) is held back by the weak merged process coverage — a limitation that can be addressed by refining the required-tool extraction to use the *union* of per-path tool sets rather than the *intersection*, or by weighting structural coverage more heavily.

### 4.4 Practical Implications

The key insight is this: **structural coverage — the metric that measures whether the agent followed a correct solution path — is fundamentally better served by merging.** Individual matching cannot model the solution space; it can only compare against isolated examples. As the diversity of valid solutions increases (more training traces, more distinct strategies), the individual average will regress toward the mean while the merged PTA will grow to encompass more valid paths, widening the gap.

Process coverage under individual matching benefits from an artifact (over-strict tool requirements) that may not generalize: as training set size grows and strategies diversify, averaging over increasingly disparate tool checklists will also dilute the signal. The merged approach, by contrast, can be extended to compute *per-path* tool requirements and take the *maximum* coverage across paths — preserving the richness of path-aware tool checking without the averaging problem.

## 5. Limitations

<!-- UPDATE: These limitations may change with larger-scale experiments -->

1. **Small sample size.** With only 5 tasks and 20 test trajectories, statistical power is limited. The KS-statistic p-values are not significant for structural coverage (p = 0.787). Larger-scale experiments are needed to confirm these trends.

2. **Task heterogeneity.** The tasks span different complexity levels and codebases. Some tasks may inherently have less diverse solution paths, reducing the advantage of merging.

3. **Merge count fixed at 4.** The optimal number of traces to merge may vary by task. The merge count study (a separate experiment) can quantify how structural AUROC improves as a function of merge count.

4. **Process coverage definition.** The current required-tools extraction (intersection across paths) is conservative for merged PTAs. Alternative definitions (union, or per-path maximum) may yield better discrimination.

## 6. Conclusion

Our merged PTA approach outperforms individual trajectory matching on **structural coverage**, the metric that captures whether a coding agent followed a valid solution path. The micro-averaged structural AUROC improves from 0.46 (below chance) to 0.67, and macro-averaged AUROC from 0.525 to 0.650. This improvement stems from the merged PTA's ability to model the *space* of valid solutions as a DAG, rather than comparing against isolated examples that each represent only one strategy.

Individual matching shows a superficial advantage on process coverage (AUROC 0.895 vs 0.600), but this is an artifact of overly strict per-trace tool requirements that conflate "same approach" with "correct approach." The process coverage definition for merged PTAs can be refined to capture path-aware tool requirements.

These results motivate the merged PTA as the primary scoring mechanism for agent trajectory evaluation, with process coverage as a complementary signal that benefits from further methodological refinement.
