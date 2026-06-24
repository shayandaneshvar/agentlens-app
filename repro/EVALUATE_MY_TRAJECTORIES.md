# Evaluating your own trajectories with AgentLens

`evaluate_my_trajectories.py` scores **your** SWE-bench trajectories against the
**paper's PTAs** and reports an AgentLens quality score, tier, coverage,
coherence, and divergence point per trajectory.

## 1. One-time setup

```bash
cd /home/shayan/Desktop/agentlens-app
python3 -m venv .venv
. .venv/bin/activate
pip install pandas numpy scikit-learn scipy pyarrow tqdm
```

(If `.venv` already exists, just `. .venv/bin/activate`.)

## 2. Expected folder layout

```
my-swebench-sample/
├── result.json                       # aggregate file (ignored by the script)
├── pydata__xarray-4075__vZzDz9S/
│   ├── result.json                   # verifier_result.rewards.reward = 1.0 / 0.0
│   └── agent/trajectory.json         # ATIF trajectory (schema_version ATIF-v1.x)
└── pydata__xarray-2905__5eAEeup/
    ├── result.json
    └── agent/trajectory.json
```

- **Pass/fail** comes from each instance's `result.json` →
  `verifier_result.rewards.reward` (`1.0` = pass, `0.0` = fail).
- **Task name** comes from `result.json` (`task_name`), falling back to the
  directory name with the trailing `__<suffix>` stripped.

## 3. Run it

```bash
. .venv/bin/activate

# default: k=5, seed=42 reference; score includes the +10 pass bonus
python repro/evaluate_my_trajectories.py my-swebench-sample

# pure process score (drop the outcome term)
python repro/evaluate_my_trajectories.py my-swebench-sample --no-outcome

# use the shipped all-passing ground-truth PTA instead of a fresh k=5 merge
python repro/evaluate_my_trajectories.py my-swebench-sample --k 0

# save outputs
python repro/evaluate_my_trajectories.py my-swebench-sample --out r.json --csv r.csv
```

### Options

| Flag | Default | Meaning |
|------|---------|---------|
| `folder` (positional) | — | results folder containing `<task>__<suffix>/` dirs |
| `--k N` | `5` | merge count for the reference PTA. `N>0` builds a fresh k=N merge (seed) from the paper's released **passing** trajectories for the task. `N=0` uses the shipped `ground_truth/<task>.json` (an all-passing merge). |
| `--seed S` | `42` | donor-selection seed for the k=N merge |
| `--resamples R` | `1` | for `k>0`: build `R` k-merges from `R` donor draws (seeds `S..S+R-1`) and report **mean ± std** score. A single draw is noisy — **use `--resamples 5` (or more) for a valid, robust score** (see §6). |
| `--no-outcome` | off | score with the `0.10*outcome` term removed (pure process score) |
| `--out FILE.json` | — | write per-trajectory results as JSON |
| `--csv FILE.csv` | — | write a flat CSV |

**Recommended invocation** (robust, paper-faithful):

```bash
PYTHONHASHSEED=0 python repro/evaluate_my_trajectories.py my-swebench-sample --k 5 --resamples 5
```

## 4. What you get

```
Reference: k=5 seed=42 x5 resamples   |   Score: with +10 pass bonus

task                         pass  score  ±std         tier stbl   cov%   coh
-----------------------------------------------------------------------------
pydata__xarray-4075          True     63   3.8        solid    y   53.3 0.556
pydata__xarray-2905         False     43   2.6 partial_fail    y   57.1  0.45

========================================
SUMMARY
========================================
  total scored : 2
  passed       : 1
  failed       : 1
  tiers:
    [pass] ideal        : 0
    [pass] solid        : 1
    [pass] lucky        : 0
    [fail] partial_fail : 1
    [fail] off_track    : 0
```

- **±std** is the score spread across donor draws; **stbl** = `y` if the tier was
  the same in every draw (`N` = it flipped, so treat that tier as borderline).
- The **SUMMARY** reports total passes/fails and the count in each tier.
- A **WASTE** section reports the 5 inefficiency categories per trajectory (see below).
- Plus a **Skipped** list explaining any instance that was not scored.

### Waste report (5 categories)

Same detectors and definitions as the dataset (`build_dataset.py`), reused directly:

```
WASTE (5 categories, GT-aware)
task                           regress       retry      redund  unnec-expl      cyclic   TOTAL
pydata__xarray-2905                0/0     1.4/2.2         4/4         0/0        2/20    26.2
pydata__xarray-4075                0/0         0/0         0/0         0/0         0/0       0
```

Each cell is `count / wasted-steps` (mean over donor draws). The categories:
- **regression loops** — E→I→E backtracks (returning to exploration after editing)
- **blind retries** — 3+ consecutive identical (tool, file, stage) actions
- **redundant steps** — re-reading a file with no edit in between
- **unnecessary exploration** — post-implementation exploration of non-GT, non-test files
- **cyclic patterns** — multi-step repeated subsequences

All are **ground-truth-aware**: behavior already present in the reference PTA is not
counted as waste. `TOTAL` is total wasted steps; a per-category roll-up across all
trajectories is printed below the table. (Exported to the JSON/CSV as
`<cat>_count` / `<cat>_waste` and `total_wasted_steps`.)

### Tiers
- Passing: `ideal` (≥ 70), `solid` (47–69), `lucky` (< 47)
- Failing: `partial_fail` (≥ 40), `off_track` (< 40)

## 5. Which trajectories get scored

Only instances whose `<task>` has a paper reference are scored:
- `--k 5` (default): the task must have ≥ 5 released passing trajectories to
  build the merge.
- `--k 0`: the task must have a shipped `ground_truth/<task>_merged_pta.json`.

Everything else — other repo instances, tasks with too few passing trajectories,
or trajectories that yield an empty trace after tool adaptation — is listed under
**Skipped**. So you can point it at a folder with extra tasks; it self-filters.

## 6. Notes / caveats

1. **Tool-name adaptation.** This agent's tools are mapped to the SDK's canonical
   names so they match PTA states:
   - `file_editor` → `read_file` (view) / `create_file` (create) /
     `replace_string_in_file` (str_replace, insert)
   - `terminal` → `run_in_terminal` (the intent labeler reads the command text to
     decide exploration vs. verification, e.g. `grep` vs `pytest`)
   - `think` → **kept as Orchestration**. The paper defines Orchestration as
     "bookkeeping and reasoning steps", and `think` is a canonical SDK tool with
     an Orchestration stage hint. (The script registers the SDK's canonical tools
     as identity mappings, because the ATIF loader's `_resolve_tool` otherwise
     drops `think` as "unknown".)
   - `finish` → dropped (episode-end marker, no code operation)

   If your agent uses other tool names, add them to `FILE_EDITOR_CMD` /
   `FUNCTION_RENAME` near the top of the script.

2. **Is a single `--k 5` run valid? Use `--resamples`.** With `--k 5` the
   reference is built from one random draw of 5 donor trajectories, and the score
   carries real donor-selection noise (across draws the corpus Lucky rate swings
   7.8–11.3%, AUROC ±0.013, and individual scores move several points). A single
   draw is a **noisy point estimate**, not a robust score. Pass `--resamples 5`
   (or more) to average over donor draws and get a mean ± std plus a tier-stability
   flag (`stbl`). If `stbl=y`, the tier held across every draw and the result is
   trustworthy; if `stbl=N`, the trajectory sits on a tier boundary and the label
   is borderline. This mirrors the paper's merge-count study, which resamples for
   exactly this reason.

3. **What "the paper's PTA" means.** With `--k 5` the script builds a fresh
   k=5/seed=42 merge from the released passing trajectories — the paper's
   documented hyperparameters. The **original** 5 donors per task were excluded
   from the release (verified: each shipped GT PTA's `num_traces` = released
   passing + 5), so your k=5 reference uses *different* donors than the paper;
   scores match the paper in distribution but are not bit-identical. `--k 0` uses
   the shipped `ground_truth` PTA, which is an **all-passing** merge (larger, so
   lower coverage and a systematic downward score offset).

4. **The `+10` pass bonus.** By default `quality_score` uses the shipped SDK
   formula, which adds `0.10 * outcome` (+10 for a passing trajectory). Pass
   `--no-outcome` for a pure process score. Example: with the all-passing GT
   (`--k 0`), `pydata__xarray-4075` scores 50 with the bonus but 46 (`lucky`)
   without it — the bonus alone flips its tier.
