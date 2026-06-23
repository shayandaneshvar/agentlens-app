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
| `--no-outcome` | off | score with the `0.10*outcome` term removed (pure process score) |
| `--out FILE.json` | — | write per-trajectory results as JSON |
| `--csv FILE.csv` | — | write a flat CSV |

## 4. What you get

```
Reference: k=5 seed=42   |   Score: with +10 pass bonus

task                         pass  score         tier   cov%   coh  div
-----------------------------------------------------------------------
pydata__xarray-4075          True     61        solid   52.4 0.556    2
pydata__xarray-2905         False     45 partial_fail   54.1  0.45    2
```

Plus a **Skipped** list explaining any instance that was not scored.

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

2. **What "the paper's PTA" means.** With `--k 5` (default) the script builds a
   fresh k=5/seed=42 merge from the released passing trajectories — the paper's
   documented hyperparameters. Note the **original** 5 donors per task were
   excluded from the release (verified: each shipped GT PTA's `num_traces` =
   released passing + 5), so your k=5 reference uses *different* donors than the
   paper and scores are close-but-not-identical. `--k 0` uses the shipped
   `ground_truth` PTA, which is an **all-passing** merge (larger, lower coverage).

3. **The `+10` pass bonus.** By default `quality_score` uses the shipped SDK
   formula, which adds `0.10 * outcome` (+10 for a passing trajectory). Pass
   `--no-outcome` for a pure process score. Example: with the all-passing GT
   (`--k 0`), `pydata__xarray-4075` scores 50 with the bonus but 46 (`lucky`)
   without it — the bonus alone flips its tier.
