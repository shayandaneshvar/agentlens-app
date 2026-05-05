# SWE Trace SDK

Analyse and compare coding-agent execution traces.

The SDK loads per-run traces from agent trajectory files, merges
multiple successful runs into a ground-truth model, matches a candidate run
against that ground truth, and returns explainable coverage metrics.

The `format` parameter is required when loading traces (e.g. `"chatlog"` for
``chat-export-logs.json``, or `"trace"` for SDK-saved Trace JSON).

## Installation

```bash
pip install -e .
```

This installs everything including LLM semantic-equivalence and Azure OpenAI support.

## Quick start

```python
from swe_trace_sdk import trace, match, export

# 1. Load traces (evaluation platform format — chat-export-logs.json)
run_a = trace.load("./runs/run-111/output/vsc-output/chat-export-logs.json", format="chatlog")
run_b = trace.load("./runs/run-222/output/vsc-output/chat-export-logs.json", format="chatlog")
run_c = trace.load("./runs/run-333/output/vsc-output/chat-export-logs.json", format="chatlog")

# 2. Merge "good" runs into a ground truth
ground_truth = trace.merge([run_a, run_b, run_c])

# 3. Load a candidate and match it
candidate = trace.load("./runs/run-candidate/output/vsc-output/chat-export-logs.json", format="chatlog")
result = match.run(candidate, ground_truth)

# 4. Inspect metrics
print(result.metrics.coverage_percent)   # e.g. 85.7
print(result.metrics.terminal_state_match)  # True / False
print(result.metrics.perfect_match)      # True / False

# 5. Quality assessment — why is it failing? lucky or ideal pass?
report = match.quality_assessment(result, candidate, ground_truth)
print(report.verdict)           # e.g. "FAIL"
print(report.quality_tier)      # e.g. "partial_fail"
print(report.quality_score)     # e.g. 58
print(report.failure_reasons)   # [{reason, detail, severity}, ...]
print(report.strengths)         # ["Strong exploration phase (95% covered)"]

# 6. Cohort ranking — compare multiple trajectories
results = [
    ("Run A", match.run(run_a, ground_truth)),
    ("Run B", match.run(run_b, ground_truth)),
    ("Candidate", result),
]
ranking = match.rank_in_cohort(results)
print(ranking.passing)          # sorted by quality_score
print(ranking.failing)          # sorted by quality_score
print(ranking.summary)          # {ideal_count: 2, lucky_count: 0, ...}

# 7. Save reports and visualisations
export.match(result, "./out/match_result.html")        # HTML report
export.match(result, "./out/match_result.json", format="json")  # JSON
export.trace(ground_truth, "./out/ground_truth.html")          # interactive graph
export.trace(candidate, "./out/candidate.txt", format="txt")   # text summary
```

## Samples

Ready-to-run scripts live in `samples/`.  The sample data ships as zip files
in `data/benchmark-data/create_and_serve_minimal_html/` — run the unzip
helper first:

```bash
cd sdk/samples
python unzip_data.py        # extracts 4 trajectories into samples/data/
```

Then run any sample:

| Script | What it does |
|--------|-------------|
| `basic_workflow.py` | Full end-to-end: load 3 passing runs, merge into ground truth, match a failing candidate (gpt-4.1), print metrics, save HTML report and visualisations. |
| `batch_compare.py` | Compare all 4 runs against a shared ground truth and print a coverage table. |
| `inspect_trace.py` | Load a single trace, print a summary (states, tools, files touched), export graph/list/text visualisations, and round-trip save/reload. |

## Modules

| Module | Purpose |
|--------|---------|
| `swe_trace_sdk.trace` | `load()` and `merge()` — the main entry points |
| `swe_trace_sdk.match` | `run()` — compare candidate vs ground truth |
| `swe_trace_sdk.models` | Core types: `Trace`, `State`, `Transition`, `LogEntry` |
| `swe_trace_sdk.equivalence` | Pluggable 3-tier state equivalence (exact → heuristic → LLM) |
| `swe_trace_sdk.export` | Export traces (HTML graph, list, text) and match results (HTML, JSON) |
| `swe_trace_sdk.io` | File discovery and format detection helpers |
| `swe_trace_sdk.llm` | Optional LLM provider interface + SQLite cache |

## API reference

### `trace.load(filename, *, format) → Trace`

Load a single-run trace from a trajectory file or a previously saved trace
JSON.  The `format` parameter is **required** and selects the input type:

- `"chatlog"` — parse a raw evaluation platform `chat-export-logs.json`.
- `"trace"` — load a Trace JSON previously saved by the SDK.

### `trace.merge(traces, equivalence="default", use_llm=False) → Trace`

Merge multiple traces into a single ground-truth model.  Branch points are
tracked automatically.  The `equivalence` parameter selects the strategy
(`"default"` uses exact → heuristic, with optional LLM fallback).

### `match.run(candidate, ground_truth, matcher="subsequence_coverage", use_llm=False) → MatchResult`

Match a candidate trace against ground truth using subsequence coverage
matching.  Returns a `MatchResult` containing:

- `metrics.coverage_percent` — percentage of ground-truth steps covered.
- `metrics.terminal_state_match` — whether the final state matches.
- `metrics.perfect_match` — 100 % coverage.
- `alignment` — per-step mapping from candidate to ground truth.
- `divergence_index` — first ground-truth step the candidate misses.

### `match.quality_assessment(result, candidate, ground_truth) → QualityReport`

Assess the quality of a matched trajectory.  Answers: *Why is it failing?*
(for fails) and *Is this a lucky or ideal pass?* (for passes).

- `report.verdict` — `"PASS"` / `"LIKELY PASS"` / `"UNCERTAIN"` / `"LIKELY FAIL"` / `"FAIL"`.
- `report.quality_tier` — `"ideal"` / `"solid"` / `"lucky"` / `"partial_fail"` / `"off_track"`.
- `report.quality_score` — 0–100 composite for ranking within a cohort.
- `report.failure_reasons` — list of `{reason, detail, severity}` (empty for passes).
- `report.strengths` — what the agent did well, even in failures.
- `report.divergence_point` — `{step, description, expected_next}` — where it went wrong.
- `report.stage_coverage` — per-stage `{matched, total, percent}` for E / I / V / O.
- `report.key_metrics` — `{coverage_percent, coherence, stage_completeness, workflow_similarity}`.

### `match.rank_in_cohort(results, candidates=None, ground_truth=None) → CohortRanking`

Rank multiple trajectories within their pass/fail cohorts.

- `ranking.passing` — sorted by quality_score (best first), each with tier + rank.
- `ranking.failing` — sorted by quality_score (best first), each with top failure reason.
- `ranking.summary` — `{ideal_count, lucky_count, partial_fail_count, off_track_count, common_failure_reasons}`.

### `export.trace(trace, path, format="html")`

Render a trace visualisation.  Supported formats: `"html"` (interactive
graph), `"html_list"` (linear transition list), `"txt"` (text statistics).

### `export.match(result, path, format=None)`

Write a `MatchResult` to file.  Format is auto-detected from extension
(`.json` → JSON, otherwise → HTML).

### Equivalence

The `StateEquivalence` class provides three tiers:

1. **Exact** — same tool and identical `resulting_state` or `observation` hash.
2. **Heuristic** — tool-specific rules (file path normalisation, command
   normalisation, Jaccard word similarity for search queries).
3. **LLM** — optional semantic comparison via an OpenAI-compatible API.

LLM usage is opt-in (`use_llm=True`) and requires the `[llm]` extra.

### LLM configuration

Set `SWE_TRACE_LLM` (or `DEFAULT_LLM`) to `"provider:model[:temperature]"`:

```bash
export SWE_TRACE_LLM="openai:gpt-4o:0.3"
export OPENAI_API_KEY="sk-..."
```

## License

MIT
