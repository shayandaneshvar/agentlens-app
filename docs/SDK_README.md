# Code Agent Trace Analysis SDK (Design)

## APIs

This section shows the intended *quick-start* usage for a Python SDK that:
- loads per-run traces from agent logs
- merges multiple traces to form a ground truth model
- matches a candidate run against ground truth
- returns explainable match metrics

### Load a trace

Load a single-run trace directly from a `chat-export-logs.json` filename (no intermediate object layer).

```python
from swe_trace_sdk import trace

candidate = trace.load(
    "./coding-agent-trajectories/run-12345-instance-task-logs/output/vsc-output/chat-export-logs.json"
)

# candidate is a trace graph (states/transitions + metadata)
# candidate.states, candidate.transitions, candidate.metadata
```

Prototype mapping:
- [src/swe_pta_generator.py](../src/swe_pta_generator.py) (loads `chat-export-logs.json` and generates the trace graph)
- [src/swe_models.py](../src/swe_models.py) (trace graph schema: `PTA`, `State`, `Transition`)

Not yet in prototype (SDK API surface):
- Loaders/adapters for other trace log formats beyond `chat-export-logs.json`

### Build a ground truth by merging multiple "good" traces

```python
from swe_trace_sdk import trace

good_runs = [
  trace.load("./coding-agent-trajectories/run-111-output/vsc-output/chat-export-logs.json"),
  trace.load("./coding-agent-trajectories/run-222-output/vsc-output/chat-export-logs.json"),
  trace.load("./coding-agent-trajectories/run-333-output/vsc-output/chat-export-logs.json"),
]

ground_truth = trace.merge(
    good_runs,
    equivalence="default",   # exact -> tool-heuristic -> optional semantic
)

# Returns a merged trace that represents acceptable variants
# (e.g., trace_count and original_state_ids preserved in metadata)
```

Prototype mapping:
- [src/extract_swe_ground_truth.py](../src/extract_swe_ground_truth.py) (`merge_ptas(...)` orchestrates incremental merges)
- [src/swe_pta_merger.py](../src/swe_pta_merger.py) (`SWEPTAMerger.merge_ptas(...)` + branch tracking)
- [src/swe_state_equivalence.py](../src/swe_state_equivalence.py) (`SWEStateEquivalence.check_equivalence(...)`: exact/heuristic/LLM)


### Match a candidate run against ground truth

```python
from swe_trace_sdk import match

match_result = match.run(
    candidate=candidate,
    ground_truth=ground_truth,
    matcher="subsequence_coverage",  # enumerate domtree paths + greedy subsequence coverage
)

# match_result is a MatchResult:
# alignment mapping (candidate step -> ground-truth state), divergence/rejoin points,
# and per-step rationales
```

Prototype mapping:
- [src/swe_pta_matcher.py](../src/swe_pta_matcher.py) (`compare(...)`, `batch_compare(...)`, subsequence matching)
- [src/swe_state_equivalence.py](../src/swe_state_equivalence.py) (semantic equivalence used during matching)
- [src/swe_pta_generator.py](../src/swe_pta_generator.py) (`load_pta_or_trace(...)` path: raw trace → trace graph)


### Inspect match_result.metrics (explainability)

```python
# Example fields (names illustrative)
print(match_result.metrics.coverage_percent)
print(match_result.metrics.terminal_state_match)
print(match_result.metrics.perfect_match)
```

Prototype mapping:
- [src/swe_pta_matcher.py](../src/swe_pta_matcher.py) (coverage %, terminal match, matched/missing indexes)


### Visualizations and Reports

> In the SDK v0.1, `report` and `visualize` modules are merged to the `export` module.

```python
from swe_trace_sdk import report

report.save(match_result, "./out/match_result.html")

# Optional: visualize traces
from swe_trace_sdk import visualize
visualize.trace(ground_truth, "./out/ground_truth_graph.html", format="html")
visualize.trace(candidate, "./out/candidate_stats.txt", format="txt")
```

Prototype mapping:
- [src/swe_models.py](../src/swe_models.py) (`PTA.save(...)` / `PTA.load(...)` JSON schema)
- [src/swe_pta_visualizer.py](../src/swe_pta_visualizer.py) (writes `*_visualization.txt`, `*_graph.html`, `*_list.html`)
- [src/extract_swe_ground_truth.py](../src/extract_swe_ground_truth.py) (writes per-run trace JSONs + `merged_pta.json`)

Not yet in prototype (SDK API surface):
- A function to render/save match results as an HTML report.
- Report should have insights like fist diverging step, missing steps, etc.


## SDK Modules

A minimal, composable module layout (names are illustrative):

- `swe_trace_sdk.io`
  - File discovery and loading helpers (e.g., locate `chat-export-logs.json` inside an instance folder)
  - Validation and schema/version handling

- `swe_trace_sdk.trace`
  - `load(filename) -> trace`
  - `merge(traces, equivalence=...) -> trace`
  - Maintains merge metadata like `trace_count`, `original_state_ids`, branch points

- `swe_trace_sdk.equivalence`
  - Pluggable state equivalence strategies used by `trace.merge`
  - Built-ins:
    - Exact (same tool + observation hash)
    - Tool heuristics (e.g., same `filePath`, normalized terminal command)
    - Optional semantic equivalence (LLM-backed) behind a provider interface

- `swe_trace_sdk.match`
  - `run(candidate, ground_truth, matcher=...) -> MatchResult`
  - `MatchResult.metrics` contains explainable, stable metrics
  - Matcher: `subsequence_coverage` (enumerate domtree paths + greedy subsequence coverage)

- `swe_trace_sdk.visualize` (optional)
  - Exports trace and match visualizations (HTML graph, text summary, list view)

- `swe_trace_sdk.llm` (optional / isolated)
  - Provider interface + caching for semantic equivalence and natural-language explanations
  - Designed to be fully optional so the core remains offline-capable

- `swe_trace_sdk.report`
  - Serialization helpers for `MatchResult` (JSON)
  - Stable output schema for downstream dashboards/CI
