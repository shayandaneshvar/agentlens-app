"""Sample: Load, merge, match, and visualise traces end-to-end.

This script demonstrates the full SWE Trace SDK workflow:
1. Load three "good" runs and build a ground-truth model.
2. Load a candidate run and match it against the ground truth.
3. Print key metrics.
4. Save an HTML report and visualisations.

Prerequisites:
    Run ``python unzip_data.py`` first to extract the sample data.

Usage:
    python basic_workflow.py
"""

from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# If you installed the SDK with `pip install -e .`, you can import directly.
# Otherwise, uncomment the following two lines so Python finds the package:
# sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# ---------------------------------------------------------------------------

from swe_trace_sdk import trace, match, export

# ---------------------------------------------------------------------------
# Configuration — paths relative to sdk/samples/data/ (extracted by unzip_data.py)
# ---------------------------------------------------------------------------
DATA = Path(__file__).resolve().parent / "data"

GOOD_RUNS = [
    DATA / "create_and_serve_minimal_html-logs-claude-opus-4.5-pass/output/vsc-output/chat-export-logs.json",
    DATA / "create_and_serve_minimal_html-logs-claude-haiku-4.5-pass/output/vsc-output/chat-export-logs.json",
    DATA / "create_and_serve_minimal_html-logs-gpt-5.1-codex-pass/output/vsc-output/chat-export-logs.json",
]

CANDIDATE = (
    DATA / "create_and_serve_minimal_html-logs-gpt-4.1-fail/output/vsc-output/chat-export-logs.json"
)

OUT_DIR = Path(__file__).resolve().parent / "output"

# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


def main() -> None:
    # Validate paths
    missing = [p for p in GOOD_RUNS + [CANDIDATE] if not p.exists()]
    if missing:
        print("Missing trajectory files — run `python unzip_data.py` first.")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)

    # 1. Load traces
    print("Loading traces...")
    good_traces = [trace.load(str(p), format="chatlog") for p in GOOD_RUNS]
    candidate_trace = trace.load(str(CANDIDATE), format="chatlog")
    print(f"  Loaded {len(good_traces)} good runs and 1 candidate.")

    # 2. Merge into ground truth
    print("Merging into ground truth...")
    ground_truth = trace.merge(good_traces)
    print(f"  Ground truth: {len(ground_truth.states)} states, "
          f"{len(ground_truth.transitions)} transitions")

    # 3. Match candidate against ground truth
    print("Matching candidate against ground truth...")
    result = match.run(candidate_trace, ground_truth)

    # 4. Print metrics
    m = result.metrics
    print("\n--- Match Metrics ---")
    print(f"  Coverage:         {m.coverage_percent:.1f}%")
    print(f"  Terminal match:   {m.terminal_state_match}")
    print(f"  Perfect match:    {m.perfect_match}")
    print(f"  Matched steps:    {m.matched_count} / {m.total_ground_truth_states}")
    print(f"  Candidate states: {m.candidate_states}")
    print(f"  Best path:        #{m.best_path_index + 1} of {m.total_paths}")

    if result.divergence_index is not None:
        print(f"  First divergence: step {result.divergence_index}")

    # 5. Save outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    report_html = str(OUT_DIR / "match_report.html")
    report_json = str(OUT_DIR / "match_report.json")
    gt_graph = str(OUT_DIR / "ground_truth_graph.html")
    cand_txt = str(OUT_DIR / "candidate_stats.txt")

    export.match(result, report_html)
    export.match(result, report_json, format="json")
    export.trace(ground_truth, gt_graph, format="html")
    export.trace(candidate_trace, cand_txt, format="txt")

    print(f"\nOutputs saved to {OUT_DIR}/")
    print(f"  {report_html}")
    print(f"  {report_json}")
    print(f"  {gt_graph}")
    print(f"  {cand_txt}")


if __name__ == "__main__":
    main()
