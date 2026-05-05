"""Sample: Load, merge, match, and visualise OpenHands traces end-to-end.

This script demonstrates the full SWE Trace SDK workflow for OpenHands
agent trajectories:

1. Load three passing runs and build a ground-truth model.
2. Load a failing candidate run and match it against the ground truth.
3. Print key metrics and alignment details.
4. Save HTML reports, visualisations, and JSON outputs.

This is the OpenHands equivalent of ``basic_workflow.py`` (which uses
evaluation platform / VSBench trajectories).

Prerequisites
-------------
Run ``python unzip_data_openhands.py`` first to extract the sample data.

Usage
-----
    cd sdk/samples
    python openhands_workflow.py
"""

from collections import Counter
from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# If you installed the SDK with `pip install -e .`, you can import directly.
# Otherwise, uncomment the following two lines so Python finds the package:
# sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# ---------------------------------------------------------------------------

from swe_trace_sdk import trace, match, export
from swe_trace_sdk.io import find_openhands_trajectory_file

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA = Path(__file__).resolve().parent / "data_openhands"

# Three passing runs → used to build the merged ground truth
GOOD_RUNS = [
    DATA / "astropy__astropy-12907-logs-claude-opus-4.5-pass-22390720814",
    DATA / "astropy__astropy-12907-logs-claude-sonnet-4-pass-22391840140",
    DATA / "astropy__astropy-12907-logs-claude-sonnet-4.5-pass-22247295125",
]

# Two failing runs → compared as candidates against the ground truth
CANDIDATES = [
    DATA / "astropy__astropy-12907-logs-GPT-4.1-fail-22624495998",
    DATA / "astropy__astropy-12907-logs-gpt-4o-fail-22012704285",
]

OUT_DIR = Path(__file__).resolve().parent / "output_openhands"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_traj(instance_dir: Path) -> Path:
    """Locate trajectory_openhands.json inside an instance directory."""
    found = find_openhands_trajectory_file(instance_dir)
    if found is None:
        raise FileNotFoundError(
            f"No trajectory_openhands.json found under {instance_dir}"
        )
    return found


def _label(p: Path) -> str:
    """Short human-readable label from a run directory name."""
    name = p.name
    # e.g. "astropy__astropy-12907-logs-claude-opus-4.5-pass-22390720814"
    #  → "claude-opus-4.5-pass"
    parts = name.split("-logs-")
    if len(parts) == 2:
        tail = parts[1]  # "claude-opus-4.5-pass-22390720814"
        # Drop the trailing run-id (last numeric segment)
        segments = tail.rsplit("-", 1)
        if segments[-1].isdigit():
            return segments[0]
        return tail
    return name


def _print_trace_summary(label: str, t) -> None:
    """Print a compact one-line summary of a loaded trace."""
    tools = Counter(s.tool_used for s in t.states.values() if s.tool_used)
    tool_str = ", ".join(f"{k}:{v}" for k, v in tools.most_common(5))
    print(f"  {label:40s}  {len(t.states):>3} states  [{tool_str}]")


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # 0. Validate that sample data has been extracted
    # ------------------------------------------------------------------
    all_dirs = GOOD_RUNS + CANDIDATES
    missing = [d for d in all_dirs if not d.exists()]
    if missing:
        print("Missing data directories — run `python unzip_data_openhands.py` first.")
        for d in missing:
            print(f"  {d}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Load traces
    # ------------------------------------------------------------------
    print("=" * 64)
    print("  Step 1: Load OpenHands trajectories")
    print("=" * 64)

    good_traces = []
    for d in GOOD_RUNS:
        traj = _find_traj(d)
        t = trace.load(str(traj), format="openhands")
        good_traces.append(t)
        _print_trace_summary(_label(d), t)

    candidate_traces = []
    for d in CANDIDATES:
        traj = _find_traj(d)
        t = trace.load(str(traj), format="openhands")
        candidate_traces.append(t)
        _print_trace_summary(_label(d) + " [candidate]", t)

    # ------------------------------------------------------------------
    # 2. Merge passing runs into ground truth
    # ------------------------------------------------------------------
    print()
    print("=" * 64)
    print("  Step 2: Merge passing runs → ground truth")
    print("=" * 64)

    ground_truth = trace.merge(good_traces)
    stats = ground_truth.metadata.get("merge_stats", {})

    print(f"  States:       {len(ground_truth.states)}")
    print(f"  Transitions:  {len(ground_truth.transitions)}")
    print(f"  Traces merged:{stats.get('traces_merged', '?')}")
    print(f"  States merged:{stats.get('states_merged', '?')}")
    print(f"  Branches:     {stats.get('branches_created', '?')}")

    # Show required tools extracted from the ground truth
    required_tools = match.extract_required_tools(ground_truth)
    print(f"  Required tools: {required_tools}")

    # ------------------------------------------------------------------
    # 3. Match each candidate against ground truth
    # ------------------------------------------------------------------
    print()
    print("=" * 64)
    print("  Step 3: Match candidates against ground truth")
    print("=" * 64)

    results = []
    for d, cand_trace in zip(CANDIDATES, candidate_traces):
        label = _label(d)
        result = match.run(cand_trace, ground_truth)
        results.append((label, result))

        m = result.metrics
        print(f"\n  --- {label} ---")
        print(f"  Coverage:         {m.coverage_percent:.1f}%")
        print(f"  Terminal match:   {m.terminal_state_match}")
        print(f"  Perfect match:    {m.perfect_match}")
        print(f"  Matched steps:    {m.matched_count} / {m.total_ground_truth_states}")
        print(f"  Candidate states: {m.candidate_states}")
        print(f"  Best path:        #{m.best_path_index + 1} of {m.total_paths}")

        if result.divergence_index is not None:
            print(f"  First divergence: step {result.divergence_index}")

        # Show first few alignment entries
        print(f"  Alignment (first 5):")
        for sa in result.alignment[:5]:
            icon = "✅" if sa.matched else "❌"
            gt = sa.ground_truth_state_id or "—"
            print(f"    {icon} step {sa.candidate_step}: "
                  f"candidate={sa.candidate_state_id} → gt={gt}")

    # ------------------------------------------------------------------
    # 4. Save outputs
    # ------------------------------------------------------------------
    print()
    print("=" * 64)
    print("  Step 4: Save visualisations and reports")
    print("=" * 64)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Ground truth visualisations
    gt_graph = OUT_DIR / "ground_truth_graph.html"
    gt_list  = OUT_DIR / "ground_truth_list.html"
    gt_txt   = OUT_DIR / "ground_truth_stats.txt"
    gt_json  = OUT_DIR / "ground_truth.json"

    export.trace(ground_truth, str(gt_graph), format="html")
    export.trace(ground_truth, str(gt_list), format="html_list")
    export.trace(ground_truth, str(gt_txt), format="txt")
    ground_truth.save(str(gt_json))

    print(f"  Ground truth graph:  {gt_graph.name}")
    print(f"  Ground truth list:   {gt_list.name}")
    print(f"  Ground truth stats:  {gt_txt.name}")
    print(f"  Ground truth JSON:   {gt_json.name}")

    # Per-candidate reports
    for label, result in results:
        safe = label.replace(" ", "_")
        report_html = OUT_DIR / f"match_{safe}.html"
        report_json = OUT_DIR / f"match_{safe}.json"

        export.match(result, str(report_html), format="html")
        export.match(result, str(report_json), format="json")
        print(f"  Match report ({label}): {report_html.name}, {report_json.name}")

    # Individual candidate visualisations
    for d, cand_trace in zip(CANDIDATES, candidate_traces):
        safe = _label(d).replace(" ", "_")
        cand_graph = OUT_DIR / f"candidate_{safe}_graph.html"
        cand_txt   = OUT_DIR / f"candidate_{safe}_stats.txt"

        export.trace(cand_trace, str(cand_graph), format="html")
        export.trace(cand_trace, str(cand_txt), format="txt")
        print(f"  Candidate ({_label(d)}): {cand_graph.name}, {cand_txt.name}")

    # Round-trip test: save and reload the ground truth
    reloaded = trace.load(str(gt_json), format="trace")
    print(f"\n  Round-trip OK: {len(reloaded.states)} states, "
          f"{len(reloaded.transitions)} transitions")

    print(f"\nAll outputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
