"""Sample: Batch-compare multiple candidates against a shared ground truth.

Demonstrates:
- Loading many candidate traces from a directory.
- Using `match.run()` to compare each candidate.
- Printing a summary table sorted by coverage.

Prerequisites:
    Run ``python unzip_data.py`` first to extract the sample data.

Usage:
    python batch_compare.py
"""

from pathlib import Path
import sys

# sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swe_trace_sdk import trace, match

DATA = Path(__file__).resolve().parent / "data"

# Passing runs form the ground truth
GT_RUNS = [
    DATA / "create_and_serve_minimal_html-logs-claude-opus-4.5-pass/output/vsc-output/chat-export-logs.json",
    DATA / "create_and_serve_minimal_html-logs-claude-haiku-4.5-pass/output/vsc-output/chat-export-logs.json",
    DATA / "create_and_serve_minimal_html-logs-gpt-5.1-codex-pass/output/vsc-output/chat-export-logs.json",
    DATA / "create_and_serve_minimal_html-logs-gemini-3-flash-preview-pass/output/vsc-output/chat-export-logs.json",
]

# All 6 runs to compare (includes passing and failing)
CANDIDATE_DIR = DATA


def main() -> None:
    # Discover all chat-export-logs.json under the candidate dir
    candidate_files = sorted(CANDIDATE_DIR.rglob("chat-export-logs.json"))
    if not candidate_files:
        print(f"No trajectory files found under {CANDIDATE_DIR}")
        print("Run `python unzip_data.py` first.")
        sys.exit(1)

    # Build ground truth
    gt_traces = []
    for p in GT_RUNS:
        if p.exists():
            gt_traces.append(trace.load(str(p), format="chatlog"))
    if len(gt_traces) < 2:
        print("Need at least 2 ground-truth traces — update GT_RUNS paths.")
        sys.exit(1)

    ground_truth = trace.merge(gt_traces)
    print(f"Ground truth: {len(ground_truth.states)} states from {len(gt_traces)} runs\n")

    # Load candidates
    candidates = []
    labels = []
    for p in candidate_files:
        try:
            candidates.append(trace.load(str(p), format="chatlog"))
            # Use the parent folder chain as a label
            # Use the run folder name as a label
            labels.append(p.relative_to(CANDIDATE_DIR).parts[0])
        except Exception as exc:
            print(f"  Skipping {p}: {exc}")

    # Batch match
    results = [match.run(c, ground_truth) for c in candidates]

    # Print table
    header = f"{'Run':<55} {'Coverage':>8}  {'Terminal':>8}  {'Steps':>12}"
    print(header)
    print("-" * len(header))

    for label, r in sorted(zip(labels, results), key=lambda x: -x[1].metrics.coverage_percent):
        m = r.metrics
        term = "Yes" if m.terminal_state_match else "No"
        steps = f"{m.matched_count}/{m.total_ground_truth_states}"
        print(f"{label:<55} {m.coverage_percent:>7.1f}%  {term:>8}  {steps:>12}")


if __name__ == "__main__":
    main()
