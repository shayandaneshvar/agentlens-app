"""Sample: Inspect a single trace — load, summarise, and visualise.

Prerequisites:
    Run ``python unzip_data.py`` first to extract the sample data.

Usage:
    python inspect_trace.py
"""

from pathlib import Path
import sys

# sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swe_trace_sdk import trace, export

DATA = Path(__file__).resolve().parent / "data"

TRAJECTORY = (
    DATA / "create_and_serve_minimal_html-logs-gpt-5.1-codex-pass"
    / "output/vsc-output/chat-export-logs.json"
)

OUT_DIR = Path(__file__).resolve().parent / "output"


def main() -> None:
    if not TRAJECTORY.exists():
        print(f"Trajectory file not found: {TRAJECTORY}")
        print("Run `python unzip_data.py` first.")
        sys.exit(1)

    # Load
    t = trace.load(str(TRAJECTORY), format="chatlog")

    # Summary
    print(f"Trace loaded: {len(t.states)} states, {len(t.transitions)} transitions")
    print(f"Initial state: {t.initial_state}")
    print(f"Terminal states: {t.get_terminal_states()}")

    tools = t.get_tool_sequence()
    print(f"Tool sequence ({len(tools)} tool calls):")
    for i, tool in enumerate(tools, 1):
        print(f"  {i}. {tool}")

    # Files touched
    all_files = set()
    for s in t.states.values():
        all_files.update(s.files_touched)
    if all_files:
        print(f"\nFiles touched ({len(all_files)}):")
        for fp in sorted(all_files)[:15]:
            print(f"  {fp}")
        if len(all_files) > 15:
            print(f"  ... and {len(all_files) - 15} more")

    # Save visualisations
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    html_path = str(OUT_DIR / "trace_graph.html")
    list_path = str(OUT_DIR / "trace_list.html")
    txt_path = str(OUT_DIR / "trace_stats.txt")

    export.trace(t, html_path, format="html")
    export.trace(t, list_path, format="html_list")
    export.trace(t, txt_path, format="txt")

    print(f"\nVisualisations saved:")
    print(f"  {html_path}")
    print(f"  {list_path}")
    print(f"  {txt_path}")

    # Round-trip: save and reload
    json_path = str(OUT_DIR / "trace.json")
    t.save(json_path)
    reloaded = trace.load(json_path, format="trace")
    print(f"\nRound-trip OK: {len(reloaded.states)} states, "
          f"{len(reloaded.transitions)} transitions")


if __name__ == "__main__":
    main()
