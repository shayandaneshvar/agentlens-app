"""Unzip sample data from the data/ directory into sdk/samples/data/.

The zipped trajectory files live in:
    data/evaluation platform-vscbench/create_and_serve_minimal_html/*.zip

This script extracts them so the other sample scripts can use them
directly.

Usage:
    python unzip_data.py
"""

import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SAMPLES_DIR = Path(__file__).resolve().parent
DATA_SRC = SAMPLES_DIR.parents[1] / "data" / "evaluation platform-vscbench" / "create_and_serve_minimal_html"
DATA_DST = SAMPLES_DIR / "data"

# ---------------------------------------------------------------------------
# Which zips to extract (add/remove entries as needed)
# ---------------------------------------------------------------------------
ZIPS = [
    "create_and_serve_minimal_html-logs-claude-opus-4.5-pass.zip",
    "create_and_serve_minimal_html-logs-claude-haiku-4.5-pass.zip",
    "create_and_serve_minimal_html-logs-gpt-5.1-codex-pass.zip",
    "create_and_serve_minimal_html-logs-gemini-3-flash-preview-pass.zip",
    "create_and_serve_minimal_html-logs-gpt-4.1-fail.zip",
    "create_and_serve_minimal_html-logs-gpt-5-mini-fail.zip",
]


def main() -> None:
    if not DATA_SRC.exists():
        print(f"Source data directory not found: {DATA_SRC}")
        print("Make sure you are running from the sdk/samples/ directory")
        raise SystemExit(1)

    DATA_DST.mkdir(parents=True, exist_ok=True)

    for name in ZIPS:
        src = DATA_SRC / name
        if not src.exists():
            print(f"  SKIP (not found): {name}")
            continue

        # Each zip contains output/... at the root, so extract into a
        # named subfolder (the zip name without ".zip") to keep them
        # separate.
        dest = DATA_DST / src.stem
        dest.mkdir(parents=True, exist_ok=True)

        print(f"  Extracting: {name} -> {dest.name}/")
        with zipfile.ZipFile(src, "r") as zf:
            zf.extractall(dest)

    print(f"\nDone. Extracted to {DATA_DST}")
    # Show what we got
    logs = sorted(DATA_DST.rglob("chat-export-logs.json"))
    print(f"Found {len(logs)} trajectory file(s):")
    for p in logs:
        print(f"  {p.relative_to(DATA_DST)}")


if __name__ == "__main__":
    main()
