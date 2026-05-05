"""Unzip OpenHands sample data into sdk/samples/data_openhands/.

The zipped trajectory files live in an external directory
(``openhands-swebench/astropy__astropy-12907/``).  This script extracts
a subset of them so the OpenHands sample scripts can use them directly.

Usage:
    cd sdk/samples
    python unzip_data_openhands.py

You can change ``OPENHANDS_SRC`` below to point at a different task or
directory if needed.
"""

import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SAMPLES_DIR = Path(__file__).resolve().parent

# Source: where the raw zip files live.  Adjust if your repo layout differs.
OPENHANDS_SRC = (
    SAMPLES_DIR.parents[1].parent
    / "openhands-swebench"
    / "astropy__astropy-12907"
)

DATA_DST = SAMPLES_DIR / "data_openhands"

# ---------------------------------------------------------------------------
# Which zips to extract
# ---------------------------------------------------------------------------
# Three passing runs (used to build ground truth) + two failing candidates.
ZIPS = [
    # --- passing runs ---
    "astropy__astropy-12907-logs-claude-opus-4.5-pass-22390720814.zip",
    "astropy__astropy-12907-logs-claude-sonnet-4-pass-22391840140.zip",
    "astropy__astropy-12907-logs-claude-sonnet-4.5-pass-22247295125.zip",
    # --- failing candidates ---
    "astropy__astropy-12907-logs-GPT-4.1-fail-22624495998.zip",
    "astropy__astropy-12907-logs-gpt-4o-fail-22012704285.zip",
]


def main() -> None:
    if not OPENHANDS_SRC.exists():
        print(f"Source directory not found: {OPENHANDS_SRC}")
        print("Update OPENHANDS_SRC in this script to point at your")
        print("openhands-swebench task directory.")
        raise SystemExit(1)

    DATA_DST.mkdir(parents=True, exist_ok=True)

    for name in ZIPS:
        src = OPENHANDS_SRC / name
        if not src.exists():
            print(f"  SKIP (not found): {name}")
            continue

        # Extract into a subfolder named after the zip (without .zip)
        dest = DATA_DST / src.stem
        if dest.exists() and any(dest.rglob("trajectory_openhands.json")):
            print(f"  SKIP (already extracted): {name}")
            continue

        dest.mkdir(parents=True, exist_ok=True)
        print(f"  Extracting: {name} -> {dest.name}/")
        with zipfile.ZipFile(src, "r") as zf:
            zf.extractall(dest)

    print(f"\nDone. Extracted to {DATA_DST}")

    # Show what we got
    trajs = sorted(DATA_DST.rglob("trajectory_openhands.json"))
    print(f"Found {len(trajs)} trajectory file(s):")
    for p in trajs:
        print(f"  {p.relative_to(DATA_DST)}")


if __name__ == "__main__":
    main()
