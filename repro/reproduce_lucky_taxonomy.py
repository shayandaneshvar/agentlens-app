"""
Reproduce the paper's Lucky Pass taxonomy (Appendix D) on the released dataset.

Applies the SAME decision tree used in evaluate_my_trajectories.py
(`classify_lucky`) to the 122 Lucky-tier passing trajectories and compares the
category split + per-category profiles to the paper.

The paper fixes the order (C1->C2->C4->C3->C5) and the C1/C3/C5 rules; the C2 and
C4 thresholds are not published and were calibrated here (see classify_lucky).
"""
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "repro"))
sys.path.insert(0, str(ROOT / "sdk" / "src"))
import warnings, logging  # noqa: E402
warnings.filterwarnings("ignore")
logging.getLogger("swe_trace_sdk").setLevel(logging.ERROR)

from evaluate_my_trajectories import classify_lucky, LUCKY_CATEGORY_NAMES  # noqa: E402

PAPER_COUNTS = {"C1": 19, "C2": 42, "C3": 41, "C4": 5, "C5": 15}
PAPER_PROFILE = {  # (mean length, mean waste) from Appendix D.3
    "C1": (3.2, 0.0), "C2": (35.6, 19.6), "C3": (12.1, 1.2),
    "C4": (50.4, 3.2), "C5": (15.5, 0.7),
}


def has_incomplete(s):
    try:
        return any(r.get("reason") == "incomplete_implementation" for r in json.loads(s))
    except Exception:
        return False


def main():
    df = pd.read_parquet(ROOT / "agentlens-bench" / "annotations" / "trajectories.parquet")
    L = df[(df.passed) & (df.quality_tier == "lucky")].copy()
    L["incomplete"] = L.failure_reasons.apply(has_incomplete)

    L["cat"] = [
        classify_lucky(
            length=r.n_states,
            total_waste=r.total_wasted_steps,
            waste_severity=r.waste_severity,
            has_verification=r.stage_coverage_V > 0,
            incomplete_impl=r.incomplete,
        )
        for r in L.itertuples()
    ]
    c = Counter(L.cat)

    print(f"Lucky Pass taxonomy on {len(L)} released Lucky passes (paper: 122)\n")
    print(f"{'Cat':<4}{'name':<26}{'count':>7}{'paper':>7}{'len':>8}{'paperLen':>9}"
          f"{'waste':>8}{'paperW':>8}")
    for k in ["C1", "C2", "C3", "C4", "C5"]:
        sub = L[L.cat == k]
        pl, pw = PAPER_PROFILE[k]
        ml = sub.n_states.mean() if len(sub) else 0
        mw = sub.total_wasted_steps.mean() if len(sub) else 0
        print(f"{k:<4}{LUCKY_CATEGORY_NAMES[k]:<26}{c.get(k,0):>7}{PAPER_COUNTS[k]:>7}"
              f"{ml:>8.1f}{pl:>9}{mw:>8.1f}{pw:>8}")
    err = sum(abs(c.get(k, 0) - PAPER_COUNTS[k]) for k in PAPER_COUNTS)
    print(f"\nTotal absolute count error vs paper: {err} / {len(L)}  "
          f"(C1 and C5 are exact; C2/C4 thresholds calibrated).")


if __name__ == "__main__":
    main()
