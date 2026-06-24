"""
Reproduce the paper's 5-category waste analysis from the released annotations.

Reproduces two tables:
  - Appendix C.2 (tab:inefficiency)  — pass/fail prevalence + per-trajectory waste
  - Appendix C.3 (tab:waste_tiers)   — Ideal vs Lucky among passing trajectories

The five categories: regression loops, blind retries, redundant steps,
unnecessary exploration, cyclic patterns. All are ground-truth-aware (patterns
already present in the merged PTA are not counted).

Definitions (matching the paper captions):
  Prevalence = % of trajectories with >= 1 instance of the category.
  Waste      = mean wasted steps among trajectories that have >= 1 instance,
               i.e. mean(<cat>_waste | <cat>_count > 0).  (NOT per-instance.)
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "agentlens-bench" / "annotations" / "trajectories.parquet"

CATS = [
    ("regression_loop", "Regression loops"),
    ("blind_retry", "Blind retries"),
    ("redundant_step", "Redundant steps"),
    ("unnecessary_exploration", "Unnecessary exploration"),
    ("cyclic_pattern", "Cyclic patterns"),
]

# Paper values: (prevA, prevB, ratio, wasteA, wasteB)
PAPER_PF = {
    "regression_loop": (38.7, 39.5, 1.02, 15.5, 12.2),
    "blind_retry": (46.0, 44.9, 0.98, 5.6, 7.7),
    "redundant_step": (50.7, 47.6, 0.94, 3.7, 5.3),
    "unnecessary_exploration": (6.1, 9.6, 1.58, 2.0, 1.9),
    "cyclic_pattern": (33.6, 44.5, 1.32, 4.6, 7.8),
}
PAPER_TIERS = {
    "regression_loop": (47.6, 18.0, 0.38, 18.6, 7.1),
    "blind_retry": (44.5, 39.3, 0.88, 2.7, 11.4),
    "redundant_step": (57.2, 24.6, 0.43, 2.9, 4.3),
    "unnecessary_exploration": (6.6, 0.8, 0.13, 2.0, 1.0),
    "cyclic_pattern": (41.5, 15.6, 0.38, 3.9, 3.4),
}


def prev(sub, c):
    return 100 * (sub[f"{c}_count"] > 0).mean()


def waste(sub, c):
    s = sub[sub[f"{c}_count"] > 0]
    return s[f"{c}_waste"].mean() if len(s) else 0.0


def table(title, A, B, labelA, labelB, ratio_name, paper):
    print("\n" + "=" * 86)
    print(title)
    print("=" * 86)
    print(f"{'Category':<24}{'Prev '+labelA:>9}{'Prev '+labelB:>9}{ratio_name:>7}"
          f"{'Waste '+labelA:>10}{'Waste '+labelB:>10}   {'paper(prevA/prevB/r/wA/wB)':>10}")
    for c, name in CATS:
        pa, pb = prev(A, c), prev(B, c)
        wa, wb = waste(A, c), waste(B, c)
        r = pb / pa if pa else float("nan")
        pp = paper[c]
        print(f"{name:<24}{pa:>8.1f}%{pb:>8.1f}%{r:>7.2f}{wa:>10.1f}{wb:>10.1f}"
              f"   | {pp[0]}/{pp[1]}/{pp[2]}/{pp[3]}/{pp[4]}")


def main():
    df = pd.read_parquet(ANN)
    P = df[df.passed]
    F = df[~df.passed]
    I = P[P.quality_tier == "ideal"]
    L = P[P.quality_tier == "lucky"]

    table("Pass/Fail waste breakdown (paper Appendix C.2, tab:inefficiency)",
          P, F, "P", "F", "F/P", PAPER_PF)
    table(f"Ideal (n={len(I)}) vs Lucky (n={len(L)}) waste (paper Appendix C.3, tab:waste_tiers)",
          I, L, "I", "L", "L/I", PAPER_TIERS)

    print("\nKey paper findings:")
    print(f"  Unnecessary-exploration F/P = {prev(F,'unnecessary_exploration')/prev(P,'unnecessary_exploration'):.2f}"
          f"  (paper 1.58)")
    print(f"  Cyclic-pattern F/P          = {prev(F,'cyclic_pattern')/prev(P,'cyclic_pattern'):.2f}"
          f"  (paper 1.32)")
    print(f"  Blind-retry waste: Lucky {waste(L,'blind_retry'):.1f} vs Ideal {waste(I,'blind_retry'):.1f}"
          f"  ({waste(L,'blind_retry')/waste(I,'blind_retry'):.1f}x; paper 11.4 vs 2.7, 4.2x)")


if __name__ == "__main__":
    main()
