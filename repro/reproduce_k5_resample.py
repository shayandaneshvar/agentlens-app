"""
k=5 reproduction WITH resampling (mirrors the paper's merge-count study, which
resamples donor subsets at each k and reports mean +/- std).

A single k=5 donor draw is noisy (one run gave Lucky 7.4%). Here we draw R random
k=5 donor sets per task, score the held-out rest each time, and report the
distribution of Lucky% and pass/fail AUROC across resamples — so we can see
whether the paper's 10.7% Lucky is within donor-selection variance.

Run deterministically:  PYTHONHASHSEED=0 python repro/reproduce_k5_resample.py
"""
import logging
import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sdk" / "src"))
warnings.filterwarnings("ignore")
logging.getLogger("swe_trace_sdk").setLevel(logging.ERROR)

from swe_trace_sdk import io, match, trace as trace_api  # noqa: E402

K = 5
RESAMPLES = 6
IDEAL_MIN, LUCKY_MAX = 70, 47
TRAJ = ROOT / "agentlens-bench" / "trajectories"

_CACHE = {}


def load(path):
    if path not in _CACHE:
        _CACHE[path] = io.load_saved_trace(path)
    return _CACHE[path]


def process_score(m):
    base = (0.25 * m.coverage_percent + 0.25 * m.coherence_score * 100
            + 0.18 * m.stage_completeness * 100 + 0.12 * m.workflow_similarity * 100
            + 0.10 * m.f1_score)
    return max(0, min(100, int(round(base))))


def main():
    task_dirs = sorted(d for d in TRAJ.iterdir() if d.is_dir())
    tasks = []
    for td in task_dirs:
        p = sorted(f for f in td.glob("*.json") if "-pass-" in f.name)
        f = sorted(x for x in td.glob("*.json") if "-fail-" in x.name)
        if len(p) >= K:
            tasks.append((td.name, p, f))

    lucky_rates, ideal_rates, auroc_out, auroc_proc = [], [], [], []
    pfail_rates, offtrack_rates = [], []
    for ri in range(RESAMPLES):
        rng = random.Random(1000 + ri)
        y, s_out, s_proc, tiers, fail_scores = [], [], [], [], []
        for name, passing, failing in tasks:
            donors = passing.copy()
            rng.shuffle(donors)
            dset = set(donors[:K])
            merged = trace_api.merge([load(d) for d in dset], use_llm=False)
            for f in passing + failing:
                if f in dset:
                    continue
                passed = "-pass-" in f.name
                res = match.run(load(f), merged)
                qr = match.quality_assessment(res, load(f), merged, passed=passed)
                y.append(int(passed))
                s_out.append(qr.quality_score)               # with +10 outcome
                s_proc.append(process_score(res.metrics))    # process-only
                if passed:
                    tiers.append(qr.quality_score)
                else:
                    fail_scores.append(qr.quality_score)     # outcome=0 for fails
        y = np.array(y)
        tiers = np.array(tiers)
        fail_scores = np.array(fail_scores)
        lucky_rates.append(100 * (tiers < LUCKY_MAX).mean())
        ideal_rates.append(100 * (tiers >= IDEAL_MIN).mean())
        # failing tiers (build_dataset.revised_tier: >=40 partial_fail else off_track)
        pfail_rates.append(100 * (fail_scores >= 40).mean())
        offtrack_rates.append(100 * (fail_scores < 40).mean())
        auroc_out.append(roc_auc_score(y, s_out))
        auroc_proc.append(roc_auc_score(y, s_proc))
        print(f"  resample {ri+1}/{RESAMPLES}: lucky={lucky_rates[-1]:.1f}%  "
              f"ideal={ideal_rates[-1]:.1f}%  pfail={pfail_rates[-1]:.1f}%  "
              f"offtrack={offtrack_rates[-1]:.1f}%  AUROC(proc)={auroc_proc[-1]:.3f}",
              flush=True)

    def stat(xs):
        return f"{np.mean(xs):.1f} ± {np.std(xs):.1f}  (range {min(xs):.1f}–{max(xs):.1f})"

    def stat3(xs):
        return f"{np.mean(xs):.3f} ± {np.std(xs):.3f}  (range {min(xs):.3f}–{max(xs):.3f})"

    print("\n" + "=" * 70)
    print(f"k={K}, {RESAMPLES} resamples, {len(tasks)} tasks")
    print("=" * 70)
    print("PASSING tiers:")
    print(f"  Ideal%        (paper 20.2) : {stat(ideal_rates)}")
    print(f"  Lucky%        (paper 10.7) : {stat(lucky_rates)}")
    print("FAILING tiers:")
    print(f"  Partial-fail% (paper 54.9) : {stat(pfail_rates)}")
    print(f"  Off-track%    (paper 45.1) : {stat(offtrack_rates)}")
    print("Discrimination:")
    print(f"  AUROC with-outcome         : {stat3(auroc_out)}   (leaky; cf. column 0.886)")
    print(f"  AUROC process-only         : {stat3(auroc_proc)}   (cf. paper Table 3 = 0.766)")
    print(f"\n10.7% Lucky within resample range? "
          f"{min(lucky_rates) <= 10.7 <= max(lucky_rates)}")
    print(f"54.9% Partial-fail within range? "
          f"{min(pfail_rates) <= 54.9 <= max(pfail_rates)}")


if __name__ == "__main__":
    main()
