#!/usr/bin/env python3
"""Model comparison analysis for AgentLens experiment.

Computes per-model AUROC, accuracy, and F1 from holdout experiment results.
Shows how well the merged PTA discriminates pass/fail for each coding model.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
except ImportError:
    print("ERROR: scikit-learn is required. Install with: pip install scikit-learn")
    sys.exit(1)


MAIN_MODELS = {
    "gemini-2.5-pro", "gpt-4.1", "gpt-4o", "gpt-5.2-codex",
    "gpt-5.3-codex", "opus-4.5", "opus-4.6", "sonnet-4.5",
}

SCORE_FEATURES = [
    "structural_coverage", "process_coverage", "weighted_score",
    "stage_completeness", "workflow_similarity", "coherence_score",
    "temporal_profile_score", "bottleneck_coverage",
]


def load_holdout_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def collect_per_model_data(results: dict) -> dict:
    """Collect labels and scores per model from holdout trajectory results."""
    model_data = defaultdict(lambda: {"labels": [], "scores": defaultdict(list), "preds": []})

    for task in results["task_results"]:
        if task.get("error"):
            continue
        for traj in task["trajectory_results"]:
            if traj.get("is_train") or traj.get("error"):
                continue
            model = traj["model_name"]
            # Skip unknown/edge-case models
            if model not in MAIN_MODELS:
                continue

            label = 1 if traj["passed"] else 0
            verdict = (traj.get("predicted_verdict") or "").upper()
            pred = 1 if verdict in ("PASS", "LIKELY PASS") else 0

            model_data[model]["labels"].append(label)
            model_data[model]["preds"].append(pred)

            for feat in SCORE_FEATURES:
                val = traj.get(feat)
                if val is not None:
                    model_data[model]["scores"][feat].append(float(val))
                else:
                    model_data[model]["scores"][feat].append(0.0)

    return dict(model_data)


def compute_combined_score(scores: dict) -> list:
    """Compute combined score as average of structural and process coverage."""
    struct = scores.get("structural_coverage", [])
    proc = scores.get("process_coverage", [])
    if not struct or not proc:
        return []
    return [(s + p) / 2.0 for s, p in zip(struct, proc)]


def compute_model_metrics(model_data: dict) -> dict:
    """Compute AUROC, accuracy, F1 per model."""
    metrics = {}
    for model, data in sorted(model_data.items()):
        labels = np.array(data["labels"])
        preds = np.array(data["preds"])
        n_pass = int(labels.sum())
        n_fail = int(len(labels) - n_pass)

        m = {
            "n_total": len(labels),
            "n_pass": n_pass,
            "n_fail": n_fail,
            "accuracy": float(accuracy_score(labels, preds)),
            "f1": float(f1_score(labels, preds, zero_division=0)),
        }

        # Combined score AUROC
        combined = compute_combined_score(data["scores"])
        if combined and len(set(labels)) == 2:
            m["combined_auroc"] = float(roc_auc_score(labels, combined))
        else:
            m["combined_auroc"] = None

        # Per-feature AUROC
        for feat in SCORE_FEATURES:
            vals = np.array(data["scores"][feat])
            if len(set(labels)) == 2 and len(vals) == len(labels):
                try:
                    m[f"{feat}_auroc"] = float(roc_auc_score(labels, vals))
                except ValueError:
                    m[f"{feat}_auroc"] = None
            else:
                m[f"{feat}_auroc"] = None

        metrics[model] = m

    return metrics


def print_report(metrics: dict):
    """Print a formatted model comparison report."""
    print("\n" + "=" * 100)
    print("MODEL COMPARISON REPORT — AgentLens PTA-based Pass/Fail Discrimination")
    print("=" * 100)

    # Summary table
    print(f"\n{'Model':<20} {'N':>5} {'Pass':>5} {'Fail':>5} {'Acc%':>7} {'F1':>7} {'Comb AUROC':>11} {'Struct AUROC':>13} {'Process AUROC':>14}")
    print("-" * 100)
    for model, m in sorted(metrics.items(), key=lambda x: x[1].get("combined_auroc") or 0, reverse=True):
        auroc_str = f"{m['combined_auroc']:.4f}" if m['combined_auroc'] is not None else "N/A"
        struct_str = f"{m['structural_coverage_auroc']:.4f}" if m.get('structural_coverage_auroc') is not None else "N/A"
        proc_str = f"{m['process_coverage_auroc']:.4f}" if m.get('process_coverage_auroc') is not None else "N/A"
        print(f"{model:<20} {m['n_total']:>5} {m['n_pass']:>5} {m['n_fail']:>5} {m['accuracy']*100:>6.1f}% {m['f1']:>7.4f} {auroc_str:>11} {struct_str:>13} {proc_str:>14}")

    # Detailed per-feature table
    print(f"\n{'Model':<20} {'Weighted':>9} {'StagComp':>9} {'Workflow':>9} {'Coherence':>10} {'Temporal':>9} {'Bottlnk':>8}")
    print("-" * 80)
    for model, m in sorted(metrics.items(), key=lambda x: x[1].get("combined_auroc") or 0, reverse=True):
        vals = []
        for feat in ["weighted_score", "stage_completeness", "workflow_similarity", "coherence_score", "temporal_profile_score", "bottleneck_coverage"]:
            v = m.get(f"{feat}_auroc")
            vals.append(f"{v:.4f}" if v is not None else "N/A")
        print(f"{model:<20} {vals[0]:>9} {vals[1]:>9} {vals[2]:>9} {vals[3]:>10} {vals[4]:>9} {vals[5]:>8}")

    # Pass rate table
    print(f"\n{'Model':<20} {'Pass Rate':>10}")
    print("-" * 35)
    for model, m in sorted(metrics.items(), key=lambda x: x[1]['n_pass'] / max(x[1]['n_total'], 1), reverse=True):
        rate = m['n_pass'] / max(m['n_total'], 1)
        print(f"{model:<20} {rate*100:>9.1f}%")

    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Model comparison from holdout results")
    parser.add_argument("holdout_results", help="Path to holdout_experiment_results.json")
    parser.add_argument("--output-dir", "-o", default=None, help="Output directory for results JSON")
    args = parser.parse_args()

    results = load_holdout_results(args.holdout_results)
    model_data = collect_per_model_data(results)

    if not model_data:
        print("ERROR: No per-model data found in holdout results.")
        sys.exit(1)

    print(f"Found {len(model_data)} models with test trajectories")
    metrics = compute_model_metrics(model_data)
    print_report(metrics)

    # Save results
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.holdout_results).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_comparison_results.json"
    with open(output_path, "w") as f:
        json.dump({"models": metrics, "source": str(args.holdout_results)}, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
