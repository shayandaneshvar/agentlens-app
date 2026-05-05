#!/usr/bin/env python3
"""
Embedding Similarity Baseline Experiment.

Compares trajectory evaluation approaches:
  1. Embedding Similarity (baseline): Embed state observations as vectors
     (using Azure OpenAI text-embedding-3-large or TF-IDF fallback),
     compute BERTScore-style greedy alignment similarity between candidate
     and each training trace, then average.
  2. Merged PTA Matching (proposed): Merge training traces into one ground-
     truth PTA, then match using greedy subsequence coverage.

Uses the same holdout split as the individual-matching baseline for direct
comparison across all three approaches.

Usage:
    python run_embedding_baseline.py <data_dir> --merge-count 4 --test-pass-count 2 --test-fail-count 2
    python run_embedding_baseline.py <data_dir> --merge-count 3 --test-pass-count 1 --test-fail-count 1 --embedding-method tfidf
    python run_embedding_baseline.py <data_dir> --output-dir <dir> -v
"""

import sys
import os
import json
import argparse
import logging
import random
import tempfile
import shutil
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from swe_trace_sdk import trace as trace_api, match
from swe_trace_sdk.models import Trace, State
from swe_trace_sdk.match import extract_required_tools, check_process_coverage, extract_required_files, check_file_coverage

# Import shared experiment infrastructure
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from metrics_utils import (
    STATS_AVAILABLE,
    tqdm,
    MetricSet,
    compute_classification_metrics,
    fill_metric_set,
    get_html_styles,
    generate_confusion_matrix_html,
    generate_histogram_js,
    format_auroc_color,
)

# Import trajectory discovery / loading utilities from holdout experiment
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "holdout"))
from run_holdout_experiment import (
    TrajectoryInfo,
    discover_trajectories,
    get_trajectory_json_path,
    split_trajectories,
    generate_pta_from_trajectory,
    merge_ptas,
)

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# TEXT EXTRACTION
# ============================================================================


def _state_text(state: State) -> str:
    """Return a compact text representation of a state for embedding."""
    parts: List[str] = []
    if state.tool_used:
        parts.append(state.tool_used)
    if state.file_path:
        parts.append(state.file_path)
    if state.operation_type:
        parts.append(state.operation_type)
    if state.resulting_state:
        parts.append(state.resulting_state)
    if state.observation:
        parts.append(state.observation[:300])
    return " | ".join(parts) if parts else "empty"


def trace_to_texts(trace_obj: Trace) -> List[str]:
    """Extract ordered list of state texts from a trace, filtering LLM-only states."""
    states = sorted(trace_obj.states.values(), key=lambda s: s.step)
    return [_state_text(s) for s in states if s.tool_used != "llm"]


# ============================================================================
# EMBEDDING ENGINE
# ============================================================================


class EmbeddingEngine:
    """Wraps Azure OpenAI text-embedding-3-large (preferred) or TF-IDF fallback."""

    def __init__(self, method: str = "auto"):
        if method == "auto":
            # Try Azure if env vars are present
            endpoint = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
            method = "azure" if (endpoint and api_key) else "tfidf"

        self.method = method
        self._client = None
        self._deployment = "text-embedding-3-large"

        if method == "azure":
            from openai import AzureOpenAI

            endpoint = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
            api_version = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION") or "2024-02-01"

            if not endpoint or not api_key:
                raise ValueError(
                    "Azure embedding requires AZURE_OPENAI_EMBEDDING_ENDPOINT "
                    "and AZURE_OPENAI_EMBEDDING_API_KEY (or AZURE_OPENAI_ENDPOINT "
                    "and AZURE_OPENAI_API_KEY) to be set."
                )

            self._client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
            )
            logger.info(f"Azure OpenAI embedding engine ready ({self._deployment} @ {endpoint})")
        else:
            logger.info("Using TF-IDF embeddings (no Azure API needed).")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Return (N, D) matrix of L2-normalised embeddings."""
        if not texts:
            return np.zeros((0, 1))

        if self.method == "azure":
            return self._encode_azure(texts)
        else:
            return self._encode_tfidf(texts)

    def encode_pair(
        self, texts_a: List[str], texts_b: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode two sets of texts. For TF-IDF, uses shared vocabulary."""
        if self.method == "azure":
            return self._encode_azure(texts_a), self._encode_azure(texts_b)
        else:
            return self._encode_tfidf_pair(texts_a, texts_b)

    # ── Azure OpenAI ────────────────────────────────────────────────────

    def _encode_azure(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Encode via Azure OpenAI text-embedding-3-large in batches."""
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._client.embeddings.create(
                input=batch, model=self._deployment
            )
            # Sort by index to preserve ordering
            items = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend([item.embedding for item in items])

        vecs = np.array(all_embeddings, dtype=np.float32)
        # L2-normalise
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs /= norms
        return vecs

    # ── TF-IDF fallback ─────────────────────────────────────────────────

    def _encode_tfidf(self, texts: List[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer

        tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
        mat = tfidf.fit_transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat /= norms
        return mat

    def _encode_tfidf_pair(
        self, texts_a: List[str], texts_b: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        from sklearn.feature_extraction.text import TfidfVectorizer

        combined = texts_a + texts_b
        tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
        tfidf.fit(combined)
        mat_a = tfidf.transform(texts_a).toarray().astype(np.float32)
        mat_b = tfidf.transform(texts_b).toarray().astype(np.float32)
        for mat in (mat_a, mat_b):
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            mat /= norms
        return mat_a, mat_b


# ============================================================================
# BERTSCORE-STYLE SIMILARITY
# ============================================================================


def bertscore_f1(
    emb_candidate: np.ndarray, emb_reference: np.ndarray
) -> Tuple[float, float, float]:
    """Compute BERTScore-style precision, recall, F1 between two embedding
    matrices.

    - Precision: for each *candidate* state, max cosine sim to any reference
      state, then average.  "How much of the candidate is relevant?"
    - Recall: for each *reference* state, max cosine sim to any candidate
      state, then average.  "How much of the reference is covered?"
    - F1: harmonic mean.

    Returns (precision, recall, f1) each in [0, 1].
    """
    if emb_candidate.shape[0] == 0 or emb_reference.shape[0] == 0:
        return 0.0, 0.0, 0.0

    # (N_cand, N_ref) cosine similarity matrix (already L2-normalised)
    sim = emb_candidate @ emb_reference.T

    precision = float(np.mean(np.max(sim, axis=1)))  # best ref for each cand
    recall = float(np.mean(np.max(sim, axis=0)))  # best cand for each ref

    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


# ============================================================================
# RESULT DATACLASSES
# ============================================================================


@dataclass
class EmbeddingTrajectoryScore:
    """Per-trajectory scores for the embedding baseline."""

    trajectory_id: str
    task_name: str
    model_name: str
    is_passed: bool

    # Embedding similarity (baseline) — averaged across training traces
    embedding_precision: float = 0.0
    embedding_recall: float = 0.0
    embedding_f1: float = 0.0
    embedding_detail: List[Dict] = field(default_factory=list)

    # Merged PTA matching (proposed) — for side-by-side comparison
    merged_structural: float = 0.0
    merged_process: float = 0.0
    merged_terminal_match: bool = False

    error: str = ""

    @property
    def embedding_score_pct(self) -> float:
        """F1 as a 0-100 percentage for compatibility with metrics_utils."""
        return self.embedding_f1 * 100.0

    @property
    def merged_combined(self) -> float:
        return 0.7 * self.merged_structural + 0.3 * self.merged_process


@dataclass
class TaskEmbeddingResult:
    """Per-task embedding baseline result."""

    task_name: str
    num_train_passed: int
    num_test_passed: int
    num_failed: int
    merged_pta_states: int = 0
    embedding_method: str = ""

    embedding_f1_metrics: MetricSet = field(default_factory=MetricSet)
    embedding_recall_metrics: MetricSet = field(default_factory=MetricSet)
    merged_structural_metrics: MetricSet = field(default_factory=MetricSet)

    trajectory_results: List[Dict] = field(default_factory=list)
    error: str = ""


@dataclass
class AggregateEmbeddingResult:
    """Aggregate results."""

    total_tasks: int = 0
    total_train_passed: int = 0
    total_test_passed: int = 0
    total_failed: int = 0
    embedding_method: str = ""

    # Micro-averaged
    embedding_f1: MetricSet = field(default_factory=MetricSet)
    embedding_recall: MetricSet = field(default_factory=MetricSet)
    merged_structural: MetricSet = field(default_factory=MetricSet)
    merged_combined: MetricSet = field(default_factory=MetricSet)

    # Macro-averaged AUROC
    macro_embedding_f1_auroc: float = 0.0
    macro_embedding_recall_auroc: float = 0.0
    macro_merged_structural_auroc: float = 0.0


# ============================================================================
# PER-TASK EXPERIMENT
# ============================================================================


def run_task_embedding(
    task_name: str,
    trajectories: List[TrajectoryInfo],
    output_dir: Path,
    temp_dir: Path,
    merge_count: int,
    test_pass_count: int,
    test_fail_count: int,
    engine: EmbeddingEngine,
    use_llm: bool = False,
    seed: int = 42,
) -> Optional[TaskEmbeddingResult]:
    """Run the embedding baseline for a single task."""

    train_passed, test_passed, failed = split_trajectories(
        trajectories,
        merge_count=merge_count,
        test_pass_count=test_pass_count,
        seed=seed,
    )

    if len(train_passed) < merge_count:
        logger.warning(f"Skipping {task_name}: not enough passed for merging")
        return None
    if len(test_passed) < test_pass_count:
        logger.warning(f"Skipping {task_name}: not enough held-out passed")
        return None
    if len(failed) < test_fail_count:
        logger.warning(f"Skipping {task_name}: not enough failed")
        return None

    rng = random.Random(seed)
    failed_shuffled = failed.copy()
    rng.shuffle(failed_shuffled)
    failed = failed_shuffled[:test_fail_count]

    logger.info(
        f"Task {task_name}: train={len(train_passed)}, "
        f"test_passed={len(test_passed)}, failed={len(failed)}"
    )

    task_dir = output_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    # ── Generate traces ─────────────────────────────────────────────────
    train_traces: List[Trace] = []
    for traj in train_passed:
        pta = generate_pta_from_trajectory(traj, task_dir, temp_dir)
        if pta is None:
            logger.warning(f"Skipping {task_name}: failed to load train {traj.trajectory_id}")
            return None
        train_traces.append(pta)

    if len(train_traces) < merge_count:
        return None

    # Merge for the PTA approach
    merged_pta = merge_ptas(train_traces, use_llm=use_llm)
    if not merged_pta:
        logger.warning(f"Skipping {task_name}: merge failed")
        return None

    merged_pta.save(str(task_dir / f"{task_name}_embedding_merged_pta.json"))
    merged_required_tools = extract_required_tools(merged_pta)
    merged_required_files = extract_required_files(merged_pta)

    # Generate test traces
    test_trajectories = test_passed + failed
    for traj in test_trajectories:
        if generate_pta_from_trajectory(traj, task_dir, temp_dir) is None:
            logger.warning(f"Skipping {task_name}: failed to load test {traj.trajectory_id}")
            return None

    # ── Pre-compute training trace texts ────────────────────────────────
    train_text_lists = [trace_to_texts(t) for t in train_traces]

    # ── Score each candidate ────────────────────────────────────────────
    all_scores: List[EmbeddingTrajectoryScore] = []

    for traj in test_trajectories:
        try:
            if traj.pta_path and Path(traj.pta_path).exists():
                cand_trace = trace_api.load(traj.pta_path, format="trace")
            else:
                result = get_trajectory_json_path(traj, temp_dir)
                if not result:
                    raise ValueError(f"Missing JSON for {traj.trajectory_id}")
                cand_trace = trace_api.load(result[0], format=result[1])

            cand_texts = trace_to_texts(cand_trace)

            # ── Embedding similarity (per training trace, then average) ──
            precisions, recalls, f1s = [], [], []
            detail = []
            for i, train_texts in enumerate(train_text_lists):
                if not cand_texts or not train_texts:
                    precisions.append(0.0)
                    recalls.append(0.0)
                    f1s.append(0.0)
                    detail.append({"train_index": i, "precision": 0.0, "recall": 0.0, "f1": 0.0})
                    continue

                emb_c, emb_r = engine.encode_pair(cand_texts, train_texts)
                p, r, f = bertscore_f1(emb_c, emb_r)
                precisions.append(p)
                recalls.append(r)
                f1s.append(f)
                detail.append(
                    {"train_index": i, "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}
                )

            avg_p = float(np.mean(precisions)) if precisions else 0.0
            avg_r = float(np.mean(recalls)) if recalls else 0.0
            avg_f1 = float(np.mean(f1s)) if f1s else 0.0

            # ── Merged PTA structural match ──────────────────────────────
            match_result = match.run(cand_trace, merged_pta, use_llm=use_llm)
            m_struct = match_result.metrics.f1_score
            m_terminal = match_result.metrics.terminal_state_match

            if merged_required_files:
                file_cov, _ = check_file_coverage(cand_trace, merged_required_files)
                m_proc = file_cov * 100.0
            else:
                m_proc = 100.0

            score = EmbeddingTrajectoryScore(
                trajectory_id=traj.trajectory_id,
                task_name=traj.task_name,
                model_name=traj.model_name,
                is_passed=traj.passed,
                embedding_precision=round(avg_p, 4),
                embedding_recall=round(avg_r, 4),
                embedding_f1=round(avg_f1, 4),
                embedding_detail=detail,
                merged_structural=round(m_struct, 2),
                merged_process=round(m_proc, 2),
                merged_terminal_match=m_terminal,
            )
            all_scores.append(score)

            logger.debug(
                f"  [{traj.trajectory_id}] passed={traj.passed} | "
                f"emb_f1={avg_f1:.3f} merged_struct={m_struct:.1f}"
            )

        except Exception as e:
            logger.error(f"Error scoring {traj.trajectory_id}: {e}")
            all_scores.append(
                EmbeddingTrajectoryScore(
                    trajectory_id=traj.trajectory_id,
                    task_name=traj.task_name,
                    model_name=traj.model_name,
                    is_passed=traj.passed,
                    error=str(e),
                )
            )

    # ── Task-level metrics ──────────────────────────────────────────────
    task_result = TaskEmbeddingResult(
        task_name=task_name,
        num_train_passed=len(train_passed),
        num_test_passed=len(test_passed),
        num_failed=len(failed),
        merged_pta_states=len(merged_pta.states),
        embedding_method=engine.method,
        trajectory_results=[asdict(s) for s in all_scores],
    )

    _compute_task_metrics(task_result, all_scores)

    return task_result


def _compute_task_metrics(
    result: TaskEmbeddingResult, scores: List[EmbeddingTrajectoryScore]
) -> None:
    """Fill MetricSet fields on a TaskEmbeddingResult."""

    @dataclass
    class _Tmp:
        is_passed: bool
        structural_coverage: float
        error: str = ""

    # Embedding F1 (scaled to 0-100)
    ef1 = [_Tmp(s.is_passed, s.embedding_score_pct, s.error) for s in scores]
    fill_metric_set(
        result.embedding_f1_metrics,
        compute_classification_metrics(ef1, "structural_coverage"),
    )

    # Embedding Recall (scaled to 0-100)
    er = [_Tmp(s.is_passed, s.embedding_recall * 100, s.error) for s in scores]
    fill_metric_set(
        result.embedding_recall_metrics,
        compute_classification_metrics(er, "structural_coverage"),
    )

    # Merged structural
    ms = [_Tmp(s.is_passed, s.merged_structural, s.error) for s in scores]
    fill_metric_set(
        result.merged_structural_metrics,
        compute_classification_metrics(ms, "structural_coverage"),
    )


# ============================================================================
# AGGREGATE
# ============================================================================


def compute_aggregate(
    task_results: List[TaskEmbeddingResult],
    all_scores: List[EmbeddingTrajectoryScore],
) -> AggregateEmbeddingResult:
    agg = AggregateEmbeddingResult(
        total_tasks=len(task_results),
        total_train_passed=sum(t.num_train_passed for t in task_results),
        total_test_passed=sum(t.num_test_passed for t in task_results),
        total_failed=sum(t.num_failed for t in task_results),
        embedding_method=task_results[0].embedding_method if task_results else "",
    )

    valid = [t for t in task_results if not t.error]
    if not valid or not STATS_AVAILABLE:
        return agg

    # Macro AUROC
    ef1_aurocs = [t.embedding_f1_metrics.auroc for t in valid if not t.embedding_f1_metrics.error]
    er_aurocs = [t.embedding_recall_metrics.auroc for t in valid if not t.embedding_recall_metrics.error]
    ms_aurocs = [t.merged_structural_metrics.auroc for t in valid if not t.merged_structural_metrics.error]

    if ef1_aurocs:
        agg.macro_embedding_f1_auroc = float(np.mean(ef1_aurocs))
    if er_aurocs:
        agg.macro_embedding_recall_auroc = float(np.mean(er_aurocs))
    if ms_aurocs:
        agg.macro_merged_structural_auroc = float(np.mean(ms_aurocs))

    # Micro-averaged (pool all test scores)
    @dataclass
    class _Tmp:
        is_passed: bool
        structural_coverage: float
        error: str = ""

    ef1_tmp = [_Tmp(s.is_passed, s.embedding_score_pct, s.error) for s in all_scores]
    fill_metric_set(agg.embedding_f1, compute_classification_metrics(ef1_tmp, "structural_coverage"))

    er_tmp = [_Tmp(s.is_passed, s.embedding_recall * 100, s.error) for s in all_scores]
    fill_metric_set(agg.embedding_recall, compute_classification_metrics(er_tmp, "structural_coverage"))

    ms_tmp = [_Tmp(s.is_passed, s.merged_structural, s.error) for s in all_scores]
    fill_metric_set(agg.merged_structural, compute_classification_metrics(ms_tmp, "structural_coverage"))

    mc_tmp = [_Tmp(s.is_passed, s.merged_combined, s.error) for s in all_scores]
    fill_metric_set(agg.merged_combined, compute_classification_metrics(mc_tmp, "structural_coverage"))

    return agg


# ============================================================================
# HTML REPORT
# ============================================================================


def generate_report(
    agg: AggregateEmbeddingResult,
    task_results: List[TaskEmbeddingResult],
    all_scores: List[EmbeddingTrajectoryScore],
    output_path: str,
) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    styles = get_html_styles()

    def card(v, l, c=""):
        cls = f'class="metric-value {c}"' if c else 'class="metric-value"'
        return f'<div class="metric-card"><div {cls}>{v}</div><div class="metric-label">{l}</div></div>'

    ef = agg.embedding_f1
    ms = agg.merged_structural

    summary = "".join([
        card(str(agg.total_tasks), "Tasks"),
        card(str(agg.total_train_passed), "Train Passed"),
        card(str(agg.total_test_passed), "Test Passed"),
        card(str(agg.total_failed), "Test Failed"),
        card(agg.embedding_method.upper(), "Embedding Method"),
    ])

    headline = "".join([
        card(f"{ef.auroc:.3f}", "Embedding F1 AUROC"),
        card(f"{ms.auroc:.3f}", "Merged Struct AUROC", "good" if ms.auroc > ef.auroc else ""),
        card(f"{ef.f1:.3f}", "Embedding F1 (classif.)"),
        card(f"{ms.f1:.3f}", "Merged Struct F1", "good" if ms.f1 > ef.f1 else ""),
    ])

    # Comparison table
    er = agg.embedding_recall
    mc = agg.merged_combined

    def _delta(a, b):
        d = a - b
        c = "var(--success)" if d > 0 else ("var(--accent)" if d < 0 else "var(--text-secondary)")
        s = "+" if d > 0 else ""
        return f'<span style="color:{c}">{s}{d:.4f}</span>'

    rows = ""
    for label, attr in [("AUROC", "auroc"), ("KS-Statistic", "ks_statistic"),
                         ("Accuracy", "accuracy"), ("F1 Score", "f1"),
                         ("Precision", "precision"), ("Recall", "recall")]:
        ev = getattr(ef, attr)
        rv = getattr(er, attr)
        sv = getattr(ms, attr)
        cv = getattr(mc, attr)
        fmt = ".1%" if attr == "accuracy" else ".4f"
        rows += f"""<tr>
            <td><strong>{label}</strong></td>
            <td>{ev:{fmt}}</td><td>{sv:{fmt}}</td><td>{_delta(sv, ev)}</td>
            <td>{rv:{fmt}}</td><td>{cv:{fmt}}</td><td>{_delta(cv, rv)}</td>
        </tr>"""

    comp_table = f"""<table><thead>
        <tr><th rowspan="2">Metric</th>
            <th colspan="3" style="text-align:center">Embedding F1 vs Merged Structural</th>
            <th colspan="3" style="text-align:center">Embedding Recall vs Merged Combined</th></tr>
        <tr><th>Embed F1</th><th>Merged Struct</th><th>Delta</th>
            <th>Embed Recall</th><th>Merged Comb</th><th>Delta</th></tr>
    </thead><tbody>{rows}</tbody></table>"""

    # Confusion matrices
    cm = f"""<div class="charts-grid">
        {generate_confusion_matrix_html(ef, "Embedding F1")}
        {generate_confusion_matrix_html(ms, "Merged Structural")}
    </div>"""

    # Histograms
    valid = [s for s in all_scores if not s.error]
    ef_pass = [s.embedding_score_pct for s in valid if s.is_passed]
    ef_fail = [s.embedding_score_pct for s in valid if not s.is_passed]
    ms_pass = [s.merged_structural for s in valid if s.is_passed]
    ms_fail = [s.merged_structural for s in valid if not s.is_passed]

    hist_js = "\n".join([
        generate_histogram_js("h_ef", ef_pass, ef_fail, "Embedding F1 (%)", ef.optimal_threshold),
        generate_histogram_js("h_ms", ms_pass, ms_fail, "Merged Structural Coverage (%)", ms.optimal_threshold),
    ])

    # Per-task table
    task_rows = ""
    for t in sorted(task_results, key=lambda x: x.task_name):
        if t.error:
            task_rows += f'<tr><td>{t.task_name}</td><td colspan="7">{t.error}</td></tr>'
            continue
        ea = t.embedding_f1_metrics.auroc
        ma = t.merged_structural_metrics.auroc
        d = ma - ea
        dc = "var(--success)" if d > 0 else ("var(--accent)" if d < 0 else "var(--text-secondary)")
        task_rows += f"""<tr>
            <td>{t.task_name}</td><td>{t.num_train_passed}</td><td>{t.num_test_passed + t.num_failed}</td>
            <td style="color:{format_auroc_color(ea)}">{ea:.4f}</td>
            <td style="color:{format_auroc_color(ma)}">{ma:.4f}</td>
            <td style="color:{dc}">{d:+.4f}</td>
            <td>{t.merged_pta_states}</td>
        </tr>"""

    method_desc = (
        '<strong>Azure OpenAI text-embedding-3-large</strong> (3072-dim dense vectors)'
        if agg.embedding_method == "azure"
        else '<strong>TF-IDF (5000 features)</strong>'
    )

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>Embedding Baseline vs Merged PTA</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>{styles}</style></head><body>
<div class="container">
    <h1>Embedding Similarity Baseline vs Merged PTA Matching</h1>
    <p class="timestamp">Generated: {timestamp}</p>

    <h2>Experiment Summary</h2>
    <div class="summary-grid">{summary}</div>

    <h2>Headline: Embedding vs Merged PTA</h2>
    <div class="summary-grid">{headline}</div>

    <h2>Full Comparison (Micro-Averaged)</h2>
    <p>Delta = Merged &minus; Embedding. <span style="color:var(--success)">Green</span> = merged is better.</p>
    {comp_table}

    <h2>Confusion Matrices</h2>{cm}

    <h2>Score Distributions</h2>
    <div class="charts-grid">
        <div class="chart-container"><div id="h_ef"></div></div>
        <div class="chart-container"><div id="h_ms"></div></div>
    </div>

    <h2>Per-Task Breakdown (AUROC)</h2>
    <table><thead><tr>
        <th>Task</th><th>Train</th><th>Test</th>
        <th>Embed F1 AUROC</th><th>Merged Struct AUROC</th><th>Delta</th><th>Merged States</th>
    </tr></thead><tbody>{task_rows}</tbody></table>

    <h2>Methodology</h2>
    <div style="background:var(--bg-secondary);padding:1.5rem;border-radius:12px;margin:1rem 0">
        <h3>Embedding Similarity (Baseline)</h3>
        <p>Each state in a trace is converted to text (tool + file + operation + resulting_state +
        truncated observation). These texts are embedded using {method_desc}.
        For each (candidate, reference) pair, we compute <em>BERTScore-style</em> greedy alignment:
        precision (avg best-match sim for each candidate state), recall (avg best-match sim for each
        reference state), and F1 (harmonic mean). The F1 is averaged across training traces and used
        as the discrimination score.</p>

        <h3 style="margin-top:1rem">Merged PTA Matching (Proposed)</h3>
        <p>Training traces are merged into a single DAG. Candidates are scored via greedy
        subsequence-coverage matching against this DAG, which captures the branching solution space.</p>

        <h3 style="margin-top:1rem">Why PTA Matching Should Win</h3>
        <ul style="margin:0.5rem 0 0 1.5rem;color:var(--text-secondary)">
            <li><strong>Structure vs. semantics:</strong> Embedding similarity captures lexical/semantic
                overlap but ignores the <em>ordering</em> and <em>causal structure</em> of states.
                Two traces can have high cosine similarity yet follow completely different solution paths.</li>
            <li><strong>Solution-space modelling:</strong> The merged PTA encodes which sequences of states
                constitute valid solutions. Embedding similarity has no notion of valid paths.</li>
            <li><strong>Noise tolerance:</strong> Failing traces often contain many of the same tool calls
                (reading files, running tests) as passing traces — semantically similar but structurally
                incorrect. PTA matching penalises wrong ordering; embedding similarity does not.</li>
        </ul>
    </div>
</div>
<script>{hist_js}</script></body></html>"""

    Path(output_path).write_text(html, encoding="utf-8")
    logger.info(f"HTML report saved to {output_path}")


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Embedding similarity baseline vs merged PTA matching"
    )
    parser.add_argument("data_dir", help="Directory containing trajectory data")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--merge-count", type=int, required=True, help=">= 2")
    parser.add_argument("--test-pass-count", type=int, required=True, help=">= 1")
    parser.add_argument("--test-fail-count", type=int, required=True, help=">= 1")
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--embedding-method",
        choices=["auto", "azure", "tfidf"],
        default="auto",
        help="Embedding method: azure (text-embedding-3-large), tfidf, or auto (default)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.merge_count < 2:
        print("ERROR: --merge-count must be >= 2")
        return 1
    if args.test_pass_count < 1:
        print("ERROR: --test-pass-count must be >= 1")
        return 1
    if args.test_fail_count < 1:
        print("ERROR: --test-fail-count must be >= 1")
        return 1
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    if not STATS_AVAILABLE:
        print("ERROR: pip install scikit-learn scipy numpy")
        return 1

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: {data_dir} not found")
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "__embedding_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(tempfile.mkdtemp(prefix="pta_emb_"))

    engine = EmbeddingEngine(method=args.embedding_method)

    print(f"Discovering trajectories in: {data_dir}")
    tasks = discover_trajectories(data_dir)
    print(f"Found {len(tasks)} tasks")

    min_passed = args.merge_count + args.test_pass_count
    eligible = {}
    for name, trajs in tasks.items():
        np_ = sum(1 for t in trajs if t.passed)
        nf = sum(1 for t in trajs if not t.passed)
        if np_ >= min_passed and nf >= args.test_fail_count:
            eligible[name] = trajs
    print(
        f"Eligible tasks (>= {min_passed} passed "
        f"[{args.merge_count} merge + {args.test_pass_count} test], "
        f">= {args.test_fail_count} failed): {len(eligible)}"
    )

    if not eligible:
        print("ERROR: No eligible tasks")
        return 1

    task_results: List[TaskEmbeddingResult] = []
    all_scores: List[EmbeddingTrajectoryScore] = []

    print(
        f"\nRunning embedding baseline (method={engine.method}, "
        f"merge_count={args.merge_count}) ..."
    )

    for task_name in tqdm(sorted(eligible.keys()), desc="Processing tasks"):
        result = run_task_embedding(
            task_name=task_name,
            trajectories=eligible[task_name],
            output_dir=output_dir,
            temp_dir=temp_dir,
            merge_count=args.merge_count,
            test_pass_count=args.test_pass_count,
            test_fail_count=args.test_fail_count,
            engine=engine,
            use_llm=args.use_llm,
            seed=args.seed,
        )

        if result:
            task_results.append(result)
            for tr_dict in result.trajectory_results:
                all_scores.append(EmbeddingTrajectoryScore(**{
                    k: v for k, v in tr_dict.items()
                    if k in EmbeddingTrajectoryScore.__dataclass_fields__
                }))

    if not task_results:
        print("ERROR: No tasks completed")
        return 1

    print(f"\nCompleted {len(task_results)} tasks")

    agg = compute_aggregate(task_results, all_scores)

    # Print summary
    ef = agg.embedding_f1
    er = agg.embedding_recall
    ms = agg.merged_structural
    mc = agg.merged_combined

    print("\n" + "=" * 90)
    print(f"EMBEDDING BASELINE ({agg.embedding_method.upper()}) vs MERGED PTA MATCHING")
    print("=" * 90)
    print(
        f"\nDataset: {agg.total_tasks} tasks, "
        f"{agg.total_test_passed} test passed, {agg.total_failed} test failed"
    )

    print(
        f"\n{'Metric':<20} {'Embed F1':>12} {'Embed Recall':>14} "
        f"{'Merged Struct':>14} {'Merged Comb':>14}"
    )
    print("-" * 80)
    print(
        f"{'AUROC':<20} {ef.auroc:>12.4f} {er.auroc:>14.4f} "
        f"{ms.auroc:>14.4f} {mc.auroc:>14.4f}"
    )
    print(
        f"{'KS-Statistic':<20} {ef.ks_statistic:>12.4f} {er.ks_statistic:>14.4f} "
        f"{ms.ks_statistic:>14.4f} {mc.ks_statistic:>14.4f}"
    )
    print(
        f"{'Accuracy':<20} {ef.accuracy:>11.1%} {er.accuracy:>13.1%} "
        f"{ms.accuracy:>13.1%} {mc.accuracy:>13.1%}"
    )
    print(
        f"{'F1':<20} {ef.f1:>12.4f} {er.f1:>14.4f} "
        f"{ms.f1:>14.4f} {mc.f1:>14.4f}"
    )

    print(
        f"\n{'Macro AUROC':<20} {'Embed F1':>12} {'Embed Recall':>14} "
        f"{'Merged Struct':>14}"
    )
    print("-" * 65)
    print(
        f"{'':>20} {agg.macro_embedding_f1_auroc:>12.4f} "
        f"{agg.macro_embedding_recall_auroc:>14.4f} "
        f"{agg.macro_merged_structural_auroc:>14.4f}"
    )

    # Save JSON
    results_json = {
        "generated_at": datetime.now().isoformat(),
        "experiment_type": "embedding_baseline",
        "experiment_config": {
            "data_dir": str(data_dir),
            "merge_count": args.merge_count,
            "test_pass_count": args.test_pass_count,
            "test_fail_count": args.test_fail_count,
            "seed": args.seed,
            "embedding_method": engine.method,
        },
        "aggregate_metrics": asdict(agg),
        "task_results": [asdict(t) for t in task_results],
    }
    results_path = output_dir / "embedding_baseline_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved results to: {results_path}")

    html_path = output_dir / "embedding_baseline_report.html"
    generate_report(agg, task_results, all_scores, str(html_path))
    print(f"Generated HTML report: {html_path}")

    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass

    print("\n" + "=" * 90)


if __name__ == "__main__":
    sys.exit(main() or 0)
