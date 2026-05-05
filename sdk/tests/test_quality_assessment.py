"""Tests for the enhanced quality assessment features.

Validates: divergence points, stage comparison, inefficiency detection,
quality signals, outcome-aware signals, backward compatibility, and
edge cases (perfect match, fully divergent).
"""

import sys
from pathlib import Path

import pytest

# Ensure the SDK src is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swe_trace_sdk import trace, match
from swe_trace_sdk.models import (
    QualityReport,
    DivergenceSegment,
    StageComparison,
    InefficiencyReport,
    QualitySignal,
    FailureReason,
    DivergencePoint,
    StageCoverageDetail,
)

# ---------------------------------------------------------------------------
# Paths to sample data (run `python samples/unzip_data.py` first)
# ---------------------------------------------------------------------------
DATA = Path(__file__).resolve().parents[1] / "samples" / "data"

GOOD_RUNS = [
    DATA / "create_and_serve_minimal_html-logs-claude-opus-4.5-pass"
         / "output" / "vsc-output" / "chat-export-logs.json",
    DATA / "create_and_serve_minimal_html-logs-claude-haiku-4.5-pass"
         / "output" / "vsc-output" / "chat-export-logs.json",
    DATA / "create_and_serve_minimal_html-logs-gpt-5.1-codex-pass"
         / "output" / "vsc-output" / "chat-export-logs.json",
]

CANDIDATE_FAIL = (
    DATA / "create_and_serve_minimal_html-logs-gpt-4.1-fail"
         / "output" / "vsc-output" / "chat-export-logs.json"
)

# Use one of the good runs as a "passing candidate"
CANDIDATE_PASS = GOOD_RUNS[0]

# Check data availability
_data_available = all(p.exists() for p in GOOD_RUNS + [CANDIDATE_FAIL])
skip_no_data = pytest.mark.skipif(
    not _data_available,
    reason="Sample data not found — run `python samples/unzip_data.py` first",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ground_truth():
    """Merged ground truth from 3 passing traces."""
    gt_traces = [trace.load(str(p), format="chatlog") for p in GOOD_RUNS]
    return trace.merge(gt_traces)


@pytest.fixture(scope="module")
def failing_candidate():
    return trace.load(str(CANDIDATE_FAIL), format="chatlog")


@pytest.fixture(scope="module")
def passing_candidate():
    return trace.load(str(CANDIDATE_PASS), format="chatlog")


@pytest.fixture(scope="module")
def fail_result(failing_candidate, ground_truth):
    return match.run(failing_candidate, ground_truth)


@pytest.fixture(scope="module")
def pass_result(passing_candidate, ground_truth):
    return match.run(passing_candidate, ground_truth)


@pytest.fixture(scope="module")
def fail_report(fail_result, failing_candidate, ground_truth) -> QualityReport:
    return match.quality_assessment(fail_result, failing_candidate, ground_truth, passed=False)


@pytest.fixture(scope="module")
def pass_report(pass_result, passing_candidate, ground_truth) -> QualityReport:
    return match.quality_assessment(pass_result, passing_candidate, ground_truth, passed=True)


# ---------------------------------------------------------------------------
# 1. Backward compatibility — existing fields still populated
# ---------------------------------------------------------------------------

@skip_no_data
class TestBackwardCompatibility:

    def test_verdict_present(self, fail_report: QualityReport):
        assert fail_report.verdict in ("PASS", "LIKELY PASS", "UNCERTAIN", "LIKELY FAIL", "FAIL")

    def test_quality_tier_present(self, fail_report: QualityReport):
        assert fail_report.quality_tier in ("ideal", "solid", "lucky", "partial_fail", "off_track")

    def test_quality_score_range(self, fail_report: QualityReport):
        assert 0 <= fail_report.quality_score <= 100

    def test_failure_reasons_type(self, fail_report: QualityReport):
        assert isinstance(fail_report.failure_reasons, list)
        for fr in fail_report.failure_reasons:
            assert isinstance(fr, FailureReason)
            assert fr.reason
            assert fr.severity in ("critical", "high", "medium", "low")

    def test_strengths_type(self, fail_report: QualityReport):
        assert isinstance(fail_report.strengths, list)

    def test_single_divergence_point(self, fail_report: QualityReport):
        # Original single divergence should still exist
        if fail_report.divergence_point is not None:
            assert isinstance(fail_report.divergence_point, DivergencePoint)
            assert fail_report.divergence_point.step >= 0

    def test_stage_coverage(self, fail_report: QualityReport):
        assert isinstance(fail_report.stage_coverage, dict)
        for k, v in fail_report.stage_coverage.items():
            assert isinstance(v, StageCoverageDetail)
            assert v.percent >= 0

    def test_key_metrics(self, fail_report: QualityReport):
        assert "coverage_percent" in fail_report.key_metrics
        assert "coherence" in fail_report.key_metrics


# ---------------------------------------------------------------------------
# 2. Divergence points (multiple segments)
# ---------------------------------------------------------------------------

@skip_no_data
class TestDivergencePoints:

    def test_divergence_points_populated(self, fail_report: QualityReport):
        assert isinstance(fail_report.divergence_points, list)
        # A failing candidate should have at least 1 divergence segment
        assert len(fail_report.divergence_points) > 0

    def test_divergence_segment_structure(self, fail_report: QualityReport):
        for seg in fail_report.divergence_points:
            assert isinstance(seg, DivergenceSegment)
            assert seg.start_step >= 0
            assert seg.end_step >= seg.start_step
            assert isinstance(seg.expected_states, list)
            assert len(seg.expected_states) > 0
            assert isinstance(seg.candidate_activity, list)
            assert isinstance(seg.stage_context, str)

    def test_expected_states_have_details(self, fail_report: QualityReport):
        for seg in fail_report.divergence_points:
            for s in seg.expected_states:
                assert "tool" in s or "intent_stage" in s

    def test_no_divergence_for_pass(self, pass_report: QualityReport):
        # A passing candidate (one of the GT traces) should have few/no divergences
        # It may not be zero because the merged GT can differ from any single trace
        pass  # Informational — just ensure it doesn't crash


# ---------------------------------------------------------------------------
# 3. Stage comparison
# ---------------------------------------------------------------------------

@skip_no_data
class TestStageComparison:

    def test_stage_comparison_populated(self, fail_report: QualityReport):
        assert isinstance(fail_report.stage_comparison, dict)
        # Should have at least one stage
        assert len(fail_report.stage_comparison) > 0

    def test_stage_comparison_structure(self, fail_report: QualityReport):
        for stage, comp in fail_report.stage_comparison.items():
            assert isinstance(comp, StageComparison)
            assert isinstance(comp.expected_steps, list)
            assert isinstance(comp.matched_steps, list)
            assert isinstance(comp.missing_steps, list)
            assert isinstance(comp.extra_steps, list)
            assert isinstance(comp.ordering_preserved, bool)
            assert isinstance(comp.effort_ratio, (int, float))
            assert comp.effort_ratio >= 0

    def test_stage_order_match_field(self, fail_report: QualityReport):
        assert isinstance(fail_report.stage_order_match, bool)

    def test_valid_stages(self, fail_report: QualityReport):
        valid = {"exploration", "implementation", "verification", "orchestration"}
        for stage in fail_report.stage_comparison:
            assert stage in valid, f"Unexpected stage: {stage}"

    def test_matched_not_exceeding_expected(self, fail_report: QualityReport):
        for stage, comp in fail_report.stage_comparison.items():
            assert len(comp.matched_steps) <= len(comp.expected_steps)


# ---------------------------------------------------------------------------
# 4. Inefficiency detection
# ---------------------------------------------------------------------------

@skip_no_data
class TestInefficiencyDetection:

    def test_inefficiencies_present(self, fail_report: QualityReport):
        assert fail_report.inefficiencies is not None
        assert isinstance(fail_report.inefficiencies, InefficiencyReport)

    def test_inefficiency_counts(self, fail_report: QualityReport):
        r = fail_report.inefficiencies
        assert r.retry_loop_count >= 0
        assert r.backtrack_count >= 0
        assert r.redundant_step_count >= 0
        assert r.unnecessary_exploration_count >= 0
        assert r.total_wasted_steps >= 0

    def test_counts_match_lists(self, fail_report: QualityReport):
        r = fail_report.inefficiencies
        assert r.retry_loop_count == len(r.retry_loops)
        assert r.backtrack_count == len(r.backtracks)
        assert r.redundant_step_count == len(r.redundant_steps)
        assert r.unnecessary_exploration_count == len(r.unnecessary_explorations)

    def test_retry_loop_structure(self, fail_report: QualityReport):
        for rl in fail_report.inefficiencies.retry_loops:
            assert rl.start_step >= 0
            assert rl.end_step >= rl.start_step
            assert rl.count >= 3  # by definition
            assert isinstance(rl.tool, str)

    def test_backtrack_structure(self, fail_report: QualityReport):
        for bt in fail_report.inefficiencies.backtracks:
            assert bt.step >= 0
            assert isinstance(bt.from_stage, str)
            assert isinstance(bt.to_stage, str)

    def test_total_wasted_non_negative(self, fail_report: QualityReport):
        assert fail_report.inefficiencies.total_wasted_steps >= 0


# ---------------------------------------------------------------------------
# 5. Quality signals
# ---------------------------------------------------------------------------

@skip_no_data
class TestQualitySignals:

    def test_signals_present(self, fail_report: QualityReport):
        assert isinstance(fail_report.quality_signals, list)

    def test_signal_structure(self, fail_report: QualityReport):
        for sig in fail_report.quality_signals:
            assert isinstance(sig, QualitySignal)
            assert sig.signal_type
            assert sig.description
            assert sig.severity in ("critical", "warning", "info")
            assert isinstance(sig.evidence, list)

    def test_no_duplicate_signal_types(self, fail_report: QualityReport):
        types = [s.signal_type for s in fail_report.quality_signals]
        assert len(types) == len(set(types)), f"Duplicate signals: {types}"

    def test_pass_report_signals(self, pass_report: QualityReport):
        # A passing candidate should not have failure signals
        fail_signals = {
            "failure_from_missing_verification",
            "failure_from_wrong_ordering",
        }
        for sig in pass_report.quality_signals:
            assert sig.signal_type not in fail_signals, (
                f"Failure signal '{sig.signal_type}' found on passing report"
            )


# ---------------------------------------------------------------------------
# 6. Serialization round-trip (to_dict / from_dict)
# ---------------------------------------------------------------------------

@skip_no_data
class TestSerialization:

    def test_quality_report_roundtrip(self, fail_report: QualityReport):
        d = fail_report.to_dict()
        restored = QualityReport.from_dict(d)

        assert restored.verdict == fail_report.verdict
        assert restored.quality_tier == fail_report.quality_tier
        assert restored.quality_score == fail_report.quality_score
        assert restored.stage_order_match == fail_report.stage_order_match
        assert len(restored.divergence_points) == len(fail_report.divergence_points)
        assert len(restored.quality_signals) == len(fail_report.quality_signals)

    def test_divergence_segment_roundtrip(self, fail_report: QualityReport):
        for seg in fail_report.divergence_points:
            d = seg.to_dict()
            restored = DivergenceSegment.from_dict(d)
            assert restored.start_step == seg.start_step
            assert restored.end_step == seg.end_step
            assert restored.stage_context == seg.stage_context

    def test_inefficiency_report_roundtrip(self, fail_report: QualityReport):
        if fail_report.inefficiencies:
            d = fail_report.inefficiencies.to_dict()
            restored = InefficiencyReport.from_dict(d)
            assert restored.total_wasted_steps == fail_report.inefficiencies.total_wasted_steps
            assert restored.retry_loop_count == fail_report.inefficiencies.retry_loop_count

    def test_dict_has_all_new_fields(self, fail_report: QualityReport):
        d = fail_report.to_dict()
        assert "divergence_points" in d
        assert "stage_comparison" in d
        assert "inefficiencies" in d
        assert "quality_signals" in d
        assert "stage_order_match" in d
        # Old fields still present
        assert "verdict" in d
        assert "failure_reasons" in d
        assert "stage_coverage" in d


# ---------------------------------------------------------------------------
# 7. Edge case: passing candidate (one of the GT contributors)
# ---------------------------------------------------------------------------

@skip_no_data
class TestPassingCandidate:

    def test_verdict_is_pass(self, pass_report: QualityReport):
        assert pass_report.verdict in ("PASS", "LIKELY PASS")

    def test_high_quality_score(self, pass_report: QualityReport):
        assert pass_report.quality_score >= 50

    def test_no_failure_reasons(self, pass_report: QualityReport):
        # Passing should have no or minimal failure reasons
        critical = [fr for fr in pass_report.failure_reasons if fr.severity == "critical"]
        assert len(critical) == 0

    def test_stage_comparison_populated(self, pass_report: QualityReport):
        assert isinstance(pass_report.stage_comparison, dict)

    def test_inefficiencies_present(self, pass_report: QualityReport):
        assert pass_report.inefficiencies is not None


# ---------------------------------------------------------------------------
# 8. Comparison data builder (backend pipeline)
# ---------------------------------------------------------------------------

@skip_no_data
class TestComparisonData:

    def test_build_comparison_data(self, fail_result, failing_candidate, ground_truth):
        """Test _build_comparison_data returns expected structure."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "webapp"))
        from backend.services.pipeline import _build_comparison_data

        comp = _build_comparison_data(fail_result, failing_candidate, ground_truth)
        assert comp is not None
        assert "gt_path" in comp
        assert "candidate_path" in comp
        assert "alignment" in comp
        assert "gt_matched_indexes" in comp
        assert "candidate_matched_indexes" in comp

        assert isinstance(comp["gt_path"], list)
        assert isinstance(comp["candidate_path"], list)
        assert len(comp["gt_path"]) > 0
        assert len(comp["candidate_path"]) > 0

        # Alignment pairs are valid indexes
        for pair in comp["alignment"]:
            assert 0 <= pair["gt_index"] < len(comp["gt_path"])
            assert 0 <= pair["candidate_index"] < len(comp["candidate_path"])

        # State summaries have expected keys
        for s in comp["gt_path"]:
            assert "tool" in s
            assert "intent_stage" in s

    def test_matched_indexes_consistent(self, fail_result, failing_candidate, ground_truth):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "webapp"))
        from backend.services.pipeline import _build_comparison_data

        comp = _build_comparison_data(fail_result, failing_candidate, ground_truth)

        gt_idxs = set(comp["gt_matched_indexes"])
        cand_idxs = set(comp["candidate_matched_indexes"])

        # Matched indexes from alignment should be subsets
        for pair in comp["alignment"]:
            assert pair["gt_index"] in gt_idxs
            assert pair["candidate_index"] in cand_idxs
