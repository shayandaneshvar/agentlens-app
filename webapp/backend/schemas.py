"""Pydantic request/response schemas for the API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# ── Responses ──────────────────────────────────────────────────────────────

class TraceInfo(BaseModel):
    trace_id: str
    label: str
    format: str
    passed: Optional[bool] = None
    state_count: int
    tool_count: int
    file_count: int
    model: str = ""
    task: str = ""
    benchmark: str = ""


class TraceListResponse(BaseModel):
    traces: List[TraceInfo]
    passing_count: int
    has_ground_truth: bool
    ground_truth_id: Optional[str] = None


class ProfileResponse(BaseModel):
    trace_id: str
    state_count: int
    file_count: int
    tool_count: int
    coherence: float
    coherence_label: str
    stage_distribution: Dict[str, int]
    stage_percentages: Dict[str, float]
    tool_distribution: Dict[str, int]
    files_touched: List[str]
    fingerprint: str
    fingerprint_detail: List[str]
    operation_types: Dict[str, int]
    completed: Optional[bool] = None
    stage_sequence: List[str]
    tool_sequence: List[str]
    exploration_ratio: float
    files_modified: int
    files_read_only: int
    model: str = ""
    agent: str = ""
    task: str = ""
    benchmark: str = ""
    human_input_count: Optional[int] = None
    subagent_count: Optional[int] = None
    active_time_ms: Optional[int] = None
    compaction_count: Optional[int] = None
    # Human Experience metrics (ATIF only — null for other formats)
    wall_time_ms: Optional[int] = None
    permission_wait_ms: Optional[int] = None
    human_experience_score: Optional[float] = None
    hx_breakdown: Optional[Dict[str, float]] = None
    time_decomposition: Optional[Dict[str, int]] = None
    step_latencies: Optional[List[int]] = None
    step_token_cumulative: Optional[List[int]] = None
    human_input_positions: Optional[List[int]] = None


class GroundTruthResponse(BaseModel):
    gt_id: str
    source_count: int
    state_count: int


class AssessResponse(BaseModel):
    trace_id: str
    match_metrics: Dict[str, Any]
    quality_report: Dict[str, Any]


class AssessWithGTResponse(BaseModel):
    trace_id: str
    gt_source_count: int
    gt_state_count: int
    match_metrics: Dict[str, Any]
    quality_report: Dict[str, Any]
    process_coverage: float
    missing_tools: List[str]
    file_coverage: float
    missing_files: List[str]
    comparison: Optional[Dict[str, Any]] = None


class BatchAssessResponse(BaseModel):
    ranking: Dict[str, Any]
    trajectories: List[Dict[str, Any]]


class LLMAssessResponse(BaseModel):
    trace_id: str
    model_used: str
    assessment: Dict[str, Any]
    quality_score: int
    verdict: str


class LLMSuggestionsResponse(BaseModel):
    trace_id: str
    model_used: str
    suggestions: List[Dict[str, Any]]
    improvement_summary: str


class CompareResponse(BaseModel):
    candidates: List[Dict[str, Any]]
    gt: Dict[str, Any]
    gt_state_ids: List[str]


class LLMCompareResponse(BaseModel):
    individual: List[Dict[str, Any]]
    comparative: Dict[str, Any]
    model_used: str


# ── Requests ───────────────────────────────────────────────────────────────

class MergeRequest(BaseModel):
    trace_ids: List[str]


class CompareRequest(BaseModel):
    trace_ids: List[str]
    gt_strategy: str = "best_match"


class UpdateTraceRequest(BaseModel):
    label: Optional[str] = None
    passed: Optional[bool] = None
