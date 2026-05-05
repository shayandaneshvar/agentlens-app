"""Analysis and assessment endpoints."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from ..schemas import (
    ProfileResponse,
    GroundTruthResponse,
    AssessResponse,
    AssessWithGTResponse,
    LLMAssessResponse,
    LLMSuggestionsResponse,
    CompareResponse,
    LLMCompareResponse,
    MergeRequest,
    CompareRequest,
)
from ..services.pipeline import (
    get_profile,
    build_ground_truth,
    assess_trajectory,
    assess_with_uploaded_gt,
    assess_with_imported_gt,
    export_ground_truth,
    import_ground_truth,
    get_visualization,
    llm_assess,
    llm_suggestions,
    compare_traces,
    llm_compare,
)

router = APIRouter(prefix="/api", tags=["analysis"])


@router.get("/traces/{trace_id}/profile", response_model=ProfileResponse)
async def trace_profile(trace_id: str) -> ProfileResponse:
    """Tier 1: single trajectory profile (always available)."""
    try:
        data = get_profile(trace_id)
    except KeyError:
        raise HTTPException(404, "Trace not found")
    return ProfileResponse(**data)


@router.get("/traces/{trace_id}/visualization")
async def trace_visualization(trace_id: str) -> dict:
    """Full trace data for DAG rendering."""
    try:
        return get_visualization(trace_id)
    except KeyError:
        raise HTTPException(404, "Trace not found")


@router.post("/merge", response_model=GroundTruthResponse)
async def merge_traces(body: MergeRequest) -> GroundTruthResponse:
    """Merge passing traces into ground truth."""
    if len(body.trace_ids) < 2:
        raise HTTPException(400, "Need at least 2 traces to merge")
    try:
        data = build_ground_truth(body.trace_ids)
    except KeyError as e:
        raise HTTPException(404, f"Trace not found: {e}")
    except Exception as e:
        raise HTTPException(500, f"Merge failed: {e}")
    return GroundTruthResponse(**data)


@router.post("/assess/{trace_id}", response_model=AssessResponse)
async def assess(trace_id: str) -> AssessResponse:
    """Tier 2: assess a single trajectory against ground truth."""
    try:
        data = assess_trajectory(trace_id)
    except KeyError:
        raise HTTPException(404, "Trace not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return AssessResponse(**data)


@router.post("/assess-with-gt/{trace_id}", response_model=AssessWithGTResponse)
async def assess_with_gt(
    trace_id: str,
    files: List[UploadFile] = File(...),
) -> AssessWithGTResponse:
    """Upload passing trajectory zips, merge into GT, and assess the target trace."""
    if len(files) < 2:
        raise HTTPException(400, "Upload at least 2 passing trajectory files")
    gt_files = []
    for f in files:
        contents = await f.read()
        gt_files.append((contents, f.filename or "unknown.zip"))
    try:
        data = assess_with_uploaded_gt(trace_id, gt_files)
    except KeyError:
        raise HTTPException(404, "Trace not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Assessment failed: {e}")
    return AssessWithGTResponse(**data)


@router.get("/gt/export")
async def gt_export() -> JSONResponse:
    """Export the current ground truth as a downloadable JSON."""
    try:
        data = export_ground_truth()
    except ValueError as e:
        raise HTTPException(400, str(e))
    return JSONResponse(content=data, headers={
        "Content-Disposition": "attachment; filename=merged_gt.json"
    })


@router.post("/gt/import", response_model=GroundTruthResponse)
async def gt_import(file: UploadFile = File(...)) -> GroundTruthResponse:
    """Import a previously exported merged GT JSON file."""
    import json as json_mod
    contents = await file.read()
    try:
        gt_data = json_mod.loads(contents.decode("utf-8"))
    except (json_mod.JSONDecodeError, UnicodeDecodeError) as e:
        raise HTTPException(400, f"Invalid JSON: {e}")
    try:
        data = import_ground_truth(gt_data)
    except Exception as e:
        raise HTTPException(500, f"Import failed: {e}")
    return GroundTruthResponse(**data)


@router.post("/assess-with-imported-gt/{trace_id}", response_model=AssessWithGTResponse)
async def assess_with_imported_gt_endpoint(
    trace_id: str,
    file: UploadFile = File(...),
) -> AssessWithGTResponse:
    """Assess a trajectory against an imported merged GT JSON file."""
    import json as json_mod
    contents = await file.read()
    try:
        gt_data = json_mod.loads(contents.decode("utf-8"))
    except (json_mod.JSONDecodeError, UnicodeDecodeError) as e:
        raise HTTPException(400, f"Invalid JSON: {e}")
    try:
        data = assess_with_imported_gt(trace_id, gt_data)
    except KeyError:
        raise HTTPException(404, "Trace not found")
    except Exception as e:
        raise HTTPException(500, f"Assessment failed: {e}")
    return AssessWithGTResponse(**data)


@router.post("/traces/{trace_id}/llm-assess", response_model=LLMAssessResponse)
async def llm_assess_endpoint(trace_id: str) -> LLMAssessResponse:
    """Run LLM-based behavioral assessment on a trajectory.

    Requires a prior Tier 2 PTA assessment (ground truth must exist).
    The LLM receives both the PTA matching results and the full trajectory
    structure to produce a holistic behavioral evaluation.
    """
    try:
        data = llm_assess(trace_id)
    except KeyError:
        raise HTTPException(404, "Trace not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"LLM assessment failed: {e}")
    return LLMAssessResponse(**data)


@router.post("/traces/{trace_id}/llm-suggestions", response_model=LLMSuggestionsResponse)
async def llm_suggestions_endpoint(trace_id: str) -> LLMSuggestionsResponse:
    """Generate LLM-based actionable improvement suggestions.

    Requires a prior Tier 2 PTA assessment (ground truth must exist).
    Focused on diagnosing inefficiencies and suggesting concrete fixes.
    """
    try:
        data = llm_suggestions(trace_id)
    except KeyError:
        raise HTTPException(404, "Trace not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"LLM suggestions failed: {e}")
    return LLMSuggestionsResponse(**data)


@router.post("/compare", response_model=CompareResponse)
async def compare_endpoint(req: CompareRequest) -> CompareResponse:
    """Compare multiple candidate trajectories against the same GT."""
    try:
        data = compare_traces(req.trace_ids, gt_strategy=req.gt_strategy)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Comparison failed: {e}")
    return CompareResponse(**data)


@router.post("/compare/llm", response_model=LLMCompareResponse)
async def llm_compare_endpoint(req: CompareRequest) -> LLMCompareResponse:
    """Run comparative LLM assessment across multiple trajectories."""
    try:
        data = llm_compare(req.trace_ids)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"LLM comparison failed: {e}")
    return LLMCompareResponse(**data)
