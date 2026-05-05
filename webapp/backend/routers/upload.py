"""Upload and trace management endpoints."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File

from ..config import MAX_UPLOAD_SIZE
from ..schemas import TraceInfo, TraceListResponse, UpdateTraceRequest
from ..services.store import store, TraceRecord
from ..services.pipeline import process_upload

router = APIRouter(prefix="/api", tags=["upload"])


def _rec_to_info(rec: TraceRecord) -> TraceInfo:
    return TraceInfo(
        trace_id=rec.trace_id,
        label=rec.label,
        format=rec.format,
        passed=rec.passed,
        state_count=rec.state_count,
        tool_count=rec.tool_count,
        file_count=rec.file_count,
        model=rec.model,
        task=rec.task,
        benchmark=rec.benchmark,
    )


@router.post("/upload", response_model=List[TraceInfo])
async def upload_trajectory(file: UploadFile = File(...)) -> List[TraceInfo]:
    """Upload a trajectory file (zip or JSON).

    Returns a list of traces — usually one, but ATIF session ZIPs may
    contain multiple agent trajectories.
    """
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(413, "File too large")

    filename = file.filename or "unknown.json"
    try:
        records = process_upload(contents, filename)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to parse trajectory: {e}")

    return [_rec_to_info(r) for r in records]


@router.post("/upload-batch", response_model=List[TraceInfo])
async def upload_batch(files: List[UploadFile] = File(...)) -> List[TraceInfo]:
    """Upload multiple trajectory files at once."""
    results: List[TraceInfo] = []
    for f in files:
        contents = await f.read()
        if len(contents) > MAX_UPLOAD_SIZE:
            continue
        filename = f.filename or "unknown.json"
        if not (filename.lower().endswith(".zip") or filename.lower().endswith(".json")):
            continue
        try:
            records = process_upload(contents, filename)
            results.extend(_rec_to_info(r) for r in records)
        except Exception:
            continue  # skip files that fail to parse
    return results


@router.get("/traces", response_model=TraceListResponse)
async def list_traces() -> TraceListResponse:
    """List all uploaded traces."""
    records = store.list_records()
    gt_id = store.ground_truth_id
    return TraceListResponse(
        traces=[
            _rec_to_info(r)
            for r in records
            if r.format != "merged"  # hide all merged GTs from normal list
        ],
        passing_count=len(store.passing_ids()),
        has_ground_truth=gt_id is not None,
        ground_truth_id=gt_id,
    )


@router.patch("/traces/{trace_id}", response_model=TraceInfo)
async def update_trace(trace_id: str, body: UpdateTraceRequest) -> TraceInfo:
    """Update trace metadata (label, pass/fail)."""
    rec = store.get_record(trace_id)
    if rec is None:
        raise HTTPException(404, "Trace not found")
    if body.label is not None:
        rec.label = body.label
    if body.passed is not None:
        rec.passed = body.passed
    return _rec_to_info(rec)


@router.delete("/traces/{trace_id}")
async def delete_trace(trace_id: str) -> dict:
    """Delete an uploaded trace."""
    if not store.delete(trace_id):
        raise HTTPException(404, "Trace not found")
    return {"deleted": trace_id}
