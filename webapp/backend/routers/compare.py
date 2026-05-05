"""Batch comparison endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..schemas import BatchAssessResponse
from ..services.pipeline import assess_batch

router = APIRouter(prefix="/api", tags=["compare"])


@router.post("/assess/batch", response_model=BatchAssessResponse)
async def batch_assess() -> BatchAssessResponse:
    """Assess all trajectories against ground truth and return cohort ranking."""
    try:
        data = assess_batch()
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Batch assessment failed: {e}")
    return BatchAssessResponse(**data)
