"""FastAPI entry point for AgentLens."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import CORS_ORIGINS
from .routers import upload, analysis, compare

app = FastAPI(
    title="AgentLens",
    version="0.1.0",
    description="Qualitative assessment dashboard for coding agent trajectories.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(compare.router)
app.include_router(analysis.router)


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "app": "AgentLens"}


# Serve frontend static files if they exist (production build)
_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")
