"""Application configuration."""

from __future__ import annotations

import os


# Maximum upload file size (100 MB)
MAX_UPLOAD_SIZE = int(os.getenv("AGENTLENS_MAX_UPLOAD_MB", "100")) * 1024 * 1024

# Server port
PORT = int(os.getenv("AGENTLENS_PORT", "8000"))

# Allowed CORS origins (comma-separated)
CORS_ORIGINS = os.getenv("AGENTLENS_CORS_ORIGINS", "http://localhost:5173,http://localhost:8000").split(",")
