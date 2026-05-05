#!/bin/bash
set -e

# Install local packages if not already installed
if ! python -c "import swe_trace_sdk" 2>/dev/null; then
    echo "Installing SDK and webapp packages..."
    pip install --no-cache-dir ./sdk ./webapp
fi

# Start the server
exec gunicorn webapp.backend.app:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 600
