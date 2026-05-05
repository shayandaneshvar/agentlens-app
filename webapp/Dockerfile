# ── Stage 1: build frontend ──────────────────────────────────────────────
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY webapp/frontend/package.json webapp/frontend/package-lock.json ./
RUN npm ci
COPY webapp/frontend/ .
RUN npm run build

# ── Stage 2: Python runtime ─────────────────────────────────────────────
FROM python:3.11-slim
WORKDIR /app

# Install SDK first (layer cache)
COPY sdk/ /app/sdk/
RUN pip install --no-cache-dir /app/sdk

# Install webapp
COPY webapp/pyproject.toml /app/webapp/pyproject.toml
COPY webapp/backend/ /app/webapp/backend/
RUN pip install --no-cache-dir /app/webapp

# Copy built frontend into static serving path
COPY --from=frontend-build /app/frontend/dist /app/webapp/frontend/dist

ENV PORT=8000
EXPOSE 8000

CMD ["uvicorn", "webapp.backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
