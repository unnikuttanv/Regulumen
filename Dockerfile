# syntax=docker/dockerfile:1
#
# ReguLumen — multi-stage image for one-command local spin-up.
#
# Embeddings are a HOSTED HTTP API (HuggingFace Inference, via httpx), so the
# image stays lean: NO torch / sentence-transformers. The heaviest dependency
# pulled in is chromadb (which brings onnxruntime); that is accepted.
#
# The image can run EITHER the API or the UI — the CMD is overridable, so
# docker-compose points each service at the right command.

# ---------------------------------------------------------------------------
# Stage 1: builder — install dependencies + the package into a venv.
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Build a self-contained virtualenv we can copy into the runtime stage.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install third-party deps first (best layer caching).
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Install the package itself (editable so src/ layout resolves at runtime).
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install -e .

# ---------------------------------------------------------------------------
# Stage 2: runtime — slim image with only the venv + app code + corpus.
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    REGULUMEN_CHROMA_PATH=/app/.chroma

WORKDIR /app

# Copy the pre-built virtualenv from the builder stage.
COPY --from=builder /opt/venv /opt/venv

# Copy application code + the editable-install metadata.
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/
# Keep the corpus IN the image so the ingest service can populate the index.
COPY corpus/ ./corpus/

# Re-install the package metadata against the copied src/ (cheap; no deps).
RUN pip install --no-deps -e .

# API on 8000, Gradio UI on 7860. Both are documented; compose picks one each.
EXPOSE 8000 7860

# Default to the API; compose / `docker run ... <cmd>` overrides for the UI
# or the one-shot ingest job.
CMD ["uvicorn", "regulumen.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
