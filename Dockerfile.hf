# syntax=docker/dockerfile:1
#
# ReguLumen — HuggingFace Space (Docker SDK) image.
#
# This is the HF-Spaces variant of the local Dockerfile. It differs in three
# Space-specific ways that the local image deliberately does NOT have:
#
#   1. Runs as a NON-ROOT user with UID 1000 (HF runs containers as uid 1000;
#      a root-only image fails to write its data).
#   2. Serves a SINGLE port — the Gradio UI on 0.0.0.0:7860 (HF exposes one
#      port, declared via `app_port` in the README front-matter). The UI imports
#      the core in-process, so it is a complete app on its own.
#   3. Ingests-then-launches at boot: scripts/hf_startup.sh builds the Chroma
#      index from the bundled corpus, then execs the UI. Secrets
#      (REGULUMEN_GROQ_API_KEY / REGULUMEN_HF_API_TOKEN) arrive at RUNTIME as HF
#      Space secrets — they are NEVER baked into this image.
#
# As with the local image, embeddings are a hosted HTTP API, so the image stays
# lean: NO torch / sentence-transformers. chromadb (onnxruntime) is accepted.

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
# Stage 2: runtime — slim image, runs as the non-root `user` (uid 1000).
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

# Create a non-root user with UID 1000 + a real home (HF requirement).
RUN useradd --create-home --uid 1000 user

ENV HOME=/home/user \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Chroma index must live somewhere `user` (uid 1000) can write.
    REGULUMEN_CHROMA_PATH=/home/user/app/.chroma \
    # Make Gradio bind on all interfaces inside the container on the HF port.
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# App lives under the user's home so the whole tree is user-owned + writable.
WORKDIR /home/user/app

# Copy the pre-built virtualenv from the builder stage (owned by user).
COPY --from=builder --chown=user:user /opt/venv /opt/venv

# Copy application code + metadata + the bundled corpus (all owned by user).
COPY --chown=user:user pyproject.toml README.md ./
COPY --chown=user:user src/ ./src/
COPY --chown=user:user scripts/ ./scripts/
# Keep the corpus IN the image so startup ingest can populate the index.
COPY --chown=user:user corpus/ ./corpus/

# Re-install the package metadata against the copied src/ (cheap; no deps).
RUN pip install --no-deps -e .

# Drop privileges: everything below runs as uid 1000.
USER user

# HF Spaces expose ONE port; serve the Gradio UI here.
EXPOSE 7860

# Ingest-then-launch. Uses bash so the script's graceful-degrade logic runs.
CMD ["bash", "scripts/hf_startup.sh"]
