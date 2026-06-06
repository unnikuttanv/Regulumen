#!/usr/bin/env bash
#
# ReguLumen — HuggingFace Space startup: ingest-on-boot, then launch the UI.
#
# Contract for an HF Docker Space:
#   * The container MUST keep running (a hard crash here = a crash-loop on HF).
#   * Secrets (REGULUMEN_HF_API_TOKEN for embeddings, REGULUMEN_GROQ_API_KEY for
#     the LLM) arrive at RUNTIME as Space secrets. They may be MISSING on a fresh
#     Space — in that case we must still BOOT and let the UI show its built-in
#     "backend unavailable / set credentials" message per request.
#
# Therefore we deliberately do NOT use `set -e` around the ingest step: a failed
# ingest (missing HF token, no network) must NOT stop the Space from starting.
# We DO want a hard failure to surface if the ingest *script itself* is broken in
# a way that would also break the UI, but since the UI degrades gracefully we
# always proceed to launch it.

set -uo pipefail

echo "[hf_startup] REGULUMEN_CHROMA_PATH=${REGULUMEN_CHROMA_PATH:-<unset>}"
echo "[hf_startup] Building Chroma index from the bundled corpus ..."

# Build the real index. If the HF token is missing or the network is down,
# scripts/ingest.py exits non-zero (e.g. code 2) — we LOG and CONTINUE so the
# Space still boots. The UI then shows its "set secrets" message per question.
if python scripts/ingest.py; then
    echo "[hf_startup] Ingest complete — index built."
else
    echo "[hf_startup] WARNING: ingest failed (missing REGULUMEN_HF_API_TOKEN or"
    echo "[hf_startup]          network/provider issue). Starting the UI anyway;"
    echo "[hf_startup]          it will show a 'set secrets and restart' message."
    echo "[hf_startup]          Set the Space SECRETS, then restart the Space to"
    echo "[hf_startup]          build the index."
fi

echo "[hf_startup] Launching Gradio UI on ${GRADIO_SERVER_NAME:-0.0.0.0}:${GRADIO_SERVER_PORT:-7860} ..."

# exec so the UI becomes PID 1 and receives signals (clean HF restarts).
# It reads GRADIO_SERVER_NAME / GRADIO_SERVER_PORT from the environment.
exec python -m regulumen.ui.app
