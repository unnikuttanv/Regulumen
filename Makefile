# ReguLumen developer task runner.
#
# Local targets use the project virtualenv python. On Windows the venv python is
# at .venv\Scripts\python.exe; on POSIX it is .venv/bin/python. Override with:
#   make test PY=python
#
# If `make` is not installed (typical on Windows), every recipe below is a
# single copy-pasteable command.

PY ?= .venv/bin/python

.PHONY: install test lint ingest ingest-fake api ui up down

install:  ## Install runtime + dev dependencies and the package (editable).
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

test:  ## Run the full test suite.
	$(PY) -m pytest -q

lint:  ## Lint with ruff.
	$(PY) -m ruff check .

ingest:  ## Build the real Chroma index (needs REGULUMEN_HF_API_TOKEN).
	$(PY) scripts/ingest.py

ingest-fake:  ## Build an offline index with the FakeEmbedder (no token/network).
	$(PY) scripts/ingest.py --use-fake-embedder

api:  ## Run the FastAPI app locally on :8000.
	$(PY) -m uvicorn regulumen.api.main:app --host 0.0.0.0 --port 8000

ui:  ## Run the Gradio UI locally on :7860.
	GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860 $(PY) -m regulumen.ui.app

up:  ## Build + start API and UI via docker compose.
	docker compose up --build api ui

down:  ## Stop and remove the compose stack.
	docker compose down
