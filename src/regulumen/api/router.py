"""Thin FastAPI routes.

Routes call the :class:`RegulumenService` ONLY — no business logic, no core
imports beyond the service + schemas + the provider error type (which is mapped
to a clean 503). The service is obtained via the :func:`get_service` dependency
so tests can override it with ``app.dependency_overrides``.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from regulumen.api.schemas import (
    AskRequest,
    AskResponse,
    ClassifyRequest,
    ClassifyResponse,
    HealthResponse,
)
from regulumen.api.service import RegulumenService
from regulumen.core.generation.llm import LLMError
from regulumen.core.retrieval.embedder import EmbeddingError

# Actionable message when a provider (LLM or embeddings) is unavailable —
# almost always a missing key. Surfaced as HTTP 503, never a 500 stack trace.
_PROVIDER_UNAVAILABLE = (
    "LLM/embedding provider unavailable: set REGULUMEN_GROQ_API_KEY / "
    "REGULUMEN_HF_API_TOKEN."
)

router = APIRouter()


def get_service() -> RegulumenService:
    """Dependency placeholder for the service.

    The app factory (:func:`regulumen.api.main.create_app`) overrides this with
    a lazily-built singleton; tests override it with a fake. The bare default
    fails loudly so a misconfigured app never silently 500s.
    """
    raise RuntimeError(
        "RegulumenService dependency is not configured. Build the app via "
        "regulumen.api.main.create_app()."
    )


@router.get("/health", response_model=HealthResponse)
def health(service: RegulumenService = Depends(get_service)) -> HealthResponse:
    return service.health()


@router.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    service: RegulumenService = Depends(get_service),
) -> AskResponse:
    try:
        return service.ask(request.question)
    except (LLMError, EmbeddingError) as exc:
        raise HTTPException(status_code=503, detail=_PROVIDER_UNAVAILABLE) from exc


@router.post("/classify", response_model=ClassifyResponse)
def classify(
    request: ClassifyRequest,
    service: RegulumenService = Depends(get_service),
) -> ClassifyResponse:
    try:
        return service.classify(request.description)
    except (LLMError, EmbeddingError) as exc:
        raise HTTPException(status_code=503, detail=_PROVIDER_UNAVAILABLE) from exc


__all__ = ["get_service", "router"]
