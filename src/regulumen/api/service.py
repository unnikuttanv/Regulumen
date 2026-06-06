"""Orchestration layer: wires domain objects to API DTOs.

No HTTP here (no FastAPI imports) and no Chroma/embedder construction (that
lives in the repository). The service owns the answerer, the classifier and the
repository, calls them, and maps their results through ``schemas``.
"""

from __future__ import annotations

from regulumen.api.repository import RetrievalRepository
from regulumen.api.schemas import (
    AskResponse,
    ClassifyResponse,
    HealthResponse,
    answer_result_to_response,
    risk_assessment_to_response,
)


def _build_answerer(settings, retriever):
    """Pick the Q&A orchestrator per ``settings.answerer_mode``.

    "graph" (default) uses the LangGraph answerer with a bounded faithfulness
    refine loop; "linear" uses the single-pass Answerer. Both expose the same
    ``.answer(question) -> AnswerResult`` interface and enforce the same gate, so
    the rest of the service is agnostic to which one is wired.
    """
    mode = getattr(settings, "answerer_mode", "graph")
    if mode == "linear":
        from regulumen.core.generation.answerer import build_answerer  # noqa: PLC0415

        return build_answerer(settings, retriever)
    from regulumen.core.generation.graph_answerer import (  # noqa: PLC0415
        build_graph_answerer,
    )

    return build_graph_answerer(settings, retriever)


class RegulumenService:
    """Application service orchestrating answering, classification and health."""

    def __init__(self, answerer, classifier, repository: RetrievalRepository) -> None:
        self._answerer = answerer
        self._classifier = classifier
        self._repository = repository

    @classmethod
    def build_from_settings(cls, settings) -> RegulumenService:
        """Composition root: build repository -> retriever -> answerer/classifier.

        Needs real provider keys at runtime (it constructs the embedder and the
        LLM). Tests use the plain ``__init__`` with fakes instead. Imports are
        local so this module imports cleanly without keys.
        """
        from regulumen.core.risk.classifier import (  # noqa: PLC0415
            build_risk_classifier,
        )

        repository = RetrievalRepository.build_from_settings(settings)
        retriever = repository.get_retriever()
        answerer = _build_answerer(settings, retriever)
        classifier = build_risk_classifier(settings, retriever=retriever)
        return cls(answerer, classifier, repository)

    def ask(self, question: str) -> AskResponse:
        """Answer ``question`` (or return a structured abstention)."""
        result = self._answerer.answer(question)
        return answer_result_to_response(result)

    def classify(self, description: str) -> ClassifyResponse:
        """Classify ``description`` per IEC 62304 §4.3 (or abstain)."""
        assessment = self._classifier.classify(description)
        return risk_assessment_to_response(assessment)

    def health(self) -> HealthResponse:
        """Report liveness and the indexed chunk count."""
        return HealthResponse(
            status="ok", indexed_chunks=self._repository.indexed_count()
        )


__all__ = ["RegulumenService"]
