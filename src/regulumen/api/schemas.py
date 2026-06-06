"""Pydantic v2 request/response models + domain->schema mappers.

These DTOs are the typed HTTP contract. Domain objects (AnswerResult,
RiskAssessment, RetrievedChunk) never cross the wire directly; the mapper
helpers here translate them, so the router/service stay thin and the API shape
is decoupled from the internal dataclasses.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from regulumen.core.generation.answerer import AnswerResult
from regulumen.core.retrieval.vector_store import RetrievedChunk
from regulumen.core.risk.classifier import RiskAssessment

# Max question/description length kept generous but bounded to reject abuse.
_MAX_INPUT_LEN = 5000
# Length of the short text excerpt surfaced per citation.
_SNIPPET_LEN = 300


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
class AskRequest(BaseModel):
    """A free-text regulatory question."""

    question: str = Field(min_length=1, max_length=_MAX_INPUT_LEN)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty or whitespace")
        return value


class ClassifyRequest(BaseModel):
    """A free-text software-component description to classify (IEC 62304)."""

    description: str = Field(min_length=1, max_length=_MAX_INPUT_LEN)

    @field_validator("description")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("description must not be empty or whitespace")
        return value


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #
class CitationOut(BaseModel):
    """A single cited source chunk, trimmed for the wire."""

    citation_label: str
    doc_id: str
    clause_path: list[str]
    score: float
    snippet: str


class AskResponse(BaseModel):
    """Answer or structured abstention for one question."""

    answered: bool
    answer: str
    abstained: bool
    abstention_reason: str | None
    citations: list[CitationOut]
    disclaimer: str


class ClassifyResponse(BaseModel):
    """IEC 62304 §4.3 classification or structured abstention."""

    safety_class: str | None
    abstained: bool
    severity: str | None
    can_contribute: bool | None
    rationale: str
    citations: list[CitationOut]
    disclaimer: str


class HealthResponse(BaseModel):
    """Liveness + how many chunks are indexed (``None`` if unknown)."""

    status: str
    indexed_chunks: int | None


# --------------------------------------------------------------------------- #
# Mappers (domain -> schema)
# --------------------------------------------------------------------------- #
def chunk_to_citation(chunk: RetrievedChunk) -> CitationOut:
    """Map a :class:`RetrievedChunk` to a :class:`CitationOut` (short snippet)."""
    text = chunk.text or ""
    snippet = text[:_SNIPPET_LEN].rstrip()
    if len(text) > _SNIPPET_LEN:
        snippet += "..."
    return CitationOut(
        citation_label=chunk.citation_label,
        doc_id=chunk.doc_id,
        clause_path=list(chunk.clause_path),
        score=float(chunk.score),
        snippet=snippet,
    )


def answer_result_to_response(result: AnswerResult) -> AskResponse:
    """Map an :class:`AnswerResult` to an :class:`AskResponse`."""
    return AskResponse(
        answered=result.answered,
        answer=result.answer,
        abstained=result.abstained,
        abstention_reason=result.abstention_reason,
        citations=[chunk_to_citation(c) for c in result.citations],
        disclaimer=result.disclaimer,
    )


def risk_assessment_to_response(assessment: RiskAssessment) -> ClassifyResponse:
    """Map a :class:`RiskAssessment` to a :class:`ClassifyResponse`.

    ``safety_class`` -> enum ``.value`` (or None); ``severity`` -> enum name
    lowercased (or None).
    """
    safety_class = (
        assessment.safety_class.value if assessment.safety_class is not None else None
    )
    severity = (
        assessment.severity.name.lower() if assessment.severity is not None else None
    )
    return ClassifyResponse(
        safety_class=safety_class,
        abstained=assessment.abstained,
        severity=severity,
        can_contribute=assessment.can_contribute,
        rationale=assessment.rationale,
        citations=[chunk_to_citation(c) for c in assessment.citations],
        disclaimer=assessment.disclaimer,
    )


__all__ = [
    "AskRequest",
    "AskResponse",
    "CitationOut",
    "ClassifyRequest",
    "ClassifyResponse",
    "HealthResponse",
    "answer_result_to_response",
    "chunk_to_citation",
    "risk_assessment_to_response",
]
