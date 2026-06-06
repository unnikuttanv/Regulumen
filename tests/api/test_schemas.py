"""Schema mapping tests: domain objects -> Pydantic response models.

These are pure mapping checks (no HTTP, no providers): they prove the
translation layer pulls the right fields out of AnswerResult / RiskAssessment /
RetrievedChunk.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from regulumen.api.schemas import (
    AskRequest,
    AskResponse,
    CitationOut,
    ClassifyRequest,
    ClassifyResponse,
    answer_result_to_response,
    chunk_to_citation,
    risk_assessment_to_response,
)
from regulumen.core.generation.answerer import DISCLAIMER as ANSWER_DISCLAIMER
from regulumen.core.generation.answerer import AnswerResult
from regulumen.core.retrieval.vector_store import RetrievedChunk
from regulumen.core.risk.classifier import DISCLAIMER as RISK_DISCLAIMER
from regulumen.core.risk.classifier import RiskAssessment
from regulumen.core.risk.severity_table import SafetyClass, Severity


def _chunk(**overrides) -> RetrievedChunk:
    base = dict(
        id="c1",
        text="The quick brown fox " * 50,
        citation_label="ISO 13485 §7.5.1",
        doc_id="iso13485",
        clause_path=["7", "7.5", "7.5.1"],
        score=0.87,
        metadata={"foo": "bar"},
    )
    base.update(overrides)
    return RetrievedChunk(**base)


def test_chunk_to_citation_maps_fields_and_truncates_snippet():
    chunk = _chunk()
    out = chunk_to_citation(chunk)
    assert isinstance(out, CitationOut)
    assert out.citation_label == "ISO 13485 §7.5.1"
    assert out.doc_id == "iso13485"
    assert out.clause_path == ["7", "7.5", "7.5.1"]
    assert out.score == pytest.approx(0.87)
    # snippet is a short excerpt, not the whole (long) text.
    assert out.snippet
    assert len(out.snippet) < len(chunk.text)


def test_answer_result_to_response_answered():
    chunk = _chunk()
    result = AnswerResult(
        answered=True,
        answer="Per [ISO 13485 §7.5.1], you must validate the process.",
        citations=[chunk],
        abstained=False,
        abstention_reason=None,
        faithfulness=None,
        disclaimer=ANSWER_DISCLAIMER,
    )
    resp = answer_result_to_response(result)
    assert isinstance(resp, AskResponse)
    assert resp.answered is True
    assert resp.abstained is False
    assert resp.abstention_reason is None
    assert resp.answer.startswith("Per [ISO 13485")
    assert resp.disclaimer == ANSWER_DISCLAIMER
    assert len(resp.citations) == 1
    assert resp.citations[0].citation_label == "ISO 13485 §7.5.1"


def test_answer_result_to_response_abstained():
    result = AnswerResult(
        answered=False,
        answer="No relevant clauses were retrieved for this question.",
        citations=[],
        abstained=True,
        abstention_reason="No relevant clauses were retrieved for this question.",
        faithfulness=None,
        disclaimer=ANSWER_DISCLAIMER,
    )
    resp = answer_result_to_response(result)
    assert resp.answered is False
    assert resp.abstained is True
    assert resp.abstention_reason
    assert resp.citations == []


def test_risk_assessment_to_response_classified():
    chunk = _chunk()
    assessment = RiskAssessment(
        safety_class=SafetyClass.C,
        abstained=False,
        severity=Severity.SERIOUS,
        can_contribute=True,
        rationale="Serious injury possible -> Class C.",
        citations=[chunk],
        disclaimer=RISK_DISCLAIMER,
        extracted={"worst_case_severity": "serious"},
    )
    resp = risk_assessment_to_response(assessment)
    assert isinstance(resp, ClassifyResponse)
    # safety_class -> .value
    assert resp.safety_class == "C"
    # severity -> name lowercased
    assert resp.severity == "serious"
    assert resp.can_contribute is True
    assert resp.abstained is False
    assert resp.rationale
    assert resp.disclaimer == RISK_DISCLAIMER
    assert len(resp.citations) == 1


def test_risk_assessment_to_response_abstained():
    assessment = RiskAssessment(
        safety_class=None,
        abstained=True,
        severity=None,
        can_contribute=None,
        rationale="Severity could not be established.",
        citations=[],
        disclaimer=RISK_DISCLAIMER,
        extracted={},
    )
    resp = risk_assessment_to_response(assessment)
    assert resp.safety_class is None
    assert resp.severity is None
    assert resp.can_contribute is None
    assert resp.abstained is True
    assert resp.rationale


def test_ask_request_rejects_empty():
    with pytest.raises(ValidationError):
        AskRequest(question="   ")


def test_classify_request_rejects_empty():
    with pytest.raises(ValidationError):
        ClassifyRequest(description="")
