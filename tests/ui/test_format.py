"""Offline unit tests for the pure UI rendering helpers.

No Gradio import, no network, synthetic IP-safe content only.
"""

from __future__ import annotations

from regulumen.api.schemas import AskResponse, CitationOut, ClassifyResponse
from regulumen.ui.format import (
    format_answer_markdown,
    format_citations_markdown,
    format_risk_markdown,
)

_DISCLAIMER = "Decision support, not regulatory advice."


def _citation(label: str = "[DOC-1 §1.2]") -> CitationOut:
    return CitationOut(
        citation_label=label,
        doc_id="DOC-1",
        clause_path=["1", "2"],
        score=0.873,
        snippet="Synthetic clause text about widget configuration.",
    )


# --------------------------------------------------------------------------- #
# format_answer_markdown
# --------------------------------------------------------------------------- #
def test_answer_markdown_answered_has_text_and_disclaimer() -> None:
    resp = AskResponse(
        answered=True,
        answer="The synthetic widget must be calibrated weekly.",
        abstained=False,
        abstention_reason=None,
        citations=[_citation()],
        disclaimer=_DISCLAIMER,
    )
    out = format_answer_markdown(resp)
    assert "The synthetic widget must be calibrated weekly." in out
    assert _DISCLAIMER in out
    assert "Disclaimer" in out


def test_answer_markdown_abstention_is_distinct() -> None:
    resp = AskResponse(
        answered=False,
        answer="",
        abstained=True,
        abstention_reason="No relevant clause was retrieved.",
        citations=[],
        disclaimer=_DISCLAIMER,
    )
    out = format_answer_markdown(resp)
    # Distinct abstention marker + lowercase "abstain" present.
    assert "⚠️" in out
    assert "abstain" in out.lower()
    # Reason surfaced.
    assert "No relevant clause was retrieved." in out
    # Does NOT present as a confident answer.
    assert "## Answer" not in out
    # Disclaimer always present.
    assert _DISCLAIMER in out


# --------------------------------------------------------------------------- #
# format_citations_markdown
# --------------------------------------------------------------------------- #
def test_citations_markdown_renders_each_label_and_snippet() -> None:
    cites = [_citation("[DOC-1 §1.2]"), _citation("[DOC-2 §4.3]")]
    out = format_citations_markdown(cites)
    for label in ("[DOC-1 §1.2]", "[DOC-2 §4.3]"):
        assert label in out
        # Bold-marked.
        assert f"**{label}**" in out
    assert "Synthetic clause text about widget configuration." in out


def test_citations_markdown_empty() -> None:
    assert format_citations_markdown([]) == "_No citations._"


# --------------------------------------------------------------------------- #
# format_risk_markdown
# --------------------------------------------------------------------------- #
def test_risk_markdown_classified_shows_class_prominently() -> None:
    resp = ClassifyResponse(
        safety_class="C",
        abstained=False,
        severity="death",
        can_contribute=True,
        rationale="Failure could directly cause serious injury.",
        citations=[_citation()],
        disclaimer=_DISCLAIMER,
    )
    out = format_risk_markdown(resp)
    # Class shown prominently (top-level heading).
    assert "# Safety Class C" in out
    assert "Failure could directly cause serious injury." in out
    assert _DISCLAIMER in out


def test_risk_markdown_abstention_is_distinct() -> None:
    resp = ClassifyResponse(
        safety_class=None,
        abstained=True,
        severity=None,
        can_contribute=None,
        rationale="The description lacked enough detail to classify.",
        citations=[],
        disclaimer=_DISCLAIMER,
    )
    out = format_risk_markdown(resp)
    assert "⚠️" in out
    assert "abstain" in out.lower()
    assert "The description lacked enough detail to classify." in out
    assert "# Safety Class" not in out
    assert _DISCLAIMER in out


# --------------------------------------------------------------------------- #
# Disclaimer present in ALL render paths
# --------------------------------------------------------------------------- #
def test_disclaimer_present_in_all_answer_and_risk_paths() -> None:
    answered = AskResponse(
        answered=True,
        answer="x",
        abstained=False,
        abstention_reason=None,
        citations=[],
        disclaimer=_DISCLAIMER,
    )
    abstained_ask = AskResponse(
        answered=False,
        answer="",
        abstained=True,
        abstention_reason="r",
        citations=[],
        disclaimer=_DISCLAIMER,
    )
    classified = ClassifyResponse(
        safety_class="A",
        abstained=False,
        severity="none",
        can_contribute=False,
        rationale="r",
        citations=[],
        disclaimer=_DISCLAIMER,
    )
    abstained_risk = ClassifyResponse(
        safety_class=None,
        abstained=True,
        severity=None,
        can_contribute=None,
        rationale="r",
        citations=[],
        disclaimer=_DISCLAIMER,
    )
    for out in (
        format_answer_markdown(answered),
        format_answer_markdown(abstained_ask),
        format_risk_markdown(classified),
        format_risk_markdown(abstained_risk),
    ):
        assert _DISCLAIMER in out
