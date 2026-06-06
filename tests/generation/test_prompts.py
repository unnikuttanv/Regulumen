"""Tests for prompt construction. Pure string assertions, no network."""

from __future__ import annotations

from regulumen.core.generation.prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
)
from regulumen.core.retrieval.vector_store import RetrievedChunk


def _chunk(label: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        id=label,
        text=text,
        citation_label=label,
        doc_id="doc",
        clause_path=[],
        score=0.9,
        metadata={},
    )


def test_system_prompt_mentions_grounding_abstention_and_disclaimer():
    lowered = SYSTEM_PROMPT.lower()
    assert "only" in lowered  # answer only from excerpts
    assert "abstain" in lowered or "cannot answer" in lowered
    assert "decision support" in lowered  # the mandatory disclaimer
    assert "[[" in SYSTEM_PROMPT  # the citation format is specified


def test_build_user_prompt_includes_question_and_each_chunk():
    chunks = [
        _chunk("EU MDR, Annex I §17.2", "Software shall be developed per state of the art."),
        _chunk("IMDRF SaMD N10, §5.1", "Risk categorization framework."),
    ]
    prompt = build_user_prompt("What does the software requirement say?", chunks)

    assert "What does the software requirement say?" in prompt
    for c in chunks:
        assert c.citation_label in prompt
        assert c.text in prompt


def test_build_user_prompt_explains_citation_format():
    prompt = build_user_prompt("q", [_chunk("EU MDR §1", "text")])
    assert "[[" in prompt and "]]" in prompt


def test_build_user_prompt_handles_empty_chunks():
    # Defensive: the orchestrator early-abstains on empty retrieval, but the
    # builder should not crash if ever called with no chunks.
    prompt = build_user_prompt("q", [])
    assert "q" in prompt
