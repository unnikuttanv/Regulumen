"""Tests for the post-generation faithfulness GATE — the project centerpiece.

Every test is pure (no I/O, no network): synthetic ``RetrievedChunk`` objects
plus hand-written answer strings. The gate must downgrade an answer that cites a
clause NOT among the retrieved chunks to an abstention.
"""

from __future__ import annotations

from regulumen.core.generation.faithfulness import (
    ABSTENTION_TOKEN,
    FaithfulnessResult,
    _is_specific_citation,
    check_faithfulness,
    extract_citations,
    grounding_score,
    is_abstention,
)
from regulumen.core.retrieval.vector_store import RetrievedChunk


def _chunk(label: str, *, id_: str = "c", text: str = "some clause text") -> RetrievedChunk:
    return RetrievedChunk(
        id=id_,
        text=text,
        citation_label=label,
        doc_id="doc",
        clause_path=[],
        score=0.9,
        metadata={},
    )


# --- extract_citations -----------------------------------------------------


def test_extract_citations_pulls_double_bracket_spans():
    answer = "Devices must comply [[EU MDR, Annex I §17.2]] and also [[IMDRF SaMD N10, §5.1]]."
    assert extract_citations(answer) == ["EU MDR, Annex I §17.2", "IMDRF SaMD N10, §5.1"]


def test_extract_citations_none_returns_empty_list():
    assert extract_citations("This answer cites nothing at all.") == []


def test_extract_citations_strips_whitespace_and_dedupes_preserving_order():
    answer = "[[  EU MDR §1  ]] then again [[EU MDR §1]] and [[EU MDR §2]]"
    assert extract_citations(answer) == ["EU MDR §1", "EU MDR §2"]


# --- is_abstention ---------------------------------------------------------


def test_is_abstention_detects_token():
    assert is_abstention(f"{ABSTENTION_TOKEN} The excerpts do not cover sterilization.")


def test_is_abstention_false_for_normal_answer():
    assert not is_abstention("Devices must comply [[EU MDR §1]] with the rule.")


def test_is_abstention_is_case_insensitive_and_token_isolated():
    # Case-insensitive and surrounded by punctuation/spaces -> still a match.
    assert is_abstention("Sorry, __abstain__ — nothing relevant was provided.")


def test_is_abstention_token_at_string_boundaries():
    assert is_abstention(ABSTENTION_TOKEN)  # whole string
    assert is_abstention(f"{ABSTENTION_TOKEN} missing info")  # start
    assert is_abstention(f"see token here:{ABSTENTION_TOKEN}")  # after punctuation


def test_is_abstention_embedded_in_word_is_false():
    # Token glued inside a larger word must NOT trigger abstention.
    assert not is_abstention("x__ABSTAIN__y")
    assert not is_abstention("prefix__ABSTAIN__suffix")


# --- grounding_score -------------------------------------------------------


def test_grounding_score_all_citations_retrieved_is_one():
    retrieved = [_chunk("EU MDR, Annex I §17.2"), _chunk("IMDRF SaMD N10, §5.1")]
    answer = "Comply [[EU MDR, Annex I §17.2]] and [[IMDRF SaMD N10, §5.1]]."
    assert grounding_score(answer, retrieved) == 1.0


def test_grounding_score_hallucinated_citation_lowers_score():
    retrieved = [_chunk("EU MDR, Annex I §17.2")]
    answer = "Comply [[EU MDR, Annex I §17.2]] and [[EU MDR, Annex I §99.9]]."
    score = grounding_score(answer, retrieved)
    assert 0.0 < score < 1.0  # one of two citations is unsupported


def test_grounding_score_no_citations_is_zero():
    retrieved = [_chunk("EU MDR, Annex I §17.2")]
    assert grounding_score("An uncited claim.", retrieved) == 0.0


def test_grounding_score_vague_citation_is_zero():
    # A vague catch-all does not count as supported.
    retrieved = [_chunk("EU MDR, Annex I §17.2")]
    assert grounding_score("Comply [[EU MDR]].", retrieved) == 0.0


def test_grounding_score_accepts_precomputed_used():
    retrieved = [_chunk("EU MDR, Annex I §17.2")]
    answer = "Comply [[EU MDR, Annex I §17.2]]."
    used = extract_citations(answer)
    assert grounding_score(answer, retrieved, used=used) == 1.0


# --- _is_specific_citation -------------------------------------------------


def test_is_specific_citation_requires_clause_signal():
    assert _is_specific_citation("EU MDR, Annex I §17.2")
    assert _is_specific_citation("EU MDR Article 10")
    assert _is_specific_citation("IMDRF SaMD N10, §5.1")
    # Bare regulation names with no number are NOT specific.
    assert not _is_specific_citation("EU MDR")
    assert not _is_specific_citation("MDR")


# --- check_faithfulness ----------------------------------------------------


def test_check_faithfulness_grounded_case():
    retrieved = [_chunk("EU MDR, Annex I §17.2")]
    answer = "Devices must meet the requirement [[EU MDR, Annex I §17.2]]."
    result = check_faithfulness(answer, retrieved)
    assert isinstance(result, FaithfulnessResult)
    assert result.grounded is True
    assert result.score == 1.0
    assert result.used_citations == ["EU MDR, Annex I §17.2"]
    assert result.unsupported_citations == []


def test_check_faithfulness_hallucinated_citation_not_grounded():
    """THE money assertion: a citation not in the retrieved set is unsupported."""
    retrieved = [_chunk("EU MDR, Annex I §17.2")]
    answer = "A made-up rule applies [[EU MDR, Annex I §99.9]]."
    result = check_faithfulness(answer, retrieved)
    assert result.grounded is False
    assert "EU MDR, Annex I §99.9" in result.unsupported_citations


def test_check_faithfulness_zero_citation_non_abstention_not_grounded():
    retrieved = [_chunk("EU MDR, Annex I §17.2")]
    answer = "Devices must comply with the requirement."  # no [[...]] at all
    result = check_faithfulness(answer, retrieved)
    assert result.grounded is False
    assert result.used_citations == []


def test_check_faithfulness_mixed_supported_and_unsupported_not_grounded():
    retrieved = [_chunk("EU MDR, Annex I §17.2")]
    answer = "Comply [[EU MDR, Annex I §17.2]] and also [[EU MDR, Annex I §99.9]]."
    result = check_faithfulness(answer, retrieved)
    # Even though one citation is real, the presence of an unsupported one
    # forces grounded=False (strict gate).
    assert result.grounded is False
    assert result.unsupported_citations == ["EU MDR, Annex I §99.9"]


def test_check_faithfulness_is_binary_strict():
    """The gate is strict-binary: ANY single unsupported citation => not grounded.

    Three supported citations plus one unsupported one must still fail,
    regardless of the (transparency-only) score, and a fully-supported answer
    must pass.
    """
    retrieved = [
        _chunk("EU MDR, Annex I §17.1"),
        _chunk("EU MDR, Annex I §17.2"),
        _chunk("EU MDR, Annex I §17.3"),
    ]
    # All three supported -> grounded.
    good = (
        "A [[EU MDR, Annex I §17.1]] B [[EU MDR, Annex I §17.2]] "
        "C [[EU MDR, Annex I §17.3]]."
    )
    good_result = check_faithfulness(good, retrieved)
    assert good_result.grounded is True
    assert good_result.unsupported_citations == []

    # 3 supported + 1 unsupported -> NOT grounded (no threshold can rescue it).
    bad = good[:-1] + " D [[EU MDR, Annex I §99.9]]."
    bad_result = check_faithfulness(bad, retrieved)
    assert bad_result.grounded is False
    assert bad_result.unsupported_citations == ["EU MDR, Annex I §99.9"]


def test_check_faithfulness_vague_label_not_grounded():
    """A bare regulation name (no clause number) is non-specific -> abstain."""
    retrieved = [_chunk("EU MDR, Annex I §17.2")]
    result = check_faithfulness("Devices must comply [[EU MDR]].", retrieved)
    assert result.grounded is False
    # The vague catch-all is flagged as unsupported even though it would
    # substring-match the retrieved EU MDR chunk.
    assert "EU MDR" in result.unsupported_citations


def test_check_faithfulness_specific_label_grounded():
    """A specific clause citation matching a retrieved chunk is grounded."""
    retrieved = [_chunk("EU MDR, Annex I §17.2")]
    result = check_faithfulness(
        "Devices must comply [[EU MDR, Annex I §17.2]].", retrieved
    )
    assert result.grounded is True
    assert result.unsupported_citations == []
