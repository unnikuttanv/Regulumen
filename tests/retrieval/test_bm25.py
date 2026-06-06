"""Tests for the BM25 lexical retrieval index.

These run fully offline (no embedder, no network). The headline test is the
*exact clause-number match*: a query for a bare clause number ("5.1.1",
"Article 10") must surface the chunk that cites it, which pure dense search
typically misses.
"""

from __future__ import annotations

import pytest

from regulumen.core.ingestion.models import Chunk
from regulumen.core.retrieval.bm25 import BM25Index, tokenize
from regulumen.core.retrieval.vector_store import RetrievedChunk


def _chunks() -> list[Chunk]:
    """IP-safe synthetic chunks. Clause numbers are intentionally distinctive."""
    return [
        Chunk(
            id="c1",
            doc_id="iec-62304",
            clause_path=["62304", "5.1.1"],
            citation_label="IEC 62304 §5.1.1",
            text="The software development plan shall describe the activities.",
            metadata={"page": 12},
        ),
        Chunk(
            id="c2",
            doc_id="eu-mdr",
            clause_path=["Article 10", "10.1"],
            citation_label="EU MDR, Article 10 §10.1",
            text="The manufacturer shall establish a quality management system.",
            metadata={"page": 4},
        ),
        Chunk(
            id="c3",
            doc_id="eu-mdr",
            clause_path=["Article 83", "83.2"],
            citation_label="EU MDR, Article 83 §83.2",
            text="Post-market surveillance plan and periodic update reporting.",
            metadata={},
        ),
        Chunk(
            id="c4",
            doc_id="guidance",
            clause_path=["Section 2"],
            citation_label="Guidance, §Section 2",
            text="Quarterly revenue projections for the marketing department.",
            metadata={},
        ),
    ]


@pytest.fixture
def index() -> BM25Index:
    idx = BM25Index()
    idx.index(_chunks())
    return idx


def test_tokenize_preserves_clause_numbers() -> None:
    toks = tokenize("IEC 62304 §5.1.1")
    assert "62304" in toks
    assert "5.1.1" in toks
    # The section sign itself must not become a stray token / break parsing.
    assert "§" not in toks


def test_tokenize_handles_article_form() -> None:
    toks = tokenize("Article 10(1)")
    assert "article" in toks
    assert "10" in toks


def test_tokenize_lowercases() -> None:
    assert tokenize("Software DEVELOPMENT Plan") == ["software", "development", "plan"]


def test_count(index: BM25Index) -> None:
    assert index.count() == 4


def test_exact_clause_number_match_dotted(index: BM25Index) -> None:
    """The money test: a bare dotted clause number finds the chunk citing it."""
    results = index.search("5.1.1", k=3)
    assert results, "expected at least one lexical hit for the clause number"
    assert isinstance(results[0], RetrievedChunk)
    assert results[0].id == "c1", (
        "BM25 must surface the §5.1.1 chunk first for query '5.1.1'"
    )


def test_exact_clause_number_match_article(index: BM25Index) -> None:
    """'Article 10' lexically lands on the Article 10 chunk, not Article 83."""
    results = index.search("Article 10", k=3)
    assert results
    assert results[0].id == "c2"


def test_scores_normalized_to_unit_interval(index: BM25Index) -> None:
    results = index.search("software development plan", k=4)
    assert results
    assert all(0.0 <= r.score <= 1.0 for r in results)
    # Top hit normalized to 1.0; results sorted best-first.
    assert results[0].score == pytest.approx(1.0)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_no_match_query_does_not_crash(index: BM25Index) -> None:
    """A query with no lexical overlap returns empty (no zero-division crash)."""
    results = index.search("zzzzzz nonexistent gibberish token", k=3)
    assert results == []


def test_search_result_fields_populated(index: BM25Index) -> None:
    [top] = index.search("software development plan", k=1)
    assert top.id == "c1"
    assert top.text
    assert top.citation_label == "IEC 62304 §5.1.1"
    assert top.doc_id == "iec-62304"
    assert top.clause_path == ["62304", "5.1.1"]
    assert isinstance(top.metadata, dict)


def test_search_respects_k(index: BM25Index) -> None:
    results = index.search("plan", k=2)
    assert len(results) <= 2


def test_empty_index_returns_empty() -> None:
    idx = BM25Index()
    assert idx.count() == 0
    assert idx.search("anything", k=5) == []


def test_blank_query_returns_empty(index: BM25Index) -> None:
    """Blank or whitespace-only queries return [] without raising."""
    assert index.search("   ", k=5) == []


def test_exact_clause_number_beats_prefix_only() -> None:
    """A query for '5.1.1' must rank the exact-match chunk above prefix-only chunks.

    Guards the prefix-emission tokenizer against future precision regressions:
    emitting prefixes of clause numbers should aid partial matching but must not
    crowd out the exact hit.
    """
    exact_chunk = Chunk(
        id="exact",
        doc_id="test-doc",
        clause_path=["5.1.1"],
        citation_label="Test §5.1.1",
        text="Detailed requirements for sub-clause five one one.",
        metadata={},
    )
    prefix_chunk = Chunk(
        id="prefix_only",
        doc_id="test-doc",
        clause_path=["5"],
        citation_label="Test §5",
        text="General requirements for section five.",
        metadata={},
    )
    unrelated_chunk = Chunk(
        id="unrelated",
        doc_id="test-doc",
        clause_path=["5.2.3"],
        citation_label="Test §5.2.3",
        text="Requirements for sub-clause five two three.",
        metadata={},
    )
    idx = BM25Index()
    idx.index([exact_chunk, prefix_chunk, unrelated_chunk])

    results = idx.search("5.1.1", k=3)
    assert results, "expected at least one hit for exact clause number '5.1.1'"
    assert results[0].id == "exact", (
        f"exact clause-number chunk must rank first, got {[r.id for r in results]}"
    )
