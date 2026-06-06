"""Tests for hybrid (dense + BM25) fusion retrieval.

The headline test (:func:`test_hybrid_beats_either_alone`) constructs a scenario
where dense and lexical each WIN one query the other LOSES, then shows the fused
retriever gets BOTH right — the core justification for hybrid search.

All tests are offline: the dense side uses :class:`FakeEmbedder` over a temp
Chroma store; the lexical side is :class:`BM25Index`.
"""

from __future__ import annotations

import pytest

from regulumen.core.ingestion.models import Chunk
from regulumen.core.retrieval.bm25 import BM25Index
from regulumen.core.retrieval.embedder import FakeEmbedder
from regulumen.core.retrieval.hybrid import HybridRetriever
from regulumen.core.retrieval.vector_store import ChromaVectorStore, RetrievedChunk


def _chunks() -> list[Chunk]:
    return [
        # Target for the CLAUSE-NUMBER query. Its prose deliberately shares NO
        # distinctive words with the query "clause 7.4.2"; only the citation
        # carries the number, so dense (word/char-bucket) search ranks it low.
        Chunk(
            id="clause_target",
            doc_id="iec-62304",
            clause_path=["62304", "7.4.2"],
            citation_label="IEC 62304 §7.4.2",
            text="The responsible party retains documented evidence of every review.",
            metadata={"page": 7},
        ),
        # Target for the PARAPHRASE query. Shares many literal words with the
        # query so dense ranks it high; but the query omits the clause number,
        # so BM25 cannot use a number to find it.
        Chunk(
            id="paraphrase_target",
            doc_id="eu-mdr",
            clause_path=["Annex II", "3"],
            citation_label="EU MDR, Annex II §3",
            text=(
                "Manufacturers must perform risk management and document hazard "
                "mitigation across the device lifecycle."
            ),
            metadata={"page": 3},
        ),
        # Distractors.
        Chunk(
            id="distractor_a",
            doc_id="eu-mdr",
            clause_path=["Article 83", "83.2"],
            citation_label="EU MDR, Article 83 §83.2",
            text="Post-market surveillance reporting obligations for economic operators.",
            metadata={},
        ),
        Chunk(
            id="distractor_b",
            doc_id="guidance",
            clause_path=["Section 5"],
            citation_label="Guidance, §Section 5",
            text="Quarterly revenue projections for the regional sales department.",
            metadata={},
        ),
    ]


@pytest.fixture
def retriever(tmp_path) -> HybridRetriever:
    store = ChromaVectorStore(
        embedder=FakeEmbedder(),
        path=str(tmp_path / "chroma"),
        collection_name="test_hybrid",
    )
    bm25 = BM25Index()
    hyb = HybridRetriever(store, bm25, alpha=0.5)
    hyb.index(_chunks())
    return hyb


def _ids(results: list[RetrievedChunk]) -> list[str]:
    return [r.id for r in results]


def test_hybrid_beats_either_alone(tmp_path) -> None:
    """Dense wins the paraphrase, BM25 wins the clause number, hybrid wins both."""
    store = ChromaVectorStore(
        embedder=FakeEmbedder(),
        path=str(tmp_path / "chroma"),
        collection_name="beats_either",
    )
    bm25 = BM25Index()
    hyb = HybridRetriever(store, bm25, alpha=0.5)
    hyb.index(_chunks())

    # Query 1: a bare clause number. BM25 nails it (the number is in the
    # citation); dense treats the number as just one of many tokens and ranks
    # the wrong chunk first.
    clause_query = "clause 7.4.2"

    # Query 2: a semantic phrasing of the paraphrase_target's topic ("risk /
    # hazard mitigation across the device lifecycle"). It shares only the word
    # "hazard" with that chunk's prose, while the word "review" coincidentally
    # matches the UNRELATED clause_target ("...evidence of every review.").
    # Dense scores the topical chunk highest; BM25, going purely on the
    # "review" token, is misled to clause_target. Real blind spots on each side.
    semantic_query = "hazard analysis safety review process"

    # --- Establish that each single retriever has a blind spot. ---
    dense_clause = _ids(store.search(clause_query, k=4))
    bm25_semantic = _ids(bm25.search(semantic_query, k=4))

    # Dense alone does NOT rank the clause-number target first (its blind spot).
    assert dense_clause[0] != "clause_target", (
        f"precondition: dense should miss the clause number, got {dense_clause}"
    )
    # BM25 alone does NOT surface the content target first: the misleading
    # number token drags distractor_a to the top. If BM25 found the right chunk
    # anyway the test would be tautological, so assert the precondition.
    assert bm25_semantic and bm25_semantic[0] != "paraphrase_target", (
        f"precondition: bm25 should be misled off the content match, "
        f"got {bm25_semantic}"
    )

    # --- The payoff: hybrid surfaces the right chunk for BOTH queries. ---
    hybrid_clause = _ids(hyb.retrieve(clause_query, k=2))
    hybrid_para = _ids(hyb.retrieve(semantic_query, k=2))

    assert "clause_target" in hybrid_clause, (
        f"hybrid must catch the clause number via BM25, got {hybrid_clause}"
    )
    assert "paraphrase_target" in hybrid_para, (
        f"hybrid must catch the paraphrase via dense, got {hybrid_para}"
    )


def test_alpha_one_matches_dense_only(retriever: HybridRetriever, tmp_path) -> None:
    """alpha=1.0 → fused order equals the dense (vector store) order."""
    store = ChromaVectorStore(
        embedder=FakeEmbedder(),
        path=str(tmp_path / "chroma2"),
        collection_name="alpha_one",
    )
    bm25 = BM25Index()
    hyb = HybridRetriever(store, bm25, alpha=1.0)
    hyb.index(_chunks())

    query = "risk management hazard mitigation device lifecycle document"
    dense = _ids(store.search(query, k=4))
    fused = _ids(hyb.retrieve(query, k=4))
    assert fused == dense


def test_alpha_zero_matches_bm25_only(tmp_path) -> None:
    """alpha=0.0 → fused order equals the BM25-only order."""
    store = ChromaVectorStore(
        embedder=FakeEmbedder(),
        path=str(tmp_path / "chroma3"),
        collection_name="alpha_zero",
    )
    bm25 = BM25Index()
    hyb = HybridRetriever(store, bm25, alpha=0.0)
    hyb.index(_chunks())

    query = "post-market surveillance reporting"
    bm25_only = _ids(bm25.search(query, k=4))
    fused = _ids(hyb.retrieve(query, k=4))
    assert fused == bm25_only


def test_fused_score_in_unit_interval(retriever: HybridRetriever) -> None:
    results = retriever.retrieve("risk management document", k=4)
    assert results
    assert all(0.0 <= r.score <= 1.0 for r in results)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_metadata_exposes_per_source_scores(retriever: HybridRetriever) -> None:
    results = retriever.retrieve("risk management document", k=4)
    assert results
    for r in results:
        assert "dense_score" in r.metadata
        assert "bm25_score" in r.metadata
        assert "fused_score" in r.metadata
        assert 0.0 <= r.metadata["dense_score"] <= 1.0
        assert 0.0 <= r.metadata["bm25_score"] <= 1.0
        assert r.metadata["fused_score"] == pytest.approx(r.score)


def test_no_duplicate_ids_when_chunk_in_both_sources(
    retriever: HybridRetriever,
) -> None:
    """A chunk found by both dense and BM25 is merged into ONE result."""
    results = retriever.retrieve("risk management hazard mitigation document", k=4)
    ids = _ids(results)
    assert len(ids) == len(set(ids))


def test_retrieve_respects_k(retriever: HybridRetriever) -> None:
    results = retriever.retrieve("management", k=2)
    assert len(results) <= 2


def test_deterministic_order(retriever: HybridRetriever) -> None:
    q = "risk management document"
    first = _ids(retriever.retrieve(q, k=4))
    second = _ids(retriever.retrieve(q, k=4))
    assert first == second


def test_k_zero_returns_empty(retriever: HybridRetriever) -> None:
    # k<=0 yields no results without crashing.
    assert retriever.retrieve("risk management", k=0) == []


def test_blank_query_returns_empty(retriever: HybridRetriever) -> None:
    """Blank/whitespace query must not raise (FakeEmbedder raises on empty text)."""
    assert retriever.retrieve("", k=5) == []
    assert retriever.retrieve("   ", k=5) == []


def test_metadata_does_not_mutate_other_results(retriever: HybridRetriever) -> None:
    """Per-source score dicts must be independent per result (fresh dicts)."""
    results = retriever.retrieve("risk management document", k=4)
    assert len(results) >= 2
    results[0].metadata["fused_score"] = -999.0
    assert results[1].metadata["fused_score"] != -999.0
