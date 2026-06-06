"""Tests for the ChromaDB vector store using the deterministic FakeEmbedder.

No network and no HF token are required.
"""

from __future__ import annotations

import pytest

from regulumen.core.ingestion.models import Chunk
from regulumen.core.retrieval.embedder import FakeEmbedder
from regulumen.core.retrieval.vector_store import ChromaVectorStore, RetrievedChunk


def _chunks() -> list[Chunk]:
    return [
        Chunk(
            id="c1",
            doc_id="eu-mdr",
            clause_path=["Annex I", "17", "17.1"],
            citation_label="EU MDR, Annex I §17.1",
            text="Software safety classification assigns each item to class A, B or C.",
            metadata={"page": 12, "structure_confidence": 0.95},
        ),
        Chunk(
            id="c2",
            doc_id="eu-mdr",
            clause_path=["Article", "10"],
            citation_label="EU MDR, Article §10",
            text="Risk management file requirements and the maintenance of records.",
            metadata={"page": 4},
        ),
        Chunk(
            id="c3",
            doc_id="eu-mdr",
            clause_path=["Article", "83"],
            citation_label="EU MDR, Article §83",
            text="Post-market surveillance plan and periodic safety update reporting.",
            metadata={},
        ),
        Chunk(
            id="c4",
            doc_id="guidance",
            clause_path=["Section", "2"],
            citation_label="Guidance, Section §2",
            text="Quarterly financial revenue projections for the marketing department.",
            metadata={"nested": {"not": "scalar"}, "tags": ["a", "b"]},
        ),
    ]


@pytest.fixture
def store(tmp_path) -> ChromaVectorStore:
    return ChromaVectorStore(
        embedder=FakeEmbedder(),
        path=str(tmp_path / "chroma"),
        collection_name="test_regulumen",
    )


def test_index_and_count(store: ChromaVectorStore) -> None:
    chunks = _chunks()
    store.index(chunks)
    assert store.count() == len(chunks)


def test_search_ranks_matching_chunk_first(store: ChromaVectorStore) -> None:
    store.index(_chunks())
    results = store.search("software safety classification class", k=3)
    assert results, "expected at least one result"
    assert isinstance(results[0], RetrievedChunk)
    assert results[0].id == "c1"


def test_search_result_fields_populated(store: ChromaVectorStore) -> None:
    store.index(_chunks())
    [top] = store.search("software safety classification class", k=1)
    assert top.id == "c1"
    assert top.text
    assert top.citation_label == "EU MDR, Annex I §17.1"
    assert top.doc_id == "eu-mdr"
    assert top.clause_path == ["Annex I", "17", "17.1"]
    assert 0.0 <= top.score <= 1.0
    assert isinstance(top.metadata, dict)


def test_search_respects_k(store: ChromaVectorStore) -> None:
    store.index(_chunks())
    results = store.search("risk management", k=2)
    assert len(results) == 2


def test_scores_in_unit_interval(store: ChromaVectorStore) -> None:
    store.index(_chunks())
    results = store.search("safety", k=4)
    assert all(0.0 <= r.score <= 1.0 for r in results)
    # Scores should be sorted best-first (descending).
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_reindex_same_ids_upserts(store: ChromaVectorStore) -> None:
    chunks = _chunks()
    store.index(chunks)
    first = store.count()
    store.index(chunks)  # same ids again
    assert store.count() == first


def test_nonscalar_metadata_is_handled(store: ChromaVectorStore) -> None:
    # c4 carries nested dict + list metadata; indexing must not raise.
    store.index(_chunks())
    [res] = store.search(
        "quarterly financial revenue projections marketing", k=1
    )
    assert res.id == "c4"


def test_clause_path_roundtrip_with_separator_in_element(
    store: ChromaVectorStore,
) -> None:
    """A clause_path element containing ' > ' must round-trip exactly."""
    tricky_path = ["Part A > General", "1.1"]
    chunk = Chunk(
        id="tricky",
        doc_id="test-doc",
        clause_path=tricky_path,
        citation_label="Test > Doc §1.1",
        text="This clause element contains the separator character in its label.",
        metadata={},
    )
    store.index([chunk])
    results = store.search(
        "clause element contains separator character label", k=1
    )
    assert results, "expected at least one result"
    assert results[0].id == "tricky"
    assert results[0].clause_path == tricky_path
