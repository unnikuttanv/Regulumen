"""Regression tests for RetrievalRepository wiring.

The headline bug these guard against: ``build_from_settings`` constructed a
``BM25Index`` but never indexed it, so the live API/UI silently ran dense-only
retrieval (the hybrid feature was dead). The eval harness was unaffected because
its runner indexes BM25 itself — so only a repository-level test catches this.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from regulumen.api.repository import RetrievalRepository
from regulumen.core.ingestion.corpus import build_chunks
from regulumen.core.retrieval.bm25 import BM25Index
from regulumen.core.retrieval.embedder import FakeEmbedder
from regulumen.core.retrieval.hybrid import HybridRetriever
from regulumen.core.retrieval.vector_store import ChromaVectorStore

_CORPUS_MANIFEST = Path("corpus/manifest.json")
_CORPUS_RAW = Path("corpus/raw")
_HAS_CORPUS = _CORPUS_MANIFEST.exists() and _CORPUS_RAW.exists()


@pytest.mark.skipif(not _HAS_CORPUS, reason="corpus not present")
def test_repository_indexes_bm25_not_dense_only(tmp_path):
    """A repository built over the corpus must have a NON-empty BM25 index.

    Mirrors build_from_settings' wiring but injects a FakeEmbedder so the test
    needs no network/keys, then asserts the lexical index was actually populated
    (the bug left it empty).
    """
    from regulumen.core.config import get_settings  # noqa: PLC0415

    settings = get_settings()
    chunks = build_chunks(settings, _CORPUS_MANIFEST, _CORPUS_RAW)
    assert chunks, "corpus should produce chunks"

    embedder = FakeEmbedder(dim=64)
    vector_store = ChromaVectorStore(embedder, path=str(tmp_path / "chroma"))
    bm25 = BM25Index()
    bm25.index(chunks)
    retriever = HybridRetriever(vector_store, bm25, alpha=0.5)
    repo = RetrievalRepository(retriever)

    held = repo.get_retriever()
    # The BM25 index inside the held retriever must be non-empty — a dense-only
    # regression would leave count() == 0.
    assert held._bm25.count() > 0


def test_repository_indexed_count_is_resilient_to_unreachable_store():
    """indexed_count() never raises, even with a retriever lacking a store."""

    class _Stub:
        pass

    repo = RetrievalRepository(_Stub())  # type: ignore[arg-type]
    assert repo.indexed_count() is None
