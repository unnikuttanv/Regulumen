"""Data/index access layer.

This is the ONLY layer that knows how to construct the concrete retrieval stack
(embedder + Chroma vector store + BM25 + HybridRetriever). The service and
router never import Chroma or the embedder directly — they go through here.

Indexing the corpus from disk is the ingest script's job (Step 11). This
repository merely holds/serves a retriever; an empty index is fine (retrieval
returns nothing and the answerer abstains).
"""

from __future__ import annotations

from regulumen.core.retrieval.hybrid import HybridRetriever


class RetrievalRepository:
    """Holds and serves the :class:`HybridRetriever`.

    Construct either with a prebuilt retriever (the testing/DI path, no real
    embeddings needed) or via :meth:`build_from_settings` (the production path,
    which constructs the embedder + Chroma vector store + BM25).
    """

    def __init__(self, retriever: HybridRetriever) -> None:
        self._retriever = retriever

    @classmethod
    def build_from_settings(cls, settings) -> RetrievalRepository:
        """Construct the full retrieval stack from ``settings``.

        Imports the embedder / Chroma store lazily so merely importing this
        module (and therefore the app) never pulls heavy deps or needs keys.

        The Chroma dense index is persisted on disk (populated by
        ``scripts/ingest.py``), but the BM25 lexical index is in-memory and must
        be rebuilt from the corpus chunks on each process start — otherwise the
        hybrid retriever silently degrades to dense-only. We therefore re-derive
        the chunks (parse + chunk is offline and fast, ~1-2s) and index BM25 here
        so the live API/UI gets true hybrid retrieval, matching the eval harness.
        """
        from regulumen.core.ingestion.corpus import build_chunks  # noqa: PLC0415
        from regulumen.core.retrieval.bm25 import BM25Index  # noqa: PLC0415
        from regulumen.core.retrieval.embedder import build_embedder  # noqa: PLC0415
        from regulumen.core.retrieval.vector_store import (  # noqa: PLC0415
            ChromaVectorStore,
        )

        embedder = build_embedder(settings)
        vector_store = ChromaVectorStore(embedder, path=settings.chroma_path)
        bm25 = BM25Index()
        # Rebuild the lexical index from the same corpus the dense index was
        # built from (the corpus module's defaults point at corpus/). Resilient:
        # if the corpus isn't present, BM25 stays empty and retrieval falls back
        # to dense-only rather than crashing.
        try:
            chunks = build_chunks(settings)
            if chunks:
                bm25.index(chunks)
        except Exception:  # pragma: no cover - defensive: never block startup
            pass
        retriever = HybridRetriever(vector_store, bm25, alpha=settings.hybrid_alpha)
        return cls(retriever)

    def get_retriever(self) -> HybridRetriever:
        """Return the held :class:`HybridRetriever`."""
        return self._retriever

    def indexed_count(self) -> int | None:
        """Return the indexed chunk count if discoverable, else ``None``.

        Best-effort: walks into the vector store's ``count()`` when present,
        and never raises (an unreachable store yields ``None``).
        """
        store = getattr(self._retriever, "_vector_store", None)
        count = getattr(store, "count", None)
        if not callable(count):
            return None
        try:
            return int(count())
        except Exception:  # pragma: no cover - defensive: never break /health
            return None


__all__ = ["RetrievalRepository"]
