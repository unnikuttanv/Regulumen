"""Hybrid dense + BM25 retrieval with score fusion for ReguLumen.

This is the headline retrieval design: dense (vector) search captures *meaning*
(paraphrases, synonyms) while BM25 captures *exact lexical hits* (clause numbers
like "§5.1.1", "Article 10"). Neither alone is sufficient for citation-faithful
regulatory QA, so we run both and FUSE their normalized scores.

Fusion is a convex combination per chunk id::

    fused = alpha * dense_score + (1 - alpha) * bm25_score

where ``alpha`` is the dense weight (``alpha=1`` → dense only, ``alpha=0`` → BM25
only), sourced from :attr:`Settings.hybrid_alpha`. Both inputs are already in
``[0, 1]`` (the vector store normalizes cosine distance; :class:`BM25Index`
normalizes by the per-query max), so the fused score is also in ``[0, 1]``. A
chunk missing from one side contributes ``0`` for that side.

Per-source scores are surfaced in each result's ``metadata``
(``dense_score`` / ``bm25_score`` / ``fused_score``) for evaluation and
debugging, using *fresh* dicts so the underlying chunk metadata is never
mutated.
"""

from __future__ import annotations

from dataclasses import dataclass

from regulumen.core.ingestion.models import Chunk
from regulumen.core.retrieval.bm25 import BM25Index
from regulumen.core.retrieval.vector_store import ChromaVectorStore, RetrievedChunk

# How many candidates to pull from EACH retriever before fusing. Fetching a
# wider pool than k lets a chunk that is, say, rank-6 dense but rank-1 BM25 still
# enter fusion and win — which is the whole point of hybrid search.
_DEFAULT_FETCH_MULTIPLIER = 4
_MIN_FETCH_K = 20


@dataclass
class _Candidate:
    """Accumulates per-source scores for one chunk id during fusion."""

    hit: RetrievedChunk
    dense_score: float = 0.0
    bm25_score: float = 0.0


class HybridRetriever:
    """Fuses dense (:class:`ChromaVectorStore`) and lexical (:class:`BM25Index`)."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        bm25: BM25Index,
        *,
        alpha: float = 0.5,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self._vector_store = vector_store
        self._bm25 = bm25
        self._alpha = alpha

    def index(self, chunks: list[Chunk]) -> None:
        """Index ``chunks`` into BOTH the vector store and the BM25 index.

        Provided so callers populate both sides from one chunk list and can
        never drift out of sync.
        """
        self._vector_store.index(chunks)
        self._bm25.index(chunks)

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return up to ``k`` fused results, best-first, with score in ``[0, 1]``.

        Pulls a wider candidate pool from each retriever, fuses by id, and
        returns the top-``k`` by fused score (tie-break by id for determinism).
        """
        # Blank/whitespace queries tokenize to nothing; the dense embedder raises
        # ValueError on empty text.  Short-circuit before touching either side.
        if not query.strip():
            return []
        if k <= 0:
            return []

        fetch_k = max(k * _DEFAULT_FETCH_MULTIPLIER, _MIN_FETCH_K)
        dense_hits = self._vector_store.search(query, fetch_k)
        bm25_hits = self._bm25.search(query, fetch_k)

        candidates: dict[str, _Candidate] = {}
        for hit in dense_hits:
            candidates.setdefault(hit.id, _Candidate(hit=hit)).dense_score = hit.score
        for hit in bm25_hits:
            cand = candidates.setdefault(hit.id, _Candidate(hit=hit))
            cand.bm25_score = hit.score
            # Prefer dense hit fields if already present (they match — same
            # chunk); otherwise the bm25 hit's fields populate the candidate.

        fused: list[RetrievedChunk] = []
        for cand in candidates.values():
            fused_score = (
                self._alpha * cand.dense_score
                + (1.0 - self._alpha) * cand.bm25_score
            )
            # A zero fused score means neither weighted source found this chunk
            # relevant (e.g. a dense-only hit when alpha=0). Drop it so the
            # fused ranking degenerates cleanly to the active source(s) and
            # never pads results with irrelevant chunks.
            if fused_score <= 0.0:
                continue
            base = cand.hit
            # Fresh metadata dict: never mutate the source chunk's metadata.
            meta = dict(base.metadata)
            meta["dense_score"] = cand.dense_score
            meta["bm25_score"] = cand.bm25_score
            meta["fused_score"] = fused_score
            fused.append(
                RetrievedChunk(
                    id=base.id,
                    text=base.text,
                    citation_label=base.citation_label,
                    doc_id=base.doc_id,
                    clause_path=list(base.clause_path),
                    score=fused_score,
                    metadata=meta,
                )
            )

        # Sort by fused score descending; tie-break by id ascending (stable).
        fused.sort(key=lambda r: (-r.score, r.id))
        return fused[:k]
