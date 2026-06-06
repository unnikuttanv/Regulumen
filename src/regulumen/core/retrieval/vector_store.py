"""ChromaDB-backed dense vector store for ReguLumen.

We supply our own embeddings (via the :class:`Embedder` protocol) rather than
letting Chroma compute them, so the collection is created WITHOUT an embedding
function and we pass ``embeddings=`` explicitly on upsert/query.

The :class:`RetrievedChunk` dataclass is the stable retrieval result type
reused by Step 4b (hybrid fusion) and Step 7 (generation). Its ``score`` is a
similarity in ``[0, 1]`` (higher = better), converted from Chroma's cosine
*distance* via ``score = 1 - distance`` (clamped).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chromadb

from regulumen.core.ingestion.models import Chunk
from regulumen.core.retrieval.embedder import Embedder

# Separator used to flatten ``clause_path`` (a list) into a scalar string for
# Chroma metadata (Chroma metadata values must be str/int/float/bool).
_CLAUSE_PATH_SEP = "\x1f"  # ASCII Unit Separator — never appears in clause paths

# Upsert batch size to keep request payloads reasonable.
_BATCH_SIZE = 128


@dataclass
class RetrievedChunk:
    """A single search hit. Stable contract for downstream retrieval/generation."""

    id: str
    text: str
    citation_label: str
    doc_id: str
    clause_path: list[str]
    score: float
    metadata: dict = field(default_factory=dict)


def _to_scalar_metadata(chunk: Chunk) -> dict:
    """Build a Chroma-safe (all-scalar) metadata dict for a chunk."""
    meta: dict[str, str | int | float | bool] = {
        "doc_id": chunk.doc_id,
        "citation_label": chunk.citation_label,
        "clause_path": _CLAUSE_PATH_SEP.join(chunk.clause_path),
    }
    for key, value in chunk.metadata.items():
        if isinstance(value, bool) or isinstance(value, (str, int, float)):
            meta[key] = value
        else:
            # Non-scalar (list/dict/None/etc.) -> coerce to string so we never
            # drop information silently while still satisfying Chroma's schema.
            meta[key] = str(value)
    return meta


def _split_clause_path(value: object) -> list[str]:
    if not isinstance(value, str) or not value:
        return []
    return value.split(_CLAUSE_PATH_SEP)


def _distance_to_score(distance: float) -> float:
    """Convert a Chroma cosine distance to a similarity in ``[0, 1]``."""
    score = 1.0 - float(distance)
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


class ChromaVectorStore:
    """Persistent dense vector store over a Chroma collection."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        path: str,
        collection_name: str = "regulumen",
    ) -> None:
        self._embedder = embedder
        self._client = chromadb.PersistentClient(path=path)
        # No embedding_function: we always pass vectors ourselves. cosine space
        # gives distances we map cleanly into [0, 1] similarities.
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    def index(self, chunks: list[Chunk]) -> None:
        """Embed and upsert chunks (idempotent on ``chunk.id``)."""
        if not chunks:
            return
        for start in range(0, len(chunks), _BATCH_SIZE):
            batch = chunks[start : start + _BATCH_SIZE]
            embeddings = self._embedder.embed_documents([c.text for c in batch])
            self._collection.upsert(
                ids=[c.id for c in batch],
                embeddings=embeddings,
                documents=[c.text for c in batch],
                metadatas=[_to_scalar_metadata(c) for c in batch],
            )

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return up to ``k`` best matches for ``query`` (best score first)."""
        if k <= 0:
            return []
        query_vec = self._embedder.embed_query(query)
        result = self._collection.query(
            query_embeddings=[query_vec],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]

        hits: list[RetrievedChunk] = []
        for i, chunk_id in enumerate(ids):
            meta = dict(metas[i]) if metas and metas[i] else {}
            citation_label = str(meta.pop("citation_label", ""))
            doc_id = str(meta.pop("doc_id", ""))
            clause_path = _split_clause_path(meta.pop("clause_path", ""))
            hits.append(
                RetrievedChunk(
                    id=str(chunk_id),
                    text=docs[i] if docs else "",
                    citation_label=citation_label,
                    doc_id=doc_id,
                    clause_path=clause_path,
                    score=_distance_to_score(dists[i]) if dists else 0.0,
                    metadata=meta,
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits

    def count(self) -> int:
        """Number of indexed chunks in the collection."""
        return self._collection.count()
