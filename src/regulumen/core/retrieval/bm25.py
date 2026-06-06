"""BM25 lexical retrieval for ReguLumen.

Pure dense (vector) search embeds meaning but routinely *misses exact clause
identifiers* — querying "62304 §5.1.1" or "Article 10" should return the clause
that literally cites those numbers, yet a paraphrase-trained embedder buries them.
BM25 is the lexical counterweight: it scores documents by exact term overlap, so a
clause-number query lands squarely on the clause that bears that number. Step 4b's
:class:`~regulumen.core.retrieval.hybrid.HybridRetriever` fuses this with dense
search to get the best of both.

Tokenization (see :func:`tokenize`) is the crux: a naive ``\\w+`` split would
shatter "5.1.1" into "5", "1", "1" and a query for "5.1.1" would then match any
chunk mentioning "1". We instead keep dotted clause forms intact AND emit their
hierarchical prefixes, so "5.1.1" is a single strong token while still partially
matching "5.1".

Results are returned as :class:`RetrievedChunk` (reused from the vector store) so
dense and lexical hits share one stable type. Scores are normalized to ``[0, 1]``
(see :meth:`BM25Index.search`).
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from regulumen.core.ingestion.models import Chunk
from regulumen.core.retrieval.vector_store import RetrievedChunk

# Token grammar:
#   * ``[a-z0-9]+(?:\.[a-z0-9]+)*`` matches an alphanumeric run, optionally with
#     dotted sub-parts — so "5.1.1", "10", "62304", "article" are each ONE token
#     and the dots inside a clause number are preserved (the whole point).
# Punctuation such as "§", "(", ")", "," is simply not matched and thus dropped.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Tokenize ``text`` for BM25, preserving regulatory clause numbers.

    The text is lower-cased and split on the grammar above. In addition to each
    matched token, every *hierarchical prefix* of a dotted token is emitted so a
    deeper clause still partially matches a shallower query:

        "5.1.1" -> ["5.1.1", "5", "5.1"]

    Rationale:
      * Keeping "5.1.1" whole means a query for "5.1.1" gets a strong, specific
        hit instead of matching every chunk that merely mentions "1".
      * Emitting the "5" and "5.1" prefixes means a query for the *section*
        "5.1" still finds its sub-clauses, which matches how regulators cite.
      * "Article 10(1)" -> ["article", "10"] because "(" / ")" are not token
        characters; the section sign "§" is likewise dropped.

    Duplicate-but-ordered output is fine — BM25 term frequencies are computed
    from the list, and prefixes deliberately add weight to clause numbers.
    """
    tokens: list[str] = []
    for match in _TOKEN_RE.findall(text.lower()):
        tokens.append(match)
        if "." in match:
            parts = match.split(".")
            # Emit prefixes "5", "5.1" for "5.1.1" (skip the full form: already added).
            for end in range(1, len(parts)):
                tokens.append(".".join(parts[:end]))
    return tokens


class BM25Index:
    """In-memory BM25 (Okapi) lexical index over a chunk collection."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def index(self, chunks: list[Chunk]) -> None:
        """Build the BM25 index over ``chunks``.

        The searchable document for each chunk concatenates ``citation_label``,
        the joined ``clause_path`` and ``text`` so that a query for a clause
        NUMBER matches lexically even when the number appears only in the
        citation/path and not in the prose body.
        """
        self._chunks = list(chunks)
        if not self._chunks:
            self._bm25 = None
            return
        corpus = [tokenize(self._searchable_text(c)) for c in self._chunks]
        self._bm25 = BM25Okapi(corpus)

    @staticmethod
    def _searchable_text(chunk: Chunk) -> str:
        return " ".join(
            [chunk.citation_label, " ".join(chunk.clause_path), chunk.text]
        )

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return up to ``k`` lexical hits, best-first, scored in ``[0, 1]``.

        Normalization: BM25 scores are unbounded and corpus-dependent, so we
        divide every score by the maximum score *in this query's result set*.
        The top hit therefore scores ``1.0`` and the rest are relative. Chunks
        with a BM25 score of ``0`` (no shared terms) are dropped, so an
        all-zero query returns ``[]`` (and we never divide by zero).

        Tradeoff: per-query normalization promotes the top BM25 hit to 1.0
        regardless of absolute relevance, so a weakly-but-best-matching hit
        still contributes full weight at alpha<1.  Interpret ``alpha`` as how
        much to trust BM25's *relative ranking* within its own result set, not
        absolute lexical confidence.  Acceptable for the clause-number use case
        (match is definitive or absent); broad lexical queries may warrant
        raw-score thresholding before normalization.
        """
        # Blank/whitespace tokenizes to []; guard here so callers don't have to.
        if not query.strip():
            return []
        if k <= 0 or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        # Corpus is non-empty here (guarded above by self._bm25 is not None).
        max_score = float(max(scores))
        if max_score <= 0.0:
            return []

        # Sort by score descending; tie-break by chunk id ascending for stable,
        # deterministic ordering.
        ranked = sorted(
            range(len(scores)),
            key=lambda i: (-scores[i], self._chunks[i].id),
        )
        hits: list[RetrievedChunk] = []
        for i in ranked:
            if scores[i] <= 0.0:
                continue
            chunk = self._chunks[i]
            hits.append(
                RetrievedChunk(
                    id=chunk.id,
                    text=chunk.text,
                    citation_label=chunk.citation_label,
                    doc_id=chunk.doc_id,
                    clause_path=list(chunk.clause_path),
                    score=float(scores[i]) / float(max_score),
                    metadata=dict(chunk.metadata),
                )
            )
            if len(hits) >= k:
                break
        return hits

    def count(self) -> int:
        """Number of indexed chunks."""
        return len(self._chunks)
