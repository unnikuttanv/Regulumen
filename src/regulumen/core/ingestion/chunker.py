"""Clause-aware chunker: turn parsed :class:`Clause` trees into retrieval
:class:`Chunk` objects while preserving citation precision (Step 3c).

This is the citation-critical seam of the ingestion pipeline.  The guiding
invariant: **a chunk's ``citation_label`` must never blur two different clause
numbers**.  Three behaviours follow from that:

  * one clause -> one chunk by default;
  * tiny consecutive siblings under the SAME parent path may merge into one
    chunk that cites the shared PARENT (a truthful, less-specific citation),
    with each child's leaf number inlined into the text;
  * an overlong clause is split into several pieces that ALL keep the original
    ``clause_path`` / ``citation_label`` (the citation stays precise) and are
    disambiguated only by ``chunk.id`` suffix and a ``metadata["part"]`` tag.

When in doubt the chunker does NOT merge -- it emits separate chunks.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from regulumen.core.ingestion.models import Chunk, Clause, build_citation_label

logger = logging.getLogger(__name__)

# Default size envelope (chars).  These are read from parameters by callers and
# tuned later (Step 6/7); hoisted here only as documented defaults.
_DEFAULT_MAX_CHARS = 1200
_DEFAULT_MIN_CHARS = 200
_DEFAULT_OVERLAP_CHARS = 150

# Separator between inlined child clauses in a merged chunk.
_MERGE_JOIN = " "

# Boundary preference order when splitting an overlong clause: paragraph break,
# then sentence end, then whitespace.  Each pattern matches the *end* position
# of a good break.
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_BREAK = re.compile(r"[.!?]['\")\]]?\s")
_WHITESPACE_BREAK = re.compile(r"\s")


def chunk_clauses(
    clauses: list[Clause],
    short_title: str,
    *,
    max_chars: int = _DEFAULT_MAX_CHARS,
    min_chars: int = _DEFAULT_MIN_CHARS,
    overlap_chars: int = _DEFAULT_OVERLAP_CHARS,
    structure_confidence: float | None = None,
) -> list[Chunk]:
    """Convert ``clauses`` into retrieval :class:`Chunk` objects.

    Args:
        clauses: Parsed clauses, in document order.
        short_title: Citation prefix (e.g. ``"EU MDR"``).
        max_chars: Upper bound on a single chunk's text length; longer clauses
            are split.
        min_chars: A clause shorter than this is a merge candidate.
        overlap_chars: Character overlap between consecutive split pieces.
        structure_confidence: Optional per-document parse confidence to surface
            into every chunk's metadata.

    Returns:
        Chunks in input order; deterministic for identical input.
    """
    # 1. Normalise + drop empties, preserving order.
    units = [u for c in clauses if (u := _Unit.from_clause(c)) is not None]

    # An internal node is any clause path that is the proper prefix of another
    # clause's path (e.g. an Article heading whose numbered points exist as
    # their own clauses).  Merging such a heading under its parent would blur
    # Article-level granularity, so these units are pinned as standalone.
    internal = _internal_node_paths(units)
    for u in units:
        u.mergeable = tuple(u.path) not in internal

    # 2. Group into merge runs, then expand each run into chunks.
    chunks: list[Chunk] = []
    seen_ids: dict[str, int] = {}
    i = 0
    while i < len(units):
        run = _collect_merge_run(units, i, min_chars=min_chars, max_chars=max_chars)
        i += len(run)
        chunks.extend(
            _emit_run(
                run,
                short_title=short_title,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
                structure_confidence=structure_confidence,
                seen_ids=seen_ids,
            )
        )

    logger.info(
        "Chunked %d clauses (%s) -> %d chunks", len(clauses), short_title, len(chunks)
    )
    return chunks


# --------------------------------------------------------------------- #
# Internal unit
# --------------------------------------------------------------------- #
@dataclass
class _Unit:
    """A non-empty clause with its text stripped and parent path computed."""

    clause: Clause
    text: str
    path: list[str]
    parent: tuple[str, ...]
    leaf: str
    mergeable: bool = True

    @classmethod
    def from_clause(cls, clause: Clause) -> _Unit | None:
        text = (clause.text or "").strip()
        if not text:
            return None
        path = list(clause.path)
        parent = tuple(path[:-1]) if len(path) > 1 else ()
        # The path leaf is the unambiguous identifier to inline on merge
        # (e.g. "Article 2", "17.1"); fall back to ``number`` only if absent.
        leaf = (path[-1] if path else "") or clause.number
        return cls(clause=clause, text=text, path=path, parent=parent, leaf=leaf)


# --------------------------------------------------------------------- #
# Merge grouping
# --------------------------------------------------------------------- #
def _collect_merge_run(
    units: list[_Unit], start: int, *, min_chars: int, max_chars: int
) -> list[_Unit]:
    """Return the run of units (>=1) starting at ``start`` to treat together.

    A run is a single unit unless the unit is small (< ``min_chars``) and has
    at least one consecutive sibling sharing the SAME parent path that can be
    merged without exceeding ``max_chars``.  Merging never crosses a parent
    boundary and never merges a top-level (parentless) clause.
    """
    first = units[start]
    if (
        len(first.text) >= min_chars
        or not first.parent
        or not first.mergeable
        or not _is_numbered_parent(first.parent)
    ):
        return [first]

    run = [first]
    acc = len(first.text)
    j = start + 1
    while j < len(units):
        nxt = units[j]
        # Stop at a parent change -- crossing parents would break the citation.
        if nxt.parent != first.parent:
            break
        # Never absorb an internal-heading sibling (it owns sub-clauses).
        if not nxt.mergeable:
            break
        # Stop if adding this unit would exceed the size ceiling.
        if acc + len(nxt.text) > max_chars:
            break
        run.append(nxt)
        acc += len(nxt.text)
        j += 1
        # Once we have cleared the minimum, only keep absorbing further small
        # siblings; stop as soon as the *next* one is itself large enough to
        # stand alone (it deserves its own chunk).
        if acc >= min_chars:
            if j < len(units) and (
                units[j].parent != first.parent
                or not units[j].mergeable
                or len(units[j].text) >= min_chars
            ):
                break

    return run


# --------------------------------------------------------------------- #
# Emission
# --------------------------------------------------------------------- #
def _emit_run(
    run: list[_Unit],
    *,
    short_title: str,
    max_chars: int,
    overlap_chars: int,
    structure_confidence: float | None,
    seen_ids: dict[str, int],
) -> list[Chunk]:
    """Emit chunk(s) for one merge run (length 1 = no merge)."""
    if len(run) == 1:
        return _emit_single(
            run[0],
            short_title=short_title,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            structure_confidence=structure_confidence,
            seen_ids=seen_ids,
        )
    return [
        _emit_merged(
            run,
            short_title=short_title,
            structure_confidence=structure_confidence,
            seen_ids=seen_ids,
        )
    ]


def _emit_single(
    unit: _Unit,
    *,
    short_title: str,
    max_chars: int,
    overlap_chars: int,
    structure_confidence: float | None,
    seen_ids: dict[str, int],
) -> list[Chunk]:
    """Emit one chunk, or several split pieces if the clause is overlong."""
    label = build_citation_label(short_title, unit.path)
    base_meta = _base_metadata(unit, structure_confidence)

    if len(unit.text) <= max_chars:
        chunk_id = _unique_id(_path_id(unit.clause.doc_id, unit.path), seen_ids)
        return [
            Chunk(
                id=chunk_id,
                doc_id=unit.clause.doc_id,
                clause_path=unit.path,
                citation_label=label,
                text=unit.text,
                metadata=base_meta,
            )
        ]

    pieces = _split_text(unit.text, max_chars=max_chars, overlap_chars=overlap_chars)
    n = len(pieces)
    base_id = _path_id(unit.clause.doc_id, unit.path)
    chunks: list[Chunk] = []
    for idx, piece in enumerate(pieces, start=1):
        meta = dict(base_meta)
        meta["part"] = f"{idx}/{n}"
        chunk_id = _unique_id(f"{base_id}:{idx}", seen_ids)
        chunks.append(
            Chunk(
                id=chunk_id,
                doc_id=unit.clause.doc_id,
                clause_path=unit.path,
                citation_label=label,
                text=piece,
                metadata=meta,
            )
        )
    return chunks


def _emit_merged(
    run: list[_Unit],
    *,
    short_title: str,
    structure_confidence: float | None,
    seen_ids: dict[str, int],
) -> Chunk:
    """Emit a single chunk citing the shared parent of merged siblings."""
    first = run[0]
    parent_path = list(first.parent)
    label = build_citation_label(short_title, parent_path)
    numbers = [u.leaf for u in run]

    # Inline each child's leaf number before its body so the merged text stays
    # navigable and the (parent-level) citation remains truthful.
    text = _MERGE_JOIN.join(f"{u.leaf} {u.text}".strip() for u in run)

    meta: dict = {"merged": True, "merged_numbers": numbers}
    pages = [u.clause.page for u in run if u.clause.page is not None]
    if pages:
        meta["page"] = pages[0]
    if structure_confidence is not None:
        meta["structure_confidence"] = structure_confidence

    base_id = _path_id(first.clause.doc_id, parent_path) + "::merged"
    chunk_id = _unique_id(base_id, seen_ids)
    return Chunk(
        id=chunk_id,
        doc_id=first.clause.doc_id,
        clause_path=parent_path,
        citation_label=label,
        text=text,
        metadata=meta,
    )


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def _base_metadata(unit: _Unit, structure_confidence: float | None) -> dict:
    meta: dict = {}
    if unit.clause.page is not None:
        meta["page"] = unit.clause.page
    if unit.clause.anchor is not None:
        meta["anchor"] = unit.clause.anchor
    if structure_confidence is not None:
        meta["structure_confidence"] = structure_confidence
    return meta


def _is_numbered_parent(parent: tuple[str, ...]) -> bool:
    """Return ``True`` if the merge parent is a numbered clause, not a divider.

    Merging is only worthwhile when the shared parent is itself a numbered
    heading (e.g. ``"Article 1"``, ``"17"``): its sub-points (1.6, 17.1) are
    fine-grained leaves that a parent-level citation still pins down precisely.
    Coarse structural containers whose leaf carries no digit (``"Chapter I"``,
    ``"Annex I"``, ``"Section 2"`` is numbered but ``"Chapter I"`` is not) would
    blur distinct child sections, so we never merge directly under them.
    """
    leaf = parent[-1] if parent else ""
    return any(ch.isdigit() for ch in leaf)


def _internal_node_paths(units: list[_Unit]) -> set[tuple[str, ...]]:
    """Return clause paths that are a proper prefix of another clause's path.

    Such a clause is a structural heading (it owns sub-clauses that exist as
    their own units); merging it under its parent would erase a citable level,
    so it is kept standalone.
    """
    paths = [tuple(u.path) for u in units]
    internal: set[tuple[str, ...]] = set()
    for p in paths:
        for k in range(1, len(p)):
            internal.add(p[:k])
    return internal


def _path_id(doc_id: str, path: list[str]) -> str:
    """Deterministic base id from doc + path, e.g. ``doc::Annex I/17/17.1``."""
    return f"{doc_id}::{'/'.join(path)}"


def _unique_id(base: str, seen_ids: dict[str, int]) -> str:
    """Return ``base`` (or ``base#k``) ensuring global uniqueness in this run."""
    n = seen_ids.get(base, 0)
    seen_ids[base] = n + 1
    return base if n == 0 else f"{base}#{n}"


def _split_text(text: str, *, max_chars: int, overlap_chars: int) -> list[str]:
    """Split ``text`` into <= ``max_chars`` pieces with ``overlap_chars`` overlap.

    Splits prefer a paragraph break, then a sentence end, then whitespace near
    the limit, falling back to a hard character cut so progress is guaranteed.
    Consecutive pieces overlap by exactly ``overlap_chars`` characters (or the
    full preceding piece when it is shorter), so reconstruction by stripping
    the leading ``overlap_chars`` of each later piece recovers the original.
    """
    if len(text) <= max_chars:
        return [text]

    overlap = max(0, min(overlap_chars, max_chars - 1))
    pieces: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            end = _best_break(text, start, end)
        pieces.append(text[start:end])
        if end >= n:
            break
        # Step forward, leaving `overlap` chars of context shared with the next
        # piece.  Guarantee forward progress even if overlap >= advanced span.
        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start
    return pieces


def _best_break(text: str, start: int, hard_end: int) -> int:
    """Find a good break position at or before ``hard_end`` (but past ``start``).

    Searches a window back from ``hard_end`` for a paragraph break, then a
    sentence boundary, then any whitespace.  Returns ``hard_end`` unchanged if
    no boundary is found in the window (hard cut).
    """
    # Don't search the whole piece -- only a trailing window, so pieces stay
    # close to max_chars.
    window_start = max(start + 1, hard_end - (hard_end - start) // 3)
    segment = text[start:hard_end]
    offset_floor = window_start - start

    for pattern in (_PARAGRAPH_BREAK, _SENTENCE_BREAK, _WHITESPACE_BREAK):
        best = -1
        for m in pattern.finditer(segment):
            if m.end() >= offset_floor:
                best = m.end()
        if best != -1:
            return start + best

    return hard_end
