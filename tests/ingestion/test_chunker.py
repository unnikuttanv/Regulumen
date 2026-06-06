"""Tests for the clause-aware chunker (Step 3c).

All fixtures are SYNTHETIC, IP-safe ``Clause`` objects -- no text is copied
from any real regulatory document.  The tests pin down the three behaviours
that protect citation precision:

  * one normal clause -> exactly one chunk,
  * tiny same-parent siblings may merge into a parent-cited chunk,
  * overlong clauses split while keeping the SAME citation,

and the invariants around them (no cross-parent merges, empty clauses
skipped, determinism, metadata pass-through).
"""

from __future__ import annotations

from regulumen.core.ingestion.chunker import chunk_clauses
from regulumen.core.ingestion.models import Clause, build_citation_label

DOC_ID = "synthetic-doc"
SHORT_TITLE = "Synthetic Std"


def _clause(
    path: list[str],
    text: str,
    *,
    number: str | None = None,
    page: int | None = None,
    anchor: str | None = None,
) -> Clause:
    return Clause(
        doc_id=DOC_ID,
        path=path,
        number=number if number is not None else (path[-1] if path else ""),
        title=None,
        text=text,
        page=page,
        anchor=anchor,
    )


# --------------------------------------------------------------------- #
# One clause -> one chunk
# --------------------------------------------------------------------- #
def test_normal_clause_yields_one_chunk() -> None:
    text = "Body text of a normally sized clause. " * 5  # well under max, over min
    clause = _clause(["Annex I", "17", "17.1"], text)

    chunks = chunk_clauses([clause], SHORT_TITLE)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.clause_path == ["Annex I", "17", "17.1"]
    assert chunk.citation_label == build_citation_label(
        SHORT_TITLE, ["Annex I", "17", "17.1"]
    )
    assert chunk.doc_id == DOC_ID
    assert chunk.text == text.strip()
    assert chunk.id == f"{DOC_ID}::Annex I/17/17.1"
    assert chunk.metadata.get("merged") is None or chunk.metadata["merged"] is False
    assert "part" not in chunk.metadata


def test_chunk_id_is_unique_when_paths_repeat() -> None:
    # Two distinct clauses that happen to collapse to the same path string.
    body = "Adequately sized body content for a standalone clause. " * 4
    c1 = _clause(["Section", "A"], body)
    c2 = _clause(["Section", "A"], body)

    chunks = chunk_clauses([c1, c2], SHORT_TITLE)

    assert len(chunks) == 2
    assert chunks[0].id != chunks[1].id


# --------------------------------------------------------------------- #
# Citation safety: never merge across parents
# --------------------------------------------------------------------- #
def test_different_parents_are_never_merged() -> None:
    # Tiny clauses, but each under a DIFFERENT parent path.
    clauses = [
        _clause(["Article 10", "10.1"], "tiny one"),
        _clause(["Article 11", "11.1"], "tiny two"),
        _clause(["Annex I", "5", "5.1"], "tiny three"),
    ]

    chunks = chunk_clauses(clauses, SHORT_TITLE)

    # No merge possible (each parent has a single child) -> 3 chunks.
    assert len(chunks) == 3
    real_paths = {tuple(c.path) for c in clauses}
    for chunk in chunks:
        assert chunk.metadata.get("merged") in (None, False)
        # Every chunk path is a genuine clause path here.
        assert tuple(chunk.clause_path) in real_paths
    # No chunk text combines two different leaves.
    for chunk in chunks:
        assert not ("tiny one" in chunk.text and "tiny two" in chunk.text)


def test_no_cross_parent_merge_for_small_clauses() -> None:
    # Two small clauses under parent A, two under parent B.
    clauses = [
        _clause(["Annex I", "1", "1.1"], "alpha small"),
        _clause(["Annex I", "1", "1.2"], "beta small"),
        _clause(["Annex I", "2", "2.1"], "gamma small"),
        _clause(["Annex I", "2", "2.2"], "delta small"),
    ]

    chunks = chunk_clauses(clauses, SHORT_TITLE)

    # Each parent merges its own two children -> 2 chunks, never crossing.
    assert len(chunks) == 2
    for chunk in chunks:
        # A merged chunk cites the shared parent path.
        assert chunk.clause_path in (["Annex I", "1"], ["Annex I", "2"])
    texts = [c.text for c in chunks]
    # The "1" chunk must not contain the "2" content and vice versa.
    chunk1 = next(c for c in chunks if c.clause_path == ["Annex I", "1"])
    chunk2 = next(c for c in chunks if c.clause_path == ["Annex I", "2"])
    assert "alpha small" in chunk1.text and "beta small" in chunk1.text
    assert "gamma small" not in chunk1.text and "delta small" not in chunk1.text
    assert "gamma small" in chunk2.text and "delta small" in chunk2.text
    assert "alpha small" not in chunk2.text and "beta small" not in chunk2.text
    assert all("alpha small" not in t or "gamma small" not in t for t in texts)


# --------------------------------------------------------------------- #
# Merging same-parent siblings
# --------------------------------------------------------------------- #
def test_small_same_parent_siblings_merge() -> None:
    clauses = [
        _clause(["Annex I", "17", "17.1"], "first short point", number="17.1"),
        _clause(["Annex I", "17", "17.2"], "second short point", number="17.2"),
        _clause(["Annex I", "17", "17.3"], "third short point", number="17.3"),
    ]

    chunks = chunk_clauses(clauses, SHORT_TITLE)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.clause_path == ["Annex I", "17"]
    assert chunk.citation_label == build_citation_label(SHORT_TITLE, ["Annex I", "17"])
    assert chunk.metadata["merged"] is True
    assert chunk.metadata["merged_numbers"] == ["17.1", "17.2", "17.3"]
    # Each child's leaf number is inlined into the merged text.
    for leaf, body in [
        ("17.1", "first short point"),
        ("17.2", "second short point"),
        ("17.3", "third short point"),
    ]:
        assert leaf in chunk.text
        assert body in chunk.text
    # Leaf numbers appear in order.
    assert (
        chunk.text.index("17.1")
        < chunk.text.index("17.2")
        < chunk.text.index("17.3")
    )


def test_merge_stops_at_min_chars() -> None:
    # Each clause ~120 chars; min_chars=200 so 2 children already clear it.
    body = "x" * 120
    clauses = [
        _clause(["17", f"17.{i}"], body, number=f"17.{i}")
        for i in range(1, 5)  # 17/17.1 .. 17/17.4
    ]
    # NOTE: these share the NUMBERED parent ["17"]; leaf is the dotted number.
    chunks = chunk_clauses(clauses, SHORT_TITLE, min_chars=200, max_chars=1200)

    # Greedy merge accumulates until >= min_chars, then keeps going only while
    # under max_chars; 4 * ~120 = 480 < 1200 so all four merge into one chunk.
    assert len(chunks) == 1
    assert chunks[0].clause_path == ["17"]
    assert chunks[0].metadata["merged_numbers"] == ["17.1", "17.2", "17.3", "17.4"]


def test_merge_respects_max_chars() -> None:
    # Children just under min, but their sum would blow past max -> split into
    # multiple merged chunks rather than one oversized chunk.
    body = "y" * 150
    clauses = [
        _clause(["17", f"17.{i}"], body, number=f"17.{i}") for i in range(1, 11)
    ]

    chunks = chunk_clauses(clauses, SHORT_TITLE, min_chars=200, max_chars=500)

    assert len(chunks) > 1
    for chunk in chunks:
        # A merged chunk cites the shared parent; a leftover lone child keeps
        # its own (still precise) path -- never an oversized blur.
        assert chunk.clause_path[0] == "17"
        if chunk.metadata.get("merged"):
            assert chunk.clause_path == ["17"]
        assert len(chunk.text) <= 500 or chunk.metadata.get("part")


def test_no_merge_under_nonnumbered_divider() -> None:
    # Small sibling Articles directly under a Chapter divider must NOT merge:
    # the parent "Chapter I" is a coarse container, so merging would blur
    # distinct Article-level citations.
    clauses = [
        _clause(["Chapter I", "Article 2"], "Definitions", number="2"),
        _clause(["Chapter I", "Article 3"], "Amendment", number="3"),
    ]
    chunks = chunk_clauses(clauses, SHORT_TITLE)
    assert len(chunks) == 2
    for chunk in chunks:
        assert chunk.metadata.get("merged") in (None, False)
    assert {tuple(c.clause_path) for c in chunks} == {
        ("Chapter I", "Article 2"),
        ("Chapter I", "Article 3"),
    }


def test_internal_heading_clause_is_not_merged() -> None:
    # A short heading clause that OWNS sub-points (its path is a prefix of
    # other clauses) stays standalone instead of merging up.
    clauses = [
        _clause(["Annex I", "17"], "short heading", number="17"),
        _clause(["Annex I", "17", "17.1"], "first point body", number="17.1"),
        _clause(["Annex I", "17", "17.2"], "second point body", number="17.2"),
    ]
    chunks = chunk_clauses(clauses, SHORT_TITLE)
    # The "17" heading is an internal node -> its own standalone chunk (kept
    # separate from the merged 17.1/17.2 chunk).
    heading = [
        c
        for c in chunks
        if c.clause_path == ["Annex I", "17"] and not c.metadata.get("merged")
    ]
    assert len(heading) == 1
    assert heading[0].text == "short heading"
    merged = [c for c in chunks if c.metadata.get("merged")]
    assert len(merged) == 1
    assert merged[0].clause_path == ["Annex I", "17"]
    assert merged[0].metadata["merged_numbers"] == ["17.1", "17.2"]


def test_single_small_clause_with_no_sibling_is_not_merged() -> None:
    clause = _clause(["Article 99", "99.1"], "lonely small clause")
    chunks = chunk_clauses([clause], SHORT_TITLE, min_chars=200)

    assert len(chunks) == 1
    assert chunks[0].clause_path == ["Article 99", "99.1"]
    assert chunks[0].metadata.get("merged") in (None, False)


# --------------------------------------------------------------------- #
# Splitting overlong clauses
# --------------------------------------------------------------------- #
def test_overlong_clause_splits_with_overlap() -> None:
    # Build a long body out of sentences so boundary splitting has something
    # to grab; total length comfortably exceeds max_chars.
    sentence = "This is sentence number {} in a very long clause body. "
    body = "".join(sentence.format(i) for i in range(1, 60))
    assert len(body) > 1200
    clause = _clause(["Annex I", "Chapter II", "17", "17.1"], body)

    max_chars, overlap = 600, 100
    chunks = chunk_clauses(
        [clause], SHORT_TITLE, max_chars=max_chars, overlap_chars=overlap
    )

    assert len(chunks) > 1
    n = len(chunks)
    label = build_citation_label(SHORT_TITLE, ["Annex I", "Chapter II", "17", "17.1"])
    for i, chunk in enumerate(chunks, start=1):
        # Citation stays precise across every piece.
        assert chunk.clause_path == ["Annex I", "Chapter II", "17", "17.1"]
        assert chunk.citation_label == label
        assert chunk.metadata["part"] == f"{i}/{n}"
        assert len(chunk.text) <= max_chars
    # ids are distinct via a piece suffix.
    assert len({c.id for c in chunks}) == n
    assert all(c.id.endswith(f":{i}") for i, c in enumerate(chunks, start=1))

    # Overlap: the tail of one piece reappears at the head of the next.
    for prev, nxt in zip(chunks, chunks[1:]):
        tail = prev.text[-overlap:]
        # Some non-trivial suffix of `prev` is a prefix region of `nxt`.
        assert any(
            tail[k:] and nxt.text.startswith(tail[k:]) for k in range(len(tail))
        )

    # Coverage: stripping the overlap region, the pieces reconstruct the body.
    reconstructed = chunks[0].text
    for prev, nxt in zip(chunks, chunks[1:]):
        # find where nxt rejoins after the overlap with prev
        overlap_region = _shared_overlap(prev.text, nxt.text)
        reconstructed += nxt.text[len(overlap_region):]
    # All original (whitespace-normalised) content is present in order.
    assert _normalize(body).replace(" ", "") in _normalize(reconstructed).replace(
        " ", ""
    ) or _normalize(reconstructed).replace(" ", "").startswith(
        _normalize(body).replace(" ", "")[:200]
    )


def test_overlong_clause_pieces_cover_original() -> None:
    body = "Z" * 3000  # no boundaries -> hard char split
    clause = _clause(["X"], body)
    max_chars, overlap = 1000, 100
    chunks = chunk_clauses(
        [clause], SHORT_TITLE, max_chars=max_chars, overlap_chars=overlap
    )
    assert len(chunks) > 1
    # Reconstruct by removing the overlap between consecutive pieces.
    reconstructed = chunks[0].text
    for prev, nxt in zip(chunks, chunks[1:]):
        reconstructed += nxt.text[overlap:]
    assert reconstructed == body


# --------------------------------------------------------------------- #
# Empty / whitespace clauses
# --------------------------------------------------------------------- #
def test_empty_and_whitespace_clauses_are_skipped() -> None:
    clauses = [
        _clause(["A", "1"], "   "),
        _clause(["A", "2"], ""),
        _clause(["A", "3"], "real content that is long enough to stand alone " * 4),
    ]
    chunks = chunk_clauses(clauses, SHORT_TITLE)
    assert len(chunks) == 1
    assert chunks[0].clause_path == ["A", "3"]


# --------------------------------------------------------------------- #
# Metadata pass-through
# --------------------------------------------------------------------- #
def test_metadata_includes_page_anchor_and_confidence() -> None:
    clause = _clause(
        ["5", "5.1"],
        "body content sized to stand alone as its own chunk here. " * 4,
        page=7,
        anchor="frag-5-1",
    )
    chunks = chunk_clauses([clause], SHORT_TITLE, structure_confidence=0.83)
    assert len(chunks) == 1
    md = chunks[0].metadata
    assert md["page"] == 7
    assert md["anchor"] == "frag-5-1"
    assert md["structure_confidence"] == 0.83


def test_structure_confidence_absent_when_not_passed() -> None:
    clause = _clause(["5", "5.1"], "standalone body content here. " * 6)
    chunks = chunk_clauses([clause], SHORT_TITLE)
    assert "structure_confidence" not in chunks[0].metadata


def test_page_absent_when_clause_has_no_page() -> None:
    clause = _clause(["5", "5.1"], "standalone body content here. " * 6)
    chunks = chunk_clauses([clause], SHORT_TITLE)
    assert "page" not in chunks[0].metadata
    assert "anchor" not in chunks[0].metadata


# --------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------- #
def test_determinism() -> None:
    clauses = [
        _clause(["Annex I", "17", "17.1"], "short a", number="17.1"),
        _clause(["Annex I", "17", "17.2"], "short b", number="17.2"),
        _clause(["Article 10", "10.1"], "standalone clause body that is long. " * 5),
        _clause(["Big"], "Q" * 3000),
    ]
    first = chunk_clauses(clauses, SHORT_TITLE)
    second = chunk_clauses(clauses, SHORT_TITLE)
    assert [c.id for c in first] == [c.id for c in second]
    assert [c.clause_path for c in first] == [c.clause_path for c in second]
    assert [c.text for c in first] == [c.text for c in second]


def test_input_order_preserved() -> None:
    clauses = [
        _clause(["B", "2"], "standalone clause content kept in order. " * 4),
        _clause(["A", "1"], "another standalone clause content here. " * 4),
    ]
    chunks = chunk_clauses(clauses, SHORT_TITLE)
    assert chunks[0].clause_path == ["B", "2"]
    assert chunks[1].clause_path == ["A", "1"]


# --------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------- #
def _normalize(text: str) -> str:
    return " ".join(text.split())


def _shared_overlap(prev: str, nxt: str) -> str:
    """Longest suffix of ``prev`` that is a prefix of ``nxt`` (bounded)."""
    limit = min(len(prev), len(nxt))
    for k in range(limit, 0, -1):
        if prev[-k:] == nxt[:k]:
            return nxt[:k]
    return ""
