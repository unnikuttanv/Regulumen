"""Typed data models for the ReguLumen ingestion pipeline.

These are the stable contract used by every downstream step
(parsers, chunker, retrieval, generation, eval).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SourceDoc(BaseModel):
    """A single regulatory source document as fetched from its origin URL.

    Carries provenance metadata (url, retrieved_at, jurisdiction) alongside
    citation context so downstream modules can attribute answers correctly.
    """

    id: str
    """Slug identifier, e.g. ``"eu-mdr-2017-745"``."""

    title: str
    """Human-readable title, e.g. ``"EU MDR — Regulation (EU) 2017/745"``."""

    short_title: str
    """Citation prefix used in labels, e.g. ``"EU MDR"``."""

    source_type: Literal["eurlex_html", "guidance_pdf"]
    """How the document was sourced."""

    jurisdiction: str
    """Regulatory jurisdiction, e.g. ``"EU"``, ``"US-FDA"``, ``"IMDRF"``."""

    url: str
    """Canonical URL the document was fetched from."""

    license_note: str
    """IP/redistribution note for this document."""

    retrieved_at: str
    """ISO date string of when the document was fetched, e.g. ``"2026-06-01"``."""


class Clause(BaseModel):
    """A single numbered (or unnumbered) structural unit within a SourceDoc.

    ``path`` is the FULL structural path from the top-level section down to
    and INCLUDING this clause's own leaf identifier, e.g.
    ``["Annex I", "17", "17.1"]``.  ``number`` is just the leaf identifier
    (e.g. ``"17.1"``); use ``""`` for unnumbered paragraphs.
    """

    doc_id: str
    """Identifier of the parent :class:`SourceDoc`."""

    path: list[str]
    """Ordered structural path including the leaf, e.g. ``["Annex I", "17", "17.1"]``."""

    number: str
    """Leaf identifier only; ``""`` if unnumbered."""

    title: str | None = None
    """Optional section heading."""

    text: str
    """Full text content of this clause."""

    page: int | None = None
    """Page number (for PDFs); ``None`` for HTML sources."""

    anchor: str | None = None
    """HTML anchor fragment; ``None`` for PDF sources."""


class Chunk(BaseModel):
    """A retrieval-ready text unit derived from one :class:`Clause`.

    ``clause_path`` is copied verbatim from the originating ``Clause.path``
    and drives ``build_citation_label`` to construct the human-readable
    citation string attached to every answer.
    """

    id: str
    """Unique chunk identifier."""

    doc_id: str
    """Identifier of the parent :class:`SourceDoc`."""

    clause_path: list[str]
    """Copied from ``Clause.path``; drives citation label generation."""

    citation_label: str
    """Human-readable citation, e.g. ``"EU MDR, Annex I §17.1"``."""

    text: str
    """Chunk text content."""

    metadata: dict = Field(default_factory=dict)
    """Arbitrary metadata, e.g. ``{"structure_confidence": 0.95, "page": 12}``."""


def build_citation_label(short_title: str, clause_path: list[str]) -> str:
    """Return a human-readable citation label from a short title and clause path.

    Format: ``"{short_title}, {ancestors} §{leaf}"`` where ``leaf`` is the last
    path element and ``ancestors`` is the remaining elements joined by a space.
    If ``clause_path`` has one element, format as ``"{short_title}, §{leaf}"``.
    If ``clause_path`` is empty, return ``short_title``.

    Examples::

        build_citation_label("EU MDR", ["Annex I", "17.2"])
        # -> "EU MDR, Annex I §17.2"

        build_citation_label("EU MDR", ["Article 10", "10.1"])
        # -> "EU MDR, Article 10 §10.1"

        build_citation_label("EU MDR", ["62"])
        # -> "EU MDR, §62"

        build_citation_label("FDA SaMD", [])
        # -> "FDA SaMD"
    """
    if not clause_path:
        return short_title
    *ancestors, leaf = clause_path
    if ancestors:
        section = " ".join(ancestors)
        return f"{short_title}, {section} §{leaf}"
    return f"{short_title}, §{leaf}"
