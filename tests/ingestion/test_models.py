"""Tests for ingestion data models and citation helper (Step 2 / TDD)."""

from __future__ import annotations

import pytest

from regulumen.core.ingestion.models import (
    Chunk,
    Clause,
    SourceDoc,
    build_citation_label,
)

# ---------------------------------------------------------------------------
# SourceDoc
# ---------------------------------------------------------------------------

def test_source_doc_round_trip():
    doc = SourceDoc(
        id="eu-mdr-2017-745",
        title="EU MDR — Regulation (EU) 2017/745",
        short_title="EU MDR",
        source_type="eurlex_html",
        jurisdiction="EU",
        url="https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32017R0745",
        license_note="Public document; sourced from EUR-Lex.",
        retrieved_at="2026-06-01",
    )
    assert doc.id == "eu-mdr-2017-745"
    assert doc.title == "EU MDR — Regulation (EU) 2017/745"
    assert doc.short_title == "EU MDR"
    assert doc.source_type == "eurlex_html"
    assert doc.jurisdiction == "EU"
    assert doc.url == "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32017R0745"
    assert doc.license_note == "Public document; sourced from EUR-Lex."
    assert doc.retrieved_at == "2026-06-01"


def test_source_doc_guidance_pdf():
    doc = SourceDoc(
        id="fda-premarket-software-functions",
        title="FDA — Content of Premarket Submissions for Device Software Functions",
        short_title="FDA Premarket SW",
        source_type="guidance_pdf",
        jurisdiction="US-FDA",
        url="https://www.fda.gov/media/153781/download",
        license_note="Public FDA guidance document.",
        retrieved_at="2026-06-01",
    )
    assert doc.source_type == "guidance_pdf"
    assert doc.jurisdiction == "US-FDA"


def test_source_doc_invalid_source_type():
    with pytest.raises(Exception):
        SourceDoc(
            id="bad",
            title="Bad Doc",
            short_title="Bad",
            source_type="unknown_type",  # invalid literal
            jurisdiction="EU",
            url="https://example.com",
            license_note="n/a",
            retrieved_at="2026-06-01",
        )


# ---------------------------------------------------------------------------
# Clause
# ---------------------------------------------------------------------------

def test_clause_round_trip():
    clause = Clause(
        doc_id="eu-mdr-2017-745",
        path=["Annex I", "17", "17.1"],
        number="17.1",
        title="General safety requirements",
        text="Devices shall be designed and manufactured in such a way that...",
        page=None,
        anchor="annex-i-17-1",
    )
    assert clause.doc_id == "eu-mdr-2017-745"
    assert clause.path == ["Annex I", "17", "17.1"]
    assert clause.number == "17.1"
    assert clause.title == "General safety requirements"
    assert "Devices shall be" in clause.text
    assert clause.page is None
    assert clause.anchor == "annex-i-17-1"


def test_clause_optional_fields_default():
    clause = Clause(
        doc_id="eu-mdr-2017-745",
        path=["Article 2"],
        number="2",
        text="Definitions.",
    )
    assert clause.title is None
    assert clause.page is None
    assert clause.anchor is None


def test_clause_empty_number():
    clause = Clause(
        doc_id="eu-mdr-2017-745",
        path=["Preamble"],
        number="",
        text="Whereas...",
    )
    assert clause.number == ""


def test_clause_pdf_has_page():
    clause = Clause(
        doc_id="fda-premarket-software-functions",
        path=["Section 3", "3.1"],
        number="3.1",
        text="Software documentation...",
        page=12,
    )
    assert clause.page == 12
    assert clause.anchor is None


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------

def test_chunk_round_trip():
    chunk = Chunk(
        id="eu-mdr-2017-745__annex-i-17-1__0",
        doc_id="eu-mdr-2017-745",
        clause_path=["Annex I", "17", "17.1"],
        citation_label="EU MDR, Annex I 17 §17.1",
        text="Devices shall be designed and manufactured in such a way that...",
        metadata={"structure_confidence": 0.95, "anchor": "annex-i-17-1"},
    )
    assert chunk.id == "eu-mdr-2017-745__annex-i-17-1__0"
    assert chunk.doc_id == "eu-mdr-2017-745"
    assert chunk.clause_path == ["Annex I", "17", "17.1"]
    assert chunk.citation_label == "EU MDR, Annex I 17 §17.1"
    assert "Devices shall be" in chunk.text
    assert chunk.metadata["structure_confidence"] == 0.95


def test_chunk_metadata_defaults_empty():
    chunk = Chunk(
        id="test-chunk-1",
        doc_id="eu-mdr-2017-745",
        clause_path=["Article 10"],
        citation_label="EU MDR, §Article 10",
        text="Some text here.",
    )
    assert chunk.metadata == {}


def test_chunk_built_with_citation_label():
    label = build_citation_label("EU MDR", ["Annex I", "17.2"])
    chunk = Chunk(
        id="chunk-annex-i-17-2",
        doc_id="eu-mdr-2017-745",
        clause_path=["Annex I", "17.2"],
        citation_label=label,
        text="The device shall...",
    )
    assert chunk.citation_label == "EU MDR, Annex I §17.2"


# ---------------------------------------------------------------------------
# build_citation_label
# ---------------------------------------------------------------------------

def test_citation_label_two_elements():
    result = build_citation_label("EU MDR", ["Annex I", "17.2"])
    assert result == "EU MDR, Annex I §17.2"


def test_citation_label_article_and_sub():
    result = build_citation_label("EU MDR", ["Article 10", "10.1"])
    assert result == "EU MDR, Article 10 §10.1"


def test_citation_label_single_element():
    result = build_citation_label("EU MDR", ["62"])
    assert result == "EU MDR, §62"


def test_citation_label_empty_path():
    result = build_citation_label("FDA SaMD", [])
    assert result == "FDA SaMD"


def test_citation_label_three_elements():
    result = build_citation_label("EU MDR", ["Annex I", "17", "17.1"])
    assert result == "EU MDR, Annex I 17 §17.1"


def test_citation_label_imdrf():
    result = build_citation_label("IMDRF SaMD", ["Section 4", "4.2"])
    assert result == "IMDRF SaMD, Section 4 §4.2"
