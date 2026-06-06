"""Tests for the EUR-Lex HTML parser against a synthetic fixture.

The fixture mimics the real EUR-Lex (CONVEX/ELI) markup discovered during
inspection of the EU MDR HTML, using IP-safe placeholder text only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from regulumen.core.ingestion.models import Clause, SourceDoc
from regulumen.core.ingestion.parsers.html_eurlex import EurlexHtmlParser

FIXTURE = Path(__file__).parent / "fixtures" / "eurlex_sample.html"


def _source(source_type: str = "eurlex_html") -> SourceDoc:
    return SourceDoc(
        id="eu-mdr-2017-745",
        title="EU MDR — Regulation (EU) 2017/745",
        short_title="EU MDR",
        source_type=source_type,  # type: ignore[arg-type]
        jurisdiction="EU",
        url="https://eur-lex.europa.eu/eli/reg/2017/745",
        license_note="Public.",
        retrieved_at="2026-06-01",
    )


@pytest.fixture
def clauses() -> list[Clause]:
    return EurlexHtmlParser().parse(_source(), FIXTURE)


def _by_path(clauses: list[Clause], path: list[str]) -> Clause:
    matches = [c for c in clauses if c.path == path]
    assert matches, f"no clause with path {path}; got {[c.path for c in clauses]}"
    assert len(matches) == 1, f"duplicate clauses for path {path}"
    return matches[0]


def test_can_parse_only_eurlex_html() -> None:
    parser = EurlexHtmlParser()
    assert parser.can_parse(_source("eurlex_html")) is True
    assert parser.can_parse(_source("guidance_pdf")) is False


def test_article_points_have_expected_path_and_number(clauses: list[Clause]) -> None:
    p1 = _by_path(clauses, ["Chapter I", "Article 10", "10.1"])
    assert p1.number == "10.1"
    assert p1.text.strip()
    assert "10.1" not in p1.text[:4]  # leading enumerator stripped, text is the body

    p2 = _by_path(clauses, ["Chapter I", "Article 10", "10.2"])
    assert p2.number == "10.2"
    assert p2.text.strip()


def test_article_heading_clause_captured(clauses: list[Clause]) -> None:
    art = _by_path(clauses, ["Chapter I", "Article 10"])
    assert art.number == "10"
    assert art.title == "Placeholder obligations heading"


def test_annex_numbered_clause(clauses: list[Clause]) -> None:
    c = _by_path(clauses, ["Annex I", "Chapter II", "17", "17.1"])
    assert c.number == "17.1"
    assert c.text.strip()


def test_annex_section_heading(clauses: list[Clause]) -> None:
    c = _by_path(clauses, ["Annex I", "Chapter II", "17"])
    assert c.number == "17"
    assert c.text.strip()


def test_all_clauses_have_nonempty_text_and_doc_id(clauses: list[Clause]) -> None:
    assert clauses, "parser returned no clauses"
    for c in clauses:
        assert c.doc_id == "eu-mdr-2017-745"
        assert c.text.strip(), f"empty text for {c.path}"
        assert c.page is None


def test_anchors_captured_from_ids(clauses: list[Clause]) -> None:
    p1 = _by_path(clauses, ["Chapter I", "Article 10", "10.1"])
    assert p1.anchor == "010.001"
    annex = _by_path(clauses, ["Annex I", "Chapter II", "17", "17.1"])
    assert annex.anchor == "anx-17-1"


def test_recital_boilerplate_skipped(clauses: list[Clause]) -> None:
    for c in clauses:
        assert "recital boilerplate" not in c.text.lower()
