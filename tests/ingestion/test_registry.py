"""Tests for the parser registry."""

from __future__ import annotations

import pytest

from regulumen.core.ingestion.models import SourceDoc
from regulumen.core.ingestion.parsers.html_eurlex import EurlexHtmlParser
from regulumen.core.ingestion.parsers.pdf_guidance import PdfGuidanceParser
from regulumen.core.ingestion.registry import get_parser


def _source(source_type: str) -> SourceDoc:
    return SourceDoc(
        id="x",
        title="t",
        short_title="s",
        source_type=source_type,  # type: ignore[arg-type]
        jurisdiction="EU",
        url="https://example.org",
        license_note="Public.",
        retrieved_at="2026-06-01",
    )


def test_get_parser_returns_eurlex_for_html() -> None:
    parser = get_parser(_source("eurlex_html"))
    assert isinstance(parser, EurlexHtmlParser)


def test_get_parser_returns_pdf_for_guidance_pdf() -> None:
    parser = get_parser(_source("guidance_pdf"))
    assert isinstance(parser, PdfGuidanceParser)


def test_get_parser_raises_for_unknown_type() -> None:
    # Bypass Literal validation to simulate an unregistered source_type.
    bad = SourceDoc.model_construct(
        id="x",
        title="t",
        short_title="s",
        source_type="unknown_type",
        jurisdiction="EU",
        url="https://example.org",
        license_note="Public.",
        retrieved_at="2026-06-01",
    )
    with pytest.raises(ValueError, match="No parser"):
        get_parser(bad)
