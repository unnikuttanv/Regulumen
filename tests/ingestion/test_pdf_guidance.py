"""Tests for the guidance-PDF parser against SYNTHETIC extracted text.

The fixtures here are IP-safe placeholder text ("Lorem ipsum ...") that
mimics the *structure* of FDA / IMDRF guidance as discovered during PDF
inspection (numbered ``N.0`` / ``N.M`` headings, repeating page headers and
footers, page-number-only lines, hyphen-broken line wraps).  No text is
copied from the real documents.

The heading-detection + segmentation core is exercised through pure
functions (``detect_heading``, ``segment_pages``) so the tests stay fast and
deterministic without shipping a binary PDF fixture.
"""

from __future__ import annotations

from regulumen.core.ingestion.models import SourceDoc
from regulumen.core.ingestion.parsers.pdf_guidance import (
    PdfGuidanceParser,
    detect_heading,
    segment_pages,
)


def _source(source_type: str = "guidance_pdf") -> SourceDoc:
    return SourceDoc(
        id="imdrf-samd-key-definitions",
        title="IMDRF SaMD Key Definitions",
        short_title="IMDRF SaMD N10",
        source_type=source_type,  # type: ignore[arg-type]
        jurisdiction="IMDRF",
        url="https://example.org/doc.pdf",
        license_note="Public.",
        retrieved_at="2026-06-01",
    )


# --------------------------------------------------------------------- #
# detect_heading
# --------------------------------------------------------------------- #
def test_detect_simple_numbered_heading() -> None:
    result = detect_heading("1.0 Introduction")
    assert result is not None
    path, number, title = result
    assert path == ["1", "1.0"]
    assert number == "1.0"
    assert title == "Introduction"


def test_detect_multi_level_heading() -> None:
    result = detect_heading("5.1.2 Lorem Ipsum Definition")
    assert result is not None
    path, number, title = result
    assert path == ["5", "5.1", "5.1.2"]
    assert number == "5.1.2"
    assert title == "Lorem Ipsum Definition"


def test_detect_single_token_section() -> None:
    result = detect_heading("3 Scope")
    assert result is not None
    path, number, title = result
    assert path == ["3"]
    assert number == "3"
    assert title == "Scope"


def test_body_sentence_is_not_a_heading() -> None:
    # A full sentence beginning with a number but running long / ending in a
    # period must NOT be treated as a heading (avoids footnote/list noise).
    assert detect_heading(
        "1. Lorem ipsum dolor sit amet consectetur adipiscing elit sed do."
    ) is None


def test_plain_prose_is_not_a_heading() -> None:
    assert detect_heading("Lorem ipsum dolor sit amet.") is None


def test_blank_line_is_not_a_heading() -> None:
    assert detect_heading("   ") is None


# --------------------------------------------------------------------- #
# segment_pages
# --------------------------------------------------------------------- #
def test_segment_basic_two_headings() -> None:
    pages = [
        "1.0 Introduction\nLorem ipsum requirement one.\n"
        "2.0 Scope\nLorem ipsum scope text.",
    ]
    clauses, confidence = segment_pages(pages)
    assert [c.number for c in clauses] == ["1.0", "2.0"]
    intro = clauses[0]
    assert intro.title == "Introduction"
    assert intro.path == ["1", "1.0"]
    assert intro.page == 1
    assert intro.anchor is None
    assert "Lorem ipsum requirement one." in intro.text
    # Heading line itself is not duplicated into the body.
    assert "1.0 Introduction" not in intro.text
    assert confidence > 0.5


def test_segment_body_spans_page_break() -> None:
    pages = [
        "1.0 Introduction\nLorem ipsum first part of the",
        "sentence continues on the next page.\n2.0 Scope\nMore text.",
    ]
    clauses, _ = segment_pages(pages)
    intro = clauses[0]
    assert intro.page == 1  # heading page, not the continuation page
    assert "first part of the sentence continues on the next page." in intro.text


def test_segment_strips_repeating_header_and_footer() -> None:
    header = "GUIDANCE/PLACEHOLDER/N99:2099"
    pages = [
        f"{header}\n1.0 Introduction\nLorem ipsum one.\n9 Month 2099 Page 1 of 2",
        f"{header}\n2.0 Scope\nLorem ipsum two.\n9 Month 2099 Page 2 of 2",
    ]
    clauses, _ = segment_pages(pages)
    full_text = " ".join(c.text for c in clauses)
    assert header not in full_text
    assert "Page 1 of 2" not in full_text
    assert "Page 2 of 2" not in full_text
    assert "Lorem ipsum one." in clauses[0].text


def test_segment_strips_boilerplate_and_page_number_lines() -> None:
    pages = [
        "Contains Nonbinding Recommendations\n"
        "1.0 Introduction\nLorem ipsum body.\n4",
    ]
    clauses, _ = segment_pages(pages)
    body = clauses[0].text
    assert "Contains Nonbinding Recommendations" not in body
    # A bare page-number-only line ("4") must be dropped.
    assert body.strip().endswith("body.")


def test_segment_joins_hyphen_broken_wraps() -> None:
    pages = ["1.0 Introduction\nLorem ip-\nsum requirement text."]
    clauses, _ = segment_pages(pages)
    assert "Lorem ipsum requirement text." in clauses[0].text


def test_segment_no_headings_falls_back_to_one_clause_per_page() -> None:
    pages = [
        "Lorem ipsum dolor sit amet, just prose with no headings here.",
        "Consectetur adipiscing elit, still no headings on this page.",
    ]
    clauses, confidence = segment_pages(pages)
    assert len(clauses) == 2
    assert clauses[0].path == ["p.1"]
    assert clauses[0].number == ""
    assert clauses[0].page == 1
    assert clauses[1].path == ["p.2"]
    assert clauses[1].page == 2
    assert confidence < 0.5  # fallback => low confidence


def test_segment_preamble_before_first_heading_does_not_crash() -> None:
    pages = [
        "Some cover preamble text.\n1.0 Introduction\nLorem ipsum body.",
    ]
    clauses, _ = segment_pages(pages)
    # Only the heading-anchored clause is emitted; preamble is dropped.
    assert [c.number for c in clauses] == ["1.0"]


# --------------------------------------------------------------------- #
# Parser protocol surface
# --------------------------------------------------------------------- #
def test_can_parse_only_guidance_pdf() -> None:
    parser = PdfGuidanceParser()
    assert parser.can_parse(_source("guidance_pdf")) is True
    assert parser.can_parse(_source("eurlex_html")) is False


def test_confidence_bounded_to_one_when_heading_only() -> None:
    _, confidence = segment_pages(["1.0 Introduction"])
    assert 0.0 <= confidence <= 1.0


def test_single_digit_with_trailing_dot_accepted() -> None:
    # "1. Title" is treated as single-level heading "1"; intentional per the
    # module docstring (trailing-dot support). Asserted so it's clearly
    # load-bearing behaviour, not an oversight.
    result = detect_heading("1. Introduction")
    assert result is not None
    _, number, _ = result
    assert number == "1"


def test_clauses_carry_doc_id(tmp_path) -> None:
    # parse_with_confidence assigns doc_id from the source; verify via the
    # pure layer + a tiny synthetic PDF written with pypdf is overkill, so we
    # check the doc_id propagation indirectly through segment + parser glue.
    parser = PdfGuidanceParser()
    clauses = parser._finalize(  # noqa: SLF001 - exercising internal glue
        segment_pages(["1.0 Introduction\nLorem ipsum body."])[0],
        doc_id="imdrf-samd-key-definitions",
    )
    assert all(c.doc_id == "imdrf-samd-key-definitions" for c in clauses)
    assert clauses[0].number == "1.0"
