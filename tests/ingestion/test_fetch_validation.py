"""Unit tests for the fetch_corpus.is_valid_body validation helper."""

from __future__ import annotations

from fetch_corpus import MIN_BYTES, is_valid_body

# ---------------------------------------------------------------------------
# MIN_BYTES constant is accessible
# ---------------------------------------------------------------------------


def test_min_bytes_constant_is_ten_thousand():
    assert MIN_BYTES == 10_000


# ---------------------------------------------------------------------------
# guidance_pdf sources
# ---------------------------------------------------------------------------


def test_valid_pdf_body():
    """A body starting with %PDF and over MIN_BYTES passes."""
    content = b"%PDF-1.4 " + b"x" * (MIN_BYTES + 100)
    ok, reason = is_valid_body("guidance_pdf", content)
    assert ok is True
    assert reason == ""


def test_pdf_body_too_small():
    """A body under MIN_BYTES is rejected even if magic bytes are correct."""
    content = b"%PDF-1.4 " + b"x" * 100
    ok, reason = is_valid_body("guidance_pdf", content)
    assert ok is False
    assert "too small" in reason


def test_pdf_missing_magic_bytes():
    """A large body that doesn't start with %PDF is rejected."""
    content = b"<html><body>" + b"x" * (MIN_BYTES + 200)
    ok, reason = is_valid_body("guidance_pdf", content)
    assert ok is False
    assert "PDF magic" in reason


# ---------------------------------------------------------------------------
# eurlex_html sources
# ---------------------------------------------------------------------------


def test_valid_html_body():
    """A body containing <html in the first 5000 bytes and over MIN_BYTES passes."""
    preamble = b"<!DOCTYPE html>\n<html lang='en'><head></head><body>"
    content = preamble + b"x" * (MIN_BYTES + 500)
    ok, reason = is_valid_body("eurlex_html", content)
    assert ok is True
    assert reason == ""


def test_html_uppercase_tag():
    """<HTML (upper-case) in the first 5000 bytes should also pass."""
    preamble = b"<!DOCTYPE HTML>\n<HTML lang='en'>"
    content = preamble + b"x" * (MIN_BYTES + 500)
    ok, reason = is_valid_body("eurlex_html", content)
    assert ok is True


def test_html_body_too_small():
    """An HTML body under MIN_BYTES is rejected."""
    content = b"<html><body>tiny</body></html>"
    ok, reason = is_valid_body("eurlex_html", content)
    assert ok is False
    assert "too small" in reason


def test_html_missing_tag():
    """A large body without any <html tag is rejected."""
    content = b"Not a valid HTML document. " + b"x" * (MIN_BYTES + 300)
    ok, reason = is_valid_body("eurlex_html", content)
    assert ok is False
    assert "<html>" in reason


# ---------------------------------------------------------------------------
# unknown source_type — only MIN_BYTES check applies
# ---------------------------------------------------------------------------


def test_unknown_source_type_large_enough():
    content = b"arbitrary content " + b"x" * MIN_BYTES
    ok, reason = is_valid_body("other_type", content)
    assert ok is True


def test_unknown_source_type_too_small():
    content = b"tiny"
    ok, reason = is_valid_body("other_type", content)
    assert ok is False
    assert "too small" in reason
