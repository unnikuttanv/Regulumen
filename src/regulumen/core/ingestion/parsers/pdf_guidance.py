"""Parser for guidance PDFs (FDA / IMDRF) using heuristic heading detection.

PDF text extraction is inherently messy, so this parser is built around a
few pure, testable helpers (:func:`detect_heading`, :func:`segment_pages`)
that operate on already-extracted page text.  ``PdfGuidanceParser`` only adds
the pypdf extraction layer and doc-id glue on top.

Discovered structure of the real corpus PDFs (Step 0 inspection):

IMDRF N10 (clean):
    Numbered headings ``1.0 Introduction``, ``5.1 Software as a Medical
    Device``, ``5.2.2 In Vitro Diagnostic ...`` on their own lines.  Every
    page repeats the header ``IMDRF/SaMD WG/N10FINAL:2013`` plus an underscore
    rule, and a footer ``9 December 2013 Page 3 of 9``.

FDA SaMD clinical eval (mixed):
    Top-level sections use Roman numerals (``I. Introduction``) which we do
    NOT treat as structural headings here; dotted-decimal sub-points are
    detected where present.  Every page repeats ``pg. N ... www.fda.gov``.

FDA premarket SW (degenerate):
    Extraction yields mostly non-breaking-space filler with scattered
    sentences; almost no detectable headings -> the per-page fallback keeps
    all content while reporting low confidence.

Boilerplate stripped (documented):
  * Repeating page headers/footers -- any line that recurs on a majority of
    pages (detected dynamically), e.g. the IMDRF doc header / FDA footer.
  * ``Contains Nonbinding Recommendations`` (FDA boilerplate banner).
  * ``Page N of M`` and bare page-number-only lines.
  * Underscore / dash rule lines and empty lines.

Heading heuristic (documented):
  A line is a heading iff it starts with a dotted-decimal number (``5``,
  ``5.1``, ``5.1.2``, optionally ``5.0`` / trailing ``.``) followed by a
  short Title-like phrase that is NOT a full sentence.  Sentence-like lines
  (too long, or ending in terminal punctuation) are rejected so numbered
  lists and footnote citations do not masquerade as sections.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

from regulumen.core.ingestion.models import Clause, SourceDoc

logger = logging.getLogger(__name__)

# A dotted-decimal heading: "5", "5.0", "5.1", "5.1.2", optional trailing dot,
# then the title text.  The number must be followed by whitespace + a title.
_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(\S.*)$")

# "Page 3 of 9" footer fragment.
_PAGE_OF = re.compile(r"\bpage\s+\d+\s+of\s+\d+\b", re.IGNORECASE)

# A line consisting solely of a (small) page number.
_PAGE_NUMBER_ONLY = re.compile(r"^\d{1,4}$")

# Rule lines made of underscores / dashes / equals.
_RULE_LINE = re.compile(r"^[_\-=\s]+$")

# Common static FDA/IMDRF boilerplate banners.
_BOILERPLATE_SUBSTRINGS = (
    "contains nonbinding recommendations",
    "www.fda.gov",
)

# Heading-title sanity bounds: a real heading title is short and not a full
# sentence.  Tunables hoisted for clarity.
_MAX_HEADING_TITLE_LEN = 80
_MAX_HEADING_TITLE_WORDS = 12

# Lowercase function words ignored when checking a title is "title-cased".
_TITLE_STOPWORDS = frozenset(
    {
        "a", "an", "the", "of", "for", "and", "or", "to", "in", "on", "as",
        "at", "by", "with", "from", "such", "that", "they", "be", "may", "is",
        "are", "via", "per",
    }
)

# A line recurring on at least this fraction of pages is a running header/footer.
_RUNNING_LINE_PAGE_FRACTION = 0.5

# Below this share of text-under-headings we treat the parse as unstructured
# and fall back to one clause per page.
_FALLBACK_COVERAGE = 0.25


def _clean(text: str) -> str:
    """Collapse non-breaking spaces and whitespace runs."""
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _is_boilerplate_line(line: str) -> bool:
    """Return ``True`` for footers/banners/rules safe to drop on any page."""
    stripped = line.strip()
    if not stripped:
        return True
    if _RULE_LINE.match(stripped):
        return True
    if _PAGE_NUMBER_ONLY.match(stripped):
        return True
    if _PAGE_OF.search(stripped):
        return True
    low = stripped.lower()
    return any(sub in low for sub in _BOILERPLATE_SUBSTRINGS)


def detect_heading(line: str) -> tuple[list[str], str, str] | None:
    """Detect a dotted-decimal section heading in ``line``.

    Returns ``(path, number, title)`` where ``path`` is the full dotted
    hierarchy including the leaf (``"5.1.2"`` -> ``["5", "5.1", "5.1.2"]``)
    and ``number`` is the leaf identifier; or ``None`` if ``line`` is not a
    heading.

    The leading ``.0`` of top-level sections (e.g. ``"1.0"``) is preserved in
    ``number`` so the leaf matches the source text, but a ``"1.0"`` heading
    yields the two-level path ``["1", "1.0"]``.
    """
    stripped = _clean(line)
    match = _HEADING.match(stripped)
    if not match:
        return None

    number, title = match.group(1), match.group(2).strip()
    if not _looks_like_heading_title(title):
        return None

    return _build_path(number), number, title


def _looks_like_heading_title(title: str) -> bool:
    """Reject sentence-like titles so list items / footnotes are not headings.

    A genuine section title is short, does not end in terminal punctuation,
    and is "title-cased": every *significant* word (ignoring small function
    words) starts with a capital.  This rejects footnote bodies and numbered
    list sentences such as ``"The specific types"`` while keeping real
    headings like ``"Introduction"`` or ``"Key Definitions"``.
    """
    if not title:
        return False
    if len(title) > _MAX_HEADING_TITLE_LEN:
        return False
    if len(title.split()) > _MAX_HEADING_TITLE_WORDS:
        return False
    # Headings do not end in terminal sentence punctuation (rejects sentence-like lines).
    if title[-1] in ".:;,?!":
        return False
    # The first title word should look like a word, not a citation/URL token.
    if not title[0].isalpha():
        return False
    return _is_title_cased(title)


def _is_title_cased(title: str) -> bool:
    """Return ``True`` if every significant word in ``title`` is capitalised."""
    words = re.findall(r"[A-Za-z][A-Za-z'./()-]*", title)
    significant = [w for w in words if w.lower() not in _TITLE_STOPWORDS]
    if not significant:
        return False
    return all(w[0].isupper() for w in significant)


def _build_path(number: str) -> list[str]:
    """Build the full dotted path for a leaf number.

    ``"5.1.2"`` -> ``["5", "5.1", "5.1.2"]``.  A trailing ``.0`` is kept as a
    distinct leaf: ``"1.0"`` -> ``["1", "1.0"]``.
    """
    parts = number.split(".")
    return [".".join(parts[: i + 1]) for i in range(len(parts))]


def segment_pages(pages: list[str]) -> tuple[list[Clause], float]:
    """Segment extracted page texts into clauses + a structure-confidence score.

    Detects section headings, accumulates the body text under each heading
    (spanning page breaks), strips repeating headers/footers and boilerplate,
    and joins hyphen-broken line wraps.  ``page`` on each clause is the
    1-based page where its heading was found.

    If too little text falls under detected headings, falls back to one clause
    per page (path ``["p.<n>"]``, number ``""``) so no content is lost, and
    reports low confidence.

    Returns ``(clauses, confidence)`` with ``confidence`` in ``0.0..1.0``.
    """
    running = _running_lines(pages)

    sections: list[_Section] = []
    current: _Section | None = None
    heading_chars = 0
    body_chars = 0

    for page_no, page_text in enumerate(pages, start=1):
        for raw_line in page_text.splitlines():
            line = _clean(raw_line)
            if not line or line in running or _is_boilerplate_line(line):
                continue

            heading = detect_heading(line)
            if heading is not None:
                path, number, title = heading
                current = _Section(path=path, number=number, title=title, page=page_no)
                sections.append(current)
                heading_chars += len(line)
            elif current is not None:
                current.add(line)
                body_chars += len(line)
            else:
                # Preamble before the first heading: counts against coverage.
                body_chars += len(line)

    total = heading_chars + body_chars
    coverage = 0.0 if total == 0 else min(1.0, (heading_chars + _covered_body(sections)) / total)

    if not sections or coverage < _FALLBACK_COVERAGE:
        clauses = _fallback_per_page(pages, running)
        confidence = round(min(0.4, coverage), 3)
        logger.info(
            "PDF parsed via per-page fallback: %d page-clauses, confidence=%.3f",
            len(clauses),
            confidence,
        )
        return clauses, confidence

    clauses = [s.to_clause() for s in sections if s.text]
    confidence = round(coverage, 3)
    logger.info(
        "PDF parsed via headings: %d clauses, confidence=%.3f",
        len(clauses),
        confidence,
    )
    return clauses, confidence


def _covered_body(sections: list[_Section]) -> int:
    """Total body characters that landed under a detected heading."""
    return sum(len(s.text) for s in sections)


def _running_lines(pages: list[str]) -> set[str]:
    """Identify lines that recur on a majority of pages (headers/footers).

    ``Page N of M`` footers vary per page, so they are caught separately by
    :func:`_is_boilerplate_line`; this catches the static running header/footer
    text (e.g. the IMDRF doc identifier) that repeats verbatim.
    """
    if len(pages) < 2:
        return set()
    counts: Counter[str] = Counter()
    for page_text in pages:
        seen = {
            _clean(line)
            for line in page_text.splitlines()
            if _clean(line) and not _is_boilerplate_line(line)
        }
        counts.update(seen)
    threshold = max(2, int(len(pages) * _RUNNING_LINE_PAGE_FRACTION))
    return {
        line
        for line, n in counts.items()
        if n >= threshold and detect_heading(line) is None
    }


def _fallback_per_page(pages: list[str], running: set[str]) -> list[Clause]:
    """One clause per page when no usable heading structure is found."""
    clauses: list[Clause] = []
    for page_no, page_text in enumerate(pages, start=1):
        kept = [
            line
            for raw_line in page_text.splitlines()
            if (line := _clean(raw_line))
            and line not in running
            and not _is_boilerplate_line(line)
        ]
        text = _join_lines(kept)
        if not text:
            continue
        clauses.append(
            Clause(
                doc_id="",
                path=[f"p.{page_no}"],
                number="",
                title=None,
                text=text,
                page=page_no,
                anchor=None,
            )
        )
    return clauses


def _join_lines(lines: list[str]) -> str:
    """Join body lines, healing hyphen-broken word wraps, into one string."""
    out = ""
    for line in lines:
        if not line:
            continue
        if not out:
            out = line
        elif out.endswith("-") and not out.endswith(("--",)):
            out = out[:-1] + line
        else:
            out = f"{out} {line}"
    return _clean(out)


class _Section:
    """Mutable accumulator for one heading and its body lines."""

    def __init__(self, path: list[str], number: str, title: str, page: int) -> None:
        self.path = path
        self.number = number
        self.title = title
        self.page = page
        self._lines: list[str] = []

    def add(self, line: str) -> None:
        self._lines.append(line)

    @property
    def text(self) -> str:
        body = _join_lines(self._lines)
        # A heading with no body still deserves a citable clause; use the title.
        return body or self.title

    def to_clause(self) -> Clause:
        return Clause(
            doc_id="",
            path=self.path,
            number=self.number,
            title=self.title,
            text=self.text,
            page=self.page,
            anchor=None,
        )


class PdfGuidanceParser:
    """Parse FDA / IMDRF guidance PDFs into :class:`Clause` objects."""

    def can_parse(self, source: SourceDoc) -> bool:
        return source.source_type == "guidance_pdf"

    def parse(self, source: SourceDoc, raw_path: Path) -> list[Clause]:
        clauses, _confidence = self.parse_with_confidence(source, raw_path)
        return clauses

    def parse_with_confidence(
        self, source: SourceDoc, raw_path: Path
    ) -> tuple[list[Clause], float]:
        """Parse and also return the heuristic structure-confidence score.

        Step 3c (the chunker) will surface ``confidence`` into chunk metadata;
        for now it is computed, logged, and returned alongside the clauses.
        """
        pages = self._extract_pages(raw_path)
        clauses, confidence = segment_pages(pages)
        clauses = self._finalize(clauses, doc_id=source.id)
        logger.info(
            "Parsed guidance PDF id=%s: %d clauses, structure_confidence=%.3f",
            source.id,
            len(clauses),
            confidence,
        )
        return clauses, confidence

    @staticmethod
    def _extract_pages(raw_path: Path) -> list[str]:
        reader = PdfReader(str(raw_path))
        return [page.extract_text() or "" for page in reader.pages]

    @staticmethod
    def _finalize(clauses: list[Clause], doc_id: str) -> list[Clause]:
        """Stamp the parent ``doc_id`` onto every clause (segment layer is
        doc-agnostic so it stays purely testable)."""
        for clause in clauses:
            clause.doc_id = doc_id
        return clauses
