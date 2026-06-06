"""Parser for EUR-Lex HTML (CONVEX/ELI rendition) regulations.

Discovered structure of the EU MDR HTML (Regulation (EU) 2017/745):

Articles
    ``<div class="eli-subdivision" id="art_10">`` wraps one article and
    contains:
      * ``<p class="oj-ti-art">Article 10</p>``        -> article number
      * ``<div class="eli-title"><p class="oj-sti-art">...</p></div>`` -> title
      * ``<div id="010.001"><p class="oj-normal">1.&#160;&#160;...</p></div>``
        -> numbered points; the leading ``N.`` enumerator gives the point.
    Articles may be nested inside a chapter container
    ``<div id="cpt_I"><p class="oj-ti-section-1">CHAPTER I</p>...</div>``.

Annexes
    ``<div class="eli-container" id="anx_I">`` holds a *flat* list of
    sibling ``<p>`` elements:
      * ``<p class="oj-doc-ti">ANNEX I</p>``           -> annex label
      * ``<p class="oj-ti-grseq-1">CHAPTER II</p>``    -> chapter divider
      * ``<p class="oj-ti-grseq-1">17. heading</p>``   -> numbered heading
      * ``<p class="oj-normal">17.1. body...</p>``     -> numbered point
    Numbered points carry their full dotted number ("10", "10.4.1") as a
    leading enumerator in the text.

Simplifications (documented):
  * Recitals/citations/preamble (``rct_*``, ``cit_*``, ``pbl_*``) and the
    final/signatory parts are skipped -- only the normative Articles and
    Annexes are emitted.
  * Enumerated ``(a)/(b)`` sub-lists rendered as ``<table>`` are folded into
    their parent point's text rather than emitted as separate clauses.
"""

from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from regulumen.core.ingestion.models import Clause, SourceDoc

# Leading enumerator like "1.", "10.", "17.1.", "10.4.1." possibly followed by
# non-breaking spaces.  Captures the dotted number (without the trailing dot).
_ENUMERATOR = re.compile(r"^\s*(\d+(?:\.\d+)*)\.\s*")

# Annex chapter divider, e.g. "CHAPTER II".
_ANNEX_CHAPTER = re.compile(r"^CHAPTER\s+([IVXLCDM]+)$", re.IGNORECASE)


def _clean(text: str) -> str:
    """Collapse non-breaking spaces and whitespace runs."""
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _strip_enumerator(text: str) -> str:
    """Remove a leading "N." / "N.M." enumerator from a point's body text."""
    return _ENUMERATOR.sub("", text, count=1).strip()


def _normalize_divider(text: str) -> str:
    """Normalise a "CHAPTER V" / "SECTION 2" heading to title-case label.

    Title-cases the leading keyword while preserving the identifier
    (Roman numerals stay upper-case): "CHAPTER V" -> "Chapter V".
    """
    parts = text.split(None, 1)
    if len(parts) == 2 and parts[0].isalpha():
        return f"{parts[0].capitalize()} {parts[1]}"
    return text


class EurlexHtmlParser:
    """Parse EUR-Lex HTML regulations into :class:`Clause` objects."""

    def can_parse(self, source: SourceDoc) -> bool:
        return source.source_type == "eurlex_html"

    def parse(self, source: SourceDoc, raw_path: Path) -> list[Clause]:
        html = Path(raw_path).read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "lxml")

        clauses: list[Clause] = []
        clauses.extend(self._parse_articles(soup, source.id))
        clauses.extend(self._parse_annexes(soup, source.id))
        return clauses

    # ------------------------------------------------------------------ #
    # Articles
    # ------------------------------------------------------------------ #
    def _parse_articles(self, soup: BeautifulSoup, doc_id: str) -> list[Clause]:
        clauses: list[Clause] = []
        for art in soup.select('div.eli-subdivision[id^="art_"]'):
            clauses.extend(self._parse_one_article(art, doc_id))
        return clauses

    def _parse_one_article(self, art: Tag, doc_id: str) -> list[Clause]:
        num_el = art.find("p", class_="oj-ti-art")
        if num_el is None:
            return []
        number = self._article_number(num_el.get_text())
        if not number:
            return []

        ancestors = self._section_ancestors(art)
        base_path = [*ancestors, f"Article {number}"]

        title_el = art.select_one("div.eli-title .oj-sti-art")
        title = _clean(title_el.get_text()) if title_el else None

        clauses: list[Clause] = []
        # The article heading itself, so the article is always citable.
        clauses.append(
            Clause(
                doc_id=doc_id,
                path=base_path,
                number=number,
                title=title,
                text=title or f"Article {number}",
                anchor=art.get("id"),
            )
        )

        # Numbered points: direct child divs holding an oj-normal paragraph.
        for point in art.find_all("div", recursive=False):
            body = point.find("p", class_="oj-normal")
            if body is None:
                continue
            raw = body.get_text()
            match = _ENUMERATOR.match(raw.replace("\xa0", " "))
            if not match:
                continue
            point_no = match.group(1)
            leaf = f"{number}.{point_no}"
            text = _strip_enumerator(_clean(raw))
            if not text:
                continue
            clauses.append(
                Clause(
                    doc_id=doc_id,
                    path=[*base_path, leaf],
                    number=leaf,
                    title=None,
                    text=text,
                    anchor=point.get("id"),
                )
            )
        return clauses

    @staticmethod
    def _article_number(raw: str) -> str:
        cleaned = _clean(raw)
        m = re.match(r"Article\s+(\d+[a-z]?)", cleaned, re.IGNORECASE)
        return m.group(1) if m else ""

    @staticmethod
    def _section_ancestors(art: Tag) -> list[str]:
        """Return the chapter/section divider labels enclosing ``art``.

        EU MDR nests articles as ``cpt_V`` (Chapter) -> ``cpt_V.sct_2``
        (Section) -> ``art_52``.  Both levels carry an ``oj-ti-section-1``
        heading; we collect them outermost-first, e.g.
        ``["Chapter V", "Section 2"]``.
        """
        labels: list[str] = []
        for container in art.find_parents("div", id=re.compile(r"^cpt_")):
            heading = container.find("p", class_="oj-ti-section-1", recursive=False)
            if heading is not None:
                labels.append(_normalize_divider(_clean(heading.get_text())))
        labels.reverse()  # find_parents yields innermost-first
        return labels

    # ------------------------------------------------------------------ #
    # Annexes
    # ------------------------------------------------------------------ #
    def _parse_annexes(self, soup: BeautifulSoup, doc_id: str) -> list[Clause]:
        clauses: list[Clause] = []
        for annex in soup.select('div.eli-container[id^="anx_"]'):
            clauses.extend(self._parse_one_annex(annex, doc_id))
        return clauses

    def _parse_one_annex(self, annex: Tag, doc_id: str) -> list[Clause]:
        label_el = annex.find("p", class_="oj-doc-ti")
        if label_el is None:
            return []
        annex_label = self._annex_label(label_el.get_text())
        if not annex_label:
            return []

        clauses: list[Clause] = []
        current_chapter: str | None = None

        for p in annex.find_all("p", recursive=False):
            classes = p.get("class") or []
            text = _clean(p.get_text())
            if not text:
                continue

            # Chapter dividers reset the chapter context.
            chap = _ANNEX_CHAPTER.match(text)
            if "oj-ti-grseq-1" in classes and chap:
                current_chapter = f"Chapter {chap.group(1).upper()}"
                continue

            # Numbered headings / points carry a leading dotted enumerator.
            if {"oj-ti-grseq-1", "oj-normal"} & set(classes):
                match = _ENUMERATOR.match(text)
                if not match:
                    continue
                number = match.group(1)
                body = _strip_enumerator(text)
                if not body:
                    continue
                path = self._annex_path(annex_label, current_chapter, number)
                clauses.append(
                    Clause(
                        doc_id=doc_id,
                        path=path,
                        number=number,
                        title=None,
                        text=body,
                        anchor=p.get("id"),
                    )
                )
        return clauses

    @staticmethod
    def _annex_label(raw: str) -> str:
        cleaned = _clean(raw)
        m = re.match(r"ANNEX\s+([IVXLCDM]+)", cleaned, re.IGNORECASE)
        return f"Annex {m.group(1).upper()}" if m else ""

    @staticmethod
    def _annex_path(
        annex_label: str, chapter: str | None, number: str
    ) -> list[str]:
        """Build a full annex path including the leaf and its dotted parent.

        For a sub-numbered point like "17.1" the parent heading "17" is
        included as an ancestor, e.g.
        ``["Annex I", "Chapter II", "17", "17.1"]``.
        """
        path = [annex_label]
        if chapter:
            path.append(chapter)
        if "." in number:
            parent = number.rsplit(".", 1)[0]
            path.append(parent)
        path.append(number)
        return path
