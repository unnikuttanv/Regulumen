"""Registry that selects the right :class:`Parser` for a SourceDoc.

Add new parsers (e.g. the guidance-PDF parser in step 3b) by appending an
instance to :data:`_PARSERS`.
"""

from __future__ import annotations

from regulumen.core.ingestion.models import SourceDoc
from regulumen.core.ingestion.parsers.base import Parser
from regulumen.core.ingestion.parsers.html_eurlex import EurlexHtmlParser
from regulumen.core.ingestion.parsers.pdf_guidance import PdfGuidanceParser

# Registered parser instances, tried in order.
_PARSERS: list[Parser] = [
    EurlexHtmlParser(),
    PdfGuidanceParser(),
]


def get_parser(source: SourceDoc) -> Parser:
    """Return the first registered parser that can handle ``source``.

    Raises:
        ValueError: if no registered parser claims ``source``.
    """
    for parser in _PARSERS:
        if parser.can_parse(source):
            return parser
    raise ValueError(
        f"No parser registered for source_type={source.source_type!r} "
        f"(source id={source.id!r})"
    )
