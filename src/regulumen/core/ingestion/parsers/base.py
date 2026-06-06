"""Parser protocol shared by all source-type parsers.

Every concrete parser (EUR-Lex HTML, guidance PDF, ...) implements this
structural protocol so the registry can select one by ``can_parse`` and
invoke ``parse`` uniformly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from regulumen.core.ingestion.models import Clause, SourceDoc


@runtime_checkable
class Parser(Protocol):
    """A source parser turning a raw file into a list of :class:`Clause`."""

    def can_parse(self, source: SourceDoc) -> bool:
        """Return ``True`` if this parser handles ``source``'s source_type."""
        ...

    def parse(self, source: SourceDoc, raw_path: Path) -> list[Clause]:
        """Parse the file at ``raw_path`` into ordered :class:`Clause` objects."""
        ...
