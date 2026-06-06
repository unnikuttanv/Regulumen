"""Load :class:`SourceDoc` objects from a corpus provenance manifest.

The manifest (``corpus/manifest.json``) is a JSON array of fetch records.
Only entries that were downloaded successfully (``success == True``) are
turned into ``SourceDoc`` objects; failed fetches are ignored.
"""

from __future__ import annotations

import json
from pathlib import Path

from regulumen.core.ingestion.models import SourceDoc

# Fields on a manifest entry that map directly onto SourceDoc.
_SOURCE_DOC_FIELDS = (
    "id",
    "title",
    "short_title",
    "source_type",
    "jurisdiction",
    "url",
    "license_note",
    "retrieved_at",
)


def load_source_docs(
    manifest_path: Path = Path("corpus/manifest.json"),
) -> list[SourceDoc]:
    """Return the successfully-fetched entries from ``manifest_path`` as SourceDocs."""
    entries = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    docs: list[SourceDoc] = []
    for entry in entries:
        if not entry.get("success", False):
            continue
        docs.append(SourceDoc(**{k: entry[k] for k in _SOURCE_DOC_FIELDS}))
    return docs
