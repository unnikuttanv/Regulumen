"""Tests for loading SourceDocs from a corpus manifest."""

from __future__ import annotations

import json
from pathlib import Path

from regulumen.core.ingestion.manifest import load_source_docs
from regulumen.core.ingestion.models import SourceDoc


def test_load_source_docs_parses_successful_entries(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "id": "eu-mdr-2017-745",
                    "title": "EU MDR",
                    "short_title": "EU MDR",
                    "source_type": "eurlex_html",
                    "jurisdiction": "EU",
                    "url": "https://eur-lex.europa.eu/eli/reg/2017/745",
                    "license_note": "Public.",
                    "retrieved_at": "2026-06-01",
                    "local_filename": "eu-mdr-2017-745.html",
                    "success": True,
                    "error": None,
                },
                {
                    "id": "failed-doc",
                    "title": "Failed",
                    "short_title": "Failed",
                    "source_type": "guidance_pdf",
                    "jurisdiction": "US-FDA",
                    "url": "https://example.org/missing.pdf",
                    "license_note": "Public.",
                    "retrieved_at": "2026-06-01",
                    "local_filename": "missing.pdf",
                    "success": False,
                    "error": "HTTP 404",
                },
            ]
        ),
        encoding="utf-8",
    )

    docs = load_source_docs(manifest)

    assert len(docs) == 1
    assert isinstance(docs[0], SourceDoc)
    assert docs[0].id == "eu-mdr-2017-745"
    assert docs[0].source_type == "eurlex_html"
