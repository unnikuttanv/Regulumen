"""Dump the real chunk inventory for the ReguLumen gold-set authoring step.

Loads every successfully-fetched source from ``corpus/manifest.json``, parses
it with its registered parser, chunks the resulting clauses, and writes one
line per chunk to ``eval/gold/_chunk_inventory.jsonl`` recording the
``citation_label``, ``doc_id`` and a short text preview.

The whole pipeline (parse + chunk) is offline: no network, no embeddings. The
inventory is the grounding reference for ``eval/gold/gold_qa.jsonl`` — every
gold ``expected_citation`` must match a real ``citation_label`` listed here.

Usage::

    python scripts/dump_chunks.py
"""

from __future__ import annotations

import json
from pathlib import Path

from regulumen.core.ingestion.chunker import chunk_clauses
from regulumen.core.ingestion.manifest import load_source_docs
from regulumen.core.ingestion.parsers.pdf_guidance import PdfGuidanceParser
from regulumen.core.ingestion.registry import get_parser

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MANIFEST_PATH = ROOT / "corpus" / "manifest.json"
RAW_DIR = ROOT / "corpus" / "raw"
OUT_PATH = ROOT / "eval" / "gold" / "_chunk_inventory.jsonl"

_PREVIEW_CHARS = 200


def _local_filename(source_id: str) -> str:
    """Return the ``local_filename`` recorded for ``source_id`` in the manifest."""
    entries = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for entry in entries:
        if entry.get("id") == source_id:
            return entry["local_filename"]
    raise KeyError(f"No manifest entry for source id {source_id!r}")


def main() -> None:
    sources = load_source_docs(MANIFEST_PATH)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    per_source: dict[str, int] = {}
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for source in sources:
            raw_path = RAW_DIR / _local_filename(source.id)
            parser = get_parser(source)

            # PDFs expose a confidence we want to thread into chunk metadata.
            if isinstance(parser, PdfGuidanceParser):
                clauses, confidence = parser.parse_with_confidence(source, raw_path)
            else:
                clauses = parser.parse(source, raw_path)
                confidence = None

            chunks = chunk_clauses(
                clauses,
                source.short_title,
                structure_confidence=confidence,
            )

            for chunk in chunks:
                preview = " ".join(chunk.text.split())[:_PREVIEW_CHARS]
                out.write(
                    json.dumps(
                        {
                            "citation_label": chunk.citation_label,
                            "doc_id": chunk.doc_id,
                            "text_preview": preview,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            per_source[source.id] = len(chunks)
            total += len(chunks)

    print(f"Wrote {total} chunks to {OUT_PATH}")
    for source_id, count in per_source.items():
        print(f"  {source_id}: {count} chunks")


if __name__ == "__main__":
    main()
