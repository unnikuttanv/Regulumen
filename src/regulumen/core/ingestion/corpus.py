"""Corpus → chunks pipeline (parse + clause-aware chunk, offline, no embedding).

This lives in the core library so BOTH the ingest CLI (`scripts/ingest.py`) and
the API/UI retrieval repository can build the same chunk list from the manifest
without library code importing from ``scripts/``. It is pure (no network, no
embedding); embedding/indexing happens in the callers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from regulumen.core.ingestion.chunker import chunk_clauses
from regulumen.core.ingestion.manifest import load_source_docs
from regulumen.core.ingestion.models import Chunk
from regulumen.core.ingestion.registry import get_parser

logger = logging.getLogger(__name__)

# Repo-root-relative defaults (this file is src/regulumen/core/ingestion/).
_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MANIFEST = _REPO_ROOT / "corpus" / "manifest.json"
DEFAULT_CORPUS_RAW = _REPO_ROOT / "corpus" / "raw"


def local_filenames(manifest_path: Path) -> dict[str, str]:
    """Return a ``{source_id: local_filename}`` map read from the manifest JSON.

    ``load_source_docs`` intentionally drops provenance fields like
    ``local_filename`` from the typed ``SourceDoc``; we read them back here so we
    can resolve raw files on disk.
    """
    entries = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return {
        entry["id"]: entry.get("local_filename", entry["id"])
        for entry in entries
        if "id" in entry
    }


def build_chunks(
    settings,
    manifest_path: Path = DEFAULT_MANIFEST,
    corpus_raw_dir: Path = DEFAULT_CORPUS_RAW,
) -> list[Chunk]:
    """Parse + chunk every source in the manifest into a flat list of chunks.

    PURE: no embedding and no network. Missing raw files or unparseable sources
    are logged and skipped so one bad source never aborts the whole pipeline.

    Args:
        settings: Application settings (reserved for future chunk tuning).
        manifest_path: Path to ``corpus/manifest.json``.
        corpus_raw_dir: Directory holding the raw source files.

    Returns:
        All chunks across all parseable sources, in source/document order.
    """
    manifest_path = Path(manifest_path)
    corpus_raw_dir = Path(corpus_raw_dir)

    sources = load_source_docs(manifest_path)
    all_chunks: list[Chunk] = []
    filenames = local_filenames(manifest_path)

    for source in sources:
        raw_path = corpus_raw_dir / filenames.get(source.id, source.id)

        if not raw_path.exists():
            logger.warning(
                "Skipping source id=%s: raw file not found at %s", source.id, raw_path
            )
            continue

        try:
            parser = get_parser(source)
        except ValueError as exc:
            logger.warning("Skipping source id=%s: %s", source.id, exc)
            continue

        structure_confidence: float | None = None
        try:
            if hasattr(parser, "parse_with_confidence"):
                clauses, structure_confidence = parser.parse_with_confidence(
                    source, raw_path
                )
            else:
                clauses = parser.parse(source, raw_path)
        except Exception:  # noqa: BLE001 — one bad file must not abort the build
            logger.exception("Failed to parse source id=%s at %s", source.id, raw_path)
            continue

        # Drop degenerate parses (e.g. a PDF that extracted as a transcript with
        # confidence 0.0). A confidence of None means a well-structured source
        # (e.g. EU MDR HTML) and is always kept. The threshold comes from
        # settings.min_structure_confidence when present.
        # A threshold of 0.0 disables filtering (keep everything); otherwise a
        # source is dropped when its confidence is below the threshold.
        min_conf = getattr(settings, "min_structure_confidence", 0.0)
        if (
            min_conf > 0.0
            and structure_confidence is not None
            and structure_confidence < min_conf
        ):
            logger.warning(
                "Skipping source id=%s: structure_confidence %.3f < %.3f "
                "(degenerate parse; its chunks would pollute retrieval)",
                source.id,
                structure_confidence,
                min_conf,
            )
            continue

        chunks = chunk_clauses(
            clauses,
            source.short_title,
            structure_confidence=structure_confidence,
        )
        logger.info(
            "Source id=%s short_title=%r: %d clauses -> %d chunks (confidence=%s)",
            source.id,
            source.short_title,
            len(clauses),
            len(chunks),
            f"{structure_confidence:.3f}" if structure_confidence is not None else "n/a",
        )
        all_chunks.extend(chunks)

    return all_chunks


__all__ = ["DEFAULT_CORPUS_RAW", "DEFAULT_MANIFEST", "build_chunks", "local_filenames"]
