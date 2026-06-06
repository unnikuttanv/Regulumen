"""Corpus ingestion CLI: build the searchable Chroma index from the corpus.

Pipeline (per source in ``corpus/manifest.json``):

    raw file -> parser -> clauses -> chunker -> chunks -> embedder -> Chroma

The module is split into two seams so the heavy lifting is testable OFFLINE:

* :func:`build_chunks` is a PURE parse+chunk function — no network, no
  embedding. Unit tests run it against the REAL corpus.
* :func:`ingest` builds chunks then indexes them into a Chroma store. The
  embedder is injectable, so tests pass a ``FakeEmbedder`` and index into a
  temp directory with no HF token and no network.

The :func:`main` CLI uses the real (hosted HuggingFace) embedder. When
``REGULUMEN_HF_API_TOKEN`` is missing it prints a clear message and exits
non-zero (2) instead of crashing with a stack trace.

Run it either way::

    python -m scripts.ingest
    python scripts/ingest.py --use-fake-embedder
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from regulumen.core.config import Settings, get_settings
from regulumen.core.ingestion.corpus import (
    DEFAULT_CORPUS_RAW,
    DEFAULT_MANIFEST,
    build_chunks,
)
from regulumen.core.retrieval.embedder import (
    Embedder,
    EmbeddingError,
    build_embedder,
)
from regulumen.core.retrieval.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

# build_chunks + the corpus-path defaults now live in
# regulumen.core.ingestion.corpus so the API/UI repository can share them
# without importing from scripts/. Re-exported here for the CLI + its tests.
DEFAULT_COLLECTION = "regulumen"


def ingest(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    manifest_path: Path = DEFAULT_MANIFEST,
    corpus_raw_dir: Path = DEFAULT_CORPUS_RAW,
    chroma_path: str | None = None,
    collection_name: str = DEFAULT_COLLECTION,
) -> int:
    """Build chunks from the corpus and index them into a Chroma store.

    Args:
        settings: Application settings.
        embedder: Embedder to use; pass a ``FakeEmbedder`` for offline tests.
            Defaults to the real hosted embedder via ``build_embedder``.
        manifest_path: Path to the corpus manifest.
        corpus_raw_dir: Directory holding the raw source files.
        chroma_path: Override for the Chroma persistence path; defaults to
            ``settings.chroma_path``.
        collection_name: Chroma collection name.

    Returns:
        The number of chunks indexed (``vector_store.count()``).
    """
    chunks = build_chunks(settings, manifest_path, corpus_raw_dir)
    if not chunks:
        logger.warning("No chunks were built from the corpus; nothing to index.")
        return 0

    if embedder is None:
        embedder = build_embedder(settings)

    store = ChromaVectorStore(
        embedder,
        path=chroma_path or settings.chroma_path,
        collection_name=collection_name,
    )
    logger.info("Indexing %d chunks into collection %r ...", len(chunks), collection_name)
    store.index(chunks)
    return store.count()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ingest",
        description="Build the ReguLumen Chroma index from the corpus.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to corpus/manifest.json.",
    )
    parser.add_argument(
        "--corpus-raw",
        type=Path,
        default=DEFAULT_CORPUS_RAW,
        help="Directory holding the raw source files.",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--chroma-path",
        default=None,
        help="Override the Chroma persistence path (defaults to settings.chroma_path).",
    )
    parser.add_argument(
        "--use-fake-embedder",
        action="store_true",
        help="Use the deterministic offline FakeEmbedder (no network/token). "
        "Handy for CI/demo smoke ingests.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g. DEBUG, INFO, WARNING).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    settings = get_settings()

    embedder: Embedder | None = None
    if args.use_fake_embedder:
        from regulumen.core.retrieval.embedder import FakeEmbedder

        logger.info("Using FakeEmbedder (offline, no network).")
        embedder = FakeEmbedder(dim=64)

    try:
        count = ingest(
            settings,
            embedder=embedder,
            manifest_path=args.manifest,
            corpus_raw_dir=args.corpus_raw,
            chroma_path=args.chroma_path,
            collection_name=args.collection,
        )
    except EmbeddingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "Set REGULUMEN_HF_API_TOKEN to build the real index, "
            "or pass --use-fake-embedder for an offline smoke ingest.",
            file=sys.stderr,
        )
        return 2

    print()
    print("Ingest complete.")
    print(f"  collection : {args.collection}")
    print(f"  chroma_path: {args.chroma_path or settings.chroma_path}")
    print(f"  indexed    : {count} chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
