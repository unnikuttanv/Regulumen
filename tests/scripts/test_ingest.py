"""Offline tests for scripts/ingest.py.

These exercise the REAL corpus parse+chunk pipeline (no network) and the
indexing path with a FakeEmbedder into a temp Chroma directory (no HF token).
"""

from __future__ import annotations

import json
from pathlib import Path

# scripts/ is on sys.path via the root conftest.py.
import ingest as ingest_module  # noqa: E402
import pytest

from regulumen.core.config import Settings
from regulumen.core.ingestion.models import Chunk
from regulumen.core.retrieval.embedder import FakeEmbedder

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = _REPO_ROOT / "corpus" / "manifest.json"
_CORPUS_RAW = _REPO_ROOT / "corpus" / "raw"


@pytest.fixture
def settings() -> Settings:
    """Settings independent of any real .env on the machine."""
    return Settings(_env_file=None)


def test_build_chunks_real_corpus(settings: Settings) -> None:
    chunks = ingest_module.build_chunks(settings, _MANIFEST, _CORPUS_RAW)

    assert isinstance(chunks, list)
    assert chunks, "expected a non-empty list of chunks from the real corpus"
    assert all(isinstance(c, Chunk) for c in chunks)

    # Every chunk carries a citation label and non-empty text.
    for c in chunks:
        assert c.citation_label, f"chunk {c.id} missing citation_label"
        assert c.text.strip(), f"chunk {c.id} has empty text"

    # EU MDR and IMDRF must both be represented (by doc_id).
    doc_ids = {c.doc_id for c in chunks}
    assert "eu-mdr-2017-745" in doc_ids
    assert "imdrf-samd-key-definitions" in doc_ids


def test_low_confidence_source_is_filtered_out(settings: Settings) -> None:
    """The FDA Premarket transcript (structure_confidence 0.0) is dropped at the
    default threshold but kept when the threshold is 0 — guarding the filter that
    stops degenerate parses from polluting retrieval / over-abstaining the LLM.
    """
    default_ids = {
        c.doc_id for c in ingest_module.build_chunks(settings, _MANIFEST, _CORPUS_RAW)
    }
    assert "fda-premarket-software-functions" not in default_ids
    assert "eu-mdr-2017-745" in default_ids

    permissive = Settings(_env_file=None, min_structure_confidence=0.0)
    permissive_ids = {
        c.doc_id for c in ingest_module.build_chunks(permissive, _MANIFEST, _CORPUS_RAW)
    }
    assert "fda-premarket-software-functions" in permissive_ids


def test_ingest_with_fake_embedder(settings: Settings, tmp_path: Path) -> None:
    count = ingest_module.ingest(
        settings,
        embedder=FakeEmbedder(dim=64),
        manifest_path=_MANIFEST,
        corpus_raw_dir=_CORPUS_RAW,
        chroma_path=str(tmp_path / "chroma"),
        collection_name="test_ingest",
    )

    assert count > 0

    # Re-open the same store and confirm it persisted + is searchable offline.
    from regulumen.core.retrieval.vector_store import ChromaVectorStore

    store = ChromaVectorStore(
        FakeEmbedder(dim=64),
        path=str(tmp_path / "chroma"),
        collection_name="test_ingest",
    )
    assert store.count() == count

    hits = store.search("software safety", k=5)
    assert hits, "expected at least one search hit for a plausible query"
    assert all(h.citation_label for h in hits)


def test_ingest_missing_corpus_file_is_resilient(
    settings: Settings, tmp_path: Path
) -> None:
    """A manifest entry whose raw file is absent is skipped, not fatal."""
    # Build a manifest with one valid entry and one pointing at a missing file.
    real_entries = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    eu = next(e for e in real_entries if e["id"] == "eu-mdr-2017-745")
    bogus = dict(eu)
    bogus["id"] = "does-not-exist"
    bogus["local_filename"] = "nope-missing-file.pdf"

    tmp_manifest = tmp_path / "manifest.json"
    tmp_manifest.write_text(json.dumps([eu, bogus]), encoding="utf-8")

    # Should still produce chunks from the valid source without raising.
    chunks = ingest_module.build_chunks(settings, tmp_manifest, _CORPUS_RAW)
    assert chunks
    assert "eu-mdr-2017-745" in {c.doc_id for c in chunks}
    assert "does-not-exist" not in {c.doc_id for c in chunks}
