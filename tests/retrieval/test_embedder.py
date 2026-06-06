"""Tests for the embeddings layer.

All default tests use :class:`FakeEmbedder` and hit NO network and need NO
HF token. A single ``@pytest.mark.live`` test exercises the real
:class:`HuggingFaceEmbedder` and is skipped unless ``REGULUMEN_HF_API_TOKEN``
is set in the environment.
"""

from __future__ import annotations

import math
import os

import pytest

from regulumen.core.retrieval.embedder import (
    Embedder,
    EmbeddingError,
    FakeEmbedder,
    HuggingFaceEmbedder,
    build_embedder,
)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def test_fake_embedder_satisfies_protocol() -> None:
    emb = FakeEmbedder()
    assert isinstance(emb, Embedder)


def test_fake_embedder_dim_default_and_custom() -> None:
    assert FakeEmbedder().dim == 64
    assert FakeEmbedder(dim=128).dim == 128


def test_fake_embedder_vector_length_matches_dim() -> None:
    emb = FakeEmbedder(dim=32)
    [vec] = emb.embed_documents(["software safety classification"])
    assert len(vec) == 32
    assert len(emb.embed_query("anything")) == 32


def test_fake_embedder_is_deterministic() -> None:
    emb = FakeEmbedder()
    text = "Software safety classification class C"
    v1 = emb.embed_documents([text])[0]
    v2 = emb.embed_documents([text])[0]
    assert v1 == v2
    # Query embedding of the same text is also stable across calls.
    assert emb.embed_query(text) == emb.embed_query(text)


def test_fake_embedder_is_l2_normalized() -> None:
    emb = FakeEmbedder()
    for text in ["risk management file", "post-market surveillance plan"]:
        vec = emb.embed_documents([text])[0]
        norm = math.sqrt(sum(x * x for x in vec))
        assert math.isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-6)


def test_fake_embedder_batch_returns_one_vector_per_text() -> None:
    emb = FakeEmbedder()
    texts = ["alpha", "beta", "gamma"]
    vecs = emb.embed_documents(texts)
    assert len(vecs) == 3
    assert all(len(v) == emb.dim for v in vecs)


def test_fake_embedder_semantic_ranking() -> None:
    """A query about software safety should be closer to a passage containing
    those words than to an unrelated passage."""
    emb = FakeEmbedder()
    query = "software safety classification class"
    relevant = "Software safety classification assigns each item to class A, B or C."
    unrelated = "Quarterly financial revenue projections for the marketing department."

    q = emb.embed_query(query)
    r = emb.embed_documents([relevant])[0]
    u = emb.embed_documents([unrelated])[0]

    assert _cosine(q, r) > _cosine(q, u)


def test_fake_embedder_empty_batch() -> None:
    assert FakeEmbedder().embed_documents([]) == []


def test_build_embedder_requires_token() -> None:
    class _S:
        embedding_provider = "huggingface"
        embedding_model = "BAAI/bge-large-en-v1.5"
        hf_api_token = None

    with pytest.raises(EmbeddingError, match="REGULUMEN_HF_API_TOKEN"):
        build_embedder(_S())


def test_build_embedder_unknown_provider() -> None:
    class _S:
        embedding_provider = "mystery"
        embedding_model = "x"
        hf_api_token = "tok"

    with pytest.raises(EmbeddingError):
        build_embedder(_S())


def test_build_embedder_returns_hf_when_token_present() -> None:
    class _S:
        embedding_provider = "huggingface"
        embedding_model = "BAAI/bge-large-en-v1.5"
        hf_api_token = "fake-token"

    emb = build_embedder(_S())
    assert isinstance(emb, HuggingFaceEmbedder)
    assert emb.dim == 1024  # known dim for bge-large-en-v1.5


def test_hf_embedder_no_network_in_init() -> None:
    # Constructing must not raise / must not call the network.
    emb = HuggingFaceEmbedder(model="BAAI/bge-large-en-v1.5", api_token="tok")
    assert emb.dim == 1024


@pytest.mark.live
@pytest.mark.skipif(
    not os.getenv("REGULUMEN_HF_API_TOKEN"),
    reason="requires REGULUMEN_HF_API_TOKEN and network",
)
def test_hf_embedder_live_roundtrip() -> None:  # pragma: no cover - network gated
    token = os.environ["REGULUMEN_HF_API_TOKEN"]
    with HuggingFaceEmbedder(model="BAAI/bge-large-en-v1.5", api_token=token) as emb:
        vecs = emb.embed_documents(["Software safety classification class C."])
        assert len(vecs) == 1
        assert len(vecs[0]) == emb.dim
        q = emb.embed_query("software safety class")
        assert len(q) == emb.dim


def test_fake_embedder_rejects_empty_text() -> None:
    emb = FakeEmbedder()
    with pytest.raises(ValueError, match="empty"):
        emb.embed_documents([""])
    with pytest.raises(ValueError, match="empty"):
        emb.embed_query("   ")
