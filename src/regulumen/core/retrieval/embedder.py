"""Embeddings layer for ReguLumen.

Embeddings live behind a small :class:`Embedder` protocol so the rest of the
system (vector store, hybrid retrieval, generation) never depends on a concrete
provider. Two implementations are shipped:

* :class:`HuggingFaceEmbedder` — calls the free *hosted* HuggingFace Inference
  feature-extraction API. Needs ``REGULUMEN_HF_API_TOKEN`` and network; used in
  real indexing/querying. Handles HF cold-start (503 "model loading").
* :class:`FakeEmbedder` — deterministic, offline, dependency-free. Used by the
  unit tests so they run with no network and no token.

The bge-large-en-v1.5 model expects an instruction prefix on *queries* (not on
passages); that prefix is applied only inside ``embed_query``.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from typing import Protocol, runtime_checkable

import httpx

# bge-large-en-v1.5 convention: queries are prefixed with this instruction so
# the model maps them into the same space as the (un-prefixed) passages.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Known embedding dimensions for models we support. Falls back to inferring the
# dimension from the first API response when a model is not listed here.
_KNOWN_DIMS: dict[str, int] = {
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}

# HF hosted feature-extraction (pipeline) endpoint. Chosen because it returns a
# single pooled vector per input for sentence-embedding models, which is exactly
# what we want for retrieval.
#
# NOTE: HuggingFace decommissioned the old ``api-inference.huggingface.co`` host
# and migrated the serverless Inference API to ``router.huggingface.co`` with the
# model in the path: .../hf-inference/models/<model>/pipeline/feature-extraction
# (verified 2026-06-03 against BAAI/bge-large-en-v1.5 → 1024-dim vectors).
_HF_BASE_URL = "https://router.huggingface.co/hf-inference/models"
_HF_PIPELINE_SUFFIX = "pipeline/feature-extraction"


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be produced (config or provider failure)."""


@runtime_checkable
class Embedder(Protocol):
    """Minimal embedding interface used across the retrieval/generation stack."""

    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages (documents). Returns one vector per input text."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""
        ...


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


class FakeEmbedder:
    """Deterministic, offline embedder for tests and local development.

    Scheme: each text is lower-cased and tokenized into words and characters.
    Every token is hashed to a stable bucket in ``[0, dim)`` and accumulated,
    then the resulting vector is L2-normalized. This guarantees:

    * identical text -> identical vector (determinism), and
    * texts sharing words/characters land in overlapping buckets, so similar
      text yields a higher cosine similarity than unrelated text.

    Queries and documents share the same projection (no instruction prefix is
    applied) so a query ranks the passage containing its words highly.
    """

    def __init__(self, dim: int = 64) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def _vectorize(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Cannot embed empty or whitespace-only text")
        vec = [0.0] * self.dim
        lowered = text.lower()
        words = re.findall(r"[a-z0-9]+", lowered)
        # Word tokens carry the dominant signal; char tokens add a fuzzy floor.
        tokens = list(words) + [f"#{c}" for c in lowered if not c.isspace()]
        for tok in tokens:
            h = hashlib.sha1(tok.encode("utf-8")).digest()
            bucket = int.from_bytes(h[:4], "big") % self.dim
            # Sign derived from another hash byte to spread mass over the space.
            sign = 1.0 if h[4] & 1 else -1.0
            weight = 3.0 if not tok.startswith("#") else 1.0
            vec[bucket] += sign * weight
        return _l2_normalize(vec)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectorize(text)


class HuggingFaceEmbedder:
    """Embedder backed by the hosted HuggingFace Inference feature-extraction API.

    Network is only touched in ``embed_documents`` / ``embed_query`` — never at
    import time or in ``__init__`` (beyond constructing the HTTP client).
    """

    # Retry policy for HF cold starts (503 "model loading").
    _MAX_RETRIES = 5
    _DEFAULT_BACKOFF = 2.0

    def __init__(self, model: str, api_token: str, *, timeout: float = 30.0) -> None:
        if not api_token:
            raise EmbeddingError(
                "HuggingFaceEmbedder requires an API token. "
                "Set REGULUMEN_HF_API_TOKEN."
            )
        self.model = model
        self._url = f"{_HF_BASE_URL}/{model}/{_HF_PIPELINE_SUFFIX}"
        self._timeout = timeout
        self._client = httpx.Client(
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_token}"},
        )
        # dim is known up front for supported models; otherwise inferred lazily
        # from the first response.
        self._dim: int | None = _KNOWN_DIMS.get(model)

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise EmbeddingError(
                f"Embedding dimension for model {self.model!r} is unknown until "
                "the first request has been made."
            )
        return self._dim

    def _post(self, inputs: list[str]) -> object:
        """POST to HF, retrying on 503 model-loading cold starts."""
        last_err: Exception | None = None
        for attempt in range(self._MAX_RETRIES):
            try:
                resp = self._client.post(
                    self._url,
                    json={"inputs": inputs, "options": {"wait_for_model": True}},
                )
            except httpx.HTTPError as exc:  # network/transport failure
                last_err = exc
                time.sleep(self._DEFAULT_BACKOFF * (attempt + 1))
                continue

            if resp.status_code == 503:
                # Model is cold-starting; HF returns estimated_time.
                wait = self._DEFAULT_BACKOFF * (attempt + 1)
                try:
                    body = resp.json()
                    if isinstance(body, dict) and "estimated_time" in body:
                        wait = float(body["estimated_time"])
                except (ValueError, TypeError):
                    pass
                last_err = EmbeddingError(f"HF model loading (503): {resp.text[:200]}")
                time.sleep(min(wait, 30.0))
                continue

            if resp.status_code != 200:
                raise EmbeddingError(
                    f"HF Inference API error {resp.status_code}: {resp.text[:300]}"
                )

            try:
                return resp.json()
            except ValueError as exc:  # malformed JSON
                raise EmbeddingError("HF Inference API returned invalid JSON") from exc

        raise EmbeddingError(
            f"HF Inference API failed after {self._MAX_RETRIES} attempts: {last_err}"
        )

    def _coerce_vectors(self, payload: object, n_inputs: int) -> list[list[float]]:
        """Normalize HF responses into ``list[list[float]]`` (one per input).

        Feature-extraction may return either a 2D list (one vector per input,
        pooled) or a 3D list (per-token vectors) which we mean-pool.
        """
        if not isinstance(payload, list) or not payload:
            raise EmbeddingError("Unexpected empty embedding response from HF")

        vectors: list[list[float]]
        first = payload[0]
        if isinstance(first, (int, float)):
            # Single flat vector (single input).
            vectors = [[float(x) for x in payload]]  # type: ignore[list-item]
        elif isinstance(first, list) and first and isinstance(first[0], (int, float)):
            # 2D: one pooled vector per input.
            vectors = [[float(x) for x in row] for row in payload]  # type: ignore[union-attr]
        elif isinstance(first, list) and first and isinstance(first[0], list):
            # 3D: per-token vectors per input -> mean-pool over tokens.
            vectors = [self._mean_pool(tokens) for tokens in payload]  # type: ignore[arg-type]
        else:
            raise EmbeddingError("Unrecognized embedding response shape from HF")

        if self._dim is None and vectors:
            self._dim = len(vectors[0])
        # Defensive: HF should return n_inputs vectors.
        if len(vectors) != n_inputs:
            raise EmbeddingError(
                f"HF returned {len(vectors)} vectors for {n_inputs} inputs"
            )
        return vectors

    @staticmethod
    def _mean_pool(token_vectors: list[list[float]]) -> list[float]:
        if not token_vectors:
            raise EmbeddingError("Cannot mean-pool empty token vectors")
        dim = len(token_vectors[0])
        acc = [0.0] * dim
        for tv in token_vectors:
            for i, x in enumerate(tv):
                acc[i] += float(x)
        n = len(token_vectors)
        return [x / n for x in acc]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = self._post(texts)
        return self._coerce_vectors(payload, len(texts))

    def embed_query(self, text: str) -> list[float]:
        prefixed = f"{BGE_QUERY_INSTRUCTION}{text}"
        payload = self._post([prefixed])
        return self._coerce_vectors(payload, 1)[0]

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        self._client.close()

    def __enter__(self) -> HuggingFaceEmbedder:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def build_embedder(settings) -> Embedder:
    """Construct the configured real embedder.

    Tests instantiate :class:`FakeEmbedder` directly; this factory is for
    production use and raises a clear :class:`EmbeddingError` when the provider
    is unknown or the HF token is missing.
    """
    provider = getattr(settings, "embedding_provider", None)
    if provider != "huggingface":
        raise EmbeddingError(
            f"Unsupported embedding_provider {provider!r}; only 'huggingface' "
            "is supported."
        )
    token = getattr(settings, "hf_api_token", None)
    if not token:
        raise EmbeddingError(
            "No HuggingFace API token configured. Set REGULUMEN_HF_API_TOKEN "
            "to use the hosted embeddings API."
        )
    return HuggingFaceEmbedder(model=settings.embedding_model, api_token=token)
