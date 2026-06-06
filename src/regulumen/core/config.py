"""Typed application settings via pydantic-settings (v2)."""

from __future__ import annotations

import functools

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All ReguLumen runtime configuration.

    Values are resolved from (highest → lowest priority):
    1. Environment variables with the ``REGULUMEN_`` prefix.
    2. A ``.env`` file in the working directory.
    3. The defaults declared below.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="REGULUMEN_",
        extra="ignore",
    )

    # --- LLM / generation ---
    llm_model: str = Field(default="groq/llama-3.3-70b-versatile")
    # Total generate cycles in the LangGraph answerer; 2 = one refine retry.
    max_answer_attempts: int = Field(default=2, ge=1)
    # Orchestrator for Q&A. "graph" uses the LangGraph answerer with a bounded
    # faithfulness refine loop (recovers mis-ranked clauses); "linear" uses the
    # single-pass Answerer. Both honor the same strict faithfulness gate and
    # return the same AnswerResult.
    answerer_mode: str = Field(default="graph")

    # --- Embeddings ---
    embedding_provider: str = Field(default="huggingface")
    embedding_model: str = Field(default="BAAI/bge-large-en-v1.5")

    # --- Vector store ---
    chroma_path: str = Field(default="./.chroma")

    # --- Retrieval tuning ---
    # Defaults tuned 2026-06-03 against the 24 retrieval-scored gold items
    # (see docs/eval-tuning-notes.md): alpha=0.3 gave the best MRR (0.784) at no
    # recall cost vs 0.5/0.7; k=8 matched k=10 on recall with higher precision.
    top_k: int = Field(default=8, ge=1)
    hybrid_alpha: float = Field(default=0.3, ge=0.0, le=1.0)  # 0=BM25 only, 1=dense only

    # Minimum per-document structure confidence for a source's chunks to be
    # indexed. Sources parsed with confidence at or below this are dropped at
    # ingest time: they are degenerate parses (e.g. the FDA Premarket PDF that
    # extracted as a webinar transcript, confidence 0.0) whose noisy chunks
    # pollute top-k and push the generator toward over-abstention. Sources with
    # confidence ``None`` (well-structured HTML like EU MDR) are always kept.
    min_structure_confidence: float = Field(default=0.15, ge=0.0, le=1.0)

    # --- Abstention ---
    abstention_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    # --- API keys (optional; sourced from env) ---
    groq_api_key: str | None = Field(default=None)
    hf_api_token: str | None = Field(default=None)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton :class:`Settings` instance."""
    return Settings()
