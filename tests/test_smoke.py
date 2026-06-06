"""Smoke tests for the regulumen package scaffold (Step 1 / TDD)."""

# (a) Basic import and version
def test_version():
    import regulumen  # noqa: PLC0415

    assert regulumen.__version__ == "0.1.0"


# (b) Settings class instantiates with documented defaults
def test_settings_defaults(monkeypatch):
    from regulumen.core.config import Settings  # noqa: PLC0415

    # Hermetic: ignore any developer .env / shell env so we test the *declared*
    # defaults, not whatever happens to be configured on this machine.
    for var in (
        "REGULUMEN_TOP_K",
        "REGULUMEN_HYBRID_ALPHA",
        "REGULUMEN_ABSTENTION_THRESHOLD",
        "REGULUMEN_EMBEDDING_MODEL",
        "REGULUMEN_LLM_MODEL",
        "REGULUMEN_EMBEDDING_PROVIDER",
        "REGULUMEN_CHROMA_PATH",
        "REGULUMEN_GROQ_API_KEY",
        "REGULUMEN_HF_API_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)

    s = Settings(_env_file=None)
    assert s.top_k == 8
    assert s.hybrid_alpha == 0.3
    assert s.abstention_threshold == 0.7
    assert s.embedding_model == "BAAI/bge-large-en-v1.5"
    assert s.llm_model == "groq/llama-3.3-70b-versatile"
    assert s.embedding_provider == "huggingface"
    assert s.chroma_path == "./.chroma"
    assert s.groq_api_key is None
    assert s.hf_api_token is None


# (c) get_settings() returns a Settings instance
def test_get_settings():
    from regulumen.core.config import Settings, get_settings  # noqa: PLC0415

    result = get_settings()
    assert isinstance(result, Settings)
