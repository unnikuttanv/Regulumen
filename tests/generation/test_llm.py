"""Tests for the thin LiteLLM wrapper and its factory. All offline.

The real network path is exercised only by an opt-in ``@pytest.mark.live`` test
that is skipped unless ``REGULUMEN_GROQ_API_KEY`` is set.
"""

from __future__ import annotations

import os

import pytest

from regulumen.core.generation.llm import (
    FakeLLM,
    LLMClient,
    build_llm,
)


def test_fake_llm_returns_scripted_string():
    llm = FakeLLM("a canned answer")
    assert llm.complete("sys", "user") == "a canned answer"


def test_fake_llm_accepts_callable():
    llm = FakeLLM(lambda system, user: f"{system}|{user}")
    assert llm.complete("S", "U") == "S|U"


def test_fake_llm_counts_calls():
    llm = FakeLLM("x")
    assert llm.calls == 0
    llm.complete("s", "u")
    llm.complete("s", "u")
    assert llm.calls == 2


def test_build_llm_raises_without_key_for_groq_model():
    class _S:
        llm_model = "groq/llama-3.3-70b-versatile"
        groq_api_key = None

    with pytest.raises(Exception) as exc:  # noqa: PT011 - message is asserted below
        build_llm(_S())
    assert "REGULUMEN_GROQ_API_KEY" in str(exc.value)


def test_build_llm_returns_client_when_key_present():
    class _S:
        llm_model = "groq/llama-3.3-70b-versatile"
        groq_api_key = "sk-test"

    client = build_llm(_S())
    assert isinstance(client, LLMClient)
    assert client.model == "groq/llama-3.3-70b-versatile"


def test_build_llm_allows_keyless_non_remote_model():
    """A model that does not require a key (e.g. a local/ollama style) is fine."""

    class _S:
        llm_model = "ollama/llama3"
        groq_api_key = None

    client = build_llm(_S())
    assert isinstance(client, LLMClient)


@pytest.mark.live
def test_live_groq_completion():
    key = os.environ.get("REGULUMEN_GROQ_API_KEY")
    if not key:
        pytest.skip("REGULUMEN_GROQ_API_KEY not set; live test skipped")
    client = LLMClient(model="groq/llama-3.3-70b-versatile", api_key=key)
    out = client.complete("You are terse.", "Reply with the single word: OK")
    assert isinstance(out, str)
    assert out.strip() != ""
