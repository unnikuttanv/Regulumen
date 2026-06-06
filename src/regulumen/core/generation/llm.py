"""Thin, provider-agnostic LLM gateway for ReguLumen.

All LLM calls go through :class:`LLMClient`, a minimal wrapper over LiteLLM so
the underlying provider (Groq by default) is swappable without touching the
generation logic. Determinism matters for faithfulness, so completions default
to ``temperature=0``.

For tests there is :class:`FakeLLM` with the SAME ``complete(system, user)``
interface, configured with either a canned string or a ``(system, user) -> str``
callable. No module imports the network at import time; LiteLLM is imported
lazily inside :meth:`LLMClient.complete`.
"""

from __future__ import annotations

from collections.abc import Callable

# Model-provider prefixes that REQUIRE a hosted API key. Anything else (e.g.
# "ollama/...", a bare local model) is treated as keyless so build_llm does not
# block local/self-hosted setups.
_KEY_REQUIRING_PREFIXES = ("groq/",)


class LLMError(RuntimeError):
    """Raised when an LLM completion fails (network, auth, provider error)."""


class LLMClient:
    """Minimal LiteLLM-backed completion client.

    Parameters
    ----------
    model:
        A LiteLLM model string, e.g. ``"groq/llama-3.3-70b-versatile"``.
    api_key:
        Provider API key (optional for keyless/local models).
    temperature:
        Sampling temperature; ``0.0`` for deterministic, faithfulness-friendly
        output.
    timeout:
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._temperature = temperature
        self._timeout = timeout

    def complete(self, system: str, user: str) -> str:
        """Run a chat completion and return the assistant text content.

        Imports LiteLLM lazily so importing this module never touches the
        network or heavy deps. Raises :class:`LLMError` on any failure.
        """
        try:
            import litellm  # noqa: PLC0415 - lazy import by design
        except Exception as exc:  # pragma: no cover - import-time env issue
            raise LLMError(f"Failed to import litellm: {exc}") from exc

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=self._temperature,
                api_key=self._api_key,
                timeout=self._timeout,
            )
        except Exception as exc:
            raise LLMError(f"LLM completion failed for model {self.model!r}: {exc}") from exc

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, KeyError) as exc:
            raise LLMError(f"Malformed LLM response from {self.model!r}: {exc}") from exc

        if content is None:
            raise LLMError(f"LLM returned empty content for model {self.model!r}")
        return str(content)


class FakeLLM:
    """Test double with the same surface as :class:`LLMClient`.

    Construct with either a canned response string or a
    ``(system, user) -> str`` callable to script grounded answers, hallucinated
    answers, or abstentions. Tracks how many times :meth:`complete` was called.
    """

    def __init__(self, response: str | Callable[[str, str], str]) -> None:
        self._response = response
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        if callable(self._response):
            return self._response(system, user)
        return self._response


def _model_requires_key(model: str) -> bool:
    return any(model.startswith(prefix) for prefix in _KEY_REQUIRING_PREFIXES)


def build_llm(settings) -> LLMClient:
    """Build an :class:`LLMClient` from a settings object.

    Raises a clear error when the configured model needs a hosted key but
    ``settings.groq_api_key`` is missing, pointing the user at the env var.
    """
    model = settings.llm_model
    api_key = settings.groq_api_key
    if _model_requires_key(model) and not api_key:
        raise LLMError(
            f"Model {model!r} requires an API key but none was provided. "
            "Set REGULUMEN_GROQ_API_KEY in your environment or .env file."
        )
    return LLMClient(model=model, api_key=api_key)
