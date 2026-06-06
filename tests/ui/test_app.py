"""Light offline tests for the Gradio wiring (no launch, no network)."""

from __future__ import annotations

import importlib

import gradio as gr

from regulumen.api.schemas import AskResponse, ClassifyResponse
from regulumen.core.generation.llm import LLMError


class _FakeService:
    """Stand-in for RegulumenService returning canned DTOs."""

    def ask(self, question: str) -> AskResponse:
        return AskResponse(
            answered=True,
            answer="canned answer",
            abstained=False,
            abstention_reason=None,
            citations=[],
            disclaimer="Decision support, not regulatory advice.",
        )

    def classify(self, description: str) -> ClassifyResponse:
        return ClassifyResponse(
            safety_class="B",
            abstained=False,
            severity="non_serious_injury",
            can_contribute=True,
            rationale="canned rationale",
            citations=[],
            disclaimer="Decision support, not regulatory advice.",
        )

    def health(self):  # pragma: no cover - not exercised here
        return None


class _BrokenService:
    """Service whose calls raise to test graceful error handling."""

    def ask(self, question: str) -> AskResponse:
        raise LLMError("provider unavailable")

    def classify(self, description: str) -> ClassifyResponse:
        raise LLMError("provider unavailable")

    def health(self):  # pragma: no cover
        return None


def test_build_ui_returns_blocks_without_launching() -> None:
    from regulumen.ui.app import build_ui

    demo = build_ui(_FakeService())
    assert isinstance(demo, gr.Blocks)


def test_app_import_is_safe_without_keys(monkeypatch) -> None:
    # Unset provider keys to prove import does not need them / build a service.
    for var in ("REGULUMEN_GROQ_API_KEY", "REGULUMEN_HF_API_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    module = importlib.import_module("regulumen.ui.app")
    importlib.reload(module)  # re-exercise top-level import safely
    assert hasattr(module, "create_app")
    assert hasattr(module, "build_ui")


def test_handle_ask_returns_formatted_markdown() -> None:
    from regulumen.ui.app import handle_ask

    answer_md, citations_md = handle_ask(_FakeService(), "a question")
    assert "canned answer" in answer_md
    assert "Decision support" in answer_md
    assert citations_md == "_No citations._"


def test_handle_classify_returns_formatted_markdown() -> None:
    from regulumen.ui.app import handle_classify

    out = handle_classify(_FakeService(), "a component")
    assert "# Safety Class B" in out
    assert "canned rationale" in out


def test_handle_ask_renders_friendly_error_not_exception() -> None:
    from regulumen.ui.app import handle_ask

    answer_md, citations_md = handle_ask(_BrokenService(), "a question")
    assert "REGULUMEN_GROQ_API_KEY" in answer_md
    assert "REGULUMEN_HF_API_TOKEN" in answer_md
    assert "⚠️" in answer_md


def test_handle_classify_renders_friendly_error_not_exception() -> None:
    from regulumen.ui.app import handle_classify

    out = handle_classify(_BrokenService(), "a component")
    assert "REGULUMEN_GROQ_API_KEY" in out
