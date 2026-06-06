"""Gradio wiring for the ReguLumen demo (Approach A: imports core in-process).

This module is import-safe: importing it does NOT construct the service or need
provider keys. The service is built lazily in :func:`create_app`. No ``.launch()``
runs at import time.

Rendering logic lives in :mod:`regulumen.ui.format` (pure, unit-tested); this
module only wires the Gradio Blocks and handles errors gracefully.
"""

from __future__ import annotations

import gradio as gr

from regulumen.api.service import RegulumenService
from regulumen.ui.format import (
    format_answer_markdown,
    format_citations_markdown,
    format_risk_markdown,
)

_HEADER = (
    "# ReguLumen — Citation-Faithful Regulatory RAG\n\n"
    "> ⚠️ **Decision support, not regulatory advice.** "
    "Always verify against the primary sources and consult a qualified "
    "regulatory professional.\n"
)

_KEY_HELP = (
    "⚠️ **Backend unavailable.** The model/embedding provider could not be "
    "reached. Set the required credentials and restart:\n\n"
    "- `REGULUMEN_GROQ_API_KEY` (LLM)\n"
    "- `REGULUMEN_HF_API_TOKEN` (embeddings)\n\n"
    "_Details: {detail}_"
)


def _error_markdown(exc: Exception) -> str:
    """Render a friendly error block instead of letting the UI throw."""
    return _KEY_HELP.format(detail=f"{type(exc).__name__}: {exc}")


def handle_ask(service: RegulumenService, question: str) -> tuple[str, str]:
    """Handle an Ask submission. Returns (answer_md, citations_md)."""
    if not question or not question.strip():
        return ("_Enter a question to begin._", "_No citations._")
    try:
        resp = service.ask(question)
    except Exception as exc:  # noqa: BLE001 - UI must never throw
        return (_error_markdown(exc), "_No citations._")
    return (
        format_answer_markdown(resp),
        format_citations_markdown(resp.citations),
    )


def handle_classify(service: RegulumenService, description: str) -> str:
    """Handle a Risk Classification submission. Returns risk_md."""
    if not description or not description.strip():
        return "_Enter a component description to begin._"
    try:
        resp = service.classify(description)
    except Exception as exc:  # noqa: BLE001 - UI must never throw
        return _error_markdown(exc)
    return format_risk_markdown(resp)


def build_ui(service: RegulumenService) -> gr.Blocks:
    """Construct the Gradio Blocks app wired to ``service``. Does not launch."""
    with gr.Blocks(title="ReguLumen") as demo:
        gr.Markdown(_HEADER)

        with gr.Tabs():
            with gr.Tab("Ask"):
                question = gr.Textbox(
                    label="Regulatory question",
                    placeholder="e.g. What does the standard require for X?",
                    lines=3,
                )
                ask_btn = gr.Button("Ask", variant="primary")
                answer_md = gr.Markdown(label="Answer")
                with gr.Accordion("Citations", open=True):
                    citations_md = gr.Markdown()

                ask_btn.click(
                    fn=lambda q: handle_ask(service, q),
                    inputs=question,
                    outputs=[answer_md, citations_md],
                )

            with gr.Tab("Risk Classification"):
                description = gr.Textbox(
                    label="Software component description (IEC 62304)",
                    placeholder="Describe the software component to classify.",
                    lines=3,
                )
                classify_btn = gr.Button("Classify", variant="primary")
                risk_md = gr.Markdown(label="Classification")

                classify_btn.click(
                    fn=lambda d: handle_classify(service, d),
                    inputs=description,
                    outputs=risk_md,
                )

    return demo


def create_app() -> gr.Blocks:
    """Build the service lazily (needs keys) and wire the UI."""
    from regulumen.core.config import get_settings  # noqa: PLC0415

    service = RegulumenService.build_from_settings(get_settings())
    return build_ui(service)


def main() -> None:
    """Entry point for ``python -m regulumen.ui.app``."""
    create_app().launch()


if __name__ == "__main__":
    main()
