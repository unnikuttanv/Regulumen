"""Pure, testable Markdown rendering helpers for the Gradio demo UI.

These functions take API DTOs (``AskResponse`` / ``ClassifyResponse`` /
``CitationOut``) and return Markdown strings. They import no Gradio and need no
service, so they can be unit-tested fully offline.

UX contract (the headline requirement):
- Abstentions render DISTINCTLY from a normal answer (a clear ``⚠️`` block that
  cannot be mistaken for a confident answer).
- Citations are the visual focus (bold labels) and rendered prominently.
- The disclaimer is ALWAYS appended in a visually distinct block on every
  answer/risk render path.
"""

from __future__ import annotations

from regulumen.api.schemas import AskResponse, CitationOut, ClassifyResponse

# Marker used for all abstention/insufficient-information blocks. Tests assert on
# this so answers and abstentions never look alike.
_ABSTAIN_MARKER = "⚠️"


def _disclaimer_block(text: str) -> str:
    """Render the disclaimer in a consistent, visually distinct block."""
    return f"\n\n---\n\n> {_ABSTAIN_MARKER} **Disclaimer:** {text}\n"


def format_answer_markdown(resp: AskResponse) -> str:
    """Render an :class:`AskResponse` as Markdown.

    Abstentions get a clearly-marked ``⚠️`` block so they do not read as a
    confident answer; answered responses render the answer text. The disclaimer
    is always appended.
    """
    if resp.abstained or not resp.answered:
        reason = resp.abstention_reason or "No grounded answer could be produced."
        body = (
            f"## {_ABSTAIN_MARKER} No grounded answer — abstained\n\n"
            f"The system declined to answer because it could not ground a "
            f"response in the corpus.\n\n"
            f"**Reason:** {reason}"
        )
    else:
        body = f"## Answer\n\n{resp.answer}"

    return body + _disclaimer_block(resp.disclaimer)


def format_citations_markdown(citations: list[CitationOut]) -> str:
    """Render a list of :class:`CitationOut` as Markdown, labels as the focus."""
    if not citations:
        return "_No citations._"

    lines = ["### Citations\n"]
    for cite in citations:
        lines.append(
            f"- **{cite.citation_label}** "
            f"(score: {cite.score:.3f})\n\n"
            f"  > {cite.snippet}"
        )
    return "\n".join(lines)


def format_risk_markdown(resp: ClassifyResponse) -> str:
    """Render a :class:`ClassifyResponse` as Markdown.

    Abstentions get a clear ``⚠️`` block; classifications show the IEC 62304
    safety class prominently. The disclaimer is always appended.
    """
    if resp.abstained or resp.safety_class is None:
        body = (
            f"## {_ABSTAIN_MARKER} Abstained — insufficient information\n\n"
            f"The system could not classify this component with confidence.\n\n"
            f"**Rationale:** {resp.rationale}"
        )
        return body + _disclaimer_block(resp.disclaimer)

    severity = resp.severity if resp.severity is not None else "unspecified"
    can_contribute = (
        "unspecified"
        if resp.can_contribute is None
        else ("yes" if resp.can_contribute else "no")
    )
    parts = [
        f"# Safety Class {resp.safety_class}",
        "",
        f"- **Severity:** {severity}",
        f"- **Can contribute to a hazardous situation:** {can_contribute}",
        "",
        f"**Rationale:** {resp.rationale}",
    ]
    if resp.citations:
        parts.append("")
        parts.append(format_citations_markdown(resp.citations))

    return "\n".join(parts) + _disclaimer_block(resp.disclaimer)


__all__ = [
    "format_answer_markdown",
    "format_citations_markdown",
    "format_risk_markdown",
]
