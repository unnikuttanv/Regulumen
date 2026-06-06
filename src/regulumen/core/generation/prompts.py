"""Prompt templates for grounded, citation-faithful generation.

Citation format
---------------
The model MUST wrap every citation in DOUBLE SQUARE BRACKETS around the exact
``citation_label`` of the excerpt it relied on, e.g. ``[[EU MDR, Annex I §17.2]]``.
This format is chosen because it is:

* unambiguous and rare in regulatory prose (so it never collides with normal
  text), and
* trivially machine-extractable with a single regex
  (:func:`regulumen.core.generation.faithfulness.extract_citations`),

which lets the post-generation gate verify each citation against the actually
retrieved labels.

Abstention
----------
If the excerpts do not contain the answer, the model must respond with the
literal token ``__ABSTAIN__`` followed by a short statement of WHAT is missing.
The token is detected by
:func:`regulumen.core.generation.faithfulness.is_abstention`.
"""

from __future__ import annotations

from regulumen.core.generation.faithfulness import ABSTENTION_TOKEN
from regulumen.core.retrieval.vector_store import RetrievedChunk

SYSTEM_PROMPT = f"""\
You are ReguLumen, a regulatory-compliance assistant for medical devices.

STRICT GROUNDING RULES — follow them exactly:
1. Answer the user's question ONLY using the numbered EXCERPTS provided in the
   user message. Do NOT use any outside or prior knowledge.
2. For every claim you make, cite the exact citation label of the excerpt you
   relied on by wrapping it in double square brackets, e.g.
   [[EU MDR, Annex I §17.2]]. Use the citation labels verbatim. Cite at least
   one excerpt for any substantive answer.
3. If the excerpts do NOT contain enough information to answer, do NOT guess.
   Respond with the literal token {ABSTENTION_TOKEN} followed by one sentence
   stating specifically what information is missing.
4. Never invent citation labels. Only cite labels that appear in the excerpts.

DISCLAIMER: ReguLumen provides decision support, not regulatory advice. Your
output must be verified by a qualified person before any compliance decision.
"""


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Format the question plus numbered, labeled excerpts for the model.

    Each excerpt is numbered and tagged with its ``citation_label`` and full
    text, and the prompt restates the required ``[[citation_label]]`` citation
    format so the answer is machine-verifiable.
    """
    lines: list[str] = []
    lines.append(f"QUESTION:\n{question}\n")

    if chunks:
        lines.append("EXCERPTS (cite these by their exact label):")
        for i, chunk in enumerate(chunks, start=1):
            lines.append(f"\n[{i}] Citation label: {chunk.citation_label}")
            lines.append(chunk.text)
    else:
        lines.append("EXCERPTS:\n(none provided)")

    lines.append(
        "\nINSTRUCTIONS: Answer using ONLY the excerpts above. Cite each excerpt "
        "you rely on with its exact label in double square brackets, e.g. "
        f"[[{chunks[0].citation_label if chunks else 'EU MDR, Annex I §17.2'}]]. "
        f"If the excerpts do not contain the answer, reply with {ABSTENTION_TOKEN} "
        "and state what is missing."
    )
    return "\n".join(lines)
