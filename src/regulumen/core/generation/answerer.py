"""Answerer orchestration: question -> grounded answer OR structured abstention.

Flow (the strict-faithfulness pipeline):

1. Retrieve top-k chunks. If retrieval is empty, ABSTAIN immediately WITHOUT
   calling the LLM (saves tokens — there is nothing to ground an answer in).
2. Build the grounded prompt and call the LLM.
3. If the model explicitly abstained, pass that abstention through.
4. Otherwise run the post-generation faithfulness GATE
   (:func:`~regulumen.core.generation.faithfulness.check_faithfulness`). If the
   answer is grounded, return it with the cited chunks. If NOT grounded (e.g. it
   cited a clause that was never retrieved), REPLACE it with a structured
   abstention that names what is missing — a hallucinated answer can never reach
   the user.
5. Always attach the decision-support disclaimer.
"""

from __future__ import annotations

from dataclasses import dataclass

from regulumen.core.eval.metrics import citation_hit
from regulumen.core.generation.faithfulness import (
    FaithfulnessResult,
    check_faithfulness,
    extract_citations,
    is_abstention,
)
from regulumen.core.generation.llm import build_llm
from regulumen.core.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from regulumen.core.retrieval.vector_store import RetrievedChunk

DISCLAIMER = (
    "ReguLumen provides decision support, not regulatory advice; "
    "verify with a qualified person."
)


def cited_chunks(
    used_citations: list[str], chunks: list[RetrievedChunk]
) -> list[RetrievedChunk]:
    """Return retrieved chunks whose labels were cited, in retrieval order.

    Shared single source of truth for the "which retrieved chunks back this
    answer" logic, reused by both :class:`Answerer` and the LangGraph
    ``GraphAnswerer`` so the two orchestrators stay byte-for-byte equivalent.
    """
    result: list[RetrievedChunk] = []
    for chunk in chunks:
        if any(citation_hit(used, [chunk.citation_label]) for used in used_citations):
            result.append(chunk)
    return result


@dataclass
class AnswerResult:
    """Outcome of answering one question.

    ``answered`` is False whenever the system abstained (no relevant clauses, an
    explicit model abstention, or a failed faithfulness gate). When abstained,
    ``answer`` holds the abstention message and ``abstention_reason`` the cause.
    """

    answered: bool
    answer: str
    citations: list[RetrievedChunk]
    abstained: bool
    abstention_reason: str | None
    faithfulness: FaithfulnessResult | None
    disclaimer: str


class Answerer:
    """Orchestrates retrieval, grounded generation, and the faithfulness gate."""

    def __init__(
        self,
        retriever,
        llm,
        *,
        top_k: int,
        abstention_threshold: float,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self.top_k = top_k
        # RESERVED for a future claim-grounding / NLI second pass. The current
        # post-generation faithfulness gate is STRICT-BINARY (a citation is
        # either valid or it forces abstention) and does NOT consult this
        # threshold. Kept on the object so config stays stable for when a
        # claim-support score is introduced. See faithfulness.check_faithfulness.
        self.abstention_threshold = abstention_threshold

    def answer(self, question: str) -> AnswerResult:
        chunks = self._retriever.retrieve(question, self.top_k)

        # 1. Early abstention on empty retrieval — do NOT call the LLM.
        if not chunks:
            return self._abstain(
                "No relevant clauses were retrieved for this question.",
                faithfulness=None,
            )

        # 2. Grounded generation.
        system = SYSTEM_PROMPT
        user = build_user_prompt(question, chunks)
        raw = self._llm.complete(system, user)

        # 3. Model chose to abstain — pass the stated gap through.
        if is_abstention(raw):
            return self._abstain(raw, faithfulness=None)

        # 4. Post-generation faithfulness GATE (strict-binary citation check).
        fr = check_faithfulness(raw, chunks)
        if fr.grounded:
            cited = cited_chunks(fr.used_citations, chunks)
            return AnswerResult(
                answered=True,
                answer=raw,
                citations=cited,
                abstained=False,
                abstention_reason=None,
                faithfulness=fr,
                disclaimer=DISCLAIMER,
            )

        # Not grounded -> replace the answer with a structured abstention that
        # names what went wrong (the hallucinated/unsupported or vague
        # citations). THIS is the strict gate doing its job.
        message = (
            "Unable to provide a grounded answer. "
            + fr.reason
            + " Please consult the cited source documents directly."
        )
        return self._abstain(message, faithfulness=fr, reason=fr.reason)

    # --- helpers -----------------------------------------------------------

    def _abstain(
        self,
        message: str,
        *,
        faithfulness: FaithfulnessResult | None,
        reason: str | None = None,
    ) -> AnswerResult:
        return AnswerResult(
            answered=False,
            answer=message,
            citations=[],
            abstained=True,
            abstention_reason=reason if reason is not None else message,
            faithfulness=faithfulness,
            disclaimer=DISCLAIMER,
        )

    # Backward-compatible alias for the now-shared module-level helper.
    _cited_chunks = staticmethod(cited_chunks)


def build_answerer(settings, retriever) -> Answerer:
    """Convenience constructor wiring an LLM from ``settings`` to a retriever."""
    llm = build_llm(settings)
    return Answerer(
        retriever,
        llm,
        top_k=settings.top_k,
        abstention_threshold=settings.abstention_threshold,
    )


# Re-exported for callers that want the extractor without importing faithfulness.
__all__ = [
    "AnswerResult",
    "Answerer",
    "DISCLAIMER",
    "build_answerer",
    "cited_chunks",
    "extract_citations",
]
