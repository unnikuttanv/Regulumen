"""LangGraph multi-step Q&A orchestrator with a faithfulness refine loop.

This is a cyclic :class:`~langgraph.graph.StateGraph` alternative to the linear
:class:`~regulumen.core.generation.answerer.Answerer`. It produces the SAME
:class:`~regulumen.core.generation.answerer.AnswerResult` contract and preserves
every guarantee of the linear pipeline, while adding genuine multi-step value:

    when the strict faithfulness GATE fails (the LLM produced an ungrounded
    answer), instead of abstaining immediately, the graph REFINES the retrieval
    query once and re-retrieves, then re-generates and re-checks. Only if it
    STILL fails after the budget is exhausted does it abstain.

The graph (nodes -> edges):

    START -> retrieve -> generate --(terminal? )--> END
                              |
                              +--(else)--> gate --(grounded?)--> END
                                             |
                                             +--(budget left?)--> refine -> retrieve  (cycle)
                                             |
                                             +--(no budget)--> END (structured abstention)

CRITICAL invariants (identical to the linear Answerer):
  * The faithfulness gate is AUTHORITATIVE. ``refine`` only RETRIES — it never
    overrides or bypasses the gate. After the final attempt, an ungrounded
    answer is turned into a structured abstention; a hallucinated answer can
    never reach the user.
  * Empty retrieval -> abstain WITHOUT calling the LLM.
  * Model abstention token -> passed through unchanged.
  * The loop is bounded by ``max_attempts`` (total generate cycles), so it can
    never spin forever.

Determinism: every node is a pure-ish ``state -> partial-update`` function and
the only side effects are the injected retriever / LLM, so the whole graph is
fully testable offline with a stub retriever and ``FakeLLM``.
"""

from __future__ import annotations

import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from regulumen.core.generation.answerer import (
    DISCLAIMER,
    AnswerResult,
    cited_chunks,
)
from regulumen.core.generation.faithfulness import (
    FaithfulnessResult,
    check_faithfulness,
    is_abstention,
)
from regulumen.core.generation.llm import build_llm
from regulumen.core.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from regulumen.core.retrieval.vector_store import RetrievedChunk


class AnswerState(TypedDict, total=False):
    """Mutable state threaded through the graph.

    ``total=False`` so nodes may return partial updates; LangGraph merges each
    node's returned dict into the running state.
    """

    question: str  # the original user question (never mutated)
    query: str  # current retrieval query (starts == question, may be refined)
    chunks: list[RetrievedChunk]  # last retrieval result
    raw: str  # last raw LLM answer
    faithfulness: FaithfulnessResult | None  # last gate verdict
    attempts: int  # number of generate cycles completed
    widen: bool  # uncited-refine path: tell _retrieve to fetch top_k*2 candidates
    result: AnswerResult | None  # terminal output (set => graph should stop)


def _abstain(
    message: str,
    *,
    faithfulness: FaithfulnessResult | None,
    reason: str | None = None,
) -> AnswerResult:
    """Build a structured abstention AnswerResult (mirrors Answerer._abstain)."""
    return AnswerResult(
        answered=False,
        answer=message,
        citations=[],
        abstained=True,
        abstention_reason=reason if reason is not None else message,
        faithfulness=faithfulness,
        disclaimer=DISCLAIMER,
    )


# Matches the clause/section-number-ish tokens worth keeping from a citation
# label: dotted/plain numbers ("17.2", "10") and structural words like
# "Annex"/"Article"/"Section"/"§" optionally followed by a number ("Annex I",
# "Article 10"). Everything else in a verbose label (regulation names, prose) is
# lexical noise for the hybrid BM25+dense retriever, so we drop it.
_CLAUSE_TOKEN_RE = re.compile(
    r"(?:Annex|Article|Section|Clause|§)\s*[A-Za-z0-9.]*|\d+(?:\.\d+)*",
    re.IGNORECASE,
)


def _extract_clause_tokens(labels: list[str]) -> list[str]:
    """Pull clause/section-number-ish tokens from unsupported citation labels.

    Returns de-duplicated, order-preserving tokens (dotted numbers like "17.2",
    plain digits, and structural words such as "Annex"/"Article" plus any
    trailing number). Used to steer re-retrieval toward the specific clause the
    model reached for, without the verbose label's lexical noise. Returns ``[]``
    when nothing clause-like is found (caller then falls back to raw labels).
    """
    seen: dict[str, None] = {}
    for label in labels:
        for match in _CLAUSE_TOKEN_RE.finditer(label):
            token = match.group(0).strip()
            if token and token not in seen:
                seen[token] = None
    return list(seen)


class GraphAnswerer:
    """Cyclic LangGraph orchestrator with a bounded faithfulness refine loop (default: one retry).

    Public surface mirrors :class:`~regulumen.core.generation.answerer.Answerer`
    so the two are drop-in interchangeable.
    """

    def __init__(
        self,
        retriever,
        llm,
        *,
        top_k: int,
        abstention_threshold: float,
        max_attempts: int = 2,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self.top_k = top_k
        # RESERVED — the post-generation faithfulness gate is STRICT-BINARY and
        # does NOT consult this threshold (see Answerer / faithfulness module).
        # Kept on the object so config stays stable for a future claim-support
        # second pass.
        self.abstention_threshold = abstention_threshold
        # Total generate cycles allowed. 2 => one refine retry. Bounds the loop.
        self.max_attempts = max(1, int(max_attempts))
        self._graph = self._build_graph()

    # --- nodes -------------------------------------------------------------

    def _retrieve(self, state: AnswerState) -> AnswerState:
        # On the uncited-refine path the query is unchanged, so we instead widen
        # k to surface more candidates (see _refine). Default k otherwise.
        k = self.top_k * 2 if state.get("widen") else self.top_k
        chunks = self._retriever.retrieve(state["query"], k)
        return {"chunks": chunks}

    def _generate(self, state: AnswerState) -> AnswerState:
        # chunks is always set by retrieve, which always precedes generate.
        chunks = state.get("chunks") or []

        # Early abstention on empty retrieval — do NOT call the LLM.
        if not chunks:
            return {
                "result": _abstain(
                    "No relevant clauses were retrieved for this question.",
                    faithfulness=None,
                )
            }

        raw = self._llm.complete(
            SYSTEM_PROMPT, build_user_prompt(state["question"], chunks)
        )
        attempts = state.get("attempts", 0) + 1

        # Model chose to abstain — pass the stated gap through unchanged.
        if is_abstention(raw):
            return {
                "raw": raw,
                "attempts": attempts,
                "result": _abstain(raw, faithfulness=None),
            }

        return {"raw": raw, "attempts": attempts}

    def _gate(self, state: AnswerState) -> AnswerState:
        chunks = state.get("chunks") or []  # set by retrieve, which always precedes gate
        raw = state.get("raw", "")  # _after_generate only routes here when generate set raw
        fr = check_faithfulness(raw, chunks)

        if fr.grounded:
            return {
                "faithfulness": fr,
                "result": AnswerResult(
                    answered=True,
                    answer=raw,
                    citations=cited_chunks(fr.used_citations, chunks),
                    abstained=False,
                    abstention_reason=None,
                    faithfulness=fr,
                    disclaimer=DISCLAIMER,
                ),
            }

        # Not grounded. If no refine budget remains, abstain now with the SAME
        # structured wording the linear Answerer produces. Otherwise leave
        # ``result`` unset so the router sends us to ``refine``.
        if state.get("attempts", 0) >= self.max_attempts:
            message = (
                "Unable to provide a grounded answer. "
                + fr.reason
                + " Please consult the cited source documents directly."
            )
            return {
                "faithfulness": fr,
                "result": _abstain(message, faithfulness=fr, reason=fr.reason),
            }

        return {"faithfulness": fr}

    def _refine(self, state: AnswerState) -> AnswerState:
        """Make the next retrieval attempt genuinely DIFFERENT from the last.

        This refine loop recovers an answer ONLY when the correct clause exists
        in the index but was mis-ranked below top-k on the first pass; it canNOT
        help when the model cited a clause that does not exist in the corpus at
        all (a hallucination), because no retrieval query can surface a clause
        that was never indexed. In that case the loop simply exhausts its budget
        and the gate's abstention stands — which is the honest outcome.

        Two paths, both deterministic and test-friendly:

        * Unsupported citations present (the answer cited specific clauses that
          were not retrieved): steer the query toward those clauses by appending
          their clause/section-number-ish TOKENS (e.g. "17.2", "Annex I") rather
          than the verbose label strings, which would add lexical noise to the
          hybrid BM25+dense retriever. If token extraction yields nothing, fall
          back to appending the raw labels so the retry still differs.
        * No unsupported citations (the answer was UNCITED): the query cannot be
          improved from missing-label hints, and re-running it verbatim would be
          pure churn. Instead we set ``widen`` so ``_retrieve`` fetches
          ``top_k * 2`` candidates — a materially different retrieval that may
          surface a correct clause that was just below the original cutoff.
        """
        fr = state.get("faithfulness")
        hints = list(fr.unsupported_citations) if fr else []
        question = state["question"]

        if hints:
            tokens = _extract_clause_tokens(hints) or hints
            refined = (question + " " + " ".join(tokens)).strip()
            return {"query": refined, "widen": False}

        # Uncited answer: keep the query, widen retrieval instead of churning.
        return {"query": question, "widen": True}

    # --- routers -----------------------------------------------------------

    @staticmethod
    def _after_generate(state: AnswerState) -> str:
        # Terminal result set (empty retrieval or model abstention) -> stop.
        return "end" if state.get("result") is not None else "gate"

    @staticmethod
    def _after_gate(state: AnswerState) -> str:
        # Result set means either grounded (answer) or budget-exhausted
        # abstention — both terminal. Otherwise loop through refine.
        return "end" if state.get("result") is not None else "refine"

    # --- graph wiring ------------------------------------------------------

    def _build_graph(self):
        builder = StateGraph(AnswerState)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("generate", self._generate)
        builder.add_node("gate", self._gate)
        builder.add_node("refine", self._refine)

        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_conditional_edges(
            "generate", self._after_generate, {"gate": "gate", "end": END}
        )
        builder.add_conditional_edges(
            "gate", self._after_gate, {"refine": "refine", "end": END}
        )
        builder.add_edge("refine", "retrieve")  # the refine cycle
        return builder.compile()

    # --- public API --------------------------------------------------------

    def answer(self, question: str) -> AnswerResult:
        initial: AnswerState = {
            "question": question,
            "query": question,
            "attempts": 0,
            "widen": False,
            "faithfulness": None,
            "result": None,
        }
        final = self._graph.invoke(initial)
        result = final.get("result")
        if result is None:  # pragma: no cover - defensive; routers guarantee one
            return _abstain(
                "Unable to provide a grounded answer. "
                "Please consult the cited source documents directly.",
                faithfulness=final.get("faithfulness"),
            )
        return result


def build_graph_answerer(settings, retriever) -> GraphAnswerer:
    """Convenience constructor mirroring ``build_answerer`` for the graph variant."""
    llm = build_llm(settings)
    return GraphAnswerer(
        retriever,
        llm,
        top_k=settings.top_k,
        abstention_threshold=settings.abstention_threshold,
        max_attempts=getattr(settings, "max_answer_attempts", 2),
    )


__all__ = [
    "AnswerState",
    "GraphAnswerer",
    "build_graph_answerer",
]
