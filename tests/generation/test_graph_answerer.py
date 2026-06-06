"""Tests for the LangGraph-based GraphAnswerer with a refine loop.

All offline: stub retrievers return synthetic chunks (and can return DIFFERENT
chunks for the refined query so the refine loop is exercised), and a ``FakeLLM``
returns scripted answers (varying per call via a callable). The headline test
proves genuine multi-step value: a first ungrounded answer triggers a refine +
re-retrieve, and the second attempt recovers a grounded answer.

The faithfulness GATE stays authoritative throughout: the refine loop only
RETRIES; after the final attempt an ungrounded answer MUST become an abstention,
exactly as the linear Answerer guarantees.
"""

from __future__ import annotations

from regulumen.core.generation.answerer import Answerer, AnswerResult
from regulumen.core.generation.faithfulness import ABSTENTION_TOKEN
from regulumen.core.generation.graph_answerer import (
    GraphAnswerer,
    build_graph_answerer,
)
from regulumen.core.generation.llm import FakeLLM
from regulumen.core.retrieval.vector_store import RetrievedChunk


def _chunk(label: str, *, id_: str | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        id=id_ or label,
        text=f"text for {label}",
        citation_label=label,
        doc_id="doc",
        clause_path=[],
        score=0.9,
        metadata={},
    )


class _StubRetriever:
    """Returns the same canned chunks for every query."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks
        self.calls = 0
        self.queries: list[str] = []

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        self.calls += 1
        self.queries.append(query)
        return self._chunks[:k]


class _SequenceRetriever:
    """Returns a different chunk set on each successive call."""

    def __init__(self, sequences: list[list[RetrievedChunk]]) -> None:
        self._sequences = sequences
        self.calls = 0
        self.queries: list[str] = []

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        idx = min(self.calls, len(self._sequences) - 1)
        self.calls += 1
        self.queries.append(query)
        return self._sequences[idx][:k]


class _ExplodingLLM:
    """FakeLLM-shaped object that fails if .complete is ever called."""

    calls = 0

    def complete(self, system: str, user: str) -> str:  # pragma: no cover
        raise AssertionError("LLM was called despite empty retrieval")


def test_grounded_on_first_try_answers():
    retriever = _StubRetriever([_chunk("EU MDR, Annex I §17.2")])
    llm = FakeLLM("Software must meet state of the art [[EU MDR, Annex I §17.2]].")
    ga = GraphAnswerer(retriever, llm, top_k=8, abstention_threshold=0.7)

    result = ga.answer("What about software?")

    assert isinstance(result, AnswerResult)
    assert result.answered is True
    assert result.abstained is False
    assert len(result.citations) == 1
    assert result.citations[0].citation_label == "EU MDR, Annex I §17.2"
    assert retriever.calls == 1
    assert llm.calls == 1
    assert result.disclaimer
    assert "decision support" in result.disclaimer.lower()


def test_refine_recovers_grounded_answer():
    """THE headline test: ungrounded first answer => refine + re-retrieve =>
    grounded second answer. Proves multi-step value."""
    good = _chunk("EU MDR, Annex I §17.2")
    other = _chunk("EU MDR, Annex I §1.1")
    # First retrieval lacks the good clause; refined retrieval includes it.
    retriever = _SequenceRetriever([[other], [good]])

    def script(system: str, user: str) -> str:
        # The good clause's text only appears in the refined retrieval prompt.
        if good.text in user:
            return "Software must meet state of the art [[EU MDR, Annex I §17.2]]."
        # First attempt cites a clause that was NOT retrieved -> gate fails.
        return "The device must do X [[EU MDR, Annex I §17.2]]."

    llm = FakeLLM(script)
    ga = GraphAnswerer(retriever, llm, top_k=8, abstention_threshold=0.7, max_attempts=2)

    result = ga.answer("What about software?")

    assert result.answered is True
    assert result.abstained is False
    assert retriever.calls == 2  # original + refined re-retrieval
    assert llm.calls == 2  # one generate per retrieval
    assert result.faithfulness is not None
    assert result.faithfulness.grounded is True
    labels = [c.citation_label for c in result.citations]
    assert labels == ["EU MDR, Annex I §17.2"]


def test_refine_exhausted_abstains():
    retriever = _StubRetriever([_chunk("EU MDR, Annex I §17.2")])
    # Always cites a clause that was never retrieved -> always ungrounded.
    llm = FakeLLM("The device must do X [[EU MDR, Annex I §99.9]].")
    ga = GraphAnswerer(retriever, llm, top_k=8, abstention_threshold=0.7, max_attempts=2)

    result = ga.answer("Does the device need X?")

    assert result.answered is False
    assert result.abstained is True
    assert result.abstention_reason is not None
    assert result.faithfulness is not None
    assert result.faithfulness.grounded is False
    # Bounded: exactly max_attempts generate cycles, no infinite loop.
    assert llm.calls == 2
    assert retriever.calls == 2
    # Wording matches the linear Answerer's structured abstention.
    assert result.answer.startswith("Unable to provide a grounded answer.")
    assert "Please consult the cited source documents directly." in result.answer


def test_empty_retrieval_abstains_without_calling_llm():
    retriever = _StubRetriever([])
    llm = _ExplodingLLM()
    ga = GraphAnswerer(retriever, llm, top_k=8, abstention_threshold=0.7)

    result = ga.answer("Anything?")

    assert result.answered is False
    assert result.abstained is True
    assert llm.calls == 0
    assert result.citations == []
    assert result.disclaimer


def test_model_abstention_is_passed_through():
    retriever = _StubRetriever([_chunk("EU MDR, Annex I §17.2")])
    llm = FakeLLM(f"{ABSTENTION_TOKEN} The excerpts do not address battery safety.")
    ga = GraphAnswerer(retriever, llm, top_k=8, abstention_threshold=0.7)

    result = ga.answer("What about battery safety?")

    assert result.answered is False
    assert result.abstained is True
    assert result.abstention_reason is not None
    assert "battery safety" in result.abstention_reason
    # No refine attempted on a model abstention.
    assert llm.calls == 1


def test_disclaimer_always_present_across_outcomes():
    for raw in (
        "Grounded [[EU MDR §1]].",
        "Bad [[EU MDR §999]].",
        f"{ABSTENTION_TOKEN} nope.",
    ):
        ga = GraphAnswerer(
            _StubRetriever([_chunk("EU MDR §1")]),
            FakeLLM(raw),
            top_k=8,
            abstention_threshold=0.7,
        )
        result = ga.answer("q")
        assert result.disclaimer
        assert "regulatory advice" in result.disclaimer.lower()


def test_parity_with_linear_answerer_grounded():
    chunks = [_chunk("EU MDR §1"), _chunk("EU MDR §2")]
    raw = "Only the first matters [[EU MDR §1]]."

    linear = Answerer(_StubRetriever(list(chunks)), FakeLLM(raw), top_k=8, abstention_threshold=0.7)
    graph = GraphAnswerer(
        _StubRetriever(list(chunks)), FakeLLM(raw), top_k=8, abstention_threshold=0.7
    )

    lr = linear.answer("q")
    gr = graph.answer("q")

    assert lr.answered == gr.answered is True
    assert lr.abstained == gr.abstained is False
    assert [c.citation_label for c in lr.citations] == [c.citation_label for c in gr.citations]


def test_parity_with_linear_answerer_ungrounded():
    # max_attempts=1 => no refine possible, so both must abstain identically.
    chunks = [_chunk("EU MDR §1")]
    raw = "Hallucinated [[EU MDR §999]]."

    linear = Answerer(_StubRetriever(list(chunks)), FakeLLM(raw), top_k=8, abstention_threshold=0.7)
    graph = GraphAnswerer(
        _StubRetriever(list(chunks)),
        FakeLLM(raw),
        top_k=8,
        abstention_threshold=0.7,
        max_attempts=1,
    )

    lr = linear.answer("q")
    gr = graph.answer("q")

    assert lr.answered == gr.answered is False
    assert lr.abstained == gr.abstained is True
    assert lr.answer == gr.answer
    assert lr.abstention_reason == gr.abstention_reason


def test_refine_yields_empty_abstains():
    """First attempt cites a non-retrieved clause (gate fails), the refined
    re-retrieval returns [] so the graph abstains cleanly WITHOUT a third
    attempt and without crashing on the empty retrieval."""
    good_but_wrong = _chunk("EU MDR, Annex I §1.1")
    # First retrieval returns a real-but-wrong chunk; refined retrieval is empty.
    retriever = _SequenceRetriever([[good_but_wrong], []])
    # Always cites a clause that was never retrieved -> first attempt ungrounded.
    llm = FakeLLM("The device must do X [[EU MDR, Annex I §17.2]].")
    ga = GraphAnswerer(retriever, llm, top_k=8, abstention_threshold=0.7, max_attempts=2)

    result = ga.answer("Does the device need X?")

    assert result.answered is False
    assert result.abstained is True
    assert result.abstention_reason is not None
    # Bounded: first generate produced an answer, second retrieval was empty so
    # the LLM was NOT called again. Exactly one generate, two retrievals.
    assert llm.calls == 1
    assert retriever.calls == 2


def test_uncited_first_answer_triggers_widened_reretrieval():
    """An UNCITED first answer can't be refined from missing-label hints, so the
    refine path widens k instead of re-running the identical query: the second
    retrieval must receive a larger k (top_k * 2)."""
    captured_ks: list[int] = []
    good = _chunk("EU MDR, Annex I §17.2")

    class _RecordingRetriever:
        def __init__(self) -> None:
            self.calls = 0

        def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
            captured_ks.append(k)
            self.calls += 1
            # First call: no good chunk in scope (uncited answer follows).
            # Second (widened) call: surface the good clause.
            return [good] if self.calls >= 2 else [_chunk("EU MDR, Annex I §1.1")]

    retriever = _RecordingRetriever()

    def script(system: str, user: str) -> str:
        if good.text in user:
            return "Software must meet state of the art [[EU MDR, Annex I §17.2]]."
        # First answer makes claims with NO citation at all -> gate fails uncited.
        return "The device must simply work."

    llm = FakeLLM(script)
    ga = GraphAnswerer(retriever, llm, top_k=8, abstention_threshold=0.7, max_attempts=2)

    ga.answer("What about software?")

    assert retriever.calls == 2
    assert captured_ks == [8, 16]  # original k, then widened k = top_k * 2


def test_build_graph_answerer_factory():
    class _S:
        llm_model = "ollama/llama3"  # keyless model so build_llm won't raise
        groq_api_key = None
        top_k = 5
        abstention_threshold = 0.8
        max_answer_attempts = 3

    retriever = _StubRetriever([_chunk("EU MDR §1")])
    ga = build_graph_answerer(_S(), retriever)
    assert isinstance(ga, GraphAnswerer)
    assert ga.top_k == 5
    assert ga.abstention_threshold == 0.8
    assert ga.max_attempts == 3
