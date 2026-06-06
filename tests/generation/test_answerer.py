"""Tests for the Answerer orchestration — the headline strict-faithfulness flow.

All offline: a stub retriever returns synthetic chunks and a ``FakeLLM`` returns
scripted answers. The decisive test proves that an answer citing a clause NOT in
the retrieved set is FORCED to an abstention by the gate.
"""

from __future__ import annotations

from regulumen.core.generation.answerer import Answerer, AnswerResult
from regulumen.core.generation.faithfulness import ABSTENTION_TOKEN
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
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks
        self.calls = 0

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        self.calls += 1
        return self._chunks[:k]


class _ExplodingLLM:
    """A FakeLLM-shaped object that fails if .complete is ever called."""

    calls = 0

    def complete(self, system: str, user: str) -> str:  # pragma: no cover - must not run
        raise AssertionError("LLM was called despite empty retrieval")


def test_grounded_answer_is_returned():
    retriever = _StubRetriever([_chunk("EU MDR, Annex I §17.2")])
    llm = FakeLLM("Software must meet state of the art [[EU MDR, Annex I §17.2]].")
    answerer = Answerer(retriever, llm, top_k=8, abstention_threshold=0.7)

    result = answerer.answer("What about software?")

    assert isinstance(result, AnswerResult)
    assert result.answered is True
    assert result.abstained is False
    assert "[[EU MDR, Annex I §17.2]]" in result.answer
    assert len(result.citations) == 1
    assert result.citations[0].citation_label == "EU MDR, Annex I §17.2"
    assert result.disclaimer  # always present
    assert "decision support" in result.disclaimer.lower()


def test_hallucinated_citation_is_forced_to_abstention():
    """THE money test: a cited clause not among retrieved chunks => abstain."""
    retriever = _StubRetriever([_chunk("EU MDR, Annex I §17.2")])
    # The model cites a clause that was NEVER retrieved.
    llm = FakeLLM("The device must do X [[EU MDR, Annex I §99.9]].")
    answerer = Answerer(retriever, llm, top_k=8, abstention_threshold=0.7)

    result = answerer.answer("Does the device need X?")

    assert result.answered is False
    assert result.abstained is True
    assert result.abstention_reason is not None
    # The reason must name the unsupported (hallucinated) citation.
    assert "EU MDR, Annex I §99.9" in result.abstention_reason
    # And the faithfulness verdict must flag it as not grounded.
    assert result.faithfulness is not None
    assert result.faithfulness.grounded is False
    assert "EU MDR, Annex I §99.9" in result.faithfulness.unsupported_citations


def test_empty_retrieval_abstains_without_calling_llm():
    retriever = _StubRetriever([])
    llm = _ExplodingLLM()
    answerer = Answerer(retriever, llm, top_k=8, abstention_threshold=0.7)

    result = answerer.answer("Anything?")

    assert result.answered is False
    assert result.abstained is True
    assert llm.calls == 0
    assert result.citations == []
    assert result.disclaimer


def test_model_abstention_is_passed_through():
    retriever = _StubRetriever([_chunk("EU MDR, Annex I §17.2")])
    llm = FakeLLM(f"{ABSTENTION_TOKEN} The excerpts do not address battery safety.")
    answerer = Answerer(retriever, llm, top_k=8, abstention_threshold=0.7)

    result = answerer.answer("What about battery safety?")

    assert result.answered is False
    assert result.abstained is True
    assert result.abstention_reason is not None
    assert "battery safety" in result.abstention_reason


def test_disclaimer_always_present_across_outcomes():
    for raw in (
        "Grounded [[EU MDR §1]].",
        "Bad [[EU MDR §999]].",
        f"{ABSTENTION_TOKEN} nope.",
    ):
        answerer = Answerer(
            _StubRetriever([_chunk("EU MDR §1")]),
            FakeLLM(raw),
            top_k=8,
            abstention_threshold=0.7,
        )
        result = answerer.answer("q")
        assert result.disclaimer
        assert "regulatory advice" in result.disclaimer.lower()


def test_retriever_called_with_top_k():
    retriever = _StubRetriever([_chunk("EU MDR §1"), _chunk("EU MDR §2"), _chunk("EU MDR §3")])
    llm = FakeLLM("Answer [[EU MDR §1]].")
    answerer = Answerer(retriever, llm, top_k=2, abstention_threshold=0.7)
    answerer.answer("q")
    assert retriever.calls == 1


def test_only_used_chunks_appear_in_citations():
    retriever = _StubRetriever([_chunk("EU MDR §1"), _chunk("EU MDR §2")])
    llm = FakeLLM("Only the first matters [[EU MDR §1]].")
    answerer = Answerer(retriever, llm, top_k=8, abstention_threshold=0.7)
    result = answerer.answer("q")
    assert result.answered is True
    labels = [c.citation_label for c in result.citations]
    assert labels == ["EU MDR §1"]


def test_build_answerer_factory():
    from regulumen.core.generation import answerer as answerer_mod

    class _S:
        llm_model = "ollama/llama3"  # keyless model so build_llm won't raise
        groq_api_key = None
        top_k = 5
        abstention_threshold = 0.8

    retriever = _StubRetriever([_chunk("EU MDR §1")])
    a = answerer_mod.build_answerer(_S(), retriever)
    assert isinstance(a, Answerer)
    assert a.top_k == 5
    assert a.abstention_threshold == 0.8
