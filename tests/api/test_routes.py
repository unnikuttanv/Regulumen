"""End-to-end route tests, fully OFFLINE via dependency-override DI.

No network, no API keys: we inject either a fake RegulumenService or a real
RegulumenService wired from in-memory fakes (FakeLLM + an in-memory retriever
backed by FakeEmbedder). This proves the layering is testable without providers.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from regulumen.api.main import create_app
from regulumen.api.router import get_service
from regulumen.api.schemas import (
    AskResponse,
    CitationOut,
    ClassifyResponse,
    HealthResponse,
)
from regulumen.core.generation.llm import LLMError


class FakeService:
    """Stand-in RegulumenService scripting ask/classify/health outcomes."""

    def __init__(self, *, ask_result=None, classify_result=None, indexed=0):
        self._ask_result = ask_result
        self._classify_result = classify_result
        self._indexed = indexed

    def ask(self, question: str) -> AskResponse:  # noqa: ARG002
        if isinstance(self._ask_result, Exception):
            raise self._ask_result
        return self._ask_result

    def classify(self, description: str) -> ClassifyResponse:  # noqa: ARG002
        if isinstance(self._classify_result, Exception):
            raise self._classify_result
        return self._classify_result

    def health(self) -> HealthResponse:
        return HealthResponse(status="ok", indexed_chunks=self._indexed)


def _client(service) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app)


def _citation() -> CitationOut:
    return CitationOut(
        citation_label="ISO 13485 §7.5.1",
        doc_id="iso13485",
        clause_path=["7", "7.5", "7.5.1"],
        score=0.9,
        snippet="You shall validate the process...",
    )


def test_health_ok():
    client = _client(FakeService(indexed=42))
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["indexed_chunks"] == 42


def test_ask_answered():
    answer = AskResponse(
        answered=True,
        answer="Per [ISO 13485 §7.5.1], validate the process.",
        abstained=False,
        abstention_reason=None,
        citations=[_citation()],
        disclaimer="decision support only",
    )
    client = _client(FakeService(ask_result=answer))
    resp = client.post("/ask", json={"question": "How do I validate?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answered"] is True
    assert body["citations"]
    assert body["disclaimer"]


def test_ask_abstained():
    answer = AskResponse(
        answered=False,
        answer="No relevant clauses were retrieved.",
        abstained=True,
        abstention_reason="No relevant clauses were retrieved.",
        citations=[],
        disclaimer="decision support only",
    )
    client = _client(FakeService(ask_result=answer))
    resp = client.post("/ask", json={"question": "Unrelated question?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answered"] is False
    assert body["abstained"] is True
    assert body["abstention_reason"]


def test_ask_empty_question_422():
    client = _client(FakeService())
    resp = client.post("/ask", json={"question": "   "})
    assert resp.status_code == 422


def test_classify_classified():
    result = ClassifyResponse(
        safety_class="C",
        abstained=False,
        severity="serious",
        can_contribute=True,
        rationale="Serious injury possible -> Class C.",
        citations=[_citation()],
        disclaimer="decision support only",
    )
    client = _client(FakeService(classify_result=result))
    resp = client.post("/classify", json={"description": "Infusion pump dose controller"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["safety_class"] == "C"
    assert body["disclaimer"]


def test_classify_abstained():
    result = ClassifyResponse(
        safety_class=None,
        abstained=True,
        severity=None,
        can_contribute=None,
        rationale="Severity could not be established.",
        citations=[],
        disclaimer="decision support only",
    )
    client = _client(FakeService(classify_result=result))
    resp = client.post("/classify", json={"description": "vague thing"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["safety_class"] is None
    assert body["abstained"] is True


def test_classify_empty_422():
    client = _client(FakeService())
    resp = client.post("/classify", json={"description": ""})
    assert resp.status_code == 422


def test_ask_provider_unavailable_returns_503_not_500():
    client = _client(FakeService(ask_result=LLMError("no key")))
    resp = client.post("/ask", json={"question": "How do I validate?"})
    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert "REGULUMEN_GROQ_API_KEY" in detail
    assert "REGULUMEN_HF_API_TOKEN" in detail


def test_classify_provider_unavailable_returns_503_not_500():
    client = _client(FakeService(classify_result=LLMError("no key")))
    resp = client.post("/classify", json={"description": "Infusion pump"})
    assert resp.status_code == 503


def test_service_built_from_fakes_end_to_end():
    """Real RegulumenService wired from in-memory fakes (no network/keys)."""
    import tempfile

    from regulumen.api.repository import RetrievalRepository
    from regulumen.api.service import RegulumenService
    from regulumen.core.generation.answerer import Answerer
    from regulumen.core.generation.llm import FakeLLM
    from regulumen.core.ingestion.models import Chunk
    from regulumen.core.retrieval.bm25 import BM25Index
    from regulumen.core.retrieval.embedder import FakeEmbedder
    from regulumen.core.retrieval.hybrid import HybridRetriever
    from regulumen.core.retrieval.vector_store import ChromaVectorStore
    from regulumen.core.risk.classifier import RiskClassifier

    tmp = tempfile.mkdtemp()
    store = ChromaVectorStore(FakeEmbedder(), path=tmp, collection_name="apitest")
    bm25 = BM25Index()
    retriever = HybridRetriever(store, bm25, alpha=0.5)
    chunks = [
        Chunk(
            id="c1",
            text="Records of process validation shall be maintained per clause 7.5.1.",
            citation_label="ISO 13485 §7.5.1",
            doc_id="iso13485",
            clause_path=["7", "7.5", "7.5.1"],
            metadata={},
        )
    ]
    retriever.index(chunks)

    repo = RetrievalRepository(retriever=retriever)
    grounded = "Process validation records must be kept [[ISO 13485 §7.5.1]]."
    answerer = Answerer(retriever, FakeLLM(grounded), top_k=4, abstention_threshold=0.7)
    classifier = RiskClassifier(
        FakeLLM('{"can_contribute_to_hazard": true, "worst_case_severity": "serious", '
                '"external_control_reduces_to": null, "needs_more_info": false}'),
        retriever=retriever,
        top_k=4,
    )
    service = RegulumenService(answerer, classifier, repo)

    client = _client(service)

    h = client.get("/health")
    assert h.status_code == 200
    assert h.json()["indexed_chunks"] >= 1

    a = client.post("/ask", json={"question": "validation records 7.5.1"})
    assert a.status_code == 200
    assert a.json()["answered"] is True

    c = client.post("/classify", json={"description": "dose controller for infusion 7.5.1"})
    assert c.status_code == 200
    assert c.json()["safety_class"] == "C"


def test_app_imports_without_keys():
    """Importing the app module / create_app must not need API keys."""
    import importlib

    import regulumen.api.main as main_mod

    importlib.reload(main_mod)
    assert main_mod.app is not None
