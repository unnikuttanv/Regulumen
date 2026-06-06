"""End-to-end (offline) test of the retrieval evaluation runner.

Builds a tiny in-memory HybridRetriever from a FakeEmbedder + a handful of
synthetic chunks with KNOWN citation labels, authors a few GoldItems whose
expected_citations point at those chunks, and asserts the aggregate metrics.
No network, no token, no real corpus.
"""

from __future__ import annotations

from pathlib import Path

from regulumen.core.eval.models import GoldItem
from regulumen.core.eval.runner import evaluate_retrieval, format_report
from regulumen.core.ingestion.models import Chunk
from regulumen.core.retrieval.bm25 import BM25Index
from regulumen.core.retrieval.embedder import FakeEmbedder
from regulumen.core.retrieval.hybrid import HybridRetriever
from regulumen.core.retrieval.vector_store import ChromaVectorStore


def _build_retriever(tmp_path: Path) -> HybridRetriever:
    chunks = [
        Chunk(
            id="c1",
            doc_id="doc-eu",
            clause_path=["Annex I", "17.2"],
            citation_label="EU MDR, Annex I §17.2",
            text="Software lifecycle processes including development and risk "
            "management for medical device software.",
        ),
        Chunk(
            id="c2",
            doc_id="doc-imdrf",
            clause_path=["5.1"],
            citation_label="IMDRF SaMD N10, §5.1",
            text="Software as a medical device safety classification categories "
            "based on the significance of information.",
        ),
        Chunk(
            id="c3",
            doc_id="doc-fda",
            clause_path=["IV", "A"],
            citation_label="FDA Premarket SW, Section IV §A",
            text="Documentation level for premarket submissions of device "
            "software functions and cybersecurity considerations.",
        ),
    ]
    store = ChromaVectorStore(
        FakeEmbedder(dim=64),
        path=str(tmp_path / "chroma"),
        collection_name="eval_test",
    )
    retriever = HybridRetriever(store, BM25Index(), alpha=0.5)
    retriever.index(chunks)
    return retriever


def test_evaluate_retrieval_recall_one_for_matching_question(tmp_path: Path):
    retriever = _build_retriever(tmp_path)
    gold = [
        GoldItem(
            id="q1",
            question="software lifecycle processes risk management",
            expected_citations=["EU MDR, Annex I §17.2"],
        ),
        GoldItem(
            id="q2",
            question="software as a medical device safety classification categories",
            expected_citations=["IMDRF SaMD N10, §5.1"],
        ),
    ]
    metrics = evaluate_retrieval(gold, retriever, k=3)

    assert metrics.mean_recall_at_k == 1.0
    assert metrics.mean_citation_accuracy == 1.0
    assert metrics.mean_mrr > 0.0
    by_id = {row.id: row for row in metrics.rows}
    assert by_id["q1"].recall == 1.0
    assert "EU MDR, Annex I §17.2" in by_id["q1"].hits


def test_evaluate_retrieval_zero_recall_for_absent_citation(tmp_path: Path):
    retriever = _build_retriever(tmp_path)
    gold = [
        GoldItem(
            id="q_absent",
            question="ISO 14971 risk management application",
            expected_citations=["ISO 14971, §4.1"],  # not in the corpus
        )
    ]
    metrics = evaluate_retrieval(gold, retriever, k=3)
    assert metrics.mean_recall_at_k == 0.0
    assert metrics.rows[0].mrr == 0.0
    assert "ISO 14971, §4.1" in metrics.rows[0].misses


def test_aggregate_excludes_abstain_items_from_means(tmp_path: Path):
    # 2 normal items + 1 must_abstain (empty expected) item. The abstain item's
    # vacuous per-item recall of 1.0 must NOT be averaged into mean_recall_at_k.
    retriever = _build_retriever(tmp_path)
    gold = [
        GoldItem(
            id="q1",
            question="software lifecycle processes risk management",
            expected_citations=["EU MDR, Annex I §17.2"],  # will hit -> recall 1.0
        ),
        GoldItem(
            id="q_absent",
            question="ISO 14971 risk management application",
            expected_citations=["ISO 14971, §4.1"],  # not in corpus -> recall 0.0
        ),
        GoldItem(
            id="q_abstain",
            question="something not in the corpus at all",
            expected_citations=[],
            must_abstain=True,
        ),
    ]
    metrics = evaluate_retrieval(gold, retriever, k=3)

    assert metrics.n_items == 3
    assert metrics.n_retrieval_items == 2
    # Mean over ONLY the 2 retrieval items: (1.0 + 0.0) / 2 == 0.5.
    # If the abstain item (vacuous 1.0) were included it would be (1+0+1)/3.
    assert metrics.mean_recall_at_k == 0.5
    # The abstain item still gets a visible per-item row.
    by_id = {row.id: row for row in metrics.rows}
    assert "q_abstain" in by_id


def test_format_report_shows_both_counts(tmp_path: Path):
    retriever = _build_retriever(tmp_path)
    gold = [
        GoldItem(
            id="q1",
            question="software lifecycle processes",
            expected_citations=["EU MDR, Annex I §17.2"],
        ),
        GoldItem(
            id="q_abstain",
            question="not in corpus",
            expected_citations=[],
            must_abstain=True,
        ),
    ]
    metrics = evaluate_retrieval(gold, retriever, k=3)
    report = format_report(metrics)
    assert "n_items=2" in report
    assert "retrieval-scored=1" in report


def test_format_report_is_nonempty_with_metric_names(tmp_path: Path):
    retriever = _build_retriever(tmp_path)
    gold = [
        GoldItem(
            id="q1",
            question="software lifecycle processes",
            expected_citations=["EU MDR, Annex I §17.2"],
        )
    ]
    metrics = evaluate_retrieval(gold, retriever, k=3)
    report = format_report(metrics)
    assert isinstance(report, str)
    assert report.strip()
    # The citation_accuracy column is labelled as a recall measure to avoid
    # confusion with Step 7 generation accuracy.
    for name in ("precision", "recall", "cit_recall", "mrr"):
        assert name in report.lower()
