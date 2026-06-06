"""Retrieval evaluation runner for ReguLumen.

Two layers:

* :func:`evaluate_retrieval` — pure orchestration over a duck-typed retriever
  (anything with ``.retrieve(query, k) -> list[RetrievedChunk]``, e.g.
  :class:`~regulumen.core.retrieval.hybrid.HybridRetriever`). Fully offline-
  testable with a :class:`~regulumen.core.retrieval.embedder.FakeEmbedder`.
* A ``main()`` CLI that builds the real corpus, indexes a hybrid retriever with
  the configured embedder, runs the eval and writes a JSON artefact. The CLI
  needs network + an HF token for embeddings; it exits gracefully with a clear
  message when those are missing. The CLI is for Step 6 tuning and is NOT run in
  CI.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from regulumen.core.eval.metrics import (
    PerItemRow,
    RetrievalMetrics,
    citation_accuracy,
    citation_hit,
    mrr,
    precision_at_k,
    recall_at_k,
)
from regulumen.core.eval.models import GoldItem, load_gold

# Default gold files: the real (human-reviewed) set if present, else the
# committed example so the CLI always has something to run.
_GOLD_DIR = Path("eval/gold")
_GOLD_FILE = _GOLD_DIR / "gold_qa.jsonl"
_GOLD_EXAMPLE = _GOLD_DIR / "gold_qa.example.jsonl"
_RESULTS_DIR = Path("eval/results")


class _Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list:  # pragma: no cover - typing only
        ...


def evaluate_retrieval(
    gold: list[GoldItem], retriever: _Retriever, k: int
) -> RetrievalMetrics:
    """Evaluate ``retriever`` against ``gold`` at cutoff ``k``.

    For each gold item we call ``retriever.retrieve(item.question, k)``, collect
    the retrieved citation labels in rank order, compute the per-item metrics
    and record which expected citations hit/missed. Per-item rows are kept for
    EVERY gold item (including ``must_abstain`` / empty-expected items, so they
    stay visible in the report).

    The aggregate ``mean_*`` fields are arithmetic means computed over the
    **retrieval-scored** items only — those with a non-empty
    ``expected_citations``. Empty-expected items are excluded so their vacuous
    recall of ``1.0`` cannot inflate ``mean_recall_at_k`` while simultaneously
    deflating ``mean_mrr`` (those items are scored later by the generation /
    abstention eval in Step 7). An empty gold set, or one with no
    retrieval-scored items, yields all-zero means.
    """
    rows: list[PerItemRow] = []
    scored_rows: list[PerItemRow] = []
    for item in gold:
        results = retriever.retrieve(item.question, k)
        labels = [rc.citation_label for rc in results]
        hits = [c for c in item.expected_citations if citation_hit(c, labels[:k])]
        misses = [c for c in item.expected_citations if c not in hits]
        row = PerItemRow(
            id=item.id,
            precision=precision_at_k(item.expected_citations, labels, k),
            recall=recall_at_k(item.expected_citations, labels, k),
            citation_accuracy=citation_accuracy(item.expected_citations, labels, k),
            mrr=mrr(item.expected_citations, labels),
            hits=hits,
            misses=misses,
        )
        rows.append(row)
        if item.expected_citations:
            scored_rows.append(row)

    n_scored = len(scored_rows)

    def _mean(values: list[float]) -> float:
        return sum(values) / n_scored if n_scored else 0.0

    return RetrievalMetrics(
        k=k,
        n_items=len(rows),
        n_retrieval_items=n_scored,
        mean_precision_at_k=_mean([r.precision for r in scored_rows]),
        mean_recall_at_k=_mean([r.recall for r in scored_rows]),
        mean_citation_accuracy=_mean([r.citation_accuracy for r in scored_rows]),
        mean_mrr=_mean([r.mrr for r in scored_rows]),
        rows=rows,
    )


def format_report(metrics: RetrievalMetrics) -> str:
    """Render a readable text report: a per-item table then aggregate means."""
    lines: list[str] = []
    lines.append(
        f"Retrieval evaluation @ k={metrics.k}  "
        f"(n_items={metrics.n_items}, "
        f"retrieval-scored={metrics.n_retrieval_items})"
    )
    lines.append("=" * 72)
    # NB: 'citation_accuracy' is a RECALL measure (expected-side denominator),
    # so the column is labelled 'cit_recall(=acc)' to avoid confusion with the
    # Step 7 generation-accuracy metric.
    header = f"{'id':<10} {'precision':>10} {'recall':>8} "
    header += f"{'cit_recall(=acc)':>18} {'mrr':>6}"
    lines.append(header)
    lines.append("-" * 72)
    for r in metrics.rows:
        lines.append(
            f"{r.id:<10} {r.precision:>10.3f} {r.recall:>8.3f} "
            f"{r.citation_accuracy:>18.3f} {r.mrr:>6.3f}"
        )
        if r.misses:
            lines.append(f"           missed: {', '.join(r.misses)}")
    lines.append("-" * 72)
    lines.append(
        f"{'MEAN':<10} {metrics.mean_precision_at_k:>10.3f} "
        f"{metrics.mean_recall_at_k:>8.3f} "
        f"{metrics.mean_citation_accuracy:>18.3f} {metrics.mean_mrr:>6.3f}"
    )
    lines.append(
        f"(means over {metrics.n_retrieval_items} retrieval-scored item(s) "
        f"with non-empty expected_citations; "
        f"{metrics.n_items - metrics.n_retrieval_items} "
        "empty-expected/abstain item(s) excluded from means)"
    )
    return "\n".join(lines)


def _write_results(metrics: RetrievalMetrics) -> Path:
    """Write the metrics as a timestamped JSON artefact and return its path."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = _RESULTS_DIR / f"{stamp}.json"
    out_path.write_text(
        json.dumps(metrics.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path


def _build_real_retriever():
    """Build a HybridRetriever over the real corpus using configured embeddings.

    Imports the heavy ingestion/retrieval pieces lazily so importing this module
    (e.g. in tests) never requires Chroma/network setup.
    """
    from regulumen.core.config import get_settings
    from regulumen.core.ingestion.chunker import chunk_clauses
    from regulumen.core.ingestion.manifest import load_source_docs
    from regulumen.core.ingestion.registry import get_parser
    from regulumen.core.retrieval.bm25 import BM25Index
    from regulumen.core.retrieval.embedder import build_embedder
    from regulumen.core.retrieval.hybrid import HybridRetriever
    from regulumen.core.retrieval.vector_store import ChromaVectorStore

    settings = get_settings()
    embedder = build_embedder(settings)  # raises EmbeddingError without a token

    manifest_path = Path("corpus/manifest.json")
    raw_dir = manifest_path.parent / "raw"
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    filenames = {
        e["id"]: e.get("local_filename")
        for e in entries
        if e.get("success", False)
    }

    chunks = []
    for doc in load_source_docs(manifest_path):
        filename = filenames.get(doc.id)
        if not filename:
            continue
        parser = get_parser(doc)
        clauses = parser.parse(doc, raw_dir / filename)
        chunks.extend(chunk_clauses(clauses, doc.short_title))

    retriever = HybridRetriever(
        ChromaVectorStore(embedder, path=settings.chroma_path),
        BM25Index(),
        alpha=settings.hybrid_alpha,
    )
    retriever.index(chunks)
    return retriever, settings.top_k


def main() -> int:
    """CLI entry point. Returns a process exit code."""
    gold_path = _GOLD_FILE if _GOLD_FILE.exists() else _GOLD_EXAMPLE
    if not gold_path.exists():
        print(f"No gold file found at {_GOLD_FILE} or {_GOLD_EXAMPLE}.")
        return 1
    print(f"Loading gold set from {gold_path} ...")
    gold = load_gold(gold_path)

    try:
        retriever, k = _build_real_retriever()
    except Exception as exc:  # noqa: BLE001 - CLI must degrade gracefully
        print("Could not build the real retriever for evaluation.")
        print(f"Reason: {exc}")
        print(
            "\nReal retrieval eval needs network access and an embeddings token. "
            "Set REGULUMEN_HF_API_TOKEN and ensure corpus/ is populated, then "
            "re-run:  python -m regulumen.core.eval.runner"
        )
        # Distinct non-zero code so callers can tell "eval skipped, no infra"
        # (2) apart from "eval ran" (0) or "bad invocation / no gold" (1).
        return 2  # graceful: not a crash, just not runnable here

    metrics = evaluate_retrieval(gold, retriever, k)
    print()
    print(format_report(metrics))
    out_path = _write_results(metrics)
    print(f"\nWrote results to {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via CLI only
    sys.exit(main())
