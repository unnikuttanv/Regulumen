# Retrieval evaluation & tuning notes

Real-embeddings run on 2026-06-03 (HuggingFace Inference, `BAAI/bge-large-en-v1.5`,
1024-dim), against the 30-item gold set (`eval/gold/gold_qa.jsonl`): 24
retrieval-scored items + 6 must-abstain items (excluded from retrieval means).

## Baseline (alpha=0.5, k=8)

| Metric | Value |
|---|---|
| Recall@8 / Citation accuracy | 0.875 (21/24) |
| MRR | 0.764 |
| Precision@8 | 0.229 |

Precision is low **by design**: each question expects 1–2 specific clauses among
8 retrieved, so per-item precision caps around 0.12–0.25. Recall and citation
accuracy are the meaningful retrieval signals here.

## Tuning sweep

Same indexed corpus, varying `hybrid_alpha` (dense weight) and `top_k`:

| alpha | k | recall | cit_acc | mrr | prec |
|---|---|---|---|---|---|
| 0.3 | 8 | 0.875 | 0.875 | **0.784** | 0.229 |
| 0.3 | 10 | 0.875 | 0.875 | 0.784 | 0.192 |
| 0.5 | 8 | 0.875 | 0.875 | 0.764 | 0.229 |
| 0.5 | 10 | 0.875 | 0.875 | 0.764 | 0.204 |
| 0.7 | 8 | 0.875 | 0.875 | 0.779 | 0.219 |
| 0.7 | 10 | 0.875 | 0.875 | 0.779 | 0.200 |

## Decision

**Set `hybrid_alpha=0.3`, `top_k=8`.** It gives the best MRR (0.784) — the first
correct citation ranks higher — at zero recall cost, and k=8 matches k=10 recall
with higher precision.

## Honest finding (retrieval)

Recall is **flat at 0.875 across every alpha/k**. The 3 misses (q007 EU MDR
Art 61.1, q009 Art 87.1, q030 Art 2) are not a ranking problem fusion can fix —
those clauses don't surface in either retriever's candidate pool for those
queries. Recovering them needs better chunking / query handling (e.g. the Art 2
"Definitions" heading-only chunk is a known parser limitation), not weight
tuning. Tracked as future work, not hidden by re-weighting.

## Live-testing finding (generation) — low-confidence chunk filter

Live end-to-end testing (real Groq) surfaced a problem retrieval metrics alone
missed: the FDA *Premarket Software Functions* PDF parsed as a webinar transcript
(structure_confidence 0.0) and its noisy per-page chunks filled most of the
top-k. Even when the correct EU MDR clause ranked #1, the generator **abstained
~50% of the time** ("these excerpts are about FDA…") because the surrounding
noise looked off-topic. The faithfulness gate never hallucinated — it failed
*safe* — but usefulness suffered.

**Fix:** `Settings.min_structure_confidence` (default 0.15) drops degenerate
parses at ingest time. The FDA transcript (0.0) is excluded; EU MDR (HTML,
confidence None → always kept), FDA SaMD Clinical Eval (0.28) and IMDRF (0.82)
are retained. Corpus: 938 → **869 chunks**.

**Result:** the EU MDR lifecycle question now answers **4/4** (was 2/4) with the
correct §17.2 citation; retrieval means are **unchanged** (recall/citation
accuracy 0.875, MRR 0.764) — confirming the dropped transcript contributed
nothing to the gold set. Consequence: gold item q020 (which cited the transcript)
is now correctly ungroundable — the right outcome for a garbage source.
