# Gold-set authoring notes (Step 5b)

This file documents the real, human-reviewable gold set in `gold_qa.jsonl`
(30 items) authored against the actual on-disk corpus. It is NOT the synthetic
`gold_qa.example.jsonl` shipped with the framework.

## How it was built

1. `scripts/dump_chunks.py` parses every successful source in
   `corpus/manifest.json` with its registered parser and chunks the clauses
   offline (no network/embeddings), writing every chunk's `citation_label`,
   `doc_id` and a 200-char preview to `_chunk_inventory.jsonl`.
2. Questions were authored so each `expected_citation` is an exact (or clearly
   substring) real `citation_label` from that inventory.
3. `scripts/validate_gold.py` re-checks grounding with the harness's own
   `citation_hit` matcher.

## Chunk inventory (938 chunks total)

| source                              | short_title             | chunks |
| ----------------------------------- | ----------------------- | ------ |
| eu-mdr-2017-745                     | EU MDR                  | 837    |
| fda-premarket-software-functions    | FDA Premarket SW        | 69     |
| fda-samd-clinical-evaluation        | FDA SaMD Clinical Eval  | 16     |
| imdrf-samd-key-definitions          | IMDRF SaMD N10          | 16     |

## Counts by category

- **30** gold items total.
- **24** retrieval-answerable (`must_abstain=false`, non-empty expected_citations).
- **6** must_abstain (out-of-scope; expected_citations = []).
- **4** JUDGMENT CALL items flagged for the reviewer (see below); 3 of these are
  also among the 24 retrieval items, 1 (q030) is a retrieval item too.

### Source coverage of the 24 retrieval items

- **EU MDR** (well-parsed): q001-q013, q022, q023, q030 — Annex I software GSPRs
  (17.1/17.2/17.4), general safety GSPRs (Chapter I 1/2), and Articles 2, 10, 15,
  27, 52, 61, 62, 83, 87, 89, 120. Mix of clause-number lookups, conceptual, and
  definition questions.
- **IMDRF SaMD N10** (clean dotted sections): q014-q019 — SaMD definition,
  medical-device definition, SaMD Changes, manufacturer, scope, introduction.
- **FDA** (degenerate/partial parses): q020 (FDA Premarket SW, documentation
  level), q021 (FDA SaMD Clinical Eval, validation example scenarios). Kept
  deliberately few and flagged because these parses are low quality.

### must_abstain items (out of corpus scope)

- q024 — IEC 62304 §5.1.1 (copyrighted standard, not ingested).
- q025 — ISO 13485 management review (copyrighted standard, not ingested).
- q026 — ISO 14971 risk management file (copyrighted standard, not ingested).
- q027 — FD&C Act §524B cybersecurity (statute/guidance not in corpus).
- q028 — UK MHRA / UK MDR 2002 (different jurisdiction, not in corpus).
- q029 — GDPR patient data (data-protection law, not in corpus).

## Validation result

`scripts/validate_gold.py` (uses `regulumen.core.eval.metrics.citation_hit`):

```
Gold items: 30 (6 abstain, 24 retrieval)
Expected citations grounded: 24/24
All expected citations are grounded in the real chunk inventory.
```

**24/24 (100%)** of non-abstain expected_citations match a real chunk label.
NOTE: this checks *grounding only*. Empirical *retrievability* (do embeddings
actually surface these chunks at top-k) is a later step requiring an embeddings
token that is not set here.

## JUDGMENT CALL items — please scrutinise

- **q014** (IMDRF SaMD definition). The natural citation `§5.1` is not a distinct
  label in the corpus; the 5.1 SaMD definition lives inside the section-5 heading
  chunk `IMDRF SaMD N10, §5`. We cite `§5` as the closest real anchor whose text
  opens with the 5.1 definition. Reviewer: accept `§5` or insist on finer
  granularity?
- **q020** (FDA Premarket SW documentation level). This PDF degraded to per-page
  fallback (`§p.N`) and the extracted text is actually the *webinar transcript*,
  not the guidance's normative sections. The cited page (p.4) does discuss
  Basic vs Enhanced documentation level, but the citation precision is weak.
  **UPDATE (2026-06-03):** live testing showed this source's noisy chunks caused
  the generator to over-abstain even on good EU MDR questions, so it is now
  dropped at ingest by `min_structure_confidence` (its confidence is 0.0). q020
  is therefore **expected to fail / abstain** until a cleaner FDA Premarket
  source is sourced; it stays in the gold set as a tracked known-gap rather than
  being deleted. See `docs/eval-tuning-notes.md`.
- **q021** (FDA SaMD Clinical Eval). Parsed into only three Roman-numeral
  "sections" (`§1`/`§2`/`§3`), which are worked AI/ML example scenarios rather
  than the guidance's real section numbering. The cited `§3` discusses
  analytical/clinical validation, but the label does not map to a true section.
- **q030** (EU MDR Article 2 'medical device' definition). The parser kept only
  the Article 2 heading chunk (text just "Definitions") and dropped the numbered
  definition sub-clauses, so the cited chunk anchors the right Article but lacks
  the actual definition wording. IMDRF §5.2.1 (q015) is the better-grounded
  source for the definition text. Reviewer: keep the heading-only citation, or
  rework/remove this item?
