# Gold-set schema (ReguLumen evaluation harness)

The gold set is a **JSONL** file — one JSON object per line — where each line is
one curated evaluation question. Retrieval metrics (Step 5/6) and, later,
generation + abstention metrics (Step 7) are computed against it.

> **Scope note.** This document + `gold_qa.example.jsonl` (3 example items) ship
> with the eval *framework*. The real ~30-item gold set lives in
> `gold_qa.jsonl` and is **authored separately (Step 5b) against the real
> corpus, with human review**. Do not treat the examples as ground truth.

## Fields

| field                | type        | required | default | used by            |
| -------------------- | ----------- | -------- | ------- | ------------------ |
| `id`                 | `str`       | yes      | —       | all                |
| `question`           | `str`       | yes      | —       | all                |
| `expected_citations` | `list[str]` | yes\*    | `[]`    | retrieval metrics  |
| `answer_must_contain`| `list[str]` | no       | `[]`    | generation (Step 7)|
| `must_abstain`       | `bool`      | no       | `false` | generation (Step 7)|
| `notes`              | `str`       | no       | `""`    | human reviewer     |

\* `expected_citations` may legitimately be `[]` for a `must_abstain` item
(there is nothing to retrieve). The per-item retrieval recall for an
empty-expected item is defined as `1.0` vacuously (see below), but such items
are **excluded from the aggregate retrieval means** so they neither inflate nor
deflate the reported numbers (see "Aggregate means" below). A non-abstain item
with an empty `expected_citations` is almost always an authoring mistake; the
loader emits a runtime warning for it.

### `id`
Stable identifier, e.g. `"q001"`. Never reuse an id for a different question —
result artefacts are keyed on it.

### `question`
The natural-language query sent verbatim to the retriever.

### `expected_citations` — **use citation_label form**
A list of the citation labels (or label substrings) that **should appear among
the retrieved chunks** for a faithful answer. These MUST be written in the same
form ReguLumen prints as a chunk's `citation_label`, for example:

- `"EU MDR, Annex I §17.2"`
- `"IMDRF SaMD N10, §5.1"`
- `"FDA Premarket SW, Section IV §A"`

Matching is **normalized substring** (case-insensitive, the `§` sign and light
punctuation removed, whitespace collapsed), so the three variants
`"EU MDR, Annex I §17.2"`, `"eu mdr annex i 17.2"` and `"EU MDR Annex I § 17.2"`
all match the same chunk. Digits and the dots inside clause numbers are kept, so
`§17.2` never matches `§17.3`. Prefer the **most specific** label that a correct
answer must cite; a shorter substring (e.g. just `"Annex I §17.2"`) is also
acceptable when the prefix is unambiguous.

### `answer_must_contain` *(carried for Step 7)*
Phrases the generated answer text must include. Ignored by retrieval metrics.

### `must_abstain` *(carried for Step 7)*
`true` when the correct behaviour is to **abstain** — the question asks about
something not supported by the corpus (e.g. a standard not ingested). Ignored by
retrieval metrics except that such items typically have `expected_citations: []`.

### `notes`
Free-text provenance for the human reviewer: why this is the right answer, where
the clause lives, any ambiguity. Never read by metrics.

## Retrieval-metric edge-case conventions

- **precision@k** — fraction of the top-k *retrieved* labels matching any
  expected citation. No retrieved labels or no expected citations → `0.0`.
- **recall@k / citation_accuracy** — fraction of *expected* citations found in
  the top-k. **No expected citations → `1.0` vacuously** (nothing to find,
  nothing missed). `citation_accuracy` is an **alias of `recall@k`** — it is a
  **RECALL** measure (denominator is the number of *expected* citations), **not
  a generation-precision metric**, and is unrelated to the Step 7 generation /
  answer-accuracy eval. Reports label the column `cit_recall(=acc)` for clarity.
- **mrr** — reciprocal of the 1-based rank of the first retrieved label that
  hits any expected citation; `0.0` if none.

### Aggregate means (over retrieval-scored items only)

The report's `mean_*` numbers are averaged over the **retrieval-scored** items
only — gold items with a **non-empty `expected_citations`** list.
`must_abstain` / empty-expected items are **excluded from the retrieval means**
(they will be scored by the generation / abstention eval in Step 7), so a
vacuous per-item recall of `1.0` can no longer inflate `mean_recall_at_k` while
the same item deflates `mean_mrr`. The report still prints a per-item row for
every item, plus two counts: `n_items` (all items) and `n_retrieval_items` (the
non-empty-expected items the means are computed over).

### Clause-number boundary matching

Matching is **boundary-aware**: a normalized expected citation only matches a
retrieved label when the occurrence is not flanked by a digit or a dot on either
side. So `§17.1` does **not** match `§17.10` or `§117.1` (sibling sub-clauses),
and a short expected like `§17.2` does not prefix-match a bare `17`.

## Examples

```json
{"id": "q001", "question": "What does EU MDR require for medical device software lifecycle and development?", "expected_citations": ["EU MDR, Annex I §17.2"], "answer_must_contain": ["development life cycle", "risk management"], "must_abstain": false, "notes": "Annex I, General safety & performance requirement 17.2 covers software lifecycle."}
{"id": "q002", "question": "Does this corpus define the requirements of ISO 14971 clause 4 for the risk management file?", "expected_citations": [], "answer_must_contain": [], "must_abstain": true, "notes": "ISO 14971 is a copyrighted standard NOT in the corpus; the assistant must abstain rather than fabricate a citation."}
```
