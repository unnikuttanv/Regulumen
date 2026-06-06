"""Pure retrieval-quality metrics for the ReguLumen evaluation harness.

These functions are the correctness centerpiece of the eval step: they are
**pure** (no I/O, no global state) so every edge case is unit-testable
offline. Each definition and its edge-case convention is documented inline so
the numbers in a report are never ambiguous.

The objects being measured are CITATION LABELS — the human-readable strings
attached to every retrieved chunk, e.g. ``"EU MDR, Annex I §17.2"`` or
``"IMDRF SaMD N10, §5.1"``. A gold item's ``expected_citations`` lists the
labels (or label substrings) that SHOULD appear among the retrieved chunks for
a faithful answer. Matching is by normalized, **boundary-aware** substring (see
:func:`citation_hit` / :func:`_boundary_match`) so a clause number never matches
a sibling sub-clause (``"17.1"`` vs ``"17.10"``).

Edge-case conventions (chosen once, applied everywhere):
    * **precision@k**: fraction of the top-k *retrieved* labels that match any
      expected citation. Denominator is ``min(k, len(retrieved))``. With no
      retrieved labels -> ``0.0``. With no expected citations there is nothing
      a label can be "relevant" to, so -> ``0.0``.
    * **recall@k / citation_accuracy**: fraction of *expected* citations found
      within the top-k retrieved labels. With NO expected citations recall is
      ``1.0`` *vacuously* (a question with nothing to find — e.g. a
      ``must_abstain`` item — cannot miss). This mirrors the standard
      "vacuous truth" convention. NOTE the per-item value is still ``1.0`` for
      such items, but the **aggregate means exclude empty-expected items**
      entirely (see :func:`~regulumen.core.eval.runner.evaluate_retrieval`) so
      vacuous recall cannot inflate the reported mean.
    * **mrr**: reciprocal of the rank (1-based) of the FIRST retrieved label
      that hits ANY expected citation; ``0.0`` if none hit (or either list is
      empty, including empty expected).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# The section sign and a handful of punctuation marks that decorate citation
# labels but carry no disambiguating information once whitespace is normalized.
# We deliberately KEEP digits, letters and the dots inside clause numbers
# ("17.2") so different clauses never collapse together.
_PUNCT_RE = re.compile(r"[§,;:()\[\]–—]")  # § , ; : ( ) [ ] – —
_WS_RE = re.compile(r"\s+")


def normalize_citation(s: str) -> str:
    """Normalize a citation label for tolerant comparison.

    Steps (conservative on purpose — we never drop digits or the dots that
    separate clause numbers, so "§17.2" and "§17.3" stay distinct):

    1. lower-case;
    2. drop the section sign ``§`` and light punctuation (``, ; : ( ) [ ] – —``);
    3. collapse all whitespace runs to a single space and strip ends.

    As a result these all normalize to ``"eu mdr annex i 17.2"``::

        "EU MDR, Annex I §17.2"
        "eu mdr annex i 17.2"
        "EU MDR  Annex I  § 17.2"
    """
    lowered = s.lower()
    de_punct = _PUNCT_RE.sub(" ", lowered)
    return _WS_RE.sub(" ", de_punct).strip()


# Below this normalized length an expected citation is treated as a "short"
# token (e.g. an article number) for which we also allow it to *contain* a
# retrieved label, not just be contained by one. Keeps tiny expecteds from
# matching everything while still being forgiving.
_SHORT_EXPECTED_LEN = 6


def _boundary_match(needle: str, haystack: str) -> bool:
    """True if ``needle`` occurs in ``haystack`` not flanked by a digit or dot.

    This is the boundary-aware replacement for a bare substring test. A clause
    number like ``"17.1"`` must NOT match inside ``"17.10"`` or ``"117.1"`` —
    that would award false credit to a *sibling* sub-clause. We require that the
    occurrence is not immediately preceded or followed by a digit ``[0-9]`` or a
    dot ``.`` on either side, so ``"17.1"`` matches ``"17.1"`` and
    ``"17.1 heading"`` but never ``"17.10"`` or ``"117.1"``.
    """
    if not needle:
        return False
    return (
        re.search(r"(?<![0-9.])" + re.escape(needle) + r"(?![0-9.])", haystack)
        is not None
    )


def citation_hit(expected: str, retrieved_labels: list[str]) -> bool:
    """Return ``True`` if ``expected`` matches any of ``retrieved_labels``.

    Matching is normalized + **boundary-aware**: ``expected`` hits if its
    normalized form occurs inside a normalized retrieved label *without* being
    flanked by a digit or a dot (see :func:`_boundary_match`). This is the
    natural direction because gold expecteds are often *substrings* of the
    fuller on-chunk label (e.g. expected ``"EU MDR, Annex I §17.2"`` vs a
    retrieved label that appends a heading) — but the boundary guard stops a
    clause number from matching a sibling sub-clause (``"17.1"`` vs ``"17.10"``).

    For *short* expecteds (normalized length <= 6, e.g. ``"§10"``) we also
    accept the reverse — a retrieved label occurring (with the same boundary
    guard) inside the expected — since such tiny tokens are otherwise too
    brittle. The boundary guard means a short ``"17.2"`` will NOT prefix-match a
    bare ``"17"``. The empty expected never hits.
    """
    norm_expected = normalize_citation(expected)
    if not norm_expected:
        return False
    short = len(norm_expected) <= _SHORT_EXPECTED_LEN
    for label in retrieved_labels:
        norm_label = normalize_citation(label)
        if not norm_label:
            continue
        if _boundary_match(norm_expected, norm_label):
            return True
        if short and _boundary_match(norm_label, norm_expected):
            return True
    return False


def precision_at_k(
    expected_citations: list[str], retrieved_labels: list[str], k: int
) -> float:
    """Fraction of the top-k retrieved labels that match any expected citation.

    Denominator is ``min(k, len(retrieved_labels))``. Returns ``0.0`` when no
    labels are retrieved (or ``k <= 0``), and ``0.0`` when there are no
    expected citations (nothing for a label to be relevant to).
    """
    if k <= 0 or not retrieved_labels or not expected_citations:
        return 0.0
    top = retrieved_labels[:k]
    hits = sum(1 for label in top if _label_matches_any(label, expected_citations))
    return hits / len(top)


def recall_at_k(
    expected_citations: list[str], retrieved_labels: list[str], k: int
) -> float:
    """Fraction of expected citations found within the top-k retrieved labels.

    No expected citations -> ``1.0`` (vacuous: nothing to find, nothing
    missed). Otherwise the count of expected citations that hit, divided by the
    number of expected citations.
    """
    if not expected_citations:
        return 1.0
    if k <= 0 or not retrieved_labels:
        return 0.0
    top = retrieved_labels[:k]
    hits = sum(1 for exp in expected_citations if citation_hit(exp, top))
    return hits / len(expected_citations)


def citation_accuracy(
    expected_citations: list[str], retrieved_labels: list[str], k: int
) -> float:
    """Citation accuracy — the README metric — is recall@k by definition.

    It is ``1.0`` when ALL expected citations are hit within the top-k, else
    ``n_hit / n_expected``. That is exactly :func:`recall_at_k`; this is a named
    alias so reports can speak of "citation accuracy" without a second formula.
    The two will always return identical values for identical inputs.

    .. warning::
        Despite the name, this is a **RECALL** measure — its denominator is the
        number of *expected* citations (the expected/gold side), NOT the number
        of retrieved or generated citations. It answers "what fraction of the
        citations a faithful answer must surface did retrieval find?" It is
        **NOT** a generation-precision metric and is entirely **unrelated to the
        Step 7 generation/answer-accuracy eval**. Read every "citation_accuracy"
        in a retrieval report as "citation recall".
    """
    return recall_at_k(expected_citations, retrieved_labels, k)


def mrr(expected_citations: list[str], retrieved_labels: list[str]) -> float:
    """Reciprocal rank of the first retrieved label that hits any expected.

    Ranks are 1-based, so a hit at the top yields ``1.0``, at rank 2 ``0.5``,
    etc. Returns ``0.0`` if no retrieved label hits, or if either list is
    empty.
    """
    if not expected_citations or not retrieved_labels:
        return 0.0
    for rank, label in enumerate(retrieved_labels, start=1):
        if _label_matches_any(label, expected_citations):
            return 1.0 / rank
    return 0.0


def _label_matches_any(label: str, expected_citations: list[str]) -> bool:
    """True if a single retrieved ``label`` satisfies any expected citation.

    Defined in terms of :func:`citation_hit` (which treats its first arg as the
    expected) so precision/mrr share the exact matching semantics of recall.
    """
    return any(citation_hit(exp, [label]) for exp in expected_citations)


@dataclass
class PerItemRow:
    """Per-gold-item metric row, surfaced for inspection and JSON export."""

    id: str
    precision: float
    recall: float
    citation_accuracy: float
    mrr: float
    hits: list[str] = field(default_factory=list)
    misses: list[str] = field(default_factory=list)


@dataclass
class RetrievalMetrics:
    """Aggregate retrieval metrics over a gold set, plus per-item rows.

    The ``mean_*`` fields are arithmetic means computed over the
    **retrieval-scored** items only — i.e. gold items with a non-empty
    ``expected_citations`` list. ``k`` records the cutoff the metrics were
    computed at, ``n_items`` is the number of ALL gold items evaluated (per-item
    rows are kept for every item, including ``must_abstain`` / empty-expected
    ones), and ``n_retrieval_items`` is the number of retrieval-scored items the
    means are averaged over. Empty-expected items are excluded from the means so
    a vacuous recall of ``1.0`` cannot inflate ``mean_recall_at_k`` while the
    same item deflates ``mean_mrr``.
    """

    k: int
    n_items: int
    n_retrieval_items: int
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_citation_accuracy: float
    mean_mrr: float
    rows: list[PerItemRow] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serializable view for writing JSON result artefacts."""
        return {
            "k": self.k,
            "n_items": self.n_items,
            "n_retrieval_items": self.n_retrieval_items,
            "mean_precision_at_k": self.mean_precision_at_k,
            "mean_recall_at_k": self.mean_recall_at_k,
            "mean_citation_accuracy": self.mean_citation_accuracy,
            "mean_mrr": self.mean_mrr,
            "rows": [
                {
                    "id": r.id,
                    "precision": r.precision,
                    "recall": r.recall,
                    "citation_accuracy": r.citation_accuracy,
                    "mrr": r.mrr,
                    "hits": r.hits,
                    "misses": r.misses,
                }
                for r in self.rows
            ],
        }
