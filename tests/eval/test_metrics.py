"""Unit tests for the pure retrieval metrics.

Every case uses hand-built label lists where the correct numeric answer is
known by inspection, so a failure points straight at a metric bug rather than
at a flaky retriever.
"""

from __future__ import annotations

import math

from regulumen.core.eval.metrics import (
    citation_accuracy,
    citation_hit,
    mrr,
    normalize_citation,
    precision_at_k,
    recall_at_k,
)

# --- normalize_citation --------------------------------------------------


def test_normalize_collapses_whitespace_and_lowercases():
    assert normalize_citation("  EU   MDR ") == "eu mdr"


def test_normalize_three_spelling_variants_compare_equal():
    variants = [
        "EU MDR, Annex I §17.2",
        "eu mdr annex i 17.2",
        "EU MDR  Annex I  § 17.2",
    ]
    normalized = {normalize_citation(v) for v in variants}
    assert len(normalized) == 1


def test_normalize_does_not_collide_different_clauses():
    a = normalize_citation("EU MDR, Annex I §17.2")
    b = normalize_citation("EU MDR, Annex I §17.3")
    assert a != b


def test_normalize_section_sign_optional():
    assert normalize_citation("IMDRF SaMD N10, §5.1") == normalize_citation(
        "imdrf samd n10 5.1"
    )


# --- citation_hit --------------------------------------------------------


def test_citation_hit_substring_match():
    retrieved = ["EU MDR, Annex I §17.2 — software lifecycle"]
    assert citation_hit("EU MDR, Annex I §17.2", retrieved) is True


def test_citation_hit_spelling_variant_match():
    retrieved = ["EU MDR  Annex I  § 17.2"]
    assert citation_hit("eu mdr annex i 17.2", retrieved) is True


def test_citation_hit_no_match():
    retrieved = ["IMDRF SaMD N10, §5.1"]
    assert citation_hit("EU MDR, Annex I §17.2", retrieved) is False


def test_citation_hit_empty_retrieved():
    assert citation_hit("EU MDR, Annex I §17.2", []) is False


# --- citation_hit: clause-number boundary regressions --------------------


def test_citation_hit_does_not_match_sibling_subclause_17_1_vs_17_10():
    # "17.1" must NOT match "17.10" — that would award false credit to a sibling.
    assert (
        citation_hit("EU MDR, Annex I §17.1", ["EU MDR, Annex I §17.10 — heading"])
        is False
    )


def test_citation_hit_does_not_match_leading_digit_117_1():
    assert citation_hit("§17.1", ["§117.1 — clause"]) is False


def test_citation_hit_short_reverse_does_not_prefix_match():
    # Short reverse direction must not let "17.2" match a bare "17".
    assert citation_hit("§17.2", ["17"]) is False


def test_citation_hit_exact_still_hits():
    assert (
        citation_hit("EU MDR, Annex I §17.1", ["EU MDR, Annex I §17.1 — lifecycle"])
        is True
    )


def test_citation_hit_multi_digit_exact_still_hits():
    assert citation_hit("§17.10", ["EU MDR, Annex I §17.10 — heading"]) is True


# --- precision_at_k ------------------------------------------------------


def test_precision_perfect():
    expected = ["EU MDR, Annex I §17.2"]
    retrieved = ["EU MDR, Annex I §17.2"]
    assert precision_at_k(expected, retrieved, k=1) == 1.0


def test_precision_partial():
    # 2 of the top-4 retrieved labels match an expected citation.
    expected = ["A §1", "B §2"]
    retrieved = ["A §1", "Z §9", "B §2", "Y §8"]
    assert precision_at_k(expected, retrieved, k=4) == 0.5


def test_precision_zero_hit():
    expected = ["A §1"]
    retrieved = ["Z §9", "Y §8"]
    assert precision_at_k(expected, retrieved, k=2) == 0.0


def test_precision_respects_k_truncation():
    expected = ["A §1"]
    retrieved = ["Z §9", "A §1"]  # the hit is at rank 2
    assert precision_at_k(expected, retrieved, k=1) == 0.0
    assert precision_at_k(expected, retrieved, k=2) == 0.5


def test_precision_empty_retrieved_is_zero():
    assert precision_at_k(["A §1"], [], k=5) == 0.0


def test_precision_empty_expected_is_zero():
    # No expected citations -> nothing retrieved can be "relevant".
    assert precision_at_k([], ["A §1"], k=5) == 0.0


# --- recall_at_k ---------------------------------------------------------


def test_recall_perfect():
    expected = ["A §1", "B §2"]
    retrieved = ["A §1", "B §2"]
    assert recall_at_k(expected, retrieved, k=2) == 1.0


def test_recall_partial():
    expected = ["A §1", "B §2"]
    retrieved = ["A §1", "Z §9"]
    assert recall_at_k(expected, retrieved, k=2) == 0.5


def test_recall_respects_k_truncation():
    expected = ["A §1", "B §2"]
    retrieved = ["A §1", "Z §9", "B §2"]  # B is at rank 3
    assert recall_at_k(expected, retrieved, k=2) == 0.5
    assert recall_at_k(expected, retrieved, k=3) == 1.0


def test_recall_empty_expected_is_vacuously_one():
    assert recall_at_k([], ["A §1"], k=5) == 1.0


def test_recall_zero_hit():
    assert recall_at_k(["A §1"], ["Z §9"], k=5) == 0.0


# --- citation_accuracy ---------------------------------------------------


def test_citation_accuracy_equals_recall():
    expected = ["A §1", "B §2"]
    retrieved = ["A §1", "Z §9", "B §2"]
    for k in (1, 2, 3):
        assert citation_accuracy(expected, retrieved, k) == recall_at_k(
            expected, retrieved, k
        )


def test_citation_accuracy_all_hits_is_one():
    expected = ["A §1", "B §2"]
    retrieved = ["A §1", "B §2"]
    assert citation_accuracy(expected, retrieved, k=2) == 1.0


def test_citation_accuracy_empty_expected_vacuously_one():
    assert citation_accuracy([], ["A §1"], k=5) == 1.0


# --- mrr -----------------------------------------------------------------


def test_mrr_first_rank():
    assert mrr(["A §1"], ["A §1", "B §2"]) == 1.0


def test_mrr_second_rank():
    assert mrr(["A §1"], ["Z §9", "A §1"]) == 0.5


def test_mrr_third_rank():
    assert math.isclose(mrr(["A §1"], ["Z §9", "Y §8", "A §1"]), 1.0 / 3.0)


def test_mrr_any_expected_counts():
    # First label that hits ANY expected citation drives the reciprocal rank.
    expected = ["A §1", "B §2"]
    retrieved = ["Z §9", "B §2", "A §1"]
    assert mrr(expected, retrieved) == 0.5


def test_mrr_no_hit_is_zero():
    assert mrr(["A §1"], ["Z §9", "Y §8"]) == 0.0


def test_mrr_empty_retrieved_is_zero():
    assert mrr(["A §1"], []) == 0.0


def test_mrr_empty_expected_is_zero():
    assert mrr([], ["A §1"]) == 0.0
