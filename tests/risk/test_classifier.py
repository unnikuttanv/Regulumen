"""Tests for the hybrid (rules + LLM) IEC 62304 risk classifier.

All tests run offline using :class:`FakeLLM` and synthetic, IP-safe component
descriptions. No IEC 62304 text is reproduced.
"""

from __future__ import annotations

from regulumen.core.generation.llm import FakeLLM
from regulumen.core.risk.classifier import RiskClassifier
from regulumen.core.risk.severity_table import SafetyClass, Severity


def _classifier(response):
    return RiskClassifier(FakeLLM(response))


def test_grounded_extraction_serious_to_class_c():
    response = (
        '{"can_contribute_to_hazard": true, "worst_case_severity": "serious", '
        '"external_control_reduces_to": null, "needs_more_info": false}'
    )
    result = _classifier(response).classify(
        "Module that adjusts an infusion rate based on a sensor reading."
    )
    assert result.abstained is False
    assert result.safety_class is SafetyClass.C
    assert result.severity is Severity.SERIOUS
    assert result.can_contribute is True
    assert "serious" in result.rationale.lower()
    assert "4.3" in result.rationale
    assert result.disclaimer
    assert "not a regulatory verdict" in result.disclaimer.lower()


def test_external_control_downgrades_death_to_class_b():
    response = (
        '{"can_contribute_to_hazard": true, "worst_case_severity": "death", '
        '"external_control_reduces_to": "non_serious", "needs_more_info": false}'
    )
    result = _classifier(response).classify(
        "Controller whose worst-case failure is bounded by an independent "
        "hardware cutoff."
    )
    assert result.abstained is False
    assert result.safety_class is SafetyClass.B
    assert result.disclaimer


def test_cannot_contribute_is_class_a():
    response = (
        '{"can_contribute_to_hazard": false, "worst_case_severity": "none", '
        '"external_control_reduces_to": null, "needs_more_info": false}'
    )
    result = _classifier(response).classify(
        "A read-only report viewer with no control over therapy."
    )
    assert result.abstained is False
    assert result.safety_class is SafetyClass.A
    assert result.can_contribute is False
    assert result.disclaimer


def test_vague_description_unknown_severity_abstains():
    """Money test: severity unknown → ABSTAIN, no class guessed."""
    response = (
        '{"can_contribute_to_hazard": true, "worst_case_severity": "unknown", '
        '"external_control_reduces_to": null, "needs_more_info": true}'
    )
    result = _classifier(response).classify("Some software for a device.")
    assert result.abstained is True
    assert result.safety_class is None
    assert result.severity is None
    # rationale must name what is missing.
    assert "severity" in result.rationale.lower()
    assert result.disclaimer


def test_needs_more_info_flag_abstains_even_with_severity():
    response = (
        '{"can_contribute_to_hazard": true, "worst_case_severity": "serious", '
        '"external_control_reduces_to": null, "needs_more_info": true}'
    )
    result = _classifier(response).classify("Ambiguous component.")
    assert result.abstained is True
    assert result.safety_class is None


def test_malformed_json_abstains_not_crash():
    response = "I think this is probably Class B but I am not sure, sorry."
    result = _classifier(response).classify("Whatever.")
    assert result.abstained is True
    assert result.safety_class is None
    assert result.disclaimer


def test_missing_keys_abstains():
    response = '{"can_contribute_to_hazard": true}'
    result = _classifier(response).classify("Partial extraction.")
    assert result.abstained is True
    assert result.safety_class is None


def test_json_in_code_fences_is_parsed():
    response = (
        "Here is the extraction:\n```json\n"
        '{"can_contribute_to_hazard": true, "worst_case_severity": "non_serious", '
        '"external_control_reduces_to": null, "needs_more_info": false}\n'
        "```\nThat is my analysis."
    )
    result = _classifier(response).classify("Component with minor injury risk.")
    assert result.abstained is False
    assert result.safety_class is SafetyClass.B


def test_disclaimer_present_on_abstain_and_on_result():
    abstain = _classifier("garbage").classify("x")
    decided = _classifier(
        '{"can_contribute_to_hazard": true, "worst_case_severity": "none", '
        '"external_control_reduces_to": null, "needs_more_info": false}'
    ).classify("y")
    assert abstain.disclaimer
    assert decided.disclaimer
    assert abstain.disclaimer == decided.disclaimer


def test_extracted_fields_exposed_for_transparency():
    response = (
        '{"can_contribute_to_hazard": true, "worst_case_severity": "serious", '
        '"external_control_reduces_to": null, "needs_more_info": false}'
    )
    result = _classifier(response).classify("z")
    assert result.extracted["worst_case_severity"] == "serious"
    assert result.extracted["can_contribute_to_hazard"] is True


def test_retriever_citations_attached_when_provided():
    from regulumen.core.retrieval.vector_store import RetrievedChunk

    chunk = RetrievedChunk(
        id="c1",
        text="related guidance",
        citation_label="Guide §1",
        doc_id="d1",
        clause_path=["1"],
        score=0.9,
    )

    class _Retriever:
        def retrieve(self, query, k):
            return [chunk]

    response = (
        '{"can_contribute_to_hazard": true, "worst_case_severity": "serious", '
        '"external_control_reduces_to": null, "needs_more_info": false}'
    )
    clf = RiskClassifier(FakeLLM(response), retriever=_Retriever(), top_k=3)
    result = clf.classify("anything")
    assert result.citations == [chunk]


def test_no_retriever_yields_empty_citations():
    response = (
        '{"can_contribute_to_hazard": true, "worst_case_severity": "serious", '
        '"external_control_reduces_to": null, "needs_more_info": false}'
    )
    result = _classifier(response).classify("anything")
    assert result.citations == []


def test_external_control_higher_than_severity_abstains_not_crash():
    """If the LLM emits an incoherent control (raises severity), the table
    guard raises ValueError; the classifier must abstain, not crash."""
    response = (
        '{"can_contribute_to_hazard": true, "worst_case_severity": "non_serious", '
        '"external_control_reduces_to": "death", "needs_more_info": false}'
    )
    result = _classifier(response).classify("incoherent control")
    assert result.abstained is True
    assert result.safety_class is None
