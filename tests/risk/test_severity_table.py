"""Exhaustive tests for the IEC 62304 §4.3 severity → safety-class table.

The reviewer-of-record audits ``severity_table.py`` and relies on these tests
to prove the encoded decision flow matches the standard. Every branch of the
table is covered explicitly.
"""

from __future__ import annotations

import pytest

from regulumen.core.risk.severity_table import (
    SEVERITY_TO_CLASS,
    SafetyClass,
    Severity,
    classify,
)


def test_cannot_contribute_is_class_a_regardless_of_severity():
    """Step 1 of §4.3: if the software cannot contribute to a hazardous
    situation, it is Class A no matter what severity is supplied."""
    for severity in Severity:
        assert classify(False, severity) is SafetyClass.A


def test_can_contribute_none_severity_is_class_a():
    assert classify(True, Severity.NONE) is SafetyClass.A


def test_can_contribute_non_serious_is_class_b():
    assert classify(True, Severity.NON_SERIOUS) is SafetyClass.B


def test_can_contribute_serious_is_class_c():
    assert classify(True, Severity.SERIOUS) is SafetyClass.C


def test_can_contribute_death_is_class_c():
    assert classify(True, Severity.DEATH) is SafetyClass.C


def test_external_control_reduces_death_to_non_serious_is_class_b():
    """External (non-software) risk control lowers the EFFECTIVE severity that
    the table consumes: worst case death, but a hardware control caps the
    residual harm at a non-serious injury → Class B."""
    result = classify(
        True,
        Severity.DEATH,
        external_control_reduces_to=Severity.NON_SERIOUS,
    )
    assert result is SafetyClass.B


def test_external_control_reduces_serious_to_none_is_class_a():
    result = classify(
        True,
        Severity.SERIOUS,
        external_control_reduces_to=Severity.NONE,
    )
    assert result is SafetyClass.A


def test_external_control_cannot_raise_severity_raises():
    """Guard: controls reduce, never raise. before=NON_SERIOUS but
    reduces_to=DEATH is incoherent and must raise ValueError."""
    with pytest.raises(ValueError):
        classify(
            True,
            Severity.NON_SERIOUS,
            external_control_reduces_to=Severity.DEATH,
        )


def test_external_control_equal_severity_is_allowed():
    """Reducing to the same severity is a no-op, not a violation."""
    result = classify(
        True,
        Severity.SERIOUS,
        external_control_reduces_to=Severity.SERIOUS,
    )
    assert result is SafetyClass.C


def test_severity_to_class_covers_all_four_severities():
    assert set(SEVERITY_TO_CLASS) == set(Severity)
    assert SEVERITY_TO_CLASS[Severity.NONE] is SafetyClass.A
    assert SEVERITY_TO_CLASS[Severity.NON_SERIOUS] is SafetyClass.B
    assert SEVERITY_TO_CLASS[Severity.SERIOUS] is SafetyClass.C
    assert SEVERITY_TO_CLASS[Severity.DEATH] is SafetyClass.C
