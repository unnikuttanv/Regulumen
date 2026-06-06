"""IEC 62304 §4.3 software safety classification table (DECISION SUPPORT).

=============================================================================
AUDIENCE: This file is the auditable centerpiece of the safety-class feature.
It is meant to be READ AND VERIFIED BY A QUALIFIED DOMAIN EXPERT (the
reviewer-of-record), who need not be a programmer. It is therefore written as
plain, transparent, pure DATA plus one small pure FUNCTION — no cleverness, no
hidden state, no I/O. A subtly wrong rule here is worse than abstaining, so
every rule is spelled out and exhaustively tested in
``tests/risk/test_severity_table.py``.

INTELLECTUAL PROPERTY NOTE: This module encodes the DECISION LOGIC of the
IEC 62304 software safety classification so a tool can offer decision support.
It does NOT reproduce the copyrighted text of the IEC 62304 standard. The
standard itself is the authoritative source; consult your licensed copy.

DISCLAIMER: This encoding is DECISION SUPPORT ONLY. It is NOT a regulatory
verdict. Any classification produced from this table must be reviewed and
confirmed by a qualified person, per IEC 62304.
=============================================================================

THE STANDARD'S DECISION FLOW (IEC 62304 §4.3, as amended by Amendment 1:2015)
----------------------------------------------------------------------------
Each software system / software item is assigned a safety class based on the
POSSIBLE HARM that could result from a hazard to which the software can
contribute. The classification is made:

  * assuming the hazardous situation actually leads to harm, and
  * assuming the WORST-CASE severity of that harm, and
  * AFTER taking account of risk-control measures that are EXTERNAL to the
    software (e.g. hardware safety interlocks, independent protective devices,
    other non-software mitigations).

The three classes:

  * Class A — No injury or damage to health is possible. Either the software
              cannot contribute to a hazardous situation at all, or it can but
              the resulting possible harm is non-existent / negligible.
  * Class B — Injury is possible, but it is NOT serious (a non-serious injury).
  * Class C — Death or SERIOUS injury is possible.

"SERIOUS INJURY" (per the IEC 62304 definition) means injury or illness that,
directly or indirectly:
  (a) is life-threatening, or
  (b) results in permanent impairment of a body function or permanent damage
      to a body structure, or
  (c) necessitates medical or surgical intervention to prevent permanent
      impairment of a body function or permanent damage to a body structure.

THE TWO-STEP DECISION (Amendment 1 flow), as encoded in ``classify`` below:
  1. CAN the software contribute to a hazardous situation?
        NO  → Class A.   (Nothing further to assess.)
        YES → continue to step 2.
  2. Take the WORST-CASE harm severity, then apply any EXTERNAL (non-software)
     risk-control measure that reduces that severity. The resulting
     POST-CONTROL (residual) severity is what selects the class:
        none / negligible      → Class A
        non-serious injury      → Class B
        serious injury or death → Class C
"""

from __future__ import annotations

from enum import Enum


class Severity(Enum):
    """Worst-case POSSIBLE-HARM severity (residual, after external controls).

    Ordered from least to most severe; the integer ``value`` encodes that
    ordering so the table can verify "a control must reduce, never raise"
    severity. The ordering is an implementation detail of the guard — the
    class mapping below is what is regulatorily meaningful.
    """

    NONE = 0  # No injury / damage to health possible (negligible).
    NON_SERIOUS = 1  # Injury possible, but not serious.
    SERIOUS = 2  # Serious injury possible (see definition in the docstring).
    DEATH = 3  # Death possible.


class SafetyClass(Enum):
    """IEC 62304 software safety class."""

    A = "A"  # No injury or damage to health is possible.
    B = "B"  # Non-serious injury is possible.
    C = "C"  # Death or serious injury is possible.


# ---------------------------------------------------------------------------
# THE §4.3 SEVERITY → CLASS TABLE (pure data — audit this directly).
#
# This is the single source of truth that maps a residual (post-external-
# control) worst-case severity onto a safety class. Note that BOTH "serious
# injury" AND "death" map to Class C, exactly as the standard requires.
# ---------------------------------------------------------------------------
SEVERITY_TO_CLASS: dict[Severity, SafetyClass] = {
    Severity.NONE: SafetyClass.A,  # no possible injury → A
    Severity.NON_SERIOUS: SafetyClass.B,  # non-serious injury → B
    Severity.SERIOUS: SafetyClass.C,  # serious injury → C
    Severity.DEATH: SafetyClass.C,  # death → C
}


def classify(
    can_contribute_to_hazard: bool,
    severity_before_external_controls: Severity,
    external_control_reduces_to: Severity | None = None,
) -> SafetyClass:
    """Return the IEC 62304 §4.3 safety class for one software system / item.

    Parameters
    ----------
    can_contribute_to_hazard:
        Step 1 of the decision flow. ``False`` means the software cannot
        contribute to ANY hazardous situation → Class A immediately.
    severity_before_external_controls:
        The WORST-CASE possible-harm severity assuming the hazardous situation
        leads to harm, BEFORE applying any external (non-software) risk control.
    external_control_reduces_to:
        If an EFFECTIVE EXTERNAL (non-software) risk-control measure reduces the
        residual severity, pass the post-control severity here. The table then
        consumes this reduced severity, per §4.3's allowance for considering
        risk controls external to the software. ``None`` means no external
        control is credited and the pre-control severity is used as-is.

    Returns
    -------
    SafetyClass
        A, B, or C.

    Raises
    ------
    ValueError
        If ``external_control_reduces_to`` is HIGHER than
        ``severity_before_external_controls``. Risk controls REDUCE severity;
        they can never raise it. A request to "reduce" to a higher severity is
        incoherent and is rejected rather than silently honoured.
    """
    # --- Step 1: cannot contribute to a hazardous situation → Class A. ---
    if not can_contribute_to_hazard:
        return SafetyClass.A

    # --- Step 2: determine the EFFECTIVE (residual) severity. ---
    if external_control_reduces_to is not None:
        # GUARD: controls reduce, never raise. The Severity int values encode
        # least→most severe, so a "reduction" to a higher value is invalid.
        if external_control_reduces_to.value > severity_before_external_controls.value:
            raise ValueError(
                "external_control_reduces_to "
                f"({external_control_reduces_to.name}) is more severe than the "
                "pre-control severity "
                f"({severity_before_external_controls.name}); risk controls "
                "reduce severity, they cannot raise it."
            )
        effective_severity = external_control_reduces_to
    else:
        effective_severity = severity_before_external_controls

    # --- Map the residual severity onto the class via the audited table. ---
    return SEVERITY_TO_CLASS[effective_severity]
