"""Hybrid (rules + LLM) IEC 62304 software safety classifier — DECISION SUPPORT.

This module offers DECISION SUPPORT for the IEC 62304 §4.3 software safety
classification (Class A / B / C). It is deliberately split into two trust
layers:

  1. The LLM is used ONLY to EXTRACT structured facts from a free-text
     component description — it never picks a class letter. The fields it
     extracts mirror the §4.3 decision inputs (can the software contribute to a
     hazard, the worst-case harm severity, and any external risk control).
  2. The CLASS is then decided by the transparent, audited pure-data table in
     :mod:`regulumen.core.risk.severity_table`. This keeps the regulatorily
     meaningful logic out of the (non-deterministic) LLM.

STRICT ABSTENTION DISCIPLINE: if the severity cannot be established — the LLM
says "unknown", flags ``needs_more_info``, omits a key, or returns text that
cannot be parsed as the expected JSON — the classifier ABSTAINS. It never
guesses a class. No severity → no class.

DISCLAIMER: every result (decided or abstained) carries a disclaimer that this
is decision support only and not a regulatory verdict.

IP NOTE: no IEC 62304 copyrighted text is reproduced here; only the decision
logic is encoded (see ``severity_table.py``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from regulumen.core.generation.llm import build_llm
from regulumen.core.retrieval.vector_store import RetrievedChunk
from regulumen.core.risk.severity_table import SafetyClass, Severity, classify

# The mandatory disclaimer attached to EVERY assessment (decided or abstained).
DISCLAIMER = (
    "Decision support only; not a regulatory verdict; classification must be "
    "confirmed by a qualified person per IEC 62304."
)

# Map the LLM's lowercase severity strings onto the Severity enum. "unknown"
# (and anything not in this map) means the severity is NOT established → abstain.
_SEVERITY_STRINGS: dict[str, Severity] = {
    "none": Severity.NONE,
    "non_serious": Severity.NON_SERIOUS,
    "serious": Severity.SERIOUS,
    "death": Severity.DEATH,
}


# ---------------------------------------------------------------------------
# STRICT EXTRACTION JSON SCHEMA (what the LLM MUST return)
# ---------------------------------------------------------------------------
# The LLM must output ONLY a single JSON object with exactly these keys:
#
#   {
#     "can_contribute_to_hazard": <bool>,         // step 1 of §4.3
#     "worst_case_severity": <string>,            // one of:
#                                                 //   "none" | "non_serious"
#                                                 //   | "serious" | "death"
#                                                 //   | "unknown"
#     "external_control_reduces_to": <string|null>, // post-control severity if
#                                                 //   an EXTERNAL (non-software)
#                                                 //   control reduces it; else
#                                                 //   null. Same vocabulary as
#                                                 //   worst_case_severity
#                                                 //   (excluding "unknown").
#     "needs_more_info": <bool>                   // true if the description is
#                                                 //   too vague to establish
#                                                 //   severity confidently.
#   }
#
# If severity is "unknown", needs_more_info is true, a key is missing, or the
# text is not valid JSON → the classifier ABSTAINS.
_EXTRACTION_SYSTEM_PROMPT = (
    "You are a regulatory analysis assistant supporting an IEC 62304 software "
    "safety classification. You DO NOT assign a safety class. You only EXTRACT "
    "structured facts from the user's software-component description.\n\n"
    "Consider the worst-case possible harm if the software contributes to a "
    "hazardous situation, after accounting for risk-control measures EXTERNAL "
    "to the software (e.g. independent hardware interlocks).\n\n"
    "Severity vocabulary:\n"
    "  - none: no injury or damage to health is possible\n"
    "  - non_serious: injury possible, but not serious\n"
    "  - serious: serious injury possible (life-threatening, permanent "
    "impairment/damage, or needing intervention to prevent such)\n"
    "  - death: death possible\n"
    "  - unknown: the description does not give enough information to decide\n\n"
    "Respond with ONLY a single JSON object, no prose, no code fences, with "
    "exactly these keys: can_contribute_to_hazard (boolean), "
    "worst_case_severity (one of none|non_serious|serious|death|unknown), "
    "external_control_reduces_to (one of none|non_serious|serious|death or "
    "null), needs_more_info (boolean). If you are not confident the severity "
    "is established, set worst_case_severity to \"unknown\" and "
    "needs_more_info to true. Do not guess."
)


def build_extraction_user_prompt(description: str, context: str = "") -> str:
    """Build the user-side extraction prompt for ``description``.

    ``context`` is optional related guidance retrieved from the corpus; it is
    advisory background only (the corpus may not contain IEC 62304 text).
    """
    parts = []
    if context.strip():
        parts.append(
            "Related reference context (background only, may be incomplete):\n"
            f"{context.strip()}\n"
        )
    parts.append(
        "Software component description to analyse:\n"
        f"{description.strip()}\n\n"
        "Return ONLY the JSON object described in your instructions."
    )
    return "\n".join(parts)


@dataclass
class RiskAssessment:
    """Result of an IEC 62304 §4.3 classification attempt.

    When ``abstained`` is True, ``safety_class``, ``severity`` and (possibly)
    ``can_contribute`` are ``None`` and ``rationale`` explains what is missing.
    """

    safety_class: SafetyClass | None
    abstained: bool
    severity: Severity | None
    can_contribute: bool | None
    rationale: str
    citations: list[RetrievedChunk]
    disclaimer: str
    extracted: dict = field(default_factory=dict)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort parse of a single JSON object from ``text``.

    Handles ```json fences and leading/trailing prose. Returns ``None`` if no
    JSON object can be parsed (caller treats that as abstain, never crash).
    """
    if not text:
        return None

    # 1. Direct parse (the happy path: the model returned bare JSON).
    candidates: list[str] = [text.strip()]

    # 2. Content inside a fenced code block, e.g. ```json ... ``` or ``` ... ```.
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        candidates.append(fence.group(1).strip())

    # 3. The first {...} balanced-ish slice (greedy to last brace) as a fallback.
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        candidates.append(brace.group(0).strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


class RiskClassifier:
    """Hybrid rules + LLM IEC 62304 §4.3 classifier.

    The LLM extracts structured facts; the audited severity table decides the
    class. Strict abstention when severity is not established.
    """

    def __init__(self, llm, *, retriever=None, top_k: int = 6) -> None:
        self._llm = llm
        self._retriever = retriever
        self._top_k = top_k

    def _abstain(
        self,
        rationale: str,
        citations: list[RetrievedChunk],
        extracted: dict,
        can_contribute: bool | None = None,
    ) -> RiskAssessment:
        return RiskAssessment(
            safety_class=None,
            abstained=True,
            severity=None,
            can_contribute=can_contribute,
            rationale=rationale,
            citations=citations,
            disclaimer=DISCLAIMER,
            extracted=extracted,
        )

    def classify(self, description: str) -> RiskAssessment:
        """Classify ``description`` into a SafetyClass, or abstain.

        Steps: (1) optionally retrieve related context, (2) ask the LLM to
        EXTRACT structured fields, (3) parse robustly and abstain on any
        missing/unknown severity, (4) apply the audited §4.3 table.
        """
        # --- Step 1: optional retrieval of related context / citations. ---
        citations: list[RetrievedChunk] = []
        context = ""
        if self._retriever is not None:
            citations = list(self._retriever.retrieve(description, self._top_k))
            context = "\n\n".join(c.text for c in citations)

        # --- Step 2: LLM extraction (facts only, never a class letter). ---
        user_prompt = build_extraction_user_prompt(description, context)
        raw = self._llm.complete(_EXTRACTION_SYSTEM_PROMPT, user_prompt)

        # --- Step 3: robust parse. Unparseable → abstain (do not crash). ---
        parsed = _extract_json_object(raw)
        if parsed is None:
            return self._abstain(
                "Abstaining: the extraction step did not return parseable JSON, "
                "so the worst-case harm severity could not be established. "
                "Please provide a clearer component description.",
                citations,
                extracted={},
            )

        # Required keys must all be present; otherwise we cannot trust it.
        required = {
            "can_contribute_to_hazard",
            "worst_case_severity",
            "needs_more_info",
        }
        missing = required - parsed.keys()
        if missing:
            return self._abstain(
                "Abstaining: the extraction was incomplete (missing fields: "
                f"{', '.join(sorted(missing))}); the worst-case harm severity "
                "could not be established.",
                citations,
                extracted=parsed,
            )

        can_contribute = bool(parsed.get("can_contribute_to_hazard"))

        # Explicit "need more info" → abstain regardless of any severity given.
        if bool(parsed.get("needs_more_info")):
            return self._abstain(
                "Abstaining: the description is too vague to establish the "
                "worst-case harm severity confidently (needs_more_info). Please "
                "describe the device function, the hazard the software could "
                "contribute to, and the worst-case harm.",
                citations,
                extracted=parsed,
                can_contribute=can_contribute,
            )

        # Resolve severity. "unknown"/unrecognised → abstain (no severity).
        severity_str = str(parsed.get("worst_case_severity", "")).strip().lower()
        severity = _SEVERITY_STRINGS.get(severity_str)
        if severity is None:
            return self._abstain(
                "Abstaining: the worst-case harm severity is unknown or could "
                f"not be interpreted (got {severity_str!r}). A safety class "
                "cannot be assigned without an established severity.",
                citations,
                extracted=parsed,
                can_contribute=can_contribute,
            )

        # Resolve any external risk-control reduction (optional).
        external_str = parsed.get("external_control_reduces_to")
        external_reduces_to: Severity | None = None
        unrecognised_control = False
        if external_str is not None:
            external_reduces_to = _SEVERITY_STRINGS.get(
                str(external_str).strip().lower()
            )
            # An unrecognised non-null control value: do NOT silently guess.
            # Ignore the control (the pre-control severity stands, the SAFER
            # direction) but record that we did so, so the result is honest
            # about having dropped an uninterpretable mitigation rather than
            # swallowing it quietly.
            if external_reduces_to is None:
                unrecognised_control = True

        # --- Step 4: apply the audited §4.3 table. ---
        try:
            safety_class = classify(
                can_contribute,
                severity,
                external_control_reduces_to=external_reduces_to,
            )
        except ValueError as exc:
            # Incoherent extraction (e.g. a control that RAISES severity) →
            # abstain rather than crash.
            return self._abstain(
                f"Abstaining: the extracted risk-control measure was incoherent "
                f"({exc}); the severity could not be reliably reduced.",
                citations,
                extracted=parsed,
                can_contribute=can_contribute,
            )

        rationale = self._build_rationale(
            can_contribute,
            severity,
            external_reduces_to,
            safety_class,
            unrecognised_control=unrecognised_control,
            raw_control=external_str,
        )

        return RiskAssessment(
            safety_class=safety_class,
            abstained=False,
            severity=severity,
            can_contribute=can_contribute,
            rationale=rationale,
            citations=citations,
            disclaimer=DISCLAIMER,
            extracted=parsed,
        )

    @staticmethod
    def _build_rationale(
        can_contribute: bool,
        severity: Severity,
        external_reduces_to: Severity | None,
        safety_class: SafetyClass,
        *,
        unrecognised_control: bool = False,
        raw_control: Any = None,
    ) -> str:
        """Explain the decision in terms of the §4.3 flow + extracted fields."""
        if not can_contribute:
            return (
                "Per IEC 62304 §4.3 step 1: the software cannot contribute to a "
                "hazardous situation, so no injury or damage to health is "
                f"possible → Class {safety_class.value}."
            )

        parts = [
            "Per IEC 62304 §4.3: the software can contribute to a hazardous "
            "situation. The extracted worst-case harm severity is "
            f"'{severity.name.lower()}'."
        ]
        if external_reduces_to is not None:
            parts.append(
                "An external (non-software) risk-control measure is credited as "
                f"reducing the residual severity to "
                f"'{external_reduces_to.name.lower()}', which is the severity "
                "the §4.3 table consumes."
            )
        elif unrecognised_control:
            # Be honest that an uninterpretable mitigation was dropped rather
            # than silently applied — important for decision-support traceability.
            parts.append(
                "Note: an external risk-control value was provided "
                f"({raw_control!r}) but could not be interpreted, so it was "
                "NOT credited; the pre-control severity stands (the more "
                "conservative choice)."
            )
        parts.append(
            f"Mapping that residual severity through the §4.3 table yields "
            f"Class {safety_class.value}."
        )
        return " ".join(parts)


def build_risk_classifier(settings, retriever=None) -> RiskClassifier:
    """Build a :class:`RiskClassifier` from ``settings`` using ``build_llm``."""
    llm = build_llm(settings)
    top_k = getattr(settings, "top_k", 6)
    return RiskClassifier(llm, retriever=retriever, top_k=top_k)
