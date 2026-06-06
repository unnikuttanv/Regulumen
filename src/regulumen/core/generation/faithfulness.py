"""The post-generation FAITHFULNESS GATE — ReguLumen's #1 differentiator.

After the LLM produces an answer, we independently verify it is grounded in the
retrieved excerpts BEFORE showing it to the user. The verification is pure (no
I/O, no LLM call) so it is fully unit-testable and cannot itself hallucinate:

* :func:`extract_citations` pulls the ``[[label]]`` spans the model emitted.
* :func:`is_abstention` detects the model's explicit abstention token.
* :func:`grounding_score` scores how grounded the answer is in ``[0, 1]``.
* :func:`check_faithfulness` produces the verdict (:class:`FaithfulnessResult`).

Citation matching reuses :func:`regulumen.core.eval.metrics.citation_hit` so a
"used" citation is compared to retrieved labels with the EXACT same normalized,
boundary-aware semantics as the evaluation harness — one source of truth for
what "this citation matches a retrieved clause" means project-wide.

Limitation: citation-grounding vs claim-grounding
-------------------------------------------------
This gate verifies CITATION VALIDITY, not CLAIM SUPPORT. Concretely, it checks
that every ``[[label]]`` the answer emitted (a) is specific enough to point at a
real clause and (b) corresponds to a clause that was actually retrieved. It does
NOT verify that the PROSE around each citation is actually entailed by the cited
chunk. A model that cites a real, retrieved label but writes contradicting or
otherwise unsupported prose next to it will PASS this gate.

Closing that gap — true claim-grounding — requires a second pass that compares
each sentence's assertion against the cited chunk's text (an NLI model or an
LLM-judge), which is deliberately OUT OF SCOPE for this pure, deterministic
gate. The reserved ``abstention_threshold`` setting (see
:class:`~regulumen.core.config.Settings` and
:class:`~regulumen.core.generation.answerer.Answerer`) is where such a future
claim-support SCORE would gate; the current citation gate is strict-binary and
does not consult it.

Grounding-score formula
-----------------------
Let ``U`` be the de-duplicated citations the answer used and ``S`` the subset of
``U`` that are BOTH specific (see :func:`_is_specific_citation`) AND match (via
``citation_hit``) some retrieved label. Then::

    grounding_score = |S| / |U|        if U is non-empty
                    = 0.0              if U is empty

i.e. the fraction of the answer's citations that point at clauses we actually
retrieved. An empty-citation (uncited) answer scores ``0.0`` — an unsupported
assertion is, by construction, ungrounded. The score is retained purely for
transparency/debugging; it is NOT a gate criterion (see below).

Grounded verdict (strict binary)
--------------------------------
``check_faithfulness`` marks an answer ``grounded`` only when ALL of:
    * the answer used at least one citation (a non-abstention answer with zero
      citations is never grounded), AND
    * there are NO unsupported citations — where a citation is unsupported if it
      is non-specific (a vague catch-all like ``[[EU MDR]]``) OR it does not
      match any retrieved label (the hallucination signal).
This is a strict BINARY gate: any single unsupported citation forces a
not-grounded verdict (which the orchestrator turns into an abstention),
regardless of how many supported citations accompany it. There is no score
threshold in the citation gate — ``score`` is reported for transparency only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from regulumen.core.eval.metrics import citation_hit, normalize_citation
from regulumen.core.retrieval.vector_store import RetrievedChunk

# Machine-detectable abstention token the model is instructed to emit. Kept
# distinctive so it never appears by accident in a real answer.
ABSTENTION_TOKEN = "__ABSTAIN__"

# Matches the content inside [[ ... ]] spans (non-greedy, no nested brackets).
_CITATION_RE = re.compile(r"\[\[\s*(.+?)\s*\]\]")

# A citation is "specific" only if, after normalization, it carries a
# clause/section signal — otherwise a bare regulation name like "EU MDR" / "MDR"
# would substring-match EVERY chunk of that regulation (a catch-all bypass).
# Conservative rule: the normalized label must contain at least one DIGIT. Since
# :func:`normalize_citation` strips the section sign but KEEPS digits and the
# dots inside clause numbers, every real clause reference ("§17.2", "Article 10",
# "Annex I 17.2", "N10") contains a digit, while bare regulation names
# ("eu mdr", "mdr", "imdrf samd") do not.
_DIGIT_RE = re.compile(r"[0-9]")


def _is_specific_citation(label: str) -> bool:
    """True if ``label`` points at a specific clause, not a whole regulation.

    A citation counts as specific only if its normalized form contains a
    clause/section signal — concretely, at least one DIGIT (e.g. an article,
    annex, or clause number such as ``17.2``, ``Article 10``, or ``N10``). A
    bare regulation name like ``"EU MDR"`` or ``"MDR"`` with no number is NOT
    specific: it would substring-match every chunk of that regulation and so
    must not be allowed to satisfy the gate.
    """
    return _DIGIT_RE.search(normalize_citation(label)) is not None


def extract_citations(answer: str) -> list[str]:
    """Return the de-duplicated, order-preserving list of cited labels.

    Pulls every ``[[label]]`` span, strips surrounding whitespace, and removes
    duplicates while keeping first-seen order. Returns ``[]`` when none.
    """
    seen: dict[str, None] = {}
    for match in _CITATION_RE.finditer(answer):
        label = match.group(1).strip()
        if label and label not in seen:
            seen[label] = None
    return list(seen)


def is_abstention(answer: str) -> bool:
    """True if the answer contains the abstention token as an isolated token.

    Matched case-insensitively, but only when the token stands alone — i.e. it
    is not flanked by a ``[A-Za-z0-9_]`` character on either side (the token
    itself contains underscores, so an ordinary word boundary is insufficient).
    This stops the token from falsely triggering when it is embedded inside a
    larger word such as ``x__ABSTAIN__y``.
    """
    pattern = r"(?<![A-Za-z0-9_])" + re.escape(ABSTENTION_TOKEN) + r"(?![A-Za-z0-9_])"
    return re.search(pattern, answer, re.IGNORECASE) is not None


def _supported(label: str, retrieved_labels: list[str]) -> bool:
    """True if ``label`` is BOTH specific AND matches a retrieved label."""
    return _is_specific_citation(label) and citation_hit(label, retrieved_labels)


def grounding_score(
    answer: str,
    retrieved: list[RetrievedChunk],
    *,
    used: list[str] | None = None,
) -> float:
    """Fraction of the answer's citations that are supported, in ``[0, 1]``.

    A citation is supported only if it is specific (see
    :func:`_is_specific_citation`) AND matches a retrieved label. See the module
    docstring for the formula. An uncited answer scores ``0.0``.

    ``used`` may be supplied to reuse an already-extracted citation list and
    avoid re-parsing the answer; when omitted it is extracted here so the
    function remains usable standalone.
    """
    if used is None:
        used = extract_citations(answer)
    if not used:
        return 0.0
    retrieved_labels = [c.citation_label for c in retrieved]
    supported = sum(1 for label in used if _supported(label, retrieved_labels))
    return supported / len(used)


@dataclass
class FaithfulnessResult:
    """Verdict of the faithfulness gate for one answer.

    Attributes
    ----------
    grounded:
        Whether the answer passed the strict binary gate. NOTE: this verifies
        CITATION validity (every cited label is specific and was retrieved), NOT
        that the prose claims are entailed by those chunks — see the module
        docstring's "citation-grounding vs claim-grounding" limitation.
    score:
        The grounding score in ``[0, 1]``. Reported for transparency/debugging
        only — it is NOT a gate criterion (the gate is strict-binary).
    used_citations:
        Citations the answer emitted (de-duplicated, in order).
    unsupported_citations:
        Citations that were emitted but are unsupported — either non-specific
        (a vague catch-all) or not matching any retrieved label. This is the
        hallucination/abstention signal.
    reason:
        Human-readable explanation of the verdict.
    """

    grounded: bool
    score: float
    used_citations: list[str] = field(default_factory=list)
    unsupported_citations: list[str] = field(default_factory=list)
    reason: str = ""


def check_faithfulness(
    answer: str,
    retrieved: list[RetrievedChunk],
) -> FaithfulnessResult:
    """Independently verify ``answer``'s citations are grounded in ``retrieved``.

    Pure function and a strict BINARY gate. ``grounded`` is True only when at
    least one citation was used AND every used citation is supported — i.e. each
    is specific (not a vague whole-regulation catch-all) and matches a retrieved
    label. Any single unsupported citation forces ``grounded=False`` regardless
    of how many supported ones accompany it. ``score`` is computed and returned
    for transparency only; it is NOT a gate criterion.

    This checks CITATION validity, not CLAIM support — see the module docstring.
    """
    used = extract_citations(answer)
    retrieved_labels = [c.citation_label for c in retrieved]
    unsupported = [label for label in used if not _supported(label, retrieved_labels)]
    score = grounding_score(answer, retrieved, used=used)

    if not used:
        return FaithfulnessResult(
            grounded=False,
            score=score,
            used_citations=used,
            unsupported_citations=unsupported,
            reason="The answer made claims without citing any retrieved excerpt.",
        )

    if unsupported:
        # Distinguish the two unsupported flavours in the reason for clarity.
        vague = [label for label in unsupported if not _is_specific_citation(label)]
        missing = [label for label in unsupported if _is_specific_citation(label)]
        parts: list[str] = []
        if missing:
            parts.append(
                "cited clauses not present in the retrieved excerpts: "
                + ", ".join(missing)
            )
        if vague:
            parts.append(
                "non-specific citation(s) that do not identify a clause: "
                + ", ".join(vague)
            )
        return FaithfulnessResult(
            grounded=False,
            score=score,
            used_citations=used,
            unsupported_citations=unsupported,
            reason="The answer included unsupported citations — " + "; ".join(parts) + ".",
        )

    return FaithfulnessResult(
        grounded=True,
        score=score,
        used_citations=used,
        unsupported_citations=unsupported,
        reason=(
            "Grounded: every cited label is specific and corresponds to a "
            "retrieved excerpt (citation validity verified; claim support not "
            "checked)."
        ),
    )
