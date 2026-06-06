"""Gold-set data model and JSONL loader for the ReguLumen eval harness.

A "gold item" is one curated question with the citations a faithful answer must
surface. The schema is documented in ``eval/gold/schema.md``; the real ~30-item
gold set is authored separately (Step 5b) against the real corpus with human
review. This module only defines the typed model and a strict loader.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, model_validator


class GoldItem(BaseModel):
    """One curated evaluation question.

    Fields used by RETRIEVAL metrics (this step):
        * ``id`` — stable identifier, e.g. ``"q001"``.
        * ``question`` — the natural-language query sent to the retriever.
        * ``expected_citations`` — citation_label strings (or substrings) that
          should appear among the retrieved chunks, e.g.
          ``["EU MDR, Annex I §17.2"]``.

    Fields carried through for GENERATION metrics (Step 7) — defined now so the
    gold set is forward-compatible, but ignored by retrieval metrics:
        * ``answer_must_contain`` — phrases the generated answer must include.
        * ``must_abstain`` — ``True`` when the correct behaviour is to abstain
          (the answer is not supported by the corpus).
        * ``notes`` — free-text provenance for the human reviewer.
    """

    id: str
    question: str
    expected_citations: list[str] = Field(default_factory=list)
    answer_must_contain: list[str] = Field(default_factory=list)
    must_abstain: bool = False
    notes: str = ""

    @model_validator(mode="after")
    def _warn_on_empty_expected_for_non_abstain(self) -> GoldItem:
        """Warn when a non-abstain item has no expected_citations.

        Such an item gets a vacuous recall of ``1.0`` (nothing to find, nothing
        missed) yet is NOT marked ``must_abstain`` — almost always an authoring
        mistake (a forgotten citation) rather than an intentional abstention.
        We only warn (never raise) so loading the gold set still succeeds.
        """
        if not self.must_abstain and not self.expected_citations:
            warnings.warn(
                f"GoldItem {self.id!r} has no expected_citations but "
                "must_abstain is False; its retrieval recall will be vacuously "
                "1.0. Did you forget a citation, or mean must_abstain=True?",
                stacklevel=2,
            )
        return self


def load_gold(path: Path) -> list[GoldItem]:
    """Load gold items from a JSONL file (one JSON object per line).

    Blank lines (and whitespace-only lines) are skipped. A malformed JSON line
    or a line that fails schema validation raises :class:`ValueError` naming the
    1-based line number so a reviewer can fix the gold file quickly.
    """
    path = Path(path)
    items: list[GoldItem] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: invalid JSON on line {lineno}: {exc}"
                ) from exc
            try:
                items.append(GoldItem(**data))
            except ValidationError as exc:
                raise ValueError(
                    f"{path}: gold item failed validation on line {lineno}: {exc}"
                ) from exc
    return items
