"""Tests for the gold-set models and JSONL loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from regulumen.core.eval.models import GoldItem, load_gold


def test_gold_item_defaults():
    item = GoldItem(
        id="q001",
        question="What lifecycle process applies?",
        expected_citations=["EU MDR, Annex I §17.2"],
    )
    assert item.answer_must_contain == []
    assert item.must_abstain is False
    assert item.notes == ""


def test_warn_on_empty_expected_for_non_abstain_item():
    # A non-abstain item with no expected_citations gets vacuous recall 1.0;
    # the model should warn (but not raise) so the authoring mistake is visible.
    with pytest.warns(UserWarning, match="expected_citations"):
        GoldItem(id="q_bad", question="forgot the citation?", expected_citations=[])


def test_no_warning_for_valid_item(recwarn):
    GoldItem(
        id="q_ok",
        question="What lifecycle process applies?",
        expected_citations=["EU MDR, Annex I §17.2"],
    )
    assert len(recwarn) == 0


def test_no_warning_for_must_abstain_item(recwarn):
    GoldItem(
        id="q_abstain",
        question="something not in the corpus",
        expected_citations=[],
        must_abstain=True,
    )
    assert len(recwarn) == 0


def test_load_gold_parses_items_and_defaults(tmp_path: Path):
    path = tmp_path / "gold.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"id": "q001", "question": "Q1?", '
                '"expected_citations": ["EU MDR, Annex I §17.2"]}',
                "",  # blank line should be skipped
                '{"id": "q002", "question": "Q2?", "expected_citations": [], '
                '"must_abstain": true, "notes": "not in corpus"}',
            ]
        ),
        encoding="utf-8",
    )
    items = load_gold(path)
    assert len(items) == 2
    assert items[0].id == "q001"
    assert items[0].must_abstain is False
    assert items[1].must_abstain is True
    assert items[1].notes == "not in corpus"


def test_load_gold_bad_json_reports_line_number(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"id": "q001", "question": "ok", "expected_citations": []}',
                "{not valid json",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as exc:
        load_gold(path)
    assert "line 2" in str(exc.value)


def test_load_gold_invalid_schema_reports_line_number(tmp_path: Path):
    path = tmp_path / "bad_schema.jsonl"
    path.write_text(
        '{"question": "missing id"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as exc:
        load_gold(path)
    assert "line 1" in str(exc.value)
