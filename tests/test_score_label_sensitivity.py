"""Tests for reading the natural set under three label rules."""

from __future__ import annotations

from scripts.score_label_sensitivity import truth_for

GOLD = {
    "N001": {
        "label": "wrong",
        "label_reader_b": "wrong",
        "label_reader_a": "wrong",
        "decidable_core": True,
        "scored": True,
    },
    "N002": {
        "label": "belongs",
        "label_reader_b": "wrong",
        "label_reader_a": "belongs",
        "decidable_core": False,
        "scored": True,
    },
    "N003": {
        "label": "cannot_say",
        "label_reader_b": "cannot_say",
        "label_reader_a": "belongs",
        "decidable_core": False,
        "scored": False,
    },
}


def test_the_adjudicated_rule_reads_the_gold_label() -> None:
    assert truth_for("adjudicated", GOLD) == {"N001": True, "N002": False, "N003": None}


def test_the_core_keeps_only_agreement() -> None:
    assert truth_for("decidable_core", GOLD) == {
        "N001": True,
        "N002": None,
        "N003": None,
    }


def test_the_non_author_rule_ignores_the_author_and_the_adjudication() -> None:
    assert truth_for("non_author", GOLD) == {"N001": True, "N002": True, "N003": None}
