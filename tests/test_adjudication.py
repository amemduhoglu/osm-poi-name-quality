"""Tests for the adjudication instrument's state.

The rule the meeting follows is fixed in the protocol, so these check that the
instrument refuses what the rule refuses rather than that it merely records.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from poi_audit import adjudication, labelling


@pytest.fixture
def meeting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[labelling.Item]:
    """Give the module a labels directory holding two complete readings.

    Args:
        tmp_path: The directory the labels are written to.
        monkeypatch: Used to point the module at that directory.

    Returns:
        The items both readers covered.
    """
    items = [
        labelling.Item(item_id=f"N00{n}", position=n, shown={"name": f"P{n}"})
        for n in (1, 2, 3)
    ]
    monkeypatch.setattr(labelling, "labels_dir", lambda: tmp_path)
    monkeypatch.setattr(labelling, "read_items", lambda run=labelling.MAIN: items)
    monkeypatch.setattr(labelling, "shown_fields", lambda: ["name"])
    readings = {
        "alpha": ["belongs", "wrong", "cannot_say"],
        "beta": ["belongs", "cannot_say", "cannot_say"],
    }
    for reader, labels in readings.items():
        lines = [
            json.dumps(
                {
                    "kind": "label",
                    "run": labelling.MAIN,
                    "annotator": reader,
                    "item_id": item.item_id,
                    "position": item.position,
                    "label": label,
                    "evidence_source": "a source",
                    "evidence_url": "",
                    "no_source": False,
                    "note": "",
                }
            )
            for item, label in zip(items, labels, strict=True)
        ]
        (tmp_path / f"{reader}.jsonl").write_text("\n".join(lines) + "\n", "utf-8")
    return items


def only_disagreement() -> adjudication.Disagreement:
    """Return the single item the two readings differ on."""
    found = adjudication.disagreements()
    assert len(found) == 1
    return found[0]


def test_readers_are_those_with_a_complete_reading(meeting: Any) -> None:
    assert adjudication.readers() == ("alpha", "beta")


def test_only_disagreements_are_in_scope(meeting: Any) -> None:
    """Items the two readings agreed on are settled and never reopened."""
    assert [item.item_id for item in adjudication.disagreements()] == ["N002"]


def test_a_label_neither_reader_reached_is_refused(meeting: Any) -> None:
    """Reaching a new label would be the third reading section 6.4 forbids."""
    item = only_disagreement()
    assert item.allowed_labels() == ("wrong", "cannot_say")
    adjudication.record_open()
    with pytest.raises(ValueError, match="third reading"):
        adjudication.record_decision(item, "belongs", "sources_conflict", "")


def test_a_reason_outside_the_list_is_refused(meeting: Any) -> None:
    adjudication.record_open()
    with pytest.raises(ValueError, match="not in the protocol"):
        adjudication.record_decision(only_disagreement(), "wrong", "because", "")


def test_no_decision_before_the_meeting_is_opened(meeting: Any) -> None:
    with pytest.raises(ValueError, match="not been opened"):
        adjudication.record_decision(
            only_disagreement(), "wrong", "sources_conflict", ""
        )


def test_a_changed_decision_is_appended_and_the_last_one_counts(meeting: Any) -> None:
    """The meeting's course stays readable: nothing is overwritten."""
    item = only_disagreement()
    adjudication.record_open()
    adjudication.record_decision(item, "wrong", "source_positive_and_specific", "")
    adjudication.record_decision(
        item, "cannot_say", "sources_conflict", "on reflection"
    )
    written = [
        action
        for action in adjudication.read_actions()
        if action["kind"] == adjudication.DECISION
    ]
    assert len(written) == 2
    assert adjudication.decisions()["N002"]["label"] == "cannot_say"


def test_the_meeting_will_not_close_with_an_item_undecided(meeting: Any) -> None:
    adjudication.record_open()
    with pytest.raises(ValueError, match="still undecided"):
        adjudication.record_close()


def test_a_closed_meeting_takes_no_further_decision(meeting: Any) -> None:
    item = only_disagreement()
    adjudication.record_open()
    adjudication.record_decision(item, "wrong", "sources_conflict", "")
    adjudication.record_close()
    assert adjudication.closed()
    with pytest.raises(ValueError, match="closed"):
        adjudication.record_decision(item, "cannot_say", "sources_conflict", "")


def test_progress_hides_the_stratum_and_the_screen(meeting: Any) -> None:
    """What the labelling instrument hid stays hidden in the meeting."""
    state = adjudication.progress()
    printed = json.dumps(state)
    assert "stratum" not in printed
    assert "screen" not in printed
    assert state["total"] == 1
