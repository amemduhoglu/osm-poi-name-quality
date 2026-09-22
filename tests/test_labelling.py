"""Tests for the labelling instrument's rules.

The protocol's demands are only worth what the instrument enforces. Three of
them are checked here: the order is fixed and an item once past is never served
again, a label the protocol does not allow is refused, and one annotator's file
tells nothing about another's. The fourth, that the reader sees the record and
nothing else, is checked on what the page is actually given.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from poi_audit import labelling


@pytest.fixture(autouse=True)
def labels_in_tmp(tmp_path, monkeypatch):
    """Send every label file to a temporary directory.

    Args:
        tmp_path: The temporary directory pytest provides.
        monkeypatch: The patcher pytest provides.

    Yields:
        The directory the session writes to.
    """
    monkeypatch.setattr(labelling, "labels_dir", lambda: tmp_path)
    yield tmp_path


def items(count: int = 3) -> list[labelling.Item]:
    """Build a short run of items.

    Args:
        count: How many to build.

    Returns:
        The items in labelling order.
    """
    return [
        labelling.Item(
            item_id=f"N{index:03d}",
            position=index,
            shown={"name": f"Place {index}", "category": "amenity=cafe", "lat": 51.9},
        )
        for index in range(1, count + 1)
    ]


def label(annotator: str, item: labelling.Item, value: str = "belongs") -> Any:
    """Record one label with the evidence the protocol asks for.

    Args:
        annotator: The annotator identifier.
        item: The item being labelled.
        value: The label to give.

    Returns:
        The record written.
    """
    return labelling.record_label(
        annotator=annotator,
        item=item,
        label=value,
        evidence_source="a business register" if value == "wrong" else "",
        evidence_url="",
        note="",
        seconds=12.5,
        no_source=value != "wrong",
    )


# --- the identifier that reaches the deposit ------------------------------


def test_an_identifier_that_could_be_a_name_is_refused():
    assert labelling.valid_annotator("reader-2")
    assert labelling.valid_annotator("ab")
    assert not labelling.valid_annotator("Reader Two")
    assert not labelling.valid_annotator("a")
    assert not labelling.valid_annotator("reader_2")
    assert not labelling.valid_annotator("")


def test_the_readers_name_is_written_outside_the_label_file(labels_in_tmp):
    labelling.record_consent("reader-2", "A Colleague")
    written = (labels_in_tmp / "annotators.local.json").read_text(encoding="utf-8")
    assert "A Colleague" in written
    session = (labels_in_tmp / "reader-2.jsonl").read_text(encoding="utf-8")
    assert "A Colleague" not in session
    assert json.loads(session.splitlines()[0])["kind"] == "consent"


# --- the order ------------------------------------------------------------


def test_the_items_are_served_in_one_fixed_order():
    run = items()
    assert labelling.next_item("reader-2", run).item_id == "N001"
    label("reader-2", run[0])
    assert labelling.next_item("reader-2", run).item_id == "N002"
    label("reader-2", run[1])
    assert labelling.next_item("reader-2", run).item_id == "N003"


def test_an_item_already_answered_is_never_served_again():
    run = items()
    for item in run:
        served = labelling.next_item("reader-2", run)
        assert served.item_id == item.item_id
        label("reader-2", served)
    assert labelling.next_item("reader-2", run) is None


def test_a_session_resumes_where_it_stopped():
    run = items(5)
    for item in run[:2]:
        label("reader-2", item)
    assert labelling.progress("reader-2", run)["labelled"] == 2
    assert labelling.next_item("reader-2", run).item_id == "N003"


# --- what the protocol allows ---------------------------------------------


def test_a_label_outside_the_protocol_is_refused():
    with pytest.raises(ValueError):
        label("reader-2", items()[0], "probably")


def test_a_wrong_name_without_evidence_is_refused():
    with pytest.raises(ValueError):
        labelling.record_label(
            annotator="reader-2",
            item=items()[0],
            label="wrong",
            evidence_source="",
            evidence_url="",
            note="",
            seconds=5.0,
        )


def test_cannot_say_needs_no_evidence_but_needs_a_statement():
    record = labelling.record_label(
        annotator="reader-2",
        item=items()[0],
        label="cannot_say",
        evidence_source="",
        evidence_url="",
        note="nothing found either way",
        seconds=40.0,
        no_source=True,
    )
    assert record["label"] == "cannot_say"
    assert record["no_source"] is True


def test_an_item_decided_in_silence_is_refused():
    with pytest.raises(ValueError):
        labelling.record_label(
            annotator="reader-2",
            item=items()[0],
            label="belongs",
            evidence_source="",
            evidence_url="",
            note="",
            seconds=9.0,
        )


def test_a_recorded_source_overrides_the_statement_that_there_was_none():
    record = labelling.record_label(
        annotator="reader-2",
        item=items()[0],
        label="belongs",
        evidence_source="the chain's own store finder",
        evidence_url="",
        note="",
        seconds=19.0,
        no_source=True,
    )
    assert record["no_source"] is False


def test_every_label_is_written_as_it_is_given(labels_in_tmp):
    run = items()
    label("reader-2", run[0])
    on_disk = (labels_in_tmp / "reader-2.jsonl").read_text(encoding="utf-8")
    assert json.loads(on_disk.splitlines()[-1])["item_id"] == "N001"


def test_a_label_records_what_the_analysis_needs():
    record = label("reader-2", items()[0], "wrong")
    for key in (
        "annotator",
        "item_id",
        "position",
        "label",
        "evidence_source",
        "seconds_on_item",
        "labelled_at",
        "protocol_version",
    ):
        assert key in record


# --- independence of the two readings -------------------------------------


def test_one_annotators_file_says_nothing_about_another():
    run = items()
    label("reader-1", run[0])
    label("reader-1", run[1])
    assert labelling.read_labels("reader-2") == []
    assert labelling.next_item("reader-2", run).item_id == "N001"
    assert labelling.progress("reader-2", run)["labelled"] == 0


# --- the practice run -----------------------------------------------------


def test_the_two_runs_are_kept_in_separate_files(labels_in_tmp):
    run = items()
    labelling.record_label(
        annotator="reader-2",
        item=run[0],
        label="belongs",
        evidence_source="",
        evidence_url="",
        note="",
        seconds=8.0,
        run=labelling.PRACTICE,
        no_source=True,
    )
    assert (labels_in_tmp / "reader-2.practice.jsonl").exists()
    assert not (labels_in_tmp / "reader-2.jsonl").exists()


def test_a_practice_answer_does_not_advance_the_run_that_counts():
    run = items()
    labelling.record_label(
        annotator="reader-2",
        item=run[0],
        label="belongs",
        evidence_source="",
        evidence_url="",
        note="",
        seconds=8.0,
        run=labelling.PRACTICE,
        no_source=True,
    )
    assert labelling.next_item("reader-2", run, labelling.PRACTICE).item_id == "N002"
    assert labelling.next_item("reader-2", run, labelling.MAIN).item_id == "N001"
    assert labelling.progress("reader-2", run, labelling.MAIN)["labelled"] == 0


def test_every_label_says_which_run_it_came_from():
    record = labelling.record_label(
        annotator="reader-2",
        item=items()[0],
        label="cannot_say",
        evidence_source="",
        evidence_url="",
        note="",
        seconds=8.0,
        run=labelling.PRACTICE,
        no_source=True,
    )
    assert record["run"] == labelling.PRACTICE


def test_an_unknown_run_is_refused():
    with pytest.raises(ValueError):
        labelling.annotator_file("reader-2", "whatever")
    with pytest.raises(ValueError):
        labelling.read_items("whatever")


def test_the_practice_items_are_not_in_the_natural_set():
    practice = labelling.path("data_processed") / "natural_set" / "practice.jsonl"
    natural = labelling.path("data_processed") / "natural_set" / "records.jsonl"
    if not (practice.exists() and natural.exists()):
        pytest.skip("sets not drawn; run scripts/draw_practice_set.py")
    identity = lambda row: (row["osm_type"], row["osm_id"])  # noqa: E731
    with practice.open(encoding="utf-8") as handle:
        drawn = {identity(json.loads(line)) for line in handle if line.strip()}
    with natural.open(encoding="utf-8") as handle:
        counted = {identity(json.loads(line)) for line in handle if line.strip()}
    assert drawn and not (drawn & counted)


def test_the_reader_is_shown_the_record_and_nothing_else():
    fields = labelling.shown_fields()
    for leak in ("stratum", "screen_positive", "screen_rules", "inclusion_probability"):
        assert leak not in fields
    assert "name" in fields and "lat" in fields
