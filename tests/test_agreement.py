"""Tests for the agreement statistic and the sheet the readers adjudicate on.

Kappa carries the weight of Branch 2: it is the number that says what two
independent readings of the residual class are worth. It is checked here against
worked examples and against an independent implementation, and the adjudication
sheet is checked for the one property that matters most, which is that nothing
in this repository fills in a label.
"""

from __future__ import annotations

import csv
import json

import pytest
from statsmodels.stats.inter_rater import cohens_kappa, to_table

from poi_audit import labelling
from poi_audit.stats import cohen_kappa
from scripts import measure_agreement

LABELS = ("belongs", "wrong", "cannot_say")


def as_codes(labels: list[str]) -> list[int]:
    """Return labels as the integer codes statsmodels expects.

    Args:
        labels: The labels.

    Returns:
        One code per label, in the fixed category order.
    """
    return [LABELS.index(label) for label in labels]


# --- the statistic --------------------------------------------------------


def test_perfect_agreement_is_one():
    labels = ["belongs", "wrong", "cannot_say", "belongs"]
    assert cohen_kappa(labels, labels, LABELS).kappa == pytest.approx(1.0)


def test_agreement_no_better_than_chance_is_zero():
    # Each reader says "belongs" half the time, and they coincide exactly as
    # often as their marginals predict.
    first = ["belongs", "belongs", "wrong", "wrong"]
    second = ["belongs", "wrong", "belongs", "wrong"]
    assert cohen_kappa(first, second, LABELS).kappa == pytest.approx(0.0)


def test_systematic_opposition_is_negative():
    first = ["belongs", "belongs", "wrong", "wrong"]
    second = ["wrong", "wrong", "belongs", "belongs"]
    assert cohen_kappa(first, second, LABELS).kappa < 0


def test_raw_agreement_flatters_an_unbalanced_task():
    # Both readers say "belongs" 19 times in 20 and differ on the last item.
    # Raw agreement reads 0.95; kappa, which knows that two readers saying
    # "belongs" almost always would coincide almost always by chance, reads
    # about half that. This gap is why the protocol reports kappa.
    first = ["belongs"] * 19 + ["wrong"]
    second = ["belongs"] * 19 + ["cannot_say"]
    agreement = cohen_kappa(first, second, LABELS)
    assert agreement.observed == pytest.approx(0.95)
    assert agreement.expected == pytest.approx(0.9025)
    assert agreement.kappa == pytest.approx(0.4871794871, abs=1e-9)
    assert agreement.kappa < agreement.observed / 1.5


def test_the_statistic_matches_an_independent_implementation():
    first = ["belongs"] * 30 + ["wrong"] * 8 + ["cannot_say"] * 12 + ["belongs"] * 5
    second = (
        ["belongs"] * 28
        + ["wrong"] * 2
        + ["wrong"] * 6
        + ["belongs"] * 2
        + ["cannot_say"] * 9
        + ["belongs"] * 3
        + ["cannot_say"] * 5
    )
    ours = cohen_kappa(first, second, LABELS)
    pairs = list(zip(as_codes(first), as_codes(second), strict=True))
    theirs = cohens_kappa(to_table(pairs)[0])
    assert ours.kappa == pytest.approx(theirs.kappa, abs=1e-9)
    assert ours.low <= ours.kappa <= ours.high


def test_one_label_for_everything_leaves_kappa_undefined_and_says_so():
    labels = ["belongs"] * 10
    agreement = cohen_kappa(labels, labels, LABELS)
    assert agreement.observed == 1.0
    assert agreement.kappa == 0.0


def test_readings_of_different_lengths_are_refused():
    with pytest.raises(ValueError):
        cohen_kappa(["belongs"], ["belongs", "wrong"], LABELS)


def test_an_empty_reading_is_refused():
    with pytest.raises(ValueError):
        cohen_kappa([], [], LABELS)


def test_a_category_neither_reader_used_still_enters_the_table():
    first = ["belongs", "wrong"]
    second = ["belongs", "belongs"]
    with_all = cohen_kappa(first, second, LABELS)
    from_data = cohen_kappa(first, second)
    assert with_all.items == from_data.items
    assert with_all.expected != from_data.expected or with_all.kappa == from_data.kappa


# --- the sheet the readers adjudicate on ----------------------------------


@pytest.fixture
def two_readings(tmp_path, monkeypatch):
    """Write two short readings of the same items.

    Args:
        tmp_path: The temporary directory pytest provides.
        monkeypatch: The patcher pytest provides.

    Returns:
        The directory holding the label files.
    """
    monkeypatch.setattr(labelling, "labels_dir", lambda: tmp_path)
    rows = {
        "reader-1": [("N001", "belongs"), ("N002", "wrong"), ("N003", "cannot_say")],
        "reader-2": [("N001", "belongs"), ("N002", "cannot_say"), ("N003", "belongs")],
    }
    for annotator, labels in rows.items():
        with (tmp_path / f"{annotator}.jsonl").open("w", encoding="utf-8") as handle:
            for item_id, label in labels:
                handle.write(
                    json.dumps(
                        {
                            "kind": "label",
                            "run": "main",
                            "annotator": annotator,
                            "item_id": item_id,
                            "label": label,
                            "evidence_source": "google maps",
                            "evidence_url": "",
                            "no_source": False,
                            "note": "",
                        }
                    )
                    + "\n"
                )
    return tmp_path


def test_the_readers_are_found_from_their_files(two_readings):
    assert measure_agreement.known_readers() == ["reader-1", "reader-2"]


def test_a_practice_file_is_not_read_as_a_reading(two_readings):
    (two_readings / "reader-3.practice.jsonl").write_text(
        json.dumps({"kind": "label", "item_id": "P001", "label": "belongs"}) + "\n",
        encoding="utf-8",
    )
    assert "reader-3" not in measure_agreement.known_readers()


def test_the_adjudicated_label_is_left_for_the_readers(two_readings, monkeypatch):
    target = two_readings / "adjudication_sheet.csv"

    class Paths:
        def __truediv__(self, other):
            return target.parent / other

    monkeypatch.setattr(measure_agreement, "path", lambda key: two_readings.parent)
    written = measure_agreement.write_adjudication_sheet(
        [
            {
                "item_id": "N002",
                "name": "Somewhere",
                "city": "Cork",
                "label_reader-1": "wrong",
                "evidence_reader-1": "google maps",
                "url_reader-1": "",
                "note_reader-1": "",
                "label_reader-2": "cannot_say",
                "evidence_reader-2": "web search",
                "url_reader-2": "",
                "note_reader-2": "",
                "adjudicated_label": "",
                "adjudication_reason": "",
            }
        ],
        ("reader-1", "reader-2"),
    )
    with written.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["adjudicated_label"] == ""
    assert rows[0]["adjudication_reason"] == ""
    assert rows[0]["label_reader-1"] == "wrong"


# --- what each reader reproduces of the other -----------------------------


def test_mutual_accuracy_scores_both_directions_and_skips_the_undecided():
    """A reader who abstains has made no claim, so the item enters no score."""
    given = {
        "reader-1": {
            "N001": {"label": "wrong"},
            "N002": {"label": "belongs"},
            "N003": {"label": "wrong"},
            "N004": {"label": "cannot_say"},
        },
        "reader-2": {
            "N001": {"label": "wrong"},
            "N002": {"label": "belongs"},
            "N003": {"label": "belongs"},
            "N004": {"label": "wrong"},
        },
    }
    measured = measure_agreement.mutual_accuracy(given, ("reader-1", "reader-2"))
    assert measured["items"] == 3
    forward = measured["directions"]["reader-2_against_reader-1"]
    reverse = measured["directions"]["reader-1_against_reader-2"]
    assert forward["positives"] == 2
    assert forward["balanced_accuracy"] == 0.75
    assert reverse["positives"] == 1
    assert reverse["balanced_accuracy"] == 0.75
    assert forward["ci_low"] < forward["balanced_accuracy"] < forward["ci_high"]
