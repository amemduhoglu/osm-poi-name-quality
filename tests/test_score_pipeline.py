"""Tests for the composed pipeline and the criterion that reads it.

The composition is a recombination rather than a measurement, so what has to be
tested is that records leave the pipeline at the stage that settles them, that a
record routed to a person is counted as caught only when it really carries the
error, that the model stage is filled from the gates rather than from a raw score,
and that the criterion answers on evidence rather than on preference.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "score_pipeline", REPO_ROOT / "scripts" / "score_pipeline.py"
)
assert _spec and _spec.loader
score_pipeline = importlib.util.module_from_spec(_spec)
sys.modules["score_pipeline"] = score_pipeline
_spec.loader.exec_module(score_pipeline)


SECONDS = {"median": 30.0, "winsorized": 45.0, "untreated": 60.0}


def compose(truth, staged, candidates, flags):
    """Compose one small set and return its rows keyed by configuration.

    Args:
        truth: Whether each record carries the error.
        staged: Per record, whether the rule flagged and whether the lookup did.
        candidates: Records the model stage may be offered.
        flags: The model's flag per record.

    Returns:
        The rows, keyed by configuration tag.
    """
    rows = score_pipeline.compose(
        "test", truth, staged, candidates, flags, "a-model", SECONDS
    )
    return {row.configuration: row for row in rows}


def test_a_record_the_rule_settles_never_reaches_the_later_stages() -> None:
    """A record leaves at the first stage that flags it."""
    truth = {"a": True, "b": True}
    staged = {"a": (True, False), "b": (False, False)}
    rows = compose(truth, staged, {"a", "b"}, {"a": True, "b": True})
    # Only b is offered to the model, so only b can be routed to a person.
    assert rows["C3"].to_person_per_1k == pytest.approx(500.0)
    assert rows["C4"].to_person_per_1k == pytest.approx(500.0)


def test_the_lookup_only_sees_what_the_rule_passed() -> None:
    """The second stage is scored on the remainder, not on the whole set."""
    truth = {"a": True, "b": True, "c": False, "d": False}
    staged = {
        "a": (True, True),
        "b": (False, True),
        "c": (False, False),
        "d": (False, False),
    }
    rows = compose(truth, staged, set(truth), {})
    assert rows["C1"].caught == 1
    assert rows["C2"].caught == 2


def test_a_person_catches_only_what_is_really_wrong() -> None:
    """Routing a clean record to a person does not create an error to find."""
    truth = {"a": False, "b": False}
    staged = {"a": (False, False), "b": (False, False)}
    rows = compose(truth, staged, set(truth), {"a": True, "b": True})
    assert rows["C3"].caught == 0
    assert rows["C4"].caught == 0
    assert rows["C3"].to_person_per_1k == pytest.approx(1000.0)


def test_reviewing_everything_reaches_every_error() -> None:
    """C0 and C4 are ceilings, and C4 reaches C0 when nothing is filtered out."""
    truth = {"a": True, "b": False, "c": True}
    staged = {item: (False, False) for item in truth}
    rows = compose(truth, staged, set(truth), {})
    assert rows["C0"].recall == 1.0
    assert rows["C4"].recall == 1.0


def test_a_model_that_flags_nothing_adds_nothing() -> None:
    """C3 falls back to C2 when the model stage raises no flag."""
    truth = {"a": True, "b": False}
    staged = {"a": (False, False), "b": (False, False)}
    rows = compose(truth, staged, set(truth), {"a": False, "b": False})
    assert rows["C3"].caught == rows["C2"].caught
    assert rows["C3"].to_person_per_1k == 0.0


def test_reader_minutes_rise_across_the_three_treatments() -> None:
    """The untreated mean is the largest, which is what the breaks do to it."""
    truth = {"a": True}
    staged = {"a": (False, False)}
    rows = compose(truth, staged, {"a"}, {"a": True})
    row = rows["C3"]
    assert (
        row.reader_minutes_median_per_1k
        < row.reader_minutes_winsorized_per_1k
        < row.reader_minutes_untreated_per_1k
    )


def test_random_triage_matches_the_hypergeometric_expectation() -> None:
    """Drawing half a pool catches half its errors, in expectation."""
    expected, low, high = score_pipeline.random_triage(100, 10, 50)
    assert expected == pytest.approx(5.0)
    assert low <= 5 <= high


def test_random_triage_reviewing_everything_catches_everything() -> None:
    """A review volume equal to the pool leaves no error behind."""
    expected, low, high = score_pipeline.random_triage(20, 7, 20)
    assert expected == pytest.approx(7.0)
    assert (low, high) == (7, 7)


def test_random_triage_of_nothing_catches_nothing() -> None:
    """An empty pool, an empty review or an error-free pool all catch nothing."""
    assert score_pipeline.random_triage(0, 0, 0) == (0.0, 0, 0)
    assert score_pipeline.random_triage(10, 3, 0) == (0.0, 0, 0)
    assert score_pipeline.random_triage(10, 0, 5) == (0.0, 0, 0)


def _row(tag: str, recall: float, minutes: float, caught: int, positives: int):
    """Build one row for the criterion tests.

    Args:
        tag: The configuration.
        recall: Its recall.
        minutes: Reader minutes per thousand, used under all three treatments.
        caught: Errors it caught.
        positives: Errors present.

    Returns:
        The row.
    """
    return score_pipeline.Row(
        item_set="test",
        configuration=tag,
        model="",
        records=100,
        positives=positives,
        caught=caught,
        recall=recall,
        recall_low=recall,
        recall_high=recall,
        machine_flags_per_1k=0.0,
        to_person_per_1k=0.0,
        machine_seconds_per_1k=0.0,
        reader_minutes_median_per_1k=minutes,
        reader_minutes_winsorized_per_1k=minutes,
        reader_minutes_untreated_per_1k=minutes,
    )


def test_a_dominated_model_stage_does_not_pay() -> None:
    """C4 reaching more at no greater reader cost settles it against C3."""
    rows = [
        _row("C2", 0.2, 0.0, 2, 10),
        _row("C3", 0.5, 90.0, 5, 10),
        _row("C4", 0.9, 80.0, 9, 10),
        _row("C6", 0.2, 90.0, 2, 10),
    ]
    outcome = score_pipeline.dominated(rows)
    assert outcome["c3_dominated"] is True
    assert outcome["model_stage_pays"] is False


def test_an_undominated_stage_that_matches_chance_still_does_not_pay() -> None:
    """Sorting no better than a coin is not triage, whatever the cost curve."""
    rows = [
        _row("C2", 0.0, 0.0, 0, 10),
        _row("C3", 0.3, 90.0, 3, 10),
        _row("C4", 1.0, 600.0, 10, 10),
        _row("C6", 0.4, 90.0, 4, 10),
    ]
    outcome = score_pipeline.dominated(rows)
    assert outcome["c3_dominated"] is False
    assert outcome["beats_random_triage"] is False
    assert outcome["model_stage_pays"] is False


def test_an_undominated_stage_that_beats_chance_pays() -> None:
    """The criterion can be met, which is what makes failing it evidence."""
    rows = [
        _row("C2", 0.0, 0.0, 0, 10),
        _row("C3", 0.7, 90.0, 7, 10),
        _row("C4", 1.0, 600.0, 10, 10),
        _row("C6", 0.3, 90.0, 3, 10),
    ]
    outcome = score_pipeline.dominated(rows)
    assert outcome["model_stage_pays"] is True


def test_the_model_stage_is_filled_from_the_gates() -> None:
    """A model failing an applicable gate is not a candidate for the stage.

    The three the study reports as indistinguishable from flagging at random are
    the three that must not appear, since seating one would credit a score the
    study has already declined to read as a capability.
    """
    eligible = score_pipeline.eligible_models()
    assert eligible
    for refused in ("qwen3.5:2b-q4_K_M", "llama3.1:8b", "openbmb/minicpm5:q4_K_M"):
        assert refused not in eligible


def test_the_readers_recorded_times_and_the_treatments_order_them() -> None:
    """The measured treatments come out in the order the breaks imply."""
    seconds = score_pipeline.reader_seconds()
    assert seconds["median"] < seconds["winsorized"] < seconds["untreated"]
