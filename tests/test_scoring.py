"""Tests for how answers become measurements.

The rules checked here are the ones that decide what a score means: that an
unparseable answer is a failure rather than an absence, that an item with no
gold label is dropped rather than guessed, and that naming a field is scored
where the question arises and nowhere else.
"""

from __future__ import annotations

from typing import Any

from poi_audit import scoring


def answer(item_id: str, verdict: str | None, field: Any = None, valid: bool = True):
    """Build one answer in the shape the runner writes.

    Args:
        item_id: The item answered.
        verdict: The verdict the model gave, or None when it gave none.
        field: The field the model named.
        valid: Whether the answer parsed.

    Returns:
        The answer.
    """
    parsed = None if verdict is None else {"verdict": verdict, "field": field}
    return {"item_id": item_id, "parsed": parsed, "valid": valid}


TRUTH = {
    "A": {
        "item_id": "A",
        "city": "Cork",
        "class": "M1",
        "corrupted": True,
        "corrupted_field": "name",
    },
    "B": {
        "item_id": "B",
        "city": "Cork",
        "class": "M1",
        "corrupted": False,
        "corrupted_field": None,
    },
}

GOLD = {
    "N1": {
        "item_id": "N1",
        "city": "Brno",
        "stratum": "A",
        "label": "wrong",
        "scored": True,
        "decidable_core": True,
    },
    "N2": {
        "item_id": "N2",
        "city": "Brno",
        "stratum": "A",
        "label": "belongs",
        "scored": True,
        "decidable_core": False,
    },
    "N3": {
        "item_id": "N3",
        "city": "Brno",
        "stratum": "B",
        "label": "cannot_say",
        "scored": False,
        "decidable_core": False,
    },
}


def test_an_unparseable_answer_is_a_failure_to_flag_not_a_missing_measurement():
    judged = scoring.score_detect([answer("A", None, valid=False)], TRUTH)
    assert len(judged) == 1
    assert judged[0].flagged is False
    assert judged[0].valid is False
    assert scoring.confusion(judged).false_negative == 1


def test_an_item_with_no_gold_label_is_dropped_rather_than_counted():
    """A model cannot be scored against a label that does not exist."""
    answers = [answer("N1", "wrong"), answer("N2", "belongs"), answer("N3", "wrong")]
    judged = scoring.score_name_natural(answers, GOLD)
    assert {one.item_id for one in judged} == {"N1", "N2"}


def test_the_decidable_core_is_the_stricter_subset():
    answers = [answer("N1", "wrong"), answer("N2", "belongs"), answer("N3", "wrong")]
    core = scoring.score_name_natural(answers, GOLD, core_only=True)
    assert [one.item_id for one in core] == ["N1"]


def test_the_field_is_scored_only_where_the_question_arises():
    """Naming a field on a clean record is already a false positive."""
    judged = scoring.score_detect(
        [answer("A", "wrong", "name"), answer("B", "wrong", "phone")], TRUTH
    )
    hits, attempts = scoring.field_accuracy(judged)
    assert (hits, attempts) == (1, 1)


def test_a_missed_corruption_leaves_the_field_unscored():
    judged = scoring.score_detect([answer("A", "clean")], TRUTH)
    assert judged[0].field_hit is None
    assert scoring.field_accuracy(judged) == (0, 0)


def test_correctness_is_keyed_by_item_for_the_paired_test():
    judged = scoring.score_detect(
        [answer("A", "wrong", "name"), answer("B", "wrong", "phone")], TRUTH
    )
    assert scoring.correctness(judged) == {"A": True, "B": False}


def test_the_residual_class_scores_a_taken_name_as_the_positive():
    judged = scoring.score_name_injected(
        [answer("A", "wrong"), answer("B", "belongs")], TRUTH
    )
    counts = scoring.confusion(judged)
    assert (counts.true_positive, counts.true_negative) == (1, 1)


def test_groups_split_by_class_and_by_stratum():
    judged = scoring.score_name_natural(
        [answer("N1", "wrong"), answer("N2", "belongs")], GOLD
    )
    assert set(scoring.by_group(judged)) == {"A"}


def test_only_the_records_both_systems_answered_can_disagree():
    """An item one system never answered is not evidence about either."""
    first = {"a": True, "b": False, "c": True}
    second = {"a": False, "b": False}
    assert scoring.discordant(first, second) == (1, 0)


def test_two_identical_systems_have_no_discordant_pair():
    same = {"a": True, "b": False}
    assert scoring.discordant(same, dict(same)) == (0, 0)


def make(item_id: str, truth: bool, flagged: bool) -> scoring.Scored:
    """Build one judgement.

    Args:
        item_id: The item.
        truth: Whether the record really carried the error.
        flagged: Whether the model said it did.

    Returns:
        The judgement.
    """
    return scoring.Scored(item_id, "Cork", "A", truth, flagged, True, None)


def test_an_enriched_stratum_is_weighted_down_to_the_corpus():
    """The raw rate of an enriched sample overstates how often the class occurs."""
    judged = [make("rare", True, True), make("common", False, False)]
    # The enriched record was a hundred times likelier to be drawn.
    weighted = scoring.reweighted(judged, {"rare": 0.1, "common": 0.001})
    assert weighted["prevalence"] < 0.05
    assert weighted["recall"] == 1.0


def test_equal_probabilities_leave_the_raw_rate_alone():
    judged = [make("a", True, True), make("b", False, False)]
    weighted = scoring.reweighted(judged, {"a": 0.01, "b": 0.01})
    assert weighted["prevalence"] == 0.5


def test_an_impossible_record_cannot_have_been_drawn():
    import pytest

    with pytest.raises(ValueError):
        scoring.reweighted([make("a", True, True)], {"a": 0.0})


def test_every_item_carries_the_clustering_unit() -> None:
    """Inference that pools records is clustered, so every record names its unit."""
    units = scoring.clusters()
    injected = scoring.injected_truth()
    natural = scoring.natural_truth()
    assert set(units) == set(injected) | set(natural)
    assert units["M1-0001-1"] == injected["M1-0001-1"]["city"]
    assert units["N001"] == natural["N001"]["city"]
