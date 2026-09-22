"""Tests for the screen baseline scored against the injected residual class.

The study's gate says a model must beat the best trivial answer available
rather than the first one written. The screen's rules read strings and
dictionaries only and were never scored as an answer, so they are scored here.
The injected items were not selected by the screen, so nothing about this
comparison is circular; what it does consume is the city's corpus, which the
models are not shown, and the scorer records that asymmetry with the scores.
"""

from __future__ import annotations

from typing import Any

from poi_audit import screen
from poi_audit.corpus import fold
from scripts import score_screen_baseline
from scripts.score_screen_baseline import (
    ANY_RULE,
    as_screen_record,
    correctness,
    score,
    score_natural,
)


def corpus_record(
    osm_id: int,
    name: str,
    category: str,
    lat: float = 51.90,
    lon: float = 4.47,
    **tags: str,
) -> dict[str, Any]:
    """Build one corpus-shaped record for a test.

    Args:
        osm_id: The element identifier.
        name: The record's name.
        category: The primary category as ``key=value``.
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        **tags: Any further tags the test needs.

    Returns:
        A record carrying the fields the screen reads.
    """
    key, value = category.split("=")
    return {
        "osm_type": "node",
        "osm_id": osm_id,
        "city": "Rotterdam",
        "country": "NL",
        "lat": lat,
        "lon": lon,
        "name": name,
        "tags": {"name": name, key: value, **tags},
    }


def item(osm_id: int, name: str, category: str, corrupted: bool) -> dict[str, Any]:
    """Build one injected item for a test.

    Args:
        osm_id: The element identifier of the record shown.
        name: The name shown to the instrument.
        category: The primary category as ``key=value``.
        corrupted: Whether the item carries a borrowed name.

    Returns:
        An item in the shape the item builder writes.
    """
    return {
        "item_id": f"M1-{osm_id:04d}-{1 if corrupted else 2}",
        "class": "M1",
        "corrupted": corrupted,
        "corrupted_field": "name" if corrupted else "",
        "city": "Rotterdam",
        "country": "NL",
        "osm_type": "node",
        "osm_id": osm_id,
        "record": {"name": name, "category": category, "lat": 51.90, "lon": 4.47},
    }


def test_a_borrowed_name_is_raised_and_its_clean_twin_is_not() -> None:
    """The duplicate rule separates the two halves of the residual class."""
    donor = corpus_record(1, "Kralingse Plaslicht", "shop=deli", lat=51.94, lon=4.53)
    borrowed = corpus_record(2, "Kralingse Plaslicht", "amenity=cafe")
    twin = corpus_record(3, "Bakkerij Vermeulen", "shop=bakery")
    scored = score(
        [
            item(2, "Kralingse Plaslicht", "amenity=cafe", True),
            item(3, "Bakkerij Vermeulen", "shop=bakery", False),
        ],
        [donor, borrowed, twin],
        {},
        {},
    )
    duplicate = scored["by_rule"][screen.DUPLICATE_DISTANT_NAME]
    assert duplicate["true_positive"] == 1
    assert duplicate["false_positive"] == 0
    assert duplicate["youdens_j"] == 1.0


def test_a_brand_tagged_clean_record_is_not_raised() -> None:
    """A chain repeated across the city is not evidence of a borrowed name.

    The clean half of the residual class carries brand-tagged records that the
    corrupted half cannot, since the donor filters exclude them, so a rule that
    raised them would score on the construction rather than on the name.
    """
    first = corpus_record(1, "Coffee Company", "amenity=cafe", brand="Coffee Company")
    second = corpus_record(
        2,
        "Coffee Company",
        "shop=convenience",
        lat=51.94,
        lon=4.53,
        brand="Coffee Company",
    )
    scored = score(
        [item(1, "Coffee Company", "amenity=cafe", False)],
        [first, second],
        {},
        {},
    )
    assert scored["by_rule"][screen.DUPLICATE_DISTANT_NAME]["false_positive"] == 0


def test_a_name_borrowed_from_nearby_is_not_raised() -> None:
    """Recall is not inflated by a donor that sits inside the rule's radius."""
    donor = corpus_record(1, "Delfshaven Licht", "shop=deli", lat=51.902, lon=4.472)
    borrowed = corpus_record(2, "Delfshaven Licht", "amenity=cafe")
    scored = score(
        [item(2, "Delfshaven Licht", "amenity=cafe", True)],
        [donor, borrowed],
        {},
        {},
    )
    assert scored["by_rule"][screen.DUPLICATE_DISTANT_NAME]["true_positive"] == 0


def test_the_union_of_the_rules_is_scored_beside_each_of_them() -> None:
    """The gate compares against the best trivial answer, including the union."""
    donor = corpus_record(1, "Kralingse Plaslicht", "shop=deli", lat=51.94, lon=4.53)
    borrowed = corpus_record(2, "Kralingse Plaslicht", "amenity=cafe")
    scored = score(
        [item(2, "Kralingse Plaslicht", "amenity=cafe", True)],
        [donor, borrowed],
        {},
        {},
    )
    assert ANY_RULE in scored["by_rule"]
    assert scored["by_rule"][ANY_RULE]["true_positive"] == 1


def test_the_shown_name_is_written_onto_the_corpus_record() -> None:
    """The screen reads tags the item does not carry, so the record is rebuilt."""
    base = corpus_record(2, "Bakkerij Vermeulen", "shop=bakery", brand="Vermeulen")
    rebuilt = as_screen_record(
        item(2, "Kralingse Plaslicht", "shop=bakery", True), base
    )
    assert rebuilt["name"] == "Kralingse Plaslicht"
    assert rebuilt["tags"]["name"] == "Kralingse Plaslicht"
    assert rebuilt["tags"]["brand"] == "Vermeulen"
    assert fold(rebuilt["name"]) == fold("Kralingse Plaslicht")


def test_correctness_is_keyed_by_item_for_the_paired_comparison() -> None:
    """The gate is paired on the record, so the baseline answers per item."""
    donor = corpus_record(1, "Kralingse Plaslicht", "shop=deli", lat=51.94, lon=4.53)
    borrowed = corpus_record(2, "Kralingse Plaslicht", "amenity=cafe")
    twin = corpus_record(3, "Bakkerij Vermeulen", "shop=bakery")
    scored = score(
        [
            item(2, "Kralingse Plaslicht", "amenity=cafe", True),
            item(3, "Bakkerij Vermeulen", "shop=bakery", False),
        ],
        [donor, borrowed, twin],
        {},
        {},
    )
    right = correctness(scored, screen.DUPLICATE_DISTANT_NAME)
    assert right == {"M1-0002-1": True, "M1-0003-2": True}


def test_a_missed_borrowed_name_is_counted_as_wrong() -> None:
    """Correctness is the verdict against the truth, not the raising alone."""
    donor = corpus_record(1, "Delfshaven Licht", "shop=deli", lat=51.902, lon=4.472)
    borrowed = corpus_record(2, "Delfshaven Licht", "amenity=cafe")
    scored = score(
        [item(2, "Delfshaven Licht", "amenity=cafe", True)],
        [donor, borrowed],
        {},
        {},
    )
    assert correctness(scored, screen.DUPLICATE_DISTANT_NAME) == {"M1-0002-1": False}


def natural_record(
    item_id: str, stratum: str, rules_raised: list[str]
) -> dict[str, Any]:
    """Build one natural-set record for a test.

    Args:
        item_id: The identifier the gold labels key on.
        stratum: The stratum the record was drawn in.
        rules_raised: The screen rules recorded against it at the draw.

    Returns:
        A record in the shape the draw writes.
    """
    return {
        "item_id": item_id,
        "stratum": stratum,
        "city": "Rotterdam",
        "screen_positive": bool(rules_raised),
        "screen_rules": rules_raised,
    }


def gold(item_id: str, label: str, stratum: str = "A") -> dict[str, Any]:
    """Build one gold label for a test.

    Args:
        item_id: The identifier the record keys on.
        label: The agreed label.
        stratum: The stratum the record was drawn in.

    Returns:
        A gold row in the shape the label builder writes.
    """
    return {"item_id": item_id, "label": label, "stratum": stratum, "scored": True}


def test_the_natural_score_reads_the_random_stratum_alone() -> None:
    """Stratum B was selected by the screen, so scoring it there is circular."""
    records = [
        natural_record("N1", "A", [screen.DUPLICATE_DISTANT_NAME]),
        natural_record("N2", "A", []),
        natural_record("N3", "B", [screen.DUPLICATE_DISTANT_NAME]),
    ]
    labels = [
        gold("N1", "wrong"),
        gold("N2", "belongs"),
        gold("N3", "wrong", stratum="B"),
    ]
    scored = score_natural(records, labels, screen.DUPLICATE_DISTANT_NAME)
    assert scored["stratum_a"]["true_positive"] == 1
    assert scored["stratum_a"]["false_positive"] == 0
    assert scored["stratum_a"]["youdens_j"] == 1.0
    assert scored["stratum_b"]["true_positive"] == 1


def test_an_undecidable_record_is_not_scored() -> None:
    """A record the readers could not decide carries no truth to score against."""
    records = [
        natural_record("N1", "A", [screen.DUPLICATE_DISTANT_NAME]),
        natural_record("N2", "A", []),
    ]
    labels = [gold("N1", "cannot_say"), gold("N2", "belongs")]
    scored = score_natural(records, labels, screen.DUPLICATE_DISTANT_NAME)
    assert scored["stratum_a"]["true_positive"] == 0
    assert scored["stratum_a"]["true_negative"] == 1
    assert scored["stratum_a"]["false_negative"] == 0


def test_the_residual_comparison_covers_both_sets() -> None:
    """The screen is paired against every model on the control set and the natural set.

    The baseline-dominance gate is declared for every task and was computed for
    detection alone. The conclusion reads a capability from the residual
    question, so the gate is computed where that conclusion lives.
    """
    rows = score_screen_baseline.compare_on_residual_sets()
    assert {row["task"] for row in rows} == {"control_test", "natural_stratum_a"}
    for row in rows:
        assert row["baseline_rule"]
        assert row["passes"] in {True, False}
        assert row["model_only_correct"] >= 0
        assert row["baseline_only_correct"] >= 0
