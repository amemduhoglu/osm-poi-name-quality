"""Tests for the withheld split behind the residual control set.

The split exists so that a borrowed name is repeated nowhere an instrument can
read. The properties checked are the ones that guarantee it: the parts are
disjoint and complete, the split does not move between runs, and a clean name
is held to the same filters a donor name is.
"""

from __future__ import annotations

from collections import Counter

from poi_audit.withheld import clean_name_eligible, name_counts, split_city

TOKENS = {"pharmacy": "amenity=pharmacy"}


def records(n: int) -> list[dict]:
    return [
        {"osm_type": "node", "osm_id": i, "name": f"Place number {i}", "tags": {}}
        for i in range(n)
    ]


def test_the_split_is_disjoint_and_complete() -> None:
    withheld, visible = split_city(records(101), share=0.5, seed=42, city="Cork")
    ids_withheld = {r["osm_id"] for r in withheld}
    ids_visible = {r["osm_id"] for r in visible}
    assert not ids_withheld & ids_visible
    assert len(ids_withheld | ids_visible) == 101
    assert len(withheld) == 50


def test_the_split_is_reproducible_and_independent_of_input_order() -> None:
    rows = records(200)
    first = split_city(rows, share=0.5, seed=42, city="Cork")
    again = split_city(list(reversed(rows)), share=0.5, seed=42, city="Cork")
    assert first == again


def test_cities_are_split_differently() -> None:
    rows = records(200)
    cork = {r["osm_id"] for r in split_city(rows, 0.5, 42, "Cork")[0]}
    brno = {r["osm_id"] for r in split_city(rows, 0.5, 42, "Brno")[0]}
    assert cork != brno


def test_name_counts_fold_case_and_accents() -> None:
    counts = name_counts(
        [{"name": "Café Roma"}, {"name": "cafe roma"}, {"name": "Other place"}]
    )
    assert counts["cafe roma"] == 2


def test_a_clean_name_passing_every_filter_is_kept() -> None:
    record = {"name": "Kafe Roma", "tags": {}}
    assert clean_name_eligible(record, Counter({"kafe roma": 1}), TOKENS)


def test_a_clean_name_with_a_category_token_is_refused() -> None:
    record = {"name": "Central Pharmacy Store", "tags": {}}
    assert not clean_name_eligible(
        record, Counter({"central pharmacy store": 1}), TOKENS
    )


def test_a_repeated_clean_name_is_refused() -> None:
    record = {"name": "Kafe Roma", "tags": {}}
    assert not clean_name_eligible(record, Counter({"kafe roma": 2}), TOKENS)


def test_a_short_clean_name_is_refused() -> None:
    record = {"name": "Roma", "tags": {}}
    assert not clean_name_eligible(record, Counter({"roma": 1}), TOKENS)


def test_a_branded_clean_name_is_refused_under_either_brand_key() -> None:
    counts = Counter({"kafe roma": 1})
    assert not clean_name_eligible(
        {"name": "Kafe Roma", "tags": {"brand": "X"}}, counts, TOKENS
    )
    assert not clean_name_eligible(
        {"name": "Kafe Roma", "tags": {"brand:wikidata": "Q1"}}, counts, TOKENS
    )
