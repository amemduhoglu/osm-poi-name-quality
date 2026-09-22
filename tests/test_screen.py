"""Tests for the natural set's screen and for the draw that uses it.

Two things have to hold whatever the screen raises. The screen must read
strings and dictionaries only, because a screen run by a model would mean
evaluating models on records a model selected. And the draw must leave both
inclusion probabilities exact, because the estimates the article reports are
reweighted to the corpus from them; a stratum whose denominator is approximate
cannot be reweighted at all.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from typing import Any

import pytest

from poi_audit import screen
from poi_audit.config import get, path, seed_everything
from poi_audit.corpus import fold
from poi_audit.references import Gazetteer, Place
from scripts.draw_natural_set import allocate, redistribute

CITY_TOKENS = {"pharmacy": "amenity=pharmacy", "bakery": "shop=bakery"}


def record(
    osm_id: int,
    name: str,
    category: str,
    lat: float = 51.9,
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
        "language": "non_english",
        "maturity": "mature",
        "lat": lat,
        "lon": lon,
        "name": name,
        "tags": {key: value, "name": name, **tags},
    }


def place(name: str, lat: float, lon: float, source: str = "geonames") -> Place:
    """Build one gazetteer place for a test.

    Args:
        name: The place's name.
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        source: The authority the place is attributed to.

    Returns:
        The place, with its name folded as the gazetteer loader folds it.
    """
    return Place(
        source=source,
        place_id=name.lower(),
        names=frozenset({fold(name)}),
        display_name=name,
        lat=lat,
        lon=lon,
        country="NL",
        population=0,
    )


# --- the duplicate-name rule ----------------------------------------------


def test_a_name_on_a_distant_record_of_another_category_is_raised():
    subject = record(1, "Bloemendaal", "amenity=cafe", lat=51.90, lon=4.47)
    other = record(2, "Bloemendaal", "shop=bakery", lat=51.95, lon=4.60)
    hit = screen.duplicate_distant_name(subject, [subject, other])
    assert hit is not None
    assert hit.rule == screen.DUPLICATE_DISTANT_NAME
    assert "km away" in hit.evidence


def test_a_chain_branch_is_not_raised_by_the_duplicate_rule():
    subject = record(1, "Albert Heijn", "shop=supermarket", lat=51.90, lon=4.47)
    branch = record(2, "Albert Heijn", "shop=supermarket", lat=51.95, lon=4.60)
    assert screen.duplicate_distant_name(subject, [subject, branch]) is None


def test_a_nearby_namesake_is_not_raised_by_the_duplicate_rule():
    subject = record(1, "De Kuip", "amenity=cafe", lat=51.9000, lon=4.4700)
    close = record(2, "De Kuip", "leisure=pitch", lat=51.9005, lon=4.4705)
    assert screen.duplicate_distant_name(subject, [subject, close]) is None


def test_a_record_is_never_raised_against_itself():
    subject = record(1, "Solitary", "amenity=cafe")
    assert screen.duplicate_distant_name(subject, [subject]) is None


def test_a_name_the_city_repeats_is_not_raised_by_the_duplicate_rule():
    most = int(
        get("natural_set.screen.rules.duplicate_distant_name.max_records_sharing_name")
    )
    subject = record(1, "Potraviny", "shop=convenience", lat=51.90, lon=4.47)
    others = [
        record(index, "Potraviny", "amenity=cafe", lat=51.95, lon=4.60)
        for index in range(2, most + 3)
    ]
    assert screen.duplicate_distant_name(subject, [subject, *others]) is None


def test_a_sibling_category_is_not_raised_by_the_duplicate_rule():
    subject = record(1, "Mamasons", "amenity=cafe", lat=51.90, lon=4.47)
    other = record(2, "Mamasons", "amenity=ice_cream", lat=51.95, lon=4.60)
    assert screen.duplicate_distant_name(subject, [subject, other]) is None


def test_a_brand_tagged_record_is_not_raised_by_the_duplicate_rule():
    subject = record(
        1, "Bloemendaal", "amenity=cafe", lat=51.90, lon=4.47, brand="Bloemendaal"
    )
    other = record(2, "Bloemendaal", "shop=bakery", lat=51.95, lon=4.60)
    assert screen.duplicate_distant_name(subject, [subject, other]) is None


def test_the_first_pass_screen_raises_what_the_final_one_excludes():
    first_pass = screen.rule_settings(screen.FIRST_PASS)[screen.DUPLICATE_DISTANT_NAME]
    subject = record(1, "Mamasons", "amenity=cafe", lat=51.90, lon=4.47)
    other = record(2, "Mamasons", "amenity=ice_cream", lat=51.95, lon=4.60)
    assert screen.duplicate_distant_name(subject, [subject, other]) is None
    assert (
        screen.duplicate_distant_name(subject, [subject, other], first_pass) is not None
    )


def test_an_unknown_screen_variant_is_refused():
    with pytest.raises(ValueError):
        screen.rule_settings("whatever")


# --- the category-token rule ----------------------------------------------


def test_a_contradicting_category_token_is_raised():
    subject = record(1, "Central Pharmacy", "shop=bakery")
    hit = screen.category_token_contradiction(subject, CITY_TOKENS)
    assert hit is not None
    assert hit.rule == screen.CATEGORY_TOKEN_CONTRADICTION


def test_an_agreeing_category_token_is_not_raised():
    subject = record(1, "Central Pharmacy", "amenity=pharmacy")
    assert screen.category_token_contradiction(subject, CITY_TOKENS) is None


def test_a_sibling_value_contradiction_is_not_raised():
    tokens = {"pension": "tourism=guest_house"}
    subject = record(1, "VG Pension", "tourism=hotel")
    assert screen.category_token_contradiction(subject, tokens) is None
    first_pass = screen.rule_settings(screen.FIRST_PASS)[
        screen.CATEGORY_TOKEN_CONTRADICTION
    ]
    assert screen.category_token_contradiction(subject, tokens, first_pass) is not None


def test_a_name_without_an_indicative_token_is_not_raised():
    subject = record(1, "Bloemendaal", "amenity=cafe")
    assert screen.category_token_contradiction(subject, CITY_TOKENS) is None


# --- the gazetteer rule ---------------------------------------------------


def test_a_place_name_held_only_far_away_is_raised():
    places = screen.place_index(Gazetteer(places=[place("Maastricht", 50.85, 5.69)]))
    subject = record(1, "Maastricht", "amenity=cafe", lat=51.92, lon=4.48)
    hit = screen.gazetteer_place_name(subject, places)
    assert hit is not None
    assert hit.rule == screen.GAZETTEER_PLACE_NAME
    assert "geonames" in hit.evidence


def test_a_place_name_held_nearby_is_not_raised():
    places = screen.place_index(Gazetteer(places=[place("Rotterdam", 51.92, 4.48)]))
    subject = record(1, "Rotterdam", "amenity=cafe", lat=51.90, lon=4.47)
    assert screen.gazetteer_place_name(subject, places) is None


def test_a_name_shorter_than_the_minimum_is_not_raised():
    minimum = int(get("natural_set.screen.rules.gazetteer_place_name.min_name_chars"))
    short = "A" * (minimum - 1)
    places = screen.place_index(Gazetteer(places=[place(short, 40.0, 4.48)]))
    subject = record(1, short, "amenity=cafe", lat=51.90, lon=4.47)
    assert screen.gazetteer_place_name(subject, places) is None


def test_a_brand_tagged_record_is_not_raised_by_the_gazetteer_rule():
    places = screen.place_index(Gazetteer(places=[place("Westfield", 50.85, 5.69)]))
    subject = record(
        1, "Westfield", "shop=mall", lat=51.92, lon=4.48, brand="Westfield"
    )
    assert screen.gazetteer_place_name(subject, places) is None


def test_the_gazetteer_match_is_reported_with_its_country_and_distance():
    places = screen.place_index(Gazetteer(places=[place("Maastricht", 50.85, 5.69)]))
    subject = record(1, "Maastricht", "amenity=cafe", lat=51.92, lon=4.48)
    match = screen.place_match(subject, places)
    assert match is not None
    away, matched = match
    assert away > float(
        get("natural_set.screen.rules.gazetteer_place_name.absent_within_km")
    )
    assert matched.country == subject["country"]


# --- the owning rule ------------------------------------------------------


def test_the_most_place_shaped_rule_owns_a_record_two_rules_raised():
    hits = [
        screen.Hit(screen.CATEGORY_TOKEN_CONTRADICTION, "token"),
        screen.Hit(screen.GAZETTEER_PLACE_NAME, "place"),
    ]
    assert screen.owning_rule(hits) == get("natural_set.screen.rule_priority")[0]


def test_a_record_raised_by_nothing_has_no_owning_rule():
    with pytest.raises(ValueError):
        screen.owning_rule([])


def test_the_place_index_carries_alternate_names():
    entry = Place(
        source="wikidata",
        place_id="Q1",
        names=frozenset({fold("Den Haag"), fold("The Hague")}),
        display_name="Den Haag",
        lat=52.08,
        lon=4.31,
        country="NL",
        population=0,
    )
    places = screen.place_index(Gazetteer(places=[entry]))
    assert fold("The Hague") in places


# --- the allocation -------------------------------------------------------


def test_the_allocation_spreads_a_stratum_across_every_city():
    seed_everything()
    cities = [str(entry["city"]) for entry in get("corpus.cities")]
    for total in (int(get("natural_set.strata.A.records")), 200, 12):
        counts = allocate(total, cities)
        assert sum(counts.values()) == total
        assert set(counts) == set(cities)
        assert max(counts.values()) - min(counts.values()) <= 1


def test_a_shortfall_moves_to_the_cities_that_can_carry_it():
    cities = ["a", "b", "c"]
    quotas = {"a": 10, "b": 10, "c": 10}
    available = {"a": 2, "b": 50, "c": 50}
    adjusted = redistribute(quotas, available, cities)
    assert sum(adjusted.values()) == sum(quotas.values())
    assert adjusted["a"] == 2
    assert all(adjusted[city] <= available[city] for city in cities)


def test_a_stratum_larger_than_the_screen_raises_is_refused():
    cities = ["a", "b"]
    with pytest.raises(ValueError):
        redistribute({"a": 10, "b": 10}, {"a": 3, "b": 4}, cities)


# --- the drawn set --------------------------------------------------------


@pytest.fixture(scope="module")
def drawn() -> list[dict[str, Any]]:
    """Read the natural set, skipping the module when it has not been drawn."""
    target = path("data_processed") / "natural_set" / "records.jsonl"
    if not target.exists():
        pytest.skip("natural set not drawn; run scripts/draw_natural_set.py")
    with target.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_the_set_holds_the_sizes_the_plan_fixed(drawn):
    assert len(drawn) == int(get("natural_set.total"))
    for stratum in ("A", "B"):
        wanted = int(get(f"natural_set.strata.{stratum}.records"))
        assert sum(1 for item in drawn if item["stratum"] == stratum) == wanted


def test_the_two_strata_are_disjoint(drawn):
    identities = [(item["osm_type"], item["osm_id"]) for item in drawn]
    assert len(set(identities)) == len(identities)


def test_every_record_carries_a_usable_inclusion_probability(drawn):
    for item in drawn:
        assert 0.0 < item["inclusion_probability"] <= 1.0


def test_every_stratum_b_record_was_raised_by_the_screen(drawn):
    for item in drawn:
        if item["stratum"] == "B":
            assert item["screen_positive"]
            assert item["screen_rules"]
            assert all(rule in screen.RULE_ORDER for rule in item["screen_rules"])
            assert item["drawn_under_rule"] in item["screen_rules"]


def test_stratum_b_is_spread_equally_across_the_three_rules(drawn):
    counts = Counter(
        item["drawn_under_rule"] for item in drawn if item["stratum"] == "B"
    )
    assert set(counts) == set(screen.RULE_ORDER)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_a_record_two_rules_raised_is_counted_under_one_of_them(drawn):
    for item in drawn:
        if item["stratum"] == "B" and len(item["screen_rules"]) > 1:
            assert item["drawn_under_rule"] == screen.owning_rule(
                [screen.Hit(rule, "") for rule in item["screen_rules"]]
            )


def test_no_stratum_a_record_is_drawn_under_a_rule(drawn):
    assert all(
        item["drawn_under_rule"] is None for item in drawn if item["stratum"] == "A"
    )


def test_the_gazetteer_match_is_written_wherever_that_rule_raised_a_record(drawn):
    for item in drawn:
        if screen.GAZETTEER_PLACE_NAME in item["screen_rules"]:
            match = item["gazetteer_match"]
            assert match is not None
            assert isinstance(match["same_country"], bool)
            assert match["distance_km"] > 0


def test_the_screen_verdict_is_recorded_for_stratum_a_as_well(drawn):
    stratum_a = [item for item in drawn if item["stratum"] == "A"]
    assert all("screen_positive" in item for item in stratum_a)


def test_both_strata_reach_every_city(drawn):
    cities = {str(entry["city"]) for entry in get("corpus.cities")}
    for stratum in ("A", "B"):
        drawn_cities = {item["city"] for item in drawn if item["stratum"] == stratum}
        assert drawn_cities == cities


def test_the_item_identifiers_do_not_say_which_stratum_a_record_came_from(drawn):
    strata = [item["stratum"] for item in sorted(drawn, key=lambda i: i["item_id"])]
    assert len(set(strata[:20])) == 2


def test_the_labelling_sheet_shows_no_screen_verdict_and_no_stratum():
    target = path("data_processed") / "natural_set" / "labelling_sheet.csv"
    if not target.exists():
        pytest.skip("natural set not drawn; run scripts/draw_natural_set.py")
    header = target.read_text(encoding="utf-8").splitlines()[0]
    for leak in ("stratum", "screen", "inclusion", "osm_id"):
        assert leak not in header


def test_the_draw_repeats_under_the_fixed_seed():
    seed_everything()
    cities = [str(entry["city"]) for entry in get("corpus.cities")]
    first = allocate(200, cities)
    pool = list(range(1000))
    first_sample = random.sample(pool, 20)
    seed_everything()
    assert allocate(200, cities) == first
    assert random.sample(pool, 20) == first_sample
