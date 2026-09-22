"""Tests for the rule baseline.

The rules are the trivial answer the models have to beat, so their behaviour is
pinned here rather than left to whatever the regular expressions happen to do.
An abstention is tested as carefully as a flag: a lookup that cannot reach a
record must say so, because counting an abstention as a pass would let missing
coverage read as accuracy.
"""

from __future__ import annotations

from poi_audit import rules
from poi_audit.config import get
from poi_audit.corpus import fold
from poi_audit.references import (
    Gazetteer,
    Place,
    expected_calling_code,
    other_calling_codes,
)


def place(name: str, lat: float, lon: float, source: str = "geonames") -> Place:
    """Build a gazetteer place for a test.

    Args:
        name: The place name.
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        source: The authority the place is attributed to.

    Returns:
        The place.
    """
    return Place(
        source=source,
        place_id=name.lower(),
        names=frozenset({fold(name)}),
        display_name=name,
        lat=lat,
        lon=lon,
        country="TR",
        population=1000,
    )


# --- format rules ---------------------------------------------------------


def test_a_well_formed_phone_passes():
    assert rules.phone_syntax("+90 232 445 12 34").outcome == rules.PASS


def test_letters_in_a_phone_are_flagged():
    assert rules.phone_syntax("+90 232 44kk5 12 34").outcome == rules.FLAG


def test_too_few_digits_is_flagged():
    minimum = int(get("rule_baseline.phone_syntax.min_digits"))
    assert rules.phone_syntax("+9 " + "1" * (minimum - 2)).outcome == rules.FLAG


def test_a_repeated_plus_is_flagged():
    assert rules.phone_syntax("++90 232 445 12 34").outcome == rules.FLAG


def test_ordinary_opening_hours_pass():
    for value in ["Mo-Fr 09:00-17:00", "24/7", "Mo-Sa 08:00-12:00,13:00-18:00; Su off"]:
        assert rules.opening_hours_syntax(value).outcome == rules.PASS, value


def test_an_hour_outside_the_clock_is_flagged():
    assert rules.opening_hours_syntax("Mo-Fr 09:00-52:30").outcome == rules.FLAG


def test_prose_in_place_of_opening_hours_is_flagged():
    assert rules.opening_hours_syntax("opens in the morning").outcome == rules.FLAG


def test_a_well_formed_web_address_passes():
    assert rules.website_syntax("https://www.example.ie/menu").outcome == rules.PASS


def test_a_web_address_without_a_dotted_host_is_flagged():
    assert rules.website_syntax("https://example").outcome == rules.FLAG


def test_two_schemes_are_flagged():
    assert rules.website_syntax("https://http://example.com").outcome == rules.FLAG


def test_a_numeric_tag_inside_its_range_passes():
    verdicts = rules.value_range({"building:levels": "4"})
    assert [v.outcome for v in verdicts] == [rules.PASS]


def test_a_numeric_tag_outside_its_range_is_flagged():
    verdicts = rules.value_range({"building:levels": "350"})
    assert [v.outcome for v in verdicts] == [rules.FLAG]


def test_a_record_without_a_numeric_tag_produces_no_range_verdict():
    assert rules.value_range({"name": "Somewhere"}) == []


# --- reference lookups ----------------------------------------------------


def test_the_right_calling_code_passes():
    code = expected_calling_code("TR")
    assert (
        rules.phone_country_code(f"+{code} 232 445 12 34", "TR").outcome == rules.PASS
    )


def test_another_countrys_calling_code_is_flagged():
    other = next(
        c for c in other_calling_codes("TR") if c != expected_calling_code("TR")
    )
    assert (
        rules.phone_country_code(f"+{other} 232 445 12 34", "TR").outcome == rules.FLAG
    )


def test_a_national_phone_makes_the_lookup_abstain():
    assert rules.phone_country_code("0232 445 12 34", "TR").outcome == rules.ABSTAIN


def test_a_shared_calling_code_is_not_a_wrong_one():
    # Canada and the United States share +1, so a number under it is not wrong
    # for a Canadian record and the lookup must not say that it is.
    assert rules.phone_country_code("+1 604 555 0134", "CA").outcome == rules.PASS


def test_a_nearby_place_name_passes():
    gazetteer = Gazetteer(places=[place("İzmir", 38.4237, 27.1428)])
    verdict = rules.address_city("Izmir", 38.42, 27.14, gazetteer)
    assert verdict.outcome == rules.PASS


def test_a_distant_place_name_is_flagged():
    gazetteer = Gazetteer(places=[place("İzmir", 38.4237, 27.1428)])
    verdict = rules.address_city("Ankara", 38.42, 27.14, gazetteer)
    assert verdict.outcome == rules.FLAG


def test_the_lookup_abstains_where_it_holds_nothing():
    gazetteer = Gazetteer(places=[place("İzmir", 38.4237, 27.1428)])
    verdict = rules.address_city("Nairobi", -1.2864, 36.8172, gazetteer)
    assert verdict.outcome == rules.ABSTAIN


def test_an_abstention_is_not_a_flag():
    gazetteer = Gazetteer(places=[])
    verdicts = [rules.address_city("Anywhere", 0.0, 0.0, gazetteer)]
    assert not rules.flagged(verdicts)


def test_each_authority_answers_on_its_own():
    gazetteer = Gazetteer(
        places=[
            place("İzmir", 38.4237, 27.1428, source="geonames"),
            place("Konak", 38.4189, 27.1287, source="wikidata"),
        ]
    )
    assert rules.address_city("Konak", 38.42, 27.14, gazetteer, "geonames").outcome == (
        rules.FLAG
    )
    assert rules.address_city("Konak", 38.42, 27.14, gazetteer, "wikidata").outcome == (
        rules.PASS
    )


# --- the reported check ---------------------------------------------------


def test_a_contradicting_category_token_is_flagged():
    tokens = {"eczanesi": "amenity=pharmacy"}
    verdict = rules.category_token("Deniz Eczanesi", "amenity=restaurant", tokens)
    assert verdict.outcome == rules.FLAG


def test_an_agreeing_category_token_passes():
    tokens = {"eczanesi": "amenity=pharmacy"}
    verdict = rules.category_token("Deniz Eczanesi", "amenity=pharmacy", tokens)
    assert verdict.outcome == rules.PASS


def test_a_name_without_an_indicative_token_makes_the_check_abstain():
    tokens = {"eczanesi": "amenity=pharmacy"}
    verdict = rules.category_token("Deniz Yıldızı", "amenity=restaurant", tokens)
    assert verdict.outcome == rules.ABSTAIN
