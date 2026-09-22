"""Tests for the open references.

The gazetteer answers radius queries off a grid, and a grid is easy to get
wrong: a cell spans fewer kilometres of longitude the further it sits from the
equator, so a query that visits a fixed number of cells silently misses places
at high latitude. That is checked here rather than left to the coverage numbers
to expose, because a lookup that quietly finds nothing abstains, and an
abstention looks like missing data rather than like a bug.
"""

from __future__ import annotations

import csv

from poi_audit.corpus import fold
from poi_audit.references import (
    Gazetteer,
    Place,
    coverage,
    degrees_for_km,
    expected_calling_code,
    load_gazetteer,
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
        country="XX",
        population=0,
    )


def test_a_place_east_of_the_query_is_found_at_high_latitude():
    # At 60 degrees north a quarter-degree cell is about fourteen kilometres
    # wide, so a twenty-kilometre radius has to reach past the neighbouring
    # cell. A fixed three-by-three neighbourhood would miss this place.
    gazetteer = Gazetteer(places=[place("Eastward", 60.0, 0.35)])
    assert gazetteer.nearby(60.0, 0.0, 20.0)
    assert gazetteer.has_nearby(60.0, 0.0, 20.0)


def test_a_place_outside_the_radius_is_not_found():
    gazetteer = Gazetteer(places=[place("Far", 60.0, 1.5)])
    assert not gazetteer.nearby(60.0, 0.0, 20.0)
    assert not gazetteer.has_nearby(60.0, 0.0, 20.0)


def test_nearby_returns_the_closest_place_first():
    gazetteer = Gazetteer(
        places=[place("Further", 10.0, 0.2), place("Closer", 10.0, 0.05)]
    )
    found = gazetteer.nearby(10.0, 0.0, 50.0)
    assert [p.display_name for p in found] == ["Closer", "Further"]


def test_a_degree_of_longitude_shrinks_away_from_the_equator():
    _, at_equator = degrees_for_km(100.0, 0.0)
    _, at_sixty = degrees_for_km(100.0, 60.0)
    assert at_sixty > at_equator


def test_distant_donors_stay_inside_one_country():
    gazetteer = Gazetteer(
        places=[
            Place("geonames", "1", frozenset({"near"}), "Near", 10.0, 0.0, "TR", 0),
            Place("geonames", "2", frozenset({"far"}), "Far", 12.0, 0.0, "TR", 0),
            Place("geonames", "3", frozenset({"other"}), "Other", 12.0, 0.0, "GR", 0),
        ]
    )
    found = gazetteer.far_places("TR", 10.0, 0.0, 100.0)
    assert [p.display_name for p in found] == ["Far"]


def test_coverage_counts_each_authority_and_their_union():
    gazetteer = Gazetteer(
        places=[
            place("Alpha", 10.0, 0.0, source="geonames"),
            place("Beta", 20.0, 0.0, source="wikidata"),
        ]
    )
    records = [
        {"lat": 10.0, "lon": 0.0, "city": "A"},
        {"lat": 20.0, "lon": 0.0, "city": "B"},
        {"lat": 40.0, "lon": 0.0, "city": "C"},
    ]
    measured = coverage(gazetteer, records, radius_km=25.0)
    assert measured["geonames"]["covered"] == 1
    assert measured["wikidata"]["covered"] == 1
    assert measured["union"]["covered"] == 2
    assert measured["union"]["records"] == 3


def test_every_corpus_country_has_a_calling_code():
    from poi_audit.config import get

    for city in get("corpus.cities"):
        assert expected_calling_code(str(city["country"])) > 0


def test_the_donor_codes_leave_out_the_records_own():
    assert expected_calling_code("NL") not in other_calling_codes("NL")


def test_the_fetched_gazetteer_holds_every_corpus_country():
    import pytest

    from poi_audit.config import get
    from poi_audit.references import gazetteer_path

    if not gazetteer_path().exists():
        pytest.skip("gazetteer not fetched; run scripts/fetch_references.py")
    with gazetteer_path().open(encoding="utf-8", newline="") as handle:
        countries = {row["country"] for row in csv.DictReader(handle)}
    for city in get("corpus.cities"):
        assert str(city["country"]) in countries, city["city"]


def test_the_fetched_gazetteer_loads_and_answers():
    import pytest

    from poi_audit.config import get
    from poi_audit.references import gazetteer_path

    if not gazetteer_path().exists():
        pytest.skip("gazetteer not fetched; run scripts/fetch_references.py")
    gazetteer = load_gazetteer()
    radius = float(get("references.address_city.match_radius_km"))
    for city in get("corpus.cities"):
        names = gazetteer.names_near(float(city["lat"]), float(city["lon"]), radius)
        assert names, city["city"]
