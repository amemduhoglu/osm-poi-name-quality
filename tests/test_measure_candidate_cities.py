"""Tests for the candidate city measurement."""

from __future__ import annotations

import pytest

from poi_audit.config import get
from scripts.measure_candidate_cities import (
    build_query,
    count_metrics,
    disc_area_km2,
    metric_filters,
    validate_centre,
)


def test_every_candidate_carries_a_valid_centre():
    for candidate in get("corpus.candidates"):
        lat, lon = validate_centre(candidate)
        assert -90.0 <= lat <= 90.0
        assert -180.0 <= lon <= 180.0


def test_a_centre_off_the_earth_is_refused():
    with pytest.raises(ValueError):
        validate_centre({"city": "Nowhere", "lat": 91.0, "lon": 0.0})
    with pytest.raises(ValueError):
        validate_centre({"city": "Nowhere", "lat": 0.0, "lon": 181.0})


def test_the_candidate_pool_can_fill_every_design_cell():
    per_cell = int(get("corpus.cities_per_cell"))
    cells: dict[tuple[str, str], int] = {}
    for candidate in get("corpus.candidates"):
        key = (candidate["language"], candidate["hypothesis_maturity"])
        cells[key] = cells.get(key, 0) + 1
    assert len(cells) == 4
    for key, count in cells.items():
        assert count >= per_cell, f"cell {key} holds too few candidates"


def test_a_cell_can_be_filled_from_distinct_countries():
    per_cell = int(get("corpus.cities_per_cell"))
    countries: dict[tuple[str, str], set[str]] = {}
    for candidate in get("corpus.candidates"):
        key = (candidate["language"], candidate["hypothesis_maturity"])
        countries.setdefault(key, set()).add(candidate["country"])
    for key, names in countries.items():
        assert len(names) >= per_cell, f"cell {key} cannot avoid repeating a country"


def test_the_query_asks_for_every_point_of_interest_key():
    query = build_query(37.1591, 38.7969, 10000, 300)
    for key in get("corpus.poi_keys"):
        assert f'["{key}"]["name"]' in query
    assert "around:10000,37.1591,38.7969" in query
    assert query.endswith("out tags;")


def test_a_disjunction_becomes_a_key_regular_expression():
    rendered = metric_filters(
        {"require_all": ["name"], "require_any": ["phone", "contact:phone"]}
    )
    assert rendered == '["name"][~"^(phone|contact:phone)$"~"."]'


def test_a_metric_counts_only_elements_holding_every_required_key():
    payload = {
        "elements": [
            {"tags": {"name": "A", "phone": "+90 414 000 00 00"}},
            {"tags": {"name": "B", "contact:phone": "+90 414 000 00 01"}},
            {"tags": {"name": "C"}},
            {"tags": {"name": "D", "addr:city": "X", "addr:street": "Y"}},
            {"tags": {"name": "E", "addr:city": "X"}},
        ]
    }
    counts = count_metrics(payload)
    assert counts["named"] == 5
    assert counts["phone"] == 2
    assert counts["address"] == 1


def test_every_threshold_leaves_headroom_over_the_per_city_quota():
    cities = int(get("corpus.cities_per_cell")) * 4
    classes = get("injected_set.classes")
    per_city_phone = (classes["M2"]["items"] + classes["S1"]["items"]) / cities
    metrics = get("corpus.candidate_selection.metrics")
    assert metrics["phone"]["threshold"] >= 10 * per_city_phone
    per_city_named = (classes["M1"]["items"] + get("injected_set.clean_items")) / cities
    assert metrics["named"]["threshold"] >= 10 * per_city_named


def test_the_disc_area_matches_its_radius():
    assert disc_area_km2(10000) == pytest.approx(314.159, rel=1e-4)
