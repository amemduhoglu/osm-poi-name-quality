"""Tests for the corpus extraction."""

from __future__ import annotations

import json

from poi_audit.config import get
from scripts.extract_corpus import build_query, count_fields, to_record, write_city

CITY = {
    "city": "Test",
    "country": "TR",
    "language": "non_english",
    "maturity": "developing",
}


def test_the_query_asks_for_a_coordinate_and_the_tags():
    query = build_query(38.4237, 27.1428, 10000, 300)
    assert query.endswith("out center tags;")
    for key in get("corpus.poi_keys"):
        assert f'["{key}"]["name"]' in query


def test_the_selected_cities_fill_the_four_design_cells():
    cities = get("corpus.cities")
    per_cell = int(get("corpus.cities_per_cell"))
    cells: dict[tuple[str, str], list[str]] = {}
    for city in cities:
        cells.setdefault((city["language"], city["maturity"]), []).append(
            city["country"]
        )
    assert len(cities) == per_cell * 4
    assert len(cells) == 4
    for key, countries in cells.items():
        assert len(countries) == per_cell, key
        assert len(set(countries)) == per_cell, f"{key} repeats a country"


def test_a_node_keeps_its_own_position_and_a_way_its_centre():
    node = to_record(
        {"type": "node", "id": 1, "lat": 38.4, "lon": 27.1, "tags": {"name": "A"}}, CITY
    )
    way = to_record(
        {
            "type": "way",
            "id": 2,
            "center": {"lat": 38.5, "lon": 27.2},
            "tags": {"name": "B"},
        },
        CITY,
    )
    assert node is not None and (node["lat"], node["lon"]) == (38.4, 27.1)
    assert way is not None and (way["lat"], way["lon"]) == (38.5, 27.2)


def test_an_element_without_a_usable_position_is_dropped():
    assert to_record({"type": "way", "id": 3, "tags": {"name": "C"}}, CITY) is None
    assert (
        to_record(
            {"type": "node", "id": 4, "lat": 91.0, "lon": 0.0, "tags": {"name": "D"}},
            CITY,
        )
        is None
    )


def test_records_are_written_in_a_fixed_order(tmp_path):
    records = [
        {"osm_type": "way", "osm_id": 2, "name": "B"},
        {"osm_type": "node", "osm_id": 9, "name": "A"},
        {"osm_type": "node", "osm_id": 1, "name": "C"},
    ]
    written = write_city(records, tmp_path, "Test City")
    assert written.name == "test-city.jsonl"
    lines = [
        json.loads(line) for line in written.read_text(encoding="utf-8").splitlines()
    ]
    assert [(r["osm_type"], r["osm_id"]) for r in lines] == [
        ("node", 1),
        ("node", 9),
        ("way", 2),
    ]


def test_the_corpus_is_counted_under_the_definitions_that_selected_the_cities():
    records = [
        {"tags": {"name": "A", "phone": "+90 232 000 00 00"}},
        {"tags": {"name": "B", "contact:website": "https://example.org"}},
        {"tags": {"name": "C", "addr:city": "X", "addr:street": "Y"}},
    ]
    counts = count_fields(records)
    assert counts["named"] == 3
    assert counts["phone"] == 1
    assert counts["website"] == 1
    assert counts["address"] == 1
