"""Tests for the evidence the retrieval-augmented arm shows beside a record."""

from __future__ import annotations

from poi_audit.references import Gazetteer, Place
from scripts.build_rag_evidence import nearest_entries, similar_entries


def place(name: str, lat: float, lon: float, source: str = "wikidata_poi") -> Place:
    return Place(
        source=source,
        place_id=name,
        names=frozenset({name.lower()}),
        display_name=name,
        lat=lat,
        lon=lon,
        country="IE",
        population=0,
    )


def test_nearest_entries_are_capped_ordered_and_inside_the_radius() -> None:
    index = Gazetteer(
        places=[
            place("Far Hall", 51.9000, -8.4700),
            place("Next Door", 51.90005, -8.47),
            place("Across Street", 51.9003, -8.47),
            place("Too Far", 51.9100, -8.47),
        ]
    )
    record = {"lat": 51.9000, "lon": -8.4700}
    found = nearest_entries(record, [index], radius_m=100, limit=2)
    assert [entry["name"] for entry in found] == ["Far Hall", "Next Door"]
    assert found[1]["distance_m"] == 6


def test_similar_entries_rank_by_cosine_and_carry_distance() -> None:
    candidates = [
        place("Harbour Bakery", 51.95, -8.47),
        place("Hill Pharmacy", 51.90, -8.47),
    ]
    vectors = {
        "Kafe Harbour": [1.0, 0.0],
        "Harbour Bakery": [0.9, 0.1],
        "Hill Pharmacy": [0.0, 1.0],
    }
    record = {"name": "Kafe Harbour", "lat": 51.90, "lon": -8.47}
    found = similar_entries(record, candidates, vectors, limit=1)
    assert found == [{"name": "Harbour Bakery", "distance_km": 5.6}]
