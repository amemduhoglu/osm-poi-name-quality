"""Tests for the open authorities and for what their coverage measurement says.

The coverage numbers carry an argument: an authority may settle an individual
label and still be unusable as a pipeline check, because it holds only a small
and unevenly spread share of the corpus. That argument is only as good as the
measurement, so the parsing, the radius and the interval are all checked here.
"""

from __future__ import annotations

import io
import zipfile

import pytest

from poi_audit.config import get
from poi_audit.references import Gazetteer, Place
from poi_audit.stats import wilson
from scripts.fetch_authorities import (
    GEONAMES_POI,
    parse_geonames_poi,
    parse_wikidata_poi,
    within_disc,
)
from scripts.measure_authority_coverage import covered, load_authorities

CITIES = [{"city": "Cork", "country": "IE", "lat": 51.8985, "lon": -8.4756}]


def dump_line(
    geoname_id: str,
    name: str,
    lat: float,
    lon: float,
    feature_class: str,
    feature_code: str,
) -> str:
    """Build one GeoNames dump line.

    Args:
        geoname_id: The entry's identifier.
        name: The entry's name.
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        feature_class: The dump's one-letter class.
        feature_code: The dump's feature code.

    Returns:
        The tab-separated line, with the nineteen columns the dump writes.
    """
    columns = [""] * 19
    columns[0] = geoname_id
    columns[1] = name
    columns[2] = name
    columns[3] = ""
    columns[4] = str(lat)
    columns[5] = str(lon)
    columns[6] = feature_class
    columns[7] = feature_code
    columns[8] = "IE"
    columns[14] = "0"
    return "\t".join(columns)


def archive(country: str, lines: list[str], readme: str = "readme text") -> bytes:
    """Build a country dump archive holding a readme beside the data.

    Args:
        country: The two-letter country code the data file is named for.
        lines: The dump lines.
        readme: The readme's content.

    Returns:
        The archive's bytes, with the readme written first so that it sorts
        ahead of the data exactly as the published archives do.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as writer:
        writer.writestr("readme.txt", readme)
        writer.writestr(f"{country}.txt", "\n".join(lines) + "\n")
    return buffer.getvalue()


# --- the GeoNames country dumps -------------------------------------------


def test_the_country_file_is_read_and_not_the_readme():
    payload = archive(
        "IE", [dump_line("1", "Cork Public Museum", 51.8961, -8.4943, "S", "MUS")]
    )
    rows = parse_geonames_poi(payload, "IE", CITIES, 10.0)
    assert [row["name"] for row in rows] == ["Cork Public Museum"]
    assert rows[0]["source"] == GEONAMES_POI
    assert rows[0]["city"] == "Cork"


def test_an_archive_without_the_country_file_is_refused():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as writer:
        writer.writestr("readme.txt", "nothing else here")
    with pytest.raises(ValueError):
        parse_geonames_poi(buffer.getvalue(), "IE", CITIES, 10.0)


def test_only_the_configured_feature_classes_are_kept():
    keep = set(get("natural_set.authorities.geonames_poi.feature_classes"))
    assert "P" not in keep, "populated places belong to the baseline's gazetteer"
    payload = archive(
        "IE",
        [
            dump_line("1", "Cork Public Museum", 51.8961, -8.4943, "S", "MUS"),
            dump_line("2", "Cork", 51.8985, -8.4756, "P", "PPL"),
        ],
    )
    rows = parse_geonames_poi(payload, "IE", CITIES, 10.0)
    assert [row["name"] for row in rows] == ["Cork Public Museum"]


def test_an_entry_outside_every_disc_is_not_kept():
    payload = archive(
        "IE", [dump_line("1", "Galway Cathedral", 53.2743, -9.0576, "S", "CH")]
    )
    assert parse_geonames_poi(payload, "IE", CITIES, 10.0) == []


def test_the_disc_test_names_the_city_a_coordinate_falls_in():
    assert within_disc(51.8961, -8.4943, CITIES, 10.0) == "Cork"
    assert within_disc(53.2743, -9.0576, CITIES, 10.0) == ""


# --- the Wikidata answers -------------------------------------------------


def test_a_wikidata_binding_keeps_its_first_name_and_its_aliases():
    bindings = [
        {
            "place": {"value": "http://www.wikidata.org/entity/Q1234"},
            "type": {"value": "http://www.wikidata.org/entity/Q33506"},
            "lat": {"value": "51.8961"},
            "lon": {"value": "-8.4943"},
            "names": {"value": "Cork Public Museum|Músaem Chorcaí"},
        }
    ]
    rows = parse_wikidata_poi(bindings, "IE", "Cork")
    assert rows[0]["authority_id"] == "Q1234"
    assert rows[0]["name"] == "Cork Public Museum"
    assert rows[0]["alternate_names"] == "Músaem Chorcaí"
    assert rows[0]["kind"] == "Q33506"


def test_a_binding_without_a_name_is_dropped():
    bindings = [
        {
            "place": {"value": "http://www.wikidata.org/entity/Q1"},
            "type": {"value": "http://www.wikidata.org/entity/Q33506"},
            "lat": {"value": "51.9"},
            "lon": {"value": "-8.5"},
            "names": {"value": ""},
        }
    ]
    assert parse_wikidata_poi(bindings, "IE", "Cork") == []


# --- coverage -------------------------------------------------------------


def entry(name: str, lat: float, lon: float, source: str = "geonames_poi") -> Place:
    """Build one authority entry for a test.

    Args:
        name: The entry's name.
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        source: The authority the entry belongs to.

    Returns:
        The entry.
    """
    return Place(
        source=source,
        place_id=name.lower(),
        names=frozenset({name.lower()}),
        display_name=name,
        lat=lat,
        lon=lon,
        country="IE",
        population=0,
    )


def test_coverage_names_every_authority_holding_something_in_range():
    indexes = {
        "geonames_poi": Gazetteer(places=[entry("Museum", 51.8961, -8.4943)]),
        "wikidata_poi": Gazetteer(places=[entry("Library", 51.8930, -8.4930)]),
    }
    at_museum = {"lat": 51.8961, "lon": -8.4943}
    assert covered(at_museum, indexes, 0.1) == {"geonames_poi"}
    assert covered({"lat": 51.0, "lon": -8.0}, indexes, 0.1) == set()


def test_the_matching_radius_decides_what_counts_as_covered():
    indexes = {"geonames_poi": Gazetteer(places=[entry("Museum", 51.8961, -8.4943)])}
    # Roughly 150 m north of the entry.
    record = {"lat": 51.8974, "lon": -8.4943}
    assert covered(record, indexes, 0.1) == set()
    assert covered(record, indexes, 0.25) == {"geonames_poi"}


def test_the_authorities_file_is_read_one_index_per_source(tmp_path):
    target = tmp_path / "authorities.csv"
    target.write_text(
        "source,authority_id,name,alternate_names,lat,lon,country,city,kind\n"
        "geonames_poi,1,Cork Public Museum,Músaem Chorcaí,51.8961,-8.4943,IE,Cork,MUS\n"
        "wikidata_poi,Q1,Cork City Gaol,,51.8996,-8.4986,IE,Cork,Q33506\n",
        encoding="utf-8",
    )
    indexes = load_authorities(target)
    assert sorted(indexes) == ["geonames_poi", "wikidata_poi"]
    assert len(indexes["geonames_poi"].places) == 1
    assert "musaem chorcai" in indexes["geonames_poi"].places[0].names


def test_a_missing_authorities_file_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_authorities(tmp_path / "absent.csv")


# --- the interval every proportion carries --------------------------------


def test_the_interval_stays_inside_the_unit_interval_at_zero_and_one():
    for successes, trials in ((0, 300), (300, 300)):
        estimate = wilson(successes, trials)
        assert 0.0 <= estimate.low <= estimate.high <= 1.0
    assert wilson(0, 300).low == 0.0
    assert wilson(300, 300).high == 1.0


def test_the_interval_matches_the_published_formula():
    estimate = wilson(50, 200)
    assert estimate.estimate == pytest.approx(0.25)
    assert estimate.low == pytest.approx(0.1953, abs=5e-4)
    assert estimate.high == pytest.approx(0.3145, abs=5e-4)


def test_measuring_nothing_returns_the_whole_interval():
    estimate = wilson(0, 0)
    assert (estimate.estimate, estimate.low, estimate.high) == (0.0, 0.0, 1.0)


def test_impossible_counts_are_refused():
    with pytest.raises(ValueError):
        wilson(5, 4)
    with pytest.raises(ValueError):
        wilson(-1, 10)
