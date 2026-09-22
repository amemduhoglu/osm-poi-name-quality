"""The open references the two lookups consult.

Two things live here: the gazetteer read from the file the fetcher wrote, and
the calling codes that travel inside ``phonenumbers``. Neither is commercial and
neither is licensed on request, so the pipeline can be deposited whole
(constraint C2).

The gazetteer keeps its sources apart. A place answers as GeoNames or as
Wikidata, never as an anonymous merged record, because the coverage of each
authority over the corpus is a reported quantity and a merged table would make
it unmeasurable.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from math import cos, radians
from pathlib import Path
from typing import Any

import phonenumbers

from poi_audit.config import get, path
from poi_audit.corpus import distance_km, fold

# Places are bucketed on a quarter-degree grid and a query visits as many cells
# around its own as its radius needs, which is more cells in longitude the
# further from the equator it sits. A coarser grid would put every place in a
# metropolitan country into one cell, and the coverage pass over the corpus
# would then compare each of 188,624 records against thousands of places.
_BUCKET_DEGREES = 0.25
_KM_PER_DEGREE_LAT = 110.574
_KM_PER_DEGREE_LON = 111.320


@dataclass(frozen=True)
class Place:
    """One gazetteer place.

    Attributes:
        source: The authority the place came from.
        place_id: The authority's own identifier.
        names: Every name the authority holds for the place, folded for
            comparison.
        display_name: The authority's primary name, unfolded, for writing into
            an item or an evidence record.
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        country: Two-letter country code.
        population: Population where the authority states one, else zero.
    """

    source: str
    place_id: str
    names: frozenset[str]
    display_name: str
    lat: float
    lon: float
    country: str
    population: int


@dataclass
class Gazetteer:
    """Place names from the open authorities, kept by source.

    Attributes:
        places: Every place read from the gazetteer file.
    """

    places: list[Place] = field(default_factory=list)
    _buckets: dict[tuple[int, int], list[Place]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Bucket the places so that a radius query does not scan them all."""
        for place in self.places:
            self._buckets.setdefault(self._bucket(place.lat, place.lon), []).append(
                place
            )

    @staticmethod
    def _bucket(lat: float, lon: float) -> tuple[int, int]:
        """Return the grid cell a coordinate falls in.

        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.

        Returns:
            The cell as a pair of integers.
        """
        return int(lat // _BUCKET_DEGREES), int(lon // _BUCKET_DEGREES)

    def _candidates(self, lat: float, lon: float, radius_km: float) -> Iterator[Place]:
        """Yield the places in the grid cells a radius can reach.

        A generator rather than a list: the coverage pass asks 188,624 times
        whether anything at all is in range, and it must be able to stop at the
        first place rather than pay for a copy of every candidate first.

        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.
            radius_km: Search radius in kilometres.

        Yields:
            The places worth measuring, a superset of those inside the radius.
        """
        dlat_km, dlon_km = degrees_for_km(radius_km, lat)
        rows = int(dlat_km / _BUCKET_DEGREES) + 1
        columns = int(dlon_km / _BUCKET_DEGREES) + 1
        centre = self._bucket(lat, lon)
        for drow in range(-rows, rows + 1):
            for dcolumn in range(-columns, columns + 1):
                yield from self._buckets.get(
                    (centre[0] + drow, centre[1] + dcolumn), ()
                )

    def nearby(
        self, lat: float, lon: float, radius_km: float, source: str | None = None
    ) -> list[Place]:
        """Return the places within a radius of a coordinate.

        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.
            radius_km: Search radius in kilometres.
            source: Restrict to one authority, or None for all of them.

        Returns:
            The places inside the radius, nearest first.
        """
        found: list[tuple[float, Place]] = []
        for place in self._candidates(lat, lon, radius_km):
            if source and place.source != source:
                continue
            away = distance_km(lat, lon, place.lat, place.lon)
            if away <= radius_km:
                found.append((away, place))
        return [place for _, place in sorted(found, key=lambda pair: pair[0])]

    def has_nearby(
        self, lat: float, lon: float, radius_km: float, source: str | None = None
    ) -> bool:
        """Test whether an authority holds any place within a radius.

        The coverage pass asks this of every corpus record and needs no more
        than the answer, so it stops at the first place it finds.

        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.
            radius_km: Search radius in kilometres.
            source: Restrict to one authority, or None for all of them.

        Returns:
            True when at least one place lies inside the radius.
        """
        for place in self._candidates(lat, lon, radius_km):
            if source and place.source != source:
                continue
            if distance_km(lat, lon, place.lat, place.lon) <= radius_km:
                return True
        return False

    def names_near(
        self, lat: float, lon: float, radius_km: float, source: str | None = None
    ) -> set[str]:
        """Return the folded names of every place within a radius.

        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.
            radius_km: Search radius in kilometres.
            source: Restrict to one authority, or None for all of them.

        Returns:
            The folded names, empty when the authority holds nothing nearby.
        """
        names: set[str] = set()
        for place in self.nearby(lat, lon, radius_km, source):
            names |= place.names
        return names

    def far_places(
        self,
        country: str,
        lat: float,
        lon: float,
        min_km: float,
        source: str = "geonames",
    ) -> list[Place]:
        """Return places in one country at least a given distance away.

        This is what the M3 corruption draws from: a real place name, in the
        record's own country so that the corruption does not restate the
        language factor, far enough away that a lookup can settle it.

        Args:
            country: Two-letter country code.
            lat: Latitude of the record in degrees.
            lon: Longitude of the record in degrees.
            min_km: Minimum distance in kilometres.
            source: The authority to draw from.

        Returns:
            The eligible places, in gazetteer order.
        """
        return [
            place
            for place in self.places
            if place.source == source
            and place.country == country
            and distance_km(lat, lon, place.lat, place.lon) >= min_km
        ]

    def sources(self) -> list[str]:
        """Return the authorities present, in a fixed order."""
        return sorted({place.source for place in self.places})


def gazetteer_path() -> Path:
    """Return the file the fetcher writes the gazetteer to."""
    return path("data_interim") / "references" / "gazetteer.csv"


def load_gazetteer(source_file: Path | None = None) -> Gazetteer:
    """Read the gazetteer written by the fetcher.

    Args:
        source_file: An alternative file, used by the tests. The configured
            location is read when this is None.

    Returns:
        The loaded gazetteer.

    Raises:
        FileNotFoundError: If the gazetteer has not been fetched.
    """
    target = source_file or gazetteer_path()
    if not target.exists():
        raise FileNotFoundError(
            f"gazetteer not found: {target}; run scripts/fetch_references.py"
        )
    places: list[Place] = []
    with target.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            names = {fold(row["name"])}
            for alternate in row["alternate_names"].split(","):
                if alternate.strip():
                    names.add(fold(alternate))
            population = row["population"].strip()
            places.append(
                Place(
                    source=row["source"],
                    place_id=row["place_id"],
                    names=frozenset(n for n in names if n),
                    display_name=row["name"],
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    country=row["country"],
                    population=int(population) if population.isdigit() else 0,
                )
            )
    return Gazetteer(places=places)


def expected_calling_code(country: str) -> int:
    """Return the international calling code of a country.

    Args:
        country: Two-letter country code.

    Returns:
        The calling code as an integer.

    Raises:
        ValueError: If libphonenumber holds no code for the country, which
            would mean the corpus had gained a country the reference cannot
            arbitrate.
    """
    code = phonenumbers.country_code_for_region(country)
    if not code:
        raise ValueError(f"no calling code for country {country}")
    return int(code)


def other_calling_codes(country: str) -> list[int]:
    """Return every calling code except the given country's.

    The M2 corruption draws from this per item, so that no class corrupts its
    field with a single fixed value.

    Args:
        country: The country whose code is excluded.

    Returns:
        The remaining codes, ascending and without repetition.
    """
    mine = expected_calling_code(country)
    codes = {
        int(phonenumbers.country_code_for_region(region))
        for region in phonenumbers.SUPPORTED_REGIONS
    }
    return sorted(code for code in codes if code and code != mine)


def coverage(
    gazetteer: Gazetteer,
    records: Iterable[dict[str, Any]],
    radius_km: float | None = None,
) -> dict[str, dict[str, int]]:
    """Measure how much of the corpus each authority can arbitrate.

    A lookup that finds no place near a record abstains, and an abstention is
    not a pass. This is the measurement the article reports in place of the
    assertion that a gazetteer covers the corpus.

    Args:
        gazetteer: The loaded gazetteer.
        records: Corpus records, each carrying ``lat``, ``lon`` and ``city``.
        radius_km: The matching radius, or None to read it from the
            configuration.

    Returns:
        Per authority and for the union, the number of records with at least
        one place in range and the number examined.
    """
    radius = radius_km or float(get("references.address_city.match_radius_km"))
    keys = [*gazetteer.sources(), "union"]
    totals = {key: {"covered": 0, "records": 0} for key in keys}
    for record in records:
        lat, lon = float(record["lat"]), float(record["lon"])
        union = False
        for source in gazetteer.sources():
            hit = gazetteer.has_nearby(lat, lon, radius, source)
            totals[source]["records"] += 1
            totals[source]["covered"] += int(hit)
            union = union or hit
        totals["union"]["records"] += 1
        totals["union"]["covered"] += int(union)
    return totals


def degrees_for_km(km: float, lat: float) -> tuple[float, float]:
    """Return the latitude and longitude spans covering a distance.

    Args:
        km: The distance in kilometres.
        lat: The latitude the span is taken at, in degrees.

    Returns:
        The span in degrees of latitude and of longitude.
    """
    dlat = km / _KM_PER_DEGREE_LAT
    dlon = km / max(_KM_PER_DEGREE_LON * cos(radians(lat)), 1e-6)
    return dlat, dlon
