"""Reading the extracted corpus and the small operations every phase repeats.

The corpus itself is a directory of JSON Lines files, one per city, written by
the extraction. Item construction, the rule baseline and the natural set
all need the same handful of derived quantities from a record: which primary
category it carries, which of the item keys it has, and how far it lies from
another record. Those live here so that the three of them cannot drift apart.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterator
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Any

from poi_audit.config import get, path

EARTH_RADIUS_KM = 6371.0088

_PUNCTUATION = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def city_slug(city: str) -> str:
    """Return the file-name form of a city name.

    Args:
        city: City name as written in the configuration.

    Returns:
        The lower-case, hyphenated form the corpus files are named with.
    """
    return city.lower().replace(" ", "-")


def corpus_dir() -> Path:
    """Return the directory holding the per-city corpus files."""
    return path("data_interim") / "corpus"


def read_city(city: str) -> list[dict[str, Any]]:
    """Read one city's corpus records.

    Args:
        city: City name as written in the configuration.

    Returns:
        The records in the order they were written.

    Raises:
        FileNotFoundError: If the city has not been extracted.
    """
    target = corpus_dir() / f"{city_slug(city)}.jsonl"
    if not target.exists():
        raise FileNotFoundError(
            f"corpus file not found: {target}; run scripts/extract_corpus.py"
        )
    with target.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_corpus() -> Iterator[dict[str, Any]]:
    """Yield every corpus record, city by city, in configuration order.

    Yields:
        One corpus record at a time, so that a pass over 188,624 records does
        not need them all in memory at once.
    """
    for city in get("corpus.cities"):
        yield from read_city(str(city["city"]))


def primary_category(tags: dict[str, Any]) -> str | None:
    """Return the record's primary category as ``key=value``.

    The point-of-interest keys are ordered in the configuration, and the first
    one present decides, so that a record tagged both ``amenity`` and ``shop``
    is assigned the same category on every pass.

    Args:
        tags: The record's tag set.

    Returns:
        The category, or None when no configured key is present.
    """
    for key in get("corpus.poi_keys"):
        value = tags.get(key)
        if value:
            return f"{key}={value}"
    return None


def item_value(tags: dict[str, Any], key: str) -> str | None:
    """Return the value an item shows for one item key.

    Two OpenStreetMap conventions carry the same attribute, the bare key and its
    ``contact:`` form, and a record may use either. The bare key wins where both
    are present.

    Args:
        tags: The record's tag set.
        key: One of the configured item keys.

    Returns:
        The value, or None when the record does not carry the attribute.
    """
    if key == "category":
        return primary_category(tags)
    value = tags.get(key) or tags.get(f"contact:{key}")
    return str(value) if value else None


def numeric_tag(tags: dict[str, Any]) -> str | None:
    """Return the range-checkable numeric tag a record carries.

    The configuration lists the tags whose range is published. The first one
    present with a value that reads as a number decides, so that a record
    carrying two of them is treated the same way on every pass.

    Args:
        tags: The record's tag set.

    Returns:
        The tag key, or None when the record carries no numeric tag the rule
        knows a range for.
    """
    for key in get("references.value_ranges"):
        raw = tags.get(key)
        if raw is None:
            continue
        try:
            float(str(raw).strip())
        except ValueError:
            continue
        return key
    return None


def field_signature(tags: dict[str, Any]) -> tuple[str, ...]:
    """Return which item keys a record carries.

    Args:
        tags: The record's tag set.

    Returns:
        The present item keys in configured order, followed by the numeric tag
        when the record carries one. Two records with the same signature show
        the same fields, so pairing on it keeps field presence from telling a
        reader which member of a pair was corrupted. The numeric tag enters by
        name rather than as a slot, so that a pair shows the same tag and not
        merely some number each.
    """
    keys = get("injected_set.construction.pairing.item_keys")
    present = [key for key in keys if item_value(tags, key) is not None]
    numeric = numeric_tag(tags)
    if numeric:
        present.append(numeric)
    return tuple(present)


def item_view(record: dict[str, Any]) -> dict[str, Any]:
    """Return the fields an item shows, and nothing else.

    An item shows a fixed schema rather than the record's whole tag set,
    because a tag outside the schema is not held constant inside a pair and
    would carry information about which member was corrupted. The coordinate is
    always shown: two of the error classes are decided against position, and a
    reader who cannot see the position cannot answer them.

    Args:
        record: A corpus record.

    Returns:
        The shown fields, absent keys omitted.
    """
    tags = record["tags"]
    shown: dict[str, Any] = {}
    for key in get("injected_set.construction.pairing.item_keys"):
        value = item_value(tags, key)
        if value is not None:
            shown[key] = value
    numeric = numeric_tag(tags)
    if numeric:
        shown[numeric] = str(tags[numeric])
    shown["lat"] = round(float(record["lat"]), 5)
    shown["lon"] = round(float(record["lon"]), 5)
    return shown


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance between two coordinates in kilometres.

    Args:
        lat1: Latitude of the first point in degrees.
        lon1: Longitude of the first point in degrees.
        lat2: Latitude of the second point in degrees.
        lon2: Longitude of the second point in degrees.

    Returns:
        The distance in kilometres.

    Raises:
        ValueError: If either coordinate is outside its valid range.
    """
    for lat in (lat1, lat2):
        if not -90.0 <= lat <= 90.0:
            raise ValueError(f"latitude out of range: {lat}")
    for lon in (lon1, lon2):
        if not -180.0 <= lon <= 180.0:
            raise ValueError(f"longitude out of range: {lon}")
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = phi2 - phi1
    dlambda = radians(lon2 - lon1)
    h = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * asin(sqrt(h))


def fold(name: str) -> str:
    """Return a name folded for comparison.

    Case, accents and punctuation are removed and whitespace collapsed, so that
    a gazetteer entry matches an address written with different diacritics. The
    fold is deliberately blunt: it is used to compare place names, never to
    rewrite them, and no folded value is ever written back into the data.

    Args:
        name: The name as written.

    Returns:
        The folded form.
    """
    decomposed = unicodedata.normalize("NFKD", name.casefold())
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _WHITESPACE.sub(" ", _PUNCTUATION.sub(" ", stripped)).strip()
