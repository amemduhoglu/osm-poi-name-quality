"""Split a city so that a borrowed name is repeated nowhere it can be read.

The residual class as first built took a donor's name and left the donor in the
corpus, so every corrupted name appeared twice in its city and a duplicate check
separated the class without knowing anything about places. The control set
built on this module takes donors from a withheld part of each city, builds
both members of every pair from the visible part, and gives every instrument
that reads the city's corpus the visible part alone.

The donor filters were also applied to the donor and never to the clean
partner, so the clean half carried category tokens and short names the corrupted
half could not. `clean_name_eligible` holds the clean partner to the same filters.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from collections.abc import Iterable
from typing import Any

from poi_audit.config import get
from poi_audit.corpus import fold
from poi_audit.screen import carries_brand_tag


def split_city(
    records: list[dict[str, Any]], share: float, seed: int, city: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split one city's records into a withheld and a visible part.

    Records are put in identity order before the draw, and the generator is
    seeded from the configured seed and the city name, so the split neither
    depends on the order the corpus was read in nor moves when another city's
    record count changes.

    Args:
        records: The city's corpus records.
        share: The share withheld.
        seed: The configured seed.
        city: The city name.

    Returns:
        The withheld records and the visible records, each in identity order.
    """
    digest = hashlib.sha256(f"{seed}:{city}".encode()).hexdigest()
    generator = random.Random(int(digest[:16], 16))
    order = sorted(records, key=lambda r: (str(r["osm_type"]), int(r["osm_id"])))
    chosen = set(generator.sample(range(len(order)), int(share * len(order))))
    withheld = [record for i, record in enumerate(order) if i in chosen]
    visible = [record for i, record in enumerate(order) if i not in chosen]
    return withheld, visible


def name_counts(records: Iterable[dict[str, Any]]) -> Counter[str]:
    """Count records by folded name, the comparison every duplicate check uses.

    Args:
        records: Corpus records.

    Returns:
        Folded name to how many records carry it. Unnamed records are skipped.
    """
    counts: Counter[str] = Counter()
    for record in records:
        folded = fold(str(record.get("name") or ""))
        if folded:
            counts[folded] += 1
    return counts


def clean_name_eligible(
    record: dict[str, Any], counts: Counter[str], tokens: dict[str, str]
) -> bool:
    """Return whether a record's own name passes the filters a donor's must.

    Args:
        record: A candidate record.
        counts: Folded name counts over the part of the city the name must be
            unique in.
        tokens: The category-token dictionary.

    Returns:
        True when the name could have been a donor's.
    """
    settings = get("injected_set.construction.m1")
    name = str(record.get("name") or "")
    if len(name) < int(settings["donor_min_name_chars"]):
        return False
    if counts[fold(name)] > 1:
        return False
    if carries_brand_tag(record):
        return False
    return not any(token in tokens for token in fold(name).split())
