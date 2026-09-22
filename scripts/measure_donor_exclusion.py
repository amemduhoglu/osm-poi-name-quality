"""Measure what the residual class would lose without the donor exclusion.

The residual class borrows a name from a point of interest in a different
category, and a name that announces its own category ("Pharmacy", "Eczanesi")
would then be settled by a token rule rather than by knowing where the name
belongs. Donors carrying such a token are excluded when the set is built.

An exclusion applied to defend a claim has to be measured, or it is the claim
assuming itself. This script draws a donor for every residual-class item from
the same pool with the token exclusion removed, everything else held identical,
and reports how often the token rule would have settled the item. The number
goes into the deposit and one sentence of it into the article.

Output:
    data/processed/injected_set/donor_exclusion.json

Usage:
    python scripts/measure_donor_exclusion.py
"""

from __future__ import annotations

import json
import random
from collections import Counter
from typing import Any

from poi_audit import rules
from poi_audit.config import get, path, seed_everything
from poi_audit.corpus import distance_km, primary_category, read_city
from poi_audit.logsetup import setup
from scripts.build_injected_set import MAX_DONOR_TRIES
from scripts.score_rule_baseline import read_category_tokens, read_items

logger = setup("measure_donor_exclusion")


def unexcluded_pool(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the donor pool the set would have used without the exclusion.

    Every other eligibility rule is kept: the name is unique in the city, the
    record carries no brand, and the name is long enough to be more than a
    category word.

    Args:
        records: One city's corpus records.

    Returns:
        The donor records, each carrying its primary category.
    """
    settings = get("injected_set.construction.m1")
    min_chars = int(settings["donor_min_name_chars"])
    counts = Counter(record["name"] for record in records)
    pool: list[dict[str, Any]] = []
    for record in records:
        name = str(record["name"])
        if len(name) < min_chars or counts[name] > 1:
            continue
        if settings["donor_excludes_brand_names"] and "brand" in record["tags"]:
            continue
        category = primary_category(record["tags"])
        if category is None:
            continue
        pool.append({**record, "category": category})
    return pool


def measure() -> dict[str, Any]:
    """Draw an unexcluded donor per residual-class item and score the rule.

    Returns:
        The tally of token-rule outcomes over the redrawn items.
    """
    tokens = read_category_tokens()
    min_km = float(get("injected_set.construction.m1.donor_min_distance_m")) / 1000.0
    cities = [str(city["city"]) for city in get("corpus.cities")]
    pools = {city: unexcluded_pool(read_city(city)) for city in cities}

    outcomes: Counter[str] = Counter()
    redrawn = 0
    for item in read_items():
        if item["class"] != "M1" or not item["corrupted"]:
            continue
        pool = pools[str(item["city"])]
        category = str(item["record"].get("category"))
        lat = float(item["record"]["lat"])
        lon = float(item["record"]["lon"])
        for _ in range(MAX_DONOR_TRIES):
            donor = random.choice(pool)
            if donor["category"] == category:
                continue
            if distance_km(lat, lon, float(donor["lat"]), float(donor["lon"])) < min_km:
                continue
            verdict = rules.category_token(str(donor["name"]), category, tokens)
            outcomes[verdict.outcome] += 1
            redrawn += 1
            break
    return {"items_redrawn": redrawn, "category_token": dict(outcomes)}


def main() -> None:
    """Measure the exclusion and write the result."""
    seed_everything()
    measured = measure()
    total = int(measured["items_redrawn"])
    flagged = int(measured["category_token"].get(rules.FLAG, 0))
    measured["settled_share"] = round(flagged / total, 4) if total else None
    measured["note"] = (
        "Share of residual-class items a category-token rule would settle if the "
        "donor pool did not exclude category-indicative names. Every other "
        "eligibility rule is held identical."
    )

    target = path("data_processed") / "injected_set" / "donor_exclusion.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(measured, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(f"wrote {target}")
    logger.info(
        f"without the exclusion a token rule settles {flagged} of {total} items "
        f"({measured['settled_share']})"
    )


if __name__ == "__main__":
    main()
