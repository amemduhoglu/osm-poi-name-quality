"""Build the injected item set: nine hundred corrupted items and their pairs.

The set is built as pairs. For every corrupted item a clean one is drawn beside
it from the same city and the same field-presence signature, and neither member
is touched again. Field presence therefore says nothing about which member was
corrupted, and a reader who flagged every record carrying a phone number would
gain nothing.

Nothing here is invented. Every item is a real OpenStreetMap record extracted in
the corpus extraction, and every corruption replaces a value the record carried with
another real value or with a malformed version of its own. The class sizes, the
eligibility rules and the corruption kinds all come from the configuration.

Two properties of the finished set are recorded rather than assumed, because
both bear on what the scores mean:

    Pre-existing flags. A clean item is clean of injected error, not of every
    error a mapper ever made. The rule baseline is run over each item's fields
    before any corruption and whatever it flags is written into the item, so
    that the analysis can report scores with and without those records instead
    of discovering the problem afterwards.

    Detectability by construction. The syntactic classes and the wrong calling
    code are injected against the same definitions the rule tests, so the rule
    settles them by construction. The builder checks that this holds for every
    item, and the check is what the exit criterion asks for.

Outputs:
    data/processed/injected_set/items.jsonl      the item set, tracked
    data/processed/injected_set/manifest.json    sizes, draws, seed, provenance
    data/processed/injected_set/class_counts.csv per class and city
    data/processed/category_tokens.csv           the token dictionary

Usage:
    python scripts/build_injected_set.py
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Any

from poi_audit import rules
from poi_audit.config import get, path, seed_everything
from poi_audit.corpus import (
    distance_km,
    field_signature,
    fold,
    item_value,
    item_view,
    numeric_tag,
    primary_category,
    read_city,
)
from poi_audit.corruptions import (
    Corruption,
    corrupt_address_city,
    corrupt_calling_code,
    corrupt_name,
    corrupt_opening_hours,
    corrupt_phone_syntax,
    corrupt_value_range,
    corrupt_website,
)
from poi_audit.logsetup import setup
from poi_audit.references import Gazetteer, load_gazetteer, other_calling_codes

logger = setup("build_injected_set")

CLASS_ORDER = ["M1", "M2", "M3", "S1", "S2", "S3", "S4"]
MAX_DONOR_TRIES = 200


# ---------------------------------------------------------------------------
# The category-token dictionary
# ---------------------------------------------------------------------------


def category_tokens(records_by_city: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    """Derive the tokens whose presence in a name indicates a category.

    A token counts as indicative when it appears often enough in the corpus and
    a large enough share of its occurrences fall in one primary category. The
    dictionary is derived from the corpus rather than written by hand, so that
    it holds in every corpus language without a translator's judgement entering
    the construction.

    Args:
        records_by_city: The corpus, keyed by city.

    Returns:
        Each indicative token mapped to the category it indicates.
    """
    min_count = int(get("injected_set.construction.m1.category_token_min_count"))
    min_share = float(get("injected_set.construction.m1.category_token_min_share"))
    totals: Counter[str] = Counter()
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    for records in records_by_city.values():
        for record in records:
            category = primary_category(record["tags"])
            if not category:
                continue
            for token in set(fold(record["name"]).split()):
                totals[token] += 1
                by_category[token][category] += 1
    tokens: dict[str, str] = {}
    for token, total in totals.items():
        if total < min_count:
            continue
        category, count = by_category[token].most_common(1)[0]
        if count / total >= min_share:
            tokens[token] = category
    return tokens


def write_category_tokens(tokens: dict[str, str]) -> Path:
    """Write the token dictionary as evidence.

    Args:
        tokens: Token to indicated category.

    Returns:
        The file written.
    """
    target = path("data_processed") / "category_tokens.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["token", "category"])
        for token in sorted(tokens):
            writer.writerow([token, tokens[token]])
    logger.info(f"wrote {target} with {len(tokens)} tokens")
    return target


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------


def parsable_international_phone(value: str | None) -> bool:
    """Test whether a phone value carries a readable international prefix.

    Args:
        value: The phone value, or None.

    Returns:
        True when the calling code can be replaced without touching anything
        else in the number.
    """
    if not value or not value.strip().startswith("+"):
        return False
    import phonenumbers

    try:
        parsed = phonenumbers.parse(value.strip(), None)
    except phonenumbers.NumberParseException:
        return False
    return bool(parsed.country_code) and str(parsed.country_code) in value


def eligible(record: dict[str, Any], item_class: str) -> bool:
    """Test whether a record can carry one class's corruption.

    Eligibility is about the field the class needs, never about whether the
    record's values happen to satisfy the rule. Filtering on rule-cleanliness
    would empty the baseline's false-positive rate of its meaning, since the
    clean half would then be clean by selection rather than by the mappers.

    Args:
        record: A corpus record.
        item_class: The class the record is being considered for.

    Returns:
        True when the record carries what the class corrupts.
    """
    tags = record["tags"]
    if item_class == "M1":
        return bool(record["name"]) and primary_category(tags) is not None
    if item_class == "M2":
        return parsable_international_phone(item_value(tags, "phone"))
    if item_class == "M3":
        return item_value(tags, "addr:city") is not None
    if item_class == "S1":
        return item_value(tags, "phone") is not None
    if item_class == "S2":
        return item_value(tags, "opening_hours") is not None
    if item_class == "S3":
        return item_value(tags, "website") is not None
    if item_class == "S4":
        return numeric_tag(tags) is not None
    raise ValueError(f"unknown item class: {item_class}")


def donor_pool(
    records: list[dict[str, Any]], tokens: dict[str, str]
) -> list[dict[str, Any]]:
    """Build the pool of names the residual class may borrow inside one city.

    A donor name must be distinctive enough to belong to one place: unique in
    the city, long enough to be more than a category word, not a brand that
    legitimately repeats, and carrying no category-indicative token. The last
    exclusion is what keeps the class residual: a name that announces its own
    category would be settled by a token rule rather than by knowing where the
    name belongs.

    Args:
        records: One city's corpus records.
        tokens: The category-token dictionary.

    Returns:
        The eligible donor records, each carrying its primary category.
    """
    settings = get("injected_set.construction.m1")
    min_chars = int(settings["donor_min_name_chars"])
    counts = Counter(record["name"] for record in records)
    pool: list[dict[str, Any]] = []
    for record in records:
        name = str(record["name"])
        if len(name) < min_chars:
            continue
        if settings["donor_name_unique_in_city"] and counts[name] > 1:
            continue
        if settings["donor_excludes_brand_names"] and "brand" in record["tags"]:
            continue
        category = primary_category(record["tags"])
        if category is None:
            continue
        if settings["donor_excludes_category_tokens"] and any(
            token in tokens for token in fold(name).split()
        ):
            continue
        pool.append({**record, "category": category})
    return pool


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------


def allocate(total: int, cities: list[str]) -> dict[str, int]:
    """Spread one class's items equally across the cities.

    Args:
        total: The class's item count.
        cities: The city names, in configuration order.

    Returns:
        Items per city. A count that does not divide by the number of cities
        gives its remainder to cities drawn from the seeded sequence, and the
        draw is written into the manifest.
    """
    base, remainder = divmod(total, len(cities))
    counts = dict.fromkeys(cities, base)
    for city in random.sample(cities, remainder):
        counts[city] += 1
    return counts


# ---------------------------------------------------------------------------
# Corruption
# ---------------------------------------------------------------------------


def draw_name_donor(
    record: dict[str, Any], pool: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Draw a donor name for one residual-class item.

    Args:
        record: The record being corrupted.
        pool: The city's donor pool.

    Returns:
        The donor with its distance from the record, or None when the pool
        holds nothing eligible for this record.
    """
    settings = get("injected_set.construction.m1")
    min_km = float(settings["donor_min_distance_m"]) / 1000.0
    same_category = bool(settings["donor_same_category"])
    category = primary_category(record["tags"])
    for _ in range(MAX_DONOR_TRIES):
        donor = random.choice(pool)
        if (
            donor["osm_id"] == record["osm_id"]
            and donor["osm_type"] == record["osm_type"]
        ):
            continue
        if same_category != (donor["category"] == category):
            continue
        if fold(str(donor["name"])) == fold(str(record["name"])):
            continue
        away = distance_km(
            float(record["lat"]),
            float(record["lon"]),
            float(donor["lat"]),
            float(donor["lon"]),
        )
        if away < min_km:
            continue
        return {**donor, "distance_km": away}
    return None


def draw_city_donor(
    record: dict[str, Any], places: list[Any], gazetteer: Gazetteer
) -> dict[str, Any] | None:
    """Draw a distant place name for one address-consistency item.

    Args:
        record: The record being corrupted.
        places: The country's gazetteer places.
        gazetteer: The loaded gazetteer, consulted to make sure the drawn name
            does not also name somewhere near the record.

    Returns:
        The donor place with its distance, or None when none is eligible.
    """
    min_km = float(get("injected_set.construction.m3.donor_min_distance_km"))
    radius = float(get("references.address_city.match_radius_km"))
    lat, lon = float(record["lat"]), float(record["lon"])
    nearby = gazetteer.names_near(lat, lon, radius)
    for _ in range(MAX_DONOR_TRIES):
        place = random.choice(places)
        away = distance_km(lat, lon, place.lat, place.lon)
        if away < min_km:
            continue
        if fold(place.display_name) in nearby:
            continue
        return {
            "name": place.display_name,
            "source": place.source,
            "place_id": place.place_id,
            "distance_km": away,
        }
    return None


def corrupt(
    item_class: str,
    record: dict[str, Any],
    context: dict[str, Any],
) -> Corruption | None:
    """Apply one class's corruption to one record.

    Args:
        item_class: The class being built.
        record: The record to corrupt.
        context: The per-city material the classes draw from: the donor pool,
            the country's gazetteer places, the calling codes and the gazetteer.

    Returns:
        The corruption, or None when this record turned out to have no eligible
        donor and another record should be drawn instead.

    Raises:
        RuntimeError: If no configured malformation produced a value the rule
            flags, which would mean a class was not settled by its rule.
    """
    tags = record["tags"]
    if item_class == "M1":
        donor = draw_name_donor(record, context["donor_pool"])
        return corrupt_name(donor, str(record["name"])) if donor else None
    if item_class == "M2":
        value = str(item_value(tags, "phone"))
        applied = corrupt_calling_code(value, context["calling_codes"])
        verdict = rules.phone_country_code(applied.injected, str(record["country"]))
        if verdict.outcome != rules.FLAG:
            return None
        return applied
    if item_class == "M3":
        donor = draw_city_donor(record, context["places"], context["gazetteer"])
        if donor is None:
            return None
        return corrupt_address_city(str(item_value(tags, "addr:city")), donor)

    kinds = list(get(f"injected_set.construction.{item_class.lower()}.malformations"))
    random.shuffle(kinds)
    for kind in kinds:
        if item_class == "S1":
            applied = corrupt_phone_syntax(str(item_value(tags, "phone")), kind)
            verdict = rules.phone_syntax(applied.injected)
        elif item_class == "S2":
            applied = corrupt_opening_hours(
                str(item_value(tags, "opening_hours")), kind
            )
            verdict = rules.opening_hours_syntax(applied.injected)
        else:
            applied = corrupt_website(str(item_value(tags, "website")), kind)
            verdict = rules.website_syntax(applied.injected)
        if applied.injected != applied.original and verdict.outcome == rules.FLAG:
            return applied
    raise RuntimeError(f"no malformation of {item_class} was flagged by its rule")


def corrupt_range(record: dict[str, Any]) -> Corruption | None:
    """Move one numeric tag outside the range the configuration declares.

    Args:
        record: The record to corrupt.

    Returns:
        The corruption, or None when no kind moved the record's value, which
        happens where the record already carries a value outside the range.
    """
    tag = numeric_tag(record["tags"])
    assert tag is not None
    kinds = list(get("injected_set.construction.s4.out_of_range_kinds"))
    random.shuffle(kinds)
    for kind in kinds:
        applied = corrupt_value_range(tag, str(record["tags"][tag]), kind)
        if applied is None:
            continue
        verdicts = rules.value_range({tag: applied.injected})
        if any(verdict.outcome == rules.FLAG for verdict in verdicts):
            return applied
    return None


# ---------------------------------------------------------------------------
# Item assembly
# ---------------------------------------------------------------------------


def shown_tags(record: dict[str, Any]) -> dict[str, Any]:
    """Return the item's fields as a tag set the rules can read.

    Args:
        record: A corpus record.

    Returns:
        The shown fields without the coordinate, which the rules take
        separately.
    """
    view = item_view(record)
    return {key: value for key, value in view.items() if key not in {"lat", "lon"}}


def preexisting_flags(
    record: dict[str, Any], gazetteer: Gazetteer
) -> list[dict[str, str]]:
    """Run the baseline over a record before anything is injected into it.

    Args:
        record: A corpus record.
        gazetteer: The loaded gazetteer.

    Returns:
        One entry per check that already flagged the untouched record.
    """
    probe = {**record, "tags": shown_tags(record)}
    return [
        {"check": verdict.check, "field": verdict.field or "", "detail": verdict.detail}
        for verdict in rules.baseline(probe, gazetteer)
        if verdict.outcome == rules.FLAG
    ]


def build_item(
    record: dict[str, Any],
    item_class: str,
    pair_id: str,
    item_id: str,
    corruption: Corruption | None,
    gazetteer: Gazetteer,
) -> dict[str, Any]:
    """Assemble one item, corrupted or clean.

    Args:
        record: The corpus record the item is built from.
        item_class: The class the pair belongs to.
        pair_id: The identifier shared by the two members of the pair.
        item_id: The item's own identifier.
        corruption: The corruption applied, or None for the clean member.
        gazetteer: The loaded gazetteer, for the pre-existing flag record.

    Returns:
        The item as it is written to disk.
    """
    view = item_view(record)
    flags = preexisting_flags(record, gazetteer)
    if corruption is not None:
        view[corruption.field] = corruption.injected
    return {
        "item_id": item_id,
        "pair_id": pair_id,
        "class": item_class,
        "corrupted": corruption is not None,
        "corrupted_field": corruption.field if corruption else None,
        "city": record["city"],
        "country": record["country"],
        "language": record["language"],
        "maturity": record["maturity"],
        "osm_type": record["osm_type"],
        "osm_id": record["osm_id"],
        "field_signature": list(field_signature(record["tags"])),
        "record": view,
        "corruption": (
            {
                "field": corruption.field,
                "kind": corruption.kind,
                "original": corruption.original,
                "injected": corruption.injected,
                "donor": corruption.donor,
            }
            if corruption
            else None
        ),
        "preexisting_flags": flags,
    }


def signature_groups(
    records: Iterable[dict[str, Any]], used: set[tuple[str, int]], item_class: str
) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    """Group a city's unused eligible records by field-presence signature.

    Args:
        records: The city's corpus records.
        used: Records already spent on another item.
        item_class: The class being built.

    Returns:
        Signature to the records carrying it, keeping only the groups holding
        at least two records, since a pair needs both members from one group.
    """
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if (record["osm_type"], record["osm_id"]) in used:
            continue
        if not eligible(record, item_class):
            continue
        groups[field_signature(record["tags"])].append(record)
    return {key: value for key, value in groups.items() if len(value) >= 2}


def build_city_class(
    city: str,
    records: list[dict[str, Any]],
    item_class: str,
    count: int,
    used: set[tuple[str, int]],
    context: dict[str, Any],
    gazetteer: Gazetteer,
    counter: Counter[str],
) -> list[dict[str, Any]]:
    """Build one class's pairs for one city.

    Args:
        city: The city name.
        records: The city's corpus records.
        item_class: The class being built.
        count: How many pairs to build.
        used: Records already spent, updated in place.
        context: The per-city material the corruptions draw from.
        gazetteer: The loaded gazetteer.
        counter: Running item counter per class, for the item identifiers.

    Returns:
        The items, two per pair.

    Raises:
        RuntimeError: If the city cannot supply the pairs the allocation asks
            for, which means the allocation and the corpus disagree and the
            design has to be revisited rather than quietly shrunk.
    """
    groups = signature_groups(records, used, item_class)
    items: list[dict[str, Any]] = []
    built = 0
    attempts = 0
    while built < count:
        attempts += 1
        if attempts > count * 50 + 200:
            raise RuntimeError(
                f"{city} cannot supply {count} {item_class} pairs; built {built}"
            )
        viable = [key for key, group in groups.items() if len(group) >= 2]
        if not viable:
            raise RuntimeError(f"{city} ran out of {item_class} signature groups")
        weights = [len(groups[key]) for key in viable]
        signature = random.choices(viable, weights=weights, k=1)[0]
        group = groups[signature]
        first, second = random.sample(range(len(group)), 2)
        target, partner = group[first], group[second]
        if random.random() < 0.5:
            target, partner = partner, target

        applied = (
            corrupt_range(target)
            if item_class == "S4"
            else corrupt(item_class, target, context)
        )
        if applied is None:
            group.remove(target)
            if len(group) < 2:
                groups.pop(signature, None)
            continue

        counter[item_class] += 1
        pair_id = f"{item_class}-{counter[item_class]:04d}"
        # The two members are written in a random order and numbered by
        # position, so that an identifier never says which one was corrupted.
        members = [(target, applied), (partner, None)]
        random.shuffle(members)
        for index, (record, corruption) in enumerate(members, start=1):
            items.append(
                build_item(
                    record,
                    item_class,
                    pair_id,
                    f"{pair_id}-{index}",
                    corruption,
                    gazetteer,
                )
            )
        for record in (target, partner):
            used.add((record["osm_type"], record["osm_id"]))
            group.remove(record)
        if len(group) < 2:
            groups.pop(signature, None)
        built += 1
    return items


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def write_outputs(items: list[dict[str, Any]], manifest: dict[str, Any]) -> Path:
    """Write the item set, its manifest and the per-class counts.

    Args:
        items: Every item.
        manifest: The construction record.

    Returns:
        The item file written.
    """
    out_dir = path("data_processed") / "injected_set"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "items.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"wrote {target} with {len(items)} items")

    rows: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for item in items:
        key = (str(item["class"]), str(item["city"]))
        rows[key]["corrupted" if item["corrupted"] else "clean"] += 1
    counts_path = out_dir / "class_counts.csv"
    with counts_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "city", "corrupted", "clean"])
        for (item_class, city), tally in sorted(rows.items()):
            writer.writerow([item_class, city, tally["corrupted"], tally["clean"]])
    logger.info(f"wrote {counts_path}")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(f"wrote {manifest_path}")
    return target


def main() -> None:
    """Build the injected set and write it with its construction record."""
    seed = seed_everything()
    gazetteer = load_gazetteer()
    cities = [str(city["city"]) for city in get("corpus.cities")]
    records_by_city = {city: read_city(city) for city in cities}
    logger.info(f"read {sum(len(r) for r in records_by_city.values())} corpus records")

    tokens = category_tokens(records_by_city)
    write_category_tokens(tokens)
    logger.info(f"{len(tokens)} category-indicative tokens")

    classes: dict[str, Any] = get("injected_set.classes")
    allocation = {
        name: allocate(int(classes[name]["items"]), cities) for name in CLASS_ORDER
    }

    used: set[tuple[str, int]] = set()
    counter: Counter[str] = Counter()
    items: list[dict[str, Any]] = []
    for city in cities:
        records = records_by_city[city]
        country = str(records[0]["country"])
        context = {
            "donor_pool": donor_pool(records, tokens),
            "calling_codes": other_calling_codes(country),
            "places": gazetteer.far_places(
                country,
                float(records[0]["lat"]),
                float(records[0]["lon"]),
                0.0,
            ),
            "gazetteer": gazetteer,
        }
        logger.info(f"{city}: {len(context['donor_pool'])} donor names available")
        for item_class in CLASS_ORDER:
            count = allocation[item_class][city]
            items.extend(
                build_city_class(
                    city,
                    records,
                    item_class,
                    count,
                    used,
                    context,
                    gazetteer,
                    counter,
                )
            )
        logger.info(f"{city}: {sum(1 for i in items if i['city'] == city)} items")

    manifest = {
        "built_on": date.today().isoformat(),
        "seed": seed,
        "corpus_extraction_date": str(get("corpus.extraction_date")),
        "items": len(items),
        "corrupted": sum(1 for item in items if item["corrupted"]),
        "clean": sum(1 for item in items if not item["corrupted"]),
        "classes": {name: int(classes[name]["items"]) for name in CLASS_ORDER},
        "allocation": allocation,
        "category_tokens": len(tokens),
        "pairing": get("injected_set.construction.pairing"),
        "references": get("references.calling_codes"),
        "items_with_preexisting_flag": sum(
            1 for item in items if item["preexisting_flags"]
        ),
    }
    write_outputs(items, manifest)
    logger.info(
        f"built {manifest['corrupted']} corrupted and {manifest['clean']} clean items"
    )


if __name__ == "__main__":
    main()
