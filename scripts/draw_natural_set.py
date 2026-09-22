"""Draw the natural set: two strata of two hundred real records each.

The injected set answers what a model does to an error whose ground truth is
exact. It cannot answer whether the residual class occurs in real data, which is
what the natural set is for.

Two strata, both with known inclusion probabilities:

    Stratum A, drawn at random from the frame. It establishes the base rate and
    supplies negatives no screen has touched.

    Stratum B, drawn from the records the string and dictionary screen raises.
    It buys labelling effort on the class the article's claim depends on, at the
    cost of a known and correctable bias.

The two are disjoint: A is drawn first from the whole frame, B from the
screen-positive records A did not take, so neither inclusion probability has to
be approximated. A is allocated equally across the twelve cities and B equally
across the three screen rules and the twelve cities alike, so that every design
cell is measurable and no rule supplies stratum B in proportion to how much of
the corpus it happens to raise. Each cell's probability is written into every
record drawn from it, so that estimates reweight to the corpus.

Every drawn record carries the screen's verdict whether or not the screen is why
it was drawn. The screen's own precision is therefore estimable from stratum A,
where the screen took no part in the selection.

Outputs:
    data/processed/natural_set/records.jsonl        the 400 records, tracked
    data/processed/natural_set/labelling_sheet.csv  blind, one shuffled order
    data/processed/natural_set/screen_counts.csv    the screen over the corpus
    data/processed/natural_set/manifest.json        frames, draws, probabilities

A second phase, stratum C, is drawn with `--phase 2` for the redesign of
2026-09: a random draw from the named records the first draw did not take,
allocated to the design cells by configured shares and equally across each
cell's cities, sized by a rule fixed in the configuration before the draw.

Outputs of the second phase:
    data/processed/natural_set/records_phase2.jsonl
    data/processed/natural_set/labelling_sheet_phase2.csv
    data/processed/natural_set/manifest_phase2.json

Usage:
    python -m scripts.draw_natural_set
    python -m scripts.draw_natural_set --phase 2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from poi_audit import screen
from poi_audit.config import get, path, seed_everything
from poi_audit.corpus import fold, item_view, read_city
from poi_audit.logsetup import setup
from poi_audit.references import load_gazetteer

logger = setup("draw_natural_set")


def output_dir() -> Path:
    """Return the directory the natural set is written to."""
    target = path("data_processed") / "natural_set"
    target.mkdir(parents=True, exist_ok=True)
    return target


def allocate(total: int, cities: list[str]) -> dict[str, int]:
    """Spread a stratum equally across the cities.

    Args:
        total: The stratum's record count.
        cities: The city names, in configuration order.

    Returns:
        Records per city. A count that does not divide by the number of cities
        gives its remainder to cities drawn from the seeded sequence, and the
        draw is written into the manifest.
    """
    base, remainder = divmod(total, len(cities))
    counts = dict.fromkeys(cities, base)
    for city in random.sample(cities, remainder):
        counts[city] += 1
    return counts


def screen_city(
    records: list[dict[str, Any]],
    tokens: dict[str, str],
    places: dict[str, list[Any]],
    settings: dict[str, dict[str, Any]],
) -> dict[tuple[str, int], list[screen.Hit]]:
    """Run one screen variant over one city.

    Args:
        records: The city's corpus records.
        tokens: Category-indicative token to the category it indicates.
        places: Folded place name to the places holding it.
        settings: The variant's rule settings.

    Returns:
        The identity of every raised record mapped to its hits. A record absent
        from the mapping is screen-negative.
    """
    grouped = screen.name_index(records)
    raised: dict[tuple[str, int], list[screen.Hit]] = {}
    for record in records:
        folded = fold(str(record.get("name") or ""))
        hits = screen.screen_record(
            record, grouped.get(folded, []), tokens, places, settings
        )
        if hits:
            raised[screen.identity(record)] = hits
    return raised


def draw_stratum(
    pool: list[dict[str, Any]], quota: int
) -> tuple[list[dict[str, Any]], int]:
    """Draw one city's share of one stratum.

    Args:
        pool: The records the stratum may draw from, in corpus order.
        quota: How many the allocation asks for.

    Returns:
        The drawn records and the size of the pool they were drawn from, which
        is the denominator of their inclusion probability. A pool smaller than
        the quota is taken whole and the shortfall is reported by the caller.
    """
    if len(pool) <= quota:
        return list(pool), len(pool)
    return random.sample(pool, quota), len(pool)


def redistribute(
    quotas: dict[str, int], available: dict[str, int], cities: list[str]
) -> dict[str, int]:
    """Move a city's shortfall to the cities that can carry it.

    A city whose screen raises fewer records than its quota asks for cannot
    supply the difference, and the stratum is kept at its configured size rather
    than shrinking silently. The surplus goes to the cities with the most
    unclaimed positives, one record at a time, so that no city takes a share of
    the stratum out of proportion to what its screen actually raised.

    Args:
        quotas: The equal allocation before any shortfall is known.
        available: How many records each city can supply.
        cities: The city names, in configuration order.

    Returns:
        The adjusted quotas, which sum to the same total unless no city has any
        record left to give.

    Raises:
        ValueError: If the adjusted quotas cannot be met at all.
    """
    adjusted = {city: min(quotas[city], available[city]) for city in cities}
    shortfall = sum(quotas.values()) - sum(adjusted.values())
    while shortfall > 0:
        spare = {
            city: available[city] - adjusted[city]
            for city in cities
            if available[city] > adjusted[city]
        }
        if not spare:
            raise ValueError(
                f"the screen raises {sum(adjusted.values())} records in total, "
                f"fewer than the {sum(quotas.values())} the stratum asks for"
            )
        city = max(sorted(spare), key=lambda name: spare[name])
        adjusted[city] += 1
        shortfall -= 1
    return adjusted


def as_item(
    record: dict[str, Any],
    stratum: str,
    probability: float,
    hits: list[screen.Hit],
    owning_rule: str | None = None,
    match: tuple[float, Any] | None = None,
) -> dict[str, Any]:
    """Write one drawn record into its natural-set form.

    Args:
        record: The corpus record.
        stratum: ``A`` or ``B``.
        probability: The record's inclusion probability in its stratum.
        hits: The screen's verdict on the record, whatever stratum it came from.
        owning_rule: The rule whose cell the record was drawn from, for stratum
            B, or None for stratum A, which no rule selected.
        match: The gazetteer place the record's name belongs to and how far
            away it is, where the gazetteer rule raised the record.

    Returns:
        The record as it is deposited: the fields a reader is shown, the
        provenance a reader needs to check it, and the sampling quantities the
        analysis reweights with.
    """
    tags = record["tags"]
    place_match = None
    if match is not None:
        away, place = match
        place_match = {
            "source": place.source,
            "place_id": place.place_id,
            "name": place.display_name,
            "distance_km": round(away, 1),
            "same_country": place.country == record["country"],
        }
    return {
        "osm_type": record["osm_type"],
        "osm_id": record["osm_id"],
        "city": record["city"],
        "country": record["country"],
        "language": record["language"],
        "maturity": record["maturity"],
        "stratum": stratum,
        "inclusion_probability": round(probability, 8),
        "screen_positive": bool(hits),
        "screen_rules": [hit.rule for hit in hits],
        "screen_evidence": [hit.evidence for hit in hits],
        "drawn_under_rule": owning_rule,
        "gazetteer_match": place_match,
        "carries_brand_tag": any(
            key == "brand" or key.startswith("brand:") for key in tags
        ),
        "shown": item_view(record),
    }


def write_records(items: list[dict[str, Any]]) -> Path:
    """Write the drawn records with their sampling quantities.

    Args:
        items: The 400 records, already in the shuffled labelling order and
            numbered by position.

    Returns:
        The file written.
    """
    target = output_dir() / "records.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"wrote {target} with {len(items)} records")
    return target


def write_labelling_sheet(
    items: list[dict[str, Any]], name: str = "labelling_sheet.csv"
) -> Path:
    """Write the sheet a reader labels from.

    The sheet carries the record and nothing else: no stratum, no screen
    verdict, no evidence and no model output. A reader who could see which
    records the screen raised would be reading the screen rather than the
    record, and the label would no longer be independent of the selection.

    Args:
        items: The records in the shuffled labelling order.
        name: The sheet's file name.

    Returns:
        The file written.
    """
    keys = list(get("injected_set.construction.pairing.item_keys"))
    target = output_dir() / name
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", *keys, "lat", "lon", "label", "evidence_url"])
        for item in items:
            shown = item["shown"]
            writer.writerow(
                [
                    item["item_id"],
                    *[shown.get(key, "") for key in keys],
                    shown["lat"],
                    shown["lon"],
                    "",
                    "",
                ]
            )
    logger.info(f"wrote {target}")
    return target


def write_screen_counts(
    frame: dict[str, int],
    counts: dict[str, tuple[dict[str, int], dict[str, Counter[str]]]],
) -> Path:
    """Write what each screen variant did to each city.

    Both variants are written. The final screen is what stratum B is drawn
    from; the first-pass screen is reported beside it so that the tightening is
    a number in the deposit rather than a claim in the text.

    Args:
        frame: Records in each city's frame.
        counts: Variant name to its per-city positives and its per-city
            per-rule counts. A record raised by two rules counts once under
            each, so the rule columns do not sum to the positive count.

    Returns:
        The file written.
    """
    target = output_dir() / "screen_counts.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["variant", "city", "frame", "screen_positive", "positive_share"]
            + list(screen.RULE_ORDER)
        )
        for variant in (screen.FIRST_PASS, screen.FINAL):
            positives, by_rule = counts[variant]
            for city in frame:
                share = positives[city] / frame[city] if frame[city] else 0.0
                writer.writerow(
                    [variant, city, frame[city], positives[city], round(share, 6)]
                    + [by_rule[city][rule] for rule in screen.RULE_ORDER]
                )
            total_frame = sum(frame.values())
            total_positive = sum(positives.values())
            writer.writerow(
                [
                    variant,
                    "all",
                    total_frame,
                    total_positive,
                    round(total_positive / total_frame, 6) if total_frame else 0.0,
                ]
                + [
                    sum(by_rule[city][rule] for city in frame)
                    for rule in screen.RULE_ORDER
                ]
            )
    logger.info(f"wrote {target}")
    return target


# ---------------------------------------------------------------------------
# Second phase
# ---------------------------------------------------------------------------


def phase2_size(
    rates: dict[str, float],
    shares: dict[str, float],
    existing_expected: float,
    target: int,
) -> int:
    """Return the smallest second-phase draw expected to reach the target.

    Args:
        rates: Wrong names per labelled record, per design cell.
        shares: Share of the draw per design cell.
        existing_expected: Wrong names expected across every record drawn before.
        target: Wrong names wanted across every labelled record.

    Returns:
        Records to draw; zero when the target is already expected.
    """
    per_record = sum(rates[cell] * shares[cell] for cell in shares)
    missing = target - existing_expected
    if missing <= 0:
        return 0
    # The epsilon keeps a size that meets the target exactly from rounding up
    # past it on floating-point noise.
    return math.ceil(missing / per_record - 1e-9)


def cell_quotas(total: int, shares: dict[str, float]) -> dict[str, int]:
    """Split a draw across design cells by largest remainder.

    Args:
        total: Records to draw.
        shares: Share per cell, summing to one.

    Returns:
        Records per cell, summing to the total. Ties in the remainder go to the
        cell listed first, so the split does not depend on a generator.
    """
    exact = {cell: total * share for cell, share in shares.items()}
    quotas = {cell: math.floor(value) for cell, value in exact.items()}
    left = total - sum(quotas.values())
    order = sorted(shares, key=lambda cell: -(exact[cell] - quotas[cell]))
    for cell in order[:left]:
        quotas[cell] += 1
    return quotas


def expected_from_first_draw(
    records: list[dict[str, Any]], rates: dict[str, float], labelled: int, observed: int
) -> float:
    """Return the wrong names expected across every record of the first draw.

    Args:
        records: The first draw, numbered in its fixed order.
        rates: Wrong names per labelled record, per design cell.
        labelled: How many of the fixed order were labelled.
        observed: Wrong names found among them.

    Returns:
        The observed count plus the expectation over the unlabelled rest.
    """
    rest = [item for item in records if int(str(item["item_id"])[1:]) > labelled]
    return observed + sum(
        rates[f"{item['language']}, {item['maturity']}"] for item in rest
    )


def main_phase2() -> None:
    """Draw stratum C and write it beside the first draw."""
    seed_everything()
    settings = get("natural_set.phase2")
    random.seed(int(settings["seed"]))
    shares = {str(k): float(v) for k, v in settings["cell_shares"].items()}
    rates = {str(k): float(v) for k, v in settings["wrong_per_labelled_record"].items()}

    with (output_dir() / "records.jsonl").open(encoding="utf-8") as handle:
        first = [json.loads(line) for line in handle if line.strip()]
    existing = expected_from_first_draw(
        first,
        rates,
        int(get("natural_set.labelled.records")),
        int(settings["observed_wrong_first_250"]),
    )
    size = phase2_size(
        rates, shares, existing, int(settings["target_wrong_all_labelled"])
    )
    quotas = cell_quotas(size, shares)
    logger.info(
        f"expected from the first draw {existing:.2f}; drawing {size}: {quotas}"
    )

    taken_before = {(str(item["osm_type"]), int(item["osm_id"])) for item in first}
    tokens = screen.load_category_tokens()
    gazetteer = load_gazetteer()
    places = screen.place_index(gazetteer)
    settings_final = screen.rule_settings(screen.FINAL)

    by_cell: dict[str, list[str]] = {cell: [] for cell in shares}
    for entry in get("corpus.cities"):
        by_cell[f"{entry['language']}, {entry['maturity']}"].append(str(entry["city"]))

    drawn: list[dict[str, Any]] = []
    record_of_draw: dict[str, dict[str, Any]] = {}
    for cell, cities in by_cell.items():
        per_city = allocate(quotas[cell], cities)
        for city in cities:
            named = [record for record in read_city(city) if record.get("name")]
            pool = [r for r in named if screen.identity(r) not in taken_before]
            taken, pool_size = draw_stratum(pool, per_city[city])
            if len(taken) < per_city[city]:
                raise RuntimeError(f"{city} cannot supply {per_city[city]} records")
            grouped = screen.name_index(named)
            for record in taken:
                hits = screen.screen_record(
                    record,
                    grouped.get(fold(str(record.get("name") or "")), []),
                    tokens,
                    places,
                    settings_final,
                )
                drawn.append(
                    as_item(
                        record,
                        str(settings["stratum"]),
                        per_city[city] / pool_size,
                        hits,
                        match=screen.place_match(record, places),
                    )
                )
            record_of_draw[city] = {
                "cell": cell,
                "quota": per_city[city],
                "pool": pool_size,
                "inclusion_probability": round(per_city[city] / pool_size, 8),
            }
            logger.info(f"{city}: {per_city[city]} of {pool_size}")

    random.shuffle(drawn)
    start = int(settings["first_item_number"])
    for position, item in enumerate(drawn, start=start):
        item["item_id"] = f"N{position:03d}"

    target = output_dir() / "records_phase2.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        for item in drawn:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    write_labelling_sheet(drawn, "labelling_sheet_phase2.csv")
    manifest = {
        "built_on": date.today().isoformat(),
        "seed": int(settings["seed"]),
        "corpus_extraction_date": str(get("corpus.extraction_date")),
        "expected_wrong_from_first_draw": round(existing, 4),
        "size": size,
        "cell_quotas": quotas,
        "expected_wrong_after_phase2": round(
            existing + sum(rates[cell] * quotas[cell] for cell in quotas), 4
        ),
        "settings": settings,
        "by_city": record_of_draw,
    }
    (output_dir() / "manifest_phase2.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(f"wrote {target} with {len(drawn)} records")


def main() -> None:
    """Screen the corpus, draw both strata and write the natural set."""
    seed = seed_everything()
    cities = [str(entry["city"]) for entry in get("corpus.cities")]
    wanted_a = int(get("natural_set.strata.A.records"))
    wanted_b = int(get("natural_set.strata.B.records"))

    tokens = screen.load_category_tokens()
    gazetteer = load_gazetteer()
    places = screen.place_index(gazetteer)
    logger.info(
        f"screening with {len(tokens)} category tokens and "
        f"{len(gazetteer.places)} gazetteer places under {len(places)} names"
    )

    records_by_city: dict[str, list[dict[str, Any]]] = {}
    hits_by_city: dict[str, dict[tuple[str, int], list[screen.Hit]]] = {}
    frame: dict[str, int] = {}
    counts: dict[str, tuple[dict[str, int], dict[str, Counter[str]]]] = {
        variant: ({}, {}) for variant in (screen.FIRST_PASS, screen.FINAL)
    }
    for city in cities:
        records = [record for record in read_city(city) if record.get("name")]
        records_by_city[city] = records
        frame[city] = len(records)
        for variant in (screen.FIRST_PASS, screen.FINAL):
            raised = screen_city(records, tokens, places, screen.rule_settings(variant))
            positives, by_rule = counts[variant]
            positives[city] = len(raised)
            counter: Counter[str] = Counter()
            for hits in raised.values():
                for hit in hits:
                    counter[hit.rule] += 1
            by_rule[city] = counter
            if variant == screen.FINAL:
                hits_by_city[city] = raised
        logger.info(
            f"{city}: {frame[city]} in frame, "
            f"{counts[screen.FIRST_PASS][0][city]} raised by the first-pass "
            f"screen, {counts[screen.FINAL][0][city]} by the final one"
        )

    final_positive = counts[screen.FINAL][0]
    owner: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for city in cities:
        by_owner: dict[str, list[dict[str, Any]]] = {
            rule: [] for rule in screen.RULE_ORDER
        }
        for record in records_by_city[city]:
            hits = hits_by_city[city].get(screen.identity(record))
            if hits:
                by_owner[screen.owning_rule(hits)].append(record)
        owner[city] = by_owner

    quota_a = allocate(wanted_a, cities)
    priority = [str(rule) for rule in get("natural_set.screen.rule_priority")]
    per_rule = allocate(wanted_b, priority)
    quota_b = {
        rule: redistribute(
            allocate(per_rule[rule], cities),
            {city: len(owner[city][rule]) for city in cities},
            cities,
        )
        for rule in priority
    }

    drawn: list[dict[str, Any]] = []
    draw_record: dict[str, dict[str, Any]] = {}
    for city in cities:
        taken_a, pool_a = draw_stratum(records_by_city[city], quota_a[city])
        taken = {screen.identity(record) for record in taken_a}
        for record in taken_a:
            drawn.append(
                as_item(
                    record,
                    "A",
                    quota_a[city] / pool_a,
                    hits_by_city[city].get(screen.identity(record), []),
                    match=screen.place_match(record, places),
                )
            )
        cells: dict[str, dict[str, Any]] = {}
        for rule in priority:
            available = [
                record
                for record in owner[city][rule]
                if screen.identity(record) not in taken
            ]
            taken_b, pool_b = draw_stratum(available, quota_b[rule][city])
            for record in taken_b:
                drawn.append(
                    as_item(
                        record,
                        "B",
                        quota_b[rule][city] / pool_b if pool_b else 1.0,
                        hits_by_city[city][screen.identity(record)],
                        rule,
                        screen.place_match(record, places),
                    )
                )
            cells[rule] = {
                "quota": quota_b[rule][city],
                "drawn": len(taken_b),
                "pool": pool_b,
                "inclusion_probability": (
                    round(quota_b[rule][city] / pool_b, 8) if pool_b else None
                ),
            }
        draw_record[city] = {
            "frame": frame[city],
            "screen_positive": final_positive[city],
            "stratum_a": {
                "quota": quota_a[city],
                "drawn": len(taken_a),
                "pool": pool_a,
                "inclusion_probability": round(quota_a[city] / pool_a, 8),
            },
            "stratum_b": cells,
        }

    random.shuffle(drawn)
    for position, item in enumerate(drawn, start=1):
        item["item_id"] = f"N{position:03d}"

    write_records(drawn)
    write_labelling_sheet(drawn)
    write_screen_counts(frame, counts)

    manifest = {
        "built_on": date.today().isoformat(),
        "seed": seed,
        "corpus_extraction_date": str(get("corpus.extraction_date")),
        "frame": {
            "definition": "every corpus record carrying a name",
            "records": sum(frame.values()),
        },
        "allocation": get("natural_set.allocation"),
        "strata_disjoint": bool(get("natural_set.strata_disjoint")),
        "screen": {
            "kind": get("natural_set.screen.kind"),
            "models_permitted": get("natural_set.screen.models_permitted"),
            "allocation": get("natural_set.screen.allocation"),
            "rule_priority": priority,
            "rules": get("natural_set.screen.rules"),
            "first_pass_rules": get("natural_set.screen.first_pass_rules"),
            "positive": {
                variant: sum(counts[variant][0].values())
                for variant in (screen.FIRST_PASS, screen.FINAL)
            },
            "positive_share": {
                variant: round(
                    sum(counts[variant][0].values()) / sum(frame.values()), 6
                )
                for variant in (screen.FIRST_PASS, screen.FINAL)
            },
            "owned_by_rule": {
                rule: sum(len(owner[city][rule]) for city in cities)
                for rule in priority
            },
            "stratum_b_per_rule": per_rule,
        },
        "gazetteer": {
            "sources": gazetteer.sources(),
            "places": len(gazetteer.places),
        },
        "drawn": {
            "A": sum(1 for item in drawn if item["stratum"] == "A"),
            "B": sum(1 for item in drawn if item["stratum"] == "B"),
            "total": len(drawn),
        },
        "by_city": draw_record,
    }
    target = output_dir() / "manifest.json"
    target.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(f"wrote {target}")
    logger.info(
        f"natural set holds {len(drawn)} records: "
        f"{manifest['drawn']['A']} in stratum A and {manifest['drawn']['B']} in B"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", type=int, choices=(1, 2), default=1)
    if parser.parse_args().phase == 2:
        main_phase2()
    else:
        main()
