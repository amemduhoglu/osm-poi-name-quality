"""Build the residual control set: borrowed names that no shortcut can find.

The residual class of the injected set is separable by its construction, which
the decision of 2026-09-11 named and scripts/measure_item_artefacts.py had
already measured. Two cues carry it. The donor keeps its record, so every
borrowed name is repeated in its city; and the donor filters were applied to
the donor alone, so the clean half carries short names and category tokens the
corrupted half cannot. This set is built so that neither cue exists:

    Each city is split once into a withheld and a visible part. Donor names are
    drawn from the withheld part only and must be unique across the whole city,
    so a borrowed name appears nowhere in the visible part. Both members of every
    pair are visible records, and every instrument that reads the city's corpus
    is given the visible part alone.

    The clean partner's own name is held to the donor filters, counted against
    the visible part, so the two halves are selected on the same terms.

Everything else follows the injected set's own builder, whose functions are
imported rather than copied: pairing on city and field signature, a donor from a
different primary category at least the configured distance away, the item
schema, the pre-existing flag record. Records spent on the injected set are not
reused. Pairs are assigned to train, development and test splits for the
adaptation arms.

Outputs:
    data/processed/injected_set/residual_control.jsonl
    data/processed/injected_set/residual_control_visible.json
    data/processed/injected_set/residual_control_manifest.json

Usage:
    python -m scripts.build_residual_control
"""

from __future__ import annotations

import json
import random
import subprocess
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from poi_audit import screen
from poi_audit.config import get, path, seed_everything
from poi_audit.corpus import field_signature, fold, read_city
from poi_audit.corruptions import corrupt_name
from poi_audit.logsetup import setup
from poi_audit.references import load_gazetteer
from poi_audit.withheld import clean_name_eligible, name_counts, split_city
from scripts.build_injected_set import (
    allocate,
    build_item,
    donor_pool,
    draw_name_donor,
    eligible,
)

logger = setup("build_residual_control")

RESIDUAL = "M1"
PAIR_PREFIX = "M1C"


def out_dir() -> Path:
    """Return the directory the control set is written to."""
    target = path("data_processed") / "injected_set"
    target.mkdir(parents=True, exist_ok=True)
    return target


def injected_identities() -> set[tuple[str, int]]:
    """Return the records the injected set already spent."""
    source = out_dir() / "items.jsonl"
    with source.open(encoding="utf-8") as handle:
        return {
            (str(item["osm_type"]), int(item["osm_id"]))
            for item in (json.loads(line) for line in handle if line.strip())
        }


def city_donors(
    records: list[dict[str, Any]],
    withheld: list[dict[str, Any]],
    tokens: dict[str, str],
) -> list[dict[str, Any]]:
    """Return the donor pool for one city, withheld records only.

    The injected builder's pool is computed over the whole city, which makes its
    uniqueness test a test over the whole city, and is then restricted to the
    withheld part. The folded count and the stricter brand test are applied on
    top, so a donor passes exactly the filters a clean partner is held to.

    Args:
        records: The city's corpus records.
        withheld: The city's withheld part.
        tokens: The category-token dictionary.

    Returns:
        The eligible donor records.
    """
    hidden = {screen.identity(record) for record in withheld}
    counts = name_counts(records)
    return [
        donor
        for donor in donor_pool(records, tokens)
        if screen.identity(donor) in hidden
        and clean_name_eligible(donor, counts, tokens)
    ]


def candidate_groups(
    visible: list[dict[str, Any]],
    spent: set[tuple[str, int]],
    tokens: dict[str, str],
) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    """Group the visible records that may enter a pair by field signature.

    Args:
        visible: The city's visible part.
        spent: Records already used, by the injected set or by this set.
        tokens: The category-token dictionary.

    Returns:
        Signature to its candidate records, keeping groups of two or more.
    """
    counts = name_counts(visible)
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in visible:
        if screen.identity(record) in spent:
            continue
        if not eligible(record, RESIDUAL):
            continue
        if not clean_name_eligible(record, counts, tokens):
            continue
        groups[field_signature(record["tags"])].append(record)
    return {key: rows for key, rows in groups.items() if len(rows) >= 2}


def build_city(
    city: str,
    visible: list[dict[str, Any]],
    donors: list[dict[str, Any]],
    count: int,
    spent: set[tuple[str, int]],
    tokens: dict[str, str],
    gazetteer: Any,
    counter: Counter[str],
) -> list[dict[str, Any]]:
    """Build one city's pairs.

    Args:
        city: The city name.
        visible: The city's visible part.
        donors: The city's donor pool.
        count: How many pairs to build.
        spent: Records already used, updated in place.
        tokens: The category-token dictionary.
        gazetteer: The loaded gazetteer, for the pre-existing flag record.
        counter: Running pair counter.

    Returns:
        The items, two per pair.

    Raises:
        RuntimeError: If the city cannot supply the pairs, which means the
            allocation and the visible part disagree and the design has to be
            revisited rather than quietly shrunk.
    """
    groups = candidate_groups(visible, spent, tokens)
    items: list[dict[str, Any]] = []
    built = 0
    attempts = 0
    while built < count:
        attempts += 1
        if attempts > count * 50 + 200:
            raise RuntimeError(f"{city} cannot supply {count} pairs; built {built}")
        viable = [key for key, rows in groups.items() if len(rows) >= 2]
        if not viable:
            raise RuntimeError(f"{city} ran out of signature groups at {built}")
        weights = [len(groups[key]) for key in viable]
        signature = random.choices(viable, weights=weights, k=1)[0]
        group = groups[signature]
        first, second = random.sample(range(len(group)), 2)
        target, partner = group[first], group[second]
        if random.random() < 0.5:
            target, partner = partner, target

        donor = draw_name_donor(target, donors)
        if donor is None:
            group.remove(target)
            if len(group) < 2:
                groups.pop(signature, None)
            continue
        applied = corrupt_name(donor, str(target["name"]))

        counter[PAIR_PREFIX] += 1
        pair_id = f"{PAIR_PREFIX}-{counter[PAIR_PREFIX]:04d}"
        members = [(target, applied), (partner, None)]
        random.shuffle(members)
        for index, (record, corruption) in enumerate(members, start=1):
            items.append(
                build_item(
                    record,
                    RESIDUAL,
                    pair_id,
                    f"{pair_id}-{index}",
                    corruption,
                    gazetteer,
                )
            )
        for record in (target, partner):
            spent.add(screen.identity(record))
            group.remove(record)
        if len(group) < 2:
            groups.pop(signature, None)
        built += 1
    return items


def assign_splits(items: list[dict[str, Any]], seed: int) -> None:
    """Assign every pair to one split, in place.

    Pairs are shuffled under their own generator and cut at the configured
    shares, so the split sizes are exact rather than expected, and both members
    of a pair always share a split.

    Args:
        items: The items, carrying their pair identifiers.
        seed: The configured seed.
    """
    shares: dict[str, float] = get("residual_control.splits")
    pairs = sorted({str(item["pair_id"]) for item in items})
    random.Random(seed).shuffle(pairs)
    cuts: dict[str, str] = {}
    start = 0
    names = list(shares)
    for position, name in enumerate(names):
        size = (
            len(pairs) - start
            if position == len(names) - 1
            else round(shares[name] * len(pairs))
        )
        for pair in pairs[start : start + size]:
            cuts[pair] = name
        start += size
    for item in items:
        item["split"] = cuts[str(item["pair_id"])]


def config_commit() -> str:
    """Return the commit the configuration was read at."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def main() -> None:
    """Build the control set and write it with its construction record."""
    seed = seed_everything()
    settings = get("residual_control")
    gazetteer = load_gazetteer()
    tokens = screen.load_category_tokens()
    cities = [str(city["city"]) for city in get("corpus.cities")]
    spent = injected_identities() if settings["exclude_injected_set_records"] else set()
    allocation = allocate(int(settings["items"]), cities)

    counter: Counter[str] = Counter()
    items: list[dict[str, Any]] = []
    visible_by_city: dict[str, dict[str, Any]] = {}
    pool_sizes: dict[str, dict[str, int]] = {}
    for city in cities:
        records = read_city(city)
        withheld, visible = split_city(
            records, float(settings["withheld_share"]), int(settings["seed"]), city
        )
        donors = city_donors(records, withheld, tokens)
        built = build_city(
            city,
            visible,
            donors,
            allocation[city],
            spent,
            tokens,
            gazetteer,
            counter,
        )
        items.extend(built)
        visible_by_city[city] = {
            "records": [list(screen.identity(record)) for record in visible],
            "names": [str(record.get("name") or "") for record in visible],
        }
        pool_sizes[city] = {
            "records": len(records),
            "withheld": len(withheld),
            "visible": len(visible),
            "donors": len(donors),
            "pairs": allocation[city],
        }
        logger.info(
            f"{city}: {len(donors)} donors, {allocation[city]} pairs, "
            f"{len(visible)} visible records"
        )

    assign_splits(items, int(settings["seed"]))

    target = out_dir() / "residual_control.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (out_dir() / "residual_control_visible.json").write_text(
        json.dumps(visible_by_city, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    split_counts = Counter(str(item["split"]) for item in items)
    manifest = {
        "built_on": date.today().isoformat(),
        "seed": seed,
        "config_commit": config_commit(),
        "corpus_extraction_date": str(get("corpus.extraction_date")),
        "items": len(items),
        "corrupted": sum(1 for item in items if item["corrupted"]),
        "clean": sum(1 for item in items if not item["corrupted"]),
        "splits_items": dict(sorted(split_counts.items())),
        "cities": pool_sizes,
        "settings": settings,
        "shown_names_folded_unique": len(
            {fold(str(i["record"]["name"])) for i in items}
        ),
    }
    (out_dir() / "residual_control_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(
        f"wrote {target}: {manifest['corrupted']} corrupted, {manifest['clean']} clean"
    )


if __name__ == "__main__":
    main()
