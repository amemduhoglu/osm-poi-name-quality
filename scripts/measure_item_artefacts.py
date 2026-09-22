"""Measure what the residual class's construction leaves on the items.

Five filters choose the name the corrupted half carries: the donor sits in the
same city and at least two kilometres away, its name is unique in that city,
carries no brand tag, carries no category-indicative token, and is at least six
characters long. The pairing that supplies the clean half matches on the city
and the field signature and says nothing about the name, so the clean half
carries whatever the record already had.

Anything that separates the two halves without reading a map is a way to score
the class without answering its question. Two kinds are measured apart, because
they are available to different instruments:

    What one record shows. A name shorter than the donor minimum, or one
    carrying a category token the donor filter excluded, cannot have been
    injected, and a model shown that record alone can act on it.

    What the city's corpus shows. The donor keeps its own record, so an
    injected name is repeated in the city by construction, and a brand-tagged
    name is never injected. A corpus-level rule can act on those; a model shown
    one record cannot.

The residual control set (scripts/build_residual_control.py) is measured the
same way with `--set residual_control`, against the visible part of each city
that every corpus-level instrument is given, and with a third shortcut beside
the two: the best single cut on the shown name's length. Its outputs carry the
acceptance verdict the configuration fixed before the set was built.

Outputs:
    data/processed/analysis/item_artefacts.csv
    data/processed/analysis/item_artefacts.json
    data/processed/analysis/item_artefacts_control.csv       (--set residual_control)
    data/processed/analysis/item_artefacts_control.json      (--set residual_control)

Usage:
    python scripts/measure_item_artefacts.py
    python -m scripts.measure_item_artefacts --set residual_control
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from poi_audit import screen
from poi_audit.config import get, path
from poi_audit.corpus import fold, read_corpus
from poi_audit.logsetup import setup
from poi_audit.stats import Confusion, wilson

logger = setup("measure_item_artefacts")

RESIDUAL = "M1"


def features(item: dict[str, Any], tokens: dict[str, str]) -> dict[str, Any]:
    """Return the construction's marks on one item.

    Args:
        item: One residual-class item, carrying the corpus facts beside it.
        tokens: Category-indicative token to the category it indicates.

    Returns:
        The features the two halves can differ on.
    """
    name = str(item["record"].get("name", ""))
    return {
        "item_id": item["item_id"],
        "corrupted": bool(item["corrupted"]),
        "name_chars": len(name),
        "name_tokens": len(name.split()),
        "carries_category_token": screen.indicated_category(name, tokens) is not None,
        "brand_tagged": bool(item.get("brand_tagged")),
        "repeated_in_city": bool(item.get("repeated_in_city")),
    }


def score_flags(
    items: list[dict[str, Any]],
    tokens: dict[str, str],
    flags: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    """Score a rule that separates the halves without reading a map.

    Args:
        items: The residual class.
        tokens: Category-indicative token to the category it indicates.
        flags: Whether the rule calls one item's measured features corrupted.

    Returns:
        The confusion and its scores, with the interval on Youden's J, since
        what the shortcut settles is read against what the instruments reach.
    """
    counts = {"tp": 0, "fn": 0, "tn": 0, "fp": 0}
    for item in items:
        measured = features(item, tokens)
        called = flags(measured)
        if measured["corrupted"]:
            counts["tp" if called else "fn"] += 1
        else:
            counts["fp" if called else "tn"] += 1
    confusion = Confusion(
        true_positive=counts["tp"],
        false_negative=counts["fn"],
        true_negative=counts["tn"],
        false_positive=counts["fp"],
    )
    low, high = confusion.youdens_j_interval()
    return {
        "true_positive": confusion.true_positive,
        "false_negative": confusion.false_negative,
        "true_negative": confusion.true_negative,
        "false_positive": confusion.false_positive,
        "recall": round(confusion.sensitivity().estimate, 4),
        "false_positive_rate": round(1.0 - confusion.specificity().estimate, 4),
        "balanced_accuracy": round(confusion.balanced_accuracy, 4),
        "youdens_j": round(confusion.youdens_j, 4),
        "youdens_j_low": round(low, 4),
        "youdens_j_high": round(high, 4),
    }


def read_items() -> list[dict[str, Any]]:
    """Read the residual class with the corpus facts its halves differ on.

    Returns:
        Every residual-class item, carrying whether the underlying record is
        brand-tagged and whether the name shown is repeated in its city.
    """
    target = path("data_processed") / "injected_set" / "items.jsonl"
    with target.open(encoding="utf-8") as handle:
        items = [json.loads(line) for line in handle if line.strip()]
    items = [one for one in items if one["class"] == RESIDUAL]

    by_city: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_identity: dict[tuple[str, int], dict[str, Any]] = {}
    for record in read_corpus():
        by_city[str(record["city"])].append(record)
        by_identity[screen.identity(record)] = record
    indexes = {city: screen.name_index(rows) for city, rows in by_city.items()}

    for item in items:
        base = by_identity[(str(item["osm_type"]), int(item["osm_id"]))]
        item["brand_tagged"] = screen.carries_brand_tag(base)
        same = indexes[str(item["city"])].get(fold(str(item["record"]["name"])), [])
        item["repeated_in_city"] = any(
            screen.identity(one) != screen.identity(base) for one in same
        )
    return items


def oracle_cut(
    items: list[dict[str, Any]], tokens: dict[str, str], feature: str
) -> dict[str, Any]:
    """Score the best single cut on one numeric feature, in either direction.

    The cut is chosen on the scored items themselves, so the result is an upper
    bound on what the feature can separate rather than a rule anyone could
    have written in advance.

    Args:
        items: The residual class.
        tokens: Category-indicative token to the category it indicates.
        feature: A numeric feature returned by `features`.

    Returns:
        The best cut, its direction and its scores.
    """
    measured = [features(item, tokens) for item in items]
    values = sorted({int(one[feature]) for one in measured})
    best: dict[str, Any] | None = None
    for cut in values:
        for direction in ("at_least", "below"):

            def flags(
                one: dict[str, Any], cut: int = cut, direction: str = direction
            ) -> bool:
                above = int(one[feature]) >= cut
                return above if direction == "at_least" else not above

            scored = score_flags(items, tokens, flags)
            if best is None or scored["youdens_j"] > best["youdens_j"]:
                best = {
                    "feature": feature,
                    "cut": cut,
                    "direction": direction,
                    **scored,
                }
    assert best is not None
    return best


def read_control_items() -> list[dict[str, Any]]:
    """Read the residual control set with the corpus facts its halves differ on.

    The facts are read against the visible part of each city, which is the only
    part of the corpus any instrument is given for this set.

    Returns:
        Every control item, carrying whether its record is brand-tagged and
        whether the name shown is carried by another visible record.
    """
    base = path("data_processed") / "injected_set"
    with (base / "residual_control.jsonl").open(encoding="utf-8") as handle:
        items = [json.loads(line) for line in handle if line.strip()]
    visible = json.loads((base / "residual_control_visible.json").read_text("utf-8"))
    wanted = {
        city: {tuple(i) for i in part["records"]} for city, part in visible.items()
    }
    by_city: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_identity: dict[tuple[str, int], dict[str, Any]] = {}
    for record in read_corpus():
        identity = screen.identity(record)
        if identity in wanted.get(str(record["city"]), set()):
            by_city[str(record["city"])].append(record)
            by_identity[identity] = record
    indexes = {city: screen.name_index(rows) for city, rows in by_city.items()}
    for item in items:
        own = (str(item["osm_type"]), int(item["osm_id"]))
        item["brand_tagged"] = screen.carries_brand_tag(by_identity[own])
        same = indexes[str(item["city"])].get(fold(str(item["record"]["name"])), [])
        # The corrupted member's own record now shows the borrowed name, so any
        # visible record carrying that name other than this one is a repeat.
        item["repeated_in_city"] = any(screen.identity(one) != own for one in same)
    return items


def summarize(items: list[dict[str, Any]], tokens: dict[str, str]) -> dict[str, Any]:
    """Count each feature on each half of the class.

    Args:
        items: The residual class.
        tokens: Category-indicative token to the category it indicates.

    Returns:
        Per half, the share of items carrying each feature.
    """
    halves: dict[str, list[dict[str, Any]]] = {"corrupted": [], "clean": []}
    for item in items:
        halves["corrupted" if item["corrupted"] else "clean"].append(
            features(item, tokens)
        )

    rows: list[dict[str, Any]] = []
    for half, measured in halves.items():
        total = len(measured)
        for feature in ("carries_category_token", "brand_tagged", "repeated_in_city"):
            carrying = sum(1 for one in measured if one[feature])
            share = wilson(carrying, total)
            rows.append(
                {
                    "half": half,
                    "feature": feature,
                    "items": total,
                    "carrying": carrying,
                    "share": round(share.estimate, 4),
                    "share_low": round(share.low, 4),
                    "share_high": round(share.high, 4),
                }
            )
        lengths = sorted(one["name_chars"] for one in measured)
        rows.append(
            {
                "half": half,
                "feature": "name_chars_below_donor_minimum",
                "items": total,
                "carrying": sum(1 for one in lengths if one < 6),
                "share": round(sum(1 for one in lengths if one < 6) / total, 4),
                "share_low": "",
                "share_high": "",
            }
        )
    return {"rows": rows}


def write_outputs(
    rows: list[dict[str, Any]],
    shortcut: dict[str, Any],
    corpus_shortcut: dict[str, Any],
    stem: str = "item_artefacts",
    extra: dict[str, Any] | None = None,
) -> None:
    """Write the measurement where the analysis reads it.

    Args:
        rows: The per-half feature shares.
        shortcut: The scores of the rule reading what one record shows.
        corpus_shortcut: The scores of the rule reading what the city shows.
        stem: The output file stem.
        extra: Further fields for the JSON output, written after the rest.
    """
    out = path("data_processed") / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    fields = [
        "half",
        "feature",
        "items",
        "carrying",
        "share",
        "share_low",
        "share_high",
    ]
    with (out / f"{stem}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out / f"{stem}.json").write_text(
        json.dumps(
            {
                "by_half": rows,
                "shortcut_from_the_construction": shortcut,
                "corpus_shortcut_from_the_construction": corpus_shortcut,
                "note": (
                    "The first shortcut reads the name's length and its "
                    "category token, both of which one record shows. The "
                    "second reads whether the name is repeated in its city, "
                    "which the donor's own surviving record makes true of "
                    "every injected name and of no injected name only. "
                    "Neither reads anything about the place, so what they "
                    "score is the construction and not the residual "
                    "question. The second bounds what a corpus-level string "
                    "rule can reach on this class from the construction "
                    "alone, which is the evidence the screen baseline reads."
                ),
                **(extra or {}),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("wrote {}", out / f"{stem}.csv")


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--set", choices=("injected", "residual_control"), default="injected"
    )
    arguments = parser.parse_args()
    control = arguments.set == "residual_control"

    items = read_control_items() if control else read_items()
    tokens = screen.load_category_tokens()
    minimum = int(get("injected_set.construction.m1.donor_min_name_chars"))
    summary = summarize(items, tokens)
    shortcut = score_flags(
        items,
        tokens,
        lambda measured: (
            measured["name_chars"] >= minimum and not measured["carries_category_token"]
        ),
    )
    corpus_shortcut = score_flags(
        items, tokens, lambda measured: measured["repeated_in_city"]
    )
    extra: dict[str, Any] | None = None
    if not control:
        # The same cut the control set is held to, measured on the first
        # construction so that the two can be read side by side.
        extra = {"name_length_oracle": oracle_cut(items, tokens, "name_chars")}
    if control:
        length = oracle_cut(items, tokens, "name_chars")
        words = oracle_cut(items, tokens, "name_tokens")
        acceptance = get("residual_control.acceptance")
        limit = float(acceptance["max_youdens_j"])
        reached = {
            "single_record": shortcut["youdens_j"],
            "corpus": corpus_shortcut["youdens_j"],
            "name_length_oracle": length["youdens_j"],
        }
        extra = {
            "name_length_oracle": length,
            "name_word_count_oracle": words,
            "acceptance": {
                "max_youdens_j": limit,
                "reached": reached,
                "passed": all(
                    reached[name] <= limit for name in acceptance["shortcuts"]
                ),
                "note": (
                    "Fixed in the configuration before the set was built. The word "
                    "count cut is measured beside the criterion and is not part of it."
                ),
            },
        }
    write_outputs(
        summary["rows"],
        shortcut,
        corpus_shortcut,
        stem="item_artefacts_control" if control else "item_artefacts",
        extra=extra,
    )
    for row in summary["rows"]:
        logger.info(
            "{} {}: {} of {} ({})",
            row["half"],
            row["feature"],
            row["carrying"],
            row["items"],
            row["share"],
        )
    for name, scored in (
        ("what one record shows", shortcut),
        ("what the city's corpus shows", corpus_shortcut),
    ):
        logger.info(
            "shortcut from the construction, {}: J {} [{}, {}] "
            "(recall {}, false positive rate {})",
            name,
            scored["youdens_j"],
            scored["youdens_j_low"],
            scored["youdens_j_high"],
            scored["recall"],
            scored["false_positive_rate"],
        )
    if extra is not None and "acceptance" in extra:
        logger.info("acceptance: {}", extra["acceptance"])


if __name__ == "__main__":
    main()
