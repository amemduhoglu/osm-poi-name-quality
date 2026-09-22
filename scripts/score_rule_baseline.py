"""Score the rule baseline against the injected item set.

This is the answer a model has to beat, and it is produced before any model
runs. Three quantities come out of it, and they are not equally interesting:

    Recall on the syntactic classes and on the wrong calling code. Near one, and
    near one by construction: those items were injected against the same
    definitions the rules test. The number is reported so that the construction
    is visible, not offered as a finding.

    Recall on the residual class. This is what the article rests on. No rule in
    the baseline reads a name, and the category-token check is scored beside the
    baseline to show what the donor exclusion was worth.

    The flag rate on the nine hundred clean items. Nothing about this is
    arranged. Those are real values written by mappers, and whatever the rules
    say about them is the cost a practitioner pays for running them.

Coverage is reported with the scores, per authority and for the union, because a
lookup that abstains on a record has not cleared it.

Outputs:
    data/processed/injected_set/rule_baseline_by_class.csv
    data/processed/injected_set/rule_baseline_by_class_paired.csv
    data/processed/injected_set/rule_baseline_by_check.csv
    data/processed/injected_set/rule_baseline.json

Usage:
    python scripts/score_rule_baseline.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from poi_audit import rules
from poi_audit.config import get, path
from poi_audit.corpus import read_corpus
from poi_audit.logsetup import setup
from poi_audit.references import Gazetteer, coverage, load_gazetteer
from poi_audit.stats import Confusion, wilson

logger = setup("score_rule_baseline")


def item_path() -> Path:
    """Return the file the item builder writes."""
    return path("data_processed") / "injected_set" / "items.jsonl"


def read_items() -> list[dict[str, Any]]:
    """Read the injected item set.

    Returns:
        Every item.

    Raises:
        FileNotFoundError: If the set has not been built.
    """
    target = item_path()
    if not target.exists():
        raise FileNotFoundError(
            f"item set not found: {target}; run scripts/build_injected_set.py"
        )
    with target.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_category_tokens() -> dict[str, str]:
    """Read the category-token dictionary the item builder wrote.

    Returns:
        Token to indicated category.

    Raises:
        FileNotFoundError: If the dictionary has not been written.
    """
    target = path("data_processed") / "category_tokens.csv"
    if not target.exists():
        raise FileNotFoundError(
            f"token dictionary not found: {target}; run scripts/build_injected_set.py"
        )
    with target.open(encoding="utf-8", newline="") as handle:
        return {row["token"]: row["category"] for row in csv.DictReader(handle)}


def as_record(item: dict[str, Any]) -> dict[str, Any]:
    """Turn an item back into the shape the rules read.

    Args:
        item: One item of the injected set.

    Returns:
        A record carrying the item's shown fields as tags, with its coordinate
        and country beside them.
    """
    shown = dict(item["record"])
    lat = float(shown.pop("lat"))
    lon = float(shown.pop("lon"))
    return {
        "tags": shown,
        "country": item["country"],
        "city": item["city"],
        "lat": lat,
        "lon": lon,
    }


def score(
    items: list[dict[str, Any]], gazetteer: Gazetteer, tokens: dict[str, str]
) -> dict[str, Any]:
    """Run the baseline over every item and tally the outcomes.

    Args:
        items: The injected item set.
        gazetteer: The loaded gazetteer.
        tokens: The category-token dictionary, for the reported check.

    Returns:
        Per class and per check tallies, and the per-item verdicts.
    """
    by_class: dict[str, Counter[str]] = defaultdict(Counter)
    by_check: dict[str, Counter[str]] = defaultdict(Counter)
    token_by_class: dict[str, Counter[str]] = defaultdict(Counter)
    detail: list[dict[str, Any]] = []

    for item in items:
        record = as_record(item)
        verdicts = rules.baseline(record, gazetteer)
        flagged = rules.flagged(verdicts)
        group = f"{item['class']}-{'corrupted' if item['corrupted'] else 'clean'}"
        by_class[group]["items"] += 1
        by_class[group]["flagged"] += int(flagged)

        # A hit is a flag that names the field the corruption actually touched.
        if item["corrupted"]:
            field = item["corrupted_field"]
            by_class[group]["field_hit"] += int(
                any(v.outcome == rules.FLAG and v.field == field for v in verdicts)
            )

        for verdict in verdicts:
            by_check[verdict.check][verdict.outcome] += 1
            if item["corrupted"]:
                by_check[verdict.check][f"corrupted_{verdict.outcome}"] += 1
            else:
                by_check[verdict.check][f"clean_{verdict.outcome}"] += 1

        name = str(item["record"].get("name", ""))
        category = str(item["record"].get("category", ""))
        token_verdict = rules.category_token(name, category, tokens)
        token_by_class[group][token_verdict.outcome] += 1

        detail.append(
            {
                "item_id": item["item_id"],
                "class": item["class"],
                "corrupted": item["corrupted"],
                "flagged": flagged,
                "checks_flagged": sorted(
                    {v.check for v in verdicts if v.outcome == rules.FLAG}
                ),
                "category_token": token_verdict.outcome,
            }
        )
    return {
        "by_class": {key: dict(value) for key, value in by_class.items()},
        "by_check": {key: dict(value) for key, value in by_check.items()},
        "category_token_by_class": {
            key: dict(value) for key, value in token_by_class.items()
        },
        "detail": detail,
    }


def write_outputs(scored: dict[str, Any], covered: dict[str, dict[str, int]]) -> None:
    """Write the baseline's tables and its summary.

    Args:
        scored: The tallies from the scoring pass.
        covered: The per-authority coverage measurement over the corpus.
    """
    out_dir = path("data_processed") / "injected_set"
    out_dir.mkdir(parents=True, exist_ok=True)

    class_path = out_dir / "rule_baseline_by_class.csv"
    with class_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "group",
                "items",
                "flagged",
                "flag_rate",
                "flag_rate_low",
                "flag_rate_high",
                "field_hit",
                "field_hit_rate",
            ]
        )
        for group in sorted(scored["by_class"]):
            tally = scored["by_class"][group]
            items = int(tally["items"])
            flagged = int(tally.get("flagged", 0))
            hits = int(tally.get("field_hit", 0))
            rate = wilson(flagged, items)
            writer.writerow(
                [
                    group,
                    items,
                    flagged,
                    round(rate.estimate, 4) if items else "",
                    round(rate.low, 4) if items else "",
                    round(rate.high, 4) if items else "",
                    hits if group.endswith("corrupted") else "",
                    (
                        round(hits / items, 4)
                        if items and group.endswith("corrupted")
                        else ""
                    ),
                ]
            )
    logger.info(f"wrote {class_path}")

    # The recall of a check and the cost of running it are one measurement, so
    # the two halves of a class are written on one row: a rule that settles a
    # class outright and flags a sixth of the clean records has not settled it
    # for free, and the figure the article draws reads this table.
    cost_path = out_dir / "rule_baseline_by_class_paired.csv"
    with cost_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "class",
                "recall",
                "recall_low",
                "recall_high",
                "false_flag_rate",
                "false_flag_rate_low",
                "false_flag_rate_high",
                "specificity",
                "balanced_accuracy",
                "balanced_accuracy_low",
                "balanced_accuracy_high",
            ]
        )
        classes = sorted(
            {group.rsplit("-", 1)[0] for group in scored["by_class"]},
            key=lambda name: (name[0], name),
        )
        for name in classes:
            corrupted = scored["by_class"].get(f"{name}-corrupted", {})
            clean = scored["by_class"].get(f"{name}-clean", {})
            counts = Confusion(
                true_positive=int(corrupted.get("flagged", 0)),
                false_negative=int(corrupted.get("items", 0))
                - int(corrupted.get("flagged", 0)),
                true_negative=int(clean.get("items", 0)) - int(clean.get("flagged", 0)),
                false_positive=int(clean.get("flagged", 0)),
            )
            recall = counts.sensitivity()
            specificity = counts.specificity()
            false_flag = wilson(counts.false_positive, counts.negatives)
            balanced = counts.balanced_accuracy_interval()
            writer.writerow(
                [
                    name,
                    round(recall.estimate, 4),
                    round(recall.low, 4),
                    round(recall.high, 4),
                    round(false_flag.estimate, 4),
                    round(false_flag.low, 4),
                    round(false_flag.high, 4),
                    round(specificity.estimate, 4),
                    round(counts.balanced_accuracy, 4),
                    round(balanced[0], 4),
                    round(balanced[1], 4),
                ]
            )
    logger.info(f"wrote {cost_path}")

    check_path = out_dir / "rule_baseline_by_check.csv"
    with check_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["check", "pass", "flag", "abstain", "clean_flag", "corrupted_flag"]
        )
        for check in sorted(scored["by_check"]):
            tally = scored["by_check"][check]
            writer.writerow(
                [
                    check,
                    tally.get(rules.PASS, 0),
                    tally.get(rules.FLAG, 0),
                    tally.get(rules.ABSTAIN, 0),
                    tally.get(f"clean_{rules.FLAG}", 0),
                    tally.get(f"corrupted_{rules.FLAG}", 0),
                ]
            )
    logger.info(f"wrote {check_path}")

    summary = {
        "by_class": scored["by_class"],
        "by_check": scored["by_check"],
        "category_token_by_class": scored["category_token_by_class"],
        "corpus_coverage": {
            source: {
                **tally,
                "rate": (
                    round(tally["covered"] / tally["records"], 4)
                    if tally["records"]
                    else None
                ),
            }
            for source, tally in covered.items()
        },
        "address_city_radius_km": get("references.address_city.match_radius_km"),
    }
    summary_path = out_dir / "rule_baseline.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(f"wrote {summary_path}")


def main() -> None:
    """Score the baseline and measure the authorities' coverage."""
    gazetteer = load_gazetteer()
    tokens = read_category_tokens()
    items = read_items()
    logger.info(f"scoring {len(items)} items")
    scored = score(items, gazetteer, tokens)

    logger.info("measuring authority coverage over the corpus")
    covered = coverage(gazetteer, read_corpus())
    for source, tally in covered.items():
        rate = tally["covered"] / tally["records"] if tally["records"] else 0.0
        logger.info(f"{source}: {rate:.3f} of {tally['records']} corpus records")

    write_outputs(scored, covered)
    for group in sorted(scored["by_class"]):
        tally = scored["by_class"][group]
        rate = int(tally.get("flagged", 0)) / int(tally["items"])
        logger.info(f"{group}: flagged {rate:.3f}")


if __name__ == "__main__":
    main()
