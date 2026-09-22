"""Account for the baseline's flags on the untouched half of every class.

Each class pairs a corrupted record with an untouched one, and a flag on the
untouched record is counted as a false positive. That counts an assumption
rather than a measurement: the untouched record is a live value written by a
mapper, and a live value can be malformed on its own. The item set records the
violations each record already held when it was drawn, so the flags can be
checked against them instead of assumed.

What the study reports as the practitioner's cost therefore needs reading twice.
A flag that lands on a violation the record already held is the rule doing its
job on an unlabelled positive; a flag with nothing behind it is the false alarm
the cost was meant to describe. The two are counted apart here and the article
can then say which of them it means.

Outputs:
    data/processed/analysis/clean_flags.csv
    data/processed/analysis/clean_flags.json

Usage:
    python scripts/measure_clean_flags.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from typing import Any

from poi_audit import rules
from poi_audit.config import path
from poi_audit.logsetup import setup
from poi_audit.references import load_gazetteer
from poi_audit.stats import wilson
from scripts.score_rule_baseline import as_record, read_items

logger = setup("measure_clean_flags")


def explain(items: list[dict[str, Any]]) -> dict[str, int]:
    """Count the flags on untouched records against what those records held.

    Args:
        items: Untouched items, each carrying the checks the baseline flagged
            and the violations the record already held.

    Returns:
        How many were flagged, how many of those land on a violation the record
        already held, and how many do not. A record the baseline left alone is
        counted in neither, since silence needs no account.
    """
    flagged = 0
    explained = 0
    unexplained = 0
    for item in items:
        raised = list(item["checks_flagged"])
        if not raised:
            continue
        flagged += 1
        held = {str(one["check"]) for one in item["preexisting_flags"]}
        if set(raised) <= held:
            explained += 1
        else:
            unexplained += 1
    return {
        "flagged": flagged,
        "explained": explained,
        "unexplained": unexplained,
    }


def clean_items() -> list[dict[str, Any]]:
    """Return the untouched half of every class with the baseline's verdicts.

    Returns:
        One entry per untouched item, carrying the checks the baseline flagged
        and the violations the record already held.
    """
    gazetteer = load_gazetteer()
    entries: list[dict[str, Any]] = []
    for item in read_items():
        if item["corrupted"]:
            continue
        verdicts = rules.baseline(as_record(item), gazetteer)
        entries.append(
            {
                "item_id": item["item_id"],
                "class": item["class"],
                "checks_flagged": sorted(
                    {one.check for one in verdicts if one.outcome == rules.FLAG}
                ),
                "preexisting_flags": item["preexisting_flags"],
            }
        )
    return entries


def main() -> None:
    """Command line entry point."""
    entries = clean_items()
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_class[str(entry["class"])].append(entry)

    rows: list[dict[str, Any]] = []
    for name, group in [("all", entries)] + sorted(by_class.items()):
        counted = explain(group)
        rate = wilson(counted["flagged"], len(group))
        false_rate = wilson(counted["unexplained"], len(group))
        rows.append(
            {
                "class": name,
                "clean_items": len(group),
                **counted,
                "flag_rate": round(rate.estimate, 4),
                "flag_rate_low": round(rate.low, 4),
                "flag_rate_high": round(rate.high, 4),
                "false_flag_rate": round(false_rate.estimate, 4),
                "false_flag_rate_low": round(false_rate.low, 4),
                "false_flag_rate_high": round(false_rate.high, 4),
            }
        )

    out = path("data_processed") / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "clean_flags.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "clean_flags.json").write_text(
        json.dumps(
            {
                "rows": rows,
                "note": (
                    "A flag on an untouched record is explained when every "
                    "check it raised is a violation the record already held "
                    "when it was drawn. An explained flag is the rule finding "
                    "a real defect in an unlabelled negative, not a false "
                    "alarm."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("wrote {}", out / "clean_flags.csv")
    for row in rows:
        logger.info(
            "{}: {} flagged of {}, {} on a violation already held, {} false",
            row["class"],
            row["flagged"],
            row["clean_items"],
            row["explained"],
            row["unexplained"],
        )


if __name__ == "__main__":
    main()
