"""Build the natural set's gold labels from the two readings and the meeting.

A label enters the gold standard by one of two routes, and the file records
which: the two readers reached it independently, or the adjudication meeting
settled it under the rule in section 6 of the protocol. Nothing else produces a
label, and no model takes any part.

The file carries the two sets the analysis scores on, decided before any model
answered a natural-set item: every item with a definite label, and the decidable
core, which is the items both readers reached the same definite label on
independently. An item left at cannot_say has no ground truth and is scored
against by neither; its share is reported instead, because how much of the class
no evidence settles is a measurement of the class.

The stratum and the inclusion probability travel with each row, since the
corpus-level estimate is recovered by weighting rather than by the shape of the
draw.

Outputs:
    data/processed/natural_set/gold.jsonl   one row per labelled record
    data/processed/natural_set/gold.json    the counts the article reports

Usage:
    python -m scripts.build_gold_labels
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from poi_audit import adjudication, labelling
from poi_audit.config import path
from poi_audit.logsetup import setup

logger = setup("build_gold_labels")

AGREED = "agreed"
ADJUDICATED = "adjudicated"


def natural_records() -> dict[str, dict[str, Any]]:
    """Return the drawn records keyed by item, for the strata and the weights.

    Returns:
        Item identifier to its record.
    """
    source = path("data_processed") / "natural_set" / "records.jsonl"
    with source.open(encoding="utf-8") as handle:
        return {
            str(record["item_id"]): record
            for record in (json.loads(line) for line in handle if line.strip())
        }


def build() -> list[dict[str, Any]]:
    """Return one gold row per labelled record.

    Returns:
        The rows, in labelling order.

    Raises:
        ValueError: If the meeting is unfinished, since a gold standard built
            from a meeting still in progress would change under the analysis.
    """
    if not adjudication.closed():
        raise ValueError(
            "the adjudication meeting is not closed; run it to the end first"
        )
    first, second = adjudication.readers()
    readings = {
        reader: {
            str(row["item_id"]): row
            for row in labelling.read_labels(reader, labelling.MAIN)
        }
        for reader in (first, second)
    }
    settled = adjudication.decisions()
    records = natural_records()

    rows: list[dict[str, Any]] = []
    for item in labelling.read_items(labelling.MAIN):
        left = readings[first][item.item_id]["label"]
        right = readings[second][item.item_id]["label"]
        agreed = left == right
        if agreed:
            label, source = left, AGREED
        else:
            label, source = settled[item.item_id]["label"], ADJUDICATED
        record = records[item.item_id]
        rows.append(
            {
                "item_id": item.item_id,
                "position": item.position,
                "city": record["city"],
                "stratum": record["stratum"],
                "inclusion_probability": record["inclusion_probability"],
                "label": label,
                "label_source": source,
                f"label_{first}": left,
                f"label_{second}": right,
                # The two sets the analysis scores on. An item with no definite
                # label is in neither, and the core is the stricter one: both
                # readers reached the same definite label with no conversation
                # between them, which is the evidence that the item admits an
                # answer at all.
                "scored": label != "cannot_say",
                "decidable_core": agreed and label != "cannot_say",
            }
        )
    return rows


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the counts the article reports.

    Args:
        rows: The gold rows.

    Returns:
        Counts overall and by stratum, for both scored sets.
    """
    strata = sorted({str(row["stratum"]) for row in rows})

    def counts(kept: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "items": len(kept),
            "labels": dict(Counter(str(row["label"]) for row in kept)),
            "by_stratum": {
                stratum: {
                    "items": sum(1 for row in kept if row["stratum"] == stratum),
                    "wrong": sum(
                        1
                        for row in kept
                        if row["stratum"] == stratum and row["label"] == "wrong"
                    ),
                }
                for stratum in strata
            },
        }

    return {
        "built_at": datetime.now(UTC).isoformat(),
        "protocol_version": adjudication.PROTOCOL_VERSION,
        "labelled": counts(rows),
        "label_source": dict(Counter(str(row["label_source"]) for row in rows)),
        "adjudicated": counts([row for row in rows if row["scored"]]),
        "decidable_core": counts([row for row in rows if row["decidable_core"]]),
        # Reported rather than hidden: how much of the residual class no
        # evidence settles is a property of the class, not a gap in the study.
        "cannot_say": counts([row for row in rows if row["label"] == "cannot_say"]),
    }


def main() -> None:
    """Build the gold labels and write them beside their summary."""
    rows = build()
    out_dir = path("data_processed") / "natural_set"
    target = out_dir / "gold.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info(f"wrote {target} with {len(rows)} rows")

    summary = summarise(rows)
    (out_dir / "gold.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(f"wrote {out_dir / 'gold.json'}")
    logger.info(
        f"{summary['adjudicated']['items']} scorable, "
        f"{summary['decidable_core']['items']} in the decidable core, "
        f"{summary['cannot_say']['items']} left at cannot_say"
    )


if __name__ == "__main__":
    main()
