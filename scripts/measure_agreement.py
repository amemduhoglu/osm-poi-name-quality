"""Measure how far the two readings agree, and write the adjudication sheet.

Branch 2 asks for two independent readings of the labelled set and reports
Cohen's kappa on all of it, whatever it is. The labelled set is the first
records of the fixed order, as the configuration sets it, and not the whole
draw: the records past the stop are deposited and weighted against but never
read. This script computes the kappa, breaks it down by stratum and by city, and
writes the sheet the two readers work from in the single adjudication meeting
the protocol allows.

It reports what it finds. A low kappa is a finding about how hard the residual
class is to judge, which is a result this study is entitled to report and not a
defect to be repaired by relabelling.

What it will not do is decide anything. The adjudicated column is left empty for
the readers to fill against the evidence, because a label produced here would be
a label produced by a program on a task the article claims resists programs.

Outputs:
    data/processed/natural_set/agreement.json          kappa and its breakdowns
    data/processed/natural_set/adjudication_sheet.csv  the disagreements

Usage:
    python -m scripts.measure_agreement
    python -m scripts.measure_agreement --readers reader_a reader_b
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from poi_audit import labelling
from poi_audit.config import get, path
from poi_audit.logsetup import setup
from poi_audit.stats import Agreement, Confusion, cohen_kappa, wilson

logger = setup("measure_agreement")


def known_readers() -> list[str]:
    """Return the annotator identifiers that have labelled anything.

    Returns:
        The identifiers, in alphabetical order, taken from the files the
        instrument has written for the run that counts.
    """
    found: list[str] = []
    for candidate in sorted(labelling.labels_dir().glob("*.jsonl")):
        if candidate.name.endswith(".practice.jsonl"):
            continue
        annotator = candidate.stem
        if labelling.read_labels(annotator):
            found.append(annotator)
    return found


def labels_by_item(annotator: str) -> dict[str, dict[str, Any]]:
    """Return one reader's labels, keyed by item.

    Args:
        annotator: The annotator identifier.

    Returns:
        Item identifier to the label record.
    """
    return {str(row["item_id"]): row for row in labelling.read_labels(annotator)}


def read_natural_set() -> dict[str, dict[str, Any]]:
    """Return the natural set keyed by item, for the strata and the cities.

    Returns:
        Item identifier to its record.

    Raises:
        FileNotFoundError: If the natural set has not been drawn.
    """
    source = path("data_processed") / "natural_set" / "records.jsonl"
    if not source.exists():
        raise FileNotFoundError(
            f"natural set not drawn: {source}; run scripts/draw_natural_set.py"
        )
    with source.open(encoding="utf-8") as handle:
        drawn = [json.loads(line) for line in handle if line.strip()]
    # The draw and the labelled set are two different numbers, and this script
    # measures the second: a reading that covers every labelled record is
    # finished, and calling it interim because the draw is larger would report a
    # closed phase as an open one.
    limit = int(get("natural_set.labelled.records"))
    ordered = sorted(drawn, key=lambda record: str(record["item_id"]))
    return {str(record["item_id"]): record for record in ordered[:limit]}


def as_dict(agreement: Agreement) -> dict[str, Any]:
    """Return one agreement measurement as a JSON-ready mapping.

    Args:
        agreement: The measurement.

    Returns:
        Its fields, rounded for the deposit.
    """
    return {
        "items": agreement.items,
        "observed_agreement": round(agreement.observed, 6),
        "expected_agreement": round(agreement.expected, 6),
        "kappa": round(agreement.kappa, 6),
        "standard_error": round(agreement.standard_error, 6),
        "ci_low": round(agreement.low, 6),
        "ci_high": round(agreement.high, 6),
    }


def mutual_accuracy(
    given: dict[str, dict[str, dict[str, Any]]], readers: tuple[str, str]
) -> dict[str, Any]:
    """Return how well each reader reproduces the other, with intervals.

    Cohen's kappa falls with the share of positives, and the wrong names are a
    small minority of this set, so a moderate kappa understates what the two
    readings agree on. Each reader is therefore scored against the other on the
    summary the models are scored on, which is available here without a gold
    standard and so without the circularity of scoring a reader against labels
    that reader helped settle. Items either reader left undecided enter neither
    direction, since a reader who abstains has made no claim to score.

    Args:
        given: Each reader's labels, keyed by reader and then by item.
        readers: The two annotator identifiers.

    Returns:
        The item count, the positives each direction counts, and the balanced
        accuracy of each direction with its interval.
    """
    first, second = readers
    decided = [
        item
        for item in sorted(set(given[first]) & set(given[second]))
        if given[first][item]["label"] != labelling.UNDECIDED
        and given[second][item]["label"] != labelling.UNDECIDED
    ]
    directions: dict[str, Any] = {}
    for predicted, reference in ((second, first), (first, second)):
        counts = Counter(
            (given[reference][item]["label"], given[predicted][item]["label"])
            for item in decided
        )
        confusion = Confusion(
            true_positive=counts[(labelling.WRONG, labelling.WRONG)],
            false_negative=counts[(labelling.WRONG, labelling.BELONGS)],
            true_negative=counts[(labelling.BELONGS, labelling.BELONGS)],
            false_positive=counts[(labelling.BELONGS, labelling.WRONG)],
        )
        low, high = confusion.balanced_accuracy_interval()
        directions[f"{predicted}_against_{reference}"] = {
            "positives": confusion.positives,
            "balanced_accuracy": round(confusion.balanced_accuracy, 6),
            "ci_low": round(low, 6),
            "ci_high": round(high, 6),
        }
    return {"items": len(decided), "directions": directions}


def write_adjudication_sheet(
    rows: list[dict[str, Any]], readers: tuple[str, str]
) -> Path:
    """Write the sheet the two readers settle their disagreements on.

    The adjudicated column is empty by design. The protocol settles a
    disagreement against the evidence both readers recorded, and where the
    evidence does not settle it the item becomes ``cannot_say``; neither
    decision belongs to this script.

    Args:
        rows: One row per item the two readers labelled differently.
        readers: The two annotator identifiers.

    Returns:
        The file written.
    """
    target = path("data_processed") / "natural_set" / "adjudication_sheet.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    first, second = readers
    fields = [
        "item_id",
        "name",
        "city",
        f"label_{first}",
        f"evidence_{first}",
        f"url_{first}",
        f"note_{first}",
        f"label_{second}",
        f"evidence_{second}",
        f"url_{second}",
        f"note_{second}",
        "adjudicated_label",
        "adjudication_reason",
    ]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"wrote {target} with {len(rows)} disagreements")
    return target


def main() -> None:
    """Measure agreement over whatever both readers have labelled so far."""
    parser = argparse.ArgumentParser(description="Measure inter-annotator agreement.")
    parser.add_argument(
        "--readers",
        nargs=2,
        metavar=("FIRST", "SECOND"),
        help="the two annotator identifiers; taken from the label files if absent",
    )
    arguments = parser.parse_args()

    readers = tuple(arguments.readers) if arguments.readers else tuple(known_readers())
    if len(readers) != 2:
        raise ValueError(
            f"two readings are needed and {len(readers)} were found: {readers}. "
            "Name them with --readers once both have labelled."
        )
    first, second = readers
    given = {reader: labels_by_item(reader) for reader in readers}
    shared = sorted(set(given[first]) & set(given[second]))
    if not shared:
        raise ValueError("the two readers have no item in common yet")

    natural = read_natural_set()
    total = len(natural)
    logger.info(
        f"{first}: {len(given[first])} of {total} labelled; "
        f"{second}: {len(given[second])} of {total}; "
        f"{len(shared)} in common"
    )
    if len(shared) < total:
        logger.warning(
            "the readings are not finished; every number below is interim and "
            "the phase does not close on it"
        )

    left = [str(given[first][item]["label"]) for item in shared]
    right = [str(given[second][item]["label"]) for item in shared]
    overall = cohen_kappa(left, right, labelling.LABELS)

    by_stratum: dict[str, Any] = {}
    for stratum in ("A", "B"):
        items = [item for item in shared if natural[item]["stratum"] == stratum]
        if items:
            by_stratum[stratum] = as_dict(
                cohen_kappa(
                    [str(given[first][item]["label"]) for item in items],
                    [str(given[second][item]["label"]) for item in items],
                    labelling.LABELS,
                )
            )

    by_city: dict[str, Any] = {}
    for city in sorted({str(natural[item]["city"]) for item in shared}):
        items = [item for item in shared if natural[item]["city"] == city]
        if len(items) > 1:
            by_city[city] = as_dict(
                cohen_kappa(
                    [str(given[first][item]["label"]) for item in items],
                    [str(given[second][item]["label"]) for item in items],
                    labelling.LABELS,
                )
            )

    matrix = Counter(zip(left, right, strict=True))
    disagreements = [
        {
            "item_id": item,
            "name": natural[item]["shown"].get("name", ""),
            "city": natural[item]["city"],
            f"label_{first}": given[first][item]["label"],
            f"evidence_{first}": given[first][item]["evidence_source"],
            f"url_{first}": given[first][item]["evidence_url"],
            f"note_{first}": given[first][item]["note"],
            f"label_{second}": given[second][item]["label"],
            f"evidence_{second}": given[second][item]["evidence_source"],
            f"url_{second}": given[second][item]["evidence_url"],
            f"note_{second}": given[second][item]["note"],
            "adjudicated_label": "",
            "adjudication_reason": "",
        }
        for item in shared
        if given[first][item]["label"] != given[second][item]["label"]
    ]
    write_adjudication_sheet(disagreements, (first, second))

    disagreed = wilson(len(disagreements), len(shared))
    summary = {
        "measured_on": date.today().isoformat(),
        "readers": list(readers),
        "protocol_version": labelling.PROTOCOL_VERSION,
        "items_in_natural_set": total,
        "items_labelled_by_both": len(shared),
        "complete": len(shared) == total,
        "overall": as_dict(overall),
        "mutual_accuracy": mutual_accuracy(given, (first, second)),
        "by_stratum": by_stratum,
        "by_city": by_city,
        "label_counts": {
            first: dict(Counter(left)),
            second: dict(Counter(right)),
        },
        "confusion": {f"{a}|{b}": count for (a, b), count in sorted(matrix.items())},
        "disagreements": {
            "items": len(disagreements),
            "share": round(disagreed.estimate, 6),
            "ci_low": round(disagreed.low, 6),
            "ci_high": round(disagreed.high, 6),
        },
        "sources_consulted": {
            reader: dict(
                Counter(
                    source.strip()
                    for item in shared
                    for source in str(given[reader][item]["evidence_source"]).split(";")
                    if source.strip()
                )
            )
            for reader in readers
        },
        "items_with_no_source": {
            reader: sum(1 for item in shared if given[reader][item].get("no_source"))
            for reader in readers
        },
    }
    target = path("data_processed") / "natural_set" / "agreement.json"
    target.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(f"wrote {target}")
    logger.info(
        f"kappa {overall.kappa:.3f} (95% {overall.low:.3f} to {overall.high:.3f}) "
        f"on {overall.items} items, {len(disagreements)} of them disagreements"
    )


if __name__ == "__main__":
    main()
