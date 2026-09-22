"""Estimate how often the residual class occurs in untouched records.

The natural set was drawn in two strata: a random one and one the screen
enriched. The share of wrong names in the sample is therefore not the share in
the corpus, and each record carries the inclusion probability that lets the
second be recovered from the first. The design also crossed language against
attribute completeness, and the rate is read on that grid as well as whole,
because a rate that varies across the grid is a property of the map rather than
of the sample.

Nothing here reads a model. The estimate rests on the two readers' agreed
labels and on the draw's own probabilities.

Outputs:
    data/processed/analysis/prevalence.csv
    data/processed/analysis/prevalence.json

Usage:
    python scripts/measure_prevalence.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from typing import Any

import numpy as np

from poi_audit.config import get, path
from poi_audit.logsetup import setup
from poi_audit.stats import wilson

logger = setup("measure_prevalence")

DECIDED = {"belongs", "wrong"}


def cell_of(record: dict[str, Any]) -> str:
    """Return the design cell a record was drawn in.

    Args:
        record: One natural-set record.

    Returns:
        The two corpus factors the design crossed, as one label.
    """
    return f"{record['language']}, {record['maturity']}"


def prevalence(records: list[dict[str, Any]], labels: dict[str, str]) -> dict[str, Any]:
    """Weight one group of records to the corpus.

    Args:
        records: The records in the group.
        labels: Item identifier to its agreed label.

    Returns:
        The counts and the weighted rate. A record the readers left undecided
        carries no rate and leaves the denominator, and is counted so that the
        share it represents stays visible rather than silently absent.

    Raises:
        ValueError: If an inclusion probability is not positive, since its
            reciprocal is the weight and an impossible record cannot have been
            drawn.
    """
    decided = 0
    wrong = 0
    undecided = 0
    total_weight = 0.0
    wrong_weight = 0.0
    undecided_weight = 0.0
    for record in records:
        label = labels.get(str(record["item_id"]))
        if label is None:
            continue
        probability = float(record["inclusion_probability"])
        if probability <= 0:
            raise ValueError(f"inclusion probability not positive: {record['item_id']}")
        weight = 1.0 / probability
        if label not in DECIDED:
            undecided += 1
            undecided_weight += weight
            continue
        decided += 1
        total_weight += weight
        if label == "wrong":
            wrong += 1
            wrong_weight += weight
    raw = wilson(wrong, decided) if decided else None
    drawn_weight = total_weight + undecided_weight
    return {
        "decided": decided,
        "wrong": wrong,
        "cannot_say": undecided,
        "raw_prevalence": round(raw.estimate, 4) if raw else "",
        "raw_low": round(raw.low, 4) if raw else "",
        "raw_high": round(raw.high, 4) if raw else "",
        "weighted_prevalence": (
            round(wrong_weight / total_weight, 4) if total_weight else 0.0
        ),
        # What the undecided records could have been worth, as a bound rather
        # than as an assumption. Dropping them from the rate assumes they go
        # wrong at the rate the decided ones do, and they do not: the readers
        # leave most of them undecided where wrong names are commonest. The
        # bounds hold whatever the truth on those records is.
        "bound_low": (round(wrong_weight / drawn_weight, 4) if drawn_weight else 0.0),
        "bound_high": (
            round((wrong_weight + undecided_weight) / drawn_weight, 4)
            if drawn_weight
            else 0.0
        ),
    }


def clustered_interval(
    records: list[dict[str, Any]],
    labels: dict[str, str],
    draws: int,
    minimum_clusters: int,
    seed: int = 42,
) -> tuple[float, float, str, int]:
    """Return the weighted rate's spread with the city resampled.

    A percentile bootstrap needs enough clusters for its tails to carry
    information. With three cities to a design cell the 2.5th and 97.5th
    percentiles of the resampled rate are the smallest and the largest city
    rate and nothing else, so what comes back is the range of three numbers
    wearing an interval's name. Under the configured minimum the range is
    returned and named as one, because a spread that cannot be a 95 per cent
    interval should not be printed as though it were.

    Args:
        records: The records in the group.
        labels: Item identifier to its agreed label.
        draws: How many resamples to take.
        minimum_clusters: How many clusters a percentile interval needs.
        seed: The seed, fixed so a reported interval is the same on every run.

    Returns:
        The lower end, the upper end, what the spread is (a bootstrap
        interval, a range across the cities, or nothing), and how many
        clusters the group holds.
    """
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for record in records:
        label = labels.get(str(record["item_id"]))
        if label not in DECIDED:
            continue
        weight = 1.0 / float(record["inclusion_probability"])
        grouped[str(record["city"])].append(
            (weight, weight if label == "wrong" else 0.0)
        )
    units = sorted(grouped)
    if not units:
        return 0.0, 0.0, "none", 0
    totals = np.array([sum(one[0] for one in grouped[u]) for u in units])
    wrongs = np.array([sum(one[1] for one in grouped[u]) for u in units])
    if len(units) < minimum_clusters:
        if len(units) < 2:
            return 0.0, 0.0, "none", len(units)
        per_city = wrongs / totals
        return (
            round(float(per_city.min()), 4),
            round(float(per_city.max()), 4),
            "city range",
            len(units),
        )
    generator = np.random.default_rng(seed)
    picks = generator.integers(0, len(units), size=(draws, len(units)))
    resampled = wrongs[picks].sum(axis=1) / totals[picks].sum(axis=1)
    low, high = np.quantile(resampled, [0.025, 0.975])
    return (
        round(float(low), 4),
        round(float(high), 4),
        "cluster bootstrap",
        len(units),
    )


def read_inputs() -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Read the natural set and its agreed labels.

    Returns:
        The records as the draw wrote them, and item identifier to label.
    """
    natural = path("data_processed") / "natural_set"
    records = [
        json.loads(line)
        for line in (natural / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels = {
        str(one["item_id"]): str(one["label"])
        for one in (
            json.loads(line)
            for line in (natural / "gold.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    }
    return records, labels


def main() -> None:
    """Command line entry point."""
    records, labels = read_inputs()
    draws = int(get("analysis.cluster_bootstrap_draws"))
    minimum_clusters = int(get("analysis.min_clusters_for_interval"))

    groups: list[tuple[str, str, list[dict[str, Any]]]] = [("all", "corpus", records)]
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_city: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_cell[cell_of(record)].append(record)
        by_city[str(record["city"])].append(record)
    groups += [("design cell", key, by_cell[key]) for key in sorted(by_cell)]
    groups += [("city", key, by_city[key]) for key in sorted(by_city)]

    rows: list[dict[str, Any]] = []
    for kind, name, group in groups:
        measured = prevalence(group, labels)
        low, high, spread, clusters = clustered_interval(
            group, labels, draws, minimum_clusters
        )
        reportable = kind != "city" and spread != "none"
        rows.append(
            {
                "grouping": kind,
                "group": name,
                **measured,
                # A single city has one cluster, so the resampled interval is a
                # point and is left empty rather than printed as certainty.
                "weighted_low": low if reportable else "",
                "weighted_high": high if reportable else "",
                "spread": spread if reportable else "none",
                "clusters": clusters,
            }
        )

    out = path("data_processed") / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (out / "prevalence.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out / "prevalence.json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("wrote {}", out / "prevalence.csv")
    for row in rows:
        if row["grouping"] != "city":
            logger.info(
                "{}: {} wrong of {} decided, weighted {} [{}, {}], {} undecided",
                row["group"],
                row["wrong"],
                row["decided"],
                row["weighted_prevalence"],
                row["weighted_low"],
                row["weighted_high"],
                row["cannot_say"],
            )


if __name__ == "__main__":
    main()
