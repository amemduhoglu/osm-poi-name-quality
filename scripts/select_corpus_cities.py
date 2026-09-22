"""Apply the fixed selection rule to the measured candidates.

This script reads the candidate counts, eliminates the candidates that cannot
carry a city's share of the item set, assigns maturity from measurement rather
than from the hypothesis recorded against each candidate, and reports the twelve
cities the rule selects.

It writes a proposed selection and nothing else. Filling ``corpus.cities`` in
the configuration file is a design decision and is made by hand after the
proposal is read, because a script that edits its own parameters is no longer
measurable.

The rule, in order:

1. A candidate failing any configured threshold is eliminated.
2. Within each language group the remaining candidates are ordered by the
   configured ranking metric, which is attribute completeness: the mean of the
   shares of named points of interest carrying each of the measured attributes.
3. The three most complete form the mature cell and the three least complete
   the developing cell, and no country appears twice inside a cell. A candidate
   whose country is already taken is skipped and the next one is considered.
4. The gap between the cells is reported, together with every candidate whose
   measured maturity contradicts its hypothesis.

The ranking metric was changed once, after the first twenty-one candidates were
measured and the ranking it produced was read. The first rule ordered by named
points of interest per square kilometre, which measures how dense a city is
rather than how completely its records are attributed; it placed Lima and Quito
in the mature cell and Utrecht in the developing one. The change is recorded in
the progress file with the selection the superseded rule produced, because a
rule changed after its output was seen has to stay visible. The elimination
thresholds were not touched.

Usage:
    python scripts/select_corpus_cities.py
"""

from __future__ import annotations

import csv
from typing import Any

from poi_audit.config import get, path
from poi_audit.logsetup import setup

logger = setup("select_corpus_cities")


def read_counts() -> list[dict[str, Any]]:
    """Read the candidate measurement table.

    Returns:
        One record per measured candidate, with numeric fields converted.

    Raises:
        FileNotFoundError: If the measurement has not been run.
    """
    counts_path = path("data_processed") / "candidate_city_counts.csv"
    if not counts_path.exists():
        raise FileNotFoundError(
            f"{counts_path} is absent; run scripts/measure_candidate_cities.py first"
        )
    with counts_path.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    metrics = list(get("corpus.candidate_selection.metrics"))
    for record in records:
        for name in metrics:
            record[name] = int(record[name])
        record["named_per_km2"] = float(record["named_per_km2"])
        record["eligible"] = record["eligible"] == "True"
        record["attribute_completeness"] = attribute_completeness(record)
    return records


def attribute_completeness(record: dict[str, Any]) -> float:
    """Return a candidate's attribute completeness as a percentage.

    Completeness is the mean of the shares of named points of interest that
    carry each configured attribute. The mean of several shares is used rather
    than any single one because a single tag can be absent for a national
    convention rather than for want of data, as ``addr:city`` is in Australia.

    Args:
        record: One measured candidate.

    Returns:
        The mean share, in percent of named points of interest.
    """
    fields = list(get("corpus.candidate_selection.completeness_metrics"))
    named = int(record["named"])
    if named == 0:
        return 0.0
    return round(
        sum(100.0 * int(record[field]) / named for field in fields) / len(fields), 2
    )


def eliminate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop the candidates that failed a threshold.

    Args:
        records: Every measured candidate.

    Returns:
        The candidates that met every threshold.
    """
    kept = [r for r in records if r["eligible"]]
    for record in records:
        if not record["eligible"]:
            logger.info(
                f"eliminated {record['city']} ({record['country']}): "
                f"{record['failed_thresholds']}"
            )
    return kept


def take_distinct_countries(
    ordered: list[dict[str, Any]], size: int
) -> list[dict[str, Any]]:
    """Take the first candidates of an ordering without repeating a country.

    Args:
        ordered: Candidates in the order the rule prefers them.
        size: How many to take.

    Returns:
        The selected candidates.

    Raises:
        ValueError: If the ordering cannot supply that many distinct countries.
    """
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in ordered:
        if record["country"] in seen:
            continue
        chosen.append(record)
        seen.add(record["country"])
        if len(chosen) == size:
            return chosen
    raise ValueError(f"only {len(chosen)} distinct countries available, need {size}")


def select(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the selection rule and return the chosen cities.

    Args:
        records: The eligible candidates.

    Returns:
        The selected cities, each carrying its assigned maturity.

    Raises:
        ValueError: If a cell cannot be filled, or if the two cells of a
            language group overlap on the ranking metric, which would mean the
            maturity contrast the design rests on was not measured.
    """
    per_cell = int(get("corpus.cities_per_cell"))
    metric = str(get("corpus.candidate_selection.ranking_metric"))
    selected: list[dict[str, Any]] = []
    for language in get("corpus.design_cells.language"):
        group = [r for r in records if r["language"] == language]
        ordered = sorted(group, key=lambda r: r[metric], reverse=True)
        mature = take_distinct_countries(ordered, per_cell)
        developing = take_distinct_countries(list(reversed(ordered)), per_cell)
        if {r["city"] for r in mature} & {r["city"] for r in developing}:
            raise ValueError(f"{language}: too few eligible candidates to split")
        floor = min(r[metric] for r in mature)
        ceiling = max(r[metric] for r in developing)
        if floor <= ceiling:
            raise ValueError(f"{language}: the two cells overlap on {metric}")
        logger.info(
            f"{language}: mature floor {floor:.1f}, developing ceiling "
            f"{ceiling:.1f}, ratio {floor / ceiling:.1f} on {metric}"
        )
        for record in mature:
            selected.append({**record, "maturity": "mature"})
        for record in developing:
            selected.append({**record, "maturity": "developing"})
    return selected


def report(selected: list[dict[str, Any]]) -> None:
    """Write the proposed selection and note every contradicted hypothesis.

    Args:
        selected: The chosen cities.
    """
    out_path = path("data_processed") / "city_selection.csv"
    fields = [
        "city",
        "country",
        "language",
        "maturity",
        "hypothesis_maturity",
        "lat",
        "lon",
        "named",
        "named_per_km2",
        "attribute_completeness",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    logger.info(f"wrote {len(selected)} selected cities to {out_path}")
    for record in selected:
        if record["maturity"] != record["hypothesis_maturity"]:
            logger.warning(
                f"{record['city']} was measured {record['maturity']} against a "
                f"{record['hypothesis_maturity']} hypothesis"
            )


def main() -> None:
    """Select the corpus cities from the measured candidates."""
    records = read_counts()
    kept = eliminate(records)
    logger.info(f"{len(kept)} of {len(records)} candidates eligible")
    selected = select(kept)
    for record in selected:
        logger.info(
            f"{record['language']:>11} {record['maturity']:>10}  "
            f"{record['city']} ({record['country']}) "
            f"completeness {record['attribute_completeness']:.1f}%, "
            f"{record['named_per_km2']:.1f} named/km2"
        )
    report(selected)


if __name__ == "__main__":
    main()
