"""Measure what share of the corpus each open authority holds at all.

A reviewer will ask why an authority good enough to settle a label is not
simply added to the rule baseline. The answer is coverage, and this study
measures it rather than asserting it: the baseline's own lookups cover the
corpus in full, whereas these authorities hold only the places they happen to
hold, and the share is neither large nor evenly spread across the twelve
cities.

Coverage is deliberately the weakest possible question: does the authority hold
any entry within the matching radius of this record. Whether the entry's name
agrees with the record's is the verification outcome, which the pilot measures
on labelled items and which is not decided here, because a name comparison run
over the whole corpus would be a labelling pass by another name.

Outputs:
    data/processed/authorities/coverage_by_city.csv    per source and city
    data/processed/authorities/coverage.json           corpus and natural set

Usage:
    python -m scripts.measure_authority_coverage
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from poi_audit.config import get, path
from poi_audit.corpus import fold, read_city
from poi_audit.logsetup import setup
from poi_audit.references import Gazetteer, Place
from poi_audit.stats import wilson
from scripts.fetch_authorities import authorities_path, manifest_path

logger = setup("measure_authority_coverage")

UNION = "any_authority"


def load_authorities(source_file: Path | None = None) -> dict[str, Gazetteer]:
    """Read the authority entries, one spatial index per source.

    The sources are kept apart rather than merged, so that each one's coverage
    is reported on its own and the union is reported beside them.

    Args:
        source_file: An alternative file, used by the tests. The configured
            location is read when this is None.

    Returns:
        Source name to its index.

    Raises:
        FileNotFoundError: If the authorities have not been fetched.
    """
    target = source_file or authorities_path()
    if not target.exists():
        raise FileNotFoundError(
            f"authorities not found: {target}; run scripts/fetch_authorities.py"
        )
    grouped: dict[str, list[Place]] = defaultdict(list)
    with target.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            names = {fold(row["name"])}
            for alternate in row["alternate_names"].split(","):
                if alternate.strip():
                    names.add(fold(alternate))
            grouped[row["source"]].append(
                Place(
                    source=row["source"],
                    place_id=row["authority_id"],
                    names=frozenset(name for name in names if name),
                    display_name=row["name"],
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    country=row["country"],
                    population=0,
                )
            )
    return {source: Gazetteer(places=places) for source, places in grouped.items()}


def covered(
    record: dict[str, Any], indexes: dict[str, Gazetteer], radius_km: float
) -> set[str]:
    """Return the authorities holding an entry near one record.

    Args:
        record: A corpus record.
        indexes: Source name to its index.
        radius_km: The matching radius in kilometres.

    Returns:
        The sources that hold something within the radius, empty when none
        does.
    """
    lat, lon = float(record["lat"]), float(record["lon"])
    return {
        source
        for source, index in indexes.items()
        if index.has_nearby(lat, lon, radius_km)
    }


def natural_set_records() -> list[dict[str, Any]]:
    """Read the drawn natural set, or return nothing when it is not drawn.

    Returns:
        The 400 records, or an empty list.
    """
    target = path("data_processed") / "natural_set" / "records.jsonl"
    if not target.exists():
        return []
    with target.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_by_city(
    sources: list[str],
    frame: dict[str, int],
    hits: dict[str, Counter[str]],
) -> Path:
    """Write coverage per source and city with its interval.

    Args:
        sources: The source names in reporting order, the union last.
        frame: Records measured in each city.
        hits: Per source, how many of each city's records it covers.

    Returns:
        The file written.
    """
    target = path("data_processed") / "authorities" / "coverage_by_city.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["source", "city", "records", "covered", "coverage", "ci_low", "ci_high"]
        )
        for source in sources:
            for city, total in frame.items():
                estimate = wilson(hits[source][city], total)
                writer.writerow(
                    [
                        source,
                        city,
                        total,
                        hits[source][city],
                        round(estimate.estimate, 6),
                        round(estimate.low, 6),
                        round(estimate.high, 6),
                    ]
                )
            estimate = wilson(sum(hits[source].values()), sum(frame.values()))
            writer.writerow(
                [
                    source,
                    "all",
                    sum(frame.values()),
                    sum(hits[source].values()),
                    round(estimate.estimate, 6),
                    round(estimate.low, 6),
                    round(estimate.high, 6),
                ]
            )
    logger.info(f"wrote {target}")
    return target


def main() -> None:
    """Measure coverage over the corpus and over the natural set."""
    radius_km = float(get("natural_set.authorities.match_radius_m")) / 1000.0
    indexes = load_authorities()
    sources = sorted(indexes)
    if not sources:
        raise ValueError("no authority holds a single entry; nothing to measure")
    logger.info(
        f"measuring {len(sources)} authorities at {radius_km * 1000:.0f} m: "
        + ", ".join(f"{source} ({len(indexes[source].places)})" for source in sources)
    )

    cities = [str(entry["city"]) for entry in get("corpus.cities")]
    frame: dict[str, int] = {}
    hits: dict[str, Counter[str]] = {source: Counter() for source in [*sources, UNION]}
    for city in cities:
        records = [record for record in read_city(city) if record.get("name")]
        frame[city] = len(records)
        for record in records:
            found = covered(record, indexes, radius_km)
            for source in found:
                hits[source][city] += 1
            if found:
                hits[UNION][city] += 1
        logger.info(
            f"{city}: {hits[UNION][city]} of {len(records)} covered by some authority"
        )

    write_by_city([*sources, UNION], frame, hits)

    drawn = natural_set_records()
    natural: dict[str, Any] = {}
    if drawn:
        for source in [*sources, UNION]:
            by_stratum: dict[str, list[int]] = {"A": [0, 0], "B": [0, 0]}
            for item in drawn:
                stratum = str(item["stratum"])
                by_stratum[stratum][1] += 1
                found = covered(
                    {"lat": item["shown"]["lat"], "lon": item["shown"]["lon"]},
                    indexes,
                    radius_km,
                )
                if source in found or (source == UNION and found):
                    by_stratum[stratum][0] += 1
            natural[source] = {
                stratum: {
                    "covered": counts[0],
                    "records": counts[1],
                    "coverage": round(wilson(counts[0], counts[1]).estimate, 6),
                    "ci_low": round(wilson(counts[0], counts[1]).low, 6),
                    "ci_high": round(wilson(counts[0], counts[1]).high, 6),
                }
                for stratum, counts in by_stratum.items()
            }
        logger.info(
            "natural set covered by some authority: "
            + ", ".join(
                f"{stratum} {natural[UNION][stratum]['covered']}/"
                f"{natural[UNION][stratum]['records']}"
                for stratum in ("A", "B")
            )
        )

    manifest = json.loads(manifest_path().read_text(encoding="utf-8"))
    summary = {
        "measured_on": date.today().isoformat(),
        "match_radius_m": get("natural_set.authorities.match_radius_m"),
        "corpus_records": sum(frame.values()),
        "sources": {
            source: {
                "entries": len(indexes[source].places),
                "licence": manifest["sources"].get(source, {}).get("licence"),
                "covered": sum(hits[source].values()),
                "coverage": round(
                    wilson(sum(hits[source].values()), sum(frame.values())).estimate, 6
                ),
                "ci_low": round(
                    wilson(sum(hits[source].values()), sum(frame.values())).low, 6
                ),
                "ci_high": round(
                    wilson(sum(hits[source].values()), sum(frame.values())).high, 6
                ),
            }
            for source in sources
        },
        "union": {
            "covered": sum(hits[UNION].values()),
            "coverage": round(
                wilson(sum(hits[UNION].values()), sum(frame.values())).estimate, 6
            ),
        },
        "natural_set": natural,
    }
    target = path("data_processed") / "authorities" / "coverage.json"
    target.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(f"wrote {target}")


if __name__ == "__main__":
    main()
