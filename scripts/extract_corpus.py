"""Extract the corpus for the twelve selected cities.

One record per named point of interest inside each city's disc, carrying a
representative coordinate and the element's full tag set. The coordinate is kept
because the density map needs it and because two of the error classes are
decided against position; the full tag set is kept because item construction
draws its fields from it, and a corpus that dropped a tag would have to be
extracted again.

The extraction date is part of the corpus definition and is written into the
manifest beside the counts, not left to the file system. Records are written in
a fixed order so that two extractions of the same download compare equal.

Outputs:
    data/interim/corpus/<city>.jsonl   one record per line, deposited not tracked
    data/processed/corpus_counts.csv   per-city counts, tracked as evidence
    data/processed/corpus_manifest.json  extraction dates, geometry, licence

Usage:
    python scripts/extract_corpus.py
"""

from __future__ import annotations

import asyncio
import csv
import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx

from poi_audit.config import get, path
from poi_audit.logsetup import setup
from scripts.measure_candidate_cities import (
    USER_AGENT,
    disc_area_km2,
    run_query,
    validate_centre,
)

logger = setup("extract_corpus")


def build_query(lat: float, lon: float, radius_m: int, timeout_sec: int) -> str:
    """Build the extraction query for one city.

    ``out center tags`` returns a coordinate for every element: its own for a
    node, the centre of its bounding box for a way or a relation. That is the
    representative point the density map draws and the reference lookups test.

    Args:
        lat: Centre latitude in degrees.
        lon: Centre longitude in degrees.
        radius_m: Radius of the extraction disc in metres.
        timeout_sec: Server-side timeout written into the query.

    Returns:
        The Overpass QL source.
    """
    poi_keys = list(get("corpus.poi_keys"))
    around = f"(around:{radius_m},{lat},{lon})"
    statements = "".join(f'nwr{around}["{key}"]["name"];' for key in poi_keys)
    return f"[out:json][timeout:{timeout_sec}];({statements});out center tags;"


def to_record(element: dict[str, Any], city: dict[str, Any]) -> dict[str, Any] | None:
    """Turn one Overpass element into a corpus record.

    Args:
        element: One element of an Overpass response.
        city: The city entry it was extracted for.

    Returns:
        The record, or None when the element carries no usable coordinate. A
        way whose members lie outside the downloaded area has no centre, and a
        record without a position cannot be scored on either reference lookup.
    """
    if "lat" in element and "lon" in element:
        lat, lon = float(element["lat"]), float(element["lon"])
    elif "center" in element:
        lat, lon = float(element["center"]["lat"]), float(element["center"]["lon"])
    else:
        return None
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        return None
    return {
        "city": city["city"],
        "country": city["country"],
        "language": city["language"],
        "maturity": city["maturity"],
        "osm_type": element["type"],
        "osm_id": element["id"],
        "lat": lat,
        "lon": lon,
        "name": element["tags"]["name"],
        "tags": element["tags"],
    }


def write_city(records: list[dict[str, Any]], out_dir: Path, city: str) -> Path:
    """Write one city's records in a fixed order.

    Args:
        records: The city's corpus records.
        out_dir: Directory the corpus is written to.
        city: City name, used for the file name.

    Returns:
        The file written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{city.lower().replace(' ', '-')}.jsonl"
    ordered = sorted(records, key=lambda r: (r["osm_type"], r["osm_id"]))
    with target.open("w", encoding="utf-8") as handle:
        for record in ordered:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return target


def count_fields(records: list[dict[str, Any]]) -> dict[str, int]:
    """Count how many records carry each field the item classes need.

    Args:
        records: One city's corpus records.

    Returns:
        One count per configured metric, under the same definitions the
        candidate measurement used, so that the corpus can be checked against
        the counts that selected the city.
    """
    metrics: dict[str, Any] = get("corpus.candidate_selection.metrics")
    totals = dict.fromkeys(metrics, 0)
    for record in records:
        tags = record["tags"]
        for name, spec in metrics.items():
            if not all(key in tags for key in spec.get("require_all", [])):
                continue
            any_keys = spec.get("require_any", [])
            if any_keys and not any(key in tags for key in any_keys):
                continue
            totals[name] += 1
    return totals


async def extract_all() -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Extract every selected city and return the per-city counts.

    Returns:
        The count rows and the extraction date per city.
    """
    selection: dict[str, Any] = get("corpus.candidate_selection")
    radius_m = int(selection["radius_m"])
    timeout_sec = int(selection["overpass_timeout_sec"])
    area = disc_area_km2(radius_m)
    raw_dir = path("data_raw") / "corpus"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir = path("data_interim") / "corpus"

    rows: list[dict[str, Any]] = []
    dates: dict[str, str] = {}
    timeout = httpx.Timeout(float(timeout_sec) + 60.0)
    async with httpx.AsyncClient(
        timeout=timeout, headers={"User-Agent": USER_AGENT}
    ) as client:
        for city in get("corpus.cities"):
            lat, lon = validate_centre(city)
            name = str(city["city"])
            cached = raw_dir / f"{name.lower().replace(' ', '-')}.json"
            if cached.exists():
                payload = json.loads(cached.read_text(encoding="utf-8"))
                logger.info(f"reading {name} from disk")
            else:
                logger.info(f"extracting {name} ({city['country']})")
                payload = await run_query(
                    client,
                    str(selection["overpass_url"]),
                    build_query(lat, lon, radius_m, timeout_sec),
                )
                cached.write_text(
                    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                )
                await asyncio.sleep(float(selection["pause_between_cities_sec"]))

            elements = payload.get("elements", [])
            records = [r for r in (to_record(e, city) for e in elements) if r]
            dropped = len(elements) - len(records)
            write_city(records, out_dir, name)
            counts = count_fields(records)
            rows.append(
                {
                    "city": name,
                    "country": city["country"],
                    "language": city["language"],
                    "maturity": city["maturity"],
                    "records": len(records),
                    "without_coordinate": dropped,
                    **counts,
                    "records_per_km2": round(len(records) / area, 2),
                    "extracted_on": date.fromtimestamp(
                        cached.stat().st_mtime
                    ).isoformat(),
                }
            )
            dates[name] = rows[-1]["extracted_on"]
            logger.info(
                f"{name}: {len(records)} records, {dropped} without a coordinate"
            )
    return rows, dates


def write_outputs(rows: list[dict[str, Any]], dates: dict[str, str]) -> None:
    """Write the per-city counts and the corpus manifest.

    Args:
        rows: One count row per city.
        dates: Extraction date per city.
    """
    counts_path = path("data_processed") / "corpus_counts.csv"
    with counts_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"wrote {counts_path}")

    manifest = {
        "source": get("corpus.source"),
        "licence": get("corpus.licence"),
        "script": get("corpus.script"),
        "poi_keys": get("corpus.poi_keys"),
        "radius_m": get("corpus.candidate_selection.radius_m"),
        "endpoint": get("corpus.candidate_selection.overpass_url"),
        "extraction_dates": dates,
        "records": sum(int(row["records"]) for row in rows),
    }
    manifest_path = path("data_processed") / "corpus_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(f"wrote {manifest_path}")


async def main() -> None:
    """Extract the corpus and report its size."""
    rows, dates = await extract_all()
    write_outputs(rows, dates)
    total = sum(int(row["records"]) for row in rows)
    logger.info(f"corpus holds {total} records across {len(rows)} cities")


if __name__ == "__main__":
    asyncio.run(main())
