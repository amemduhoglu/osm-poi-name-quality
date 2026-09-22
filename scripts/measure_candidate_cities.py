"""Measure the candidate cities so that twelve can be selected by a fixed rule.

Every candidate in ``corpus.candidates`` is counted inside a disc of the
configured radius around its centre. The counts decide two things: whether a
candidate can carry its share of the item set at all, and where it falls on the
scale that assigns maturity. Both rules live in the configuration file.

The script writes one row per candidate to the processed data directory and the
raw Overpass responses to the interim one, and prints the eliminated candidates.
A city whose response is already on disk is not queried again, so the run
resumes after an interruption and a candidate added later costs one query.

It selects nothing: selection is a separate step that reads this output, so that
the measurement can be repeated without repeating the choice.

Usage:
    python scripts/measure_candidate_cities.py
"""

from __future__ import annotations

import asyncio
import csv
import json
from datetime import date
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from poi_audit.config import get, path
from poi_audit.logsetup import setup

logger = setup("measure_candidate_cities")

LAT_RANGE = (-90.0, 90.0)
LON_RANGE = (-180.0, 180.0)

# Overpass refuses a request that carries a library's default user agent. The
# string names the project rather than its author, so that it discloses nothing
# under anonymous review.
USER_AGENT = "poi-name-audit/0.1 (OSM attribute quality research)"


def validate_centre(candidate: dict[str, Any]) -> tuple[float, float]:
    """Return a candidate's centre after checking it lies on the Earth.

    Args:
        candidate: One entry of ``corpus.candidates``.

    Returns:
        The latitude and longitude in degrees.

    Raises:
        ValueError: If either coordinate is outside its valid range.
    """
    lat = float(candidate["lat"])
    lon = float(candidate["lon"])
    if not LAT_RANGE[0] <= lat <= LAT_RANGE[1]:
        raise ValueError(f"latitude out of range for {candidate['city']}: {lat}")
    if not LON_RANGE[0] <= lon <= LON_RANGE[1]:
        raise ValueError(f"longitude out of range for {candidate['city']}: {lon}")
    return lat, lon


def metric_filters(spec: dict[str, Any]) -> str:
    """Render one metric's tag filters as Overpass filter syntax.

    Args:
        spec: A metric definition holding ``require_all`` and optionally
            ``require_any``.

    Returns:
        The concatenated filter clauses. A ``require_any`` list becomes a key
        regular expression, which is how Overpass expresses a disjunction over
        keys.
    """
    clauses = "".join(f'["{key}"]' for key in spec.get("require_all", []))
    any_keys = spec.get("require_any", [])
    if any_keys:
        alternation = "|".join(any_keys)
        clauses += f'[~"^({alternation})$"~"."]'
    return clauses


def build_query(lat: float, lon: float, radius_m: int, timeout_sec: int) -> str:
    """Build one Overpass query returning the tags of every named point of interest.

    The tags are downloaded once per city and every metric is then counted
    locally. Asking the server for one count per metric was tried first and
    times out: a disjunction over keys is expensive, and six of them in one
    query exceed the endpoint's gateway limit. Counting locally also keeps the
    metric definitions changeable without a second round of queries.

    Args:
        lat: Centre latitude in degrees.
        lon: Centre longitude in degrees.
        radius_m: Radius of the counting disc in metres.
        timeout_sec: Server-side timeout written into the query.

    Returns:
        The Overpass QL source.
    """
    poi_keys = list(get("corpus.poi_keys"))
    around = f"(around:{radius_m},{lat},{lon})"
    statements = "".join(f'nwr{around}["{key}"]["name"];' for key in poi_keys)
    return f"[out:json][timeout:{timeout_sec}];({statements});out tags;"


def count_metrics(payload: dict[str, Any]) -> dict[str, int]:
    """Count how many downloaded elements satisfy each configured metric.

    Args:
        payload: A parsed Overpass response holding elements with their tags.

    Returns:
        One total per metric, keyed by metric name.
    """
    metrics: dict[str, Any] = get("corpus.candidate_selection.metrics")
    totals = dict.fromkeys(metrics, 0)
    for element in payload.get("elements", []):
        tags = element.get("tags", {})
        for name, spec in metrics.items():
            if not all(key in tags for key in spec.get("require_all", [])):
                continue
            any_keys = spec.get("require_any", [])
            if any_keys and not any(key in tags for key in any_keys):
                continue
            totals[name] += 1
    return totals


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=5, max=120))
async def run_query(client: httpx.AsyncClient, url: str, query: str) -> dict[str, Any]:
    """Send one Overpass query and return its parsed response.

    Args:
        client: An open HTTP client.
        url: The Overpass endpoint.
        query: The Overpass QL source.

    Returns:
        The parsed JSON response.

    Raises:
        httpx.HTTPStatusError: If the endpoint refuses or fails the request.
    """
    response = await client.post(url, data={"data": query})
    response.raise_for_status()
    return response.json()


def disc_area_km2(radius_m: int) -> float:
    """Return the area of the counting disc in square kilometres.

    Args:
        radius_m: Radius in metres.

    Returns:
        The area, used to turn a count into a density.
    """
    from math import pi

    return pi * (radius_m / 1000.0) ** 2


async def measure_all() -> list[dict[str, Any]]:
    """Measure every candidate and return one record each.

    Returns:
        A record per candidate holding its identity, its counts, its density
        and whether it met every threshold.
    """
    selection: dict[str, Any] = get("corpus.candidate_selection")
    metrics: dict[str, Any] = selection["metrics"]
    metric_names = list(metrics)
    radius_m = int(selection["radius_m"])
    area = disc_area_km2(radius_m)
    raw_dir = path("data_interim") / "candidate_cities_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    timeout = httpx.Timeout(float(selection["overpass_timeout_sec"]) + 60.0)
    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        for candidate in get("corpus.candidates"):
            lat, lon = validate_centre(candidate)
            city = str(candidate["city"])
            slug = city.lower().replace(" ", "-")
            cached = raw_dir / f"{slug}.json"
            # A city already downloaded is not queried again: the run resumes
            # after an interruption, and a candidate added later costs only its
            # own query rather than the whole pool's.
            if cached.exists():
                payload = json.loads(cached.read_text(encoding="utf-8"))
                logger.info(f"reading {city} ({candidate['country']}) from disk")
            else:
                query = build_query(
                    lat, lon, radius_m, int(selection["overpass_timeout_sec"])
                )
                logger.info(f"measuring {city} ({candidate['country']})")
                payload = await run_query(client, str(selection["overpass_url"]), query)
                cached.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                await asyncio.sleep(float(selection["pause_between_cities_sec"]))
            counts = count_metrics(payload)
            failed = [
                name
                for name in metric_names
                if counts[name] < int(metrics[name]["threshold"])
            ]
            record: dict[str, Any] = {
                "city": city,
                "country": candidate["country"],
                "language": candidate["language"],
                "hypothesis_maturity": candidate["hypothesis_maturity"],
                "lat": lat,
                "lon": lon,
                **counts,
                "named_per_km2": round(counts["named"] / area, 2),
                "eligible": not failed,
                "failed_thresholds": ";".join(failed),
                # The access date belongs to the download, not to this run, so
                # it is read from the file that holds the response.
                "measured_on": date.fromtimestamp(cached.stat().st_mtime).isoformat(),
            }
            records.append(record)
            counted = ", ".join(f"{n}={counts[n]}" for n in metric_names)
            verdict = "eligible" if not failed else f"eliminated on {','.join(failed)}"
            logger.info(f"{city}: {counted} ({verdict})")
    return records


def write_table(records: list[dict[str, Any]]) -> None:
    """Write the measurement table to the interim data directory.

    Args:
        records: The measured candidates.
    """
    out_path = path("data_processed") / "candidate_city_counts.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    logger.info(f"wrote {len(records)} rows to {out_path}")


async def main() -> None:
    """Measure every candidate city and report which ones remain eligible."""
    records = await measure_all()
    write_table(records)
    eligible = [r for r in records if r["eligible"]]
    logger.info(f"{len(eligible)} of {len(records)} candidates met every threshold")
    for record in records:
        if not record["eligible"]:
            logger.warning(
                f"{record['city']} eliminated on {record['failed_thresholds']}"
            )


if __name__ == "__main__":
    asyncio.run(main())
