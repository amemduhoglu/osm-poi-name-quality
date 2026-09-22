"""Fetch the open reference data the two lookups and the M3 corruption need.

Two gazetteers, kept apart rather than merged, so that each one's coverage over
the corpus is reported on its own and the union is reported beside them:

    GeoNames cities500 (CC-BY 4.0)  every populated place of 500 or more people
                                    in the twelve corpus countries, with its
                                    alternate names. This is the bulk source and
                                    the one the M3 corruption draws distant place
                                    names from.
    Wikidata (CC0)                  human settlements within a fixed radius of
                                    each city centre, with labels and aliases.
                                    Independent of GeoNames and of OpenStreetMap,
                                    which answers the objection that a reference
                                    derived from the data under test cannot
                                    arbitrate it.

Constraint C2 governs the choice: both are redistributable, so the whole
pipeline can be deposited. The calling codes the other lookup needs travel
inside the ``phonenumbers`` package and need no download.

Outputs:
    data/raw/references/cities500.zip            the download, kept, not tracked
    data/interim/references/gazetteer.csv        both sources, one row per place
    data/processed/references/reference_manifest.json
                                                 URLs, licences, access dates,
                                                 row counts, tracked as evidence

Usage:
    python scripts/fetch_references.py
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import zipfile
from datetime import date
from pathlib import Path
from typing import Any

import httpx
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

from poi_audit.config import get, path
from poi_audit.corpus import distance_km
from poi_audit.logsetup import setup
from scripts.measure_candidate_cities import USER_AGENT, validate_centre

logger = setup("fetch_references")

FIELDNAMES = [
    "source",
    "place_id",
    "name",
    "alternate_names",
    "lat",
    "lon",
    "country",
    "feature_code",
    "population",
]

# The GeoNames dump is a headerless tab-separated file. Only these columns are
# read; the rest (admin codes, elevation, timezone) carry nothing this study
# asks of a gazetteer.
GEONAMES_COLUMNS = {
    "geonameid": 0,
    "name": 1,
    "asciiname": 2,
    "alternatenames": 3,
    "latitude": 4,
    "longitude": 5,
    "feature_code": 7,
    "country_code": 8,
    "population": 14,
}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30))
async def download(client: httpx.AsyncClient, url: str) -> bytes:
    """Download one file, retrying with exponential backoff.

    Args:
        client: The HTTP client to use.
        url: The address to fetch.

    Returns:
        The response body.

    Raises:
        httpx.HTTPStatusError: If the server answers with an error status.
    """
    response = await client.get(url)
    response.raise_for_status()
    return response.content


def parse_geonames(payload: bytes, countries: set[str]) -> list[dict[str, Any]]:
    """Read the GeoNames dump and keep the corpus countries.

    Args:
        payload: The downloaded zip archive.
        countries: The two-letter country codes to keep.

    Returns:
        One row per populated place, with its alternate names preserved as the
        dump writes them, separated by commas.
    """
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".txt"))
        with archive.open(name) as handle:
            text = io.TextIOWrapper(handle, encoding="utf-8")
            for line in text:
                parts = line.rstrip("\n").split("\t")
                if len(parts) <= GEONAMES_COLUMNS["population"]:
                    continue
                if parts[GEONAMES_COLUMNS["country_code"]] not in countries:
                    continue
                alternates = parts[GEONAMES_COLUMNS["alternatenames"]]
                ascii_name = parts[GEONAMES_COLUMNS["asciiname"]]
                if ascii_name and ascii_name not in alternates:
                    alternates = (
                        f"{alternates},{ascii_name}" if alternates else ascii_name
                    )
                rows.append(
                    {
                        "source": "geonames",
                        "place_id": parts[GEONAMES_COLUMNS["geonameid"]],
                        "name": parts[GEONAMES_COLUMNS["name"]],
                        "alternate_names": alternates,
                        "lat": float(parts[GEONAMES_COLUMNS["latitude"]]),
                        "lon": float(parts[GEONAMES_COLUMNS["longitude"]]),
                        "country": parts[GEONAMES_COLUMNS["country_code"]],
                        "feature_code": parts[GEONAMES_COLUMNS["feature_code"]],
                        "population": parts[GEONAMES_COLUMNS["population"]],
                    }
                )
    return rows


def wikidata_query(lat: float, lon: float, radius_km: float) -> str:
    """Build the settlement query for one city centre.

    Labels and aliases are collected together, because an address may be written
    with either and the lookup only asks whether the written form names a place
    near the coordinate.

    Args:
        lat: Centre latitude in degrees.
        lon: Centre longitude in degrees.
        radius_km: Search radius in kilometres.

    Returns:
        The SPARQL source.
    """
    classes = " ".join(
        f"wd:{name}"
        for name in get("references.gazetteers.wikidata.settlement_classes")
    )
    languages = ", ".join(
        f'"{code}"' for code in get("references.gazetteers.wikidata.label_languages")
    )
    return f"""
SELECT ?place ?lat ?lon (GROUP_CONCAT(DISTINCT ?nm; separator="|") AS ?names)
WHERE {{
  SERVICE wikibase:around {{
    ?place wdt:P625 ?location .
    bd:serviceParam wikibase:center "Point({lon} {lat})"^^geo:wktLiteral .
    bd:serviceParam wikibase:radius "{radius_km}" .
  }}
  VALUES ?type {{ {classes} }}
  ?place wdt:P31 ?type .
  ?place p:P625/psv:P625 ?node .
  ?node wikibase:geoLatitude ?lat ; wikibase:geoLongitude ?lon .
  {{ ?place rdfs:label ?nm }} UNION {{ ?place skos:altLabel ?nm }}
  FILTER(LANG(?nm) IN ({languages}))
}}
GROUP BY ?place ?lat ?lon
""".strip()


@retry(
    stop=stop_after_attempt(5), wait=wait_exponential(multiplier=80, min=80, max=400)
)
async def run_sparql(client: httpx.AsyncClient, url: str, query: str) -> dict[str, Any]:
    """Run one SPARQL query against the Wikidata endpoint.

    The endpoint throttles bursts and answers a throttled request with status
    429, so the backoff here is measured in minutes rather than seconds.

    Args:
        client: The HTTP client to use.
        url: The endpoint address.
        query: The SPARQL source.

    Returns:
        The parsed JSON bindings.

    Raises:
        httpx.HTTPStatusError: If the endpoint answers with an error status.
    """
    response = await client.get(
        url,
        params={"query": query, "format": "json"},
        headers={"Accept": "application/sparql-results+json"},
    )
    if response.status_code == 429:
        logger.warning("Wikidata is throttling; backing off")
    response.raise_for_status()
    return response.json()


def parse_wikidata(payload: dict[str, Any], country: str) -> list[dict[str, Any]]:
    """Turn one Wikidata answer into gazetteer rows.

    Args:
        payload: The SPARQL result.
        country: The country the query was centred in, recorded so that the
            distant-donor draw can stay inside one country.

    Returns:
        One row per settlement, its first name taken as the place name and the
        rest kept as alternates.
    """
    rows: list[dict[str, Any]] = []
    for binding in payload.get("results", {}).get("bindings", []):
        names = [n for n in binding["names"]["value"].split("|") if n]
        if not names:
            continue
        rows.append(
            {
                "source": "wikidata",
                "place_id": binding["place"]["value"].rsplit("/", 1)[-1],
                "name": names[0],
                "alternate_names": ",".join(names[1:]),
                "lat": float(binding["lat"]["value"]),
                "lon": float(binding["lon"]["value"]),
                "country": country,
                "feature_code": "settlement",
                "population": "",
            }
        )
    return rows


def deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop rows repeated across city queries, keeping the first.

    Args:
        rows: Gazetteer rows, possibly holding a place twice because two city
            discs overlap.

    Returns:
        The rows in their original order, each place once per source.
    """
    seen: set[tuple[str, str]] = set()
    kept: list[dict[str, Any]] = []
    for row in rows:
        key = (row["source"], row["place_id"])
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept


async def fetch_all() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch both gazetteers and return their rows and the source record.

    Returns:
        Every gazetteer row, and the manifest describing where each came from.
    """
    cities = list(get("corpus.cities"))
    countries = {str(city["country"]) for city in cities}
    geonames_cfg = get("references.gazetteers.geonames")
    wikidata_cfg = get("references.gazetteers.wikidata")

    raw_dir = path("data_raw") / "references"
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "cities500.zip"

    manifest: dict[str, Any] = {"gazetteers": {}}
    rows: list[dict[str, Any]] = []

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(300.0),
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
    ) as client:
        if archive.exists():
            logger.info(f"reading {archive} from disk")
            payload = archive.read_bytes()
            accessed = date.fromtimestamp(archive.stat().st_mtime).isoformat()
        else:
            logger.info(f"downloading {geonames_cfg['url']}")
            payload = await download(client, str(geonames_cfg["url"]))
            archive.write_bytes(payload)
            accessed = date.today().isoformat()
        geonames_rows = parse_geonames(payload, countries)
        rows.extend(geonames_rows)
        manifest["gazetteers"]["geonames"] = {
            "url": geonames_cfg["url"],
            "dataset": geonames_cfg["dataset"],
            "licence": geonames_cfg["licence"],
            "access_date": accessed,
            "countries": sorted(countries),
            "places": len(geonames_rows),
        }
        logger.info(
            f"geonames: {len(geonames_rows)} places in {len(countries)} countries"
        )

        radius_km = float(wikidata_cfg["radius_km"])
        wikidata_rows: list[dict[str, Any]] = []
        cache_dir = raw_dir / "wikidata"
        cache_dir.mkdir(parents=True, exist_ok=True)
        unavailable: str | None = None
        for city in cities:
            if unavailable:
                break
            lat, lon = validate_centre(city)
            name = str(city["city"])
            cached = cache_dir / f"{name.lower().replace(' ', '-')}.json"
            if cached.exists():
                logger.info(f"reading Wikidata for {name} from disk")
                answer = json.loads(cached.read_text(encoding="utf-8"))
            else:
                logger.info(f"querying Wikidata around {name}")
                try:
                    answer = await run_sparql(
                        client,
                        str(wikidata_cfg["url"]),
                        wikidata_query(lat, lon, radius_km),
                    )
                except (RetryError, httpx.HTTPError) as error:
                    # The second authority is not allowed to hold up the phase,
                    # and a partial one is worse than none: a gazetteer covering
                    # some cities and not others would make coverage differ by
                    # city for a reason that has nothing to do with the data.
                    unavailable = f"{type(error).__name__} while querying {name}"
                    logger.warning(f"Wikidata unavailable: {unavailable}")
                    break
                cached.write_text(
                    json.dumps(answer, ensure_ascii=False), encoding="utf-8"
                )
                await asyncio.sleep(float(wikidata_cfg["pause_between_queries_sec"]))
            wikidata_rows.extend(parse_wikidata(answer, str(city["country"])))

        if unavailable:
            wikidata_rows = []
            for stale in cache_dir.glob("*.json"):
                stale.unlink()
            manifest["gazetteers"]["wikidata"] = {
                "url": wikidata_cfg["url"],
                "licence": wikidata_cfg["licence"],
                "status": "unavailable",
                "attempted_on": date.today().isoformat(),
                "reason": unavailable,
                "places": 0,
            }
            logger.warning(
                "the gazetteer holds GeoNames only; re-run when the endpoint answers"
            )
        else:
            wikidata_rows = deduplicate(wikidata_rows)
            manifest["gazetteers"]["wikidata"] = {
                "url": wikidata_cfg["url"],
                "licence": wikidata_cfg["licence"],
                "settlement_classes": wikidata_cfg["settlement_classes"],
                "radius_km": radius_km,
                "access_date": date.today().isoformat(),
                "places": len(wikidata_rows),
            }
            logger.info(f"wikidata: {len(wikidata_rows)} settlements")
        rows.extend(wikidata_rows)

    manifest["calling_codes"] = dict(get("references.calling_codes"))
    return rows, manifest


def write_outputs(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> Path:
    """Write the gazetteer and its manifest.

    Args:
        rows: Every gazetteer row.
        manifest: The source record.

    Returns:
        The gazetteer file written.
    """
    out_dir = path("data_interim") / "references"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "gazetteer.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"wrote {target} with {len(rows)} rows")

    processed = path("data_processed") / "references"
    processed.mkdir(parents=True, exist_ok=True)
    manifest["rows"] = len(rows)
    manifest_path = processed / "reference_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(f"wrote {manifest_path}")
    return target


def report_span(rows: list[dict[str, Any]]) -> None:
    """Log how far each city centre is from its nearest gazetteer place.

    A lookup that has nothing near a city cannot arbitrate that city's records,
    and this is the cheapest place to notice it.

    Args:
        rows: Every gazetteer row.
    """
    for city in get("corpus.cities"):
        lat, lon = validate_centre(city)
        nearest = min(
            (distance_km(lat, lon, row["lat"], row["lon"]) for row in rows),
            default=float("inf"),
        )
        logger.info(f"{city['city']}: nearest gazetteer place {nearest:.1f} km away")


async def main() -> None:
    """Fetch both gazetteers, write them, and report their reach."""
    rows, manifest = await fetch_all()
    write_outputs(rows, manifest)
    report_span(rows)


if __name__ == "__main__":
    asyncio.run(main())
