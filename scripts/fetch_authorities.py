"""Fetch the open authorities a natural-set label may rest on.

The rule baseline's gazetteers hold populated places, which settle an address
and say nothing about whether a name belongs to the establishment carrying it.
Answering that needs sources that hold the establishments themselves. Two are
open enough to deposit whole:

    GeoNames per-country dumps (CC-BY 4.0)  named facilities, parks and areas
                                            inside each corpus city's disc,
                                            classes S and L. Class P is left to
                                            the baseline's own gazetteer.
    Wikidata (CC0)                          named places of the kinds the corpus
                                            keys admit, within the corpus disc
                                            of each city, with labels and
                                            aliases.

Municipal and national registers are added to the configuration one at a time,
each after its licence has been read. A city with no redistributable register
keeps none, because which cities can be audited this way is a finding rather
than a gap to fill.

Constraint C2 governs every source here: nothing that cannot be redistributed
under CC-BY or CC0 enters the pipeline, whatever it would add to coverage.

Outputs:
    data/raw/authorities/geonames/{CC}.zip      the downloads, kept, not tracked
    data/raw/authorities/wikidata/{city}.json   one cached answer per city
    data/interim/authorities/authorities.csv    every entry, one row each
    data/processed/authorities/authority_manifest.json
                                                URLs, licences, access dates and
                                                row counts, tracked as evidence

Usage:
    python -m scripts.fetch_authorities
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
from scripts.fetch_references import GEONAMES_COLUMNS, download
from scripts.measure_candidate_cities import USER_AGENT, validate_centre

logger = setup("fetch_authorities")

FIELDNAMES = [
    "source",
    "authority_id",
    "name",
    "alternate_names",
    "lat",
    "lon",
    "country",
    "city",
    "kind",
]

GEONAMES_POI = "geonames_poi"
WIKIDATA_POI = "wikidata_poi"

# The dump's feature class, which the reference fetcher has no use for and so
# does not name. Class S is spots, buildings and farms; class L is parks and
# areas; class P is populated places and belongs to the baseline's gazetteer.
FEATURE_CLASS_COLUMN = 6


def authorities_path() -> Path:
    """Return the file the fetcher writes every authority entry to."""
    return path("data_interim") / "authorities" / "authorities.csv"


def manifest_path() -> Path:
    """Return the file the fetcher writes the source record to."""
    return path("data_processed") / "authorities" / "authority_manifest.json"


def within_disc(lat: float, lon: float, cities: list[dict[str, Any]], km: float) -> str:
    """Return the corpus city whose disc a coordinate falls in.

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        cities: The corpus city entries.
        km: The disc radius in kilometres.

    Returns:
        The city's name, or an empty string when the coordinate falls in no
        disc. An entry outside every disc cannot cover a corpus record and is
        not kept, which is what holds the file to the area the corpus was
        extracted from.
    """
    for city in cities:
        centre_lat, centre_lon = float(city["lat"]), float(city["lon"])
        if distance_km(lat, lon, centre_lat, centre_lon) <= km:
            return str(city["city"])
    return ""


def parse_geonames_poi(
    payload: bytes, country: str, cities: list[dict[str, Any]], km: float
) -> list[dict[str, Any]]:
    """Read one country's GeoNames dump and keep its facilities in the discs.

    Args:
        payload: The downloaded zip archive.
        country: The two-letter country code the dump is for.
        cities: The corpus city entries in that country.
        km: The disc radius in kilometres.

    Returns:
        One row per entry of a kept feature class inside one of the discs, with
        its alternate names preserved as the dump writes them.
    """
    keep = {
        str(code)
        for code in get("natural_set.authorities.geonames_poi.feature_classes")
    }
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        # The country archives hold a readme beside the data, and the readme
        # sorts first. Naming the file the country names it keeps the parse off
        # a text that yields no rows and no error.
        name = f"{country}.txt"
        if name not in archive.namelist():
            raise ValueError(
                f"{name} is not in the archive: {sorted(archive.namelist())}"
            )
        with archive.open(name) as handle:
            for line in io.TextIOWrapper(handle, encoding="utf-8"):
                parts = line.rstrip("\n").split("\t")
                if len(parts) <= GEONAMES_COLUMNS["population"]:
                    continue
                if parts[FEATURE_CLASS_COLUMN] not in keep:
                    continue
                lat = float(parts[GEONAMES_COLUMNS["latitude"]])
                lon = float(parts[GEONAMES_COLUMNS["longitude"]])
                city = within_disc(lat, lon, cities, km)
                if not city:
                    continue
                alternates = parts[GEONAMES_COLUMNS["alternatenames"]]
                ascii_name = parts[GEONAMES_COLUMNS["asciiname"]]
                if ascii_name and ascii_name not in alternates:
                    alternates = (
                        f"{alternates},{ascii_name}" if alternates else ascii_name
                    )
                rows.append(
                    {
                        "source": GEONAMES_POI,
                        "authority_id": parts[GEONAMES_COLUMNS["geonameid"]],
                        "name": parts[GEONAMES_COLUMNS["name"]],
                        "alternate_names": alternates,
                        "lat": lat,
                        "lon": lon,
                        "country": country,
                        "city": city,
                        "kind": parts[GEONAMES_COLUMNS["feature_code"]],
                    }
                )
    return rows


def wikidata_poi_query(
    lat: float, lon: float, radius_km: float, classes: list[str]
) -> str:
    """Build the point-of-interest query for one city centre.

    Args:
        lat: Centre latitude in degrees.
        lon: Centre longitude in degrees.
        radius_km: Search radius in kilometres.
        classes: The Wikidata class identifiers to ask for.

    Returns:
        The SPARQL source.
    """
    values = " ".join(f"wd:{name}" for name in classes)
    languages = ", ".join(
        f'"{code}"'
        for code in get("natural_set.authorities.wikidata_poi.label_languages")
    )
    return f"""
SELECT ?place ?type ?lat ?lon
       (GROUP_CONCAT(DISTINCT ?nm; separator="|") AS ?names)
WHERE {{
  SERVICE wikibase:around {{
    ?place wdt:P625 ?location .
    bd:serviceParam wikibase:center "Point({lon} {lat})"^^geo:wktLiteral .
    bd:serviceParam wikibase:radius "{radius_km}" .
  }}
  VALUES ?type {{ {values} }}
  ?place wdt:P31 ?type .
  ?place p:P625/psv:P625 ?node .
  ?node wikibase:geoLatitude ?lat ; wikibase:geoLongitude ?lon .
  {{ ?place rdfs:label ?nm }} UNION {{ ?place skos:altLabel ?nm }}
  FILTER(LANG(?nm) IN ({languages}))
}}
GROUP BY ?place ?type ?lat ?lon
""".strip()


@retry(
    stop=stop_after_attempt(5), wait=wait_exponential(multiplier=80, min=80, max=400)
)
async def run_sparql(client: httpx.AsyncClient, url: str, query: str) -> dict[str, Any]:
    """Run one SPARQL query against the Wikidata endpoint.

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


async def query_city(
    client: httpx.AsyncClient,
    url: str,
    lat: float,
    lon: float,
    radius_km: float,
    classes: list[str],
    pause_sec: float,
) -> list[dict[str, Any]]:
    """Ask the endpoint for one city, splitting the class list if it times out.

    A dense city asked for every class at once can exceed the endpoint's own
    query deadline. Splitting the class list and asking twice returns the same
    entries for more requests, which is the trade this study wants: a city
    covered less than another for a reason that has nothing to do with the data
    would make the coverage measurement meaningless.

    Args:
        client: The HTTP client to use.
        url: The endpoint address.
        lat: Centre latitude in degrees.
        lon: Centre longitude in degrees.
        radius_km: Search radius in kilometres.
        classes: The Wikidata class identifiers to ask for.
        pause_sec: How long to wait after a request.

    Returns:
        The bindings from one or several answers.

    Raises:
        RetryError: If the endpoint fails for a single class.
        httpx.HTTPError: If the endpoint fails for a single class.
    """
    try:
        answer = await run_sparql(
            client, url, wikidata_poi_query(lat, lon, radius_km, classes)
        )
        await asyncio.sleep(pause_sec)
        return list(answer.get("results", {}).get("bindings", []))
    except (RetryError, httpx.HTTPError):
        if len(classes) == 1 or not get(
            "natural_set.authorities.wikidata_poi.split_on_timeout"
        ):
            raise
        middle = len(classes) // 2
        logger.warning(
            f"splitting {len(classes)} classes into {middle} and "
            f"{len(classes) - middle} after a failed request"
        )
        bindings = await query_city(
            client, url, lat, lon, radius_km, classes[:middle], pause_sec
        )
        bindings.extend(
            await query_city(
                client, url, lat, lon, radius_km, classes[middle:], pause_sec
            )
        )
        return bindings


def parse_wikidata_poi(
    bindings: list[dict[str, Any]], country: str, city: str
) -> list[dict[str, Any]]:
    """Turn one city's answers into authority rows.

    Args:
        bindings: The SPARQL bindings.
        country: The city's country code.
        city: The city the query was centred on.

    Returns:
        One row per place, its first name taken as the name and the rest kept
        as alternates.
    """
    rows: list[dict[str, Any]] = []
    for binding in bindings:
        names = [n for n in binding["names"]["value"].split("|") if n]
        if not names:
            continue
        rows.append(
            {
                "source": WIKIDATA_POI,
                "authority_id": binding["place"]["value"].rsplit("/", 1)[-1],
                "name": names[0],
                "alternate_names": ",".join(names[1:]),
                "lat": float(binding["lat"]["value"]),
                "lon": float(binding["lon"]["value"]),
                "country": country,
                "city": city,
                "kind": binding["type"]["value"].rsplit("/", 1)[-1],
            }
        )
    return rows


def deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop entries repeated across queries, keeping the first.

    Args:
        rows: Authority rows, possibly holding one entry twice because a class
            list was split or two discs overlap.

    Returns:
        The rows in their original order, each entry once per source.
    """
    seen: set[tuple[str, str]] = set()
    kept: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row["source"]), str(row["authority_id"]))
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept


async def fetch_geonames(
    client: httpx.AsyncClient, cities: list[dict[str, Any]], radius_km: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch and read every corpus country's GeoNames dump.

    Args:
        client: The HTTP client to use.
        cities: The corpus city entries.
        radius_km: The disc radius in kilometres.

    Returns:
        The rows and the source record.
    """
    settings = get("natural_set.authorities.geonames_poi")
    raw_dir = path("data_raw") / "authorities" / "geonames"
    raw_dir.mkdir(parents=True, exist_ok=True)
    by_country: dict[str, list[dict[str, Any]]] = {}
    for city in cities:
        by_country.setdefault(str(city["country"]), []).append(city)

    rows: list[dict[str, Any]] = []
    accessed: dict[str, str] = {}
    for country, country_cities in by_country.items():
        archive = raw_dir / f"{country}.zip"
        if archive.exists():
            logger.info(f"reading {archive} from disk")
            payload = archive.read_bytes()
            accessed[country] = date.fromtimestamp(archive.stat().st_mtime).isoformat()
        else:
            url = str(settings["url_template"]).format(country=country)
            logger.info(f"downloading {url}")
            payload = await download(client, url)
            archive.write_bytes(payload)
            accessed[country] = date.today().isoformat()
        country_rows = parse_geonames_poi(payload, country, country_cities, radius_km)
        rows.extend(country_rows)
        logger.info(f"{country}: {len(country_rows)} entries inside the discs")
    return rows, {
        "url_template": settings["url_template"],
        "licence": settings["licence"],
        "feature_classes": list(settings["feature_classes"]),
        "access_dates": accessed,
        "entries": len(rows),
    }


async def fetch_wikidata(
    client: httpx.AsyncClient, cities: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Query Wikidata for every corpus city, one city at a time.

    Args:
        client: The HTTP client to use.
        cities: The corpus city entries.

    Returns:
        The rows and the source record. An endpoint failure leaves the source
        marked unavailable and its cache cleared rather than shipping an
        authority that covers some cities and not others.
    """
    settings = get("natural_set.authorities.wikidata_poi")
    radius_km = float(settings["radius_km"])
    pause_sec = float(settings["pause_between_queries_sec"])
    classes = [str(name) for name in settings["classes"]]
    cache_dir = path("data_raw") / "authorities" / "wikidata"
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    per_city: dict[str, int] = {}
    for city in cities:
        lat, lon = validate_centre(city)
        name = str(city["city"])
        cached = cache_dir / f"{name.lower().replace(' ', '-')}.json"
        if cached.exists():
            logger.info(f"reading Wikidata for {name} from disk")
            bindings = json.loads(cached.read_text(encoding="utf-8"))
        else:
            logger.info(f"querying Wikidata around {name}")
            try:
                bindings = await query_city(
                    client,
                    str(settings["url"]),
                    lat,
                    lon,
                    radius_km,
                    classes,
                    pause_sec,
                )
            except (RetryError, httpx.HTTPError) as error:
                for stale in cache_dir.glob("*.json"):
                    stale.unlink()
                logger.warning(f"Wikidata unavailable: {type(error).__name__}")
                return [], {
                    "url": settings["url"],
                    "licence": settings["licence"],
                    "status": "unavailable",
                    "attempted_on": date.today().isoformat(),
                    "failed_on_city": name,
                    "entries": 0,
                }
            cached.write_text(
                json.dumps(bindings, ensure_ascii=False), encoding="utf-8"
            )
        city_rows = parse_wikidata_poi(bindings, str(city["country"]), name)
        per_city[name] = len(city_rows)
        rows.extend(city_rows)
        logger.info(f"{name}: {len(city_rows)} Wikidata entries")
    return rows, {
        "url": settings["url"],
        "licence": settings["licence"],
        "radius_km": radius_km,
        "classes": classes,
        "access_date": date.today().isoformat(),
        "entries_per_city": per_city,
        "entries": len(rows),
    }


def write_outputs(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> Path:
    """Write the authority entries and the source record.

    Args:
        rows: Every authority row.
        manifest: The source record.

    Returns:
        The entry file written.
    """
    target = authorities_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"wrote {target} with {len(rows)} entries")

    record = manifest_path()
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(f"wrote {record}")
    return target


async def main() -> None:
    """Fetch every configured authority and write it out."""
    cities = [dict(city) for city in get("corpus.cities")]
    radius_km = float(get("corpus.candidate_selection.radius_m")) / 1000.0
    registers = list(get("natural_set.authorities.registers"))

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(
            float(get("natural_set.authorities.wikidata_poi.timeout_sec"))
        ),
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
    ) as client:
        geonames_rows, geonames_record = await fetch_geonames(client, cities, radius_km)
        wikidata_rows, wikidata_record = await fetch_wikidata(client, cities)

    rows = deduplicate([*geonames_rows, *wikidata_rows])
    manifest = {
        "built_on": date.today().isoformat(),
        "disc_radius_km": radius_km,
        "match_radius_m": get("natural_set.authorities.match_radius_m"),
        "sources": {
            GEONAMES_POI: geonames_record,
            WIKIDATA_POI: wikidata_record,
        },
        "registers": registers,
        "entries": len(rows),
    }
    write_outputs(rows, manifest)
    if not registers:
        logger.info("no municipal or national register is configured yet")


if __name__ == "__main__":
    asyncio.run(main())
