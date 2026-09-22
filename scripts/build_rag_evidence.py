"""Retrieve the evidence the retrieval-augmented arm shows beside each record.

For every item the arm is asked about, two lists are drawn from the open
point-of-interest authorities: the entries nearest the record's position, and
the entries within the city disc whose names are closest to the record's name
by the retrieval baseline's encoder. Nothing is drawn from the corpus, so the
evidence carries no trace of how an item set was built, and nothing is decided
here: the lists are written as retrieved and the model reads them.

The encoder is served by the configured release of the serving stack, the same
one every scored run is held to.

Outputs:
    data/processed/retrieval_augmented/evidence.jsonl
    data/processed/retrieval_augmented/manifest.json

Usage:
    python -m scripts.build_rag_evidence
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

from poi_audit.config import get, secret
from poi_audit.corpus import distance_km
from poi_audit.inference import evidence_path, item_set
from poi_audit.logsetup import setup
from poi_audit.references import Gazetteer, Place
from poi_audit.retrieval import Encoder, cosine
from scripts.measure_authority_coverage import load_authorities

logger = setup("build_rag_evidence")

SETS = ("residual_control", "natural", "natural_extension")
BATCH = 256


def nearest_entries(
    record: dict[str, Any], indexes: list[Gazetteer], radius_m: float, limit: int
) -> list[dict[str, Any]]:
    """Return the authority entries nearest a record's position.

    Args:
        record: Fields carrying ``lat`` and ``lon``.
        indexes: One spatial index per authority.
        radius_m: The search radius in metres.
        limit: The most entries returned.

    Returns:
        Each entry's name and its distance in whole metres, nearest first.
    """
    found: list[tuple[float, Place]] = []
    for index in indexes:
        for entry in index.nearby(record["lat"], record["lon"], radius_m / 1000.0):
            away = distance_km(record["lat"], record["lon"], entry.lat, entry.lon)
            found.append((away, entry))
    found.sort(key=lambda pair: (pair[0], pair[1].display_name))
    return [
        {"name": entry.display_name, "distance_m": round(away * 1000.0)}
        for away, entry in found[:limit]
    ]


def similar_entries(
    record: dict[str, Any],
    candidates: list[Place],
    vectors: dict[str, list[float]],
    limit: int,
) -> list[dict[str, Any]]:
    """Return the candidate entries whose names are closest to a record's name.

    Args:
        record: Fields carrying ``name``, ``lat`` and ``lon``.
        candidates: The authority entries within the city disc.
        vectors: Every embedded text keyed by itself.
        limit: The most entries returned.

    Returns:
        Each entry's name and its distance in kilometres to one decimal, most
        similar first.
    """
    query = vectors[str(record["name"])]
    scored = sorted(
        candidates,
        key=lambda entry: (
            -cosine(query, vectors[entry.display_name]),
            entry.display_name,
        ),
    )
    return [
        {
            "name": entry.display_name,
            "distance_km": round(
                distance_km(record["lat"], record["lon"], entry.lat, entry.lon), 1
            ),
        }
        for entry in scored[:limit]
    ]


async def embed_all(encoder: Encoder, texts: list[str]) -> dict[str, list[float]]:
    """Embed every text once, in batches.

    Args:
        encoder: The embedding client.
        texts: The texts, deduplicated by the caller.

    Returns:
        Text to vector.
    """
    vectors: dict[str, list[float]] = {}
    for start in range(0, len(texts), BATCH):
        chunk = texts[start : start + BATCH]
        for text, vector in zip(chunk, await encoder.embed(chunk), strict=True):
            vectors[text] = vector
        if start // BATCH % 10 == 0:
            logger.info(f"embedded {len(vectors)} of {len(texts)}")
    return vectors


async def run() -> None:
    """Retrieve and write the evidence for every item the arm is asked about."""
    settings = get("retrieval_augmented")
    authorities = load_authorities()
    indexes = [authorities[source] for source in settings["sources"]]
    items = [(kind, item) for kind in SETS for item in item_set(kind)]
    radius_km = float(settings["similar_names"]["radius_km"])

    candidates: dict[str, list[Place]] = {}
    texts: set[str] = set()
    for _, item in items:
        shown = item.shown
        nearby = [
            entry
            for index in indexes
            for entry in index.nearby(shown["lat"], shown["lon"], radius_km)
        ]
        candidates[item.item_id] = nearby
        texts.add(str(shown["name"]))
        texts.update(entry.display_name for entry in nearby)
    logger.info(f"{len(items)} items, {len(texts)} distinct texts to embed")

    base_url = secret("models.serving.base_url_env")
    async with Encoder(base_url, str(settings["encoder"])) as encoder:
        vectors = await embed_all(encoder, sorted(texts))

    target = evidence_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for kind, item in items:
            evidence = {
                "radius_m": int(settings["nearest"]["radius_m"]),
                "radius_km": int(radius_km),
                "nearest": nearest_entries(
                    item.shown,
                    indexes,
                    float(settings["nearest"]["radius_m"]),
                    int(settings["nearest"]["max_entries"]),
                ),
                "similar": similar_entries(
                    item.shown,
                    candidates[item.item_id],
                    vectors,
                    int(settings["similar_names"]["max_entries"]),
                ),
            }
            handle.write(
                json.dumps(
                    {"item_id": item.item_id, "set": kind, "evidence": evidence},
                    ensure_ascii=False,
                )
                + "\n"
            )
    manifest = {
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "settings": settings,
        "items": {kind: sum(1 for k, _ in items if k == kind) for kind in SETS},
        "texts_embedded": len(texts),
        "with_nearest_entry": sum(
            1
            for _, item in items
            if nearest_entries(
                item.shown, indexes, float(settings["nearest"]["radius_m"]), 1
            )
        ),
    }
    (target.parent / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(f"wrote {target}")


def main() -> None:
    """Command line entry point."""
    asyncio.run(run())


if __name__ == "__main__":
    main()
