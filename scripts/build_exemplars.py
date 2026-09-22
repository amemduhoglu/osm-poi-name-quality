"""Generate the few-shot exemplars from an invented vocabulary.

The few-shot arm needs worked examples of the residual question, and no model
may write or label one. Each exemplar here is composed by fixed rules from fixed
lists, so its label is a fact about how it was made rather than a judgement:

    A belonging exemplar carries an invented proper name followed by a
    descriptive noun that names what its category is (an establishment called a
    roastery, tagged as a cafe).

    A borrowed exemplar carries a descriptive noun from a different, unrelated
    category (a roastery tagged as a dentist), which is what a name taken from
    another place looks like on the record it lands on.

Proper names, streets and settlements are composed from syllables that spell no
real place, coordinates lie in the open South Pacific, and a descriptive noun is
used only if no folded token of it is in the corpus category-token dictionary,
so the exemplars teach neither a token rule nor a recalled place. The output is
byte-identical on every run.

Outputs:
    prompts/exemplars.generated.v1.json

Usage:
    python -m scripts.build_exemplars
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from poi_audit import screen
from poi_audit.config import get
from poi_audit.corpus import fold
from poi_audit.logsetup import setup

logger = setup("build_exemplars")

TARGET = (
    Path(__file__).resolve().parent.parent / "prompts" / "exemplars.generated.v1.json"
)

# What each category's own name would call the place. Candidates are listed in
# order; the first whose folded tokens the token dictionary does not hold is used.
NOUNS: dict[str, list[str]] = {
    "amenity=cafe": ["Demitasse Room", "Percolator", "Cafetiere Loft"],
    "shop=bakery": ["Crustworks", "Loafery", "Oven Hall"],
    "amenity=dentist": ["Molar Studio", "Enamel Rooms", "Orthodontics"],
    "shop=books": ["Folio House", "Quire Rooms", "Chapterhouse"],
    "leisure=fitness_centre": ["Kettlebell Hall", "Strength Loft", "Athletica"],
    "shop=hairdresser": ["Shearling Studio", "Tress Rooms", "Coiffure"],
    "tourism=hotel": ["Lodgings", "Guest Rooms", "Hostelry"],
    "shop=furniture": ["Joinery Rooms", "Cabinetworks", "Settle House"],
}
# Borrowed pairs: the noun's category, and the unrelated category it lands on.
BORROWED_PAIRS = [
    ("amenity=cafe", "amenity=dentist"),
    ("shop=books", "leisure=fitness_centre"),
    ("tourism=hotel", "shop=hairdresser"),
]
BELONGING = ["shop=bakery", "shop=furniture", "amenity=cafe"]
ONSETS = ["Br", "Qu", "Th", "Vel", "Mor", "Sk", "Dr", "Ess", "Gl", "Wr"]
NUCLEI = ["an", "oth", "ey", "ilm", "ar", "ow", "uth", "enn"]
CODAS = ["dale", "moor", "wick", "stead", "holt", "mere", "combe", "garth"]
STREETS = ["Row", "Lane", "Walk", "Yard", "Close"]
HOURS = ["Mo-Fr 08:10-17:40", "Tu-Sa 09:20-18:50", "We-Su 10:35-19:15"]


def word(generator: random.Random) -> str:
    """Return one invented proper word."""
    return generator.choice(ONSETS) + generator.choice(NUCLEI) + generator.choice(CODAS)


def noun_for(category: str, tokens: dict[str, str]) -> str:
    """Return the first descriptive noun of a category the token dictionary lacks.

    Raises:
        ValueError: If every candidate carries a dictionary token.
    """
    for candidate in NOUNS[category]:
        if not any(token in tokens for token in fold(candidate).split()):
            return candidate
    raise ValueError(f"every noun for {category} carries a category token")


def exemplar(
    identifier: str,
    category: str,
    noun_category: str,
    generator: random.Random,
    tokens: dict[str, str],
) -> dict[str, Any]:
    """Compose one exemplar record with its label fixed by construction."""
    settlement = word(generator)
    return {
        "id": identifier,
        "source": "generated",
        "label": "clean" if category == noun_category else "wrong",
        "field": None if category == noun_category else "name",
        "reason": (
            "The name describes the kind of place the record is."
            if category == noun_category
            else "The name describes another kind of place than the category."
        ),
        "record": {
            "name": f"{word(generator)} {noun_for(noun_category, tokens)}",
            "category": category,
            "addr:city": settlement,
            "addr:street": f"{word(generator)} {generator.choice(STREETS)}",
            "addr:housenumber": str(generator.randint(2, 98)),
            "opening_hours": generator.choice(HOURS),
            "lat": round(generator.uniform(-49.5, -44.5), 5),
            "lon": round(generator.uniform(-139.5, -121.5), 5),
        },
    }


def build(seed: int, tokens: dict[str, str]) -> list[dict[str, Any]]:
    """Return the six exemplars in a fixed, shuffled order."""
    generator = random.Random(seed)
    entries = [
        exemplar("", landing, noun, generator, tokens)
        for noun, landing in BORROWED_PAIRS
    ] + [exemplar("", category, category, generator, tokens) for category in BELONGING]
    generator.shuffle(entries)
    for position, entry in enumerate(entries, start=1):
        entry["id"] = f"G{position}"
    return entries


def main() -> None:
    """Write the generated exemplars."""
    settings = get("few_shot")
    entries = build(int(settings["seed"]), screen.load_category_tokens())
    if sum(1 for e in entries if e["label"] == "wrong") != int(settings["borrowed"]):
        raise ValueError("the borrowed count does not match the configuration")
    payload = {
        "version": 1,
        "purpose": "Generated demonstration records for the few-shot prompt.",
        "note": (
            "Composed by scripts/build_exemplars.py from fixed lists and a fixed seed. "
            "Every value is invented and every label follows from the composition; "
            "no model wrote or labelled any of them."
        ),
        "exemplars": entries,
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")
    logger.info(f"wrote {len(entries)} exemplars to {TARGET}")


if __name__ == "__main__":
    main()
