"""The string and dictionary screen that enriches stratum B of the natural set.

The injected set gives exact ground truth and scale; it cannot show that the
residual class occurs in real data. The natural set does, and half of it is
drawn from records this screen raises, so that 200 labelling decisions are not
spent on a class whose base rate may be well under one per cent.

Three rules, any one of which raises a record. All three read strings and
dictionaries only. **No model and no evaluated system takes part**, because a
screen run by a model would mean evaluating models on records a model selected.

The screen is not the rule baseline and is never scored as one. Its job is to
raise the share of wrong names in stratum B, and how well it does that is
measured by the labels rather than asserted here. Its verdicts carry evidence so
that a reader can see why each record was raised, and the inclusion probability
it implies is written into every drawn record so that estimates reweight to the
corpus.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from poi_audit import rules
from poi_audit.config import get
from poi_audit.corpus import distance_km, fold, primary_category
from poi_audit.references import Gazetteer, Place

DUPLICATE_DISTANT_NAME = "duplicate_distant_name"
CATEGORY_TOKEN_CONTRADICTION = "category_token_contradiction"
GAZETTEER_PLACE_NAME = "gazetteer_place_name"

RULE_ORDER = [
    DUPLICATE_DISTANT_NAME,
    CATEGORY_TOKEN_CONTRADICTION,
    GAZETTEER_PLACE_NAME,
]

FINAL = "final"
FIRST_PASS = "first_pass"


@dataclass(frozen=True)
class Hit:
    """One screen rule raising one record.

    Attributes:
        rule: The rule's name.
        evidence: A short English statement of what the rule saw, written for
            the released output and for the labelling protocol.
    """

    rule: str
    evidence: str


def rule_settings(variant: str = FINAL) -> dict[str, dict[str, Any]]:
    """Return one screen variant's rule settings.

    Two variants are configured. The final screen is the one stratum B is drawn
    from; the first-pass screen is the one first written, kept and run so that
    what the tightening changed is reported as a number rather than described.

    Args:
        variant: ``final`` or ``first_pass``.

    Returns:
        Rule name to its settings.

    Raises:
        ValueError: If the variant is not one of the two configured.
    """
    if variant == FINAL:
        return dict(get("natural_set.screen.rules"))
    if variant == FIRST_PASS:
        return dict(get("natural_set.screen.first_pass_rules"))
    raise ValueError(f"unknown screen variant: {variant}")


def category_key(record: dict[str, Any]) -> str:
    """Return the top-level point-of-interest key a record is categorised under.

    Args:
        record: A corpus record.

    Returns:
        The key alone (``amenity`` rather than ``amenity=cafe``), or an empty
        string when the record carries no configured category key.
    """
    category = primary_category(record["tags"])
    return category.split("=")[0] if category else ""


def carries_brand_tag(record: dict[str, Any]) -> bool:
    """Test whether a record states that its name is a chain's.

    Args:
        record: A corpus record.

    Returns:
        True when the record carries ``brand`` or any ``brand:`` tag.
    """
    return any(key == "brand" or key.startswith("brand:") for key in record["tags"])


def identity(record: dict[str, Any]) -> tuple[str, int]:
    """Return the OpenStreetMap identity of a record.

    Args:
        record: A corpus record.

    Returns:
        The element type and identifier, which together are unique in the
        corpus and are what every phase uses to say that two records are the
        same one.
    """
    return str(record["osm_type"]), int(record["osm_id"])


def load_category_tokens(source_file: Path | None = None) -> dict[str, str]:
    """Read the category-token dictionary derived from the corpus.

    Args:
        source_file: An alternative file, used by the tests. The configured
            location is read when this is None.

    Returns:
        Each indicative token mapped to the category it indicates.

    Raises:
        FileNotFoundError: If the dictionary has not been written.
    """
    configured = get(
        "natural_set.screen.rules.category_token_contradiction.token_dictionary"
    )
    target = source_file or Path(configured)
    if not target.exists():
        raise FileNotFoundError(
            f"category tokens not found: {target}; run scripts/build_injected_set.py"
        )
    with target.open(encoding="utf-8", newline="") as handle:
        return {row["token"]: row["category"] for row in csv.DictReader(handle)}


def name_index(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group one city's records by their folded name.

    Args:
        records: One city's corpus records.

    Returns:
        Folded name to the records carrying it. A name held by one record only
        cannot raise the duplicate rule, and grouping first keeps the rule from
        comparing every record against every other one.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        folded = fold(str(record.get("name") or ""))
        if folded:
            grouped[folded].append(record)
    return dict(grouped)


def place_index(gazetteer: Gazetteer) -> dict[str, list[Place]]:
    """Index the gazetteer by folded place name.

    Args:
        gazetteer: The loaded gazetteer.

    Returns:
        Folded name, including alternate names, to the places holding it.
    """
    grouped: dict[str, list[Place]] = defaultdict(list)
    for place in gazetteer.places:
        for name in place.names:
            grouped[name].append(place)
    return dict(grouped)


def duplicate_distant_name(
    record: dict[str, Any],
    same_name: list[dict[str, Any]],
    settings: dict[str, Any] | None = None,
) -> Hit | None:
    """Test whether the same name sits on a distant record of another category.

    A name repeated across a city is ordinary: a chain writes its own name on
    every branch, and a generic word is written by everyone. What the residual
    class looks like is a name that belongs to some other place, so the rule
    asks for a name the city barely repeats, carried by a record far enough away
    to be a different establishment and categorised under a different top-level
    key.

    Args:
        record: The record under test.
        same_name: Every record in the same city carrying the same folded name,
            including this one.
        settings: The rule's settings, or None to read the final screen's.

    Returns:
        The hit, or None when no other record qualifies.
    """
    if settings is None:
        settings = rule_settings()[DUPLICATE_DISTANT_NAME]
    min_km = float(settings["min_distance_km"])
    different_category = bool(settings["require_different_category"])
    different_key = bool(settings.get("require_different_category_key"))
    most_sharing = settings.get("max_records_sharing_name")
    if most_sharing is not None and len(same_name) > int(most_sharing):
        return None
    if settings.get("exclude_brand_tagged") and carries_brand_tag(record):
        return None
    category = primary_category(record["tags"])
    key = category_key(record)
    for other in same_name:
        if identity(other) == identity(record):
            continue
        if different_category and primary_category(other["tags"]) == category:
            continue
        if different_key and category_key(other) == key:
            continue
        away = distance_km(
            float(record["lat"]),
            float(record["lon"]),
            float(other["lat"]),
            float(other["lon"]),
        )
        if away >= min_km:
            return Hit(
                DUPLICATE_DISTANT_NAME,
                f"the same name is on {other['osm_type']} {other['osm_id']} of "
                f"category {primary_category(other['tags'])}, {away:.1f} km away",
            )
    return None


def indicated_category(name: str, tokens: dict[str, str]) -> str | None:
    """Return the category a name's first indicative token points at.

    The first token decides, which is the order the baseline's own check reads
    them in, so the screen and the check never disagree about which token spoke.

    Args:
        name: The record's name.
        tokens: Category-indicative token to the category it indicates.

    Returns:
        The indicated category as ``key=value``, or None when the name carries
        no indicative token.
    """
    for token in fold(name).split():
        indicated = tokens.get(token)
        if indicated:
            return indicated
    return None


def category_token_contradiction(
    record: dict[str, Any],
    tokens: dict[str, str],
    settings: dict[str, Any] | None = None,
) -> Hit | None:
    """Test whether the name's category token contradicts the record's category.

    The rule itself is the one already written for the reported check, so the
    screen and the baseline read one definition of an indicative token. What the
    screen adds is a restriction to contradictions across top-level keys, which
    separates a name that belongs to another kind of place from a sibling-value
    nuance such as a guest house tagged as a hotel.

    Args:
        record: The record under test.
        tokens: Category-indicative token to the category it indicates.
        settings: The rule's settings, or None to read the final screen's.

    Returns:
        The hit, or None when the name carries no contradicting token.
    """
    if settings is None:
        settings = rule_settings()[CATEGORY_TOKEN_CONTRADICTION]
    category = primary_category(record["tags"])
    if not category:
        return None
    verdict = rules.category_token(str(record.get("name") or ""), category, tokens)
    if verdict.outcome != rules.FLAG:
        return None
    indicated = indicated_category(str(record.get("name") or ""), tokens)
    if (
        settings.get("require_different_category_key")
        and indicated
        and indicated.split("=")[0] == category.split("=")[0]
    ):
        return None
    return Hit(CATEGORY_TOKEN_CONTRADICTION, verdict.detail)


def gazetteer_place_name(
    record: dict[str, Any],
    places: dict[str, list[Place]],
    settings: dict[str, Any] | None = None,
) -> Hit | None:
    """Test whether the name is a place name belonging somewhere else.

    A record named after a place that sits near it is unremarkable: a café in
    Brno may be named for its own district. The rule therefore asks for a name
    an open gazetteer holds as a place while holding no place of that name
    within the radius the address lookup matches on.

    Args:
        record: The record under test.
        places: Folded place name to the places holding it.
        settings: The rule's settings, or None to read the final screen's.

    Returns:
        The hit, or None when the name is not a place name, or is one that
        belongs nearby.
    """
    match = place_match(record, places, settings)
    if match is None:
        return None
    away, place = match
    return Hit(
        GAZETTEER_PLACE_NAME,
        f"{place.source} holds {place.display_name} under this name, "
        f"nearest {away:.0f} km away",
    )


def place_match(
    record: dict[str, Any],
    places: dict[str, list[Place]],
    settings: dict[str, Any] | None = None,
) -> tuple[float, Place] | None:
    """Return the nearest gazetteer place the record's name belongs to.

    The rule is written here and read twice: once to raise a record, and once to
    write the match into a drawn record, so that the article can report the
    screen's precision split by whether the matched place sits in the record's
    own country without drawing a second sample.

    Args:
        record: The record under test.
        places: Folded place name to the places holding it.
        settings: The rule's settings, or None to read the final screen's.

    Returns:
        The distance in kilometres and the place, or None when the rule does
        not raise the record.
    """
    if settings is None:
        settings = rule_settings()[GAZETTEER_PLACE_NAME]
    if settings.get("exclude_brand_tagged") and carries_brand_tag(record):
        return None
    radius_km = float(settings["absent_within_km"])
    min_chars = int(settings["min_name_chars"])
    folded = fold(str(record.get("name") or ""))
    if len(folded) < min_chars:
        return None
    holders = places.get(folded)
    if not holders:
        return None
    away, place = min(
        (
            (
                distance_km(
                    float(record["lat"]), float(record["lon"]), place.lat, place.lon
                ),
                place,
            )
            for place in holders
        ),
        key=lambda pair: pair[0],
    )
    if away <= radius_km:
        return None
    return away, place


def screen_record(
    record: dict[str, Any],
    same_name: list[dict[str, Any]],
    tokens: dict[str, str],
    places: dict[str, list[Place]],
    settings: dict[str, dict[str, Any]] | None = None,
) -> list[Hit]:
    """Run every screen rule over one record.

    Args:
        record: The record under test.
        same_name: Every record in the same city carrying the same folded name.
        tokens: Category-indicative token to the category it indicates.
        places: Folded place name to the places holding it.
        settings: One variant's rule settings, or None for the final screen's.

    Returns:
        The hits, in the configured rule order. A record with at least one hit
        is screen-positive.
    """
    if settings is None:
        settings = rule_settings()
    found = [
        duplicate_distant_name(record, same_name, settings[DUPLICATE_DISTANT_NAME]),
        category_token_contradiction(
            record, tokens, settings[CATEGORY_TOKEN_CONTRADICTION]
        ),
        gazetteer_place_name(record, places, settings[GAZETTEER_PLACE_NAME]),
    ]
    return [hit for hit in found if hit is not None]


def owning_rule(hits: list[Hit]) -> str:
    """Return the rule a record raised by several is counted under.

    Stratum B is allocated equally across the rules, so a record raised twice
    has to belong to exactly one cell. The configured priority runs from the
    most place-name-shaped rule to the least, so a record the gazetteer rule
    raised is counted there whatever else also raised it.

    Args:
        hits: The record's hits, which must not be empty.

    Returns:
        The owning rule's name.

    Raises:
        ValueError: If the record was raised by nothing.
    """
    raised = {hit.rule for hit in hits}
    if not raised:
        raise ValueError("a record raised by no rule has no owning rule")
    for rule in get("natural_set.screen.rule_priority"):
        if rule in raised:
            return str(rule)
    return sorted(raised)[0]
