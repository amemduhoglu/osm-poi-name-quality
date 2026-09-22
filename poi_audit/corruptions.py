"""The corruptions that build the injected item set.

Every function here takes a real value from a real record and returns a wrong
one, together with the kind of wrongness it applied. Three rules govern all of
them, and each repairs a known failure mode of this kind of item set:

    The value is drawn at random per item. No class corrupts its field with one
    fixed value, so a model cannot learn the class from a constant.

    No corruption value restates a corpus factor. A name comes from the record's
    own city and a place name from the record's own country, so that language,
    script and local convention are held fixed and the only thing wrong with the
    item is the thing the class is about.

    Nothing depends on what a record ought to have carried. Every corruption
    changes a value the record already has. Whether a record should have held an
    attribute is a property of the edit rather than of the record shown, and
    could not be scored against the record alone.

The randomness is the module-level ``random`` sequence, seeded once by the
caller from the configured seed, so that the whole set rebuilds identically.
"""

from __future__ import annotations

import random
import re
import string
from dataclasses import dataclass
from typing import Any

import phonenumbers

from poi_audit.config import get

_DIGIT = re.compile(r"\d")


@dataclass(frozen=True)
class Corruption:
    """One applied corruption.

    Attributes:
        field: The field whose value was replaced.
        original: The value the record carried.
        injected: The value the item shows.
        kind: The kind of corruption drawn for this item.
        donor: Where the injected value came from, for a class that draws one,
            else None.
    """

    field: str
    original: str
    injected: str
    kind: str
    donor: dict[str, Any] | None = None


def corrupt_name(donor: dict[str, Any], original: str) -> Corruption:
    """Replace a name with the name of another place in the same city.

    Args:
        donor: The donor record, chosen by the caller under the eligibility
            rules in the configuration.
        original: The name the record carried.

    Returns:
        The corruption.
    """
    return Corruption(
        field="name",
        original=original,
        injected=str(donor["name"]),
        kind="name_of_another_place",
        donor={
            "city": donor["city"],
            "osm_type": donor["osm_type"],
            "osm_id": donor["osm_id"],
            "category": donor["category"],
            "distance_km": round(float(donor["distance_km"]), 3),
        },
    )


def corrupt_calling_code(original: str, codes: list[int]) -> Corruption:
    """Replace a phone number's calling code with another country's.

    The national part of the number and every separator are left exactly as the
    mapper wrote them, so that the calling code is the only thing that changed.

    Args:
        original: The phone value, which must begin with an international
            prefix.
        codes: The calling codes to draw from, the record's own excluded.

    Returns:
        The corruption.

    Raises:
        ValueError: If the value carries no readable international prefix.
    """
    text = original.strip()
    parsed = phonenumbers.parse(text, None)
    old = str(parsed.country_code)
    prefix = text.index(old, text.index("+"))
    new = str(random.choice(codes))
    injected = text[:prefix] + new + text[prefix + len(old) :]
    return Corruption(
        field="phone",
        original=original,
        injected=injected,
        kind=f"calling_code_{new}",
    )


def corrupt_address_city(original: str, donor: dict[str, Any]) -> Corruption:
    """Replace an address city with a distant place in the same country.

    Args:
        original: The ``addr:city`` value the record carried.
        donor: The donor place, with its name and its distance from the record.

    Returns:
        The corruption.
    """
    return Corruption(
        field="addr:city",
        original=original,
        injected=str(donor["name"]),
        kind="distant_place_same_country",
        donor={
            "source": donor["source"],
            "place_id": donor["place_id"],
            "distance_km": round(float(donor["distance_km"]), 1),
        },
    )


def _insert_at_random(text: str, fragment: str) -> str:
    """Insert a fragment at a random position inside a value.

    Args:
        text: The value.
        fragment: What to insert.

    Returns:
        The value with the fragment inserted.
    """
    position = random.randrange(1, max(len(text), 2))
    return text[:position] + fragment + text[position:]


def corrupt_phone_syntax(original: str, kind: str) -> Corruption:
    """Break the syntax of a phone number.

    Args:
        original: The phone value as written.
        kind: One of the configured malformations.

    Returns:
        The corruption.

    Raises:
        ValueError: If the kind is not one this function implements.
    """
    text = original.strip()
    if kind == "letters_inserted":
        injected = _insert_at_random(text, random.choice(string.ascii_lowercase) * 2)
    elif kind == "digits_removed":
        keep = random.randint(2, 4)
        seen = 0
        kept: list[str] = []
        for char in text:
            if char.isdigit():
                seen += 1
                if seen > keep:
                    continue
            kept.append(char)
        injected = "".join(kept)
    elif kind == "separator_garbage":
        injected = _insert_at_random(
            text, random.choice(["**", "##", "!!", "??", "%%"])
        )
    elif kind == "doubled_plus":
        injected = "+" + text if text.startswith("+") else "++" + text
    elif kind == "plus_misplaced":
        stripped = text.lstrip("+")
        injected = _insert_at_random(stripped, "+")
    else:
        raise ValueError(f"unknown phone malformation: {kind}")
    return Corruption("phone", original, injected, kind)


def corrupt_opening_hours(original: str, kind: str) -> Corruption:
    """Break the syntax of an opening-hours value.

    Args:
        original: The opening-hours value as written.
        kind: One of the configured malformations.

    Returns:
        The corruption.

    Raises:
        ValueError: If the kind is not one this function implements.
    """
    text = original.strip()
    if kind == "unknown_day_token":
        token = "".join(random.choice(string.ascii_uppercase) for _ in range(2))
        injected = f"{token} " + text
    elif kind == "hour_out_of_clock":
        hour = random.randint(49, 99)
        minute = random.choice(["00", "15", "30", "45"])
        injected = re.sub(r"\d{1,2}:\d{2}", f"{hour}:{minute}", text, count=1)
        if injected == text:
            injected = f"Mo-Fr {hour}:{minute}-{hour + 1}:00"
    elif kind == "missing_separator":
        # The separator that goes is the one inside a time span. Removing the
        # first hyphen in the value would take the one out of a day range, and
        # "MoFr 09:00-17:00" is still readable enough that the rule lets it by.
        injected, replaced = re.subn(
            r"(\d{1,2}:\d{2})-(\d{1,2}:\d{2})", r"\1\2", text, count=1
        )
        if not replaced:
            injected = text + " 0900 1700"
    elif kind == "unclosed_range":
        hour = random.randint(6, 20)
        injected = f"Mo-Fr {hour:02d}:00-"
    elif kind == "prose_instead_of_syntax":
        injected = random.choice(
            [
                "opens in the morning and closes late",
                "please call ahead for the hours",
                "closed whenever the owner travels",
                "hours vary with the season",
                "open most days until evening",
            ]
        )
    else:
        raise ValueError(f"unknown opening-hours malformation: {kind}")
    return Corruption("opening_hours", original, injected, kind)


def corrupt_website(original: str, kind: str) -> Corruption:
    """Break the syntax of a web address.

    Args:
        original: The website value as written.
        kind: One of the configured malformations.

    Returns:
        The corruption.

    Raises:
        ValueError: If the kind is not one this function implements.
    """
    text = original.strip()
    body = text.split("://", 1)[-1]
    if kind == "scheme_removed_and_broken":
        injected = random.choice(["htp:/", "ttps:/", "http:/", "://"]) + body
    elif kind == "space_inserted":
        injected = _insert_at_random(text, " ")
    elif kind == "tld_removed":
        host = body.split("/", 1)[0]
        # Every dot goes, not only the last one: a host that keeps a dot after
        # losing its top-level domain still reads as a dotted name and the rule
        # would let it through.
        stripped = re.sub(r"\.[A-Za-z]{2,}$", "", host).replace(".", "")
        injected = text.replace(host, stripped or "host", 1)
    elif kind == "double_scheme":
        injected = random.choice(["https://", "http://"]) + text
    elif kind == "unbalanced_brackets":
        injected = _insert_at_random(text, random.choice(["[", "]", "{", "}"]))
    else:
        raise ValueError(f"unknown website malformation: {kind}")
    return Corruption("website", original, injected, kind)


def corrupt_value_range(tag: str, original: str, kind: str) -> Corruption | None:
    """Move a numeric value outside the range the rule publishes.

    The range comes from the same configured table the rule reads, so that the
    published bound has one home. That the rule then settles this class is a
    property of the construction and is reported as such.

    Args:
        tag: The numeric tag being corrupted.
        original: The value the record carried.
        kind: One of the configured kinds.

    Returns:
        The corruption, or None when the drawn value equals the one the record
        already carried. A handful of records hold a value that is already
        outside its range, and writing the same value back would produce an
        item whose corrupted member is identical to the record it came from.

    Raises:
        ValueError: If the kind is not one this function implements, or if the
            drawn value did not land outside the range.
    """
    low, high = (float(bound) for bound in get("references.value_ranges")[tag])
    if kind == "negative":
        value = (
            -abs(low) - random.randint(1, 500) if low <= 0 else -random.randint(1, 500)
        )
    elif kind == "far_above_maximum":
        value = high * random.randint(3, 40) + random.randint(1, 99)
    elif kind == "zero_where_positive":
        value = 0.0 if low > 0 else low - random.randint(1, 50)
    else:
        raise ValueError(f"unknown range corruption: {kind}")
    if low <= value <= high:
        raise ValueError(f"corruption stayed inside the range for {tag}: {value}")
    injected = str(int(value)) if float(value).is_integer() else f"{value:g}"
    if injected == original.strip():
        return None
    return Corruption(tag, original, injected, kind)
