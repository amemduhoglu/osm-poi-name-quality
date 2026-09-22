"""The rule baseline: four format rules and two reference lookups.

This is the trivial answer a model has to beat, and it is written before any
model runs. Each check returns a verdict for one record and one field, and each
may abstain. An abstention is not a pass. A lookup that cannot find a reference
for a record has not cleared that record, and counting it as clean would let
missing coverage read as accuracy.

Two of the classes are settled by a rule that shares its definition with the
corruption that produced them: the out-of-range class is injected outside the
same declared range the rule tests against, and the wrong-calling-code class is
injected by replacing the code the lookup expects. Recall on those classes is
therefore near one by construction and is reported as such, not as a discovery.
The quantity that is not settled by construction, and the one the article reads,
is how often each rule fires on the nine hundred clean records: those carry real
values written by mappers, and nothing about them is arranged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import phonenumbers

from poi_audit.config import get
from poi_audit.corpus import fold, item_value
from poi_audit.references import Gazetteer, expected_calling_code

PASS = "pass"
FLAG = "flag"
ABSTAIN = "abstain"

_PHONE_ALLOWED = re.compile(r"^[+0-9 ()\-./]+$")
_DIGITS = re.compile(r"\d")
_WEBSITE = re.compile(
    r"^(?P<scheme>[a-zA-Z][a-zA-Z0-9+.\-]*)://(?P<host>[^/\s?#]+)(?P<rest>[^\s]*)$"
)
_HOST = re.compile(r"^[A-Za-z0-9\-._~%]+(\.[A-Za-z0-9\-._~%]+)+(:\d+)?$", re.UNICODE)

# A restrained reading of the opening-hours syntax: a semicolon-separated list
# of rules, each an optional selector followed by time spans, "off" or "closed".
# It is deliberately not the whole specification, which no mapper writes in full
# and no practitioner implements in full either. Whatever it rejects among real
# values is counted as a false positive and reported, which is the honest way to
# state the cost of a format rule.
_TIME = re.compile(r"^(?P<h>\d{1,2}):(?P<m>\d{2})$")
_SELECTOR = re.compile(
    r"^(Mo|Tu|We|Th|Fr|Sa|Su|PH|SH|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
    r"|week|\d{4}|\[[-\d,]+\]|[-,:0-9 ])+$"
)
_MAX_HOUR = 48


@dataclass(frozen=True)
class Verdict:
    """One check's answer about one field of one record.

    Attributes:
        check: The check's name.
        field: The field the verdict is about, or None where it is about the
            record as a whole.
        outcome: One of ``pass``, ``flag`` or ``abstain``.
        detail: A short English explanation, written for the released output.
    """

    check: str
    field: str | None
    outcome: str
    detail: str


def phone_syntax(value: str) -> Verdict:
    """Test whether a phone number is well formed.

    The rule is about syntax alone: which characters appear and how many digits
    they hold. Whether the number belongs to the city is the lookup's question,
    not this one's.

    Args:
        value: The phone value as written.

    Returns:
        The verdict for the phone field.
    """
    text = value.strip()
    bounds = get("rule_baseline.phone_syntax")
    if not text:
        return Verdict("phone_syntax", "phone", FLAG, "empty value")
    if not _PHONE_ALLOWED.match(text):
        return Verdict(
            "phone_syntax", "phone", FLAG, "character outside the phone alphabet"
        )
    if text.count("+") > 1 or ("+" in text and not text.startswith("+")):
        return Verdict("phone_syntax", "phone", FLAG, "misplaced or repeated plus sign")
    digits = len(_DIGITS.findall(text))
    if digits < int(bounds["min_digits"]):
        return Verdict("phone_syntax", "phone", FLAG, f"only {digits} digits")
    if digits > int(bounds["max_digits"]):
        return Verdict("phone_syntax", "phone", FLAG, f"{digits} digits")
    return Verdict("phone_syntax", "phone", PASS, "well formed")


def _valid_timespan(span: str) -> bool:
    """Test one time span of an opening-hours rule.

    Args:
        span: A span such as ``09:00-17:30``, or ``off``.

    Returns:
        True when the span is well formed.
    """
    text = span.strip()
    if text.lower() in {"off", "closed", "open", "unknown"}:
        return True
    parts = text.split("-")
    if len(parts) != 2:
        return False
    for part in parts:
        match = _TIME.match(part.strip())
        if not match:
            return False
        hour, minute = int(match.group("h")), int(match.group("m"))
        if hour > _MAX_HOUR or minute > 59:
            return False
    return True


def opening_hours_syntax(value: str) -> Verdict:
    """Test whether an opening-hours value is well formed.

    Args:
        value: The opening-hours value as written.

    Returns:
        The verdict for the opening-hours field.
    """
    text = value.strip()
    if not text:
        return Verdict("opening_hours_syntax", "opening_hours", FLAG, "empty value")
    if text in {"24/7", "24/7 open"}:
        return Verdict("opening_hours_syntax", "opening_hours", PASS, "well formed")
    for rule in text.split(";"):
        body = rule.strip().lstrip("|").strip()
        if not body:
            return Verdict("opening_hours_syntax", "opening_hours", FLAG, "empty rule")
        if body == "24/7":
            continue
        head, _, tail = body.partition(" ")
        selector, spans = (head, tail) if tail else ("", body)
        if selector and not _SELECTOR.match(selector):
            return Verdict(
                "opening_hours_syntax",
                "opening_hours",
                FLAG,
                f"unreadable selector: {selector}",
            )
        for span in spans.split(","):
            if not _valid_timespan(span):
                return Verdict(
                    "opening_hours_syntax",
                    "opening_hours",
                    FLAG,
                    f"unreadable time span: {span.strip()}",
                )
    return Verdict("opening_hours_syntax", "opening_hours", PASS, "well formed")


def website_syntax(value: str) -> Verdict:
    """Test whether a web address is well formed.

    Args:
        value: The website value as written.

    Returns:
        The verdict for the website field.
    """
    text = value.strip()
    if not text:
        return Verdict("website_syntax", "website", FLAG, "empty value")
    if " " in text:
        return Verdict(
            "website_syntax", "website", FLAG, "whitespace inside the address"
        )
    match = _WEBSITE.match(text)
    if not match:
        return Verdict("website_syntax", "website", FLAG, "no scheme and host")
    schemes = [str(s).lower() for s in get("rule_baseline.website_syntax.schemes")]
    if match.group("scheme").lower() not in schemes:
        return Verdict(
            "website_syntax", "website", FLAG, f"scheme {match.group('scheme')}"
        )
    if "://" in match.group("rest"):
        return Verdict("website_syntax", "website", FLAG, "two schemes")
    if not _HOST.match(match.group("host")):
        return Verdict("website_syntax", "website", FLAG, "host is not a dotted name")
    return Verdict("website_syntax", "website", PASS, "well formed")


def value_range(tags: dict[str, Any]) -> list[Verdict]:
    """Test every numeric tag against the range this study declares for it.

    The ranges are the study's own and are declared in the configuration, which
    also documents where each bound comes from. They are not an external
    standard: OpenStreetMap documents these keys without publishing a numeric
    range for any of them.

    Args:
        tags: The record's tag set.

    Returns:
        One verdict per range-checkable tag the record carries, and an empty
        list when it carries none, which is an abstention by absence.
    """
    ranges: dict[str, list[float]] = get("references.value_ranges")
    verdicts: list[Verdict] = []
    for key, (low, high) in ranges.items():
        if key not in tags:
            continue
        raw = str(tags[key]).strip()
        try:
            number = float(raw)
        except ValueError:
            verdicts.append(Verdict("value_range", key, FLAG, f"not a number: {raw}"))
            continue
        if number < float(low) or number > float(high):
            verdicts.append(
                Verdict("value_range", key, FLAG, f"{raw} outside [{low}, {high}]")
            )
        else:
            verdicts.append(
                Verdict("value_range", key, PASS, "inside the declared range")
            )
    return verdicts


def phone_country_code(value: str, country: str) -> Verdict:
    """Test a phone number's country code against the city's country.

    The comparison is on calling codes rather than on regions, because several
    countries share one code and a number written under a shared code is not
    wrong for the city.

    Args:
        value: The phone value as written.
        country: Two-letter code of the country the record sits in.

    Returns:
        The verdict, abstaining when the value carries no international prefix
        the reference can read.
    """
    text = value.strip()
    if not text.startswith("+"):
        return Verdict(
            "phone_country_code", "phone", ABSTAIN, "no international prefix to read"
        )
    try:
        parsed = phonenumbers.parse(text, None)
    except phonenumbers.NumberParseException:
        return Verdict("phone_country_code", "phone", ABSTAIN, "prefix does not parse")
    expected = expected_calling_code(country)
    if int(parsed.country_code) != expected:
        return Verdict(
            "phone_country_code",
            "phone",
            FLAG,
            f"calling code +{parsed.country_code} in {country}, which is +{expected}",
        )
    return Verdict("phone_country_code", "phone", PASS, f"calling code +{expected}")


def address_city(
    value: str,
    lat: float,
    lon: float,
    gazetteer: Gazetteer,
    source: str | None = None,
    radius_km: float | None = None,
) -> Verdict:
    """Test an address city against the places near the record's coordinate.

    Args:
        value: The ``addr:city`` value as written.
        lat: Record latitude in degrees.
        lon: Record longitude in degrees.
        gazetteer: The loaded gazetteer.
        source: Restrict to one authority, or None to use every one of them.
        radius_km: The matching radius, or None to read it from the
            configuration.

    Returns:
        The verdict, abstaining where the authority holds no place within the
        radius, because it has then said nothing about the record.
    """
    radius = radius_km or float(get("references.address_city.match_radius_km"))
    names = gazetteer.names_near(lat, lon, radius, source)
    if not names:
        return Verdict(
            "address_city", "addr:city", ABSTAIN, f"no place within {radius:g} km"
        )
    if fold(value) in names:
        return Verdict("address_city", "addr:city", PASS, "names a place nearby")
    return Verdict(
        "address_city",
        "addr:city",
        FLAG,
        f"no place within {radius:g} km is named {value}",
    )


def category_token(name: str, category: str, tokens: dict[str, str]) -> Verdict:
    """Test whether a name's category-indicative token contradicts its category.

    This is not part of the baseline. It is scored against the finished item set
    so that the exclusion applied when the residual class was built is reported
    as a measurement: if the rule settled the residual class, the boundary this
    article draws would sit somewhere else.

    Args:
        name: The record's name.
        category: The record's primary category as ``key=value``.
        tokens: Category-indicative token to the category it indicates, derived
            from the corpus.

    Returns:
        The verdict, abstaining when the name carries no indicative token.
    """
    for token in fold(name).split():
        indicated = tokens.get(token)
        if not indicated:
            continue
        if indicated != category:
            return Verdict(
                "category_token",
                "name",
                FLAG,
                f"the name says {indicated} and the record says {category}",
            )
        return Verdict(
            "category_token", "name", PASS, "the name agrees with the category"
        )
    return Verdict("category_token", "name", ABSTAIN, "no category-indicative token")


def format_rules(tags: dict[str, Any]) -> list[Verdict]:
    """Run every format rule over one record.

    Args:
        tags: The record's tag set.

    Returns:
        Every verdict the format rules produced. A field the record does not
        carry produces no verdict.
    """
    verdicts: list[Verdict] = []
    phone = item_value(tags, "phone")
    if phone is not None:
        verdicts.append(phone_syntax(phone))
    hours = item_value(tags, "opening_hours")
    if hours is not None:
        verdicts.append(opening_hours_syntax(hours))
    website = item_value(tags, "website")
    if website is not None:
        verdicts.append(website_syntax(website))
    verdicts.extend(value_range(tags))
    return verdicts


def reference_lookups(
    tags: dict[str, Any],
    country: str,
    lat: float,
    lon: float,
    gazetteer: Gazetteer,
) -> list[Verdict]:
    """Run both reference lookups over one record.

    Args:
        tags: The record's tag set.
        country: Two-letter code of the country the record sits in.
        lat: Record latitude in degrees.
        lon: Record longitude in degrees.
        gazetteer: The loaded gazetteer.

    Returns:
        Every verdict the lookups produced, abstentions included.
    """
    verdicts: list[Verdict] = []
    phone = item_value(tags, "phone")
    if phone is not None:
        verdicts.append(phone_country_code(phone, country))
    city = item_value(tags, "addr:city")
    if city is not None:
        verdicts.append(address_city(city, lat, lon, gazetteer))
    return verdicts


def baseline(record: dict[str, Any], gazetteer: Gazetteer) -> list[Verdict]:
    """Run the whole rule baseline over one record.

    Args:
        record: A corpus or item record carrying ``tags``, ``country``, ``lat``
            and ``lon``.
        gazetteer: The loaded gazetteer.

    Returns:
        Every verdict, in a fixed order: the format rules, then the lookups.
    """
    tags = record["tags"]
    return [
        *format_rules(tags),
        *reference_lookups(
            tags,
            str(record["country"]),
            float(record["lat"]),
            float(record["lon"]),
            gazetteer,
        ),
    ]


def flagged(verdicts: list[Verdict]) -> bool:
    """Test whether any verdict flagged the record.

    Args:
        verdicts: The verdicts for one record.

    Returns:
        True when at least one check flagged.
    """
    return any(verdict.outcome == FLAG for verdict in verdicts)
