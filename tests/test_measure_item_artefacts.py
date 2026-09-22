"""Tests for the residual class's construction asymmetries.

The corrupted half of the residual class carries a name five filters chose and
the clean half carries whatever the record already had, since the pairing
matches on the city and the field signature and says nothing about the name.
Anything that separates the two halves without reading a map is a way to score
the class without answering it, and this measures how far that goes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from scripts.measure_item_artefacts import features, score_flags

TOKENS = {"bakery": "shop=bakery", "pharmacy": "amenity=pharmacy"}


def record_shortcut(min_chars: int) -> Callable[[dict[str, Any]], bool]:
    """Return the rule reading what one record shows.

    Args:
        min_chars: The donor's minimum name length.

    Returns:
        Whether an item's measured features are ones the donor filters could
        have produced.
    """
    return lambda measured: (
        measured["name_chars"] >= min_chars and not measured["carries_category_token"]
    )


def item(
    name: str, corrupted: bool, brand: bool = False, repeated: bool = False
) -> dict[str, Any]:
    """Build one residual-class item for a test.

    Args:
        name: The name shown.
        corrupted: Whether the item carries a borrowed name.
        brand: Whether the underlying record carries a brand tag.
        repeated: Whether the name is carried by another record in the city.

    Returns:
        An item with the corpus facts the measurement needs beside it.
    """
    return {
        "item_id": f"M1-{name}-{int(corrupted)}",
        "class": "M1",
        "corrupted": corrupted,
        "record": {"name": name, "category": "amenity=cafe"},
        "brand_tagged": brand,
        "repeated_in_city": repeated,
    }


def test_the_donor_filters_leave_their_marks_on_the_name() -> None:
    """Every filter that constrains the donor is measured on the name shown."""
    measured = features(item("Kralingse Plaslicht", True), TOKENS)
    assert measured["name_chars"] == 19
    assert measured["carries_category_token"] is False
    assert features(item("Vermeulen Bakery", False), TOKENS)["carries_category_token"]
    assert features(item("Roos", False), TOKENS)["name_chars"] == 4


def test_a_name_no_donor_could_have_carried_is_clean_without_a_map() -> None:
    """The shortcut answers from the construction rather than from the record.

    A name shorter than the donor minimum, or one carrying a category token the
    donor filter excluded, cannot have been injected. Answering `belongs` to
    those is right every time and reads nothing about the place.
    """
    items = [
        item("Kralingse Plaslicht", True),
        item("Roos", False),
        item("Vermeulen Bakery", False),
    ]
    scored = score_flags(items, TOKENS, record_shortcut(6))
    assert scored["true_positive"] == 1
    assert scored["false_positive"] == 0
    assert scored["youdens_j"] == 1.0


def test_a_clean_name_the_filters_allow_is_not_separated() -> None:
    """The shortcut is not a name check; it only reads what construction fixed."""
    items = [item("Kralingse Plaslicht", True), item("Bakkerij Vermeulen", False)]
    scored = score_flags(items, TOKENS, record_shortcut(6))
    assert scored["false_positive"] == 1


def test_the_corpus_shortcut_reads_the_donor_s_surviving_record() -> None:
    """A rule asking only whether the name repeats separates the halves.

    The donor keeps its own record, so an injected name is repeated in its city
    by construction and a clean name is repeated only where the city happens to
    carry it twice. A corpus-level rule can act on that difference without
    reading anything about the place, which is what bounds what the screen
    baseline settles on this class.
    """
    items = [
        item("Kralingse Plaslicht", True, repeated=True),
        item("Kralingse Zoom", True, repeated=True),
        item("Roos", False, repeated=False),
        item("Vermeulen Bakery", False, repeated=False),
    ]
    scored = score_flags(items, TOKENS, lambda measured: measured["repeated_in_city"])
    assert scored["true_positive"] == 2
    assert scored["false_positive"] == 0
    assert scored["youdens_j"] == 1.0
    assert scored["youdens_j_low"] < scored["youdens_j"]


def test_the_oracle_cut_finds_a_length_that_separates_the_halves() -> None:
    from scripts.measure_item_artefacts import oracle_cut

    items = [item("Long borrowed name", True) for _ in range(10)] + [
        item("Short", False) for _ in range(10)
    ]
    best = oracle_cut(items, TOKENS, "name_chars")
    assert best["youdens_j"] == 1.0
    assert best["direction"] == "at_least"


def test_the_oracle_cut_finds_nothing_when_lengths_match() -> None:
    from scripts.measure_item_artefacts import oracle_cut

    items = [item("Same length A", True) for _ in range(10)] + [
        item("Same length B", False) for _ in range(10)
    ]
    assert oracle_cut(items, TOKENS, "name_chars")["youdens_j"] == 0.0
