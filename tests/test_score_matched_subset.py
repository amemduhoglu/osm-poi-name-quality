"""Tests for scoring the residual class on pairs the construction did not mark.

Five filters choose the name the corrupted half carries and the pairing that
supplies the clean half constrains nothing about the name, so two of the
filters leave a mark a reader can act on without reading a map: the clean twin
may carry a category-indicative token, which no injected name does, and it may
be shorter than the donor minimum, which no injected name is. A pair whose
clean twin passes those two filters carries neither mark, and a score read on
those pairs alone is a score the marks cannot explain.

One asymmetry is not repaired this way and is not meant to be: the donor keeps
its own record, so an injected name is repeated in its city by construction.
That is a property of injection rather than of the pairing.
"""

from __future__ import annotations

from typing import Any

from scripts.score_matched_subset import eligible_pairs

TOKENS = {"bakery": "shop=bakery"}


def item(pair: str, name: str, corrupted: bool) -> dict[str, Any]:
    """Build one residual-class item for a test.

    Args:
        pair: The identifier the two halves share.
        name: The name shown.
        corrupted: Whether the item carries a borrowed name.

    Returns:
        An item in the shape the item builder writes.
    """
    return {
        "item_id": f"{pair}-{1 if corrupted else 2}",
        "pair_id": pair,
        "class": "M1",
        "corrupted": corrupted,
        "record": {"name": name},
    }


def test_a_pair_whose_clean_twin_carries_a_category_token_is_dropped() -> None:
    """No injected name carries one, so carrying one answers the question."""
    items = [
        item("M1-0001", "Kralingse Plaslicht", True),
        item("M1-0001", "Vermeulen Bakery", False),
    ]
    assert eligible_pairs(items, TOKENS, min_chars=6) == set()


def test_a_pair_whose_clean_twin_is_shorter_than_the_donor_minimum_is_dropped() -> None:
    """No injected name is shorter, so being shorter answers the question."""
    items = [
        item("M1-0002", "Kralingse Plaslicht", True),
        item("M1-0002", "Roos", False),
    ]
    assert eligible_pairs(items, TOKENS, min_chars=6) == set()


def test_a_pair_the_construction_left_unmarked_is_kept() -> None:
    """Both halves could have been either, so the pair asks the question."""
    items = [
        item("M1-0003", "Kralingse Plaslicht", True),
        item("M1-0003", "Bakkerij Vermeulen", False),
    ]
    assert eligible_pairs(items, TOKENS, min_chars=6) == {"M1-0003"}
