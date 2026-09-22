"""Tests for what the baseline's flags on untouched records actually are.

The clean half of every class is a live record nobody corrupted, and the study
scores a flag on it as a false positive. That is an assumption about the data
rather than a measurement of it: a live record can carry a malformed phone
number of its own. Each item records the violations it already held, and this
checks the flags against them.
"""

from __future__ import annotations

from typing import Any

from scripts.measure_clean_flags import explain


def item(checks: list[str], preexisting: list[str]) -> dict[str, Any]:
    """Build one clean item and the checks the baseline raised on it.

    Args:
        checks: The checks the baseline flagged.
        preexisting: The checks the record already violated when it was drawn.

    Returns:
        The pair the explanation reads.
    """
    return {
        "item_id": "S1-0001-2",
        "class": "S1",
        "checks_flagged": checks,
        "preexisting_flags": [{"check": one} for one in preexisting],
    }


def test_a_flag_on_a_violation_the_record_already_held_is_not_false() -> None:
    """The rule found what was there; the label said the record was clean."""
    explained = explain([item(["phone_syntax"], ["phone_syntax"])])
    assert explained["flagged"] == 1
    assert explained["explained"] == 1
    assert explained["unexplained"] == 0


def test_a_flag_on_a_check_the_record_did_not_violate_is_false() -> None:
    """A flag with nothing behind it is the false positive the study assumed."""
    explained = explain([item(["phone_syntax"], ["opening_hours_syntax"])])
    assert explained["explained"] == 0
    assert explained["unexplained"] == 1


def test_a_record_the_baseline_left_alone_is_not_counted_either_way() -> None:
    """Only flags are explained; silence needs no account."""
    explained = explain([item([], ["phone_syntax"])])
    assert explained["flagged"] == 0
    assert explained["explained"] == 0
    assert explained["unexplained"] == 0
