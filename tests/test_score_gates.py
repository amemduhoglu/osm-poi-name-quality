"""Tests for the audit gates and for how the four views are reported.

The residual question is asked four ways and the gate reads one of them: the
view holding neither the name nor the tags, which answers whether a model reads
the record at all. The other two answer what it reads the record *for*, and a
score that survives one removal and not the other says which part of the record
carried it. They are summarized here so that the article reports all four
rather than the decisive one alone.
"""

from __future__ import annotations

from typing import Any

from scripts.score_gates import POOL, view_summary


def dependence_row(
    model: str, view: str, j_full: float, j_reduced: float
) -> dict[str, Any]:
    """Build one input-dependence row for a test.

    Args:
        model: The model's tag.
        view: The reduced view.
        j_full: Youden's J on the full record.
        j_reduced: Youden's J on the reduced view.

    Returns:
        A row in the shape the dependence comparison writes.
    """
    return {
        "model": model,
        "family": "Test",
        "parameters_b": 1.0,
        "view_removed": view,
        "j_full": j_full,
        "j_reduced": j_reduced,
    }


def test_every_view_is_reported_beside_the_full_record() -> None:
    """A reader sees all four scores, not only the one the gate turns on."""
    rows = [
        dependence_row("a", "name_only", 0.60, 0.01),
        dependence_row("a", "tags_only", 0.60, -0.02),
        dependence_row("a", "neither", 0.60, 0.00),
    ]
    summary = view_summary(rows)
    entry = next(one for one in summary if one["model"] == "a")
    assert entry["j_full"] == 0.60
    assert entry["j_name_only"] == 0.01
    assert entry["j_tags_only"] == -0.02
    assert entry["j_neither"] == 0.00


def test_the_pool_median_is_reported_for_every_view() -> None:
    """The pool's position is what the article states, model by model beside it."""
    rows = [
        dependence_row("a", "name_only", 0.60, 0.02),
        dependence_row("a", "tags_only", 0.60, 0.00),
        dependence_row("a", "neither", 0.60, 0.00),
        dependence_row("b", "name_only", 0.40, 0.00),
        dependence_row("b", "tags_only", 0.40, 0.04),
        dependence_row("b", "neither", 0.40, 0.00),
    ]
    summary = view_summary(rows)
    pooled = next(one for one in summary if one["model"] == POOL)
    assert pooled["j_full"] == 0.50
    assert pooled["j_name_only"] == 0.01
    assert pooled["j_tags_only"] == 0.02
    assert pooled["j_neither"] == 0.00


def test_a_model_missing_a_view_is_reported_without_it() -> None:
    """A view that was not run leaves a blank rather than a zero."""
    rows = [dependence_row("a", "neither", 0.60, 0.00)]
    summary = view_summary(rows)
    entry = next(one for one in summary if one["model"] == "a")
    assert entry["j_name_only"] == ""
    assert entry["j_neither"] == 0.00
