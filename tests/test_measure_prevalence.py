"""Tests for the residual class's corpus-level rate.

The natural set is half a random draw and half a screened one, so the share of
wrong names in the sample is not the share in the corpus. Each record was drawn
with a known inclusion probability and is weighted by its reciprocal, which is
what recovers a corpus-level rate from a design that deliberately over-sampled
one stratum.
"""

from __future__ import annotations

from typing import Any

from scripts.measure_prevalence import cell_of, prevalence


def record(
    item_id: str,
    probability: float,
    language: str = "english",
    maturity: str = "mature",
    city: str = "Cork",
) -> dict[str, Any]:
    """Build one natural-set record for a test.

    Args:
        item_id: The identifier the gold labels key on.
        probability: The record's inclusion probability.
        language: The corpus factor the design balances on.
        maturity: The other corpus factor.
        city: The clustering unit.

    Returns:
        A record in the shape the draw writes.
    """
    return {
        "item_id": item_id,
        "inclusion_probability": probability,
        "language": language,
        "maturity": maturity,
        "city": city,
    }


def test_a_rare_record_carries_the_weight_its_probability_gives_it() -> None:
    """A record drawn one time in a hundred stands for a hundred."""
    records = [record("N1", 0.01), record("N2", 0.5)]
    labels = {"N1": "wrong", "N2": "belongs"}
    measured = prevalence(records, labels)
    assert measured["decided"] == 2
    assert measured["wrong"] == 1
    # 100 weighted against 2: the rare record dominates, and the raw half does
    # not survive the weighting.
    assert round(measured["weighted_prevalence"], 4) == round(100 / 102, 4)


def test_an_undecided_record_leaves_the_denominator_and_is_still_counted() -> None:
    """A record the readers could not decide carries no rate, and is reported."""
    records = [record("N1", 0.5), record("N2", 0.5), record("N3", 0.5)]
    labels = {"N1": "wrong", "N2": "belongs", "N3": "cannot_say"}
    measured = prevalence(records, labels)
    assert measured["decided"] == 2
    assert measured["cannot_say"] == 1
    assert measured["weighted_prevalence"] == 0.5


def test_the_design_cell_names_both_factors() -> None:
    """The corpus was built two by two and the rate is read the same way."""
    assert cell_of(record("N1", 0.5, "non_english", "developing")) == (
        "non_english, developing"
    )
