"""Tests for the sizing and allocation of the natural set's second phase."""

from __future__ import annotations

from scripts.draw_natural_set import cell_quotas, expected_from_first_draw, phase2_size


def test_size_is_the_smallest_meeting_the_target() -> None:
    rates = {"a": 0.25, "b": 0.05}
    shares = {"a": 0.5, "b": 0.5}
    size = phase2_size(rates, shares, existing_expected=30.0, target=120)
    assert size == 600
    assert 30.0 + 0.15 * (size - 1) < 120 <= 30.0 + 0.15 * size + 1e-9


def test_no_draw_when_the_target_is_already_met() -> None:
    assert phase2_size({"a": 0.2}, {"a": 1.0}, existing_expected=130.0, target=120) == 0


def test_cell_quotas_sum_to_the_total_by_largest_remainder() -> None:
    shares = {"a": 0.35, "b": 0.35, "c": 0.15, "d": 0.15}
    quotas = cell_quotas(473, shares)
    assert sum(quotas.values()) == 473
    assert quotas == {"a": 166, "b": 165, "c": 71, "d": 71}


def test_the_expectation_counts_only_the_unlabelled_rest() -> None:
    records = [
        {"item_id": "N001", "language": "english", "maturity": "mature"},
        {"item_id": "N002", "language": "english", "maturity": "developing"},
        {"item_id": "N003", "language": "english", "maturity": "developing"},
    ]
    rates = {"english, mature": 0.1, "english, developing": 0.5}
    assert expected_from_first_draw(records, rates, labelled=1, observed=2) == 3.0
