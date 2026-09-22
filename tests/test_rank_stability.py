"""Tests for rank stability under a city-clustered bootstrap.

The measure exists because an ordering read off a small set can be an accident
of which cities were drawn. A system that is better everywhere must stay first
under resampling, and two systems that answer identically must share the lead.
"""

from __future__ import annotations

import pytest

from scripts.score_rank_stability import rank_stability


def make(
    system_recall: dict[str, float],
) -> tuple[dict[str, dict[str, tuple[bool, bool]]], dict[str, str]]:
    """Build twelve cities of twenty records, five positive, for each system."""
    clusters: dict[str, str] = {}
    correct: dict[str, dict[str, tuple[bool, bool]]] = {s: {} for s in system_recall}
    for city in range(12):
        for k in range(20):
            item = f"c{city}i{k}"
            clusters[item] = f"city{city}"
            positive = k < 5
            for system, recall in system_recall.items():
                correct[system][item] = (positive, positive and k < 5 * recall)
    return correct, clusters


def test_a_clearly_better_system_stays_first() -> None:
    correct, clusters = make({"strong": 1.0, "weak": 0.2})
    result = rank_stability(correct, clusters, replicates=200, seed=42)
    assert result["strong"]["top_share"] == pytest.approx(1.0)
    assert result["strong"]["rank_high"] == pytest.approx(1.0)
    assert result["weak"]["rank_low"] == pytest.approx(2.0)


def test_identical_systems_share_the_lead() -> None:
    correct, clusters = make({"a": 0.6, "b": 0.6})
    result = rank_stability(correct, clusters, replicates=200, seed=42)
    assert result["a"]["top_share"] == pytest.approx(0.5)
    assert result["b"]["top_share"] == pytest.approx(0.5)


def test_full_set_j_is_reported() -> None:
    correct, clusters = make({"strong": 1.0, "weak": 0.2})
    result = rank_stability(correct, clusters, replicates=50, seed=42)
    assert result["strong"]["j"] == pytest.approx(1.0)
    assert result["weak"]["j"] == pytest.approx(0.2)


def test_the_result_is_reproducible() -> None:
    correct, clusters = make({"a": 0.6, "b": 0.4, "c": 0.5})
    first = rank_stability(correct, clusters, replicates=100, seed=42)
    again = rank_stability(correct, clusters, replicates=100, seed=42)
    assert first == again


def test_systems_must_answer_the_same_items() -> None:
    correct, clusters = make({"a": 0.6, "b": 0.4})
    del correct["b"]["c0i0"]
    with pytest.raises(ValueError):
        rank_stability(correct, clusters, replicates=10, seed=42)
