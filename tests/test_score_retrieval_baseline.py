"""Tests for scoring the dense-retrieval comparator.

Three properties are worth a test here, and each protects the calibration
discipline the comparator's own module states: a threshold chosen on the
records it is then scored on is not a result, the split that keeps the two
apart is reproducible, and the oracle threshold names an upper bound rather
than a candidate for the calibrated cut.
"""

from __future__ import annotations

from scripts import score_retrieval_baseline


def test_the_calibration_split_is_disjoint_from_the_reported_one() -> None:
    """A threshold chosen on the records it is then scored on is not a result."""
    items = [{"item_id": f"i{n}"} for n in range(100)]
    calibration, reported = score_retrieval_baseline.split(items, share=0.2, seed=42)
    assert len(calibration) == 20
    assert len(reported) == 80
    assert not {item["item_id"] for item in calibration} & {
        item["item_id"] for item in reported
    }


def test_the_split_is_the_same_on_every_run() -> None:
    """The seed fixes it, so a rerun scores the same records."""
    items = [{"item_id": f"i{n}"} for n in range(100)]
    first, _ = score_retrieval_baseline.split(items, share=0.2, seed=42)
    second, _ = score_retrieval_baseline.split(items, share=0.2, seed=42)
    assert [item["item_id"] for item in first] == [item["item_id"] for item in second]


def test_the_oracle_threshold_is_reported_as_an_upper_bound() -> None:
    """The best threshold in hindsight bounds the comparator; it does not tune it."""
    scores = [0.1, 0.2, 0.8, 0.9]
    truth = [True, True, False, False]
    threshold, j = score_retrieval_baseline.oracle_threshold(scores, truth)
    assert 0.2 < threshold <= 0.8
    assert j == 1.0
