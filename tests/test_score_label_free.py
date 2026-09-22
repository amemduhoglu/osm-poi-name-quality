"""Tests for the precision bound that needs no label."""

from __future__ import annotations

import pytest

from scripts.score_label_free import precision_bound


def test_a_model_flagging_more_than_the_wrong_names_can_hold_is_bounded() -> None:
    assert precision_bound(
        {"a": 100, "b": 100}, {"a": 0.2, "b": 0.1}, 100
    ) == pytest.approx(0.3)


def test_the_bound_never_exceeds_one() -> None:
    assert precision_bound({"a": 100}, {"a": 0.5}, 10) == 1.0


def test_a_model_flagging_nothing_has_no_bound() -> None:
    with pytest.raises(ValueError):
        precision_bound({"a": 100}, {"a": 0.5}, 0)
