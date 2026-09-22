"""Tests for the control-set scores and the rule choosing the adaptation models."""

from __future__ import annotations

import pytest

from scripts.score_control import select_models


def test_the_best_and_the_lower_median_are_chosen() -> None:
    scores = {"a": 0.30, "b": 0.10, "c": 0.20, "d": 0.05}
    assert select_models(scores) == {"best": "a", "median": "b"}


def test_an_odd_count_takes_the_middle_model() -> None:
    scores = {"a": 0.30, "b": 0.10, "c": 0.20}
    assert select_models(scores) == {"best": "a", "median": "c"}


def test_ties_are_broken_by_tag() -> None:
    scores = {"b": 0.2, "a": 0.2, "c": 0.1}
    assert select_models(scores)["best"] == "a"


def test_one_model_is_both() -> None:
    assert select_models({"a": 0.1}) == {"best": "a", "median": "a"}


def test_no_model_is_refused() -> None:
    with pytest.raises(ValueError):
        select_models({})


def test_no_excluded_model_is_pooled_or_chosen() -> None:
    import csv
    import json

    from poi_audit.config import get, path

    out = path("data_processed") / "analysis"
    if not (out / "control_scores.csv").exists():
        pytest.skip("control not scored")
    excluded = set(get("run.replay_excluded_shown"))
    with (out / "control_scores.csv").open(encoding="utf-8") as handle:
        assert not {row["model"] for row in csv.DictReader(handle)} & excluded
    chosen = json.loads((out / "adaptation_models.json").read_text("utf-8"))
    assert not set(chosen["eligible"]) & excluded
