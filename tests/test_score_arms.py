"""Tests for the arm-against-base comparisons."""

from __future__ import annotations

from poi_audit.config import get
from scripts.score_arms import comparisons


def test_the_adapted_model_is_compared_with_its_own_base_on_both_sets() -> None:
    rows = [row for row in comparisons() if row["arm"] == "low_rank_adaptation"]
    assert {row["set"] for row in rows} == {"control_test", "natural"}
    assert all(row["base_model"] == get("lora.base_tag") for row in rows)
    assert all(row["arm_run"] == row["base_run"] for row in rows)


def test_every_arm_changes_one_component_only() -> None:
    for row in comparisons():
        if row["arm"] == "retrieval_augmented":
            assert row["arm_model"] == row["base_model"]
            assert row["arm_run"] != row["base_run"]
