"""Tests for the configuration loader."""

from __future__ import annotations

import random

import pytest

from poi_audit import config


def test_dotted_key_reads_a_nested_value() -> None:
    """A dotted key addresses a value at any depth."""
    assert config.get("injected_set.classes.M1.items") == 300


def test_missing_key_raises_rather_than_defaulting_silently() -> None:
    """A parameter that is not in the file is an error, not a silent default."""
    with pytest.raises(KeyError):
        config.get("injected_set.classes.M9.items")


def test_missing_key_returns_an_explicit_default() -> None:
    """An explicit default is honoured."""
    assert config.get("injected_set.classes.M9.items", 0) == 0


def test_class_sizes_sum_to_the_declared_total() -> None:
    """The item counts are internally consistent, so no script has to add them up."""
    classes = config.get("injected_set.classes")
    corrupted = sum(entry["items"] for entry in classes.values())
    clean = config.get("injected_set.clean_items")
    assert corrupted == clean
    assert corrupted + clean == config.get("injected_set.total")


def test_natural_set_strata_sum_to_its_total() -> None:
    """The two strata account for every record in the natural set."""
    strata = config.get("natural_set.strata")
    assert sum(entry["records"] for entry in strata.values()) == config.get(
        "natural_set.total"
    )


def test_seed_everything_is_reproducible() -> None:
    """Seeding twice produces the same draw."""
    config.seed_everything()
    first = [random.random() for _ in range(5)]
    config.seed_everything()
    assert [random.random() for _ in range(5)] == first


def test_the_word_budget_follows_the_editor_rather_than_the_venue_median() -> None:
    """The editor called the manuscript far too long, so the ceiling fell.

    The floor is gone with it: a floor existed because the reframe was growing
    the article towards a venue median, and the editor overruled that reading.
    """
    manuscript = config.get("manuscript")
    assert manuscript["body_word_target"] == 10500
    assert manuscript["body_word_limit"] == 11000
    assert "body_word_floor" not in manuscript


def test_the_abstract_and_the_highlights_carry_the_editor_s_numbers() -> None:
    """Two hundred words for the abstract, 15 to 30 for each highlight."""
    manuscript = config.get("manuscript")
    assert manuscript["abstract_word_limit"] == 200
    assert manuscript["highlight_words_min"] == 15
    assert manuscript["highlight_words_max"] == 30


# A test asserting that section_budgets sums to within one tolerance step of
# body_word_target - 1720 is deferred to Task 3.1, which is where the
# allocation itself moves to the new target. Adding it here, against the old
# section_budgets this task leaves untouched, would fail on arithmetic alone
# (the old allocation sums to 11,850 against a target-derived expectation of
# 7,780) and would drop the suite below its current pass count.


def test_the_retrieval_baseline_is_configured_and_not_hardcoded() -> None:
    """The comparator's every parameter lives in the configuration.

    The threshold is not among them: it is calibrated on a split fixed by the
    seed, which is what keeps it from being chosen after the scores are seen.
    """
    block = config.get("retrieval_baseline")
    assert block["encoder"]
    assert block["radius_km"] > 0
    assert 0 < block["calibration_share"] < 1
    assert block["seed"] == 42
    assert block["similarity"] == "cosine"


def test_the_ceiling_covers_the_two_commercial_references() -> None:
    """Two commercial frontier models join the two open-weights references.

    Two vendors rather than one, so that a ceiling belonging to a lineage cannot
    be read as a ceiling belonging to the task.
    """
    api = config.get("models")["reference_api"]
    assert api["spend_ceiling_usd"] == 12.00
    assert len(api["models"]) == 4
    assert {entry["model"] for entry in api["models"]} >= {
        "anthropic/claude-opus-5",
        "openai/gpt-5.6-sol",
    }


def test_each_reference_declares_the_parameters_it_accepts() -> None:
    """The two commercial models do not take the same decoding parameters.

    The catalogue states that ``anthropic/claude-opus-5`` accepts no seed and
    ``openai/gpt-5.6-sol`` accepts no temperature. Sending one anyway, under the
    provider pinning this study uses, has the request refused or routed
    elsewhere, which is a silent change of instrument.
    """
    api = config.get("models")["reference_api"]
    for entry in api["models"]:
        assert entry["accepts"], entry["model"]
    accepts = {entry["model"]: set(entry["accepts"]) for entry in api["models"]}
    assert "seed" not in accepts["anthropic/claude-opus-5"]
    assert "temperature" not in accepts["openai/gpt-5.6-sol"]
