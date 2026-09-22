"""Tests for how a reference is scored and reported.

The properties checked here are the ones that keep a reference a reference: that
it is compared against both the largest and the best pool model, since those
answer different questions, and that its scores are written to tables of their
own rather than into the pool's, where a later analysis could read them without
meaning to.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "score_reference.py"
ANALYSIS = REPO / "data" / "processed" / "analysis"


def load() -> ModuleType:
    """Return the script as a module.

    Returns:
        The imported script, which reads the configuration on import.
    """
    spec = importlib.util.spec_from_file_location("score_reference", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows(name: str) -> list[dict[str, str]]:
    """Return one written table.

    Args:
        name: The file name.

    Returns:
        Its rows, or an empty list when it has not been written.
    """
    target = ANALYSIS / name
    if not target.exists():
        return []
    with target.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_a_reference_is_compared_against_the_largest_and_the_best():
    """The two answer different questions and neither replaces the other."""
    pairs = rows("reference_pairs.csv")
    if not pairs:
        return  # the references have not been run in this working tree
    for measure in {row["measure"] for row in pairs}:
        for reference in {row["reference"] for row in pairs}:
            roles = {
                row["against_role"]
                for row in pairs
                if row["measure"] == measure and row["reference"] == reference
            }
            assert "largest in the pool" in roles


def test_a_pairing_carries_what_it_needs_to_be_read_correctly():
    """A win on record correctness can be restraint rather than capability."""
    pairs = rows("reference_pairs.csv")
    if not pairs:
        return
    for row in pairs:
        assert row["reference_balanced_accuracy"]
        assert row["other_balanced_accuracy"]
        assert row["reference_flag_rate"]
        assert row["other_flag_rate"]


def test_reference_scores_stay_out_of_the_pool_tables():
    """A pool comparison, figure or fit must not pick a reference up."""
    for name in ("model_scores.csv", "model_scores_by_group.csv", "gates.csv"):
        for row in rows(name):
            assert "reference" not in row.get("model", "").lower()
            assert "glm" not in row.get("model", "").lower()
            assert "397b" not in row.get("model", "").lower()


def test_the_gate_is_applied_to_a_reference_as_it_is_to_a_pool_model():
    """A score whose answers do not move without the record is not a capability."""
    scored = rows("reference_scores.csv")
    if not scored:
        return
    gated = [row for row in scored if row.get("gate_p_value")]
    assert gated
    for row in gated:
        assert row["passes_input_dependence"] in {"True", "False"}


def test_the_comparators_are_read_from_the_pool_rather_than_named_here():
    """A hardcoded comparator would go stale the moment the pool changed."""
    module = load()
    source = SCRIPT.read_text(encoding="utf-8")
    assert "qwen3.5" not in source
    assert "phi4" not in source
    assert callable(module.comparators)
