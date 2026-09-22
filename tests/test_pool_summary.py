"""Tests for the pool-level summary the results text quotes.

The article prints a median over the twelve models in four places, and each of
those medians was once taken as the upper of the two middle values, which is
not a median and overstated every one of them. The properties checked here are
the ones that would have caught it: an even-sized pool averages its two middle
values, a model that does not fall is counted rather than absorbed into a
sentence claiming the whole pool falls, and the count below chance is read
against the metric's own chance level.
"""

from __future__ import annotations

import importlib.util
import statistics
from pathlib import Path
from types import ModuleType

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_models.py"


def load() -> ModuleType:
    """Return the script as a module.

    Returns:
        The imported script, which reads the configuration on import.
    """
    spec = importlib.util.spec_from_file_location("score_models", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows(
    run: str, values: list[float], scored_set: str = "all"
) -> list[dict[str, object]]:
    """Return one scoring row per value.

    Args:
        run: The run the rows belong to.
        values: The balanced accuracy of each model, in pool order.
        scored_set: The set the rows were scored on.

    Returns:
        Rows shaped as the scoring writes them.
    """
    return [
        {
            "run": run,
            "model": f"model-{index}",
            "scored_set": scored_set,
            "balanced_accuracy": value,
            "youdens_j": round(2 * value - 1, 4),
        }
        for index, value in enumerate(values)
    ]


def cell(
    summary: list[dict[str, object]], quantity: str, run: str
) -> dict[str, object]:
    """Return the one summary row for a quantity and run.

    Args:
        summary: The summary rows.
        quantity: The quantity wanted.
        run: The run wanted.

    Returns:
        The matching row.
    """
    found = [
        row for row in summary if row["quantity"] == quantity and row["run"] == run
    ]
    assert len(found) == 1
    return found[0]


def test_an_even_pool_averages_its_two_middle_values() -> None:
    """The median of twelve models is not the seventh of them sorted."""
    module = load()
    values = [0.40, 0.45, 0.47, 0.48, 0.49, 0.50, 0.55, 0.56, 0.58, 0.59, 0.60, 0.61]
    summary = module.pool_summary(rows("a_run", values))
    row = cell(summary, "balanced_accuracy", "a_run")
    assert row["median"] == round(statistics.median(values), 4)
    assert row["median"] != values[len(values) // 2]
    assert row["maximum"] == 0.61
    assert row["minimum"] == 0.4


def test_below_chance_is_counted_against_the_metric_chance_level() -> None:
    """Balanced accuracy is below chance at 0.5 and Youden's J at zero."""
    module = load()
    values = [0.30, 0.45, 0.50, 0.70]
    summary = module.pool_summary(rows("a_run", values))
    assert cell(summary, "balanced_accuracy", "a_run")["below_chance"] == 2
    assert cell(summary, "youdens_j", "a_run")["below_chance"] == 2


def test_a_model_that_does_not_fall_is_counted() -> None:
    """The fall row reports how many models did not fall between the runs."""
    module = load()
    before = rows(module.FALL_FROM, [0.80, 0.60, 0.50, 0.55])
    after = rows(module.FALL_TO, [0.60, 0.50, 0.50, 0.60], scored_set="adjudicated")
    summary = module.pool_summary(before + after)
    row = cell(
        summary,
        "fall_in_balanced_accuracy",
        f"{module.FALL_FROM} to {module.FALL_TO}",
    )
    assert row["not_lower_than_source"] == 2
    assert row["maximum"] == 0.2
    assert row["minimum"] == -0.05
