"""Tests for the size-trend measurement that decides the frontier reference.

The properties checked here are the ones the decision rests on: that the weight
each item carries makes a model's weighted mean its balanced accuracy whatever
the class sizes are, that the slope is read on one degree of freedom per cluster
less one rather than one per item, and that a pool with no trend in it is not
reported as having one.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType

from poi_audit import scoring

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_size_trend.py"


def load() -> ModuleType:
    """Return the script as a module.

    Returns:
        The imported script, which reads the configuration on import.
    """
    spec = importlib.util.spec_from_file_location("score_size_trend", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Card:
    """The little of a model card the script reads."""

    def __init__(self, tag: str, parameters_b: float) -> None:
        """Build a card.

        Args:
            tag: The model's tag.
            parameters_b: Its parameter count in billions.
        """
        self.tag = tag
        self.family = "family"
        self.parameters_b = parameters_b


def judged(correct: list[bool], positives: int, cities: list[str]):
    """Build judgements whose truth classes and cities are known.

    Args:
        correct: Whether each item was answered correctly.
        positives: How many of the items carry the error asked about, taken from
            the front of the list.
        cities: The city each item came from.

    Returns:
        The judgements.
    """
    return [
        scoring.Scored(
            item_id=f"I{index}",
            city=cities[index],
            group="M1",
            truth_positive=index < positives,
            flagged=(index < positives) == right,
            valid=True,
            field_hit=None,
        )
        for index, right in enumerate(correct)
    ]


def test_weights_give_balanced_accuracy_on_an_unbalanced_class_split():
    """A model's weighted mean is its balanced accuracy, not its raw accuracy."""
    module = load()
    # Six positives, two of them right; two negatives, both right. The raw
    # accuracy is 0.5 and the balanced accuracy is (2/6 + 2/2) / 2.
    correct = [True, True, False, False, False, False, True, True]
    rows = module.rows_for(Card("m", 4.0), judged(correct, 6, ["A"] * 8))
    assert math.isclose(module.balanced_accuracy(rows), (2 / 6 + 1.0) / 2)
    assert not math.isclose(module.balanced_accuracy(rows), 0.5)


def test_a_model_answering_one_truth_class_is_refused():
    """A model with no negatives has no balanced accuracy to contribute."""
    module = load()
    rows = judged([True, True], positives=2, cities=["A", "B"])
    try:
        module.rows_for(Card("m", 4.0), rows)
    except ValueError as error:
        assert "truth class" in str(error)
    else:
        raise AssertionError("a single-class model was accepted")


def test_the_slope_is_read_on_the_clusters_rather_than_on_the_items():
    """The degrees of freedom count clusters, so 7,200 items do not buy precision."""
    module = load()
    rows = []
    for size, rate in ((1.0, 0.5), (2.0, 0.6), (4.0, 0.7), (8.0, 0.8)):
        for index in range(40):
            rows.append(
                {
                    "model": f"m{size}",
                    "family": "family",
                    "parameters_b": size,
                    "log2_parameters": math.log2(size),
                    "cluster": f"C{index % 5}",
                    "correct": float(index < rate * 40),
                    "weight": 1 / 40,
                }
            )
    fit = module.pooled_fit(rows, "test")
    assert fit["clusters"] == 5
    assert fit["items"] == 160
    assert fit["df"] == 4
    assert math.isclose(fit["slope_per_doubling"], 0.1, abs_tol=0.001)


def test_a_pool_with_no_trend_is_not_reported_as_having_one():
    """A flat pool gives a slope near zero and no positive trend."""
    module = load()
    scores = [
        {"log2_parameters": math.log2(size), "balanced_accuracy": accuracy}
        for size, accuracy in ((1.0, 0.7), (2.0, 0.68), (4.0, 0.71), (8.0, 0.69))
    ]
    fit = module.model_level_fit(scores, "test")
    assert abs(fit["slope_per_doubling"]) < 0.02
    assert not fit["positive_trend"]
    assert not module.rank_fit(scores, "test")["positive_trend"]


def test_a_gate_decided_on_another_task_does_not_refuse_a_model(tmp_path: Path):
    """The trend is on the residual class, so only its own gates refuse a model.

    Baseline dominance is decided on the detection task and label stability
    belongs to the label set rather than to a model. A reading that took either
    for a refusal would empty the gate-passing fit, since no model passes
    baseline dominance on detection.

    Args:
        tmp_path: Where the gate table is written.
    """
    module = load()
    table = tmp_path / "gates.csv"
    table.write_text(
        "model,gate,comparison,statistic,value,low,high,p_value,passes\n"
        "alpha,baseline_dominance,detection,odds ratio,0.4,,,0.001,False\n"
        "alpha,flag_rate_correction,residual,Youden's J,0.5,0.4,0.6,,True\n"
        "alpha,input_dependence,full against position,Youden's J,0.0,,,0.001,True\n"
        "beta,baseline_dominance,detection,odds ratio,0.2,,,0.001,False\n"
        "beta,flag_rate_correction,residual,Youden's J,0.0,-0.1,0.1,,False\n"
        "beta,input_dependence,full against position,Youden's J,0.0,,,0.5,True\n"
        ",label_stability,the label set,kappa,0.538,0.408,0.667,,\n",
        encoding="utf-8",
    )
    module.out_dir = lambda: tmp_path

    assert module.gate_failures() == {"beta"}
