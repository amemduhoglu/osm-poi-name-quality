"""Test whether accuracy on the residual class rises with model size.

One measurement, and it decides one thing. The model pool document fixed a rule
on 2026-08-03, before any score existed: if the analysis finds a significant
positive trend of balanced accuracy with model size on the residual class
(p < 0.001 on the pooled comparison), the possibility that the class yields to
capacity stays open and a frontier reference is added at revision and reported
as such; if the trend is flat or inconclusive, none is added and the article
reports the flatness as what was measured across the range the pool spans.

The pooled fit is the comparison that rule names. One row per answered item,
weighted so that each model's weighted mean is its balanced accuracy, size
entering as a doubling because the pool spans a factor of thirteen and a slope
per billion parameters would be read off the largest models alone. The variance
is clustered on the study's clustering unit and the slope is read on a t
distribution with one degree of freedom per cluster less one.

Three readings are computed beside it and none replaces it. The pooled fit
treats the city as the independent unit, while size varies between models and
not within a city, so a fit on the twelve model-level scores is the conservative
reading of the same trend; a rank correlation assumes no functional form at all;
and the last asks whether the trend survives dropping the models the gates
refused, whose scores are the majority-class rate rather than a measured
capability.

Outputs:
    data/processed/analysis/size_trend.csv     one row per fit
    data/processed/analysis/size_trend.json    the decision and its provenance

Usage:
    python scripts/score_size_trend.py
"""

from __future__ import annotations

import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import statsmodels.api as sm
from scipy.stats import spearmanr

from poi_audit import scoring
from poi_audit.config import get, path
from poi_audit.inference import ModelCard, planned_runs, pool, response_path
from poi_audit.logsetup import setup

logger = setup("score_size_trend")

RUN = str(get("analysis.size_trend.run"))
CLUSTER_UNIT = str(get("analysis.cluster_unit"))
THRESHOLD = float(get("analysis.significance_threshold"))
CONFIDENCE = float(get("analysis.confidence"))


def out_dir() -> Path:
    """Return the directory the analysis outputs are written to."""
    target = path("data_processed") / "analysis"
    target.mkdir(parents=True, exist_ok=True)
    return target


# The gates that decide whether a model's score on the residual class may be
# read as a capability, which is the score this trend is fitted to. Baseline
# dominance is decided on the detection task and label stability belongs to the
# label set rather than to a model, so neither refuses a model here. The
# composed pipeline reads the table the same way.
APPLICABLE = ("flag_rate_correction", "input_dependence")


def gate_failures() -> set[str]:
    """Return the models a gate refuses on the residual class.

    Returns:
        The tags whose answers yield no capability conclusion, read from the
        gate table rather than restated here.

    Raises:
        FileNotFoundError: If the gates have not been measured.
    """
    target = out_dir() / "gates.csv"
    if not target.exists():
        raise FileNotFoundError(f"run scripts/score_gates.py first: {target} is absent")
    refused: set[str] = set()
    with target.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row["model"] or row["gate"] not in APPLICABLE:
                continue
            if row["passes"] != "True":
                refused.add(row["model"])
    return refused


def answered(card: ModelCard, truth: dict[str, dict[str, Any]]) -> list[scoring.Scored]:
    """Return one model's judged answers on the residual class.

    Args:
        card: The model.
        truth: The injected truth keyed by item.

    Returns:
        The judgements, empty when the run has no answers on disk.
    """
    target = response_path(RUN, card)
    if not target.exists():
        return []
    answers = scoring.read_answers(target)
    return scoring.score_name_injected(answers, truth) if answers else []


def rows_for(card: ModelCard, judged: list[scoring.Scored]) -> list[dict[str, Any]]:
    """Return one weighted row per answered item.

    Args:
        card: The model that answered.
        judged: Its judgements.

    Returns:
        One row per item, carrying the model's size, the item's cluster, whether
        the answer was right and the weight that makes the model's weighted mean
        its balanced accuracy. The weight is half the class's share, so a class
        holding fewer items counts for as much as the other one and a difference
        in class sizes between models cannot move the fit.
    """
    counts = {True: 0, False: 0}
    for one in judged:
        counts[one.truth_positive] += 1
    if not all(counts.values()):
        raise ValueError(f"{card.tag} answered only one truth class on {RUN}")
    return [
        {
            "model": card.tag,
            "family": card.family,
            "parameters_b": card.parameters_b,
            "log2_parameters": math.log2(card.parameters_b),
            "cluster": getattr(one, CLUSTER_UNIT),
            "correct": float(one.flagged == one.truth_positive),
            "weight": 0.5 / counts[one.truth_positive],
        }
        for one in judged
    ]


def balanced_accuracy(rows: list[dict[str, Any]]) -> float:
    """Return the weighted mean of one model's rows, which is its balanced accuracy.

    Args:
        rows: One model's item rows.

    Returns:
        The balanced accuracy.
    """
    weight = sum(row["weight"] for row in rows)
    return sum(row["weight"] * row["correct"] for row in rows) / weight


def pooled_fit(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    """Fit the pooled trend with a variance clustered on the study's unit.

    Args:
        rows: The item rows, over every model entering the fit.
        name: The fit's name in the output table.

    Returns:
        The slope in balanced accuracy per doubling of size, its interval, the
        statistic, the p-value and the counts the reading depends on.
    """
    design = sm.add_constant(np.array([row["log2_parameters"] for row in rows]))
    outcome = np.array([row["correct"] for row in rows])
    weights = np.array([row["weight"] for row in rows])
    clusters = np.array([row["cluster"] for row in rows])
    fit = sm.WLS(outcome, design, weights=weights).fit(
        cov_type="cluster",
        cov_kwds={"groups": clusters, "use_correction": True, "df_correction": True},
        use_t=True,
    )
    low, high = fit.conf_int(alpha=1.0 - CONFIDENCE)[1]
    return {
        "fit": name,
        "estimator": "weighted_least_squares",
        "variance": f"cluster_robust_by_{CLUSTER_UNIT}",
        "models": len({row["model"] for row in rows}),
        "items": len(rows),
        "clusters": len(set(clusters)),
        "slope_per_doubling": round(float(fit.params[1]), 4),
        "slope_low": round(float(low), 4),
        "slope_high": round(float(high), 4),
        "statistic": round(float(fit.tvalues[1]), 4),
        # The degrees of freedom the slope is read on, which is one per cluster
        # less one rather than one per item: the residual count would read a
        # pooled fit as though every item were an independent observation.
        "df": int(fit.df_resid_inference),
        "p_value": float(fit.pvalues[1]),
        "positive_trend": bool(fit.params[1] > 0 and float(fit.pvalues[1]) < THRESHOLD),
    }


def model_level_fit(scores: list[dict[str, Any]], name: str) -> dict[str, Any]:
    """Fit the trend on the model-level scores, the conservative reading.

    Args:
        scores: One row per model, carrying its size and its balanced accuracy.
        name: The fit's name in the output table.

    Returns:
        The same columns as the pooled fit, with the model as the unit.
    """
    design = sm.add_constant(np.array([row["log2_parameters"] for row in scores]))
    outcome = np.array([row["balanced_accuracy"] for row in scores])
    fit = sm.OLS(outcome, design).fit()
    low, high = fit.conf_int(alpha=1.0 - CONFIDENCE)[1]
    return {
        "fit": name,
        "estimator": "ordinary_least_squares",
        "variance": "model_level",
        "models": len(scores),
        "items": 0,
        "clusters": len(scores),
        "slope_per_doubling": round(float(fit.params[1]), 4),
        "slope_low": round(float(low), 4),
        "slope_high": round(float(high), 4),
        "statistic": round(float(fit.tvalues[1]), 4),
        "df": int(fit.df_resid),
        "p_value": float(fit.pvalues[1]),
        "positive_trend": bool(fit.params[1] > 0 and float(fit.pvalues[1]) < THRESHOLD),
    }


def rank_fit(scores: list[dict[str, Any]], name: str) -> dict[str, Any]:
    """Correlate size and score by rank, which assumes no functional form.

    Args:
        scores: One row per model.
        name: The fit's name in the output table.

    Returns:
        The correlation in the slope column, with no interval, since the reading
        it supports is the direction of the trend rather than its size.
    """
    sizes = [row["log2_parameters"] for row in scores]
    accuracies = [row["balanced_accuracy"] for row in scores]
    result = spearmanr(sizes, accuracies)
    return {
        "fit": name,
        "estimator": "spearman_rank",
        "variance": "model_level",
        "models": len(scores),
        "items": 0,
        "clusters": len(scores),
        "slope_per_doubling": round(float(result.statistic), 4),
        "slope_low": "",
        "slope_high": "",
        "statistic": round(float(result.statistic), 4),
        "df": len(scores) - 2,
        "p_value": float(result.pvalue),
        "positive_trend": bool(
            result.statistic > 0 and float(result.pvalue) < THRESHOLD
        ),
    }


def write_table(rows: list[dict[str, Any]]) -> None:
    """Write the fits.

    Args:
        rows: One row per fit.
    """
    target = out_dir() / "size_trend.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"wrote {target} with {len(rows)} rows")


def main() -> None:
    """Test the size trend and record what the pre-registered rule decides."""
    if RUN not in {str(run["name"]) for run in planned_runs()}:
        raise ValueError(f"{RUN} is absent from the run plan")
    truth = scoring.injected_truth()
    refused = gate_failures()
    item_rows: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    for card in pool():
        judged = answered(card, truth)
        if not judged:
            logger.warning(f"{card.tag} has no answers on {RUN}")
            continue
        rows = rows_for(card, judged)
        item_rows.extend(rows)
        scores.append(
            {
                "model": card.tag,
                "family": card.family,
                "parameters_b": card.parameters_b,
                "log2_parameters": math.log2(card.parameters_b),
                "balanced_accuracy": balanced_accuracy(rows),
                "passes_gates": card.tag not in refused,
            }
        )
    if len(scores) < 3:
        raise ValueError(f"only {len(scores)} models answered {RUN}")

    kept_rows = [row for row in item_rows if row["model"] not in refused]
    kept_scores = [row for row in scores if row["passes_gates"]]
    fits = [
        pooled_fit(item_rows, "pooled"),
        model_level_fit(scores, "model_level"),
        rank_fit(scores, "rank"),
        pooled_fit(kept_rows, "pooled_gate_passing"),
        model_level_fit(kept_scores, "model_level_gate_passing"),
        rank_fit(kept_scores, "rank_gate_passing"),
    ]
    write_table(fits)

    decisive = fits[0]
    decision = {
        "run": RUN,
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rule": (
            "a frontier reference is added at revision if balanced accuracy on "
            "the residual class rises with model size at p < "
            f"{THRESHOLD}, and is not added if the trend is flat or "
            "inconclusive; fixed 2026-08-03, before any score existed"
        ),
        "decisive_fit": decisive["fit"],
        "slope_per_doubling": decisive["slope_per_doubling"],
        "p_value": decisive["p_value"],
        "frontier_reference_required": bool(decisive["positive_trend"]),
        "models_refused_by_a_gate": sorted(refused),
        "scores": [
            {
                "model": row["model"],
                "parameters_b": row["parameters_b"],
                "balanced_accuracy": round(row["balanced_accuracy"], 4),
                "passes_gates": row["passes_gates"],
            }
            for row in scores
        ],
    }
    target = out_dir() / "size_trend.json"
    target.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    logger.info(f"wrote {target}")
    verdict = "requires" if decision["frontier_reference_required"] else "does not add"
    logger.info(
        f"slope {decisive['slope_per_doubling']} per doubling, "
        f"p = {decisive['p_value']:.3g}: the rule {verdict} a frontier reference"
    )


if __name__ == "__main__":
    main()
