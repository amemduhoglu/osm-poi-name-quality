"""Apply the audit gates, which decide whether a score may be read as a capability.

Four gates, stated once in the method and applied to every task:

    Baseline dominance. A model beats the best trivial answer available, which
    here is the rule baseline, not the first thing written down. Measured in
    scripts/score_models.py by an exact McNemar paired on the record.

    Label stability. The labels survive an independent check. Measured for the
    natural set by the inter-annotator kappa, and given by construction for the
    injected set.

    Input dependence. A model demonstrably uses the input the task claims to
    test. Measured here: the residual question is asked with the record in full,
    then with the name alone, then with everything except the name, then with
    the position alone. A model whose answers do not move when the record is
    taken away was never reading it.

    Flag-rate correction. A detection score is read beside how often the model
    flags at all, which every table in scripts/score_models.py carries.

This script measures the third and gathers all four into one statement per
model. It decides nothing on its own: a gate reports pass or fail, and a task
failing any gate yields no capability conclusion whatever its accuracy.

Outputs:
    data/processed/analysis/input_dependence.csv
    data/processed/analysis/views.csv
    data/processed/analysis/gates.csv

Usage:
    python scripts/score_gates.py
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from poi_audit import scoring
from poi_audit.config import get, path
from poi_audit.inference import planned_runs, pool, response_path
from poi_audit.logsetup import setup
from poi_audit.references import load_gazetteer
from poi_audit.stats import exact_mcnemar

logger = setup("score_gates")

FULL = "m1_name_full_v1"
DETECT = "injected_detect_v1"
VIEWS = {
    "name_only": "m1_name_name_only_v1",
    "tags_only": "m1_name_tags_only_v1",
    "neither": "m1_name_neither_v1",
}
THRESHOLD = float(get("analysis.significance_threshold"))
POOL = "pool median"


def out_dir() -> Path:
    """Return the directory the analysis outputs are written to."""
    target = path("data_processed") / "analysis"
    target.mkdir(parents=True, exist_ok=True)
    return target


def judged_for(run_name: str, card: Any, truth: dict[str, Any]) -> list[scoring.Scored]:
    """Return one model's judgements for one run, or an empty list.

    Args:
        run_name: The run.
        card: The model.
        truth: The injected truth keyed by item.

    Returns:
        The judgements, empty when the run has no answers on disk.
    """
    target = response_path(run_name, card)
    if not target.exists():
        return []
    answers = scoring.read_answers(target)
    return scoring.score_name_injected(answers, truth) if answers else []


def input_dependence() -> list[dict[str, Any]]:
    """Compare the full view against each reduced one, model by model.

    Returns:
        One row per model and reduced view, carrying the two Youden's J values,
        the discordant counts and the exact test. The comparison is paired on
        the record: the same items are asked under both views, so a difference
        between summary rates would throw away what the pairing knows.
    """
    truth = scoring.injected_truth()
    rows: list[dict[str, Any]] = []
    for card in pool():
        full = judged_for(FULL, card, truth)
        if not full:
            continue
        full_right = scoring.correctness(full)
        full_counts = scoring.confusion(full)
        for view, run_name in VIEWS.items():
            reduced = judged_for(run_name, card, truth)
            if not reduced:
                continue
            first, second = scoring.discordant(full_right, scoring.correctness(reduced))
            result = exact_mcnemar(first, second)
            reduced_counts = scoring.confusion(reduced)
            rows.append(
                {
                    "model": card.tag,
                    "family": card.family,
                    "parameters_b": card.parameters_b,
                    "view_removed": view,
                    "j_full": round(full_counts.youdens_j, 4),
                    "j_reduced": round(reduced_counts.youdens_j, 4),
                    "j_drop": round(
                        full_counts.youdens_j - reduced_counts.youdens_j, 4
                    ),
                    "full_only_correct": result.first_only,
                    "reduced_only_correct": result.second_only,
                    "p_value": result.p_value,
                    "odds_ratio": result.odds_ratio,
                    "odds_ratio_low": result.odds_ratio_low,
                    "odds_ratio_high": result.odds_ratio_high,
                    "moves_with_input": result.p_value < THRESHOLD,
                }
            )
    return rows


def view_summary(dependence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Gather every model's score under all four views.

    The gate reads one view, the one holding neither the name nor the tags,
    because that is what answers whether a model reads the record at all. Which
    part of the record carried the score is a different question, and the two
    remaining views answer it: a score that survives neither removal was
    carried by the name and the tags together rather than by either alone.
    Reporting the decisive view on its own leaves that unsaid.

    Args:
        dependence: The input-dependence rows.

    Returns:
        One row per model carrying the four scores, and a pooled row carrying
        the median of each. A view a model did not run is left blank rather
        than filled with a zero, which would read as a measured chance score.
    """
    by_model: dict[str, dict[str, Any]] = {}
    for row in dependence:
        entry = by_model.setdefault(
            str(row["model"]),
            {
                "model": row["model"],
                "family": row["family"],
                "parameters_b": row["parameters_b"],
                "j_full": row["j_full"],
                **{f"j_{view}": "" for view in VIEWS},
            },
        )
        entry[f"j_{row['view_removed']}"] = row["j_reduced"]

    rows = [by_model[key] for key in sorted(by_model)]
    if rows:
        pooled: dict[str, Any] = {
            "model": POOL,
            "family": "",
            "parameters_b": "",
        }
        for column in ["j_full", *[f"j_{view}" for view in VIEWS]]:
            values = [float(row[column]) for row in rows if row[column] != ""]
            pooled[column] = round(statistics.median(values), 4) if values else ""
        rows.append(pooled)
    return rows


def label_stability() -> dict[str, Any]:
    """Read the agreement measurement the natural set's labels were checked by.

    Returns:
        The overall kappa with its interval and the band it falls in.

    Raises:
        FileNotFoundError: If agreement has not been measured.
    """
    target = path("data_processed") / "natural_set" / "agreement.json"
    if not target.exists():
        raise FileNotFoundError(
            f"agreement not measured: {target}; run scripts/measure_agreement.py"
        )
    return json.loads(target.read_text(encoding="utf-8"))["overall"]


def baseline_dominance() -> dict[str, dict[str, Any]]:
    """Compare every model against the rule baseline, paired on the record.

    The gate asks whether a score beats the best trivial answer available, which
    on both tasks is the rule a maintainer could have written instead of running
    a model. The comparison is made on the detection task, where the baseline
    settles six classes outright and the question is whether a model adds
    anything to it.

    Returns:
        One entry per model, keyed by tag.
    """
    truth = scoring.injected_truth()
    baseline = scoring.score_baseline(list(truth.values()), load_gazetteer())
    baseline_right = scoring.correctness(baseline)
    results: dict[str, dict[str, Any]] = {}
    for card in pool():
        target = response_path(DETECT, card)
        if not target.exists():
            continue
        answers = scoring.read_answers(target)
        if not answers:
            continue
        judged = scoring.score_detect(answers, truth)
        first, second = scoring.discordant(scoring.correctness(judged), baseline_right)
        result = exact_mcnemar(first, second)
        results[card.tag] = {
            "model_only_correct": first,
            "baseline_only_correct": second,
            "p_value": result.p_value,
            "odds_ratio": result.odds_ratio,
            "odds_ratio_low": result.odds_ratio_low,
            "odds_ratio_high": result.odds_ratio_high,
            # Beating the baseline means winning the discordant records, not
            # merely differing from it: a model significantly worse than the
            # rule fails this gate as surely as one indistinguishable from it.
            "passes": result.p_value < THRESHOLD and result.odds_ratio > 1.0,
        }
    return results


def flag_rate_correction() -> dict[str, dict[str, Any]]:
    """Read each model's residual-question score against its own flag rate.

    On balanced halves a model flagging at random reaches a recall equal to its
    flag rate, so the corrected summary is Youden's J and the gate is whether
    its interval clears zero. A model whose bounds straddle zero has not been
    told apart from one that suspects everything, whatever its recall.

    Returns:
        One entry per model, keyed by tag.
    """
    truth = scoring.injected_truth()
    results: dict[str, dict[str, Any]] = {}
    for card in pool():
        judged = judged_for(FULL, card, truth)
        if not judged:
            continue
        counts = scoring.confusion(judged)
        low, high = counts.youdens_j_interval()
        results[card.tag] = {
            "flag_rate": round(counts.flag_rate().estimate, 4),
            "youdens_j": round(counts.youdens_j, 4),
            "youdens_j_low": round(low, 4),
            "youdens_j_high": round(high, 4),
            "passes": low > 0.0,
        }
    return results


def gates(dependence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Gather one row per model and gate, and one for the labels.

    Args:
        dependence: The input-dependence rows.

    Returns:
        One row per model and gate, plus the label-stability row the natural
        set carries once, since a label is a property of the item and not of the
        model that answered it. A model passes the whole audit only when every
        gate that applies to it passes.
    """
    decisive = {
        row["model"]: row for row in dependence if row["view_removed"] == "neither"
    }
    dominance = baseline_dominance()
    corrected = flag_rate_correction()
    rows: list[dict[str, Any]] = []

    for model in sorted(set(decisive) | set(dominance) | set(corrected)):
        if model in dominance:
            entry = dominance[model]
            rows.append(
                {
                    "model": model,
                    "gate": "baseline_dominance",
                    "task": "detection task",
                    "comparison": "detection against the rule baseline",
                    "statistic": "odds ratio of the discordant pairs",
                    "value": round(float(entry["odds_ratio"]), 4),
                    "low": "",
                    "high": "",
                    "p_value": entry["p_value"],
                    "passes": entry["passes"],
                }
            )
        if model in corrected:
            entry = corrected[model]
            rows.append(
                {
                    "model": model,
                    "gate": "flag_rate_correction",
                    "task": "residual question",
                    "comparison": (
                        f"residual question, flagging {entry['flag_rate']:.3f} "
                        "of records"
                    ),
                    "statistic": "Youden's J",
                    "value": entry["youdens_j"],
                    "low": entry["youdens_j_low"],
                    "high": entry["youdens_j_high"],
                    "p_value": "",
                    "passes": entry["passes"],
                }
            )
        if model in decisive:
            entry = decisive[model]
            rows.append(
                {
                    "model": model,
                    "gate": "input_dependence",
                    "task": "residual question",
                    "comparison": "full record against position alone",
                    "statistic": "Youden's J with the record taken away",
                    "value": entry["j_reduced"],
                    "low": "",
                    "high": "",
                    "p_value": entry["p_value"],
                    "passes": bool(entry["moves_with_input"]),
                }
            )

    rows.extend(residual_dominance())

    overall = label_stability()
    rows.append(
        {
            "model": "",
            "gate": "label_stability",
            "task": "the label set",
            "comparison": "natural set, two readers working a fixed protocol",
            "statistic": "Cohen's kappa",
            "value": round(float(overall["kappa"]), 4),
            "low": round(float(overall["ci_low"]), 4),
            "high": round(float(overall["ci_high"]), 4),
            "p_value": "",
            # No threshold on kappa was fixed before the labelling, so none is
            # applied afterwards: the method says this gate is measured, and a
            # cut written now would be a cut chosen with the value in view. The
            # measurement is reported and read against published bands in the
            # article, which is what the design promised.
            "passes": "",
        }
    )
    return rows


RESIDUAL_TASKS = {
    "control_test": "residual question, control test items",
    "natural_stratum_a": "residual question, the natural set",
}


def residual_dominance() -> list[dict[str, Any]]:
    """Read the baseline-dominance gate as it falls on the residual question.

    The gate was computed for the detection task alone, where the cheapest
    instrument is the format rule. On the residual question that rule reads no
    name and is silent by construction, so the comparison is against the screen
    and it is made by scripts/score_screen_baseline.py. This reads what that
    script wrote, so that every gate's verdict reaches the article from one
    table.

    Returns:
        One row per model and set, empty when the comparison has not been run.
    """
    source = out_dir() / "against_screen_residual.csv"
    if not source.exists():
        logger.warning(
            f"no residual dominance at {source}; "
            "run scripts/score_screen_baseline.py"
        )
        return []
    with source.open(encoding="utf-8") as handle:
        scored = list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    for row in scored:
        odds = row["odds_ratio"]
        rows.append(
            {
                "model": row["model"],
                "gate": "baseline_dominance",
                "task": RESIDUAL_TASKS[row["task"]],
                "comparison": "against the string and dictionary screen",
                "statistic": "odds ratio of the discordant pairs",
                "value": round(float(odds), 4) if odds else "",
                "low": "",
                "high": "",
                "p_value": row["p_value"],
                "passes": row["passes"] == "True",
            }
        )
    return rows


def write(name: str, rows: list[dict[str, Any]]) -> None:
    """Write one table.

    Args:
        name: The file name.
        rows: The rows.
    """
    if not rows:
        logger.warning(f"nothing to write for {name}")
        return
    target = out_dir() / name
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"wrote {target} with {len(rows)} rows")


def main() -> None:
    """Measure the input-dependence gate and report it per model."""
    known = {str(run["name"]) for run in planned_runs()}
    missing = {FULL, *VIEWS.values()} - known
    if missing:
        raise ValueError(f"runs absent from the plan: {sorted(missing)}")
    dependence = input_dependence()
    write("input_dependence.csv", dependence)
    write("views.csv", view_summary(dependence))
    passed = gates(dependence)
    write("gates.csv", passed)
    for gate in ("baseline_dominance", "flag_rate_correction", "input_dependence"):
        rows = [row for row in passed if row["gate"] == gate]
        if rows:
            logger.info(
                f"{gate}: {sum(1 for row in rows if row['passes'])} of {len(rows)} "
                "models pass"
            )


if __name__ == "__main__":
    main()
