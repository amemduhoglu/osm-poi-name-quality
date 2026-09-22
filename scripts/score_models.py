"""Score every model on every run, against the truth each item set carries.

This produces the numbers the results sections read and nothing else: no figure
is drawn here, no display item is assigned, and no analysis is run that has no
place in the article's outline.

Every proportion carries a Wilson score interval. Detection is summarized by
Youden's J with the flag rate beside it, because a model that flags everything
reaches perfect recall and has decided nothing. Comparisons across groups whose
corrupted share differs are made on balanced accuracy. Each model is compared
against the rule baseline by an exact McNemar test on the records where the two
disagreed, reported with the odds ratio of the discordant pairs.

The natural set is scored twice, on the adjudicated set and on the decidable
core, because a single score there cannot separate a model that cannot answer
the residual question from a question the record does not answer.

Outputs:
    data/processed/analysis/model_scores.csv        one row per run and model
    data/processed/analysis/model_scores_by_group.csv  per class and stratum
    data/processed/analysis/against_baseline.csv    the paired comparisons
    data/processed/analysis/pool_summary.csv        the pool-level figures
    data/processed/analysis/model_scores.json       the run's own provenance

Usage:
    python scripts/score_models.py
"""

from __future__ import annotations

import csv
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from poi_audit import scoring
from poi_audit.config import get, path
from poi_audit.inference import planned_runs, pool, response_path
from poi_audit.logsetup import setup
from poi_audit.references import load_gazetteer
from poi_audit.stats import Confusion, clustered_paired_difference, exact_mcnemar

logger = setup("score_models")

CONFIDENCE = float(get("analysis.confidence"))


def out_dir() -> Path:
    """Return the directory the analysis outputs are written to."""
    target = path("data_processed") / "analysis"
    target.mkdir(parents=True, exist_ok=True)
    return target


def judge(
    run: dict[str, Any],
    answers: list[dict[str, Any]],
    injected: dict[str, dict[str, Any]],
    natural: dict[str, dict[str, Any]],
    core_only: bool = False,
) -> list[scoring.Scored]:
    """Judge one run's answers with the scorer its item set calls for.

    Args:
        run: The planned run.
        answers: The answers as the runner wrote them.
        injected: The injected truth keyed by item.
        natural: The gold labels keyed by item.
        core_only: Score the natural set on the decidable core alone.

    Returns:
        The judgements.
    """
    if run["items"] == "natural":
        return scoring.score_name_natural(answers, natural, core_only=core_only)
    if run["task"] == "detect":
        return scoring.score_detect(answers, injected)
    return scoring.score_name_injected(answers, injected)


def measures(counts: Confusion) -> dict[str, Any]:
    """Return the measures every scored set reports.

    Args:
        counts: The confusion.

    Returns:
        The point estimates with their intervals, flattened for a table.
    """
    sensitivity = counts.sensitivity(CONFIDENCE)
    specificity = counts.specificity(CONFIDENCE)
    flag = counts.flag_rate(CONFIDENCE)
    balanced = counts.balanced_accuracy_interval(CONFIDENCE)
    index = counts.youdens_j_interval(CONFIDENCE)
    return {
        "items": counts.total,
        "positives": counts.positives,
        "true_positive": counts.true_positive,
        "false_positive": counts.false_positive,
        "recall": round(sensitivity.estimate, 4),
        "recall_low": round(sensitivity.low, 4),
        "recall_high": round(sensitivity.high, 4),
        "specificity": round(specificity.estimate, 4),
        "specificity_low": round(specificity.low, 4),
        "specificity_high": round(specificity.high, 4),
        "flag_rate": round(flag.estimate, 4),
        "flag_rate_low": round(flag.low, 4),
        "flag_rate_high": round(flag.high, 4),
        "balanced_accuracy": round(counts.balanced_accuracy, 4),
        "balanced_accuracy_low": round(balanced[0], 4),
        "balanced_accuracy_high": round(balanced[1], 4),
        "youdens_j": round(counts.youdens_j, 4),
        "youdens_j_low": round(index[0], 4),
        "youdens_j_high": round(index[1], 4),
    }


def score_all() -> dict[str, list[dict[str, Any]]]:
    """Score every model on every run that has answers on disk.

    Returns:
        The three tables, each as a list of rows.

    Raises:
        FileNotFoundError: If the gold labels have not been built.
    """
    injected = scoring.injected_truth()
    natural = scoring.natural_truth()
    gazetteer = load_gazetteer()
    baseline = scoring.score_baseline(list(injected.values()), gazetteer)
    baseline_right = scoring.correctness(baseline)
    units = scoring.clusters()
    logger.info(
        f"rule baseline: J {scoring.confusion(baseline).youdens_j:.3f} "
        f"over {len(baseline)} items"
    )

    rows: list[dict[str, Any]] = []
    grouped: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []

    for run in planned_runs():
        for card in pool():
            target = response_path(str(run["name"]), card)
            if not target.exists():
                continue
            answers = scoring.read_answers(target)
            if not answers:
                continue
            # The natural set is scored on both of the sets fixed in advance;
            # every other run has one.
            sets = [("all", False)]
            if run["items"] == "natural":
                sets = [("adjudicated", False), ("decidable_core", True)]
            for name, core_only in sets:
                judged = judge(run, answers, injected, natural, core_only)
                counts = scoring.confusion(judged)
                hits, attempts = scoring.field_accuracy(judged)
                # The natural set is a stratified sample with known inclusion
                # probabilities, so its estimates are reported twice: raw
                # within stratum, and weighted back to the corpus.
                weighted: dict[str, Any] = {}
                if run["items"] == "natural":
                    probabilities = {
                        item: float(gold["inclusion_probability"])
                        for item, gold in natural.items()
                    }
                    for key, value in scoring.reweighted(judged, probabilities).items():
                        weighted[f"weighted_{key}"] = round(value, 4)
                rows.append(
                    {
                        "run": run["name"],
                        "model": card.tag,
                        "family": card.family,
                        "parameters_b": card.parameters_b,
                        "task": run["task"],
                        # Named item_set, not items: the measures below carry
                        # their own `items`, which is a count, and a key
                        # collision here silently dropped this one.
                        "item_set": run["items"],
                        "view": run["view"],
                        "prompt_version": run["prompt_version"],
                        "constrained": run["constrained"],
                        "scored_set": name,
                        "answers": len(answers),
                        # Counted over every answer in the file rather than over
                        # the scored subset, so that answers minus invalid is a
                        # true count. The scored subset is smaller on the
                        # natural runs, where items with no gold label are
                        # dropped, and counting invalid there reported valid
                        # answers that do not exist.
                        "invalid": sum(
                            1 for answer in answers if not answer.get("valid")
                        ),
                        "field_hits": hits,
                        "field_attempts": attempts,
                        **measures(counts),
                        **weighted,
                    }
                )
                for group, group_counts in scoring.by_group(judged).items():
                    grouped.append(
                        {
                            "run": run["name"],
                            "model": card.tag,
                            "scored_set": name,
                            "group": group,
                            **measures(group_counts),
                        }
                    )
                # The baseline answers the injected set only, so the paired
                # comparison is made where both systems saw the same records.
                if run["items"] != "natural":
                    first, second = scoring.discordant(
                        scoring.correctness(judged), baseline_right
                    )
                    result = exact_mcnemar(first, second)
                    # The exact test asks whether the two differ; the clustered
                    # difference asks by how much, with the city as the unit
                    # that could have come out otherwise.
                    spread = clustered_paired_difference(
                        scoring.correctness(judged), baseline_right, units
                    )
                    paired.append(
                        {
                            "run": run["name"],
                            "model": card.tag,
                            "model_only_correct": result.first_only,
                            "baseline_only_correct": result.second_only,
                            "p_value": result.p_value,
                            "odds_ratio": result.odds_ratio,
                            "odds_ratio_low": result.odds_ratio_low,
                            "odds_ratio_high": result.odds_ratio_high,
                            "accuracy_difference": spread.difference,
                            "difference_low": spread.low,
                            "difference_high": spread.high,
                            "clustered_p_value": spread.p_value,
                            "clusters": spread.clusters,
                        }
                    )
    return {
        "rows": rows,
        "grouped": grouped,
        "paired": paired,
        "pairwise": pairwise(injected, natural),
        "summary": pool_summary(rows),
    }


# The two runs the article's model-choice question is asked on: the residual
# class where the answer is determinable by construction, and the natural set
# where it is not. A pairwise table over every run would be an analysis with no
# place in the outline.
PAIRWISE_RUNS = ("m1_name_full_v1", "natural_name_v1")


def pairwise(
    injected: dict[str, dict[str, Any]], natural: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Compare every pair of models on the runs the model-choice question uses.

    Each pair is tested individually rather than adjusted for multiplicity, and
    the conclusions the article draws rest on differences significant at the
    configured threshold; a comparison near the boundary is read as
    inconclusive rather than as evidence of equivalence.

    Args:
        injected: The injected truth keyed by item.
        natural: The gold labels keyed by item.

    Returns:
        One row per run and unordered pair of models.
    """
    runs = {str(run["name"]): run for run in planned_runs()}
    cards = list(pool())
    rows: list[dict[str, Any]] = []
    for run_name in PAIRWISE_RUNS:
        run = runs[run_name]
        answered: dict[str, dict[str, bool]] = {}
        for card in cards:
            target = response_path(run_name, card)
            if not target.exists():
                continue
            answers = scoring.read_answers(target)
            if not answers:
                continue
            answered[card.tag] = scoring.correctness(
                judge(run, answers, injected, natural)
            )
        tags = sorted(answered)
        for index, first_tag in enumerate(tags):
            for second_tag in tags[index + 1 :]:
                first, second = scoring.discordant(
                    answered[first_tag], answered[second_tag]
                )
                result = exact_mcnemar(first, second)
                rows.append(
                    {
                        "run": run_name,
                        "first": first_tag,
                        "second": second_tag,
                        "first_only_correct": result.first_only,
                        "second_only_correct": result.second_only,
                        "p_value": result.p_value,
                        "odds_ratio": result.odds_ratio,
                        "odds_ratio_low": result.odds_ratio_low,
                        "odds_ratio_high": result.odds_ratio_high,
                        "separates": result.p_value
                        < float(get("analysis.significance_threshold")),
                    }
                )
    return rows


# The run the injected residual class is scored on, and the run the same
# question is asked of records nobody corrupted. The article reads the fall
# between them, so the fall is released rather than left to be subtracted.
FALL_FROM = "m1_name_full_v1"
FALL_TO = "natural_name_v1"


def pool_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize each run over the pool, and the fall between two of them.

    The article quotes a median, a highest and a lowest value over the twelve
    models, and a number of models below chance. Each is computed here so that
    every figure the text prints is read from a released output rather than
    from a reader's arithmetic. On a fall the maximum is the largest fall and a
    negative minimum is a model that did not fall at all.

    Args:
        rows: The per-run and per-model rows from the scoring.

    Returns:
        One row per run, scored set and measure, and one row for the fall
        between the injected residual class and the natural set.
    """
    summary: list[dict[str, Any]] = []
    keys = sorted({(str(row["run"]), str(row["scored_set"])) for row in rows})
    for run_name, scored_set in keys:
        cell = [
            row
            for row in rows
            if row["run"] == run_name and row["scored_set"] == scored_set
        ]
        for measure in ("balanced_accuracy", "youdens_j"):
            values = sorted(float(row[measure]) for row in cell)
            summary.append(
                {
                    "quantity": measure,
                    "run": run_name,
                    "scored_set": scored_set,
                    "models": len(values),
                    "median": round(statistics.median(values), 4),
                    "maximum": round(values[-1], 4),
                    "minimum": round(values[0], 4),
                    "below_chance": sum(
                        1
                        for value in values
                        if value < (0.5 if measure == "balanced_accuracy" else 0.0)
                    ),
                    "not_lower_than_source": "",
                }
            )
    before = {str(row["model"]): row for row in rows if row["run"] == FALL_FROM}
    for scored_set in sorted(
        {str(row["scored_set"]) for row in rows if row["run"] == FALL_TO}
    ):
        after = {
            str(row["model"]): row
            for row in rows
            if row["run"] == FALL_TO and row["scored_set"] == scored_set
        }
        shared = sorted(set(before) & set(after))
        if not shared:
            continue
        for measure in ("balanced_accuracy", "youdens_j"):
            falls = sorted(
                float(before[tag][measure]) - float(after[tag][measure])
                for tag in shared
            )
            summary.append(
                {
                    "quantity": f"fall_in_{measure}",
                    "run": f"{FALL_FROM} to {FALL_TO}",
                    "scored_set": scored_set,
                    "models": len(falls),
                    "median": round(statistics.median(falls), 4),
                    "maximum": round(falls[-1], 4),
                    "minimum": round(falls[0], 4),
                    "below_chance": "",
                    # A model that does not fall contradicts any sentence
                    # claiming the whole pool does, so the count is released.
                    "not_lower_than_source": sum(1 for fall in falls if fall <= 0),
                }
            )
    return summary


def write(tables: dict[str, list[dict[str, Any]]]) -> None:
    """Write the three tables and the provenance beside them.

    Args:
        tables: The tables from the scoring.
    """
    targets = {
        "rows": "model_scores.csv",
        "grouped": "model_scores_by_group.csv",
        "paired": "against_baseline.csv",
        "pairwise": "model_pairs.csv",
        "summary": "pool_summary.csv",
    }
    for key, name in targets.items():
        rows = tables[key]
        if not rows:
            continue
        # Only the natural-set rows carry the weighted estimates, so the header
        # is the union of what the rows hold rather than the first row's keys.
        fieldnames: list[str] = []
        for row in rows:
            fieldnames.extend(key for key in row if key not in fieldnames)
        target = out_dir() / name
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"wrote {target} with {len(rows)} rows")

    (out_dir() / "model_scores.json").write_text(
        json.dumps(
            {
                "scored_at": datetime.now(UTC).isoformat(),
                "confidence": CONFIDENCE,
                "interval": get("analysis.interval"),
                "detection_summary": get("analysis.detection_summary"),
                "paired_test": get("analysis.paired_test"),
                "runs_scored": sorted({str(row["run"]) for row in tables["rows"]}),
                "models_scored": sorted({str(row["model"]) for row in tables["rows"]}),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Score every answer on disk and write the tables."""
    tables = score_all()
    if not tables["rows"]:
        logger.warning("no answers on disk yet; nothing scored")
        return
    write(tables)
    logger.info(
        f"scored {len(tables['rows'])} run and model combinations over "
        f"{len(set(row['model'] for row in tables['rows']))} models"
    )


if __name__ == "__main__":
    main()
