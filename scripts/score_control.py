"""Score the residual control set, and choose the models the adaptation arms use.

Every run over the control set is scored per model on the whole set and on each
split, with Youden's J, its interval, recall, the false positive rate and the
flag rate beside it. The development split exists to choose the models the
retrieval-augmented arm runs on, by the rule fixed in the configuration before
any control answer existed: the best and the median deployable model by J on
the development split under prompt version 1, among the models the replay under
the configured serving release kept. The choice is written to disk before any
arm is run, and the article reads the test split.

Outputs:
    data/processed/analysis/control_scores.csv
    data/processed/analysis/adaptation_models.json
    data/processed/analysis/excluded_shown.csv       models the replay rule excluded

Usage:
    python -m scripts.score_control
"""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from poi_audit import scoring
from poi_audit.config import get, path
from poi_audit.inference import planned_runs, pool, response_path
from poi_audit.logsetup import setup
from poi_audit.stats import Confusion, exact_mcnemar

logger = setup("score_control")

SELECTION_RUN = "m1c_name_full_v1"
SELECTION_SPLIT = "dev"
REPLAY_RUN = "natural_name_v1_replay_0326"


def control_truth() -> dict[str, dict[str, Any]]:
    """Return the control set keyed by item."""
    source = path("data_processed") / "injected_set" / "residual_control.jsonl"
    with source.open(encoding="utf-8") as handle:
        return {
            str(row["item_id"]): row
            for row in (json.loads(line) for line in handle if line.strip())
        }


def measures(counts: Confusion) -> dict[str, Any]:
    """Return the detection measures every score row carries."""
    low, high = counts.youdens_j_interval()
    return {
        "items": counts.true_positive
        + counts.false_negative
        + counts.true_negative
        + counts.false_positive,
        "positives": counts.true_positive + counts.false_negative,
        "recall": round(counts.sensitivity().estimate, 4),
        "false_positive_rate": round(1.0 - counts.specificity().estimate, 4),
        "flag_rate": round(counts.flag_rate().estimate, 4),
        "balanced_accuracy": round(counts.balanced_accuracy, 4),
        "youdens_j": round(counts.youdens_j, 4),
        "youdens_j_low": round(low, 4),
        "youdens_j_high": round(high, 4),
    }


def select_models(scores: dict[str, float]) -> dict[str, str]:
    """Apply the configured rule: the best model and the median model by J.

    Models are ordered by J from highest to lowest, ties broken by tag. The
    median of an even count is the lower of the two middle models, the one at
    position n // 2 of that order, so the rule never picks a model better than
    the middle of the pool.

    Args:
        scores: Model tag to its J on the selection split.

    Returns:
        The best and the median model.

    Raises:
        ValueError: If no model is eligible.
    """
    if not scores:
        raise ValueError("no eligible model")
    order = sorted(scores, key=lambda tag: (-scores[tag], tag))
    return {"best": order[0], "median": order[len(order) // 2]}


def kept_models() -> set[str] | None:
    """Return the models the replay under the configured release kept, if measured."""
    source = path("data_processed") / "analysis" / f"replay_stability_{REPLAY_RUN}.json"
    if not source.exists():
        return None
    return set(json.loads(source.read_text(encoding="utf-8"))["kept"])


def deployable() -> set[str]:
    """Return the models the footprint profile measured as deployable."""
    source = path("data_processed") / "efficiency" / "footprint.json"
    profile = json.loads(source.read_text(encoding="utf-8"))
    return {
        str(row["tag"])
        for row in profile["models"]
        if row.get("measured_class") == "deployable"
    }


GATE_VIEWS = {"neither": "m1c_name_neither_v1", "name_only": "m1c_name_name_only_v1"}


def write_gates(truth: dict[str, dict[str, Any]]) -> None:
    """Write the input-dependence gate on the control test split, per kept model.

    The record is shown in full, then reduced to its position and to its name. A
    model whose score does not fall when the record is taken away was not reading
    it; the comparison is paired on the item by exact McNemar test.
    """
    excluded = set(get("run.replay_excluded_shown"))
    test = {item for item, row in truth.items() if row["split"] == "test"}
    rows: list[dict[str, Any]] = []
    for card in pool():
        if card.tag in excluded:
            continue
        full_path = response_path("m1c_name_full_v1", card)
        if not full_path.exists():
            continue
        full = [
            one
            for one in scoring.score_name_injected(
                scoring.read_answers(full_path), truth
            )
            if one.item_id in test
        ]
        full_ok = scoring.correctness(full)
        for view, run in GATE_VIEWS.items():
            target = response_path(run, card)
            if not target.exists():
                continue
            reduced = [
                one
                for one in scoring.score_name_injected(
                    scoring.read_answers(target), truth
                )
                if one.item_id in test
            ]
            first, second = scoring.discordant(full_ok, scoring.correctness(reduced))
            paired = exact_mcnemar(first, second)
            rows.append(
                {
                    "model": card.tag,
                    "name": card.name,
                    "view": view,
                    "full_youdens_j": round(scoring.confusion(full).youdens_j, 4),
                    "view_youdens_j": round(scoring.confusion(reduced).youdens_j, 4),
                    "view_flag_rate": round(
                        scoring.confusion(reduced).flag_rate().estimate, 4
                    ),
                    "full_only_correct": first,
                    "view_only_correct": second,
                    "p_value": paired.p_value,
                    # The gate is decided on the position-only view; the name-only
                    # view is reported beside it and decides nothing.
                    "passes": paired.p_value < 0.001 if view == "neither" else "",
                }
            )
    if rows:
        target = path("data_processed") / "analysis" / "control_gates.csv"
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"wrote {target}")


def main() -> None:
    """Score every control run on disk and write the model choice."""
    truth = control_truth()
    rows: list[dict[str, Any]] = []
    selection: dict[str, float] = {}
    runs = [
        run
        for run in planned_runs("redesign")
        if str(run["items"]).startswith("residual_control")
    ]
    excluded = set(get("run.replay_excluded_shown"))
    shown: list[dict[str, Any]] = []
    for run in runs:
        for card in pool():
            target: Path = response_path(str(run["name"]), card)
            if not target.exists():
                continue
            answers = scoring.read_answers(target)
            if len(answers) < len(truth):
                logger.warning(f"{run['name']} {card.tag}: {len(answers)} answers")
                continue
            judged = scoring.score_name_injected(answers, truth)
            for split in ("all", "train", "dev", "test"):
                part = [
                    one
                    for one in judged
                    if split == "all" or truth[one.item_id]["split"] == split
                ]
                counts = scoring.confusion(part)
                # A model the replay rule excluded is written apart and enters
                # neither the pooled rows nor the model rule.
                (shown if card.tag in excluded else rows).append(
                    {
                        "run": run["name"],
                        "model": card.tag,
                        "name": card.name,
                        "split": split,
                        "invalid": sum(1 for one in part if not one.valid),
                        **measures(counts),
                    }
                )
                if (
                    run["name"] == SELECTION_RUN
                    and split == SELECTION_SPLIT
                    and card.tag not in excluded
                ):
                    selection[card.tag] = counts.youdens_j

    out = path("data_processed") / "analysis"
    with (out / "control_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"wrote {len(rows)} score rows")
    if shown:
        with (out / "excluded_shown.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=[*shown[0], "excluded_by"])
            writer.writeheader()
            writer.writerows(
                {**row, "excluded_by": "replay_min_verdict_agreement"} for row in shown
            )

    write_gates(truth)

    kept = kept_models()
    eligible = {
        tag: score
        for tag, score in selection.items()
        if tag in deployable() and (kept is None or tag in kept)
    }
    if eligible:
        chosen = select_models(eligible)
        (out / "adaptation_models.json").write_text(
            json.dumps(
                {
                    "chosen_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "rule": get("retrieval_augmented.model_rule"),
                    "run": SELECTION_RUN,
                    "split": SELECTION_SPLIT,
                    "replay_filter_applied": kept is not None,
                    "eligible": {
                        tag: round(j, 4) for tag, j in sorted(eligible.items())
                    },
                    "chosen": chosen,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        logger.info(f"adaptation models: {chosen}")


if __name__ == "__main__":
    main()
