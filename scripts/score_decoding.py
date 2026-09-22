"""Measure what the decoding constraint is worth, for the deposit.

The plan sends this whole analysis to the supplementary material and leaves one
line of it in the article, so nothing here is a table the reader of the article
has to absorb. It exists so that the one line is backed by a number anyone can
recompute.

Two runs answer the residual question with the JSON schema constraint suspended,
and each has a constrained twin asked in every other respect identically. Three
quantities come out of the comparison:

    How often the answer stopped satisfying the schema at all.

    What the score becomes under the rule this study fixed in advance, which is
    that an answer the schema rejects did not answer.

    What the score would become under the looser reading, which honours a
    verdict that can still be read out of a rejected answer. Many rejected
    answers are rejected for carrying an extra field beside a perfectly usable
    verdict, and the two readings differ by as much as a tenth of a point.

The strict reading is the study's, and it is the one the article reports. The
loose reading is here because the difference between them is a property of the
decoding constraint rather than a matter of opinion, and a reader is entitled to
see how much of the arm's result rests on where that line was drawn.

Outputs:
    data/processed/analysis/decoding_constraint.csv
    data/processed/analysis/decoding_constraint.json

Usage:
    python scripts/score_decoding.py
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from poi_audit import scoring
from poi_audit.config import path
from poi_audit.inference import planned_runs, pool, response_path
from poi_audit.logsetup import setup
from poi_audit.stats import Confusion

logger = setup("score_decoding")

# Each unconstrained run and the constrained run it is the twin of.
TWINS = {
    "m1_name_full_v1_unconstrained": "m1_name_full_v1",
    "natural_name_v1_unconstrained": "natural_name_v1",
}


def readable_verdict(answer: dict[str, Any]) -> str | None:
    """Return the verdict a rejected answer still carries, if it carries one.

    Args:
        answer: One answer as the runner wrote it.

    Returns:
        The verdict, or None when none can be read. This never changes how the
        study scores anything; it only measures how much the strict rule costs.
    """
    parsed = answer.get("parsed")
    if not isinstance(parsed, dict):
        return None
    verdict = parsed.get("verdict")
    return verdict if verdict in ("belongs", "wrong") else None


def count(
    answers: list[dict[str, Any]],
    truth: dict[str, dict[str, Any]],
    natural: bool,
    lenient: bool,
) -> Confusion:
    """Count one run's answers under one of the two readings.

    Args:
        answers: The answers as the runner wrote them.
        truth: The truth for this item set.
        natural: Whether the truth is the natural set's gold standard.
        lenient: Honour a verdict read out of a rejected answer.

    Returns:
        The confusion.
    """
    judged: list[scoring.Scored] = []
    for answer in answers:
        item_id = str(answer["item_id"])
        if natural:
            gold = truth[item_id]
            if not gold["scored"]:
                continue
            positive = gold["label"] == "wrong"
            city, group = str(gold["city"]), str(gold["stratum"])
        else:
            item = truth[item_id]
            positive = bool(item["corrupted"])
            city, group = str(item["city"]), str(item["class"])
        if lenient:
            flagged = readable_verdict(answer) == "wrong"
        else:
            parsed = answer.get("parsed") or {}
            flagged = bool(answer.get("valid")) and parsed.get("verdict") == "wrong"
        judged.append(
            scoring.Scored(item_id, city, group, positive, flagged, True, None)
        )
    return scoring.confusion(judged)


def build() -> list[dict[str, Any]]:
    """Compare the two readings, model by model, on both unconstrained runs.

    Returns:
        One row per run and model.
    """
    injected = scoring.injected_truth()
    gold = scoring.natural_truth()
    runs = {str(run["name"]): run for run in planned_runs()}
    rows: list[dict[str, Any]] = []
    for loose_run, tight_run in TWINS.items():
        natural = runs[loose_run]["items"] == "natural"
        truth = gold if natural else injected
        for card in pool():
            target = response_path(loose_run, card)
            twin = response_path(tight_run, card)
            if not target.exists() or not twin.exists():
                continue
            answers = scoring.read_answers(target)
            strict = count(answers, truth, natural, lenient=False)
            loose = count(answers, truth, natural, lenient=True)
            constrained = count(
                scoring.read_answers(twin), truth, natural, lenient=False
            )
            rejected = sum(1 for answer in answers if not answer.get("valid"))
            readable = sum(
                1
                for answer in answers
                if not answer.get("valid") and readable_verdict(answer)
            )
            rows.append(
                {
                    "run": loose_run,
                    "model": card.tag,
                    "answers": len(answers),
                    "schema_rejected": rejected,
                    "rejected_rate": round(rejected / len(answers), 4),
                    "rejected_but_readable": readable,
                    "j_strict": round(strict.youdens_j, 4),
                    "j_lenient": round(loose.youdens_j, 4),
                    "j_difference": round(loose.youdens_j - strict.youdens_j, 4),
                    "j_constrained_twin": round(constrained.youdens_j, 4),
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]], target: Path) -> None:
    """Write the pool-level figures the article's one line quotes.

    The line reports a median rejection rate over the pool and how many models
    still wrote a readable verdict inside a rejected answer, so both are
    released rather than left for a reader to compute from the table.

    Args:
        rows: The per-run and per-model comparison rows.
        target: The file the summary is written to.
    """
    summary = {}
    for run in sorted({str(row["run"]) for row in rows}):
        cell = [row for row in rows if row["run"] == run]
        rates = sorted(float(row["rejected_rate"]) for row in cell)
        summary[run] = {
            "models": len(cell),
            "median_rejected_rate": round(statistics.median(rates), 4),
            "maximum_rejected_rate": round(rates[-1], 4),
            "minimum_rejected_rate": round(rates[0], 4),
            "models_rejecting_every_answer": sum(1 for rate in rates if rate == 1.0),
            "models_with_a_readable_rejected_answer": sum(
                1 for row in cell if int(row["rejected_but_readable"]) > 0
            ),
        }
    target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info(f"wrote {target}")


def main() -> None:
    """Write the comparison the deposit carries."""
    rows = build()
    if not rows:
        logger.warning("no unconstrained answers on disk yet")
        return
    target = path("data_processed") / "analysis" / "decoding_constraint.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"wrote {target} with {len(rows)} rows")
    summarize(rows, target.with_suffix(".json"))
    worst = max(rows, key=lambda row: abs(row["j_difference"]))
    logger.info(
        f"the two readings differ most on {worst['model']} in {worst['run']}: "
        f"{worst['j_strict']} strict against {worst['j_lenient']} lenient"
    )


if __name__ == "__main__":
    main()
