"""Read the natural set under three label rules, and the annotators' agreement.

The natural set was labelled by two annotators, one of whom is the author, and
their disagreements were adjudicated. A result that held only under the
adjudicated labels would rest in part on the author's judgement. Every system is
therefore scored under three rules fixed in the configuration: the adjudicated
gold label; the decidable core, where both annotators independently gave the
same definite label; and the non-author annotator's own labels, which no
decision of the author touches. Agreement is reported as Cohen's kappa and
Gwet's AC1, the second because kappa collapses when one label is rare.

Outputs:
    data/processed/analysis/label_sensitivity.csv
    data/processed/analysis/annotator_agreement.json

Usage:
    python -m scripts.score_label_sensitivity
"""

from __future__ import annotations

import csv
import json
from typing import Any

from poi_audit import scoring, stats
from poi_audit.config import get, path
from poi_audit.logsetup import setup
from poi_audit.stats import Confusion

logger = setup("score_label_sensitivity")

DEFINITE = {"wrong", "belongs"}
RUNS = ("natural_name_v1", "natural_name_v1_replay_0326")


def truth_for(rule: str, gold: dict[str, dict[str, Any]]) -> dict[str, bool | None]:
    """Return each item's truth under one label rule, None where the rule gives none.

    Args:
        rule: ``adjudicated``, ``decidable_core`` or ``non_author``.
        gold: The gold rows keyed by item, carrying each annotator's label.

    Returns:
        Item to whether its name is wrong, or None where the rule leaves the item
        undecided.

    Raises:
        ValueError: If the rule is unknown.
    """
    non_author = str(get("natural_set.labelling.non_author_annotator"))
    truth: dict[str, bool | None] = {}
    for item, row in gold.items():
        if rule == "adjudicated":
            label = row["label"] if row["scored"] else None
        elif rule == "decidable_core":
            label = row["label"] if row["decidable_core"] else None
        elif rule == "non_author":
            label = row[f"label_{non_author}"]
        else:
            raise ValueError(f"unknown label rule: {rule}")
        truth[item] = (label == "wrong") if label in DEFINITE else None
    return truth


def score(answers: list[dict[str, Any]], truth: dict[str, bool | None]) -> Confusion:
    """Count one system's answers against one rule's decided items."""
    tp = fn = tn = fp = 0
    for answer in answers:
        positive = truth.get(str(answer["item_id"]))
        if positive is None:
            continue
        flagged = (
            bool(answer.get("valid"))
            and (answer.get("parsed") or {}).get("verdict") == "wrong"
        )
        if positive and flagged:
            tp += 1
        elif positive:
            fn += 1
        elif flagged:
            fp += 1
        else:
            tn += 1
    return Confusion(tp, fn, tn, fp)


def agreement(gold: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Return the two annotators' agreement, binary and over all three labels."""
    author = str(get("natural_set.labelling.author_annotator"))
    other = str(get("natural_set.labelling.non_author_annotator"))
    everything = [
        [row[f"label_{author}"], row[f"label_{other}"]] for row in gold.values()
    ]
    decided = [pair for pair in everything if set(pair) <= DEFINITE]
    result: dict[str, Any] = {}
    for name, ratings, categories in (
        ("all_three_labels", everything, ("wrong", "belongs", "cannot_say")),
        ("both_definite", decided, ("wrong", "belongs")),
    ):
        kappa = stats.cohen_kappa(
            [a for a, _ in ratings], [b for _, b in ratings], categories
        )
        low, high = stats.bootstrap_interval(
            ratings, categories, stats.gwet_ac1, replicates=2000, seed=42
        )
        result[name] = {
            "items": len(ratings),
            "observed_agreement": round(kappa.observed, 4),
            "cohen_kappa": round(kappa.kappa, 4),
            "cohen_kappa_low": round(kappa.low, 4),
            "cohen_kappa_high": round(kappa.high, 4),
            "gwet_ac1": round(stats.gwet_ac1(ratings, categories), 4),
            "gwet_ac1_low": round(low, 4),
            "gwet_ac1_high": round(high, 4),
        }
    result["wrong_labels"] = {
        author: sum(1 for row in gold.values() if row[f"label_{author}"] == "wrong"),
        other: sum(1 for row in gold.values() if row[f"label_{other}"] == "wrong"),
    }
    return result


def main() -> None:
    """Score every system under every rule and write the agreement."""
    gold = scoring.natural_truth()
    rows: list[dict[str, Any]] = []
    for rule in get("natural_set.label_rules"):
        truth = truth_for(str(rule), gold)
        for run in RUNS:
            for target in sorted((path("responses") / run).glob("*.jsonl")):
                counts = score(scoring.read_answers(target), truth)
                low, high = counts.youdens_j_interval()
                rows.append(
                    {
                        "rule": rule,
                        "run": run,
                        "system": target.stem,
                        "items": counts.total,
                        "positives": counts.true_positive + counts.false_negative,
                        "recall": round(counts.sensitivity().estimate, 4),
                        "flag_rate": round(counts.flag_rate().estimate, 4),
                        "youdens_j": round(counts.youdens_j, 4),
                        "youdens_j_low": round(low, 4),
                        "youdens_j_high": round(high, 4),
                    }
                )
    out = path("data_processed") / "analysis"
    with (out / "label_sensitivity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "annotator_agreement.json").write_text(
        json.dumps(agreement(gold), indent=2) + "\n", encoding="utf-8"
    )
    logger.info(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
