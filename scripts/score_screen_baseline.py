"""Score the screen's rules against the injected item set as a baseline.

The study's gate says a model must beat the best trivial answer available
rather than the first one written. The rule baseline is the answer scored
before any model runs, and no rule in it reads a name, so on the residual class
it is silent by construction. The screen's three rules do read names, they use
no model and no evaluated system, and they were written for a different job:
raising records for the natural set's enriched stratum. They were never scored
as an answer. This script scores them, because a trivial answer that exists in
the repository and is not scored is not a trivial answer that has been beaten.

Two properties of the comparison are stated with it rather than left to a
reader:

    The injected items were not selected by the screen, so nothing here is
    circular. On the natural set the screen chose stratum B and scoring it
    there would be, so the natural estimate is read on stratum A, the random
    draw the screen played no part in. Stratum B is reported beside it and is
    not the estimate.

    The duplicate rule reads every record of the item's city, and the gazetteer
    rule reads an open gazetteer. The models are shown one record. That
    asymmetry is the same one the reference lookup already carries, and it is
    recorded with the scores so that a reader knows what the baseline consumed.

Outputs:
    data/processed/analysis/screen_baseline.csv
    data/processed/analysis/screen_baseline.json
    data/processed/analysis/against_screen_baseline.csv
    data/processed/analysis/screen_baseline_natural.csv

Usage:
    python scripts/score_screen_baseline.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from poi_audit import scoring, screen
from poi_audit.config import get, path
from poi_audit.corpus import fold, read_corpus
from poi_audit.inference import planned_runs, pool, response_path
from poi_audit.logsetup import setup
from poi_audit.references import Place, load_gazetteer
from poi_audit.stats import Confusion, clustered_paired_difference, exact_mcnemar

logger = setup("score_screen_baseline")

ANY_RULE = "any_rule"


def item_path() -> Path:
    """Return the file the item builder writes."""
    return path("data_processed") / "injected_set" / "items.jsonl"


def read_items() -> list[dict[str, Any]]:
    """Read the injected item set.

    Returns:
        Every item.
    """
    with item_path().open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_category_tokens() -> dict[str, str]:
    """Read the category-indicative token dictionary the screen reads.

    Returns:
        Token to the category it indicates.
    """
    return screen.load_category_tokens()


def as_screen_record(item: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    """Return the item's corpus record carrying the name the item shows.

    The screen reads tags the item does not carry: the brand tag its exclusions
    turn on, and the primary category key. Those come from the corpus record the
    item was built from, and the shown name is written over it, so that a
    corrupted item is scored on the borrowed name and a clean one on its own.

    Args:
        item: One item of the injected set.
        base: The corpus record the item was built from.

    Returns:
        A corpus-shaped record carrying the item's name.
    """
    record = dict(base)
    record["tags"] = dict(base["tags"])
    name = str(item["record"].get("name", ""))
    record["name"] = name
    record["tags"]["name"] = name
    return record


def score(
    items: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    tokens: dict[str, str],
    places: dict[str, list[Place]],
) -> dict[str, Any]:
    """Run every screen rule over every item and tally the outcomes.

    Args:
        items: The injected item set.
        corpus: Every corpus record the items were drawn from.
        tokens: Category-indicative token to the category it indicates.
        places: Folded place name to the places holding it.

    Returns:
        Per rule and per class tallies, and the per-item verdicts.
    """
    by_city: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_identity: dict[tuple[str, int], dict[str, Any]] = {}
    for record in corpus:
        by_city[str(record["city"])].append(record)
        by_identity[screen.identity(record)] = record
    indexes = {city: screen.name_index(rows) for city, rows in by_city.items()}

    tallies: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    detail: list[dict[str, Any]] = []
    for item in items:
        base = by_identity.get((str(item["osm_type"]), int(item["osm_id"])))
        if base is None:
            raise SystemExit(
                f"item {item['item_id']} names a record the corpus does not hold"
            )
        record = as_screen_record(item, base)
        same_name = list(
            indexes.get(str(item["city"]), {}).get(fold(record["name"]), [])
        )
        if not any(
            screen.identity(one) == screen.identity(record) for one in same_name
        ):
            same_name.append(record)
        hits = screen.screen_record(record, same_name, tokens, places)
        raised = {hit.rule for hit in hits}

        for rule in [*screen.RULE_ORDER, ANY_RULE]:
            flagged = bool(raised) if rule == ANY_RULE else rule in raised
            key = (rule, str(item["class"]))
            tallies[key]["corrupted" if item["corrupted"] else "clean"] += 1
            if flagged:
                tallies[key][
                    "corrupted_flagged" if item["corrupted"] else "clean_flagged"
                ] += 1
        detail.append(
            {
                "item_id": item["item_id"],
                "class": item["class"],
                "corrupted": item["corrupted"],
                "rules": sorted(raised),
            }
        )
    return {
        "by_rule": summarize(tallies, by_class=False),
        "by_rule_and_class": summarize(tallies, by_class=True),
        "detail": detail,
    }


def summarize(
    tallies: dict[tuple[str, str], dict[str, int]], by_class: bool
) -> dict[str, Any]:
    """Turn tallies into scores with their intervals.

    Args:
        tallies: Rule and class to the four counts.
        by_class: Whether to keep the classes apart or pool them.

    Returns:
        Rule (or rule and class) to its confusion counts and scores.
    """
    pooled: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for (rule, klass), counts in tallies.items():
        key = f"{rule}|{klass}" if by_class else rule
        for name, value in counts.items():
            pooled[key][name] += value

    scored: dict[str, Any] = {}
    for key, counts in pooled.items():
        confusion = Confusion(
            true_positive=counts.get("corrupted_flagged", 0),
            false_negative=counts.get("corrupted", 0)
            - counts.get("corrupted_flagged", 0),
            true_negative=counts.get("clean", 0) - counts.get("clean_flagged", 0),
            false_positive=counts.get("clean_flagged", 0),
        )
        recall = confusion.sensitivity()
        specificity = confusion.specificity()
        j_low, j_high = confusion.youdens_j_interval()
        scored[key] = {
            "true_positive": confusion.true_positive,
            "false_negative": confusion.false_negative,
            "true_negative": confusion.true_negative,
            "false_positive": confusion.false_positive,
            "recall": round(recall.estimate, 4),
            "recall_low": round(recall.low, 4),
            "recall_high": round(recall.high, 4),
            "false_positive_rate": round(1.0 - specificity.estimate, 4),
            "balanced_accuracy": round(confusion.balanced_accuracy, 4),
            "youdens_j": round(confusion.youdens_j, 4),
            "youdens_j_low": round(j_low, 4),
            "youdens_j_high": round(j_high, 4),
        }
    return scored


def correctness(scored: dict[str, Any], rule: str) -> dict[str, bool]:
    """Return whether one rule answered each item correctly, keyed by item.

    The gate is paired on the record, since two systems answering the same
    items are compared on where they disagreed rather than between two summary
    rates.

    Args:
        scored: The tallies and verdicts the run produced.
        rule: The rule to read, or ``any_rule`` for the union.

    Returns:
        Item identifier to whether the rule's verdict matched the truth.
    """
    right: dict[str, bool] = {}
    for one in scored["detail"]:
        raised = bool(one["rules"]) if rule == ANY_RULE else rule in one["rules"]
        right[str(one["item_id"])] = raised == bool(one["corrupted"])
    return right


def write_outputs(scored: dict[str, Any]) -> None:
    """Write the scores where the analysis reads them.

    Args:
        scored: The tallies and scores the run produced.
    """
    out = path("data_processed") / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    fields = [
        "rule",
        "class",
        "true_positive",
        "false_negative",
        "true_negative",
        "false_positive",
        "recall",
        "recall_low",
        "recall_high",
        "false_positive_rate",
        "balanced_accuracy",
        "youdens_j",
        "youdens_j_low",
        "youdens_j_high",
    ]
    with (out / "screen_baseline.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rule, row in sorted(scored["by_rule"].items()):
            writer.writerow({"rule": rule, "class": "all", **row})
        for key, row in sorted(scored["by_rule_and_class"].items()):
            rule, klass = key.split("|")
            writer.writerow({"rule": rule, "class": klass, **row})

    summary = {
        "by_rule": scored["by_rule"],
        "by_rule_and_class": scored["by_rule_and_class"],
        "evidence_the_baseline_reads": {
            "city_corpus": True,
            "open_gazetteer": True,
            "note": (
                "The duplicate rule reads every record of the item's city and "
                "the gazetteer rule reads an open gazetteer, neither of which "
                "the models are shown. The asymmetry is the reference lookup's "
                "and is stated rather than repaired."
            ),
        },
        "selection": (
            "The injected items were not selected by the screen, so the "
            "comparison is not circular. The screen is not scored on the "
            "natural set, half of which it selected."
        ),
    }
    (out / "screen_baseline.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("wrote {}", out / "screen_baseline.csv")


def score_natural(
    records: list[dict[str, Any]], labels: list[dict[str, Any]], rule: str
) -> dict[str, Any]:
    """Score one screen rule against the natural set's labels.

    The verdicts are the ones recorded at the draw rather than recomputed, so
    that the rule is read exactly as it was run over the corpus. Stratum A is
    the estimate: it is a random draw from the frame and the screen played no
    part in it. Stratum B was selected by the screen and is reported apart.

    Args:
        records: The natural set as the draw wrote it.
        labels: The gold labels.
        rule: The rule to read, or ``any_rule`` for the union.

    Returns:
        One confusion and its scores per stratum.
    """
    decided = {
        str(one["item_id"]): str(one["label"])
        for one in labels
        if str(one["label"]) in {"belongs", "wrong"}
    }
    tallies: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for record in records:
        label = decided.get(str(record["item_id"]))
        if label is None:
            continue
        raised = (
            bool(record["screen_rules"])
            if rule == ANY_RULE
            else rule in record["screen_rules"]
        )
        key = (rule, f"stratum_{str(record['stratum']).lower()}")
        tallies[key]["corrupted" if label == "wrong" else "clean"] += 1
        if raised:
            tallies[key][
                "corrupted_flagged" if label == "wrong" else "clean_flagged"
            ] += 1
    scored = summarize(tallies, by_class=True)
    return {key.split("|")[1]: value for key, value in scored.items()}


def write_natural(scored: dict[str, Any], rule: str) -> None:
    """Write the natural-set scores where the analysis reads them.

    Args:
        scored: One row per stratum.
        rule: The rule the rows were scored under.
    """
    out = path("data_processed") / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    fields = [
        "rule",
        "stratum",
        "true_positive",
        "false_negative",
        "true_negative",
        "false_positive",
        "recall",
        "recall_low",
        "recall_high",
        "false_positive_rate",
        "balanced_accuracy",
        "youdens_j",
        "youdens_j_low",
        "youdens_j_high",
    ]
    target = out / "screen_baseline_natural.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for stratum, row in sorted(scored.items()):
            writer.writerow({"rule": rule, "stratum": stratum, **row})
    logger.info("wrote {}", target)


def compare_to_models(scored: dict[str, Any], rule: str) -> list[dict[str, Any]]:
    """Pair every model's residual-class answers against the screen baseline.

    This is the baseline-dominance gate the study declares and applies to the
    detection task alone. On the residual class the rule baseline reads no name
    and is silent, so the gate was never binding there; the screen's duplicate
    rule is the answer that has to be beaten instead.

    Args:
        scored: The tallies and verdicts the screen baseline produced.
        rule: The rule to compare against.

    Returns:
        One row per model and run, in the shape the rule baseline's comparison
        writes.
    """
    from scripts.score_models import judge

    right = correctness(scored, rule)
    injected = scoring.injected_truth()
    natural = scoring.natural_truth()
    units = scoring.clusters()

    rows: list[dict[str, Any]] = []
    for run in planned_runs():
        if run["items"] == "natural" or run["task"] == "detect":
            continue
        for card in pool():
            target = response_path(str(run["name"]), card)
            if not target.exists():
                continue
            answers = scoring.read_answers(target)
            if not answers:
                continue
            judged = judge(run, answers, injected, natural)
            first, second = scoring.discordant(scoring.correctness(judged), right)
            result = exact_mcnemar(first, second)
            spread = clustered_paired_difference(
                scoring.correctness(judged), right, units
            )
            rows.append(
                {
                    "run": run["name"],
                    "model": card.tag,
                    "baseline_rule": rule,
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
    return rows


CONTROL_RUN = "m1c_name_full_v1"
NATURAL_RUN = "natural_name_v1"
STRATUM_A = "a"


def control_screen_correctness() -> dict[str, bool]:
    """Return whether the screen answered each control test item correctly.

    The control set is read the way every instrument on it is read: against the
    visible part of each city alone, which is what the construction gives an
    instrument that consults the city's records.

    Returns:
        Item identifier to whether the screen's union of rules was right.
    """
    base = path("data_processed") / "injected_set"
    with (base / "residual_control.jsonl").open(encoding="utf-8") as handle:
        items = [json.loads(line) for line in handle if line.strip()]
    visible = json.loads(
        (base / "residual_control_visible.json").read_text(encoding="utf-8")
    )
    identities = {
        (str(kind), int(number))
        for part in visible.values()
        for kind, number in part["records"]
    }
    corpus = [
        record for record in read_corpus() if screen.identity(record) in identities
    ]
    test = [item for item in items if item["split"] == "test"]
    scored = score(
        test, corpus, read_category_tokens(), screen.place_index(load_gazetteer())
    )
    return {
        str(row["item_id"]): bool(row["rules"]) == bool(row["corrupted"])
        for row in scored["detail"]
    }


def natural_screen_correctness() -> dict[str, bool]:
    """Return whether the screen answered each stratum A record correctly.

    Stratum A is the estimate because the screen played no part in drawing it.
    Stratum B was selected by the screen, and pairing a model against it there
    would ask the screen to beat its own selection.

    Returns:
        Item identifier to whether the screen's verdict matched the gold label.
    """
    natural = path("data_processed") / "natural_set"
    records = [
        json.loads(line)
        for line in (natural / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    truth = scoring.natural_truth()
    right: dict[str, bool] = {}
    for record in records:
        item = str(record["item_id"])
        gold = truth.get(item)
        if gold is None or not gold["scored"]:
            continue
        if str(gold["stratum"]).lower() != STRATUM_A:
            continue
        right[item] = bool(record["screen_rules"]) == bool(gold["label"] == "wrong")
    return right


def compare_on_residual_sets() -> list[dict[str, Any]]:
    """Pair every model against the screen on the two sets the conclusion rests on.

    The baseline-dominance gate is declared for every task and was computed for
    the detection task alone. The article reads a capability from the residual
    question, which is answered on the control set's test items and on the
    natural set, so the gate is computed there too. The instrument compared
    against is the screen's three rules taken together, which is the cheapest
    answer to the residual question that exists in this repository: it reads
    names, uses no model, and was written for another job. The format rule is
    not the comparison here, since it reads no name and is silent on this class
    by construction.

    Returns:
        One row per model and set, carrying the exact McNemar of the discordant
        pairs and whether the model beat the screen at the study's threshold.
    """
    threshold = float(get("analysis.significance_threshold"))
    control_right = control_screen_correctness()
    natural_right = natural_screen_correctness()
    control_truth = {
        str(row["item_id"]): row
        for row in (
            json.loads(line)
            for line in (
                path("data_processed") / "injected_set" / "residual_control.jsonl"
            )
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    }
    natural_truth = scoring.natural_truth()

    rows: list[dict[str, Any]] = []
    for card in pool():
        for task, run, right in (
            ("control_test", CONTROL_RUN, control_right),
            ("natural_stratum_a", NATURAL_RUN, natural_right),
        ):
            target = response_path(run, card)
            if not target.exists():
                continue
            answers = scoring.read_answers(target)
            if not answers:
                continue
            if task == "control_test":
                judged = scoring.score_name_injected(answers, control_truth)
            else:
                judged = scoring.score_name_natural(answers, natural_truth)
            model_right = {
                item: value
                for item, value in scoring.correctness(judged).items()
                if item in right
            }
            first, second = scoring.discordant(model_right, right)
            result = exact_mcnemar(first, second)
            rows.append(
                {
                    "task": task,
                    "run": run,
                    "model": card.tag,
                    "baseline_rule": ANY_RULE,
                    "items": len(model_right),
                    "model_only_correct": result.first_only,
                    "baseline_only_correct": result.second_only,
                    "p_value": result.p_value,
                    "odds_ratio": result.odds_ratio,
                    "odds_ratio_low": result.odds_ratio_low,
                    "odds_ratio_high": result.odds_ratio_high,
                    "passes": bool(
                        result.first_only > result.second_only
                        and result.p_value < threshold
                    ),
                }
            )
    return rows


def write_residual_comparison(rows: list[dict[str, Any]]) -> None:
    """Write the residual-question comparisons where the gates read them.

    Args:
        rows: One row per model and set.
    """
    out = path("data_processed") / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "against_screen_residual.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote {}", target)


def write_comparison(rows: list[dict[str, Any]]) -> None:
    """Write the paired comparisons where the analysis reads them.

    Args:
        rows: One row per model and run.
    """
    out = path("data_processed") / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    fields = [
        "run",
        "model",
        "baseline_rule",
        "model_only_correct",
        "baseline_only_correct",
        "p_value",
        "odds_ratio",
        "odds_ratio_low",
        "odds_ratio_high",
        "accuracy_difference",
        "difference_low",
        "difference_high",
        "clustered_p_value",
        "clusters",
    ]
    target = out / "against_screen_baseline.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote {}", target)


def main() -> None:
    """Command line entry point."""
    items = read_items()
    corpus = list(read_corpus())
    tokens = read_category_tokens()
    places = screen.place_index(load_gazetteer())
    scored = score(items, corpus, tokens, places)
    write_outputs(scored)
    write_comparison(compare_to_models(scored, screen.DUPLICATE_DISTANT_NAME))
    write_residual_comparison(compare_on_residual_sets())

    natural = path("data_processed") / "natural_set"
    records = [
        json.loads(line)
        for line in (natural / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels = [
        json.loads(line)
        for line in (natural / "gold.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    on_natural = score_natural(records, labels, screen.DUPLICATE_DISTANT_NAME)
    write_natural(on_natural, screen.DUPLICATE_DISTANT_NAME)
    for stratum, row in sorted(on_natural.items()):
        logger.info(
            "{}: J {} (recall {}, {} wrong of {} decided)",
            stratum,
            row["youdens_j"],
            row["recall"],
            row["true_positive"] + row["false_negative"],
            row["true_positive"]
            + row["false_negative"]
            + row["true_negative"]
            + row["false_positive"],
        )
    for rule, row in sorted(scored["by_rule"].items()):
        logger.info(
            "{}: J {} (recall {}, false positive rate {})",
            rule,
            row["youdens_j"],
            row["recall"],
            row["false_positive_rate"],
        )


if __name__ == "__main__":
    main()
