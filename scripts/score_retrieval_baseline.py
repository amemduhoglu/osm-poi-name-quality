"""Score the dense-retrieval comparator, the editor's requested baseline.

The comparator embeds a record's name and the display names of the gazetteer
places near its coordinate, and flags the record when the best cosine score
reached falls below a threshold. It passes the study's four gates the way
every model does, against the same item sets, and three quantities come out
of scoring it.

    The residual class on the injected set, under a threshold calibrated on a
    split the reported score never saw. This is the comparator's own answer:
    the threshold is chosen once, blind to the records it is then measured on,
    which is what keeps the reported recall from being chosen after the fact.

    The same records under the oracle threshold, the best any single cut could
    reach on the exact data being scored. It is computed on the reported split
    itself rather than on the calibration split, so it is not a candidate for
    the comparator's own cut and is labelled an upper bound rather than a
    second score.

    The natural set's adjudicated records, on both thresholds. Stratum A is
    the random draw the screen played no part in; stratum B was selected by
    the screen, and scoring it there would be circular, which is the same
    reading scripts/score_screen_baseline.py applies to its own comparison.
    Both strata are written so the choice is visible in the data. Abstentions,
    the records whose neighbourhood held no candidate at all, are counted
    apart from the negatives: a lookup that found nothing has not cleared a
    record and has not condemned it.

The paired comparison against the rule baseline, by exact McNemar test with
the odds ratio of the discordant pairs, is written beside the scores. The
rule baseline reads no name and is silent on this class by construction, so
the comparison exists to show whether reading the gazetteer moved the answer
at all.

Outputs:
    data/processed/analysis/retrieval_baseline.csv
    data/processed/analysis/retrieval_baseline_pairs.csv

Usage:
    python scripts/score_retrieval_baseline.py
    python scripts/score_retrieval_baseline.py --limit 20   # a rehearsal
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

from poi_audit import scoring
from poi_audit.config import get, path, secret
from poi_audit.logsetup import setup
from poi_audit.references import Gazetteer, Place, load_gazetteer
from poi_audit.retrieval import Encoder, best_match, decide
from poi_audit.stats import Confusion, exact_mcnemar

logger = setup("score_retrieval_baseline")

# The class the article rests on: the name of another place, which no rule in
# the baseline reads. The code is the data's and the configuration's
# vocabulary, and the manuscript never prints it.
RESIDUAL_CLASS = "M1"

SCORE_FIELDS = [
    "set",
    "split",
    "threshold_kind",
    "threshold",
    "n",
    "flagged",
    "recall",
    "recall_low",
    "recall_high",
    "balanced_accuracy",
    "youden_j",
    "youden_j_low",
    "youden_j_high",
    "abstentions",
]

PAIR_FIELDS = [
    "set",
    "split",
    "threshold_kind",
    "comparator_only_correct",
    "baseline_only_correct",
    "p_value",
    "odds_ratio",
    "odds_ratio_low",
    "odds_ratio_high",
]


def item_path() -> Path:
    """Return the file the item builder writes the injected set to."""
    return path("data_processed") / "injected_set" / "items.jsonl"


def read_items() -> list[dict[str, Any]]:
    """Read the injected item set.

    Returns:
        Every item.

    Raises:
        FileNotFoundError: If the set has not been built.
    """
    target = item_path()
    if not target.exists():
        raise FileNotFoundError(
            f"item set not found: {target}; run scripts/build_injected_set.py"
        )
    with target.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def as_retrieval_record(item: dict[str, Any]) -> dict[str, Any]:
    """Return one injected item in the shape the comparator scores.

    Args:
        item: One item of the injected residual class.

    Returns:
        Its identifier, name, coordinate and whether it was corrupted.
    """
    record = item["record"]
    return {
        "item_id": str(item["item_id"]),
        "name": str(record["name"]),
        "lat": float(record["lat"]),
        "lon": float(record["lon"]),
        "corrupted": bool(item["corrupted"]),
    }


def natural_set_path() -> Path:
    """Return the directory the natural set draw writes to."""
    return path("data_processed") / "natural_set"


def read_natural_records() -> list[dict[str, Any]]:
    """Read the natural set's draw.

    Returns:
        Every drawn record.

    Raises:
        FileNotFoundError: If the set has not been drawn.
    """
    target = natural_set_path() / "records.jsonl"
    if not target.exists():
        raise FileNotFoundError(
            f"natural set not found: {target}; run scripts/draw_natural_set.py"
        )
    with target.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_natural_labels() -> list[dict[str, Any]]:
    """Read the natural set's gold labels.

    Returns:
        Every gold row.

    Raises:
        FileNotFoundError: If the labels have not been built.
    """
    target = natural_set_path() / "gold.jsonl"
    if not target.exists():
        raise FileNotFoundError(
            f"gold labels not built: {target}; run scripts/build_gold_labels.py"
        )
    with target.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def adjudicated_natural_records(
    records: list[dict[str, Any]], labels: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Return the natural set's scored records, kept apart by stratum.

    An item the two readers left undecided carries no truth to score against
    and is dropped, the same rule ``poi_audit.scoring.score_name_natural``
    applies. Stratum A is the random draw; stratum B was selected by the
    screen, so an estimate read there would be circular.

    Args:
        records: The natural set as the draw wrote it.
        labels: The gold labels.

    Returns:
        Stratum key to the records it decided, in the shape the comparator
        scores.
    """
    decided = {str(row["item_id"]): row for row in labels if row["scored"]}
    by_stratum: dict[str, list[dict[str, Any]]] = {"stratum_a": [], "stratum_b": []}
    for record in records:
        gold = decided.get(str(record["item_id"]))
        if gold is None:
            continue
        key = f"stratum_{str(record['stratum']).lower()}"
        if key not in by_stratum:
            continue
        shown = record["shown"]
        by_stratum[key].append(
            {
                "item_id": str(record["item_id"]),
                "name": str(shown["name"]),
                "lat": float(shown["lat"]),
                "lon": float(shown["lon"]),
                "corrupted": gold["label"] == scoring.WRONG,
            }
        )
    return by_stratum


def split(
    items: list[dict[str, Any]], share: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records into a calibration share and the rest.

    A threshold chosen on the records it is then scored on is not a result,
    so the draw is made once, before any score exists, on the item identifier
    alone.

    Args:
        items: The records to split, each carrying ``item_id``.
        share: The share drawn for calibration.
        seed: The seed the draw is fixed under.

    Returns:
        The calibration records and the reported records, disjoint and
        together covering every item given.
    """
    ordered = sorted(items, key=lambda item: str(item["item_id"]))
    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    cut = round(len(shuffled) * share)
    calibration_ids = {str(item["item_id"]) for item in shuffled[:cut]}
    calibration = [item for item in ordered if str(item["item_id"]) in calibration_ids]
    reported = [item for item in ordered if str(item["item_id"]) not in calibration_ids]
    return calibration, reported


def oracle_threshold(scores: list[float], truth: list[bool]) -> tuple[float, float]:
    """Return the threshold reaching the highest Youden's J on the given scores.

    This is not a candidate for the comparator's own cut: it is computed on
    the same data it is then read against, so it names the best any single
    threshold could do on that data rather than what a blind cut in fact
    achieves. Used on a calibration split and then applied to different
    records, the same routine is the calibration itself; used on the records
    it is then reported against, it is the upper bound.

    Args:
        scores: The best cosine score reached, one per record, none of them
            an abstention.
        truth: Whether each record was actually corrupted, same order.

    Returns:
        The threshold and the Youden's J it reaches. Ties favour the higher
        threshold, which flags less of the clean population for the same
        recall.

    Raises:
        ValueError: If no score was given.
    """
    if not scores:
        raise ValueError("no score to choose a threshold from")
    best_threshold = min(scores)
    best_j = -1.0
    for candidate in sorted(set(scores)):
        confusion = Confusion(
            true_positive=sum(
                1 for s, t in zip(scores, truth, strict=True) if t and s < candidate
            ),
            false_negative=sum(
                1 for s, t in zip(scores, truth, strict=True) if t and not s < candidate
            ),
            true_negative=sum(
                1
                for s, t in zip(scores, truth, strict=True)
                if not t and not s < candidate
            ),
            false_positive=sum(
                1 for s, t in zip(scores, truth, strict=True) if not t and s < candidate
            ),
        )
        j = confusion.youdens_j
        if j >= best_j:
            best_j = j
            best_threshold = candidate
    return best_threshold, best_j


async def score_records(
    records: list[dict[str, Any]],
    gazetteer: Gazetteer,
    encoder: Encoder,
    radius_km: float,
) -> dict[str, float]:
    """Score every record's name against its gazetteer neighbourhood.

    Every text the pass needs, the records' own names and every candidate's
    display name, is embedded once in one call, since the same text embeds to
    the same vector and a second call would only pay for it twice.

    Args:
        records: Records carrying ``item_id``, ``name``, ``lat`` and ``lon``.
        gazetteer: The loaded gazetteer.
        encoder: The embedding client.
        radius_km: The radius the candidates are drawn from.

    Returns:
        Item identifier to the best cosine score reached, ``nan`` where the
        neighbourhood held no candidate at all.
    """
    candidates: dict[str, list[Place]] = {}
    texts: set[str] = set()
    for record in records:
        nearby = gazetteer.nearby(record["lat"], record["lon"], radius_km)
        candidates[record["item_id"]] = nearby
        texts.add(record["name"])
        texts.update(place.display_name for place in nearby)

    ordered = sorted(texts)
    embedded = await encoder.embed(ordered) if ordered else []
    vectors = dict(zip(ordered, embedded, strict=True))

    scores: dict[str, float] = {}
    for record in records:
        score, _ = best_match(record["name"], candidates[record["item_id"]], vectors)
        scores[record["item_id"]] = score
    return scores


def confusion_at(
    scores: dict[str, float], truth: dict[str, bool], threshold: float
) -> tuple[Confusion, int]:
    """Count one threshold's verdicts over a scored population.

    An abstention, a ``nan`` score from an empty neighbourhood, is counted
    apart rather than as a negative: the comparator has not cleared that
    record and has not condemned it.

    Args:
        scores: Item identifier to its best cosine score.
        truth: Item identifier to whether it was actually corrupted.
        threshold: The calibrated or oracle cut.

    Returns:
        The confusion over the decided records, and how many abstained.
    """
    true_positive = false_negative = true_negative = false_positive = 0
    abstentions = 0
    for item_id, score in scores.items():
        if math.isnan(score):
            abstentions += 1
            continue
        flagged = decide(score, threshold)
        positive = truth[item_id]
        if positive and flagged:
            true_positive += 1
        elif positive:
            false_negative += 1
        elif flagged:
            false_positive += 1
        else:
            true_negative += 1
    return (
        Confusion(true_positive, false_negative, true_negative, false_positive),
        abstentions,
    )


def score_row(
    set_name: str,
    split_name: str,
    threshold_kind: str,
    threshold: float,
    scores: dict[str, float],
    truth: dict[str, bool],
) -> dict[str, Any]:
    """Build one row of the comparator's scored output.

    Args:
        set_name: ``injected`` or ``natural``.
        split_name: The split or stratum the row was scored on.
        threshold_kind: ``calibrated`` or ``oracle``.
        threshold: The cut applied.
        scores: Item identifier to its best cosine score, over this row's
            population.
        truth: Item identifier to whether it was actually corrupted.

    Returns:
        One row in the shape ``retrieval_baseline.csv`` writes.
    """
    confusion, abstentions = confusion_at(scores, truth, threshold)
    recall = confusion.sensitivity()
    j_low, j_high = confusion.youdens_j_interval()
    return {
        "set": set_name,
        "split": split_name,
        "threshold_kind": threshold_kind,
        "threshold": round(threshold, 4),
        "n": confusion.total,
        "flagged": confusion.true_positive + confusion.false_positive,
        "recall": round(recall.estimate, 4),
        "recall_low": round(recall.low, 4),
        "recall_high": round(recall.high, 4),
        "balanced_accuracy": round(confusion.balanced_accuracy, 4),
        "youden_j": round(confusion.youdens_j, 4),
        "youden_j_low": round(j_low, 4),
        "youden_j_high": round(j_high, 4),
        "abstentions": abstentions,
    }


def retrieval_correctness(
    scores: dict[str, float], truth: dict[str, bool], threshold: float
) -> dict[str, bool]:
    """Return whether the comparator's verdict matched the truth, per item.

    An abstention answered nothing and is left out, which is what a paired
    comparison already does with an item a system never answered.

    Args:
        scores: Item identifier to its best cosine score.
        truth: Item identifier to whether it was actually corrupted.
        threshold: The cut applied.

    Returns:
        Item identifier to whether the verdict was correct.
    """
    correct: dict[str, bool] = {}
    for item_id, score in scores.items():
        if math.isnan(score):
            continue
        correct[item_id] = decide(score, threshold) == truth[item_id]
    return correct


def paired_row(
    set_name: str,
    split_name: str,
    threshold_kind: str,
    comparator_right: dict[str, bool],
    baseline_right: dict[str, bool],
) -> dict[str, Any]:
    """Build one row of the paired comparison against the rule baseline.

    Args:
        set_name: The item set the comparison was made on.
        split_name: The split the comparison was made on.
        threshold_kind: ``calibrated`` or ``oracle``.
        comparator_right: Per-item correctness of the retrieval comparator.
        baseline_right: Per-item correctness of the rule baseline, over the
            same items.

    Returns:
        One row in the shape ``retrieval_baseline_pairs.csv`` writes.
    """
    first, second = scoring.discordant(comparator_right, baseline_right)
    result = exact_mcnemar(first, second)
    return {
        "set": set_name,
        "split": split_name,
        "threshold_kind": threshold_kind,
        "comparator_only_correct": result.first_only,
        "baseline_only_correct": result.second_only,
        "p_value": result.p_value,
        "odds_ratio": result.odds_ratio,
        "odds_ratio_low": result.odds_ratio_low,
        "odds_ratio_high": result.odds_ratio_high,
    }


def write_scores(rows: list[dict[str, Any]]) -> Path:
    """Write the comparator's scored rows.

    Args:
        rows: One row per set, split and threshold kind.

    Returns:
        The path written.
    """
    out = path("data_processed") / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "retrieval_baseline.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"wrote {target}")
    return target


def write_pairs(rows: list[dict[str, Any]]) -> Path:
    """Write the paired comparison against the rule baseline.

    Args:
        rows: One row per set, split and threshold kind compared.

    Returns:
        The path written.
    """
    out = path("data_processed") / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "retrieval_baseline_pairs.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"wrote {target}")
    return target


def _decided(
    scores: dict[str, float], truth: dict[str, bool]
) -> tuple[list[float], list[bool]]:
    """Return the scores and truths of a population with its abstentions dropped.

    Args:
        scores: Item identifier to its best cosine score.
        truth: Item identifier to whether it was actually corrupted.

    Returns:
        Parallel lists of the decided scores and their truths.
    """
    kept = [
        (score, truth[item_id])
        for item_id, score in scores.items()
        if not math.isnan(score)
    ]
    return [score for score, _ in kept], [positive for _, positive in kept]


async def run(limit: int | None) -> None:
    """Score the comparator end to end and write its outputs.

    Args:
        limit: Cap the injected residual class to this many items, for a
            rehearsal. None scores every item.
    """
    seed = int(get("retrieval_baseline.seed"))
    share = float(get("retrieval_baseline.calibration_share"))
    radius_km = float(get("retrieval_baseline.radius_km"))
    encoder_tag = str(get("retrieval_baseline.encoder"))

    gazetteer = load_gazetteer()
    injected = [item for item in read_items() if item["class"] == RESIDUAL_CLASS]
    if limit is not None:
        injected = injected[:limit]
    injected_records = [as_retrieval_record(item) for item in injected]
    injected_truth = {r["item_id"]: r["corrupted"] for r in injected_records}
    items_by_id = {str(item["item_id"]): item for item in injected}

    natural_by_stratum = adjudicated_natural_records(
        read_natural_records(), read_natural_labels()
    )

    base_url = secret("models.serving.base_url_env")
    async with Encoder(base_url, encoder_tag) as encoder:
        injected_scores = await score_records(
            injected_records, gazetteer, encoder, radius_km
        )
        natural_scores = {
            stratum: await score_records(records, gazetteer, encoder, radius_km)
            for stratum, records in natural_by_stratum.items()
        }

    calibration, reported = split(injected_records, share=share, seed=seed)
    calibration_ids = {r["item_id"] for r in calibration}
    reported_ids = {r["item_id"] for r in reported}

    calibration_scores, calibration_truth = _decided(
        {i: injected_scores[i] for i in calibration_ids},
        {i: injected_truth[i] for i in calibration_ids},
    )
    calibrated, calibrated_j = oracle_threshold(calibration_scores, calibration_truth)
    logger.info(
        f"calibrated threshold {calibrated:.4f} reaches J {calibrated_j:.4f} "
        f"on {len(calibration_scores)} calibration records"
    )

    reported_scores = {i: injected_scores[i] for i in reported_ids}
    reported_truth = {i: injected_truth[i] for i in reported_ids}
    reported_only, reported_only_truth = _decided(reported_scores, reported_truth)
    reported_oracle, reported_oracle_j = oracle_threshold(
        reported_only, reported_only_truth
    )
    logger.info(
        f"oracle threshold {reported_oracle:.4f} reaches J {reported_oracle_j:.4f} "
        f"on {len(reported_only)} reported records"
    )

    rows = [
        score_row(
            "injected",
            "reported",
            "calibrated",
            calibrated,
            reported_scores,
            reported_truth,
        ),
        score_row(
            "injected",
            "reported",
            "oracle",
            reported_oracle,
            reported_scores,
            reported_truth,
        ),
    ]

    for stratum, records in natural_by_stratum.items():
        stratum_scores = natural_scores[stratum]
        stratum_truth = {r["item_id"]: r["corrupted"] for r in records}
        decided_scores, decided_truth = _decided(stratum_scores, stratum_truth)
        stratum_oracle, stratum_oracle_j = (
            oracle_threshold(decided_scores, decided_truth)
            if decided_scores
            else (calibrated, 0.0)
        )
        rows.append(
            score_row(
                "natural",
                stratum,
                "calibrated",
                calibrated,
                stratum_scores,
                stratum_truth,
            )
        )
        rows.append(
            score_row(
                "natural",
                stratum,
                "oracle",
                stratum_oracle,
                stratum_scores,
                stratum_truth,
            )
        )
        logger.info(
            f"{stratum}: oracle J {stratum_oracle_j:.4f} on "
            f"{len(decided_scores)} decided records"
        )
    write_scores(rows)

    rule_baseline_right = scoring.correctness(
        scoring.score_baseline([items_by_id[i] for i in reported_ids], gazetteer)
    )
    pairs = [
        paired_row(
            "injected",
            "reported",
            "calibrated",
            retrieval_correctness(reported_scores, reported_truth, calibrated),
            rule_baseline_right,
        ),
        paired_row(
            "injected",
            "reported",
            "oracle",
            retrieval_correctness(reported_scores, reported_truth, reported_oracle),
            rule_baseline_right,
        ),
    ]
    write_pairs(pairs)


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None, help="items per set, for a rehearsal"
    )
    args = parser.parse_args()
    asyncio.run(run(args.limit))


if __name__ == "__main__":
    main()
