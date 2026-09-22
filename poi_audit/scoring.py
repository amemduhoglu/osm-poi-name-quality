"""Turn model answers into per-item correctness, and per-item into measures.

The scoring is kept apart from the running so that a re-scored answer never
means a re-asked one: every number the article prints comes from the files the
inference pass wrote, and those files are the deposit's.

Three truths are scored against, and each belongs to its item set. The injected
set carries its own truth by construction, since the corruption is what built
it. The natural set carries the gold labels the two readers produced, and an
item they left at cannot_say has no truth to score against and is dropped rather
than counted either way.

An answer the schema rejects is a wrong answer and never a missing one.
Dropping it would score a model on the subset of items it happened to answer in
the shape that was asked for, which flatters exactly the models the
decoding-constraint arm exists to measure.

Rejected does not always mean unreadable, and the distinction matters for one
arm. Some rejected answers carry a usable verdict beside a field the schema
forbids, and honouring those verdicts would move an unconstrained score by as
much as a tenth of a point. The rule taken here is the strict one, fixed before
the answers existed and applied to every model alike: an answer that does not
satisfy the schema did not answer. The looser reading belongs in the robustness
analysis, not in the headline number.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from poi_audit.config import get, path
from poi_audit.stats import Confusion

WRONG = "wrong"
CANNOT_SAY = "cannot_say"


@dataclass(frozen=True)
class Scored:
    """One answer, judged.

    Attributes:
        item_id: The item answered.
        city: The city the record came from, which is the clustering unit.
        group: The class or stratum the item belongs to, for the breakdowns.
        truth_positive: Whether the record really carried the error asked about.
        flagged: Whether the model said it did.
        valid: Whether the answer parsed. An invalid answer counts as a failure
            to flag, never as an absent measurement.
        field_hit: Whether the model named the corrupted field, where the task
            asked for a field and the record carried one. None where the
            question does not arise.
    """

    item_id: str
    city: str
    group: str
    truth_positive: bool
    flagged: bool
    valid: bool
    field_hit: bool | None


def injected_truth() -> dict[str, dict[str, Any]]:
    """Return the injected set keyed by item.

    Returns:
        Item identifier to its record, carrying the class, whether it was
        corrupted and which field was corrupted.
    """
    source = path("data_processed") / "injected_set" / "items.jsonl"
    with source.open(encoding="utf-8") as handle:
        return {
            str(row["item_id"]): row
            for row in (json.loads(line) for line in handle if line.strip())
        }


def clusters() -> dict[str, str]:
    """Return the clustering unit each answered record belongs to.

    A record is not an independent draw: the corpus is stratified by city and
    difficulty travels with the city, so a comparison that pools records reads
    its uncertainty on the city instead. Both item sets carry the unit, and
    they are read together because a comparison may span either.

    Returns:
        Item identifier to the cluster it belongs to.
    """
    unit = str(get("analysis.cluster_unit"))
    both = {**injected_truth(), **natural_truth()}
    return {key: str(value[unit]) for key, value in both.items()}


def natural_truth() -> dict[str, dict[str, Any]]:
    """Return the natural set's gold labels keyed by item.

    Returns:
        Item identifier to its gold row. Items left at cannot_say are present
        and are dropped by the scorer, so that the count dropped stays
        measurable rather than silently absent.

    Raises:
        FileNotFoundError: If the gold labels have not been built.
    """
    source = path("data_processed") / "natural_set" / "gold.jsonl"
    if not source.exists():
        raise FileNotFoundError(
            f"gold labels not built: {source}; run scripts/build_gold_labels.py"
        )
    with source.open(encoding="utf-8") as handle:
        return {
            str(row["item_id"]): row
            for row in (json.loads(line) for line in handle if line.strip())
        }


def read_answers(target: Path) -> list[dict[str, Any]]:
    """Return one model's answers for one run.

    Args:
        target: The response file.

    Returns:
        The answers, in the order they were written.
    """
    with target.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def score_detect(
    answers: list[dict[str, Any]], truth: dict[str, dict[str, Any]]
) -> list[Scored]:
    """Judge answers to the detection task against the injected truth.

    Args:
        answers: The answers as the runner wrote them.
        truth: The injected set keyed by item.

    Returns:
        One judgement per answer. The field is scored only where the record was
        corrupted and the model flagged it, since naming a field on a clean
        record is already counted as a false positive and would otherwise be
        counted twice.
    """
    judged: list[Scored] = []
    for answer in answers:
        item = truth[str(answer["item_id"])]
        parsed = answer.get("parsed") or {}
        valid = bool(answer.get("valid"))
        flagged = valid and parsed.get("verdict") == WRONG
        corrupted = bool(item["corrupted"])
        hit: bool | None = None
        if corrupted and flagged:
            hit = parsed.get("field") == item["corrupted_field"]
        judged.append(
            Scored(
                item_id=str(answer["item_id"]),
                city=str(item["city"]),
                group=str(item["class"]),
                truth_positive=corrupted,
                flagged=flagged,
                valid=valid,
                field_hit=hit,
            )
        )
    return judged


def score_name_injected(
    answers: list[dict[str, Any]], truth: dict[str, dict[str, Any]]
) -> list[Scored]:
    """Judge answers to the residual question on the injected residual class.

    Args:
        answers: The answers as the runner wrote them.
        truth: The injected set keyed by item.

    Returns:
        One judgement per answer. A corrupted item is one whose name was taken
        from another place, which is what the question asks about.
    """
    judged: list[Scored] = []
    for answer in answers:
        item = truth[str(answer["item_id"])]
        parsed = answer.get("parsed") or {}
        valid = bool(answer.get("valid"))
        judged.append(
            Scored(
                item_id=str(answer["item_id"]),
                city=str(item["city"]),
                group=str(item["class"]),
                truth_positive=bool(item["corrupted"]),
                flagged=valid and parsed.get("verdict") == WRONG,
                valid=valid,
                field_hit=None,
            )
        )
    return judged


def score_name_natural(
    answers: list[dict[str, Any]],
    truth: dict[str, dict[str, Any]],
    core_only: bool = False,
) -> list[Scored]:
    """Judge answers to the residual question on the natural set.

    Args:
        answers: The answers as the runner wrote them.
        truth: The gold labels keyed by item.
        core_only: Keep only the decidable core, which is the items both readers
            reached the same definite label on independently.

    Returns:
        One judgement per scorable answer. An item with no definite label is
        dropped: a model cannot be scored against a label that does not exist,
        and counting it either way would invent one.
    """
    judged: list[Scored] = []
    for answer in answers:
        gold = truth[str(answer["item_id"])]
        if not gold["scored"]:
            continue
        if core_only and not gold["decidable_core"]:
            continue
        parsed = answer.get("parsed") or {}
        valid = bool(answer.get("valid"))
        judged.append(
            Scored(
                item_id=str(answer["item_id"]),
                city=str(gold["city"]),
                group=str(gold["stratum"]),
                truth_positive=gold["label"] == WRONG,
                flagged=valid and parsed.get("verdict") == WRONG,
                valid=valid,
                field_hit=None,
            )
        )
    return judged


def confusion(judged: list[Scored]) -> Confusion:
    """Count one set of judgements into a confusion.

    Args:
        judged: The judgements.

    Returns:
        The four counts the detection measures read.
    """
    return Confusion(
        true_positive=sum(1 for s in judged if s.truth_positive and s.flagged),
        false_negative=sum(1 for s in judged if s.truth_positive and not s.flagged),
        true_negative=sum(1 for s in judged if not s.truth_positive and not s.flagged),
        false_positive=sum(1 for s in judged if not s.truth_positive and s.flagged),
    )


def by_group(judged: list[Scored]) -> dict[str, Confusion]:
    """Split judgements by their group and count each.

    Args:
        judged: The judgements.

    Returns:
        Group to its confusion, for the per-class and per-stratum breakdowns.
    """
    grouped: dict[str, list[Scored]] = defaultdict(list)
    for one in judged:
        grouped[one.group].append(one)
    return {group: confusion(items) for group, items in sorted(grouped.items())}


def correctness(judged: list[Scored]) -> dict[str, bool]:
    """Return whether each item was answered correctly, keyed by item.

    Args:
        judged: The judgements.

    Returns:
        Item identifier to whether the model's verdict matched the truth. This
        is what a paired comparison between two systems is made on, since
        McNemar reads the records where they disagreed.
    """
    return {one.item_id: one.flagged == one.truth_positive for one in judged}


def field_accuracy(judged: list[Scored]) -> tuple[int, int]:
    """Return how often a flagged corruption was attributed to the right field.

    Args:
        judged: The judgements.

    Returns:
        The hits and the attempts. The attempts are the corrupted records the
        model flagged, so this measures naming rather than finding, which is
        the second half of the question the article asks of model choice.
    """
    attempts = [one for one in judged if one.field_hit is not None]
    return sum(1 for one in attempts if one.field_hit), len(attempts)


def as_record(item: dict[str, Any]) -> dict[str, Any]:
    """Return one injected item in the shape the rule baseline reads.

    Args:
        item: The injected item.

    Returns:
        The record, with the coordinate lifted out of the tags.
    """
    shown = dict(item["record"])
    lat = float(shown.pop("lat"))
    lon = float(shown.pop("lon"))
    return {
        "tags": shown,
        "country": item["country"],
        "city": item["city"],
        "lat": lat,
        "lon": lon,
    }


def score_baseline(items: list[dict[str, Any]], gazetteer: Any) -> list[Scored]:
    """Judge the rule baseline on the same items, for the paired comparison.

    The baseline is the answer a model has to beat, and a comparison against it
    is paired on the record rather than made between two summary rates, since
    the two systems are answering the same items and McNemar reads only the
    records where they disagreed.

    Args:
        items: The injected item set.
        gazetteer: The loaded gazetteer the reference lookups read.

    Returns:
        One judgement per item, in the same shape a model's answers produce.
    """
    from poi_audit import rules

    judged: list[Scored] = []
    for item in items:
        verdicts = rules.baseline(as_record(item), gazetteer)
        flagged = rules.flagged(verdicts)
        corrupted = bool(item["corrupted"])
        hit: bool | None = None
        if corrupted and flagged:
            hit = any(
                verdict.outcome == rules.FLAG
                and verdict.field == item["corrupted_field"]
                for verdict in verdicts
            )
        judged.append(
            Scored(
                item_id=str(item["item_id"]),
                city=str(item["city"]),
                group=str(item["class"]),
                truth_positive=corrupted,
                flagged=flagged,
                valid=True,
                field_hit=hit,
            )
        )
    return judged


def discordant(first: dict[str, bool], second: dict[str, bool]) -> tuple[int, int]:
    """Count the records two systems answered differently.

    Args:
        first: Per-item correctness of one system.
        second: Per-item correctness of the other.

    Returns:
        How many only the first got right, and how many only the second did,
        over the items both answered. An item one system never answered is not
        a disagreement and is left out.
    """
    shared = set(first) & set(second)
    return (
        sum(1 for item in shared if first[item] and not second[item]),
        sum(1 for item in shared if second[item] and not first[item]),
    )


def reweighted(judged: list[Scored], weights: dict[str, float]) -> dict[str, float]:
    """Return corpus-level estimates from a stratified sample.

    Both strata were drawn with known inclusion probabilities, so a corpus-level
    quantity is recovered by weighting each record by the reciprocal of its own
    probability rather than by the shape of the draw. Reporting the raw rate of
    an enriched sample as if it were a random one would overstate how often the
    residual class occurs.

    Args:
        judged: The judgements, each carrying the item it came from.
        weights: Item identifier to its inclusion probability.

    Returns:
        The weighted prevalence, recall and flag rate. An empty set returns
        zeros rather than dividing by nothing.

    Raises:
        KeyError: If an item has no inclusion probability.
        ValueError: If a probability is not positive, since its reciprocal is
            the weight and an impossible record cannot have been drawn.
    """
    if not judged:
        return {"prevalence": 0.0, "recall": 0.0, "flag_rate": 0.0}
    total = 0.0
    positive = 0.0
    flagged = 0.0
    caught = 0.0
    for one in judged:
        probability = weights[one.item_id]
        if probability <= 0:
            raise ValueError(f"inclusion probability not positive: {one.item_id}")
        weight = 1.0 / probability
        total += weight
        if one.truth_positive:
            positive += weight
            if one.flagged:
                caught += weight
        if one.flagged:
            flagged += weight
    return {
        "prevalence": positive / total,
        "recall": caught / positive if positive else 0.0,
        "flag_rate": flagged / total,
    }
