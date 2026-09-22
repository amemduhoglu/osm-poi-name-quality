"""Measure how stable the ordering of systems on the natural set is.

The decision of 2026-09-11 found the ordering of systems on the natural set too
unstable to carry a claim, since it rests on 29 wrong names. This measures that
directly rather than arguing it. Cities are resampled with replacement, which is
the unit the corpus is stratified by and the unit difficulty travels with, and
every system is rescored by Youden's J on the same resample. Each system's share
of first places and the 95% interval of its rank are reported.

Beside it, the arithmetic floor on what the set can separate at all: an exact
McNemar test needs at least eleven discordant pairs split eleven to none before
it reaches p < 0.001, so two systems must differ in recall by at least eleven
over the positives before the study's own threshold can tell them apart.

Outputs:
    data/processed/analysis/rank_stability.csv
    data/processed/analysis/detectability.json

Usage:
    python -m scripts.score_rank_stability
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from poi_audit import scoring, stats
from poi_audit.config import get, path
from poi_audit.inference import pool
from poi_audit.logsetup import setup
from poi_audit.stats import Confusion
from scripts.run_reference import slug as reference_slug

logger = setup("score_rank_stability")


def out_dir() -> Path:
    """Return the directory the analysis outputs are written to."""
    target = path("data_processed") / "analysis"
    target.mkdir(parents=True, exist_ok=True)
    return target


def youden(pairs: list[tuple[bool, bool]]) -> float:
    """Return Youden's J over (is_positive, flagged) pairs.

    Args:
        pairs: One pair per record.

    Returns:
        J, computed by the same confusion the article's scores use.
    """
    return Confusion(
        true_positive=sum(1 for positive, flagged in pairs if positive and flagged),
        false_negative=sum(
            1 for positive, flagged in pairs if positive and not flagged
        ),
        true_negative=sum(
            1 for positive, flagged in pairs if not positive and not flagged
        ),
        false_positive=sum(
            1 for positive, flagged in pairs if not positive and flagged
        ),
    ).youdens_j


def rank_stability(
    correct: dict[str, dict[str, tuple[bool, bool]]],
    clusters: dict[str, str],
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Return each system's share of first places and its 95% rank interval.

    Args:
        correct: Per system, per item, whether the item is positive and whether
            the system flagged it.
        clusters: Item to city.
        replicates: How many city resamples.
        seed: The generator seed.

    Returns:
        Per system: J on the full set, the share of replicates in which it ranks
        first (shared equally among systems tied for first), and the 2.5th and
        97.5th percentiles of its rank, where rank 1 is best and ties take the
        average rank.

    Raises:
        ValueError: If the systems were not scored on the same items, or an item
            carries no city.
    """
    systems = sorted(correct)
    items = set(correct[systems[0]])
    if any(set(correct[system]) != items for system in systems):
        raise ValueError("every system must be scored on the same items")
    missing = items - set(clusters)
    if missing:
        raise ValueError(f"{len(missing)} item(s) carry no city")

    by_city: dict[str, list[str]] = defaultdict(list)
    for item in sorted(items):
        by_city[clusters[item]].append(item)
    cities = sorted(by_city)

    generator = np.random.default_rng(seed)
    ranks: dict[str, list[float]] = {system: [] for system in systems}
    top: dict[str, float] = dict.fromkeys(systems, 0.0)
    for _ in range(replicates):
        drawn = generator.choice(len(cities), size=len(cities), replace=True)
        resample = [item for index in drawn for item in by_city[cities[index]]]
        scores = [
            youden([correct[system][item] for item in resample]) for system in systems
        ]
        order = rankdata([-score for score in scores], method="average")
        best = max(scores)
        leaders = [
            system
            for system, score in zip(systems, scores, strict=True)
            if score == best
        ]
        for system, rank in zip(systems, order, strict=True):
            ranks[system].append(float(rank))
        for leader in leaders:
            top[leader] += 1.0 / len(leaders)

    result: dict[str, dict[str, float]] = {}
    for system in systems:
        low, high = np.quantile(ranks[system], [0.025, 0.975])
        result[system] = {
            "j": youden(list(correct[system].values())),
            "top_share": top[system] / replicates,
            "rank_low": float(low),
            "rank_high": float(high),
        }
    return result


def display_names() -> dict[str, str]:
    """Return each response file's stem mapped to the name the article prints."""
    names = {card.slug: card.name for card in pool()}
    for entry in get("models.reference_api.models", []):
        names[f"reference-{reference_slug(str(entry['model']))}"] = str(entry["name"])
    return names


def load_run(
    run: str, truth: dict[str, dict]
) -> tuple[dict[str, dict[str, tuple[bool, bool]]], dict[str, str]]:
    """Judge every system's answers to one natural-set run.

    Args:
        run: The run name under the responses directory.
        truth: The gold labels keyed by item.

    Returns:
        Per system the judged pairs on the items every system was scored on,
        and item to city.
    """
    correct: dict[str, dict[str, tuple[bool, bool]]] = {}
    clusters: dict[str, str] = {}
    # Outside the reported run, only the models the replay kept are ranked: an
    # adapted model and a model the replay excluded are not pool members there.
    kept = None
    if run != "natural_name_v1":
        kept = {
            card.slug
            for card in pool()
            if card.tag not in set(get("run.replay_excluded_shown"))
        }
    for target in sorted((path("responses") / run).glob("*.jsonl")):
        if kept is not None and target.stem not in kept:
            continue
        judged = scoring.score_name_natural(scoring.read_answers(target), truth)
        correct[target.stem] = {
            one.item_id: (one.truth_positive, one.flagged) for one in judged
        }
        clusters.update({one.item_id: one.city for one in judged})
    shared = set.intersection(*(set(pairs) for pairs in correct.values()))
    for system, pairs in correct.items():
        dropped = len(pairs) - len(shared)
        if dropped:
            logger.warning(f"{run}: {system} loses {dropped} item(s) not shared")
        correct[system] = {item: pairs[item] for item in shared}
    return correct, clusters


def main() -> None:
    """Score rank stability for the configured runs and write the floor beside it."""
    settings = get("analysis.rank_stability")
    alpha = float(get("analysis.significance_threshold"))
    truth = scoring.natural_truth()
    names = display_names()
    rows: list[dict[str, object]] = []
    floors: dict[str, dict[str, object]] = {}
    for run in settings["runs"]:
        correct, clusters = load_run(run, truth)
        result = rank_stability(
            correct, clusters, int(settings["replicates"]), int(settings["seed"])
        )
        for system, measures in sorted(
            result.items(), key=lambda pair: -pair[1]["top_share"]
        ):
            rows.append(
                {
                    "run": run,
                    "system": system,
                    "name": names.get(system, system),
                    "youdens_j": round(measures["j"], 4),
                    "top_share": round(measures["top_share"], 4),
                    "rank_low": round(measures["rank_low"], 2),
                    "rank_high": round(measures["rank_high"], 2),
                    "systems": len(result),
                }
            )
        any_system = next(iter(correct.values()))
        positives = sum(1 for positive, _ in any_system.values() if positive)
        floors[run] = {
            "items": len(any_system),
            "positives": positives,
            "cities": len(set(clusters.values())),
            "alpha": alpha,
            "min_discordant_pairs": stats.min_discordant_for_significance(alpha),
            "min_detectable_recall_difference": round(
                stats.min_detectable_recall_difference(positives, alpha), 4
            ),
        }
        logger.info(f"{run}: {len(result)} systems, {positives} positives")

    target = out_dir() / "rank_stability.csv"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir() / "detectability.json").write_text(
        json.dumps(
            {
                "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "replicates": int(settings["replicates"]),
                "seed": int(settings["seed"]),
                "runs": floors,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info(f"wrote {target}")


if __name__ == "__main__":
    main()
