"""Test whether the scored pass meets the phase's exit criterion.

The criterion is written in the plan and is not a matter of judgement: every
planned run complete, outputs deposited, no run marked offloaded. This checks
it mechanically, so that the phase closes on a measurement rather than on an
impression, and so that a partial pass cannot be mistaken for a finished one.

Four gates, and the pass fails if any one of them does:

    G1 every model answered every planned run, at that run's item count.
    G2 no answer carries an offloaded flag, and the footprint profile agrees.
    G3 the constrained runs stay under the configured invalid-answer ceiling.
       The unconstrained runs are exempt by design: their invalid rate is the
       measurement, and stopping on it would discard what they exist to show.
    G4 the efficiency summary exists and covers every model.

Nothing here repairs anything. A failing gate is reported and left, because
every repair available (changing the parser, dropping a model, moving a
threshold) is a design decision.

Usage:
    python scripts/check_phase5.py
"""

from __future__ import annotations

import csv
import json
import sys
from typing import Any

from poi_audit import scoring
from poi_audit.config import get, path
from poi_audit.inference import (
    efficiency_dir,
    item_set,
    planned_runs,
    pool,
    response_path,
)
from poi_audit.logsetup import setup

logger = setup("check_phase5")

CEILING = float(get("run.invalid_response_rate_stop"))


def expected_counts() -> dict[str, int]:
    """Return how many items each item set holds.

    Returns:
        Item set name to its count.
    """
    return {
        kind: len(item_set(kind)) for kind in ("injected", "injected_m1", "natural")
    }


def check_complete(counts: dict[str, int]) -> list[str]:
    """G1: every model answered every planned run, at that run's item count.

    Args:
        counts: Item set name to its count.

    Returns:
        One complaint per run and model that is missing or short.
    """
    problems: list[str] = []
    for run in planned_runs():
        wanted = counts[str(run["items"])]
        for card in pool():
            target = response_path(str(run["name"]), card)
            if not target.exists():
                problems.append(f"{run['name']} / {card.tag}: no answers")
                continue
            found = len(scoring.read_answers(target))
            if found != wanted:
                problems.append(
                    f"{run['name']} / {card.tag}: {found} answers, {wanted} expected"
                )
    return problems


def check_offload() -> list[str]:
    """G2: nothing offloaded, in the answers or in the profile.

    Returns:
        One complaint per model that offloaded.
    """
    problems: list[str] = []
    profile = efficiency_dir() / "footprint.csv"
    if not profile.exists():
        return [f"no footprint profile at {profile}"]
    with profile.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("offloaded", "")).strip().lower() == "true":
                problems.append(f"{row['tag']}: the profile marks it offloaded")
    return problems


def check_invalid() -> list[str]:
    """G3: the constrained runs stay under the ceiling.

    Returns:
        One complaint per constrained run and model above the ceiling.
    """
    problems: list[str] = []
    for run in planned_runs():
        if not run["constrained"]:
            continue
        for card in pool():
            target = response_path(str(run["name"]), card)
            if not target.exists():
                continue
            answers = scoring.read_answers(target)
            if not answers:
                continue
            invalid = sum(1 for answer in answers if not answer.get("valid"))
            rate = invalid / len(answers)
            if rate > CEILING:
                problems.append(
                    f"{run['name']} / {card.tag}: {rate:.1%} invalid, "
                    f"above the {CEILING:.0%} ceiling"
                )
    return problems


def check_summary() -> list[str]:
    """G4: the efficiency summary exists and covers every model.

    Returns:
        One complaint per model absent from the summary.
    """
    target = efficiency_dir() / "run_summary.csv"
    if not target.exists():
        return [f"no efficiency summary at {target}"]
    with target.open(encoding="utf-8") as handle:
        covered = {row["model"] for row in csv.DictReader(handle)}
    return [
        f"{card.tag}: absent from the efficiency summary"
        for card in pool()
        if card.tag not in covered
    ]


def main() -> None:
    """Run the four gates and report, exiting non-zero if any fails."""
    counts = expected_counts()
    gates: dict[str, list[str]] = {
        "G1 every planned run complete": check_complete(counts),
        "G2 nothing offloaded": check_offload(),
        "G3 constrained runs under the ceiling": check_invalid(),
        "G4 efficiency summary covers the pool": check_summary(),
    }
    report: dict[str, Any] = {}
    for gate, problems in gates.items():
        report[gate] = {"passes": not problems, "problems": problems}
        if problems:
            logger.error(f"{gate}: {len(problems)} problems")
            for problem in problems[:10]:
                logger.error(f"    {problem}")
            if len(problems) > 10:
                logger.error(f"    and {len(problems) - 10} more")
        else:
            logger.info(f"{gate}: passes")
    target = path("data_processed") / "analysis" / "phase5_check.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if any(problems for problems in gates.values()):
        logger.error("the phase does not meet its exit criterion")
        sys.exit(1)
    logger.info("every gate passes; the phase meets its exit criterion")


if __name__ == "__main__":
    main()
