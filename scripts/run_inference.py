"""Run the model pool over the planned runs and write every answer to disk.

This is the study's measuring pass. Eleven runs are configured and every model
answers all of them: the injected set under both prompt versions, the residual
class under the four views of the input-dependence gate, the natural set, and
the two unconstrained runs that measure what the decoding constraint is worth.

Four rules govern the pass, and each is here rather than in a habit:

    The device decides whether anything may be scored. The footprint profile
    written by the preflight has to exist, name this card, and hold no model
    that offloads. Without it the run refuses to write into the response
    directory at all.

    One model is resident at a time, and the pool is walked from the smallest
    artifact upward, so that a pass stopped part way through has already covered
    the cheap end of the range.

    Every answer reaches disk before the next item is sent. A run resumes by
    item, so an interruption costs the items it had not reached.

    The stop rule is applied as it was written. A run whose invalid answers pass
    the configured rate stops the whole pass and reports, and the parser is not
    adjusted to get past it. The two unconstrained runs are exempt, and only
    they: the share of unparseable answers is the measurement there, so stopping
    on it would discard the number the run exists to produce.

Outputs:
    data/responses/<run>/<model>.jsonl
    data/processed/efficiency/run_summary.csv

Usage:
    python -m scripts.run_inference
    python -m scripts.run_inference --rehearse --limit 5   # before the card
    python -m scripts.run_inference --runs natural_name_v1 --models llama3.1:8b
    python -m scripts.run_inference --summary-only
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

import httpx

from poi_audit import prompting
from poi_audit.config import REPO_ROOT, get, path, secret
from poi_audit.inference import (
    ModelCard,
    ScoredItem,
    Server,
    adapted_cards,
    answered_items,
    device_memory_gb,
    efficiency_dir,
    invalid_answers,
    item_set,
    parse_answer,
    planned_runs,
    pool,
    relative,
)
from poi_audit.logsetup import setup

logger = setup("run_inference")

DEVICE_TOLERANCE_GB = 1.0
REHEARSAL_DIR = REPO_ROOT / "data" / "interim" / "rehearsal"


class StopRule(RuntimeError):
    """Raised when a run's invalid-answer rate passes the configured ceiling."""


def profile_path() -> Path:
    """Return the footprint profile the preflight writes."""
    return efficiency_dir() / "footprint.json"


def check_device(rehearse: bool) -> dict[str, Any] | None:
    """Refuse to score on anything but the measurement device.

    Args:
        rehearse: Whether this pass is a rehearsal, which scores nothing.

    Returns:
        The footprint profile, or None during a rehearsal.

    Raises:
        RuntimeError: If this is not the measurement device, if the profile is
            missing, or if it holds a model that offloads or fails a check.
    """
    configured = float(get("device.vram_gb"))
    device = device_memory_gb()
    if rehearse:
        logger.warning(
            f"rehearsal on {device} GB: answers go to {relative(REHEARSAL_DIR)} "
            "and no number produced here is a result"
        )
        return None
    if device is None or device < configured - DEVICE_TOLERANCE_GB:
        raise RuntimeError(
            f"device memory is {device} GB against a configured {configured} GB; "
            "constraint C3 puts every score on one card"
        )
    target = profile_path()
    if not target.exists():
        raise RuntimeError(
            f"no footprint profile at {relative(target)}; "
            "run scripts.preflight_models first"
        )
    profile = json.loads(target.read_text(encoding="utf-8"))
    stopped = [row["tag"] for row in profile["models"] if row.get("problem")]
    if stopped:
        raise RuntimeError(
            "the preflight stopped these models and they are not scored: "
            + ", ".join(stopped)
        )
    return dict(profile)


def output_root(rehearse: bool) -> Path:
    """Return the directory answers are written under.

    Args:
        rehearse: Whether this pass is a rehearsal.

    Returns:
        The response directory, or the rehearsal directory, which git ignores so
        that a rehearsal cannot be mistaken for a result.
    """
    return REHEARSAL_DIR if rehearse else path("responses")


def selected(
    names: list[str] | None, tags: list[str] | None, plan: str = "planned"
) -> tuple[list, list]:
    """Return the runs and models this pass covers.

    Args:
        plan: The configured run list to read.
        names: Run names to keep, or None for every planned run.
        tags: Model tags to keep, or None for the whole pool.

    Returns:
        The runs and the models.

    Raises:
        ValueError: If a named run or tag is not configured.
    """
    runs = planned_runs(plan)
    # An adapted model is not a pool member: only the redesign's runs may ask it,
    # and only when it is named, so a pass over the pool never reaches it.
    cards = pool()
    if plan == "redesign" and tags:
        cards += [card for card in adapted_cards() if card.tag in set(tags)]
    if names:
        known = {run["name"] for run in runs}
        unknown = set(names) - known
        if unknown:
            raise ValueError(f"unknown run: {', '.join(sorted(unknown))}")
        runs = [run for run in runs if run["name"] in set(names)]
    if tags:
        known = {card.tag for card in cards}
        unknown = set(tags) - known
        if unknown:
            raise ValueError(f"model outside the pool: {', '.join(sorted(unknown))}")
        cards = [card for card in cards if card.tag in set(tags)]
    return runs, cards


async def answer_run(
    server: Server,
    card: ModelCard,
    run: dict[str, Any],
    items: list[ScoredItem],
    target: Path,
    server_version: str | None = None,
) -> dict[str, int]:
    """Ask one model every item of one run, writing each answer as it arrives.

    Args:
        server: The local server.
        card: The model.
        run: The run's configuration.
        items: The run's items.
        target: The file the answers are written to.
        server_version: The serving stack's version, written beside every answer.

    Returns:
        How many items were asked and how many answers were invalid.

    Raises:
        StopRule: If the invalid rate passes the configured ceiling on a run the
            rule applies to.
        httpx.HTTPError: If the server fails past its retries.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    done = answered_items(target)
    remaining = [item for item in items if item.item_id not in done]
    if not remaining:
        logger.info(f"{card.tag} {run['name']}: complete, {len(done)} answers")
        return {"asked": 0, "invalid": 0}

    ceiling = float(get("run.invalid_response_rate_stop"))
    floor = int(get("run.invalid_response_min_items"))
    constrained = bool(run["constrained"])
    # The rule reads the run rather than the session: a resumed run carries the
    # invalid answers it already holds, or a run stopped for its invalid rate
    # would pass the same rule on the next attempt.
    answered = len(done)
    invalid = invalid_answers(target)
    asked = 0

    with target.open("a", encoding="utf-8") as handle:
        for item in remaining:
            rendered = prompting.render(
                str(run["task"]),
                int(run["prompt_version"]),
                item.shown,
                str(run["view"]),
                exemplar_view=(
                    str(run["exemplar_view"]) if run.get("exemplar_view") else None
                ),
                evidence=item.evidence,
            )
            schema = rendered.schema if constrained else None
            text, metrics = await server.ask(
                card.tag, rendered.system, rendered.user, schema
            )
            answer = parse_answer(text, rendered.schema, constrained)
            handle.write(
                json.dumps(
                    {
                        "run": run["name"],
                        "model": card.tag,
                        "item_id": item.item_id,
                        "task": run["task"],
                        "prompt_version": run["prompt_version"],
                        "view": run["view"],
                        "exemplar_view": run.get("exemplar_view"),
                        "constrained": constrained,
                        "answered_at": datetime.now(UTC).isoformat(),
                        "response_raw": answer.raw,
                        "parsed": answer.parsed,
                        "valid": answer.valid,
                        "server_version": server_version,
                        # The server's own timings and token counts, written
                        # beside the answer so that the efficiency profile needs
                        # no second pass over the pool.
                        **metrics,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()
            asked += 1
            answered += 1
            invalid += 0 if answer.valid else 1
            if (
                bool(run["stop_on_invalid"])
                and answered >= floor
                and invalid / answered > ceiling
            ):
                raise StopRule(
                    f"{card.tag} {run['name']}: {invalid} of {answered} answers "
                    f"invalid, above the configured {ceiling:.0%}"
                )
            if asked % 100 == 0:
                logger.info(
                    f"{card.tag} {run['name']}: {asked} of {len(remaining)}, "
                    f"{invalid} invalid"
                )

    logger.info(
        f"{card.tag} {run['name']}: {asked} asked, {invalid} invalid in all, "
        f"{answered} of {len(items)} complete"
    )
    return {"asked": asked, "invalid": invalid}


async def run_pass(
    rehearse: bool,
    limit: int | None,
    names: list[str] | None,
    tags: list[str] | None,
    plan: str = "planned",
) -> int:
    """Walk the pool over the planned runs.

    Args:
        rehearse: Whether this pass is a rehearsal on another device.
        limit: How many items of each run to ask, for a rehearsal.
        names: Run names to cover, or None for all.
        tags: Model tags to cover, or None for the whole pool.
        plan: The configured run list to read.

    Returns:
        A process exit status.
    """
    check_device(rehearse)
    runs, cards = selected(names, tags, plan)
    root = output_root(rehearse)
    sets = {kind: item_set(kind) for kind in {str(run["items"]) for run in runs}}
    planned = sum(
        min(len(sets[str(run["items"])]), limit or len(sets[str(run["items"])]))
        for run in runs
    ) * len(cards)
    logger.info(
        f"{len(cards)} models over {len(runs)} runs: {planned} calls at most, "
        f"answers under {relative(root)}"
    )

    base_url = secret("models.serving.base_url_env")
    timeout = float(get("models.serving.timeout_sec"))
    async with Server(base_url, timeout) as server:
        server_version = await server.version()
        logger.info(f"serving stack at {base_url}: version {server_version}")
        expected = str(get("device.ollama_version"))
        mismatched = [str(run["name"]) for run in runs if run.get("stack_check", True)]
        if server_version != expected and mismatched and not rehearse:
            # The same weights answer differently under two releases of the
            # server, so a scored run under another release is not the
            # instrument. Only a run that exists to measure that is exempt.
            logger.error(
                f"the server reports {server_version} and the configured instrument "
                f"is {expected}; refusing {', '.join(mismatched)}"
            )
            return 5
        for card in cards:
            for run in runs:
                items = sets[str(run["items"])]
                if limit:
                    items = items[:limit]
                target = root / str(run["name"]) / f"{card.slug}.jsonl"
                try:
                    await answer_run(server, card, run, items, target, server_version)
                except StopRule as stop:
                    logger.error(str(stop))
                    logger.error("the pass is stopped; the parser is not to be changed")
                    return 3
                except httpx.HTTPError as error:
                    logger.error(f"{card.tag} {run['name']}: server failed: {error}")
                    return 4
            await server.unload(card.tag)
            logger.info(f"{card.tag}: unloaded")

    # The efficiency summary describes the reported runs; the redesign's runs
    # are summarised by their own analysis, so the tracked summary does not move.
    if plan == "planned":
        summarise(root, rehearse)
    return 0


def summarise(root: Path, rehearse: bool) -> Path | None:
    """Write the efficiency summary from the answers on disk.

    The summary is rebuilt from the response files rather than accumulated in
    memory, so that it can be produced again from the deposit without repeating
    a single call.

    Args:
        root: The directory the answers were written under.
        rehearse: Whether this pass was a rehearsal, which writes no summary.

    Returns:
        The file written, or None during a rehearsal.
    """
    if rehearse:
        logger.warning("rehearsal: no summary written")
        return None
    rows: list[dict[str, Any]] = []
    for run in planned_runs():
        directory = root / str(run["name"])
        if not directory.exists():
            continue
        for target in sorted(directory.glob("*.jsonl")):
            answers = [
                json.loads(line)
                for line in target.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not answers:
                continue
            latencies = [a["total_sec"] for a in answers if a.get("total_sec")]
            rates = [
                a["output_tokens_per_sec"]
                for a in answers
                if a.get("output_tokens_per_sec")
            ]
            output = [a["output_tokens"] for a in answers if a.get("output_tokens")]
            invalid = sum(1 for a in answers if not a["valid"])
            rows.append(
                {
                    "run": run["name"],
                    "model": answers[0]["model"],
                    "task": run["task"],
                    "prompt_version": run["prompt_version"],
                    "view": run["view"],
                    "constrained": run["constrained"],
                    "answers": len(answers),
                    "invalid": invalid,
                    "invalid_rate": round(invalid / len(answers), 4),
                    "median_total_sec": (
                        round(median(latencies), 3) if latencies else None
                    ),
                    "median_output_tokens_per_sec": (
                        round(median(rates), 2) if rates else None
                    ),
                    "median_output_tokens": (
                        round(median(output), 1) if output else None
                    ),
                    "wall_sec": round(sum(latencies), 1) if latencies else None,
                }
            )
    if not rows:
        logger.warning("no answers on disk yet; no summary written")
        return None
    target = efficiency_dir() / "run_summary.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"wrote {relative(target)} with {len(rows)} rows")
    return target


def main() -> None:
    """Read the arguments and run the pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rehearse",
        action="store_true",
        help="run on a device that is not the measurement device, scoring nothing",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="items per run, for a rehearsal"
    )
    parser.add_argument("--runs", nargs="*", default=None, help="run names to cover")
    parser.add_argument("--models", nargs="*", default=None, help="model tags to cover")
    parser.add_argument(
        "--plan",
        choices=("planned", "redesign"),
        default="planned",
        help="the configured run list: the reported runs or the redesign's",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="rebuild the efficiency summary from the answers already on disk",
    )
    arguments = parser.parse_args()
    if arguments.summary_only:
        summarise(output_root(False), False)
        return
    raise SystemExit(
        asyncio.run(
            run_pass(
                arguments.rehearse,
                arguments.limit,
                arguments.runs,
                arguments.models,
                arguments.plan,
            )
        )
    )


if __name__ == "__main__":
    main()
