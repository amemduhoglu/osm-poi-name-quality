"""Ask the frontier reference the residual question, under the pool's conditions.

The size rule fixed on 2026-08-03 admitted a frontier reference if balanced
accuracy on the residual class rose with model size; the trend was measured on
2026-08-07 and it did. This script runs that reference. It is not a member of
the pool and its answers never enter a pool comparison: it is an evaluated
reference whose score is reported beside the pool's, which is the only role the
plan gives a model this study did not measure on its own card.

Four properties, and each is a rule rather than a convenience.

    The same question, under the same conditions. The prompts are rendered by
    the same renderer from the same templates and item sets the pool answered.
    Temperature, seed, the output cap, the disabled reasoning trace and the
    schema constraint are the configuration's and are shared by both
    references, so a difference between a reference score and a pool score is a
    difference of model.

    Four references, in two pairs. The first pair is open weights, and one of
    them extends a ladder the pool already carries; a single reference could
    not separate a ceiling that belongs to the task from one that belongs to a
    lineage, and no pair of the twelve shares a training recipe across a wide
    enough range of sizes to separate capacity from recipe on its own. The
    second pair is commercial and closed weight, added on 2026-08-14 to
    answer an editor's request for a state-of-the-art comparison and for the
    use of large language models to be addressed. None of the four enters the
    pre-registered size-trend fit, which was defined on the twelve before any
    score existed; every reference is reported as a point beside the pool
    rather than a member of the fit.

    One provider, one quantization. The model is served by many providers at
    several quantizations, and a freely routed run would be answered at more
    than one. The provider is pinned, fallbacks are refused, and the provider
    that actually answered is written beside every answer, so a silent switch is
    visible in the deposit rather than absorbed into the score.

    The ceiling is enforced against what was actually charged, over the
    references together. Every response carries the provider's own cost for that
    call; the running total is checked before the next item is sent, and the
    pass stops at the configured ceiling rather than after it.

    The pilot decides whether the rest runs. It is a real run under the real
    settings, written to the same files and kept. It measures the cost of an
    item and whether the reasoning setting and the schema hold; the full pass is
    a separate invocation, taken only once those numbers have been read.

Outputs:
    data/responses/<run>/reference-<model>.jsonl
    data/processed/analysis/reference_cost.json

Usage:
    python -m scripts.run_reference --pilot
    python -m scripts.run_reference
    python -m scripts.run_reference --runs m1_name_full_v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from datetime import UTC, datetime
from itertools import groupby
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from poi_audit import prompting
from poi_audit.config import get, path, secret
from poi_audit.inference import (
    answered_items,
    invalid_answers,
    item_set,
    parse_answer,
    planned_runs,
)
from poi_audit.logsetup import setup

logger = setup("run_reference")

CONFIG = "models.reference_api"


# Statuses a shared endpoint answers with when it is busy rather than when the
# request is wrong. They are waited out rather than counted as failures: the
# study's rule is that a transport failure is retried and an answer the model
# actually gave is never retried, and none of these is an answer.
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
MAX_BACKOFF_SEC = 120.0


class Throttled(RuntimeError):
    """Raised when the endpoint asks the caller to wait and try again."""


class CeilingReached(RuntimeError):
    """Raised when the configured spend ceiling stops the pass."""


class StopRule(RuntimeError):
    """Raised when a run's invalid answers pass the configured rate."""


def decoding_payload(entry: dict[str, Any]) -> dict[str, Any]:
    """Return the decoding parameters one reference actually accepts.

    The four references do not take the same parameters. The two open-weights
    references take a temperature and a seed, the OpenAI reference takes a seed
    and no temperature, and the pinned endpoint of the Claude reference takes
    neither. With the provider pinned and fallbacks refused, a parameter the
    endpoint does not accept has the request refused or served by somebody
    else, either of which changes the instrument without the study noticing. A
    parameter absent from the entry's ``accepts`` list is therefore never built
    into the payload, and what was sent is written beside every answer.

    Args:
        entry: The reference as the configuration declares it, carrying an
            ``accepts`` list naming the decoding parameters it takes.

    Returns:
        The subset of the configured decoding options this reference accepts,
        keyed the way the request body expects them.
    """
    options = dict(get(f"{CONFIG}.options"))
    accepts = set(entry.get("accepts") or [])
    available: dict[str, Any] = {
        "temperature": float(options["temperature"]),
        "seed": int(options["seed"]),
        "max_tokens": int(options["max_tokens"]),
        "reasoning": {"enabled": bool(options["reasoning_enabled"])},
    }
    return {key: value for key, value in available.items() if key in accepts}


def slug(model: str) -> str:
    """Return a filename-safe form of a model identifier.

    Args:
        model: The identifier the endpoint serves the model under.

    Returns:
        The slug the response file is named with.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model)


def response_path(run_name: str, model: str) -> Path:
    """Return the file one run's reference answers are written to.

    Args:
        run_name: The run's configured name.
        model: The reference model's identifier.

    Returns:
        The path. The name marks the answers as a reference so that no pool
        comparison can read them by accident.
    """
    target = path("responses") / run_name / f"reference-{slug(model)}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def spent_so_far(model: str) -> float:
    """Return what the reference has already cost, read from the answers on disk.

    A resumed pass carries the cost it already incurred into the ceiling. Adding
    only this session's calls would let a pass stopped at the ceiling spend the
    same amount again on the next attempt.

    Args:
        model: The reference model's identifier.

    Returns:
        The total charged for every answer already written, in United States
        dollars.
    """
    total = 0.0
    for run in planned_runs():
        target = response_path(str(run["name"]), model)
        if not target.exists():
            continue
        with target.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    total += float(json.loads(line).get("cost_usd") or 0.0)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
    return total


class Reference:
    """The reference endpoint, asked one item at a time.

    Attributes:
        model: The identifier the endpoint serves the model under.
    """

    def __init__(
        self, card: dict[str, Any], base_url: str, api_key: str, timeout_sec: float
    ) -> None:
        """Open a client against the endpoint, for one reference.

        Args:
            card: The reference as the configuration declares it.
            base_url: The endpoint's address.
            api_key: The key the endpoint authenticates with.
            timeout_sec: How long one request may take.
        """
        self.card = dict(card)
        self.model = str(card["model"])
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_sec),
            headers={"Authorization": f"Bearer {api_key}"},
        )

    async def __aenter__(self) -> Reference:
        """Enter the client's context."""
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        """Close the client."""
        await self._client.aclose()

    async def _send(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Post one request, waiting out a busy endpoint.

        Args:
            payload: The request body.

        Returns:
            The decoded response.

        Raises:
            Throttled: If the endpoint is busy, after waiting as long as it
                asked to be waited for.
            httpx.HTTPStatusError: If the endpoint refuses the request itself,
                which is a fault in the request rather than in its timing and is
                not worth retrying.
        """
        response = await self._client.post("/chat/completions", json=payload)
        if response.status_code in RETRYABLE_STATUS:
            header = response.headers.get("retry-after", "")
            delay = float(header) if header.replace(".", "", 1).isdigit() else 0.0
            logger.warning(
                f"{self.model}: endpoint answered {response.status_code}, "
                f"waiting {delay or 'the backoff'} and asking again"
            )
            if delay:
                await asyncio.sleep(min(delay, MAX_BACKOFF_SEC))
            raise Throttled(f"{response.status_code} from the endpoint")
        response.raise_for_status()
        body = dict(response.json())
        # A busy endpoint sometimes answers 200 with an error object carrying
        # the same status codes RETRYABLE_STATUS already waits out at the
        # transport level. It is waited out here too: the study's rule is
        # that a transport failure is retried and an answer the model
        # actually gave is never retried, and an embedded "Overloaded" is not
        # an answer.
        error = body.get("error")
        if isinstance(error, dict) and error.get("code") in RETRYABLE_STATUS:
            logger.warning(
                f"{self.model}: endpoint answered 200 with "
                f"{error.get('code')} ({error.get('message')}), "
                "waiting the backoff and asking again"
            )
            raise Throttled(f"{error.get('code')} embedded in a 200 response")
        return body

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, Throttled)),
        stop=stop_after_attempt(8),
        wait=wait_exponential(multiplier=2, min=2, max=MAX_BACKOFF_SEC),
        reraise=True,
    )
    async def ask(
        self, system: str, user: str, schema: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Send one prompt and return the answer with what the call cost.

        Args:
            system: The system message.
            user: The user message.
            schema: The schema the answer is constrained to.

        Returns:
            The answer text and the call's own record: the provider that served
            it, the token counts, and the cost the provider charged.

        Raises:
            Throttled: If the endpoint is still busy after the backoff.
            httpx.HTTPError: If the endpoint keeps failing.
        """
        decoding = decoding_payload(self.card)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **decoding,
            "provider": dict(self.card["provider"]),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "strict": True,
                    "schema": schema,
                },
            },
            # The provider's own accounting for this call, which is what the
            # ceiling is enforced against. An estimate made here would be a
            # guess about a price list rather than a measurement.
            "usage": {"include": True},
        }
        body = await self._send(payload)
        if "error" in body:  # the endpoint answers 200 with an error object
            raise httpx.HTTPError(str(body["error"]))
        choice = (body.get("choices") or [{}])[0]
        message = dict(choice.get("message") or {})
        usage = dict(body.get("usage") or {})
        return str(message.get("content") or ""), {
            "provider": body.get("provider"),
            "served_model": body.get("model"),
            "finish_reason": choice.get("finish_reason"),
            # Written whether it is empty or not: an empty trace under a
            # disabled reasoning setting is the evidence the setting held.
            "reasoning_text": message.get("reasoning"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "cost_usd": usage.get("cost"),
            # What was actually sent, beside the provider that served it, so
            # the deposit records what each model was actually asked rather
            # than what the configuration asks for every reference.
            "sent_parameters": decoding,
        }


async def answer_run(
    client: Reference, run: dict[str, Any], limit: int | None, ceiling: float
) -> dict[str, Any]:
    """Ask the reference one run, writing every answer before the next is sent.

    Args:
        client: The endpoint.
        run: The planned run.
        limit: How many unanswered items to ask, or None for all of them.
        ceiling: The spend ceiling in United States dollars.

    Returns:
        What the run asked, what it cost and how many answers failed the parser.

    Raises:
        CeilingReached: If the ceiling is reached before the run finishes.
        StopRule: If the run's invalid answers pass the configured rate.
    """
    items = item_set(str(run["items"]))
    target = response_path(str(run["name"]), client.model)
    done = answered_items(target)
    remaining = [item for item in items if item.item_id not in done]
    if limit is not None:
        remaining = remaining[:limit]
    if not remaining:
        logger.info(f"{run['name']}: nothing to ask, {len(done)} answers on disk")
        return {"asked": 0, "invalid": 0, "cost": 0.0}

    rate_ceiling = float(get("run.invalid_response_rate_stop"))
    floor = int(get("run.invalid_response_min_items"))
    answered = len(done)
    invalid = invalid_answers(target)
    asked = 0
    cost = 0.0

    with target.open("a", encoding="utf-8") as handle:
        for item in remaining:
            if spent_so_far(client.model) + cost >= ceiling:
                raise CeilingReached(
                    f"{run['name']}: stopped at the ${ceiling:.2f} ceiling with "
                    f"{asked} of {len(remaining)} items asked"
                )
            rendered = prompting.render(
                str(run["task"]),
                int(run["prompt_version"]),
                item.shown,
                str(run["view"]),
            )
            text, record = await client.ask(
                rendered.system, rendered.user, rendered.schema
            )
            answer = parse_answer(text, rendered.schema, constrained=True)
            handle.write(
                json.dumps(
                    {
                        "run": run["name"],
                        "model": client.model,
                        "role": "reference",
                        "item_id": item.item_id,
                        "task": run["task"],
                        "prompt_version": run["prompt_version"],
                        "view": run["view"],
                        "constrained": True,
                        "answered_at": datetime.now(UTC).isoformat(),
                        "response_raw": answer.raw,
                        "parsed": answer.parsed,
                        "valid": answer.valid,
                        **record,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()
            asked += 1
            answered += 1
            cost += float(record.get("cost_usd") or 0.0)
            invalid += 0 if answer.valid else 1
            if (
                bool(run["stop_on_invalid"])
                and answered >= floor
                and invalid / answered > rate_ceiling
            ):
                raise StopRule(
                    f"{run['name']}: {invalid} of {answered} answers invalid, "
                    f"above the configured {rate_ceiling:.0%}"
                )
            if asked % 50 == 0:
                logger.info(
                    f"{run['name']}: {asked} of {len(remaining)}, "
                    f"{invalid} invalid, ${cost:.4f} spent this session"
                )

    logger.info(
        f"{run['name']}: {asked} asked, {invalid} invalid in all, "
        f"${cost:.4f} this session"
    )
    return {"asked": asked, "invalid": invalid, "cost": cost}


def plan_runs() -> list[dict[str, Any]]:
    """Return one entry per reference and run, in the order they are answered.

    Extracted from ``run_pass`` so that the plan can be read without a network
    call. The order is the configuration's and is not sorted here: the runs that
    decide something come first, so a pass stopped at the ceiling has finished
    them.

    Returns:
        One dictionary per planned run, carrying ``reference``, ``model`` and
        ``run``.
    """
    cards = [dict(card) for card in get(f"{CONFIG}.models")]
    runs = [str(name) for name in get(f"{CONFIG}.runs")]
    return [
        {"reference": str(card["name"]), "model": str(card["model"]), "run": run}
        for card in cards
        for run in runs
    ]


def report(rows: list[dict[str, Any]], stopped: str | None) -> Path:
    """Write what the reference pass asked and what it cost.

    Args:
        rows: One entry per reference and run.
        stopped: Why the pass stopped early, or None if it did not.

    Returns:
        The path written.
    """
    cards = [dict(card) for card in get(f"{CONFIG}.models")]
    target = path("data_processed") / "analysis" / "reference_cost.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "references": cards,
                "options": get(f"{CONFIG}.options"),
                "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "spend_ceiling_usd": get(f"{CONFIG}.spend_ceiling_usd"),
                "spent_usd": {
                    str(card["model"]): round(spent_so_far(str(card["model"])), 6)
                    for card in cards
                },
                "stopped_because": stopped,
                "runs": rows,
            },
            indent=2,
            # A card's weights_checked is read back from the configuration as
            # a date rather than a string; written as the string it prints,
            # never re-parsed from this file.
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info(f"wrote {target}")
    return target


async def run_pass(
    names: list[str] | None, models: list[str] | None, limit: int | None
) -> int:
    """Ask every configured reference every configured run, in order.

    The references are asked one after another rather than together, so that a
    pass stopped at the ceiling has finished the first reference rather than
    left both half answered.

    Args:
        names: The runs to ask, or None for the configured list.
        models: The references to ask, or None for all of them.
        limit: How many unanswered items to ask per run, or None for all.

    Returns:
        A process exit code: zero when the pass finished, one when it stopped.

    Raises:
        RuntimeError: If the endpoint is disabled in the configuration.
        ValueError: If a run or a reference is asked for that is not configured.
    """
    if not bool(get(f"{CONFIG}.enabled")):
        raise RuntimeError("the reference endpoint is disabled in the configuration")
    configured = [str(name) for name in get(f"{CONFIG}.runs")]
    wanted = names or configured
    unknown = set(wanted) - set(configured)
    if unknown:
        raise ValueError(
            f"runs the reference is not configured to ask: {sorted(unknown)}"
        )
    cards = [dict(card) for card in get(f"{CONFIG}.models")]
    if models:
        known = {str(card["model"]) for card in cards}
        missing = set(models) - known
        if missing:
            raise ValueError(f"references not configured: {sorted(missing)}")
        cards = [card for card in cards if str(card["model"]) in models]
    run_defs = {str(run["name"]): dict(run) for run in planned_runs()}
    ceiling = float(get(f"{CONFIG}.spend_ceiling_usd"))
    base_url = str(get(f"{CONFIG}.base_url"))
    api_key = secret(f"{CONFIG}.api_key_env")
    timeout = float(get("models.serving.timeout_sec"))

    by_model = {str(card["model"]): card for card in cards}
    entries = [
        entry
        for entry in plan_runs()
        if entry["run"] in wanted and entry["model"] in by_model
    ]

    rows: list[dict[str, Any]] = []
    stopped: str | None = None
    for model, group in groupby(entries, key=lambda entry: entry["model"]):
        card = by_model[model]
        logger.info(
            f"reference {model}: ${spent_so_far(model):.4f} spent so far, "
            f"${ceiling:.2f} ceiling over all references"
        )
        async with Reference(card, base_url, api_key, timeout) as client:
            for entry in group:
                name = entry["run"]
                try:
                    outcome = await answer_run(client, run_defs[name], limit, ceiling)
                except (CeilingReached, StopRule) as stop:
                    stopped = str(stop)
                    logger.error(stopped)
                    break
                rows.append({"model": model, "run": name, **outcome})
        if stopped:
            break

    report(rows, stopped)
    return 1 if stopped else 0


def main() -> None:
    """Run the reference pass from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="ask the configured pilot count per run and stop",
    )
    parser.add_argument("--runs", nargs="*", help="ask only these runs")
    parser.add_argument("--models", nargs="*", help="ask only these references")
    parser.add_argument("--limit", type=int, help="ask this many items per run")
    args = parser.parse_args()
    limit = args.limit
    if args.pilot:
        limit = int(get(f"{CONFIG}.pilot_items"))
    raise SystemExit(asyncio.run(run_pass(args.runs, args.models, limit)))


if __name__ == "__main__":
    main()
