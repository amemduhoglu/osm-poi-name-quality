"""The scored inference layer: one local server, one model at a time.

Every number the article reports about a model is produced here, so this module
is written around four properties rather than around convenience.

    One device, one resident model. Constraint C3 puts every score and every
    rate on the same card. The server is asked to unload a model when its runs
    finish, so the footprint measured for the next one is that model's own, and
    a model that does not fit is stopped rather than published with an offload
    qualification.

    Conditions held identical. Temperature, seed, context window, output cap and
    the reasoning setting come from the configuration and are the same for every
    model, so that a difference between two scores is a difference of model.

    Every response written raw and immediately. The answer reaches disk before
    the next item is sent, with the server's own timings beside it, so an
    interrupted run resumes by item and the efficiency profile needs no second
    pass.

    A fixed parser. What counts as a valid answer is decided here, before any
    run, and is not adjusted after a rate is seen. A transport failure is
    retried; an answer the model actually gave is never retried, because a
    second draw at temperature zero returns the same text and keeping the better
    of two readings would round an ambiguous answer toward the correct one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from poi_audit.config import REPO_ROOT, get, path

NANOSECONDS = 1_000_000_000

# The first balanced JSON object in a response, used by the unconstrained arm
# alone. The rule is fixed here rather than in the run that reads it: a model
# writing its answer inside a sentence has still answered, and a model writing
# something else has not.
_OBJECT_START = "{"


@dataclass(frozen=True)
class ModelCard:
    """One model as the configuration declares it.

    Attributes:
        tag: The tag the local registry serves it under.
        name: What the article prints. The tag is the registry's word and does
            not appear in the manuscript.
        family: The family it belongs to, so that a ladder can be read.
        parameters_b: Parameters in billions, read from the model's metadata.
        weights_gb: The published artifact's size, recorded as provenance.
        expected_class: The eligibility class expected before measurement.
    """

    tag: str
    name: str
    family: str
    parameters_b: float
    weights_gb: float
    expected_class: str

    @property
    def slug(self) -> str:
        """Return a filename-safe form of the tag."""
        return re.sub(r"[^A-Za-z0-9._-]+", "-", self.tag)


@dataclass(frozen=True)
class ScoredItem:
    """One item as a model is asked about it.

    Attributes:
        item_id: The item's identifier in its own set.
        shown: The fields the item shows, before any view is applied.
        evidence: Entries retrieved for the item, for the retrieval-augmented
            arm, or None.
    """

    item_id: str
    shown: dict[str, Any]
    evidence: dict[str, Any] | None = None


@dataclass(frozen=True)
class Answer:
    """One model response, as it arrived and as it parsed.

    Attributes:
        raw: The response text exactly as the server returned it.
        parsed: The parsed object, or None when the text is not one.
        valid: Whether the parsed object satisfies the task's schema.
        metrics: The server's own timings and token counts.
    """

    raw: str
    parsed: dict[str, Any] | None
    valid: bool
    metrics: dict[str, Any] = field(default_factory=dict)


def pool() -> list[ModelCard]:
    """Return the approved model pool in the configured run order.

    Returns:
        Every model, smallest artifact first, so that a run stopped part way
        through has already covered the cheap end of the range.
    """
    cards = [
        ModelCard(
            tag=str(entry["tag"]),
            name=str(entry["name"]),
            family=str(entry["family"]),
            parameters_b=float(entry["parameters_b"]),
            weights_gb=float(entry["weights_gb"]),
            expected_class=str(entry["expected_class"]),
        )
        for entry in get("models.pool")
    ]
    key = str(get("run.order_models_by"))
    return sorted(cards, key=lambda card: getattr(card, key))


def adapted_cards() -> list[ModelCard]:
    """Return the adapted models the redesign runs, which are not pool members.

    Returns:
        One card per adapted model, carrying its base's family, size and weights,
        since the adapter changes neither the architecture nor the footprint class
        in any way the pool's split depends on.
    """
    served = get("lora.served", None)
    if not served:
        return []
    base = {card.tag: card for card in pool()}[str(get("lora.base_tag"))]
    return [
        ModelCard(
            tag=str(served["tag"]),
            name=str(served["name"]),
            family=base.family,
            parameters_b=base.parameters_b,
            weights_gb=base.weights_gb,
            expected_class=base.expected_class,
        )
    ]


def planned_runs(plan: str = "planned") -> list[dict[str, Any]]:
    """Return the planned runs in the order the configuration lists them.

    Args:
        plan: ``planned`` for the runs every reported analysis reads, or
            ``redesign`` for the runs added for the third venue.

    Returns:
        One dictionary per run, each naming its task, item set, prompt version,
        view, whether the answer is constrained to the schema, and whether the
        invalid-response stop rule applies to it.
    """
    if plan not in {"planned", "redesign"}:
        raise ValueError(f"unknown plan: {plan}")
    return [dict(entry) for entry in get(f"run.{plan}")]


RAG_SUFFIX = "_rag"


def evidence_path() -> Path:
    """Return the file the retrieval-augmented arm's evidence is written to."""
    return path("data_processed") / "retrieval_augmented" / "evidence.jsonl"


def attach_evidence(items: list[ScoredItem]) -> list[ScoredItem]:
    """Return the items with the evidence retrieved for each.

    Args:
        items: Items of one set.

    Returns:
        The same items carrying their evidence.

    Raises:
        FileNotFoundError: If the evidence has not been retrieved.
        KeyError: If an item has no evidence, which would silently turn the arm
            into version 1 for that item.
    """
    source = evidence_path()
    if not source.exists():
        raise FileNotFoundError(
            f"evidence not found: {source}; run scripts/build_rag_evidence.py"
        )
    with source.open(encoding="utf-8") as handle:
        evidence = {
            str(row["item_id"]): row["evidence"]
            for row in (json.loads(line) for line in handle if line.strip())
        }
    return [
        ScoredItem(
            item_id=item.item_id, shown=item.shown, evidence=evidence[item.item_id]
        )
        for item in items
    ]


def item_set(kind: str) -> list[ScoredItem]:
    """Read one of the item sets a run can ask about.

    Args:
        kind: ``injected`` for the whole injected set, ``injected_m1`` for its
            residual class alone, ``natural`` for the natural set's labelled
            records, ``residual_control`` for the residual control set, or
            ``natural_extension`` for the natural-set records past the first
            labelling stop together with the second-phase draw. Any of these
            with ``_rag`` appended carries the evidence retrieved for each item.
            ``residual_control_dev`` is the control set's development split.

    Returns:
        The items in their file order, which is the order they were written in;
        the natural kinds in identifier order.

    Raises:
        ValueError: If the kind is not one of these.
        FileNotFoundError: If the set has not been built.
    """
    if kind.endswith(RAG_SUFFIX):
        return attach_evidence(item_set(kind[: -len(RAG_SUFFIX)]))
    if kind == "residual_control_dev":
        # The development split alone, which prompt selection is scored on and
        # the article never reads.
        source = path("data_processed") / "injected_set" / "residual_control.jsonl"
        with source.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        return [
            ScoredItem(item_id=str(row["item_id"]), shown=dict(row["record"]))
            for row in rows
            if row["split"] == "dev"
        ]
    processed = path("data_processed")
    sources: list[Path]
    if kind in {"injected", "injected_m1"}:
        sources = [processed / "injected_set" / "items.jsonl"]
        builder = "scripts/build_injected_set.py"
        shown_key, class_key = "record", "M1"
    elif kind == "residual_control":
        sources = [processed / "injected_set" / "residual_control.jsonl"]
        builder = "scripts/build_residual_control.py"
        shown_key, class_key = "record", None
    elif kind in {"natural", "natural_extension"}:
        sources = [processed / "natural_set" / "records.jsonl"]
        builder = "scripts/draw_natural_set.py"
        shown_key, class_key = "shown", None
    else:
        raise ValueError(f"unknown item set: {kind}")
    if not sources[0].exists():
        raise FileNotFoundError(f"item set not found: {sources[0]}; run {builder}")
    second_phase = processed / "natural_set" / "records_phase2.jsonl"

    def read(source: Path) -> list[ScoredItem]:
        rows: list[ScoredItem] = []
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if kind == "injected_m1" and record.get("class") != class_key:
                    continue
                rows.append(
                    ScoredItem(
                        item_id=str(record["item_id"]), shown=dict(record[shown_key])
                    )
                )
        return rows

    items = read(sources[0])
    if kind in {"natural", "natural_extension"}:
        # The draw is larger than the first labelled set, and the records past
        # the stop carried no gold label when the reported runs were made, so
        # the reported runs ask only up to it. The extension asks the rest, and
        # the second-phase draw once it exists, for the labels the redesign adds.
        limit = int(get("natural_set.labelled.records"))
        ordered = sorted(items, key=lambda item: item.item_id)
        if kind == "natural":
            return ordered[:limit]
        items = ordered[limit:]
        if second_phase.exists():
            items += sorted(read(second_phase), key=lambda item: item.item_id)
    return items


def response_path(run_name: str, card: ModelCard) -> Path:
    """Return the file one model's answers for one run are written to.

    Args:
        run_name: The run's configured name.
        card: The model.

    Returns:
        The path. Its parent is not created here.
    """
    return path("responses") / run_name / f"{card.slug}.jsonl"


def answered_items(target: Path) -> set[str]:
    """Return the items a response file already holds.

    A run resumes by item rather than by restarting, so a stopped run costs the
    items it had not reached and nothing more.

    Args:
        target: The response file.

    Returns:
        The identifiers already answered. A truncated last line is ignored, so a
        run interrupted mid-write repeats one item rather than losing the file.
    """
    if not target.exists():
        return set()
    done: set[str] = set()
    with target.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                done.add(str(json.loads(line)["item_id"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def invalid_answers(target: Path) -> int:
    """Return how many answers already on disk failed the parser.

    A resumed run carries its predecessor's invalid answers into the stop rule.
    Counting only the current session would let a run that was stopped for its
    invalid rate pass the same rule on the next attempt.

    Args:
        target: The response file.

    Returns:
        The number of invalid answers it holds.
    """
    if not target.exists():
        return 0
    invalid = 0
    with target.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                invalid += 0 if json.loads(line)["valid"] else 1
            except (json.JSONDecodeError, KeyError):
                continue
    return invalid


def parse_answer(text: str, schema: dict[str, Any], constrained: bool) -> Answer:
    """Parse one response and decide whether it is a valid answer.

    Args:
        text: The response text as the server returned it.
        schema: The task's response schema.
        constrained: Whether the server was asked to constrain the answer to the
            schema. When it was not, the first balanced JSON object in the text
            is taken as the answer, because a model writing its answer inside a
            sentence has still answered.

    Returns:
        The answer, carrying the raw text whether it parsed or not.
    """
    candidate = text.strip() if constrained else _first_object(text)
    if not candidate:
        return Answer(raw=text, parsed=None, valid=False)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return Answer(raw=text, parsed=None, valid=False)
    if not isinstance(parsed, dict):
        return Answer(raw=text, parsed=None, valid=False)
    return Answer(raw=text, parsed=parsed, valid=satisfies(parsed, schema))


def _first_object(text: str) -> str:
    """Return the first balanced JSON object in a text, or an empty string.

    Args:
        text: The text to read.

    Returns:
        The substring from the first opening brace to the brace that closes it,
        counting braces inside strings as text rather than as structure.
    """
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, character in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == _OBJECT_START:
            if depth == 0:
                start = index
            depth += 1
        elif character == "}" and depth:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return ""


def satisfies(parsed: dict[str, Any], schema: dict[str, Any]) -> bool:
    """Test one parsed answer against the task's response schema.

    The schemas this study sends are flat objects of enumerated values, so the
    check is written out rather than taken from a validator: it is short, it has
    no dependency, and what counts as valid is visible in the file that decides
    it.

    Args:
        parsed: The parsed answer.
        schema: The task's response schema.

    Returns:
        True when every required property is present, every value is one the
        schema admits, and no property is present that the schema forbids.
    """
    properties = dict(schema.get("properties", {}))
    if schema.get("additionalProperties") is False:
        if set(parsed) - set(properties):
            return False
    for name in schema.get("required", []):
        if name not in parsed:
            return False
    for name, value in parsed.items():
        rule = properties.get(name)
        if rule is None:
            continue
        if "enum" in rule and value not in rule["enum"]:
            return False
    return True


class Server:
    """The local inference server, asked one model and one item at a time.

    Attributes:
        base_url: Where the server answers.
    """

    def __init__(self, base_url: str, timeout_sec: float) -> None:
        """Open a client against the local server.

        Args:
            base_url: The server's address.
            timeout_sec: How long one request may take.
        """
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url, timeout=httpx.Timeout(timeout_sec)
        )

    async def __aenter__(self) -> Server:
        """Enter the client's context."""
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        """Close the client."""
        await self.close()

    async def close(self) -> None:
        """Close the underlying client."""
        await self._client.aclose()

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        stop=stop_after_attempt(int(get("models.serving.max_retries", 3))),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    async def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Post one request, retrying transport failures only.

        Args:
            endpoint: The endpoint path.
            payload: The request body.

        Returns:
            The decoded response.

        Raises:
            httpx.HTTPError: If the server keeps failing.
        """
        response = await self._client.post(endpoint, json=payload)
        response.raise_for_status()
        return dict(response.json())

    async def tags(self) -> list[dict[str, Any]]:
        """Return the models the local registry holds.

        Returns:
            One entry per model, as the server reports it.

        Raises:
            httpx.HTTPError: If the server does not answer.
        """
        response = await self._client.get("/api/tags")
        response.raise_for_status()
        return list(response.json().get("models", []))

    async def version(self) -> str | None:
        """Return the serving stack's own version string.

        The version is part of the instrument: the same weights answer
        differently under two releases of the server, so it is written beside
        every answer rather than assumed from the configuration.

        Returns:
            The version, or None when the server does not report one.
        """
        try:
            response = await self._client.get("/api/version")
            response.raise_for_status()
        except httpx.HTTPError:
            return None
        return response.json().get("version")

    async def show(self, tag: str) -> dict[str, Any]:
        """Return one model's own metadata.

        Args:
            tag: The model tag.

        Returns:
            The server's description of the model, which is where the
            quantization and the parameter count are read from. Neither is taken
            from the tag, because the two do not always agree.

        Raises:
            httpx.HTTPError: If the model is unknown or the server fails.
        """
        return await self._post("/api/show", {"model": tag})

    async def resident(self) -> list[dict[str, Any]]:
        """Return the models currently loaded, with their memory use.

        Returns:
            One entry per resident model, carrying the total size and the part
            of it that sits in device memory. The two differ exactly when the
            model is partly on the host, which constraint C3 forbids.

        Raises:
            httpx.HTTPError: If the server does not answer.
        """
        response = await self._client.get("/api/ps")
        response.raise_for_status()
        return list(response.json().get("models", []))

    async def unload(self, tag: str) -> None:
        """Ask the server to release one model from memory.

        Args:
            tag: The model tag.
        """
        try:
            await self._client.post(
                "/api/generate", json={"model": tag, "keep_alive": 0}
            )
        except httpx.HTTPError:  # a model that will not unload is not a result
            return

    async def ask(
        self,
        tag: str,
        system: str,
        user: str,
        schema: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any]]:
        """Send one prompt and return the answer text with the server's timings.

        Args:
            tag: The model tag.
            system: The system message.
            user: The user message.
            schema: The schema to constrain the answer to, or None for the
                decoding-constraint check.

        Returns:
            The answer text and the server's own metrics for the call.

        Raises:
            httpx.HTTPError: If the server keeps failing.
        """
        options = dict(get("run.options"))
        keep_alive = options.pop("keep_alive")
        think = options.pop("think")
        payload: dict[str, Any] = {
            "model": tag,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": think,
            "keep_alive": keep_alive,
            "options": {
                **options,
                "temperature": float(get("models.serving.temperature")),
                "seed": int(get("models.serving.seed")),
            },
        }
        if schema is not None:
            payload["format"] = schema
        body = await self._post("/api/chat", payload)
        text = str(body.get("message", {}).get("content", ""))
        return text, timings(body)


def timings(body: dict[str, Any]) -> dict[str, Any]:
    """Return the efficiency metrics one response carries.

    The server reports durations in nanoseconds and counts in tokens. They are
    converted once, here, so that every run writes the same units and the
    efficiency profile needs no second pass over the responses.

    Args:
        body: The decoded response.

    Returns:
        Durations in seconds, token counts, and the generation rate in tokens
        per second. A count the server did not report is written as None rather
        than as zero, which would otherwise read as a measurement.
    """
    eval_count = body.get("eval_count")
    eval_duration = body.get("eval_duration")
    rate = None
    if eval_count and eval_duration:
        rate = round(float(eval_count) / (float(eval_duration) / NANOSECONDS), 3)
    return {
        "prompt_tokens": body.get("prompt_eval_count"),
        "output_tokens": eval_count,
        "load_sec": _seconds(body.get("load_duration")),
        "prompt_eval_sec": _seconds(body.get("prompt_eval_duration")),
        "eval_sec": _seconds(eval_duration),
        "total_sec": _seconds(body.get("total_duration")),
        "output_tokens_per_sec": rate,
    }


def _seconds(nanoseconds: Any) -> float | None:
    """Return a nanosecond duration in seconds, or None when it is absent.

    Args:
        nanoseconds: The duration as the server reported it.

    Returns:
        The duration in seconds, rounded to milliseconds.
    """
    if nanoseconds is None:
        return None
    return round(float(nanoseconds) / NANOSECONDS, 3)


def device_memory_gb() -> float | None:
    """Return the visible device's total memory in gigabytes.

    The value decides whether a run may be scored at all: constraint C3 puts
    every reported rate on one card, and a run made on a smaller one is not that
    measurement.

    Returns:
        The total memory of the first visible device, or None when no device
        can be read.
    """
    import shutil
    import subprocess

    binary = shutil.which("nvidia-smi")
    if binary is None:
        return None
    try:
        output = subprocess.run(
            [binary, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    first = output.strip().splitlines()[0] if output.strip() else ""
    try:
        return round(float(first) / 1024.0, 2)
    except ValueError:
        return None


def efficiency_dir() -> Path:
    """Return the directory the efficiency profile is written to.

    Returns:
        The path, created if it does not exist.
    """
    target = path("data_processed") / "efficiency"
    target.mkdir(parents=True, exist_ok=True)
    return target


def relative(target: Path) -> str:
    """Return a path relative to the repository root, for logging.

    Args:
        target: The path.

    Returns:
        The relative path as text, or the absolute path when it lies outside.
    """
    try:
        return str(target.relative_to(REPO_ROOT))
    except ValueError:
        return str(target)
