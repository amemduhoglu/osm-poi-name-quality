"""Tests for the frontier-reference pass.

The properties checked here are the ones the reference's comparability and its
cost depend on: that the request carries the pool's own decoding conditions and
the pinned provider, that the ceiling is enforced against what was actually
charged including a resumed pass's earlier spend, and that a reference answer is
written where no pool comparison will read it.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import httpx

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_reference.py"


def load() -> ModuleType:
    """Return the script as a module.

    Returns:
        The imported script, which reads the configuration on import.
    """
    spec = importlib.util.spec_from_file_location("run_reference", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def client_against(
    module: ModuleType, seen: list[dict], card: dict | None = None
) -> object:
    """Build a reference client whose requests are captured rather than sent.

    Args:
        module: The loaded script.
        seen: The list each request body is appended to.
        card: The reference card, or None for the first configured one.

    Returns:
        The client.
    """
    from poi_audit.config import get

    card = card or dict(get("models.reference_api.models")[0])

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "provider": "Novita",
                "model": "z-ai/glm-5.2",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"verdict": "belongs"}'},
                    }
                ],
                "usage": {
                    "prompt_tokens": 400,
                    "completion_tokens": 8,
                    "cost": 0.0004,
                },
            },
        )

    client = module.Reference(card, "https://example.invalid/api/v1", "key", 30.0)
    client._client = httpx.AsyncClient(
        base_url="https://example.invalid/api/v1",
        transport=httpx.MockTransport(handle),
        headers={"Authorization": "Bearer key"},
    )
    return client


def test_the_request_carries_the_pools_conditions_and_the_pinned_provider():
    """A reference asked under other conditions would not be comparable."""
    module = load()
    seen: list[dict] = []
    client = client_against(module, seen)
    schema = {"type": "object", "properties": {"verdict": {"enum": ["belongs"]}}}
    asyncio.run(client.ask("system", "user", schema))
    asyncio.run(client._client.aclose())

    assert len(seen) == 1
    body = seen[0]
    assert body["temperature"] == 0.0
    assert body["seed"] == 42
    assert body["reasoning"] == {"enabled": False}
    assert body["response_format"]["json_schema"]["schema"] == schema
    assert body["provider"]["allow_fallbacks"] is False
    assert body["provider"]["quantizations"] == ["fp8"]
    assert body["provider"]["only"]
    assert body["provider"]["require_parameters"] is True
    # The ceiling is enforced against the provider's own accounting, which the
    # endpoint returns only when the call asks for it.
    assert body["usage"] == {"include": True}


def test_the_answer_records_what_the_call_cost_and_who_served_it():
    """A silent switch of provider has to be visible in the deposit."""
    module = load()
    seen: list[dict] = []
    client = client_against(module, seen)
    text, record = asyncio.run(client.ask("system", "user", {"type": "object"}))
    asyncio.run(client._client.aclose())

    assert text == '{"verdict": "belongs"}'
    assert record["provider"] == "Novita"
    assert record["cost_usd"] == 0.0004
    assert record["prompt_tokens"] == 400
    assert record["output_tokens"] == 8


def test_a_resumed_pass_carries_the_cost_it_already_incurred(tmp_path, monkeypatch):
    """Counting only this session would let a stopped pass spend twice."""
    module = load()
    written: dict[str, Path] = {}

    def fake_path(run_name: str, model: str) -> Path:
        target = tmp_path / f"{run_name}.jsonl"
        written[run_name] = target
        return target

    monkeypatch.setattr(module, "response_path", fake_path)
    monkeypatch.setattr(module, "planned_runs", lambda: [{"name": "m1_name_full_v1"}])
    target = tmp_path / "m1_name_full_v1.jsonl"
    target.write_text(
        "\n".join(
            json.dumps({"item_id": f"I{index}", "cost_usd": 0.001})
            for index in range(5)
        )
        + "\n",
        encoding="utf-8",
    )
    assert module.spent_so_far("z-ai/glm-5.2") == 0.005


def test_reference_answers_are_written_where_no_pool_comparison_reads_them():
    """The pool's files are named by model tag; a reference is marked as one."""
    module = load()
    target = module.response_path("m1_name_full_v1", "z-ai/glm-5.2")
    assert target.name.startswith("reference-")
    assert target.parent.name == "m1_name_full_v1"


def test_each_reference_sends_its_own_pinned_provider():
    """A shared pin would route one reference through the other's endpoint."""
    from poi_audit.config import get

    module = load()
    cards = [dict(card) for card in get("models.reference_api.models")]
    assert len(cards) >= 2
    pins = []
    for card in cards:
        seen: list[dict] = []
        client = client_against(module, seen, card)
        asyncio.run(client.ask("system", "user", {"type": "object"}))
        asyncio.run(client._client.aclose())
        assert seen[0]["model"] == card["model"]
        pins.append(tuple(seen[0]["provider"]["only"]))
    assert len(set(pins)) == len(pins)


def test_a_busy_endpoint_is_waited_out_rather_than_counted_as_a_failure():
    """A rate limit is a matter of timing, so the item is asked again."""
    from poi_audit.config import get

    module = load()
    card = dict(get("models.reference_api.models")[0])
    attempts = {"n": 0}

    def handle(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(429, headers={"retry-after": "0"}, json={})
        return httpx.Response(
            200,
            json={
                "provider": "DeepInfra",
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {"cost": 0.0002},
            },
        )

    client = module.Reference(card, "https://example.invalid/api/v1", "key", 30.0)
    client._client = httpx.AsyncClient(
        base_url="https://example.invalid/api/v1",
        transport=httpx.MockTransport(handle),
    )
    text, record = asyncio.run(client.ask("system", "user", {"type": "object"}))
    asyncio.run(client._client.aclose())

    assert attempts["n"] == 3
    assert text == "{}"
    assert record["cost_usd"] == 0.0002


def test_every_configured_reference_is_planned() -> None:
    """The runner reads the list; it does not know how long the list is.

    A third reference was added on the editor's request, and a runner written
    for two would have run two of three and reported a complete pass.
    """
    from poi_audit import config

    module = load()
    plan = module.plan_runs()
    configured = {
        entry["name"] for entry in config.get("models")["reference_api"]["models"]
    }
    assert {entry["reference"] for entry in plan} == configured


def test_a_reference_is_sent_only_the_parameters_it_accepts() -> None:
    """A parameter the model does not take is not sent to it.

    With ``require_parameters`` set and fallbacks refused, an unsupported
    parameter has the request refused or served by somebody else. Either way the
    instrument changed without the study noticing.
    """
    module = load()
    entry = {
        "model": "anthropic/claude-opus-5",
        "accepts": ["temperature", "max_tokens", "reasoning"],
    }
    payload = module.decoding_payload(entry)
    assert "seed" not in payload
    assert payload["temperature"] == 0.0
    assert payload["reasoning"] == {"enabled": False}


def test_a_reference_that_takes_a_seed_and_no_temperature_is_sent_only_that() -> None:
    """The reverse pair takes the opposite parameter, not the same two."""
    module = load()
    entry = {
        "model": "openai/gpt-5.6-sol",
        "accepts": ["seed", "max_tokens", "reasoning"],
    }
    payload = module.decoding_payload(entry)
    assert "temperature" not in payload
    assert payload["seed"] == 42
    assert payload["max_tokens"] == 256


def test_the_sent_parameters_travel_with_the_answer():
    """The deposit records what was actually asked.

    Not the pool's full options block.
    """
    module = load()
    seen: list[dict] = []
    card = {
        "model": "anthropic/claude-opus-5",
        "accepts": ["temperature", "max_tokens", "reasoning"],
        "provider": {"only": ["anthropic"], "require_parameters": True},
    }
    client = client_against(module, seen, card)
    _, record = asyncio.run(client.ask("system", "user", {"type": "object"}))
    asyncio.run(client._client.aclose())

    assert "seed" not in seen[0]
    assert record["sent_parameters"] == {
        "temperature": 0.0,
        "max_tokens": 256,
        "reasoning": {"enabled": False},
    }


def test_an_overload_answered_as_200_is_waited_out_rather_than_crashing():
    """An overload answered as a 200 is waited out.

    A busy endpoint sometimes answers 200 with an error object instead of a
    retryable status code.

    None of these is an answer either, so it is retried the same way a
    transport-level 503 already is, rather than raised as a fault in the
    request.
    """
    from poi_audit.config import get

    module = load()
    card = dict(get("models.reference_api.models")[0])
    attempts = {"n": 0}

    def handle(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(
                200, json={"error": {"message": "Overloaded", "code": 503}}
            )
        return httpx.Response(
            200,
            json={
                "provider": "DeepInfra",
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {"cost": 0.0002},
            },
        )

    client = module.Reference(card, "https://example.invalid/api/v1", "key", 30.0)
    client._client = httpx.AsyncClient(
        base_url="https://example.invalid/api/v1",
        transport=httpx.MockTransport(handle),
    )
    text, record = asyncio.run(client.ask("system", "user", {"type": "object"}))
    asyncio.run(client._client.aclose())

    assert attempts["n"] == 3
    assert text == "{}"
    assert record["cost_usd"] == 0.0002


def test_a_refused_request_is_not_retried():
    """A fault in the request is not fixed by asking again."""
    from poi_audit.config import get

    module = load()
    card = dict(get("models.reference_api.models")[0])
    attempts = {"n": 0}

    def handle(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(404, json={"error": "no allowed provider"})

    client = module.Reference(card, "https://example.invalid/api/v1", "key", 30.0)
    client._client = httpx.AsyncClient(
        base_url="https://example.invalid/api/v1",
        transport=httpx.MockTransport(handle),
    )
    try:
        asyncio.run(client.ask("system", "user", {"type": "object"}))
    except httpx.HTTPStatusError:
        pass
    else:
        raise AssertionError("a refused request was accepted")
    finally:
        asyncio.run(client._client.aclose())
    assert attempts["n"] == 1
