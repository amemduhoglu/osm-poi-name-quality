"""Tests for the scored inference layer.

Three properties are worth a test here, and each of them protects a number the
article prints. The request has to carry the study's decoding conditions, or a
score is a property of a server default rather than of a model. The parser has
to be fixed and readable, because the invalid-response rate is both a stop rule
and, on the unconstrained runs, a result. And a run has to resume by item, or an
interrupted pass either loses answers or repeats calls that were already paid
for on the one measurement device.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from poi_audit import inference, prompting
from poi_audit.config import get

DETECT_SCHEMA = prompting.load_template("detect", 1)["response_schema"]
NAME_SCHEMA = prompting.load_template("name", 1)["response_schema"]

BODY = {
    "message": {"content": '{"verdict": "belongs"}'},
    "prompt_eval_count": 310,
    "prompt_eval_duration": 500_000_000,
    "eval_count": 8,
    "eval_duration": 2_000_000_000,
    "load_duration": 1_000_000_000,
    "total_duration": 3_500_000_000,
}


def transport(recorder: list[dict]) -> httpx.MockTransport:
    """Return a transport that records requests and answers with one body.

    Args:
        recorder: The list each decoded request body is appended to.

    Returns:
        The transport.
    """

    def handle(request: httpx.Request) -> httpx.Response:
        recorder.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json=BODY)

    return httpx.MockTransport(handle)


def server_with(recorder: list[dict]) -> inference.Server:
    """Return a server whose client answers from a recording transport.

    Args:
        recorder: The list requests are recorded into.

    Returns:
        The server.
    """
    server = inference.Server("http://localhost:11434", 30.0)
    server._client = httpx.AsyncClient(
        base_url=server.base_url, transport=transport(recorder)
    )
    return server


async def ask_once(recorder: list[dict], schema: dict | None) -> tuple[str, dict]:
    """Send one request through a recording transport.

    Args:
        recorder: The list requests are recorded into.
        schema: The schema to constrain the answer to, or None.

    Returns:
        The answer text and the call's metrics.
    """
    async with server_with(recorder) as server:
        return await server.ask("m", "system", "user", schema)


# --- what the request carries ---------------------------------------------


def test_the_request_carries_the_configured_decoding_conditions():
    recorder: list[dict] = []
    asyncio.run(ask_once(recorder, NAME_SCHEMA))
    sent = recorder[0]
    assert sent["options"]["temperature"] == get("models.serving.temperature")
    assert sent["options"]["seed"] == get("models.serving.seed")
    assert sent["options"]["num_ctx"] == get("run.options.num_ctx")
    assert sent["options"]["num_predict"] == get("run.options.num_predict")
    assert sent["think"] == get("run.options.think")
    assert sent["keep_alive"] == get("run.options.keep_alive")
    assert sent["stream"] is False


def test_the_schema_is_sent_only_when_the_answer_is_constrained():
    recorder: list[dict] = []
    asyncio.run(ask_once(recorder, NAME_SCHEMA))
    asyncio.run(ask_once(recorder, None))
    assert recorder[0]["format"] == NAME_SCHEMA
    assert "format" not in recorder[1]


def test_the_timings_are_converted_once_into_seconds_and_a_rate():
    recorder: list[dict] = []
    metrics = asyncio.run(ask_once(recorder, NAME_SCHEMA))[1]
    assert metrics["total_sec"] == 3.5
    assert metrics["output_tokens"] == 8
    assert metrics["output_tokens_per_sec"] == 4.0


def test_a_count_the_server_did_not_report_is_not_written_as_zero():
    metrics = inference.timings({"eval_count": 4})
    assert metrics["total_sec"] is None
    assert metrics["output_tokens_per_sec"] is None


# --- what counts as an answer ---------------------------------------------


def test_a_constrained_answer_matching_the_schema_is_valid():
    answer = inference.parse_answer('{"verdict": "belongs"}', NAME_SCHEMA, True)
    assert answer.valid and answer.parsed == {"verdict": "belongs"}


def test_a_verdict_outside_the_schema_is_not_valid():
    answer = inference.parse_answer('{"verdict": "maybe"}', NAME_SCHEMA, True)
    assert answer.parsed == {"verdict": "maybe"}
    assert not answer.valid


def test_a_missing_required_property_is_not_valid():
    answer = inference.parse_answer('{"verdict": "wrong"}', DETECT_SCHEMA, True)
    assert not answer.valid


def test_a_property_the_schema_forbids_is_not_valid():
    text = '{"verdict": "belongs", "confidence": 0.9}'
    assert not inference.parse_answer(text, NAME_SCHEMA, True).valid


def test_text_that_is_not_an_object_is_not_an_answer():
    for text in ("belongs", "", "[1, 2]", '"belongs"'):
        answer = inference.parse_answer(text, NAME_SCHEMA, True)
        assert answer.parsed is None and not answer.valid


def test_the_raw_text_is_kept_whether_it_parsed_or_not():
    answer = inference.parse_answer("no idea", NAME_SCHEMA, True)
    assert answer.raw == "no idea"


def test_an_unconstrained_answer_may_sit_inside_a_sentence():
    text = 'Looking at the record, I would say {"verdict": "wrong"} here.'
    assert not inference.parse_answer(text, NAME_SCHEMA, True).valid
    unconstrained = inference.parse_answer(text, NAME_SCHEMA, False)
    assert unconstrained.valid and unconstrained.parsed == {"verdict": "wrong"}


def test_a_brace_inside_a_string_does_not_end_the_object():
    text = 'answer: {"verdict": "belongs", "note": "a } brace"} and that is all'
    parsed = inference._first_object(text)
    assert json.loads(parsed)["note"] == "a } brace"


def test_a_nested_object_is_read_to_its_own_end():
    text = '{"verdict": "wrong", "where": {"field": "name"}}'
    assert inference._first_object(text) == text


def test_an_unfinished_object_is_not_an_answer():
    answer = inference.parse_answer('{"verdict": "belo', NAME_SCHEMA, False)
    assert answer.parsed is None and not answer.valid


# --- resuming --------------------------------------------------------------


def test_a_run_resumes_from_the_items_already_answered(tmp_path):
    target = tmp_path / "answers.jsonl"
    target.write_text(
        '{"item_id": "M1-0001-1"}\n{"item_id": "M1-0001-2"}\n', encoding="utf-8"
    )
    assert inference.answered_items(target) == {"M1-0001-1", "M1-0001-2"}


def test_a_truncated_last_line_costs_one_item_rather_than_the_file(tmp_path):
    target = tmp_path / "answers.jsonl"
    target.write_text('{"item_id": "N001"}\n{"item_id": "N0', encoding="utf-8")
    assert inference.answered_items(target) == {"N001"}


def test_a_run_that_never_started_has_answered_nothing(tmp_path):
    assert inference.answered_items(tmp_path / "absent.jsonl") == set()


def test_a_resumed_run_carries_the_invalid_answers_it_already_has(tmp_path):
    # Counting the current session alone would let a run stopped for its invalid
    # rate pass the same rule on the next attempt.
    target = tmp_path / "answers.jsonl"
    target.write_text(
        '{"item_id": "a", "valid": true}\n'
        '{"item_id": "b", "valid": false}\n'
        '{"item_id": "c", "valid": false}\n',
        encoding="utf-8",
    )
    assert inference.invalid_answers(target) == 2
    assert inference.invalid_answers(tmp_path / "absent.jsonl") == 0


def test_an_answer_is_written_with_the_timings_the_server_reported(tmp_path):
    # The efficiency profile is built from these lines and from nothing else, so
    # an answer written without them would cost a second pass over the pool.
    from scripts import run_inference

    recorder: list[dict] = []
    card = inference.pool()[0]
    run = next(
        entry
        for entry in inference.planned_runs()
        if entry["name"] == "natural_name_v1"
    )
    items = inference.item_set("natural")[:2]
    target = tmp_path / "answers.jsonl"

    async def answer() -> None:
        async with server_with(recorder) as server:
            await run_inference.answer_run(server, card, run, items, target)

    asyncio.run(answer())
    written = [json.loads(line) for line in target.read_text().splitlines()]
    assert [row["item_id"] for row in written] == [item.item_id for item in items]
    for row in written:
        assert row["total_sec"] == 3.5
        assert row["output_tokens"] == 8
        assert row["prompt_tokens"] == 310
        assert row["valid"] is True
        assert row["run"] == "natural_name_v1"


def test_a_model_stays_resident_between_two_items():
    # A model asked to stay resident for no time at all is reloaded for every
    # item, and the pass measures loading rather than inference.
    assert float(get("run.options.keep_alive")) > 0


# --- the pool and the planned runs -----------------------------------------


def test_the_pool_is_walked_from_the_smallest_artifact_upward():
    weights = [card.weights_gb for card in inference.pool()]
    assert weights == sorted(weights)
    assert len(weights) == len(get("models.pool"))


def test_a_tag_becomes_a_filename_without_losing_its_identity():
    cards = {card.tag: card.slug for card in inference.pool()}
    assert cards["openbmb/minicpm5:q4_K_M"] == "openbmb-minicpm5-q4_K_M"
    assert len(set(cards.values())) == len(cards)


def test_every_planned_run_names_a_task_a_view_and_an_item_set():
    for run in inference.planned_runs():
        assert run["task"] in prompting.TASKS
        assert run["view"] in prompting.VIEWS
        assert run["items"] in {"injected", "injected_m1", "natural"}
        assert isinstance(run["constrained"], bool)


def test_the_detection_task_is_planned_on_the_full_view_only():
    for run in inference.planned_runs():
        if run["task"] == prompting.DETECT:
            assert run["view"] == prompting.FULL


def test_the_stop_rule_is_suspended_on_the_unconstrained_runs_and_only_there():
    for run in inference.planned_runs():
        assert run["stop_on_invalid"] is bool(run["constrained"])


def test_the_four_views_are_all_asked_of_the_residual_class():
    views = {
        run["view"]
        for run in inference.planned_runs()
        if run["items"] == "injected_m1" and run["prompt_version"] == 1
    }
    assert views == set(prompting.VIEWS)


def test_the_prompt_is_varied_across_the_whole_pool_on_the_full_view():
    versions = {
        (run["items"], run["prompt_version"])
        for run in inference.planned_runs()
        if run["view"] == prompting.FULL and run["constrained"]
    }
    for items in ("injected", "injected_m1", "natural"):
        assert (items, 1) in versions and (items, 2) in versions


# --- the item sets ---------------------------------------------------------


def test_each_item_set_holds_what_the_design_says_it_holds():
    assert len(inference.item_set("injected")) == get("injected_set.total")
    residual = inference.item_set("injected_m1")
    assert len(residual) == 2 * get("injected_set.classes.M1.items")


def test_the_natural_set_is_asked_about_only_where_it_was_labelled():
    """A record past the labelling stop carries no gold label to score against."""
    asked = inference.item_set("natural")
    assert len(asked) == get("natural_set.labelled.records")
    assert len(asked) < get("natural_set.total")
    assert [item.item_id for item in asked] == sorted(item.item_id for item in asked)


def test_no_item_is_asked_twice_inside_a_set():
    for kind in ("injected", "injected_m1", "natural"):
        items = inference.item_set(kind)
        assert len({item.item_id for item in items}) == len(items)


def test_an_unknown_item_set_is_refused():
    with pytest.raises(ValueError):
        inference.item_set("everything")


def test_every_item_renders_under_every_view_it_is_asked_in():
    items = {kind: inference.item_set(kind)[:5] for kind in ("injected", "natural")}
    for run in inference.planned_runs():
        sample = items["natural" if run["items"] == "natural" else "injected"]
        for item in sample:
            rendered = prompting.render(
                run["task"], run["prompt_version"], item.shown, run["view"]
            )
            assert rendered.user


# --- the redesign's runs and item sets -------------------------------------


def test_the_redesign_runs_are_kept_apart_from_the_planned_runs():
    planned = {run["name"] for run in inference.planned_runs()}
    redesign = {run["name"] for run in inference.planned_runs("redesign")}
    assert redesign
    assert not planned & redesign


def test_an_unknown_plan_is_refused():
    with pytest.raises(ValueError):
        inference.planned_runs("everything")


def test_the_control_set_is_asked_whole():
    items = inference.item_set("residual_control")
    assert len(items) == 2 * get("residual_control.items")
    assert all(item.item_id.startswith("M1C-") for item in items)


def test_the_extension_asks_exactly_the_records_the_first_stop_left():
    first = {item.item_id for item in inference.item_set("natural")}
    extension = inference.item_set("natural_extension")
    ids = {item.item_id for item in extension}
    assert not first & ids
    drawn = get("natural_set.total") - get("natural_set.labelled.records")
    assert len([i for i in ids if int(i[1:]) <= get("natural_set.total")]) == drawn


def test_an_answer_records_the_serving_release_it_was_produced_under(tmp_path):
    from scripts import run_inference

    recorder: list[dict] = []
    card = inference.pool()[0]
    run = next(
        entry
        for entry in inference.planned_runs()
        if entry["name"] == "natural_name_v1"
    )
    items = inference.item_set("natural")[:1]
    target = tmp_path / "answers.jsonl"

    async def answer() -> None:
        async with server_with(recorder) as server:
            await run_inference.answer_run(server, card, run, items, target, "0.32.6")

    asyncio.run(answer())
    written = json.loads(target.read_text().splitlines()[0])
    assert written["server_version"] == "0.32.6"


def test_only_the_stack_measurement_is_exempt_from_the_stack_check():
    exempt = [
        run["name"]
        for plan in ("planned", "redesign")
        for run in inference.planned_runs(plan)
        if run.get("stack_check", True) is False
    ]
    assert exempt == ["natural_name_v1_replay"]


def test_the_adapted_model_is_asked_only_by_name_under_the_redesign_plan():
    from scripts import run_inference

    adapted = get("lora.served.tag")
    _, named = run_inference.selected(["m1c_name_full_v1"], [adapted], "redesign")
    assert [card.tag for card in named] == [adapted]
    _, everyone = run_inference.selected(["m1c_name_full_v1"], None, "redesign")
    assert adapted not in {card.tag for card in everyone}
    with pytest.raises(ValueError):
        run_inference.selected(["natural_name_v1"], [adapted], "planned")


def test_the_development_split_is_asked_alone():
    dev = inference.item_set("residual_control_dev")
    whole = inference.item_set("residual_control")
    assert dev
    assert len(dev) == round(get("residual_control.splits.dev") * len(whole) / 2) * 2
    assert {item.item_id for item in dev} < {item.item_id for item in whole}
