"""Tests for the replay comparison that decides which models enter the redesign."""

from __future__ import annotations

import pytest

from scripts.score_replay import compare


def answer(item: str, verdict: str | None, raw: str = "x") -> dict:
    return {
        "item_id": item,
        "parsed": {"verdict": verdict} if verdict else None,
        "valid": verdict is not None,
        "response_raw": raw,
    }


def test_identical_answers_agree_fully() -> None:
    old = [answer("N001", "belongs"), answer("N002", "wrong")]
    result = compare(old, list(old), minimum=0.95)
    assert result["verdict_agreement"] == pytest.approx(1.0)
    assert result["raw_identical"] == pytest.approx(1.0)
    assert result["kept"]


def test_a_moved_verdict_counts_against_agreement() -> None:
    old = [answer(f"N{i:03d}", "belongs") for i in range(20)]
    new = [answer(f"N{i:03d}", "wrong" if i == 0 else "belongs") for i in range(20)]
    result = compare(old, new, minimum=0.95)
    assert result["verdict_agreement"] == pytest.approx(0.95)
    assert result["changed"] == 1
    assert result["kept"]
    assert not compare(old, new, minimum=0.96)["kept"]


def test_an_invalid_answer_matches_only_another_invalid_answer() -> None:
    old = [answer("N001", None), answer("N002", "belongs")]
    new = [answer("N001", None), answer("N002", None)]
    assert compare(old, new, minimum=0.5)["verdict_agreement"] == pytest.approx(0.5)


def test_items_missing_from_the_replay_are_refused() -> None:
    with pytest.raises(ValueError):
        compare([answer("N001", "belongs")], [], minimum=0.95)


def test_a_changed_prompt_token_count_is_reported() -> None:
    old = [{**answer("N001", "belongs"), "prompt_tokens": 430}]
    new = [{**answer("N001", "belongs"), "prompt_tokens": 512}]
    result = compare(old, new, minimum=0.95)
    assert result["prompt_tokens_identical"] == 0.0
    assert result["kept"]
