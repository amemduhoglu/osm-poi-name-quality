"""Tests for the generated few-shot exemplars and version 3 of the prompt."""

from __future__ import annotations

import json

import pytest

from poi_audit import prompting, screen
from poi_audit.config import get, path
from poi_audit.corpus import fold
from scripts.build_exemplars import TARGET, build

IDENTIFYING = {"name", "addr:city", "addr:street", "opening_hours"}


def generated() -> list[dict]:
    if not TARGET.exists():
        pytest.skip("exemplars not generated")
    return json.loads(TARGET.read_text("utf-8"))["exemplars"]


def test_labels_follow_the_construction() -> None:
    entries = generated()
    assert sum(1 for e in entries if e["label"] == "wrong") == get("few_shot.borrowed")
    assert len(entries) == get("few_shot.shots")


def test_the_file_is_what_the_generator_writes() -> None:
    rebuilt = build(int(get("few_shot.seed")), screen.load_category_tokens())
    assert rebuilt == generated()


def test_no_generated_name_carries_a_category_token() -> None:
    tokens = screen.load_category_tokens()
    for entry in generated():
        assert not any(t in tokens for t in fold(entry["record"]["name"]).split())


def test_no_generated_value_appears_in_any_item_set() -> None:
    written = {
        fold(str(entry["record"][key]))
        for entry in generated()
        for key in IDENTIFYING
        if key in entry["record"]
    }
    processed = path("data_processed")
    sources = [
        (processed / "injected_set" / "items.jsonl", "record"),
        (processed / "injected_set" / "residual_control.jsonl", "record"),
        (processed / "natural_set" / "records.jsonl", "shown"),
        (processed / "natural_set" / "records_phase2.jsonl", "shown"),
    ]
    for source, key in sources:
        with source.open(encoding="utf-8") as handle:
            for row in map(json.loads, handle):
                for field, value in row[key].items():
                    if field in IDENTIFYING and isinstance(value, str):
                        assert fold(value) not in written, (source.name, field, value)


def test_every_category_is_one_the_corpus_uses() -> None:
    with (path("data_processed") / "injected_set" / "items.jsonl").open() as handle:
        categories = {json.loads(line)["record"].get("category") for line in handle}
    assert all(entry["record"]["category"] in categories for entry in generated())


def test_version_three_differs_from_version_one_in_the_examples_alone() -> None:
    first = prompting.load_template(prompting.NAME, 1)
    third = prompting.load_template(prompting.NAME, 3)
    same = {"system", "instruction", "views", "response_schema"}
    assert all(first[key] == third[key] for key in same)
    assert third["exemplar_ids"][:2] == first["exemplar_ids"]
    available = prompting.load_exemplars()
    assert all(identifier in available for identifier in third["exemplar_ids"])
