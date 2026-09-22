"""Tests for the training file of the low-rank adaptation arm."""

from __future__ import annotations

import json

import pytest

from poi_audit.config import path
from scripts.build_lora_data import example_for

BUILT = path("data_processed") / "lora" / "train.jsonl"


def test_an_example_is_the_pool_prompt_with_the_schema_answer() -> None:
    item = {
        "item_id": "M1C-0001-1",
        "corrupted": True,
        "record": {
            "name": "Kafe Roma",
            "category": "amenity=cafe",
            "lat": 1.0,
            "lon": 2.0,
        },
    }
    example = example_for(item)
    assert example["messages"][0]["role"] == "system"
    assert example["messages"][1]["role"] == "user"
    assert "Kafe Roma" in example["messages"][1]["content"]
    assert json.loads(example["messages"][2]["content"]) == {"verdict": "wrong"}


def test_the_built_file_holds_the_training_split_and_nothing_else() -> None:
    if not BUILT.exists():
        pytest.skip("training file not built")
    source = path("data_processed") / "injected_set" / "residual_control.jsonl"
    with source.open(encoding="utf-8") as handle:
        split = {row["item_id"]: row["split"] for row in map(json.loads, handle)}
    with BUILT.open(encoding="utf-8") as handle:
        ids = [json.loads(line)["item_id"] for line in handle]
    assert ids
    assert all(split.get(item_id) == "train" for item_id in ids)
    assert len(ids) == sum(1 for value in split.values() if value == "train")
