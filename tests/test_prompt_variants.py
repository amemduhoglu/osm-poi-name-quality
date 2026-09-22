"""Tests for the instruction variants prompt selection chooses among."""

from __future__ import annotations

import re

import pytest

from poi_audit import prompting
from poi_audit.config import get

VARIANTS = [v for v in get("prompt_selection.versions") if v != 1]
FORBIDDEN = re.compile(
    r"\b(percent|%|half|most|few|rare|often|usually|share|proportion|injected|corrupt|"
    r"constructed|benchmark|test set|dataset)\b",
    re.IGNORECASE,
)


@pytest.mark.parametrize("version", VARIANTS)
def test_a_variant_differs_from_version_one_in_its_instruction_alone(
    version: int,
) -> None:
    first = prompting.load_template(prompting.NAME, 1)
    variant = prompting.load_template(prompting.NAME, version)
    for key in ("system", "views", "exemplar_intro", "exemplar_ids", "response_schema"):
        assert variant[key] == first[key], key
    assert variant["instruction"] != first["instruction"]


@pytest.mark.parametrize("version", VARIANTS)
def test_a_variant_says_nothing_about_rates_or_construction(version: int) -> None:
    instruction = prompting.load_template(prompting.NAME, version)["instruction"]
    assert not FORBIDDEN.search(instruction), instruction


def test_the_instructions_are_distinct() -> None:
    texts = [
        prompting.load_template(prompting.NAME, v)["instruction"]
        for v in [1, *VARIANTS]
    ]
    assert len(set(texts)) == len(texts)
