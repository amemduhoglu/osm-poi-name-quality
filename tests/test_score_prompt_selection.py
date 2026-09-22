"""Tests for the prompt selection rule."""

from __future__ import annotations

from scripts.score_prompt_selection import choose


def test_the_highest_j_is_chosen() -> None:
    assert choose({1: 0.30, 5: 0.41, 6: 0.12}) == 5


def test_a_tie_goes_to_the_lowest_version() -> None:
    assert choose({7: 0.4, 1: 0.4, 5: 0.2}) == 1
