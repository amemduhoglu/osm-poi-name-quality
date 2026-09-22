"""Properties the finished residual control set must hold, read off the built files.

These run on the tracked outputs, so a rebuild that broke a property fails here
rather than in a score. They are skipped when the set has not been built.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from poi_audit.config import get, path
from poi_audit.corpus import fold
from scripts.build_residual_control import assign_splits

BUILT = path("data_processed") / "injected_set" / "residual_control.jsonl"
VISIBLE = path("data_processed") / "injected_set" / "residual_control_visible.json"
INJECTED = path("data_processed") / "injected_set" / "items.jsonl"


@pytest.fixture(scope="module")
def items() -> list[dict]:
    if not BUILT.exists():
        pytest.skip("control set not built")
    with BUILT.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@pytest.fixture(scope="module")
def visible() -> dict:
    if not VISIBLE.exists():
        pytest.skip("control set not built")
    return json.loads(VISIBLE.read_text(encoding="utf-8"))


def test_counts_match_the_configuration(items: list[dict]) -> None:
    corrupted = [item for item in items if item["corrupted"]]
    assert len(corrupted) == get("residual_control.items")
    assert len(items) == 2 * len(corrupted)


def test_every_pair_has_one_corrupted_and_one_clean_member(items: list[dict]) -> None:
    by_pair: dict[str, list[bool]] = {}
    for item in items:
        by_pair.setdefault(item["pair_id"], []).append(item["corrupted"])
    assert all(sorted(members) == [False, True] for members in by_pair.values())


def test_no_shown_name_repeats_in_the_visible_part(
    items: list[dict], visible: dict
) -> None:
    counts = {
        city: Counter(fold(name) for name in part["names"] if name)
        for city, part in visible.items()
    }
    for item in items:
        shown = fold(item["record"]["name"])
        allowed = 0 if item["corrupted"] else 1
        assert counts[item["city"]][shown] == allowed, item["item_id"]


def test_every_member_is_a_visible_record(items: list[dict], visible: dict) -> None:
    identities = {
        city: {tuple(identity) for identity in part["records"]}
        for city, part in visible.items()
    }
    for item in items:
        assert (item["osm_type"], item["osm_id"]) in identities[item["city"]]


def test_no_record_is_shared_with_the_injected_set(items: list[dict]) -> None:
    with INJECTED.open(encoding="utf-8") as handle:
        spent = {
            (row["osm_type"], row["osm_id"])
            for row in (json.loads(line) for line in handle if line.strip())
        }
    assert not {(item["osm_type"], item["osm_id"]) for item in items} & spent


def test_pairs_never_straddle_splits(items: list[dict]) -> None:
    by_pair: dict[str, set[str]] = {}
    for item in items:
        by_pair.setdefault(item["pair_id"], set()).add(item["split"])
    assert all(len(splits) == 1 for splits in by_pair.values())


def test_split_sizes_follow_the_configured_shares() -> None:
    shares = get("residual_control.splits")
    rows = [{"pair_id": f"P{i:03d}"} for i in range(300) for _ in range(2)]
    assign_splits(rows, seed=42)
    pairs = Counter(row["split"] for row in rows[::2])
    assert pairs == {name: round(share * 300) for name, share in shares.items()}


def test_the_exemplars_share_no_identifying_value_with_the_new_sets(
    items: list[dict],
) -> None:
    """The rule the injected set is held to, applied to the redesign's sets."""
    from poi_audit import prompting

    identifying = {
        "name",
        "addr:city",
        "addr:street",
        "phone",
        "opening_hours",
        "website",
    }
    written = {
        fold(value)
        for entry in prompting.load_exemplars().values()
        for key, value in entry["record"].items()
        if isinstance(value, str) and key in identifying
    }
    phase2 = path("data_processed") / "natural_set" / "records_phase2.jsonl"
    shown = [item["record"] for item in items]
    if phase2.exists():
        with phase2.open(encoding="utf-8") as handle:
            shown += [json.loads(line)["shown"] for line in handle if line.strip()]
    for record in shown:
        for key, value in record.items():
            if key in identifying and isinstance(value, str):
                assert fold(value) not in written, (key, value)
