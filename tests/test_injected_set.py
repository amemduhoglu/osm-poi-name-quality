"""Tests for the injected item set and the corruptions that build it.

The construction rules in the study plan are not commentary: each one repairs a
known failure mode of this kind of item set, and each is checked here against
the finished set rather than against the intention of the code. A rule that
cannot be checked would be a rule nobody has to keep.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from typing import Any

import pytest

from poi_audit import rules
from poi_audit.config import get, seed_everything
from poi_audit.corpus import distance_km, fold
from poi_audit.corruptions import (
    corrupt_calling_code,
    corrupt_opening_hours,
    corrupt_phone_syntax,
    corrupt_value_range,
    corrupt_website,
)
from poi_audit.references import expected_calling_code, other_calling_codes
from scripts.build_injected_set import CLASS_ORDER, allocate
from scripts.score_rule_baseline import item_path

EXEMPLAR_FILE = "prompts/exemplars.v1.json"


@pytest.fixture(scope="module")
def items() -> list[dict[str, Any]]:
    """Read the injected set, skipping the module when it has not been built."""
    target = item_path()
    if not target.exists():
        pytest.skip("injected set not built; run scripts/build_injected_set.py")
    with target.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# --- the corruptions, on their own ----------------------------------------


def test_every_phone_malformation_is_flagged_by_the_rule():
    seed_everything()
    for kind in get("injected_set.construction.s1.malformations"):
        applied = corrupt_phone_syntax("+90 232 445 12 34", kind)
        assert applied.injected != applied.original, kind
        assert rules.phone_syntax(applied.injected).outcome == rules.FLAG, kind


def test_every_opening_hours_malformation_is_flagged_by_the_rule():
    seed_everything()
    for kind in get("injected_set.construction.s2.malformations"):
        applied = corrupt_opening_hours("Mo-Fr 09:00-17:00", kind)
        assert applied.injected != applied.original, kind
        assert rules.opening_hours_syntax(applied.injected).outcome == rules.FLAG, kind


def test_every_website_malformation_is_flagged_by_the_rule():
    seed_everything()
    for kind in get("injected_set.construction.s3.malformations"):
        applied = corrupt_website("https://www.example.ie/menu", kind)
        assert applied.injected != applied.original, kind
        assert rules.website_syntax(applied.injected).outcome == rules.FLAG, kind


def test_every_range_corruption_lands_outside_the_published_range():
    seed_everything()
    for tag in get("references.value_ranges"):
        for kind in get("injected_set.construction.s4.out_of_range_kinds"):
            applied = corrupt_value_range(tag, "4", kind)
            verdicts = rules.value_range({tag: applied.injected})
            assert [v.outcome for v in verdicts] == [rules.FLAG], (tag, kind)


def test_the_calling_code_corruption_changes_only_the_calling_code():
    seed_everything()
    original = "+90 232 445 12 34"
    applied = corrupt_calling_code(original, other_calling_codes("TR"))
    assert applied.injected.endswith("232 445 12 34")
    assert rules.phone_country_code(applied.injected, "TR").outcome == rules.FLAG


def test_the_calling_code_corruption_does_not_draw_one_fixed_value():
    seed_everything()
    drawn = {
        corrupt_calling_code("+90 232 445 12 34", other_calling_codes("TR")).kind
        for _ in range(40)
    }
    assert len(drawn) > 1


def test_no_corruption_reuses_the_records_own_calling_code():
    seed_everything()
    mine = expected_calling_code("TR")
    assert all(code != mine for code in other_calling_codes("TR"))


def test_the_allocation_spreads_a_class_across_every_city():
    random.seed(int(get("project.seed")))
    cities = [str(city["city"]) for city in get("corpus.cities")]
    counts = allocate(100, cities)
    assert sum(counts.values()) == 100
    assert min(counts.values()) >= 100 // len(cities)
    assert max(counts.values()) - min(counts.values()) <= 1


# --- the finished set -----------------------------------------------------


def test_the_set_holds_the_sizes_the_plan_fixed(items):
    assert len(items) == int(get("injected_set.total"))
    corrupted = [item for item in items if item["corrupted"]]
    clean = [item for item in items if not item["corrupted"]]
    assert len(clean) == int(get("injected_set.clean_items"))
    assert len(corrupted) == len(clean)
    per_class = Counter(item["class"] for item in corrupted)
    for name in CLASS_ORDER:
        assert per_class[name] == int(get(f"injected_set.classes.{name}.items")), name


def test_every_pair_matches_on_city_and_on_field_presence(items):
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        pairs[item["pair_id"]].append(item)
    assert len(pairs) == int(get("injected_set.clean_items"))
    for pair_id, members in pairs.items():
        assert len(members) == 2, pair_id
        assert sum(1 for m in members if m["corrupted"]) == 1, pair_id
        assert members[0]["city"] == members[1]["city"], pair_id
        assert members[0]["field_signature"] == members[1]["field_signature"], pair_id


def test_field_presence_cannot_tell_a_corrupted_item_from_a_clean_one(items):
    corrupted = Counter(
        tuple(item["field_signature"]) for item in items if item["corrupted"]
    )
    clean = Counter(
        tuple(item["field_signature"]) for item in items if not item["corrupted"]
    )
    assert corrupted == clean


def test_no_record_is_used_twice(items):
    used = [(item["osm_type"], item["osm_id"]) for item in items]
    assert len(set(used)) == len(used)


def test_the_items_are_spread_evenly_across_the_cities(items):
    per_city = Counter(item["city"] for item in items)
    cities = [str(city["city"]) for city in get("corpus.cities")]
    assert set(per_city) == set(cities)
    assert max(per_city.values()) - min(per_city.values()) <= 2 * len(CLASS_ORDER)


def test_no_class_corrupts_its_field_with_one_fixed_value(items):
    by_class: dict[str, set[str]] = defaultdict(set)
    for item in items:
        if item["corrupted"]:
            by_class[item["class"]].add(item["corruption"]["injected"])
    for name, values in by_class.items():
        assert len(values) > 1, name


def test_every_corruption_replaces_a_value_the_record_already_carried(items):
    for item in items:
        if not item["corrupted"]:
            continue
        corruption = item["corruption"]
        assert corruption["original"], item["item_id"]
        assert corruption["injected"] != corruption["original"], item["item_id"]
        assert corruption["field"] in item["record"], item["item_id"]
        assert item["record"][corruption["field"]] == corruption["injected"]


def test_the_residual_class_borrows_a_name_from_the_same_city(items):
    radius_km = float(get("corpus.candidate_selection.radius_m")) / 1000.0
    minimum = float(get("injected_set.construction.m1.donor_min_distance_m")) / 1000.0
    for item in items:
        if item["class"] != "M1" or not item["corrupted"]:
            continue
        donor = item["corruption"]["donor"]
        assert donor["city"] == item["city"], item["item_id"]
        assert donor["distance_km"] >= minimum, item["item_id"]
        assert donor["distance_km"] <= 2 * radius_km, item["item_id"]


def test_the_residual_class_carries_no_category_indicative_token(items):
    from scripts.score_rule_baseline import read_category_tokens

    tokens = read_category_tokens()
    for item in items:
        if item["class"] != "M1" or not item["corrupted"]:
            continue
        name = str(item["record"]["name"])
        assert not any(token in tokens for token in fold(name).split()), item["item_id"]


def test_the_address_class_borrows_a_distant_place(items):
    minimum = float(get("injected_set.construction.m3.donor_min_distance_km"))
    for item in items:
        if item["class"] != "M3" or not item["corrupted"]:
            continue
        assert item["corruption"]["donor"]["distance_km"] >= minimum, item["item_id"]


def test_the_rules_settle_the_classes_they_are_meant_to_settle(items):
    from poi_audit.references import load_gazetteer
    from scripts.score_rule_baseline import as_record

    gazetteer = load_gazetteer()
    for item in items:
        if not item["corrupted"] or item["class"] not in {"S1", "S2", "S3", "S4", "M2"}:
            continue
        verdicts = rules.baseline(as_record(item), gazetteer)
        field = item["corrupted_field"]
        assert any(
            v.outcome == rules.FLAG and v.field == field for v in verdicts
        ), item["item_id"]


def test_no_item_shows_a_field_outside_the_item_schema(items):
    allowed = set(get("injected_set.construction.pairing.item_keys"))
    allowed |= set(get("references.value_ranges"))
    allowed |= {"lat", "lon"}
    for item in items:
        assert set(item["record"]) <= allowed, item["item_id"]


def test_the_coordinates_are_valid(items):
    for item in items:
        assert -90.0 <= float(item["record"]["lat"]) <= 90.0, item["item_id"]
        assert -180.0 <= float(item["record"]["lon"]) <= 180.0, item["item_id"]


def read_exemplars() -> list[dict[str, Any]]:
    """Read the hand-written exemplar records."""
    from poi_audit.config import REPO_ROOT

    return json.loads((REPO_ROOT / EXEMPLAR_FILE).read_text(encoding="utf-8"))[
        "exemplars"
    ]


def test_the_exemplars_share_no_identifying_value_with_any_item(items):
    # The fields compared are the ones that could carry an answer. A category, a
    # house number and a numeric tag are shared vocabulary rather than content:
    # every corpus holds "3" as a house number, and an exemplar that showed no
    # category would not be a point-of-interest record at all. The fields left
    # out are named here rather than in a comment inside the loop, so that
    # widening the exemption is a visible edit.
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
        for exemplar in read_exemplars()
        for key, value in exemplar["record"].items()
        if isinstance(value, str) and key in identifying
    }
    assert written
    for item in items:
        for key, value in item["record"].items():
            if key in identifying and isinstance(value, str) and fold(value) in written:
                raise AssertionError(f"{item['item_id']} repeats an exemplar {key}")


def test_no_corruption_value_appears_in_an_exemplar(items):
    # Compared as written rather than folded. The fold exists to match place
    # names across diacritics and punctuation, and applying it here would equate
    # a level of -17 with a house number of 17, which share nothing an exemplar
    # could teach a model.
    written = {
        str(value).strip()
        for exemplar in read_exemplars()
        for value in exemplar["record"].values()
        if isinstance(value, str)
    }
    for item in items:
        if not item["corrupted"]:
            continue
        assert str(item["corruption"]["injected"]).strip() not in written, item[
            "item_id"
        ]


def test_the_pre_existing_flags_are_recorded(items):
    # Not an assertion about how many there are: a clean item is clean of
    # injected error, not of every error a mapper ever made, and the analysis
    # needs to know which items those are rather than to be told there are none.
    assert all("preexisting_flags" in item for item in items)
    assert all(isinstance(item["preexisting_flags"], list) for item in items)


def test_the_donor_distance_is_measured_the_same_way_everywhere():
    # The corpus discs are 10 km, so two records in one city cannot be further
    # apart than 20 km; a donor beyond that would mean the pool leaked a city.
    assert distance_km(38.4237, 27.1428, 38.4237, 27.1428) == 0.0
    assert distance_km(0.0, 0.0, 0.0, 1.0) == pytest.approx(111.19, abs=0.5)
