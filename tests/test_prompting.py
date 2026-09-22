"""Tests for the scored prompts and the four views of the ablation.

Two things have to hold of every prompt this study sends. It must carry the task
and nothing about the data, because a model told how the set was built could
score well without reading a record. And a view must remove exactly what it says
it removes, because the input-dependence gate reads a difference between views
as a difference of input.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from poi_audit import prompting
from poi_audit.config import get, path

SHOWN = {
    "name": "Karşıyaka Kitabevi",
    "category": "shop=books",
    "addr:city": "İzmir",
    "addr:street": "Cemal Gürsel Caddesi",
    "phone": "+90 232 555 11 22",
    "opening_hours": "Mo-Sa 09:00-19:00",
    "website": "https://example.org",
    "lat": 38.4611,
    "lon": 27.1122,
}

# Words a prompt must not carry, because each would tell a model something about
# the data rather than about the task.
LEAKS = (
    "half",
    "50%",
    "900",
    "1800",
    "1,800",
    "corrupted",
    "injected",
    "balanced",
    "base rate",
    "equally likely",
    "M1",
    "gazetteer",
)


def templates() -> list[dict[str, Any]]:
    """Return every versioned template in the repository.

    Returns:
        The loaded templates.
    """
    return [
        prompting.load_template(task, version)
        for task in prompting.TASKS
        for version in (1, 2)
    ]


# --- what the prompt may say ----------------------------------------------


def test_no_template_says_anything_about_the_data():
    for template in templates():
        text = " ".join(
            str(template[key]) for key in ("system", "instruction") if key in template
        ).lower()
        for leak in LEAKS:
            assert leak.lower() not in text, (
                template["task"],
                template["version"],
                leak,
            )


def test_every_task_has_two_versions_so_the_prompt_can_be_varied():
    for task in prompting.TASKS:
        for version in (1, 2):
            assert prompting.load_template(task, version)["version"] == version


def test_the_two_versions_of_a_task_differ_in_their_instruction():
    for task in prompting.TASKS:
        first = prompting.load_template(task, 1)["instruction"]
        second = prompting.load_template(task, 2)["instruction"]
        assert first != second


def test_a_version_that_was_never_written_is_refused():
    with pytest.raises(FileNotFoundError):
        prompting.load_template(prompting.DETECT, 99)


def test_an_unknown_task_is_refused():
    with pytest.raises(ValueError):
        prompting.template_path("whatever", 1)


# --- the views ------------------------------------------------------------


def test_the_full_view_shows_the_record_as_it_is():
    assert prompting.apply_view(SHOWN, prompting.FULL) == SHOWN


def test_the_name_only_view_shows_the_name_and_the_coordinate():
    shown = prompting.apply_view(SHOWN, prompting.NAME_ONLY)
    assert set(shown) == {"name", "lat", "lon"}


def test_the_tags_only_view_removes_the_name_and_nothing_else():
    shown = prompting.apply_view(SHOWN, prompting.TAGS_ONLY)
    assert "name" not in shown
    assert set(shown) == set(SHOWN) - {"name"}


def test_the_neither_view_shows_the_coordinate_alone():
    shown = prompting.apply_view(SHOWN, prompting.NEITHER)
    assert set(shown) == {"lat", "lon"}


def test_every_view_keeps_the_coordinate():
    # Two of the injected classes are decided against position, so a view that
    # hid the coordinate would ask a question the task never asks.
    for view in prompting.VIEWS:
        shown = prompting.apply_view(SHOWN, view)
        assert shown["lat"] == SHOWN["lat"]
        assert shown["lon"] == SHOWN["lon"]


def test_an_unknown_view_is_refused():
    with pytest.raises(ValueError):
        prompting.apply_view(SHOWN, "sideways")


# --- the rendered prompt --------------------------------------------------


def test_the_name_is_absent_from_the_two_views_that_remove_it():
    for view in (prompting.TAGS_ONLY, prompting.NEITHER):
        rendered = prompting.render(prompting.NAME, 1, SHOWN, view)
        assert SHOWN["name"] not in rendered.user, view


def test_the_tags_are_absent_from_the_views_that_remove_them():
    for view in (prompting.NAME_ONLY, prompting.NEITHER):
        rendered = prompting.render(prompting.NAME, 1, SHOWN, view)
        assert SHOWN["phone"] not in rendered.user, view
        assert SHOWN["opening_hours"] not in rendered.user, view


def test_the_full_view_carries_every_value():
    rendered = prompting.render(prompting.NAME, 1, SHOWN, prompting.FULL)
    for value in SHOWN.values():
        assert str(value) in rendered.user


def test_the_detection_task_is_asked_on_the_full_record_only():
    with pytest.raises(ValueError):
        prompting.render(prompting.DETECT, 1, SHOWN, prompting.NAME_ONLY)


def test_the_same_item_renders_identically_twice():
    first = prompting.render(prompting.NAME, 1, SHOWN, prompting.FULL)
    second = prompting.render(prompting.NAME, 1, SHOWN, prompting.FULL)
    assert first == second


def test_the_record_is_serialised_in_a_fixed_key_order():
    shuffled = dict(reversed(list(SHOWN.items())))
    assert prompting.render_record(shuffled) == prompting.render_record(SHOWN)


def test_the_response_schema_constrains_the_answer():
    detect = prompting.render(prompting.DETECT, 1, SHOWN).schema
    assert detect["properties"]["verdict"]["enum"] == ["clean", "wrong"]
    assert detect["additionalProperties"] is False
    name = prompting.render(prompting.NAME, 1, SHOWN).schema
    assert name["properties"]["verdict"]["enum"] == ["belongs", "wrong"]
    assert "field" not in name["properties"]


def test_the_schema_names_every_field_an_item_can_show():
    # An item shows the paired keys and, when the record carries one, a numeric
    # tag from the declared range table. Both sources are asked for here: a
    # test that read the paired keys alone would pass while the schema left a
    # model unable to name the field a whole error class corrupts.
    keys = set(get("injected_set.construction.pairing.item_keys"))
    keys |= set(get("references.value_ranges"))
    for task, version in ((prompting.DETECT, 1), (prompting.DETECT, 2)):
        allowed = set(
            prompting.render(task, version, SHOWN).schema["properties"]["field"]["enum"]
        )
        assert keys <= allowed


def test_the_schema_can_name_the_field_of_every_corrupted_item():
    # The property that matters is about the finished set rather than about the
    # configuration: every field the item set actually corrupts has to be
    # expressible in the constrained answer, or a model is forced to name a
    # field it was not shown.
    items = path("data_processed") / "injected_set" / "items.jsonl"
    if not items.exists():  # the set is built by the item builder, not by the tests
        pytest.skip("injected item set not built")
    corrupted = {
        json.loads(line)["corrupted_field"]
        for line in items.read_text(encoding="utf-8").splitlines()
        if line.strip()
    } - {None}
    for version in (1, 2):
        allowed = set(
            prompting.render(prompting.DETECT, version, SHOWN).schema["properties"][
                "field"
            ]["enum"]
        )
        assert corrupted <= allowed


# --- the exemplars --------------------------------------------------------


def test_the_exemplars_are_shown_under_the_same_view_as_the_item():
    rendered = prompting.render(prompting.NAME, 1, SHOWN, prompting.NEITHER)
    exemplars = prompting.load_exemplars()
    for entry in exemplars.values():
        assert str(entry["record"]["name"]) not in rendered.user


def test_every_exemplar_a_template_asks_for_exists():
    available = prompting.load_exemplars()
    for template in templates():
        for identifier in template["exemplar_ids"]:
            assert str(identifier) in available


def test_no_exemplar_value_appears_in_the_item_set():
    """No identifying exemplar value is shared with a scored item.

    The category is exempt and has to be: it is drawn from OpenStreetMap's own
    closed vocabulary, so an exemplar carrying a category no record carries
    would not be a plausible record at all. Every other field is a value a
    reader could recognise, and none of them may be shared.
    """
    target = get("paths.data_processed")
    items = f"{target}/injected_set/items.jsonl"
    try:
        with open(items, encoding="utf-8") as handle:
            written = handle.read()
    except FileNotFoundError:
        pytest.skip("injected set not built")
    for entry in prompting.load_exemplars().values():
        for key, value in entry["record"].items():
            if key == "category" or not isinstance(value, str) or len(value) <= 6:
                continue
            assert value not in written, (entry["id"], key, value)


def test_the_name_task_labels_an_exemplar_by_whether_its_name_is_wrong():
    exemplars = prompting.load_exemplars()
    rendered = prompting.render_exemplar(
        exemplars["E2"], prompting.NAME, prompting.FULL
    )
    assert json.loads(rendered.splitlines()[-1])["verdict"] == "wrong"
    rendered = prompting.render_exemplar(
        exemplars["E1"], prompting.NAME, prompting.FULL
    )
    assert json.loads(rendered.splitlines()[-1])["verdict"] == "belongs"


def test_the_exemplars_follow_the_item_s_view_by_default() -> None:
    """The worked example demonstrates the question actually asked."""
    shown = {
        "name": "Bakkerij Vermeulen",
        "category": "shop=bakery",
        "lat": 51.9,
        "lon": 4.47,
    }
    rendered = prompting.render("name", 1, shown, prompting.NAME_ONLY)
    assert "craft" not in rendered.user
    assert "amenity" not in rendered.user


def test_the_exemplars_can_be_held_at_the_full_record() -> None:
    """An ablation that changes the item and the example changes two things.

    The gate asks whether a score moves with the input. With the example
    reduced beside the item, a score that falls may have fallen because the
    demonstration stopped demonstrating anything, which is a different finding.
    Holding the example at the full record leaves the item as the only thing
    the ablation changes.
    """
    shown = {
        "name": "Bakkerij Vermeulen",
        "category": "shop=bakery",
        "lat": 51.9,
        "lon": 4.47,
    }
    rendered = prompting.render(
        "name", 1, shown, prompting.NAME_ONLY, exemplar_view=prompting.FULL
    )
    assert "craft=bookbinder" in rendered.user
    assert '"category"' in rendered.user.split("Bakkerij Vermeulen")[0]


def test_the_item_stays_reduced_when_the_exemplars_are_held() -> None:
    """Holding the examples does not restore the item's own removed fields."""
    shown = {
        "name": "Bakkerij Vermeulen",
        "category": "shop=bakery",
        "lat": 51.9,
        "lon": 4.47,
    }
    rendered = prompting.render(
        "name", 1, shown, prompting.NAME_ONLY, exemplar_view=prompting.FULL
    )
    item = rendered.user.split("Bakkerij Vermeulen")[1]
    assert "shop=bakery" not in item


# --- the retrieval-augmented version -----------------------------------------

EVIDENCE = {
    "radius_m": 100,
    "radius_km": 10,
    "nearest": [{"name": "Old Mill Museum", "distance_m": 42}],
    "similar": [],
}


def test_version_four_differs_from_version_one_only_by_the_evidence_block() -> None:
    shown = {"name": "Kafe Roma", "category": "amenity=cafe", "lat": 1.0, "lon": 2.0}
    first = prompting.render(prompting.NAME, 1, shown)
    fourth = prompting.render(prompting.NAME, 4, shown, evidence=EVIDENCE)
    assert fourth.system == first.system
    assert fourth.schema == first.schema
    assert fourth.user.startswith(first.user + "\n\n")
    block = fourth.user[len(first.user) + 2 :]
    assert "- Old Mill Museum (42 m)" in block
    assert block.count("(none)") == 1


def test_evidence_is_refused_by_a_template_without_an_evidence_block() -> None:
    shown = {"name": "Kafe Roma", "lat": 1.0, "lon": 2.0}
    with pytest.raises(ValueError):
        prompting.render(prompting.NAME, 1, shown, evidence=EVIDENCE)


def test_version_four_refuses_to_render_without_evidence() -> None:
    shown = {"name": "Kafe Roma", "lat": 1.0, "lon": 2.0}
    with pytest.raises(ValueError):
        prompting.render(prompting.NAME, 4, shown)
