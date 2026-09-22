"""Render the scored prompts, and the four views the ablation asks for.

A score is a property of a model **and** of the prompt that produced it, so the
prompt is versioned, deposited, and varied across the whole pool rather than a
subset. This module turns a versioned template and an item into the exact
messages a model is sent, so that the article can say what was asked and a
reader can reproduce it.

Two rules govern what may reach a model, and both are enforced here rather than
trusted to a template:

    Nothing beyond the task description. The model is never told how many items
    are wrong, what share of them are wrong, which error classes exist, how the
    set was built, or what the base rate of a field is. A model told that half
    the set is corrupted could score well without reading a record.

    The view is the input. The residual class carries the input-dependence gate
    natively: the record is shown in full, then as name only, then as tags only,
    then as neither. A model that scores above chance on the view holding
    neither the name nor the tags is not reading the record, and its score on
    the other views cannot be read as evidence that it can.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"

DETECT = "detect"
NAME = "name"
TASKS = (DETECT, NAME)

FULL = "full"
NAME_ONLY = "name_only"
TAGS_ONLY = "tags_only"
NEITHER = "neither"
VIEWS = (FULL, NAME_ONLY, TAGS_ONLY, NEITHER)

# The coordinate is shown in every view, including the one that shows neither
# the name nor the tags. Two of the injected classes are decided against
# position, and a view that hid it would test something the task never asks.
COORDINATE_KEYS = ("lat", "lon")


@dataclass(frozen=True)
class Prompt:
    """One rendered prompt, exactly as a model receives it.

    Attributes:
        task: ``detect`` or ``name``.
        version: The template version the text came from.
        view: Which view of the record was shown.
        system: The system message.
        user: The user message, exemplars included.
        schema: The JSON schema the response is constrained to.
    """

    task: str
    version: int
    view: str
    system: str
    user: str
    schema: dict[str, Any]


def template_path(task: str, version: int) -> Path:
    """Return the file one versioned template lives in.

    Args:
        task: ``detect`` or ``name``.
        version: The template version.

    Returns:
        The path.

    Raises:
        ValueError: If the task is not one of the two.
    """
    if task not in TASKS:
        raise ValueError(f"unknown task: {task}")
    return PROMPT_DIR / f"{task}.v{version}.json"


def load_template(task: str, version: int) -> dict[str, Any]:
    """Read one versioned prompt template.

    Args:
        task: ``detect`` or ``name``.
        version: The template version.

    Returns:
        The template.

    Raises:
        FileNotFoundError: If that version was never written.
    """
    target = template_path(task, version)
    if not target.exists():
        raise FileNotFoundError(f"prompt template not found: {target}")
    return json.loads(target.read_text(encoding="utf-8"))


def load_exemplars() -> dict[str, dict[str, Any]]:
    """Read the hand-written exemplar records, keyed by identifier.

    Returns:
        Each exemplar by its identifier. Every value in them is invented, so no
        item can be answered by recalling one.

    Raises:
        FileNotFoundError: If the exemplar file is missing.
    """
    target = PROMPT_DIR / "exemplars.v1.json"
    if not target.exists():
        raise FileNotFoundError(f"exemplars not found: {target}")
    entries = json.loads(target.read_text(encoding="utf-8"))["exemplars"]
    # The generated exemplars of the few-shot arm live in their own file, so the
    # hand-written file every earlier prompt reads is not touched.
    generated = PROMPT_DIR / "exemplars.generated.v1.json"
    if generated.exists():
        entries += json.loads(generated.read_text(encoding="utf-8"))["exemplars"]
    return {str(entry["id"]): entry for entry in entries}


def apply_view(shown: dict[str, Any], view: str) -> dict[str, Any]:
    """Return the fields one view of a record shows.

    Args:
        shown: The record's shown fields, coordinate included.
        view: One of the four views.

    Returns:
        The fields that view leaves visible. The coordinate survives every view;
        the name and the remaining tags are removed according to the view.

    Raises:
        ValueError: If the view is not one of the four.
    """
    if view not in VIEWS:
        raise ValueError(f"unknown view: {view}")
    coordinate = {key: shown[key] for key in COORDINATE_KEYS if key in shown}
    if view == FULL:
        return dict(shown)
    if view == NAME_ONLY:
        return {**({"name": shown["name"]} if "name" in shown else {}), **coordinate}
    if view == TAGS_ONLY:
        return {key: value for key, value in shown.items() if key != "name"}
    return coordinate


def render_record(shown: dict[str, Any]) -> str:
    """Return one record as the model is shown it.

    Args:
        shown: The fields the view leaves visible.

    Returns:
        The record as JSON with its keys in a fixed order, so that two runs of
        the same item are byte-identical and a difference between runs is a
        difference of model rather than of serialisation.
    """
    return json.dumps(shown, ensure_ascii=False, sort_keys=True, indent=2)


def render_exemplar(entry: dict[str, Any], task: str, view: str) -> str:
    """Return one worked example as the model is shown it.

    Args:
        entry: The exemplar.
        task: ``detect`` or ``name``.
        view: The view the item will be shown under, applied to the exemplar
            too, so that the example demonstrates the question actually asked.

    Returns:
        The exemplar's record and its answer.
    """
    shown = apply_view(dict(entry["record"]), view)
    if task == DETECT:
        answer = {"verdict": "clean" if entry["label"] == "clean" else "wrong"}
        answer["field"] = entry["field"] if answer["verdict"] == "wrong" else None
    else:
        wrong_name = entry["label"] == "wrong" and entry["field"] == "name"
        answer = {"verdict": "wrong" if wrong_name else "belongs"}
    return f"{render_record(shown)}\n{json.dumps(answer, sort_keys=True)}"


def exemplars_for(template: dict[str, Any], task: str, view: str) -> list[str]:
    """Return the worked examples one template asks for.

    Args:
        template: The loaded template.
        task: ``detect`` or ``name``.
        view: The view to render them under.

    Returns:
        One rendered example per configured identifier.
    """
    available = load_exemplars()
    return [
        render_exemplar(available[str(identifier)], task, view)
        for identifier in template["exemplar_ids"]
    ]


def render_evidence(template: dict[str, Any], evidence: dict[str, Any]) -> str:
    """Return the retrieved-evidence block a template asks for.

    Args:
        template: A template carrying the evidence wording.
        evidence: The entries retrieved for one item: ``nearest`` with each
            entry's name and distance in metres, ``similar`` with each entry's
            name and distance in kilometres, and the radii they were drawn from.

    Returns:
        The block, one line per entry, with the template's word for an empty
        list where a search found nothing.
    """
    none = str(template["evidence_none"])
    nearest = [
        f"- {entry['name']} ({entry['distance_m']} m)" for entry in evidence["nearest"]
    ]
    similar = [
        f"- {entry['name']} ({entry['distance_km']} km away)"
        for entry in evidence["similar"]
    ]
    return "\n".join(
        [
            str(template["evidence_intro"]),
            str(template["evidence_nearest"]).format(radius_m=evidence["radius_m"]),
            *(nearest or [none]),
            str(template["evidence_similar"]).format(radius_km=evidence["radius_km"]),
            *(similar or [none]),
        ]
    )


def render(
    task: str,
    version: int,
    shown: dict[str, Any],
    view: str = FULL,
    exemplar_view: str | None = None,
    evidence: dict[str, Any] | None = None,
) -> Prompt:
    """Render one item into the prompt a model is sent.

    Args:
        task: ``detect`` or ``name``.
        version: The template version.
        shown: The item's shown fields, as the item set wrote them.
        view: Which view of the record to show. The detection task is always
            asked on the full record; the residual question is asked on all
            four.
        exemplar_view: The view to render the worked examples under, or None to
            render them under the item's own view. Holding them at the full
            record leaves the item as the only thing an ablation changes, which
            separates a score that fell because the model stopped reading the
            input from one that fell because the demonstration stopped
            demonstrating anything.
        evidence: Entries retrieved for the item, required by a template that
            carries an evidence block and refused by one that does not.

    Returns:
        The rendered prompt.

    Raises:
        ValueError: If the task is asked under a view it does not define, or if
            evidence is given to a template without an evidence block or withheld
            from one with it.
    """
    template = load_template(task, version)
    if task == DETECT and view != FULL:
        raise ValueError("the detection task is asked on the full record only")
    if task == NAME and view not in template["views"]:
        raise ValueError(f"the template defines no view {view}")

    parts: list[str] = [str(template["instruction"])]
    if task == NAME:
        parts.append(str(template["views"][view]))
    parts.append(str(template["exemplar_intro"]))
    parts.extend(exemplars_for(template, task, exemplar_view or view))
    parts.append(render_record(apply_view(shown, view)))
    if ("evidence_intro" in template) != (evidence is not None):
        raise ValueError(
            f"template {task} v{version} and the evidence given do not match"
        )
    if evidence is not None:
        parts.append(render_evidence(template, evidence))

    return Prompt(
        task=task,
        version=int(template["version"]),
        view=view,
        system=str(template["system"]),
        user="\n\n".join(parts),
        schema=dict(template["response_schema"]),
    )
