"""The state a labelling session keeps, kept apart from how it is served.

The protocol makes three demands that a paper form cannot enforce and a program
can: the items are labelled in one fixed order, an item is not revisited once it
is past, and every label is written to disk as it is given so that a session
resumed after an interruption continues where it stopped rather than starting
again.

Everything here is about one annotator's own file. No function in this module
can read another annotator's labels, which is what keeps the two readings
independent.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from poi_audit.config import get, path

PROTOCOL_VERSION = "1"
LABELS = ("belongs", "wrong", "cannot_say")
BELONGS, WRONG, UNDECIDED = LABELS
EVIDENCE_REQUIRED_FOR = ("wrong",)

# The two runs an annotator works through. The practice run is fifteen real records
# drawn from outside the natural set, worked before the labelling that counts;
# its labels live in their own file and never reach the analysis.
MAIN = "main"
PRACTICE = "practice"
RUNS = (MAIN, PRACTICE)

_ANNOTATOR_ID = re.compile(r"^[a-z0-9][a-z0-9-]{1,30}$")


@dataclass(frozen=True)
class Item:
    """One item as the annotator sees it.

    Attributes:
        item_id: The identifier printed on the sheet, a position in the shuffled
            order and nothing else.
        position: Where the item falls in the order, counting from one.
        shown: The fields the record carries, exactly what the labelling sheet
            shows and nothing beyond it.
    """

    item_id: str
    position: int
    shown: dict[str, Any]


def labels_dir() -> Path:
    """Return the directory holding one file per annotator."""
    target = path("data_processed") / "natural_set" / "labels"
    target.mkdir(parents=True, exist_ok=True)
    return target


def annotator_file(annotator: str, run: str = MAIN) -> Path:
    """Return one annotator's label file for one run.

    Args:
        annotator: The annotator identifier.
        run: ``main`` or ``practice``.

    Returns:
        The file the session appends to. The two runs are kept in separate
        files, so a practice answer cannot reach the analysis by accident.

    Raises:
        ValueError: If the run is neither of the two.
    """
    if run not in RUNS:
        raise ValueError(f"unknown run: {run}")
    suffix = "" if run == MAIN else ".practice"
    return labels_dir() / f"{annotator}{suffix}.jsonl"


def valid_annotator(annotator: str) -> bool:
    """Test whether an annotator identifier is one this study will deposit.

    The identifier travels into the released labels, so it is restricted to a
    short lower-case slug. An annotator's own name is recorded separately, on the
    machine that collected the labels, and never here.

    Args:
        annotator: The identifier to check.

    Returns:
        True when the identifier is acceptable.
    """
    return bool(_ANNOTATOR_ID.match(annotator))


def record_name(annotator: str, name: str) -> Path:
    """Write the annotator's own name to the local, untracked record.

    The written agreement and the acknowledgement both need a name; the
    repository and the deposit must not carry one. The mapping therefore lives
    in a file that git ignores, on the machine the labelling was done on.

    Args:
        annotator: The annotator identifier.
        name: The annotator's name as they gave it.

    Returns:
        The file written.
    """
    target = labels_dir() / "annotators.local.json"
    known: dict[str, Any] = {}
    if target.exists():
        known = json.loads(target.read_text(encoding="utf-8"))
    known[annotator] = {
        "name": name,
        "first_seen": known.get(annotator, {}).get(
            "first_seen", datetime.now(UTC).isoformat()
        ),
    }
    target.write_text(
        json.dumps(known, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return target


def read_items(run: str = MAIN) -> list[Item]:
    """Read one run's items in their labelling order.

    Args:
        run: ``main`` or ``practice``.

    Returns:
        The items to be labelled, in the single shuffled order the set was
        written in, showing only what the sheet shows. The main run stops at the
        configured number of labelled records; the practice run is served whole.

    Raises:
        FileNotFoundError: If the run's items have not been drawn.
        ValueError: If the run is neither of the two.
    """
    if run not in RUNS:
        raise ValueError(f"unknown run: {run}")
    name = "records.jsonl" if run == MAIN else "practice.jsonl"
    builder = "draw_natural_set" if run == MAIN else "draw_practice_set"
    source = path("data_processed") / "natural_set" / name
    if not source.exists():
        raise FileNotFoundError(
            f"{run} items not drawn: {source}; run scripts/{builder}.py"
        )
    items: list[Item] = []
    with source.open(encoding="utf-8") as handle:
        for position, line in enumerate(
            (line for line in handle if line.strip()), start=1
        ):
            record = json.loads(line)
            items.append(
                Item(
                    item_id=str(record["item_id"]),
                    position=position,
                    shown=dict(record["shown"]),
                )
            )
    ordered = sorted(items, key=lambda item: item.item_id)
    if run != MAIN:
        return ordered
    # Labelling stops at the first records of the fixed order, and this is what
    # enforces it: an annotator who reaches the stop is shown the closing page
    # rather than the next item. The draw is untouched, so the records past the
    # stop are still drawn, deposited and weighted against; they are simply
    # never served.
    limit = int(get("natural_set.labelled.records"))
    return [item for item in ordered if item.position <= limit]


def read_labels(annotator: str, run: str = MAIN) -> list[dict[str, Any]]:
    """Read one annotator's labels so far in one run.

    Args:
        annotator: The annotator identifier.
        run: ``main`` or ``practice``.

    Returns:
        The labels in the order they were given, empty when the session has not
        started.
    """
    target = annotator_file(annotator, run)
    if not target.exists():
        return []
    with target.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return [row for row in rows if row.get("kind") == "label"]


def consented(annotator: str) -> bool:
    """Test whether this annotator has accepted the protocol.

    Args:
        annotator: The annotator identifier.

    Returns:
        True when a consent record is already on file.
    """
    target = annotator_file(annotator)
    if not target.exists():
        return False
    with target.open(encoding="utf-8") as handle:
        return any(
            json.loads(line).get("kind") == "consent" for line in handle if line.strip()
        )


def append(annotator: str, record: dict[str, Any], run: str = MAIN) -> None:
    """Append one record to an annotator's file and flush it to disk.

    A label that reached a buffer and not a disk is a label that an interrupted
    session loses, so the handle is flushed on every write.

    Args:
        annotator: The annotator identifier.
        record: The record to write.
        run: ``main`` or ``practice``.
    """
    with annotator_file(annotator, run).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def record_consent(annotator: str, name: str) -> None:
    """Write the consent record that opens a session.

    Args:
        annotator: The annotator identifier.
        name: The annotator's name, written to the untracked local record and not
            into the label file.
    """
    record_name(annotator, name)
    append(
        annotator,
        {
            "kind": "consent",
            "annotator": annotator,
            "protocol_version": PROTOCOL_VERSION,
            "accepted_at": datetime.now(UTC).isoformat(),
        },
    )


def next_item(annotator: str, items: list[Item], run: str = MAIN) -> Item | None:
    """Return the item this annotator labels next in one run.

    Args:
        annotator: The annotator identifier.
        items: Every item in labelling order.
        run: ``main`` or ``practice``.

    Returns:
        The next unlabelled item, or None when all of them are done. The order
        is fixed and an item already past is never returned again.
    """
    done = len(read_labels(annotator, run))
    return items[done] if done < len(items) else None


def record_label(
    annotator: str,
    item: Item,
    label: str,
    evidence_source: str,
    evidence_url: str,
    note: str,
    seconds: float,
    run: str = MAIN,
    no_source: bool = False,
) -> dict[str, Any]:
    """Write one label.

    Args:
        annotator: The annotator identifier.
        item: The item being labelled.
        label: One of the three labels the protocol allows.
        evidence_source: What the annotator consulted, in their own words.
        evidence_url: The address consulted, where there was one.
        note: Anything the annotator wants recorded about the item.
        seconds: How long the item was on screen, reported by the page.
        run: ``main`` or ``practice``.
        no_source: The annotator's statement that they consulted nothing for this
            item, which is a record in its own right and not an empty field.

    Returns:
        The record written.

    Raises:
        ValueError: If the label is not one the protocol allows, if a label that
            requires evidence carries none, or if an item leaves neither
            evidence nor a statement that nothing was consulted.
    """
    if label not in LABELS:
        raise ValueError(f"label not in the protocol: {label}")
    consulted = bool(evidence_source or evidence_url)
    if label in EVIDENCE_REQUIRED_FOR and not consulted:
        raise ValueError(f"the protocol requires evidence for a {label} label")
    if not consulted and not no_source:
        raise ValueError(
            "record what you consulted, or state that you consulted nothing"
        )
    record = {
        "kind": "label",
        "run": run,
        "annotator": annotator,
        "item_id": item.item_id,
        "position": item.position,
        "label": label,
        "evidence_source": evidence_source,
        "evidence_url": evidence_url,
        "no_source": bool(no_source) and not consulted,
        "note": note,
        "seconds_on_item": round(float(seconds), 1),
        "labelled_at": datetime.now(UTC).isoformat(),
        "protocol_version": PROTOCOL_VERSION,
    }
    append(annotator, record, run)
    return record


def progress(annotator: str, items: list[Item], run: str = MAIN) -> dict[str, Any]:
    """Return how far one annotator has come in one run.

    Args:
        annotator: The annotator identifier.
        items: Every item in labelling order.
        run: ``main`` or ``practice``.

    Returns:
        The counts a session shows the annotator, and nothing about any other
        annotator.
    """
    done = read_labels(annotator, run)
    return {
        "run": run,
        "labelled": len(done),
        "total": len(items),
        "remaining": len(items) - len(done),
        "consented": consented(annotator),
    }


def shown_fields() -> list[str]:
    """Return the field names an item may show, in a fixed display order.

    Returns:
        The configured item keys followed by the coordinate.
    """
    keys = [str(key) for key in get("injected_set.construction.pairing.item_keys")]
    return [*keys, "lat", "lon"]
