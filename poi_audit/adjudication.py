"""The state the single adjudication meeting keeps, apart from how it is served.

Section 6 of the protocol fixes the rule before the disagreements exist: the two
readers meet once, review only the items they disagree on, take the label the
recorded evidence supports, and leave the item at ``cannot_say`` where the
evidence settles nothing. No item is decided by seniority, by majority or by a
third reading.

This module enforces what a rule written on paper cannot. It offers a decision
only the labels the two readers actually reached, refuses a reason outside the
fixed list, and writes every action to disk as it happens. It decides nothing
itself, here as everywhere: no model and no program produces a label on a task
this study reports as resisting programs.

Everything is appended. A decision changed later is a new line rather than an
overwrite, so the meeting's course stays readable and the last line for an item
is the one that counts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from poi_audit import labelling

PROTOCOL_VERSION = labelling.PROTOCOL_VERSION
CANNOT_SAY = "cannot_say"
FILE = "adjudication.jsonl"

# Why the rule settled the item the way it did, as a fixed list rather than as
# free text, so that how the rule was applied across the meeting can be reported
# rather than described. Every entry names a property of the recorded evidence
# and none names a reader, because section 6.4 forbids deciding by standing.
REASONS = (
    "source_positive_and_specific",
    "sources_conflict",
    "no_source_positive",
    "one_reader_consulted_no_source",
)

OPENED = "opened"
DECISION = "decision"
CLOSED = "closed"


@dataclass(frozen=True)
class Disagreement:
    """One item the two readers labelled differently.

    Attributes:
        item_id: The identifier the labelling sheet printed.
        position: Where the item fell in the labelling order.
        shown: The fields the record carries, exactly what the readers saw.
        readings: Each reader's own label, evidence and note, keyed by reader.
    """

    item_id: str
    position: int
    shown: dict[str, Any]
    readings: dict[str, dict[str, Any]]

    def allowed_labels(self) -> tuple[str, ...]:
        """Return the labels this item may be adjudicated to.

        Returns:
            The labels the two readers reached, and ``cannot_say``, which
            section 6.3 keeps open for an item the evidence does not settle. A
            label neither reader reached is absent, because reaching one here
            would be the third reading section 6.4 forbids.
        """
        given = {str(reading["label"]) for reading in self.readings.values()}
        given.add(CANNOT_SAY)
        return tuple(label for label in labelling.LABELS if label in given)


def record_file() -> Path:
    """Return the file every action of the meeting is appended to."""
    return labelling.labels_dir() / FILE


def readers() -> tuple[str, ...]:
    """Return the readers who completed the labelled set, in a fixed order.

    Returns:
        The annotator identifiers, sorted, so that the two columns never change
        places between one item and the next.

    Raises:
        ValueError: If the number of complete readings is not two, since the
            rule this instrument serves is written for exactly two.
    """
    items = labelling.read_items(labelling.MAIN)
    complete = []
    for target in sorted(labelling.labels_dir().glob("*.jsonl")):
        name = target.stem
        if name.endswith(".practice") or name == Path(FILE).stem:
            continue
        if len(labelling.read_labels(name, labelling.MAIN)) == len(items):
            complete.append(name)
    if len(complete) != 2:
        raise ValueError(
            f"the meeting needs exactly two complete readings, and {len(complete)} "
            "were found"
        )
    return tuple(complete)


def disagreements() -> list[Disagreement]:
    """Return the items the two readers labelled differently, in order.

    Returns:
        One entry per disagreement, carrying what each reader recorded. Items
        the readers agreed on are absent: two independent readings already
        settled them, and reopening one would be a third reading.
    """
    first, second = readers()
    given = {
        reader: {
            str(row["item_id"]): row
            for row in labelling.read_labels(reader, labelling.MAIN)
        }
        for reader in (first, second)
    }
    found: list[Disagreement] = []
    for item in labelling.read_items(labelling.MAIN):
        rows = [given[reader].get(item.item_id) for reader in (first, second)]
        if any(row is None for row in rows):
            continue
        if rows[0]["label"] == rows[1]["label"]:
            continue
        found.append(
            Disagreement(
                item_id=item.item_id,
                position=item.position,
                shown=dict(item.shown),
                readings={
                    reader: {
                        "label": row["label"],
                        "evidence_source": row.get("evidence_source", ""),
                        "evidence_url": row.get("evidence_url", ""),
                        "no_source": bool(row.get("no_source")),
                        "note": row.get("note", ""),
                    }
                    for reader, row in zip((first, second), rows, strict=True)
                },
            )
        )
    return found


def read_actions() -> list[dict[str, Any]]:
    """Return every action the meeting has taken, in the order taken.

    Returns:
        The opening, the decisions and the closing, or an empty list before the
        meeting starts.
    """
    target = record_file()
    if not target.exists():
        return []
    with target.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append(record: dict[str, Any]) -> None:
    """Append one action and flush it to disk.

    Args:
        record: The action to write.
    """
    with record_file().open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def decisions() -> dict[str, dict[str, Any]]:
    """Return the decision standing for each item.

    Returns:
        Item identifier to the last decision recorded for it, which is the one
        that counts. Superseded decisions stay in the file.
    """
    standing: dict[str, dict[str, Any]] = {}
    for action in read_actions():
        if action.get("kind") == DECISION:
            standing[str(action["item_id"])] = action
    return standing


def opened() -> bool:
    """Return whether the meeting has been opened."""
    return any(action.get("kind") == OPENED for action in read_actions())


def closed() -> bool:
    """Return whether the meeting has been closed."""
    return any(action.get("kind") == CLOSED for action in read_actions())


def record_open() -> dict[str, Any]:
    """Open the meeting.

    Returns:
        The record written.

    Raises:
        ValueError: If the meeting has already been closed.
    """
    if closed():
        raise ValueError("the meeting is closed")
    record = {
        "kind": OPENED,
        "readers": list(readers()),
        "items": len(disagreements()),
        "both_readers_present": True,
        "opened_at": datetime.now(UTC).isoformat(),
        "protocol_version": PROTOCOL_VERSION,
    }
    append(record)
    return record


def record_decision(
    item: Disagreement,
    label: str,
    reason: str,
    note: str,
) -> dict[str, Any]:
    """Record one adjudicated label.

    Args:
        item: The disagreement being settled.
        label: One of the labels the two readers reached, or ``cannot_say``.
        reason: Which property of the recorded evidence settled it.
        note: Anything the readers want recorded, in the language they work in.

    Returns:
        The record written.

    Raises:
        ValueError: If the meeting is closed or not yet open, if the label is
            not one this item may take, or if the reason is outside the list.
    """
    if closed():
        raise ValueError("the meeting is closed")
    if not opened():
        raise ValueError("the meeting has not been opened")
    if label not in item.allowed_labels():
        raise ValueError(
            f"neither reader reached {label} on this item, and reaching it here "
            "would be a third reading"
        )
    if reason not in REASONS:
        raise ValueError(f"reason not in the protocol's list: {reason}")
    record = {
        "kind": DECISION,
        "item_id": item.item_id,
        "position": item.position,
        "label": label,
        "reason": reason,
        "note": note,
        "readings": {
            reader: reading["label"] for reader, reading in item.readings.items()
        },
        "decided_at": datetime.now(UTC).isoformat(),
        "protocol_version": PROTOCOL_VERSION,
    }
    append(record)
    return record


def record_close() -> dict[str, Any]:
    """Close the meeting, which freezes it.

    Returns:
        The record written.

    Raises:
        ValueError: If the meeting is already closed, or if any item is still
            undecided, since a closed meeting is the record that the rule was
            applied to every disagreement.
    """
    if closed():
        raise ValueError("the meeting is already closed")
    outstanding = [
        item.item_id for item in disagreements() if item.item_id not in decisions()
    ]
    if outstanding:
        raise ValueError(f"{len(outstanding)} items are still undecided")
    record = {
        "kind": CLOSED,
        "decided": len(decisions()),
        "closed_at": datetime.now(UTC).isoformat(),
        "protocol_version": PROTOCOL_VERSION,
    }
    append(record)
    return record


def progress() -> dict[str, Any]:
    """Return what the page needs to draw itself.

    Returns:
        The readers, every disagreement with the decision standing against it,
        and whether the meeting is open or closed.
    """
    items = disagreements()
    standing = decisions()
    return {
        "readers": list(readers()),
        "reasons": list(REASONS),
        "fields": labelling.shown_fields(),
        "opened": opened(),
        "closed": closed(),
        "total": len(items),
        "decided": sum(1 for item in items if item.item_id in standing),
        "items": [
            {
                "item_id": item.item_id,
                "position": item.position,
                "shown": item.shown,
                "readings": item.readings,
                "allowed_labels": list(item.allowed_labels()),
                "decision": standing.get(item.item_id),
            }
            for item in items
        ],
    }
