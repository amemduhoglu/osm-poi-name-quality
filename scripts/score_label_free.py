"""Bound what the models can be doing on the unlabelled natural-set records.

The 623 natural-set records past the first 250 carry no label, and none will be
collected. Two things are still measurable on them. The first is a ceiling on
precision: the first 250 bound, cell by cell, how many wrong names a record set
of known composition can hold, so a model that flags more records than that
cannot be right about most of them, whatever the labels would have said. The
second is whether the models share a signal there: agreement among them on these
records is set beside their agreement on the control test split, where the truth
is known and a shared signal exists.

Outputs:
    data/processed/analysis/label_free_bounds.csv
    data/processed/analysis/model_agreement.json

Usage:
    python -m scripts.score_label_free
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from typing import Any

from poi_audit import scoring, stats
from poi_audit.config import get, path
from poi_audit.inference import pool, response_path
from poi_audit.logsetup import setup
from scripts.score_control import control_truth

logger = setup("score_label_free")


def precision_bound(
    cells: dict[str, int], upper: dict[str, float], flagged: int
) -> float:
    """Return the highest precision a model flagging this many records can reach.

    Args:
        cells: Records per design cell.
        upper: Upper bound of the wrong-name rate per cell.
        flagged: Records the model flags.

    Returns:
        The most wrong names the records can hold over the flags, capped at one.

    Raises:
        ValueError: If the model flags nothing, where precision is undefined.
    """
    if flagged <= 0:
        raise ValueError("no flag, no precision")
    most = sum(count * upper[cell] for cell, count in cells.items())
    return min(1.0, most / flagged)


def cell_bounds() -> dict[str, float]:
    """Return the upper Wilson bound of the wrong-name rate per cell, first 250."""
    records = {}
    for name in ("records.jsonl",):
        with (path("data_processed") / "natural_set" / name).open(
            encoding="utf-8"
        ) as handle:
            for row in map(json.loads, handle):
                records[row["item_id"]] = row
    counts: dict[str, Counter] = {}
    for item, row in scoring.natural_truth().items():
        if not row["scored"]:
            continue
        source = records[item]
        cell = f"{source['language']}, {source['maturity']}"
        counts.setdefault(cell, Counter())[row["label"]] += 1
    return {
        cell: stats.wilson(tally["wrong"], tally["wrong"] + tally["belongs"]).high
        for cell, tally in counts.items()
    }


def extension_cells() -> dict[str, dict[str, str]]:
    """Return the design cell of every extension record."""
    cells: dict[str, dict[str, str]] = {}
    limit = int(get("natural_set.labelled.records"))
    for name in ("records.jsonl", "records_phase2.jsonl"):
        with (path("data_processed") / "natural_set" / name).open(
            encoding="utf-8"
        ) as handle:
            for row in map(json.loads, handle):
                if name == "records.jsonl" and int(row["item_id"][1:]) <= limit:
                    continue
                cells[row["item_id"]] = f"{row['language']}, {row['maturity']}"
    return cells


def verdicts(run: str, tag: str) -> dict[str, str] | None:
    """Return one model's verdicts on one run, keyed by item, or None if absent."""
    card = {c.tag: c for c in pool()}[tag]
    target = response_path(run, card)
    if not target.exists():
        return None
    return {
        str(a["item_id"]): (
            (a.get("parsed") or {}).get("verdict") if a.get("valid") else "invalid"
        )
        for a in scoring.read_answers(target)
    }


def main() -> None:
    """Write the precision bounds and the agreement among the kept models."""
    settings = get("analysis.label_free")
    kept_file = (
        path("data_processed")
        / "analysis"
        / "replay_stability_natural_name_v1_replay_0326.json"
    )
    kept = json.loads(kept_file.read_text("utf-8"))["kept"]
    shown = list(get("run.replay_excluded_shown"))
    upper = cell_bounds()
    cells_of = extension_cells()
    composition = Counter(cells_of.values())

    rows: list[dict[str, Any]] = []
    extension: dict[str, dict[str, str]] = {}
    for tag in kept + shown:
        answers = verdicts(str(settings["run"]), tag)
        if answers is None or len(answers) < len(cells_of):
            logger.warning(f"{tag}: extension incomplete")
            continue
        if tag in kept:
            extension[tag] = answers
        flagged = sum(1 for v in answers.values() if v == "wrong")
        rows.append(
            {
                "model": tag,
                "pooled": tag in kept,
                "records": len(answers),
                "flagged": flagged,
                "flag_rate": round(flagged / len(answers), 4),
                "most_wrong_names": round(
                    sum(n * upper[c] for c, n in composition.items()), 1
                ),
                "precision_at_most": (
                    round(precision_bound(dict(composition), upper, flagged), 4)
                    if flagged
                    else ""
                ),
            }
        )

    truth = control_truth()
    test_items = sorted(
        i for i, row in truth.items() if row["split"] == settings["control_split"]
    )
    control = {tag: verdicts(str(settings["control_run"]), tag) for tag in extension}
    categories = ("wrong", "belongs", "invalid")
    ext_items = sorted(cells_of)
    agreement = {
        "models": sorted(extension),
        "extension": round(
            stats.fleiss_kappa(
                [[extension[t][i] for t in sorted(extension)] for i in ext_items],
                categories,
            ),
            4,
        ),
        "control_test": round(
            stats.fleiss_kappa(
                [[control[t][i] for t in sorted(extension)] for i in test_items],
                categories,
            ),
            4,
        ),
        "cell_upper_bounds": {cell: round(value, 4) for cell, value in upper.items()},
        "extension_composition": dict(composition),
    }
    out = path("data_processed") / "analysis"
    with (out / "label_free_bounds.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "model_agreement.json").write_text(
        json.dumps(agreement, indent=2) + "\n", "utf-8"
    )
    logger.info(
        f"agreement: {agreement['extension']} extension, "
        f"{agreement['control_test']} control test"
    )


if __name__ == "__main__":
    main()
