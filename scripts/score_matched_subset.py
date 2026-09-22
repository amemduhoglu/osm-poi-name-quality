"""Score the residual class on the pairs its construction did not mark.

Two of the five donor filters leave a mark on the clean half rather than on the
corrupted one, because the pairing matches on the city and the field signature
and constrains nothing about the name. No injected name carries a
category-indicative token and no injected name is shorter than the donor
minimum, so a clean twin that carries a token, or that is shorter, can be
answered from the construction alone.

Dropping those pairs costs items and buys a score the marks cannot explain. It
is a matched subsample of a set already answered, so it needs no new inference:
the same answers are read on fewer records.

One asymmetry survives and is meant to. The donor keeps its own record, so an
injected name is repeated in its city by construction, and no choice of clean
twin repairs that. It is a property of injection, it is what the corpus rule
reads, and it is reported rather than removed.

Outputs:
    data/processed/analysis/matched_subset.csv
    data/processed/analysis/matched_subset.json

Usage:
    python scripts/score_matched_subset.py
"""

from __future__ import annotations

import csv
import json
from typing import Any

from poi_audit import scoring, screen
from poi_audit.config import get, path
from poi_audit.inference import pool, response_path
from poi_audit.logsetup import setup

logger = setup("score_matched_subset")

RUN = "m1_name_full_v1"


def eligible_pairs(
    items: list[dict[str, Any]], tokens: dict[str, str], min_chars: int
) -> set[str]:
    """Return the pairs whose clean twin could itself have been a donor.

    Args:
        items: The residual class, both halves.
        tokens: Category-indicative token to the category it indicates.
        min_chars: The donor's minimum name length.

    Returns:
        The pair identifiers to keep.
    """
    keep: set[str] = set()
    for item in items:
        if item["corrupted"]:
            continue
        name = str(item["record"].get("name", ""))
        if len(name) < min_chars:
            continue
        if screen.indicated_category(name, tokens) is not None:
            continue
        keep.add(str(item["pair_id"]))
    return keep


def read_items() -> list[dict[str, Any]]:
    """Read the residual class.

    Returns:
        Every residual-class item.
    """
    target = path("data_processed") / "injected_set" / "items.jsonl"
    with target.open(encoding="utf-8") as handle:
        return [
            one
            for one in (json.loads(line) for line in handle if line.strip())
            if one["class"] == "M1"
        ]


def main() -> None:
    """Command line entry point."""
    items = read_items()
    tokens = screen.load_category_tokens()
    minimum = int(get("injected_set.construction.m1.donor_min_name_chars"))
    keep = eligible_pairs(items, tokens, minimum)
    kept = {str(one["item_id"]) for one in items if str(one["pair_id"]) in keep}
    logger.info("{} pairs of {} kept, {} items", len(keep), len(items) // 2, len(kept))

    truth = scoring.injected_truth()
    rows: list[dict[str, Any]] = []
    for card in pool():
        target = response_path(RUN, card)
        if not target.exists():
            continue
        answers = scoring.read_answers(target)
        if not answers:
            continue
        judged = scoring.score_name_injected(answers, truth)
        whole = scoring.confusion(judged)
        subset = scoring.confusion([one for one in judged if one.item_id in kept])
        rows.append(
            {
                "model": card.tag,
                "parameters_b": card.parameters_b,
                "items_whole": whole.total,
                "youdens_j_whole": round(whole.youdens_j, 4),
                "balanced_accuracy_whole": round(whole.balanced_accuracy, 4),
                "items_matched": subset.total,
                "youdens_j_matched": round(subset.youdens_j, 4),
                "balanced_accuracy_matched": round(subset.balanced_accuracy, 4),
                "youdens_j_change": round(subset.youdens_j - whole.youdens_j, 4),
            }
        )

    out = path("data_processed") / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "matched_subset.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "matched_subset.json").write_text(
        json.dumps(
            {
                "run": RUN,
                "pairs_kept": len(keep),
                "pairs_whole": len(items) // 2,
                "dropped_because": (
                    "the clean twin carries a category-indicative token, which "
                    "no injected name does, or is shorter than the donor's "
                    "minimum name length, which no injected name is"
                ),
                "not_repaired": (
                    "the donor keeps its own record, so an injected name is "
                    "repeated in its city by construction"
                ),
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("wrote {}", out / "matched_subset.csv")
    for row in sorted(rows, key=lambda one: -float(one["youdens_j_whole"])):
        logger.info(
            "{}: J {} whole, {} matched ({})",
            row["model"],
            row["youdens_j_whole"],
            row["youdens_j_matched"],
            row["youdens_j_change"],
        )


if __name__ == "__main__":
    main()
