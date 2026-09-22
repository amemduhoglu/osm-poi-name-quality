"""Score the trivial answers on the residual control set.

The decision of 2026-09-11 found that a string or dictionary screen separates
the residual class as first built. This scores the same screen, rule by rule,
on the control set, reading only the visible part of each city, which is all
any instrument is given for this set; and it scores the dense-retrieval
comparator, calibrated on the training and development splits and read on the
test split. Both are what a model on the control set has to beat.

Outputs:
    data/processed/analysis/control_screen_baseline.csv
    data/processed/analysis/control_retrieval_baseline.csv

Usage:
    python -m scripts.score_control_baselines
"""

from __future__ import annotations

import asyncio
import csv
import json
from typing import Any

from poi_audit import screen
from poi_audit.config import get, path, secret
from poi_audit.corpus import read_corpus
from poi_audit.logsetup import setup
from poi_audit.references import load_gazetteer
from poi_audit.retrieval import Encoder
from scripts.score_retrieval_baseline import (
    _decided,
    as_retrieval_record,
    oracle_threshold,
    score_records,
    score_row,
)
from scripts.score_screen_baseline import score

logger = setup("score_control_baselines")


def read_control() -> tuple[list[dict[str, Any]], set[tuple[str, int]]]:
    """Return the control items and the identities of every visible record."""
    base = path("data_processed") / "injected_set"
    with (base / "residual_control.jsonl").open(encoding="utf-8") as handle:
        items = [json.loads(line) for line in handle if line.strip()]
    visible = json.loads((base / "residual_control_visible.json").read_text("utf-8"))
    identities = {
        (str(kind), int(number))
        for part in visible.values()
        for kind, number in part["records"]
    }
    return items, identities


def write(rows: list[dict[str, Any]], name: str) -> None:
    """Write rows to one analysis CSV."""
    target = path("data_processed") / "analysis" / name
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"wrote {target}")


def screen_rows(items: list[dict[str, Any]], visible: set[tuple[str, int]]) -> list:
    """Score every screen rule on the control set against the visible corpus."""
    corpus = [record for record in read_corpus() if screen.identity(record) in visible]
    places = screen.place_index(load_gazetteer())
    tokens = screen.load_category_tokens()
    rows = []
    for split in ("all", "test"):
        wanted = [i for i in items if split == "all" or i["split"] == split]
        result = score(wanted, corpus, tokens, places)
        for rule, measures in sorted(result["by_rule"].items()):
            rows.append({"split": split, "rule": rule, **measures})
    return rows


async def retrieval_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score the dense-retrieval comparator on the control set."""
    calibrate_on = set(get("residual_control.baseline_calibration_splits"))
    records = [as_retrieval_record(item) for item in items]
    split_of = {str(item["item_id"]): item["split"] for item in items}
    truth = {r["item_id"]: r["corrupted"] for r in records}
    async with Encoder(
        secret("models.serving.base_url_env"), str(get("retrieval_baseline.encoder"))
    ) as encoder:
        scores = await score_records(
            records,
            load_gazetteer(),
            encoder,
            float(get("retrieval_baseline.radius_km")),
        )
    calibration = {i: s for i, s in scores.items() if split_of[i] in calibrate_on}
    test = {i: s for i, s in scores.items() if split_of[i] == "test"}
    cal_scores, cal_truth = _decided(calibration, {i: truth[i] for i in calibration})
    calibrated, _ = oracle_threshold(cal_scores, cal_truth)
    test_scores, test_truth = _decided(test, {i: truth[i] for i in test})
    oracle, _ = oracle_threshold(test_scores, test_truth)
    return [
        score_row("control", "test", "calibrated", calibrated, test, truth),
        score_row("control", "test", "oracle", oracle, test, truth),
    ]


def main() -> None:
    """Score both baselines on the control set."""
    items, visible = read_control()
    write(screen_rows(items, visible), "control_screen_baseline.csv")
    write(asyncio.run(retrieval_rows(items)), "control_retrieval_baseline.csv")


if __name__ == "__main__":
    main()
