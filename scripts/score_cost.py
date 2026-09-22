"""Join what each model costs to run with what it is worth on the residual class.

Section 5.4 asks a practitioner's question rather than a benchmark one: for a
model that could be deployed beside the community's existing validators, what
does an hour of it buy. That needs three numbers per model held together, and
they live in three files until here: the measured footprint from the preflight,
the median rate from the scored runs, and the detection score from the analysis.

Every rate is a property of one card and of the serving stack named in the
configuration. The article states that once. A model above the deployment
threshold is named a reference wherever its score appears, and the threshold is
a convention about who can run the result rather than a property of the
instrument.

Outputs:
    data/processed/analysis/cost.csv

Usage:
    python scripts/score_cost.py
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from poi_audit.config import get, path
from poi_audit.inference import efficiency_dir
from poi_audit.logsetup import setup

logger = setup("score_cost")

# The run the cost is quoted against: the residual class asked in full, which is
# the only question the article says still needs a model.
QUOTED_RUN = "m1_name_full_v1"


def read_csv(target: Path) -> list[dict[str, str]]:
    """Return one CSV as a list of rows.

    Args:
        target: The file.

    Returns:
        The rows, empty when the file is absent.
    """
    if not target.exists():
        return []
    with target.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build() -> list[dict[str, Any]]:
    """Join the footprint, the rate and the score, one row per model.

    Returns:
        One row per model that has been both profiled and scored.

    Raises:
        FileNotFoundError: If the footprint profile has not been written.
    """
    profile = efficiency_dir() / "footprint.csv"
    if not profile.exists():
        raise FileNotFoundError(
            f"no footprint profile at {profile}; run scripts/preflight_models.py"
        )
    footprints = {row["tag"]: row for row in read_csv(profile)}
    summary = {
        row["model"]: row
        for row in read_csv(efficiency_dir() / "run_summary.csv")
        if row["run"] == QUOTED_RUN
    }
    scores = {
        row["model"]: row
        for row in read_csv(path("data_processed") / "analysis" / "model_scores.csv")
        if row["run"] == QUOTED_RUN and row["scored_set"] == "all"
    }
    threshold = float(get("device.deployment_threshold_gb"))

    rows: list[dict[str, Any]] = []
    for tag, footprint in footprints.items():
        rate = summary.get(tag)
        score = scores.get(tag)
        if not rate or not score:
            continue
        seconds = float(rate["median_total_sec"])
        measured = float(footprint["footprint_gb"])
        rows.append(
            {
                "model": tag,
                "family": footprint["family"],
                "parameters_b": footprint["configured_parameters_b"],
                "footprint_gb": measured,
                "deployable": measured <= threshold,
                "measured_class": footprint["measured_class"],
                "median_sec_per_record": round(seconds, 3),
                "records_per_hour": round(3600 / seconds) if seconds else None,
                "median_tokens_per_sec": float(rate["median_output_tokens_per_sec"]),
                "youdens_j": float(score["youdens_j"]),
                "flag_rate": float(score["flag_rate"]),
            }
        )
    return sorted(rows, key=lambda row: float(row["footprint_gb"]))


def main() -> None:
    """Write the cost table."""
    rows = build()
    if not rows:
        logger.warning("nothing to join yet: the runs or the scores are missing")
        return
    target = path("data_processed") / "analysis" / "cost.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"wrote {target} with {len(rows)} models")
    best = max(rows, key=lambda row: row["youdens_j"])
    deployable = [row for row in rows if row["deployable"]]
    if deployable:
        best_deployable = max(deployable, key=lambda row: row["youdens_j"])
        logger.info(
            f"best on the residual class: {best['model']} "
            f"at J {best['youdens_j']:.3f}"
        )
        logger.info(
            f"best inside the deployment threshold: {best_deployable['model']} "
            f"at J {best_deployable['youdens_j']:.3f}"
        )


if __name__ == "__main__":
    main()
