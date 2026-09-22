"""Choose each model's prompt version on the control development split.

Nine versions are candidates: version 1 and eight variants differing from it in
the instruction alone. Each is scored by Youden's J on the development split,
which the article never reads, and the highest wins, ties to the lowest version
so that version 1 is kept unless a variant does better. The choice is written
before the chosen version is asked anything else.

Outputs:
    data/processed/analysis/prompt_selection.json

Usage:
    python -m scripts.score_prompt_selection
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from poi_audit import scoring
from poi_audit.config import get, path
from poi_audit.inference import pool, response_path
from poi_audit.logsetup import setup
from scripts.score_control import control_truth

logger = setup("score_prompt_selection")


def choose(scores: dict[int, float]) -> int:
    """Return the version with the highest J, ties to the lowest version."""
    return sorted(scores, key=lambda version: (-scores[version], version))[0]


def run_for(version: int) -> str:
    """Return the run a version's development answers are read from."""
    return "m1c_name_full_v1" if version == 1 else f"m1c_name_select_v{version}"


def main() -> None:
    """Score every candidate for every chosen model and write the choice."""
    settings = get("prompt_selection")
    chosen = json.loads(
        (path("data_processed") / "analysis" / "adaptation_models.json").read_text(
            "utf-8"
        )
    )["chosen"]
    truth = control_truth()
    dev = {item for item, row in truth.items() if row["split"] == settings["select_on"]}
    cards = {card.tag: card for card in pool()}
    result: dict[str, dict] = {}
    for tag in sorted(set(chosen.values())):
        scores: dict[int, float] = {}
        for version in settings["versions"]:
            target = response_path(run_for(int(version)), cards[tag])
            answers = [a for a in scoring.read_answers(target) if a["item_id"] in dev]
            if len(answers) != len(dev):
                raise ValueError(
                    f"{tag} v{version}: {len(answers)} of {len(dev)} answers"
                )
            judged = scoring.score_name_injected(answers, truth)
            scores[int(version)] = round(scoring.confusion(judged).youdens_j, 4)
        result[tag] = {"dev_youdens_j": scores, "selected_version": choose(scores)}
        logger.info(f"{tag}: {result[tag]}")
    target = path("data_processed") / "analysis" / "prompt_selection.json"
    target.write_text(
        json.dumps(
            {
                "chosen_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "rule": settings,
                "models": result,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
