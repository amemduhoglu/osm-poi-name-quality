"""Compare the replay of natural_name_v1 with the stored run, model by model.

Two things changed between the reported runs and the redesign's: the serving
stack moved from Ollama 0.32.6 to 0.32.14, and four models were removed from
the server and pulled again. The old binary and the old weights' digests do not
survive, so whether a model is still the model that was scored is shown by its
behaviour: every model answers the same 250 records again at temperature zero,
and its verdicts are compared one by one with the ones on disk. The threshold a
model must reach was fixed in the configuration before any replay was compared.

The comparison is also the stability check the decision of 2026-09-11 asked
for: how much of a model's answer is a property of the model and how much of
the moment it was asked.

Outputs:
    data/processed/analysis/replay_stability_<replay run>.csv
    data/processed/analysis/replay_stability_<replay run>.json

Usage:
    python -m scripts.score_replay --replay natural_name_v1_replay_0326
    python -m scripts.score_replay --replay natural_name_v1_replay
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from typing import Any

from poi_audit import scoring
from poi_audit.config import get, path
from poi_audit.inference import pool, response_path
from poi_audit.logsetup import setup

logger = setup("score_replay")

STORED = "natural_name_v1"


def verdict(answer: dict[str, Any]) -> str | None:
    """Return an answer's verdict, or None when the answer was not valid."""
    if not answer.get("valid"):
        return None
    return (answer.get("parsed") or {}).get("verdict")


def compare(
    stored: list[dict[str, Any]], replay: list[dict[str, Any]], minimum: float
) -> dict[str, Any]:
    """Compare one model's stored answers with its replayed answers.

    Args:
        stored: The answers on disk from the reported run.
        replay: The answers to the same items under the current stack.
        minimum: The verdict agreement a model needs to be kept.

    Returns:
        The items compared, the verdicts that changed, the shares of matching
        verdicts and of byte-identical responses, and whether the model is kept.

    Raises:
        ValueError: If the replay does not cover every stored item.
    """
    old = {str(one["item_id"]): one for one in stored}
    new = {str(one["item_id"]): one for one in replay}
    missing = set(old) - set(new)
    if missing:
        raise ValueError(f"replay lacks {len(missing)} stored item(s)")
    items = sorted(old)
    same = sum(1 for item in items if verdict(old[item]) == verdict(new[item]))
    raw = sum(
        1
        for item in items
        if old[item].get("response_raw") == new[item].get("response_raw")
    )
    # The prompt's token count is the server's own count of what it was sent
    # after the chat template, so a template or tokenizer change shows here even
    # where the verdicts happen to agree.
    tokens = sum(
        1
        for item in items
        if old[item].get("prompt_tokens") == new[item].get("prompt_tokens")
    )
    agreement = same / len(items)
    return {
        "items": len(items),
        "changed": len(items) - same,
        "verdict_agreement": round(agreement, 4),
        "raw_identical": round(raw / len(items), 4),
        "prompt_tokens_identical": round(tokens / len(items), 4),
        "kept": agreement >= minimum,
    }


def youden_on_natural(answers: list[dict[str, Any]], truth: dict) -> float:
    """Return Youden's J of one run's answers on the adjudicated natural set."""
    return scoring.confusion(scoring.score_name_natural(answers, truth)).youdens_j


def main() -> None:
    """Compare every model whose replay is complete and write the result."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replay", required=True, help="the replay run to compare")
    replay_run = parser.parse_args().replay
    minimum = float(get("run.replay_min_verdict_agreement"))
    truth = scoring.natural_truth()
    rows: list[dict[str, Any]] = []
    for card in pool():
        stored_path = response_path(STORED, card)
        replay_path = response_path(replay_run, card)
        if not replay_path.exists():
            logger.warning(f"{card.tag}: no replay yet")
            continue
        stored = scoring.read_answers(stored_path)
        replay = scoring.read_answers(replay_path)
        if len({one["item_id"] for one in replay}) < len(stored):
            logger.warning(f"{card.tag}: replay incomplete ({len(replay)})")
            continue
        result = compare(stored, replay, minimum)
        rows.append(
            {
                "model": card.tag,
                "name": card.name,
                **result,
                "youdens_j_stored": round(youden_on_natural(stored, truth), 4),
                "youdens_j_replay": round(youden_on_natural(replay, truth), 4),
            }
        )
        logger.info(f"{card.tag}: {result}")
    target = path("data_processed") / "analysis" / f"replay_stability_{replay_run}.csv"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    target.with_suffix(".json").write_text(
        json.dumps(
            {
                "compared_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "stored_run": STORED,
                "replay_run": replay_run,
                "replay_server_versions": sorted(
                    {
                        str(one.get("server_version"))
                        for card in pool()
                        if response_path(replay_run, card).exists()
                        for one in scoring.read_answers(response_path(replay_run, card))
                    }
                ),
                "stored_stack": get("device.ollama_version"),
                "minimum_verdict_agreement": minimum,
                "models": rows,
                "kept": [row["model"] for row in rows if row["kept"]],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info(f"wrote {target}")


if __name__ == "__main__":
    main()
