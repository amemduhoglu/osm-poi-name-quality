"""Compare each adaptation arm with its own base, paired on the record.

Two arms answer the third concern of the decision of 2026-09-11. The
retrieval-augmented arm asks the chosen models version 4 of the prompt, and the
adapted model is Qwen 3.5 4B with a low-rank adapter trained on the control
training split. Each is compared with the same model's unadapted answers under
the same serving release: on the control set's test split, which no arm was
tuned on, and on the natural set's adjudicated records. The comparison is paired
on the record by an exact McNemar test, reported with the discordant odds ratio,
with Youden's J and the flag rate of both sides beside it.

Outputs:
    data/processed/analysis/arm_comparisons.csv

Usage:
    python -m scripts.score_arms
"""

from __future__ import annotations

import csv
import json
from typing import Any

from poi_audit import scoring
from poi_audit.config import get, path
from poi_audit.inference import adapted_cards, pool, response_path
from poi_audit.logsetup import setup
from poi_audit.stats import exact_mcnemar
from scripts.score_control import control_truth

logger = setup("score_arms")


def comparisons() -> list[dict[str, str]]:
    """Return every arm-against-base comparison the redesign makes."""
    rows: list[dict[str, str]] = []
    chosen_file = path("data_processed") / "analysis" / "adaptation_models.json"
    if chosen_file.exists():
        chosen = json.loads(chosen_file.read_text("utf-8"))["chosen"]
        for tag in sorted(set(chosen.values())):
            rows.append(
                {
                    "arm": "retrieval_augmented",
                    "arm_model": tag,
                    "base_model": tag,
                    "set": "control_test",
                    "arm_run": "m1c_name_rag_v4",
                    "base_run": "m1c_name_full_v1",
                }
            )
            rows.append(
                {
                    "arm": "retrieval_augmented",
                    "arm_model": tag,
                    "base_model": tag,
                    "set": "natural",
                    "arm_run": "natural_name_rag_v4",
                    "base_run": "natural_name_v1_replay_0326",
                }
            )
        for tag in sorted(set(chosen.values())):
            rows.append(
                {
                    "arm": "few_shot",
                    "arm_model": tag,
                    "base_model": tag,
                    "set": "control_test",
                    "arm_run": "m1c_name_fewshot_v3",
                    "base_run": "m1c_name_full_v1",
                }
            )
            rows.append(
                {
                    "arm": "few_shot",
                    "arm_model": tag,
                    "base_model": tag,
                    "set": "natural",
                    "arm_run": "natural_name_fewshot_v3",
                    "base_run": "natural_name_v1_replay_0326",
                }
            )
    selection_file = path("data_processed") / "analysis" / "prompt_selection.json"
    if selection_file.exists():
        selected = json.loads(selection_file.read_text("utf-8"))["models"]
        for tag, choice in sorted(selected.items()):
            version = int(choice["selected_version"])
            if version == 1:
                continue
            rows.append(
                {
                    "arm": "prompt_selection",
                    "arm_model": tag,
                    "base_model": tag,
                    "set": "control_test",
                    "arm_run": f"m1c_name_selected_v{version}",
                    "base_run": "m1c_name_full_v1",
                }
            )
            rows.append(
                {
                    "arm": "prompt_selection",
                    "arm_model": tag,
                    "base_model": tag,
                    "set": "natural",
                    "arm_run": f"natural_name_selected_v{version}",
                    "base_run": "natural_name_v1_replay_0326",
                }
            )
    for card in adapted_cards():
        base = str(get("lora.base_tag"))
        rows.append(
            {
                "arm": "low_rank_adaptation",
                "arm_model": card.tag,
                "base_model": base,
                "set": "control_test",
                "arm_run": "m1c_name_full_v1",
                "base_run": "m1c_name_full_v1",
            }
        )
        rows.append(
            {
                "arm": "low_rank_adaptation",
                "arm_model": card.tag,
                "base_model": base,
                "set": "natural",
                "arm_run": "natural_name_v1_replay_0326",
                "base_run": "natural_name_v1_replay_0326",
            }
        )
    return rows


def judged(run: str, tag: str, which: str) -> list[scoring.Scored] | None:
    """Return one model's judged answers on one comparison set, or None if absent."""
    cards = {card.tag: card for card in pool() + adapted_cards()}
    target = response_path(run, cards[tag])
    if not target.exists():
        return None
    answers = scoring.read_answers(target)
    if which == "natural":
        return scoring.score_name_natural(answers, scoring.natural_truth())
    truth = control_truth()
    return [
        one
        for one in scoring.score_name_injected(answers, truth)
        if truth[one.item_id]["split"] == "test"
    ]


def main() -> None:
    """Score every comparison whose answers are complete."""
    out: list[dict[str, Any]] = []
    for row in comparisons():
        arm = judged(row["arm_run"], row["arm_model"], row["set"].split("_")[0])
        base = judged(row["base_run"], row["base_model"], row["set"].split("_")[0])
        if arm is None or base is None:
            logger.warning(f"missing answers for {row}")
            continue
        arm_ok, base_ok = scoring.correctness(arm), scoring.correctness(base)
        if set(arm_ok) != set(base_ok):
            logger.warning(
                f"incomplete comparison {row}: {len(arm_ok)} vs {len(base_ok)}"
            )
            continue
        first, second = scoring.discordant(arm_ok, base_ok)
        test = exact_mcnemar(first, second)
        arm_c, base_c = scoring.confusion(arm), scoring.confusion(base)
        arm_low, arm_high = arm_c.youdens_j_interval()
        base_low, base_high = base_c.youdens_j_interval()
        out.append(
            {
                **row,
                "items": len(arm_ok),
                "positives": arm_c.true_positive + arm_c.false_negative,
                "arm_youdens_j": round(arm_c.youdens_j, 4),
                "arm_youdens_j_low": round(arm_low, 4),
                "arm_youdens_j_high": round(arm_high, 4),
                "base_youdens_j": round(base_c.youdens_j, 4),
                "base_youdens_j_low": round(base_low, 4),
                "base_youdens_j_high": round(base_high, 4),
                "arm_flag_rate": round(arm_c.flag_rate().estimate, 4),
                "base_flag_rate": round(base_c.flag_rate().estimate, 4),
                "arm_only_correct": first,
                "base_only_correct": second,
                "p_value": round(test.p_value, 6),
                "odds_ratio": (
                    None if test.odds_ratio is None else round(test.odds_ratio, 4)
                ),
                "arm_invalid": sum(1 for one in arm if not one.valid),
            }
        )
        logger.info(f"{row['arm']} {row['arm_model']} {row['set']}: {out[-1]}")
    if out:
        target = path("data_processed") / "analysis" / "arm_comparisons.csv"
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(out[0]))
            writer.writeheader()
            writer.writerows(out)
        logger.info(f"wrote {target}")


if __name__ == "__main__":
    main()
