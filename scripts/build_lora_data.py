"""Write the training file for the low-rank adaptation arm.

Each example is the prompt the pool is sent for the residual question under
prompt version 1, rendered by the same function the runner uses, followed by the
schema's answer for that item. Only the control set's training split is written:
the development and test splits and every natural-set record stay out of it,
which the tests check on the built file.

Outputs:
    data/processed/lora/train.jsonl
    data/processed/lora/settings.json

Usage:
    python -m scripts.build_lora_data
"""

from __future__ import annotations

import json
import random
from typing import Any

from poi_audit import prompting
from poi_audit.config import get, path
from poi_audit.logsetup import setup

logger = setup("build_lora_data")


def example_for(item: dict[str, Any]) -> dict[str, Any]:
    """Return one training example in chat form.

    Args:
        item: A control-set item.

    Returns:
        The system and user messages exactly as the pool receives them, and the
        answer the schema admits for the item.
    """
    rendered = prompting.render(prompting.NAME, 1, dict(item["record"]), prompting.FULL)
    answer = {"verdict": "wrong" if item["corrupted"] else "belongs"}
    return {
        "item_id": item["item_id"],
        "messages": [
            {"role": "system", "content": rendered.system},
            {"role": "user", "content": rendered.user},
            {"role": "assistant", "content": json.dumps(answer)},
        ],
    }


def main() -> None:
    """Build the training file from the control set's training split."""
    settings = get("lora")
    source = path("data_processed") / "injected_set" / "residual_control.jsonl"
    with source.open(encoding="utf-8") as handle:
        items = [json.loads(line) for line in handle if line.strip()]
    train = [item for item in items if item["split"] == settings["train_split"]]
    random.Random(int(settings["seed"])).shuffle(train)
    out = path("data_processed") / "lora"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "train.jsonl").open("w", encoding="utf-8") as handle:
        for item in train:
            handle.write(json.dumps(example_for(item), ensure_ascii=False) + "\n")
    (out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", "utf-8")
    logger.info(f"wrote {len(train)} training examples")


if __name__ == "__main__":
    main()
