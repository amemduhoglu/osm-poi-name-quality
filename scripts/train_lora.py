"""Train the low-rank adapter for the adaptation arm.

This script runs in its own environment, because the training stack (torch,
transformers, peft) is not part of the analysis environment and must not move
its pinned versions. It reads only the files `scripts/build_lora_data.py`
writes, trains on the measurement card, and writes the adapter together with the
exact versions and the peak device memory of the run, so that constraint C3 is
checked from the record rather than assumed.

The loss is taken on the answer alone: the system and user messages are the
prompt the pool is sent, and the model is taught to answer it, not to repeat it.
The chat template is the model's own, with thinking disabled, which is how every
scored call is made.

Outputs:
    data/processed/lora/adapter/            the adapter in PEFT form
    data/processed/lora/training_record.json

Usage (in the training environment):
    python scripts/train_lora.py --base <weights dir> --data data/processed/lora
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import peft
import torch
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoTokenizer


def encode(
    tokenizer: Any, messages: list[dict[str, str]]
) -> tuple[list[int], list[int]]:
    """Return the token ids of one example and labels masking all but the answer.

    Args:
        tokenizer: The base model's tokenizer.
        messages: System, user and assistant messages.

    Returns:
        The input ids and the labels, with -100 over the prompt.
    """
    prompt = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full = tokenizer.apply_chat_template(
        messages, tokenize=False, enable_thinking=False
    )
    if not full.startswith(prompt):
        raise ValueError("the chat template does not extend the prompt with the answer")
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("the prompt tokenizes differently inside the full example")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return full_ids, labels


def main() -> None:
    """Train and write the adapter with its record."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    arguments = parser.parse_args()

    settings = json.loads((arguments.data / "settings.json").read_text("utf-8"))
    seed = int(settings["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(arguments.base)
    with (arguments.data / "train.jsonl").open(encoding="utf-8") as handle:
        examples = [json.loads(line) for line in handle if line.strip()]
    encoded = [encode(tokenizer, example["messages"]) for example in examples]

    model = AutoModelForImageTextToText.from_pretrained(
        arguments.base, dtype=torch.bfloat16, device_map={"": 0}
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    config = LoraConfig(
        r=int(settings["rank"]),
        lora_alpha=int(settings["alpha"]),
        lora_dropout=float(settings["dropout"]),
        target_modules=settings["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(settings["learning_rate"]),
    )
    batch = int(settings["batch_size"])
    epochs = int(settings["epochs"])
    steps = epochs * ((len(encoded) + batch - 1) // batch)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, steps // 10), num_training_steps=steps
    )
    # Each example is run alone and the gradient is accumulated over the configured
    # batch, so the optimiser sees the same batch while the card holds one
    # sequence. Logits are computed only over the answer: over the whole prompt
    # the vocabulary-sized logits of one batch do not fit on the card.
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    losses: list[float] = []
    model.train()
    order = list(range(len(encoded)))
    for epoch in range(epochs):
        random.Random(seed + epoch).shuffle(order)
        for start in range(0, len(order), batch):
            chunk = [encoded[i] for i in order[start : start + batch]]
            total = 0.0
            for ids, labels in chunk:
                answer = sum(1 for label in labels if label != -100)
                if labels[-answer:] != ids[-answer:] or -100 in labels[-answer:]:
                    raise ValueError("the answer is not the tail of the example")
                input_ids = torch.tensor([ids], device=0)
                logits = model(input_ids=input_ids, logits_to_keep=answer + 1).logits
                targets = torch.tensor([ids[-answer:]], device=0)
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                )
                (loss / len(chunk)).backward()
                total += float(loss) / len(chunk)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(total)
        recent = losses[-((len(order) + batch - 1) // batch) :]
        print(f"epoch {epoch + 1}: mean loss {np.mean(recent):.4f}", flush=True)

    out = arguments.data / "adapter"
    model.save_pretrained(out)
    record = {
        "examples": len(encoded),
        "trainable_parameters": trainable,
        "steps": len(losses),
        "micro_batch": 1,
        "gradient_accumulation": batch,
        "loss_first": round(losses[0], 4),
        "loss_last": round(losses[-1], 4),
        "seconds": round(time.time() - started, 1),
        "peak_device_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "device": torch.cuda.get_device_name(0),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        "settings": settings,
    }
    (arguments.data / "training_record.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
