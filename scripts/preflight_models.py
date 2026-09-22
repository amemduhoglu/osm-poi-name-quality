"""Check the model pool against the device before any score is produced.

Nothing here scores anything. It answers the four questions that decide whether
a scored run may start at all, and it answers them by measurement rather than
from the configuration file:

    Is this the measurement device. Constraint C3 puts every reported rate on
    the 16 GB card. A run made on a smaller card is a different measurement, so
    the profile is refused unless the visible device matches, and a rehearsal
    says so on its face.

    Is every model the one the pool names. The quantization and the parameter
    count are read from each model's own metadata, because neither is predicted
    by its tag: one family publishes its default build at a different
    quantization, and one family's names describe an effective size rather than
    a resident one. A model whose build differs from the pool's quantization is
    reported and the run is refused, because a difference between two models
    would then be a difference of quantization.

    Does it fit. The footprint is measured on the device with the model loaded
    under the study's own decoding settings, not taken from the artifact's size.

    Does it offload. The server reports how much of a resident model sits in
    device memory. Any part of it on the host stops the model, and an offloaded
    rate is never published as a property of a model.

Outputs (not written by a rehearsal):
    data/processed/efficiency/footprint.csv
    data/processed/efficiency/footprint.json

Usage:
    python -m scripts.preflight_models
    python -m scripts.preflight_models --rehearse   # before the card arrives
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from datetime import UTC, datetime
from typing import Any

import httpx

from poi_audit import prompting
from poi_audit.config import get, secret
from poi_audit.inference import (
    ModelCard,
    Server,
    device_memory_gb,
    efficiency_dir,
    item_set,
    pool,
    relative,
)
from poi_audit.logsetup import setup

logger = setup("preflight_models")

BYTES_PER_GB = 1_000_000_000
# The measured footprint may fall this far under the configured card without the
# device counting as the wrong one: a card reports its usable memory, which is a
# little under the size it is sold as.
DEVICE_TOLERANCE_GB = 1.0
# A parameter count read from metadata is compared with the configured one at
# this tolerance, which is finer than any difference the pool distinguishes.
PARAMETER_TOLERANCE_B = 0.15


def parse_parameters(raw: Any) -> float | None:
    """Return a metadata parameter count in billions.

    Args:
        raw: The count as the server reports it, such as ``8.0B`` or ``752.16M``.

    Returns:
        The count in billions, or None when it cannot be read.
    """
    text = str(raw or "").strip().upper()
    if not text:
        return None
    scale = 1.0
    if text.endswith("B"):
        text = text[:-1]
    elif text.endswith("M"):
        text, scale = text[:-1], 0.001
    try:
        return round(float(text) * scale, 3)
    except ValueError:
        return None


async def probe(server: Server, card: ModelCard) -> dict[str, Any]:
    """Measure one model: its build, its footprint, and whether it offloads.

    The model is loaded by asking it a discarded item, then one item of each
    task at the study's own decoding settings, so that the footprint measured is
    the footprint the runs will use rather than an idle one. The two timed
    answers are diagnostics: they show the model answers, and the study's rates
    are medians measured over the scored runs.

    Args:
        server: The local server.
        card: The model to probe.

    Returns:
        One row of the profile.
    """
    row: dict[str, Any] = {
        "tag": card.tag,
        "family": card.family,
        "configured_parameters_b": card.parameters_b,
        "configured_weights_gb": card.weights_gb,
        "expected_class": card.expected_class,
    }
    try:
        metadata = await server.show(card.tag)
    except httpx.HTTPError as error:
        row["problem"] = f"not served by the local registry: {error}"
        return row

    details = dict(metadata.get("details", {}))
    row["quantization"] = details.get("quantization_level")
    row["metadata_parameters_b"] = parse_parameters(details.get("parameter_size"))
    row["context_length"] = next(
        (
            value
            for key, value in dict(metadata.get("model_info", {})).items()
            if key.endswith(".context_length")
        ),
        None,
    )

    injected = item_set("injected")[0]
    natural = item_set("natural")[0]
    # One answer is asked for and thrown away before anything is timed. The
    # generation that follows a load is measured a quarter to a third slow, and
    # slower still when the weights come from a cold disk, because the card is
    # still settling while the first tokens are produced. Discarding it costs one
    # call per model and keeps a number out of the deposit that would read as a
    # property of the model rather than of its first second alive.
    warm = prompting.render(prompting.DETECT, 1, injected.shown)
    try:
        await server.ask(card.tag, warm.system, warm.user, warm.schema)
    except httpx.HTTPError as error:
        row["problem"] = f"failed to load: {error}"
        return row
    for task, item in ((prompting.DETECT, injected), (prompting.NAME, natural)):
        rendered = prompting.render(task, 1, item.shown)
        try:
            _, metrics = await server.ask(
                card.tag, rendered.system, rendered.user, rendered.schema
            )
        except httpx.HTTPError as error:
            row["problem"] = f"failed to answer the {task} task: {error}"
            return row
        row[f"diagnostic_{task}_total_sec"] = metrics["total_sec"]
        row[f"diagnostic_{task}_output_tokens_per_sec"] = metrics[
            "output_tokens_per_sec"
        ]

    resident = {entry.get("model"): entry for entry in await server.resident()}
    entry = resident.get(card.tag, {})
    size = float(entry.get("size", 0) or 0)
    in_device = float(entry.get("size_vram", 0) or 0)
    row["footprint_gb"] = round(size / BYTES_PER_GB, 3) if size else None
    row["device_memory_gb"] = round(in_device / BYTES_PER_GB, 3) if in_device else None
    row["offloaded"] = bool(size and in_device < size)
    await server.unload(card.tag)
    return row


def classify(row: dict[str, Any]) -> dict[str, Any]:
    """Decide one model's eligibility from what was measured.

    Args:
        row: The probe's row, amended in place.

    Returns:
        The row, carrying its measured class and any problem that stops it.
    """
    threshold = float(get("device.deployment_threshold_gb"))
    quantization = str(get("models.serving.quantization"))
    problems: list[str] = []
    if "problem" in row:
        problems.append(str(row.pop("problem")))
    if row.get("quantization") and row["quantization"] != quantization:
        problems.append(
            f"built at {row['quantization']}, and the pool is {quantization}"
        )
    measured = row.get("metadata_parameters_b")
    if measured is not None:
        difference = abs(float(measured) - float(row["configured_parameters_b"]))
        if difference > PARAMETER_TOLERANCE_B:
            problems.append(
                f"metadata reports {measured}B against the configured "
                f"{row['configured_parameters_b']}B"
            )
    if row.get("offloaded") and not bool(get("device.offload_permitted")):
        problems.append("part of the model sits on the host, which C3 forbids")

    footprint = row.get("footprint_gb")
    row["measured_class"] = (
        None
        if footprint is None
        else ("deployable" if footprint <= threshold else "reference")
    )
    row["as_expected"] = (
        None
        if row["measured_class"] is None
        else row["measured_class"] == row["expected_class"]
    )
    row["problem"] = "; ".join(problems)
    return row


def write_profile(rows: list[dict[str, Any]], device: float | None) -> None:
    """Write the footprint profile the runs and the article read.

    Args:
        rows: One row per model.
        device: The measured device memory in gigabytes.
    """
    out_dir = efficiency_dir()
    columns = [
        "tag",
        "family",
        "configured_parameters_b",
        "metadata_parameters_b",
        "quantization",
        "context_length",
        "configured_weights_gb",
        "footprint_gb",
        "device_memory_gb",
        "offloaded",
        "expected_class",
        "measured_class",
        "as_expected",
        # Named for what they are. These four come from one call each, made to
        # load the model and to prove it answers, and they are diagnostics
        # rather than the study's efficiency result. Every published rate is a
        # median over a whole scored run and is measured by the runner.
        "diagnostic_detect_total_sec",
        "diagnostic_detect_output_tokens_per_sec",
        "diagnostic_name_total_sec",
        "diagnostic_name_output_tokens_per_sec",
        "problem",
    ]
    table = out_dir / "footprint.csv"
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info(f"wrote {relative(table)}")

    summary = {
        "measured_at": datetime.now(UTC).isoformat(),
        "device_memory_gb": device,
        "configured_device_memory_gb": get("device.vram_gb"),
        "deployment_threshold_gb": get("device.deployment_threshold_gb"),
        "quantization": get("models.serving.quantization"),
        "rates_are_diagnostic": (
            "The per-model rates in this profile come from a single call each, "
            "made to prove the model answers. Every rate the study reports is a "
            "median over a scored run, measured in the inference pass."
        ),
        "options": get("run.options"),
        "models": rows,
    }
    document = out_dir / "footprint.json"
    document.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(f"wrote {relative(document)}")


async def run(rehearse: bool, tags: list[str] | None = None) -> int:
    """Probe every model in the pool and report what stops a scored run.

    Args:
        rehearse: Whether this is a rehearsal on a device that is not the
            measurement device. A rehearsal writes no profile.
        tags: Model tags to probe, or None for the whole pool. A partial probe
            is a rehearsal aid and never writes the profile, which has to
            describe every model the runs will use.

    Returns:
        A process exit status: zero when every model is ready to be scored.
    """
    cards = pool()
    if tags:
        unknown = set(tags) - {card.tag for card in cards}
        if unknown:
            logger.error(f"model outside the pool: {', '.join(sorted(unknown))}")
            return 2
        cards = [card for card in cards if card.tag in set(tags)]
        rehearse = True

    configured = float(get("device.vram_gb"))
    device = device_memory_gb()
    if device is None:
        logger.warning("no device could be read; the footprint cannot be measured")
    else:
        logger.info(f"device memory {device} GB against a configured {configured} GB")
    wrong_device = device is None or device < configured - DEVICE_TOLERANCE_GB
    if wrong_device and not rehearse:
        logger.error(
            "this is not the measurement device constraint C3 names; "
            "run with --rehearse to check the pool without writing a profile"
        )
        return 2

    base_url = secret("models.serving.base_url_env")
    timeout = float(get("models.serving.timeout_sec"))
    rows: list[dict[str, Any]] = []
    async with Server(base_url, timeout) as server:
        served = {entry.get("name") for entry in await server.tags()}
        for card in cards:
            if card.tag not in served:
                logger.warning(f"{card.tag}: not pulled; run `ollama pull {card.tag}`")
            logger.info(f"probing {card.tag}")
            rows.append(classify(await probe(server, card)))

    for row in rows:
        if row["problem"]:
            logger.error(f"{row['tag']}: {row['problem']}")
        else:
            logger.info(
                f"{row['tag']}: {row['footprint_gb']} GB, {row['measured_class']}"
                + ("" if row["as_expected"] else ", against the expected class")
            )

    if rehearse:
        logger.warning("rehearsal: no profile written, and no number here is a result")
        return 0
    write_profile(rows, device)
    return 1 if any(row["problem"] for row in rows) else 0


def main() -> None:
    """Read the arguments and run the preflight."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rehearse",
        action="store_true",
        help="check the pool on a device that is not the measurement device",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="probe these tags only, which is a rehearsal and writes no profile",
    )
    arguments = parser.parse_args()
    raise SystemExit(asyncio.run(run(arguments.rehearse, arguments.models)))


if __name__ == "__main__":
    main()
