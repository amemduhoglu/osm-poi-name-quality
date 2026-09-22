"""Compose the three instruments into pipelines and measure what each one buys.

The instruments are scored separately everywhere else in this study. A maintainer
deciding how to check a repository does not choose an instrument; they choose an
order and a place to stop. This script measures the orders.

Six configurations are compared, on the injected set and again on the real records
that nobody touched:

    C0  every record to a person
    C1  the format rule alone
    C2  the format rule, then the open reference lookup
    C3  the rule, the lookup, then a model, whose flags go to a person
    C4  the rule, the lookup, then every remaining candidate to a person
    C5  a model alone
    C6  the same number of records to a person, chosen at random

Records enter at the cheapest stage and leave as soon as one of them settles them,
so what each stage passes on is what the next stage sees. A record routed to a
person is counted as caught when it really carries the error, because the person is
the standard the models are scored against; the cost of that routing is counted in
reader minutes.

**No model is run again.** Every number is a recombination of responses already on
disk, of the rule's per-item verdicts, of the reference lookup's, and of the
measured efficiency profile.

Reader cost is measured rather than assumed, from the seconds the two readers spent
per item, under the three treatments the reframe plan fixes: the median as the
primary estimate, a mean winsorized at each reader's own 95th percentile as the
sensitivity, and the untreated mean so that the size of the distortion stays
visible. A reader who steps away is recorded as a reader who is working, and the
distortion is one-sided.

C6 is the trivial answer C3 has to beat. A model stage that sends a person the
same number of records a coin would have sent them, and finds no more errors in
them, has triaged nothing. This configuration was added on 12 August 2026 **after**
the first composition was computed and after C3 was seen to be undominated, which
is stated here because the order matters: it is not a criterion chosen to reach a
conclusion, it is the study's own first gate, beating the cheapest instrument
capable of the same answer, applied to a configuration rather than to a model, and
its absence from the plan was an oversight rather than a decision.

On the injected set the rule's recall over the malformed and contradicted classes is
guaranteed by the construction, so composed recall there is a property of how the
items were built. The evidential weight rests on the real records.

Outputs:
    data/processed/analysis/pipeline.csv    one row per configuration and set
    data/processed/analysis/pipeline.json   the summary and the criterion's outcome

Usage:
    python -m scripts.score_pipeline
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from math import comb
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from poi_audit import rules, scoring  # noqa: E402
from poi_audit.config import get, path  # noqa: E402
from poi_audit.inference import pool, response_path  # noqa: E402
from poi_audit.logsetup import setup  # noqa: E402
from poi_audit.references import load_gazetteer  # noqa: E402
from poi_audit.stats import Confusion  # noqa: E402

logger = setup("score_pipeline")

CONFIDENCE = 0.95

# The run whose answers stand for the model stage on each set. Both ask the
# residual question under the first prompt on the full record, which is the
# condition every other view is compared against.
INJECTED_RUN = "m1_name_full_v1"
NATURAL_RUN = "natural_name_v1"

# A thousand records is the unit a maintainer budgets in, and every cost below is
# reported against it.
PER = 1000


@dataclass(frozen=True)
class Row:
    """One configuration measured on one set.

    Attributes:
        item_set: ``injected`` or ``natural``.
        configuration: The configuration's tag.
        model: The model at the model stage, or an empty string where there is none.
        records: Records the configuration saw.
        positives: Records that really carried the error.
        caught: Positives the configuration caught, by machine or by person.
        recall: Caught over positives.
        recall_low: Lower bound of the 95% Wilson interval.
        recall_high: Upper bound of the same interval.
        machine_flags_per_1k: Records the machine stages flagged, per thousand.
        to_person_per_1k: Records routed to a person, per thousand.
        machine_seconds_per_1k: Machine time, per thousand records.
        reader_minutes_median_per_1k: Reader time under the primary treatment.
        reader_minutes_winsorized_per_1k: Reader time under the sensitivity.
        reader_minutes_untreated_per_1k: Reader time with the breaks left in.
    """

    item_set: str
    configuration: str
    model: str
    records: int
    positives: int
    caught: int
    recall: float
    recall_low: float
    recall_high: float
    machine_flags_per_1k: float
    to_person_per_1k: float
    machine_seconds_per_1k: float
    reader_minutes_median_per_1k: float
    reader_minutes_winsorized_per_1k: float
    reader_minutes_untreated_per_1k: float


def out_dir() -> Path:
    """Return the directory the analysis outputs are written to.

    Returns:
        The analysis directory, created if it does not exist.
    """
    target = path("data_processed") / "analysis"
    target.mkdir(parents=True, exist_ok=True)
    return target


def reader_seconds() -> dict[str, float]:
    """Return the seconds a person takes over one record, under three treatments.

    A break between opening an item and answering it is recorded as work. The
    treatments are fixed in the reframe plan and are applied to the timing alone:
    no label is discarded and the gold standard is untouched.

    Returns:
        Seconds per record under ``median``, ``winsorized`` and ``untreated``.

    Raises:
        FileNotFoundError: If no reader recorded a time.
    """
    labels = path("data_processed") / "natural_set" / "labels"
    per_reader: dict[str, list[float]] = {}
    for source in sorted(labels.glob("*.jsonl")):
        if source.name.startswith("adjudication") or ".practice." in source.name:
            continue
        for line in source.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("kind") != "label" or record.get("run") != "main":
                continue
            seconds = record.get("seconds_on_item")
            if seconds is None:
                continue
            per_reader.setdefault(str(record["annotator"]), []).append(float(seconds))
    if not per_reader:
        raise FileNotFoundError("no reader times found beside the labels")

    pooled = [value for values in per_reader.values() for value in values]
    winsorized: list[float] = []
    for values in per_reader.values():
        ordered = sorted(values)
        cap = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
        winsorized.extend(min(value, cap) for value in values)
    return {
        "median": statistics.median(pooled),
        "winsorized": statistics.fmean(winsorized),
        "untreated": statistics.fmean(pooled),
    }


def eligible_models() -> dict[str, float]:
    """Return the models whose residual-class score survives the gates.

    Two of the four gates are per-model properties of the residual question and
    those are the two that decide whether a model may stand at the model stage:
    the flag-rate correction, which refuses a detection score to a model that
    flags at any rate, and input dependence, which refuses one to a model whose
    answers do not move when the record is taken away. Baseline dominance is a
    property of the detection task rather than of this question, and label
    stability is a property of the set. A model failing either applicable gate is
    not a candidate, because placing it at the model stage would credit a score
    the study has already declined to read as a capability.

    Returns:
        The eligible models, each with its corrected score on the residual
        question, which is what "leads the residual class" means here.
    """
    table = out_dir() / "gates.csv"
    if not table.exists():
        return {}
    passes: dict[str, dict[str, bool]] = {}
    corrected: dict[str, float] = {}
    with table.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            model = row["model"]
            if not model:
                continue
            passes.setdefault(model, {})[row["gate"]] = row["passes"] == "True"
            if row["gate"] == "flag_rate_correction":
                corrected[model] = float(row["value"])
    applicable = ("flag_rate_correction", "input_dependence")
    return {
        model: corrected.get(model, 0.0)
        for model, outcome in passes.items()
        if all(outcome.get(gate, False) for gate in applicable)
    }


def machine_seconds(model: str) -> float:
    """Return the measured seconds a model takes over one record.

    Args:
        model: The model's run name.

    Returns:
        Median seconds per record, or 0.0 when the model is not in the profile.
    """
    profile = out_dir() / "cost.csv"
    if not profile.exists():
        return 0.0
    with profile.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["model"] == model:
                return float(row["median_sec_per_record"])
    return 0.0


def random_triage(pool_size: int, pool_positives: int, reviewed: int) -> tuple:
    """Return what reviewing the same number of records at random would catch.

    Sending a person `reviewed` records drawn without replacement from a pool of
    `pool_size` containing `pool_positives` errors catches a hypergeometric
    number of them. The distribution is computed exactly rather than simulated,
    so the answer does not depend on a seed.

    Args:
        pool_size: Records the machine stages passed.
        pool_positives: Errors among them.
        reviewed: Records a person sees.

    Returns:
        The expected number caught and an equal-tailed 95% interval.
    """
    if pool_size <= 0 or reviewed <= 0 or pool_positives <= 0:
        return 0.0, 0, 0
    reviewed = min(reviewed, pool_size)
    lowest = max(0, reviewed - (pool_size - pool_positives))
    highest = min(reviewed, pool_positives)
    total = comb(pool_size, reviewed)
    counts = range(lowest, highest + 1)
    weights = [
        comb(pool_positives, k) * comb(pool_size - pool_positives, reviewed - k) / total
        for k in counts
    ]
    expected = sum(k * w for k, w in zip(counts, weights, strict=True))
    cumulative = 0.0
    low = high = lowest
    for k, weight in zip(counts, weights, strict=True):
        cumulative += weight
        if cumulative < 0.025:
            low = k + 1
        if cumulative <= 0.975:
            high = k + 1
    return expected, low, min(high, highest)


def stage_verdicts(items: list[dict[str, Any]], gazetteer: Any) -> dict[str, tuple]:
    """Run the two machine stages over every item, keeping them apart.

    The composition needs the stages separately, because a record leaves the
    pipeline at the first stage that settles it and the next stage never sees it.

    Args:
        items: The injected item set.
        gazetteer: The loaded gazetteer the lookups resolve against.

    Returns:
        Per item, whether the format rule flagged and whether the lookup did.
    """
    staged: dict[str, tuple] = {}
    for item in items:
        record = scoring.as_record(item)
        rule = rules.flagged(rules.format_rules(record["tags"]))
        lookup = rules.flagged(
            rules.reference_lookups(
                record["tags"],
                str(record["country"]),
                float(record["lat"]),
                float(record["lon"]),
                gazetteer,
            )
        )
        staged[str(item["item_id"])] = (rule, lookup)
    return staged


def model_flags(run: str, card: Any) -> dict[str, bool]:
    """Read one model's answers on one run, as a flag per item.

    Args:
        run: The run's name.
        card: The model, as the pool holds it.

    Returns:
        Whether the model flagged each item it answered. An unparseable answer
        counts as a failure to flag, never as an absent measurement.
    """
    target = response_path(run, card)
    if not target.exists():
        return {}
    flags: dict[str, bool] = {}
    for answer in scoring.read_answers(target):
        item = str(answer.get("item_id", ""))
        if not item:
            continue
        parsed = answer.get("parsed") or {}
        flags[item] = (
            bool(answer.get("valid")) and parsed.get("verdict") == scoring.WRONG
        )
    return flags


def interval(caught: int, positives: int) -> tuple[float, float, float]:
    """Return a recall and its 95% Wilson interval.

    Args:
        caught: Positives the configuration caught.
        positives: Positives present.

    Returns:
        The point estimate and the interval, all zero when nothing was present.
    """
    if positives == 0:
        return 0.0, 0.0, 0.0
    counts = Confusion(
        true_positive=caught,
        false_negative=positives - caught,
        false_positive=0,
        true_negative=0,
    )
    estimate = counts.sensitivity(CONFIDENCE)
    return estimate.estimate, estimate.low, estimate.high


def compose(
    item_set: str,
    truth: dict[str, bool],
    staged: dict[str, tuple],
    candidates: set[str],
    flags: dict[str, bool],
    model: str,
    seconds: dict[str, float],
) -> list[Row]:
    """Measure every configuration on one set.

    Args:
        item_set: The set's name, for the output.
        truth: Whether each record really carries the error.
        staged: Per record, whether the rule flagged and whether the lookup did.
        candidates: Records the model stage is offered, of those the machine
            stages pass. On the injected set this is the residual class; on the
            real records it is every record, since none was constructed.
        flags: The model's flag per record.
        model: The model's name.
        seconds: Reader seconds per record, under the three treatments.

    Returns:
        One row per configuration.
    """
    records = sorted(truth)
    total = len(records)
    positives = sum(truth.values())
    machine_sec = machine_seconds(model)

    def emit(
        tag: str,
        caught: int,
        machine_flagged: int,
        to_person: int,
        model_records: int,
        named_model: str,
    ) -> Row:
        point, low, high = interval(caught, positives)
        scale = PER / total if total else 0.0
        return Row(
            item_set=item_set,
            configuration=tag,
            model=named_model,
            records=total,
            positives=positives,
            caught=caught,
            recall=round(point, 4),
            recall_low=round(low, 4),
            recall_high=round(high, 4),
            machine_flags_per_1k=round(machine_flagged * scale, 1),
            to_person_per_1k=round(to_person * scale, 1),
            machine_seconds_per_1k=round(model_records * machine_sec * scale, 1),
            reader_minutes_median_per_1k=round(
                to_person * seconds["median"] * scale / 60, 1
            ),
            reader_minutes_winsorized_per_1k=round(
                to_person * seconds["winsorized"] * scale / 60, 1
            ),
            reader_minutes_untreated_per_1k=round(
                to_person * seconds["untreated"] * scale / 60, 1
            ),
        )

    rule_flag = {item: staged.get(item, (False, False))[0] for item in records}
    lookup_flag = {item: staged.get(item, (False, False))[1] for item in records}
    machine_flag = {item: rule_flag[item] or lookup_flag[item] for item in records}
    passed = [item for item in records if not machine_flag[item]]
    offered = [item for item in passed if item in candidates]

    rows = [
        emit("C0", positives, 0, total, 0, ""),
        emit(
            "C1",
            sum(1 for i in records if rule_flag[i] and truth[i]),
            sum(1 for i in records if rule_flag[i]),
            0,
            0,
            "",
        ),
        emit(
            "C2",
            sum(1 for i in records if machine_flag[i] and truth[i]),
            sum(1 for i in records if machine_flag[i]),
            0,
            0,
            "",
        ),
    ]
    caught_machine = sum(1 for i in records if machine_flag[i] and truth[i])
    model_flagged = [i for i in offered if flags.get(i, False)]
    rows.append(
        emit(
            "C3",
            caught_machine + sum(1 for i in model_flagged if truth[i]),
            sum(1 for i in records if machine_flag[i]) + len(model_flagged),
            len(model_flagged),
            len(offered),
            model,
        )
    )
    rows.append(
        emit(
            "C4",
            caught_machine + sum(1 for i in offered if truth[i]),
            sum(1 for i in records if machine_flag[i]),
            len(offered),
            0,
            "",
        )
    )
    alone = [i for i in records if i in candidates and flags.get(i, False)]
    rows.append(
        emit(
            "C5",
            sum(1 for i in alone if truth[i]),
            len(alone),
            len(alone),
            len([i for i in records if i in candidates]),
            model,
        )
    )
    # The trivial answer C3 has to beat: the same review volume, chosen at
    # random from the same pool. Its caught count is the expectation, rounded
    # down, and its interval is carried in the summary rather than in the row,
    # since a row reports one configuration's recall and this one's uncertainty
    # is about the draw rather than about the sample.
    expected, low, high = random_triage(
        len(offered), sum(1 for i in offered if truth[i]), len(model_flagged)
    )
    # C6 carries the model's tag even though it runs no model: its review volume
    # is the one that model asked for, so a row without the tag would be read as
    # a property of the set rather than of the comparison it belongs to.
    rows.append(
        emit(
            "C6",
            caught_machine + int(expected),
            sum(1 for i in records if machine_flag[i]),
            len(model_flagged),
            0,
            model,
        )
    )
    rows[-1] = replace(
        rows[-1],
        recall_low=round((caught_machine + low) / positives if positives else 0.0, 4),
        recall_high=round((caught_machine + high) / positives if positives else 0.0, 4),
    )
    return rows


def dominated(rows: list[Row]) -> dict[str, Any]:
    """Apply the acceptance criterion fixed before the composition was computed.

    C3 is dominated when C2 or C4 reaches at least its recall at no greater reader
    cost. A dominated model stage does not pay, and the recommendation is whichever
    configuration dominates it. The outcome is reported whichever way it falls, and
    it must hold under all three reader-time treatments.

    Args:
        rows: Every measured row for one set.

    Returns:
        The verdict, the dominating configuration where there is one, and whether
        the three treatments agree.
    """
    by_tag = {row.configuration: row for row in rows}
    three = (
        "reader_minutes_median_per_1k",
        "reader_minutes_winsorized_per_1k",
        "reader_minutes_untreated_per_1k",
    )
    verdicts: dict[str, str] = {}
    for treatment in three:
        target = by_tag["C3"]
        winner = ""
        for tag in ("C2", "C4"):
            other = by_tag[tag]
            if other.recall >= target.recall and getattr(other, treatment) <= getattr(
                target, treatment
            ):
                winner = tag
                break
        verdicts[treatment] = winner
    agreed = len(set(verdicts.values())) == 1
    is_dominated = bool(verdicts[three[0]])

    # The second test, added 12 August 2026 after the first composition was
    # computed. A model stage that sends a person the same number of records a
    # coin would have sent them, and finds no more errors among them, has
    # triaged nothing. C3 beats the random triage only when its catch clears the
    # upper bound of what the same review volume would catch by chance, which is
    # the conservative reading and the one the study applies elsewhere.
    target = by_tag["C3"]
    chance = by_tag.get("C6")
    beats_chance = bool(chance) and target.caught > chance.recall_high * max(
        target.positives, 1
    )

    if is_dominated:
        verdict = (
            f"C3 is dominated by {verdicts[three[0]]}; the model stage does not pay"
        )
    elif not beats_chance:
        verdict = (
            "C3 is undominated but does not beat a random triage of the same "
            "volume; the model stage sorts nothing and does not pay"
        )
    else:
        verdict = "C3 is undominated and beats a random triage; the model stage pays"
    if not agreed:
        verdict = "inconclusive: the reader-time treatments disagree"
    return {
        "c3_dominated": is_dominated,
        "dominating_configuration": verdicts[three[0]],
        "treatments_agree": agreed,
        "by_treatment": verdicts,
        "beats_random_triage": beats_chance,
        "random_triage_upper_bound_recall": (
            round(chance.recall_high, 4) if chance else None
        ),
        "c3_recall": round(target.recall, 4),
        "model_stage_pays": bool(agreed and not is_dominated and beats_chance),
        "verdict": verdict,
    }


def build() -> tuple[list[Row], dict[str, Any]]:
    """Measure every configuration on both sets.

    Returns:
        Every row, and the summary carrying the criterion's outcome.
    """
    seconds = reader_seconds()
    logger.info(
        "reader seconds per record: "
        f"median {seconds['median']:.1f}, "
        f"winsorized mean {seconds['winsorized']:.1f}, "
        f"untreated mean {seconds['untreated']:.1f}"
    )

    gazetteer = load_gazetteer()
    items = [
        json.loads(line)
        for line in (path("data_processed") / "injected_set" / "items.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    staged = stage_verdicts(items, gazetteer)
    injected_truth = {str(i["item_id"]): bool(i["corrupted"]) for i in items}
    # The residual class has one home, in the class registry, and is the class
    # no instrument settles. Reading it from there keeps the code from carrying a
    # second copy of a name the configuration already owns.
    classes = get("injected_set.classes")
    residual = [
        code
        for code, entry in dict(classes).items()
        if str(entry.get("settled_by")) == "none"
    ]
    if len(residual) != 1:
        raise ValueError(f"expected one residual class, found {residual}")
    residual_items = {
        str(i["item_id"]) for i in items if str(i["class"]) == residual[0]
    }

    natural = scoring.natural_truth()
    natural_truth = {
        item: bool(entry["label"] == scoring.WRONG)
        for item, entry in natural.items()
        if entry.get("scored")
    }
    natural_staged = {item: (False, False) for item in natural_truth}

    rows: list[Row] = []
    summary: dict[str, Any] = {
        "written_at": datetime.now(UTC).isoformat(),
        "reader_seconds": {k: round(v, 2) for k, v in seconds.items()},
        "sets": {},
    }
    cards = pool()
    # The model at the model stage is the one that leads the residual class
    # after the gates, not before. Selecting on a raw score would seat a model
    # the study has already refused to read as capable, which is the mistake the
    # gates exist to prevent.
    scored = eligible_models()
    leader = max(scored, key=lambda model: scored[model]) if scored else ""
    logger.info(
        f"{len(scored)} of {len(cards)} models survive the applicable gates; "
        f"the model stage is {leader or 'unavailable'}"
    )
    for item_set, truth, staged_map, candidates, run in (
        ("injected", injected_truth, staged, residual_items, INJECTED_RUN),
        ("natural", natural_truth, natural_staged, set(natural_truth), NATURAL_RUN),
    ):
        headline: list[Row] = []
        for card in cards:
            flags = model_flags(run, card)
            if not flags:
                continue
            measured = compose(
                item_set, truth, staged_map, candidates, flags, card.tag, seconds
            )
            rows.extend(measured)
            if card.tag == leader:
                headline = measured
        if not leader:
            summary["sets"][item_set] = {
                "leading_model": "",
                "criterion": {
                    "verdict": (
                        "C3 is unavailable: no model's residual-class score "
                        "survives the gates that apply to it, so no model may "
                        "stand at the model stage"
                    ),
                    "c3_dominated": None,
                },
            }
        elif headline:
            summary["sets"][item_set] = {
                "leading_model": leader,
                "criterion": dominated(headline),
            }
    return rows, summary


def _cost_rows() -> list[dict[str, str]]:
    """Read the efficiency profile.

    Returns:
        One row per model, empty when the profile has not been built.
    """
    profile = out_dir() / "cost.csv"
    if not profile.exists():
        return []
    with profile.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(rows: list[Row], summary: dict[str, Any]) -> None:
    """Write the table and the summary.

    Args:
        rows: Every measured row.
        summary: The summary carrying the criterion's outcome.
    """
    target = out_dir()
    table = target / "pipeline.csv"
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Row.__annotations__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    (target / "pipeline.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(f"wrote {len(rows)} rows to {table.relative_to(REPO_ROOT)}")


def main() -> None:
    """Measure the configurations and report the criterion's outcome."""
    rows, summary = build()
    for item_set, entry in summary["sets"].items():
        logger.info(f"{item_set}: leading model {entry['leading_model']}")
        logger.info(f"{item_set}: {entry['criterion']['verdict']}")
    write(rows, summary)


if __name__ == "__main__":
    main()
