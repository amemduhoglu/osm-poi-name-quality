"""Score the frontier references, and report them beside the pool rather than in it.

The size rule admitted the first pair to keep open the possibility that the
residual class yields to capacity; the second pair was added on 2026-08-14 to
answer an editor's request for a state-of-the-art comparison and for the use of
large language models to be addressed. All four are evaluated references:
their scores are reported, and no pool comparison, figure or fit reads them.
That separation is why this is a script of its own writing a table of its own,
rather than extra rows in the pool's table, where a later analysis could pick
them up without meaning to.

What it writes, per reference:

    The residual class under the full view, which is the run the rule turns on,
    and under the position alone, which is the input-dependence gate. A gate
    applies to a reference exactly as it applies to a pool model: a score whose
    answers do not move when the record is taken away is not a capability.

    The natural set, scored on the adjudicated set and on the decidable core, so
    that the reference is read on the same two legs the pool is.

    The paired comparison against the largest pool model, on the records where
    the two disagreed, which is what says whether the reference is
    distinguishable from a model a reader could run on their own card.

Outputs:
    data/processed/analysis/reference_scores.csv
    data/processed/analysis/reference_pairs.csv

Usage:
    python scripts/score_reference.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from poi_audit import scoring
from poi_audit.config import get, path
from poi_audit.inference import ModelCard, pool
from poi_audit.logsetup import setup
from poi_audit.stats import Confusion, clustered_paired_difference, exact_mcnemar

logger = setup("score_reference")

CONFIG = "models.reference_api"
FULL = "m1_name_full_v1"
NEITHER = "m1_name_neither_v1"
NATURAL = "natural_name_v1"
THRESHOLD = float(get("analysis.significance_threshold"))


def out_dir() -> Path:
    """Return the directory the analysis outputs are written to."""
    target = path("data_processed") / "analysis"
    target.mkdir(parents=True, exist_ok=True)
    return target


def reference_answers(run: str, model: str) -> list[dict[str, Any]]:
    """Read one reference's answers for one run.

    Args:
        run: The run's name.
        model: The reference model's identifier.

    Returns:
        The answers as the runner wrote them.

    Raises:
        FileNotFoundError: If the run has not been answered.
    """
    slug = model.replace("/", "-")
    target = path("responses") / run / f"reference-{slug}.jsonl"
    if not target.exists():
        raise FileNotFoundError(f"{model} has not answered {run}: {target} is absent")
    return scoring.read_answers(target)


def comparators(run: str, judge) -> list[tuple[str, ModelCard]]:
    """Return the pool models a reference is compared against, and why each.

    Two comparisons rather than one, because they answer different questions.
    The largest model asks what more capacity buys, which is what a reference
    was admitted to establish. The best-scoring model asks whether the reference
    is distinguishable from what a reader could actually run, which is the
    comparison this study's own gate is written in terms of: a claim is made
    against the best trivial answer available, not the first one written.

    Args:
        run: The run the comparison is made on.
        judge: A callable turning one model's answers into judgements.

    Returns:
        One entry per comparator, carrying the role it plays and its card. The
        two coincide when the largest model is also the best, and the duplicate
        is dropped.

    Raises:
        FileNotFoundError: If a pool model has not answered the run.
    """
    cards = pool()
    scored = []
    for card in cards:
        target = path("responses") / run / f"{card.slug}.jsonl"
        if not target.exists():
            raise FileNotFoundError(f"{card.tag} has not answered {run}")
        counts = scoring.confusion(judge(scoring.read_answers(target)))
        scored.append((counts.balanced_accuracy, card))
    biggest = max(cards, key=lambda card: card.parameters_b)
    best = max(scored, key=lambda entry: entry[0])[1]
    chosen = [("largest in the pool", biggest)]
    if best.tag != biggest.tag:
        chosen.append(("best in the pool", best))
    return chosen


def row(
    name: str, measure: str, counts: Confusion, extra: dict[str, Any]
) -> dict[str, Any]:
    """Return one scored row.

    Args:
        name: The reference's printed name.
        measure: What was scored.
        counts: The confusion the score comes from.
        extra: Columns particular to this measure.

    Returns:
        The row, carrying the interval every proportion in this study carries.
    """
    flag = counts.flag_rate()
    return {
        "reference": name,
        "measure": measure,
        "items": counts.total,
        "positives": counts.positives,
        "balanced_accuracy": round(counts.balanced_accuracy, 4),
        "youdens_j": round(counts.youdens_j, 4),
        "flag_rate": round(flag.estimate, 4),
        "flag_rate_low": round(flag.low, 4),
        "flag_rate_high": round(flag.high, 4),
        **extra,
    }


def score_one(card: dict[str, Any], gold: dict[str, Any], truth: dict[str, Any]):
    """Score one reference on every run it answered.

    Args:
        card: The reference as the configuration declares it.
        gold: The natural set's adjudicated labels.
        truth: The injected set's truth.

    Returns:
        Its scored rows and its paired comparisons against the largest pool
        model.
    """
    name = str(card["name"])
    model = str(card["model"])
    full = scoring.score_name_injected(reference_answers(FULL, model), truth)
    neither = scoring.score_name_injected(reference_answers(NEITHER, model), truth)
    natural = reference_answers(NATURAL, model)
    adjudicated = scoring.score_name_natural(natural, gold)
    core = scoring.score_name_natural(natural, gold, core_only=True)

    first, second = scoring.discordant(
        scoring.correctness(full), scoring.correctness(neither)
    )
    gate = exact_mcnemar(first, second)
    rows = [
        row(
            name,
            "residual class, full view",
            scoring.confusion(full),
            {
                "gate_p_value": gate.p_value,
                "passes_input_dependence": bool(gate.p_value < THRESHOLD),
            },
        ),
        row(name, "residual class, position alone", scoring.confusion(neither), {}),
        row(name, "natural set, adjudicated", scoring.confusion(adjudicated), {}),
        row(name, "natural set, decidable core", scoring.confusion(core), {}),
    ]

    pairs = []
    for measure, judged, pool_run, scorer in (
        ("residual class", full, FULL, "injected"),
        ("natural set", adjudicated, NATURAL, "natural"),
    ):
        judge = (
            (lambda answers: scoring.score_name_injected(answers, truth))
            if scorer == "injected"
            else (lambda answers: scoring.score_name_natural(answers, gold))
        )
        for role, comparator in comparators(pool_run, judge):
            against = judge(
                scoring.read_answers(
                    path("responses") / pool_run / f"{comparator.slug}.jsonl"
                )
            )
            first, second = scoring.discordant(
                scoring.correctness(judged), scoring.correctness(against)
            )
            result = exact_mcnemar(first, second)
            spread = clustered_paired_difference(
                scoring.correctness(judged),
                scoring.correctness(against),
                scoring.clusters(),
            )
            # A conclusion in this study rests on p < 0.001, so the interval a
            # lead is read from is the one that matches that threshold. A 95
            # per cent interval clearing zero says p < 0.05 and would set the
            # size of a claimed difference against a bar the article does not
            # use anywhere else.
            strict = clustered_paired_difference(
                scoring.correctness(judged),
                scoring.correctness(against),
                scoring.clusters(),
                confidence=1.0 - THRESHOLD,
            )
            # The pairing is made on whether each record was answered
            # correctly, which is how this study compares two models. On a set
            # whose corrupted share is far from a half that quantity rewards
            # restraint: a model that rarely flags is right about most records
            # and has decided very little. The two summaries that separate the
            # readings travel in the row, so that no one has to fetch them from
            # another table to read it correctly.
            mine = scoring.confusion(judged)
            theirs = scoring.confusion(against)
            pairs.append(
                {
                    "reference": name,
                    "against": comparator.name,
                    "against_role": role,
                    "against_parameters_b": comparator.parameters_b,
                    "measure": measure,
                    "reference_balanced_accuracy": round(mine.balanced_accuracy, 4),
                    "other_balanced_accuracy": round(theirs.balanced_accuracy, 4),
                    "reference_flag_rate": round(mine.flag_rate().estimate, 4),
                    "other_flag_rate": round(theirs.flag_rate().estimate, 4),
                    "reference_only_correct": result.first_only,
                    "other_only_correct": result.second_only,
                    "p_value": result.p_value,
                    "odds_ratio": result.odds_ratio,
                    "accuracy_difference": spread.difference,
                    "difference_low": spread.low,
                    "difference_high": spread.high,
                    "difference_low_at_threshold": strict.low,
                    "difference_high_at_threshold": strict.high,
                    "clustered_p_value": spread.p_value,
                    "clusters": spread.clusters,
                    "separable": bool(result.p_value < THRESHOLD),
                    # A lead is claimed only where both readings clear the
                    # threshold the article draws conclusions at: the exact
                    # test on the discordant records, and the resampled
                    # interval at the matching coverage.
                    "leads_at_threshold": bool(
                        result.p_value < THRESHOLD
                        and spread.p_value < THRESHOLD
                        and strict.low > 0.0
                    ),
                }
            )
    return rows, pairs


def write(name: str, rows: list[dict[str, Any]]) -> None:
    """Write one table.

    Args:
        name: The file name.
        rows: The rows.
    """
    target = out_dir() / name
    fields: list[str] = []
    for entry in rows:
        for key in entry:
            if key not in fields:
                fields.append(key)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"wrote {target} with {len(rows)} rows")


def main() -> None:
    """Score every configured reference and report it beside the pool."""
    truth = scoring.injected_truth()
    gold = scoring.natural_truth()
    scores: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for card in get(f"{CONFIG}.models"):
        card = dict(card)
        rows, paired = score_one(card, gold, truth)
        scores.extend(rows)
        pairs.extend(paired)
        answers = reference_answers(FULL, str(card["model"]))
        provenance.append(
            {
                "reference": card["name"],
                "model": card["model"],
                "weights": card.get("weights"),
                # The licence and the date it was read travel with the URL,
                # because a deposit that names a source without them names
                # something a reader cannot check.
                "weights_licence": card.get("weights_licence"),
                # The loader gives a date back as a date; it is written as the
                # string it was read as, so the deposit holds one form of it.
                "weights_checked": str(card.get("weights_checked") or ""),
                "parameters_b": card.get("parameters_b"),
                # A mixture of experts activates a fraction of what it holds,
                # and a capacity claim read off the total alone overstates what
                # the comparison varied. Both numbers travel together.
                "active_parameters_b": card.get("active_parameters_b"),
                # The pool is served at Q4_K_M and the references at whatever
                # the provider serves. The pool excluded a model for exactly
                # this reason, so the asymmetry is recorded rather than left to
                # the phrase "the same decoding conditions".
                "serving_quantizations": list(
                    (card.get("provider") or {}).get("quantizations") or []
                ),
                "providers_that_answered": sorted(
                    {str(one.get("provider")) for one in answers}
                ),
            }
        )
    write("reference_scores.csv", scores)
    write("reference_pairs.csv", pairs)
    (out_dir() / "reference_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(f"wrote the provenance of {len(provenance)} references")


if __name__ == "__main__":
    main()
