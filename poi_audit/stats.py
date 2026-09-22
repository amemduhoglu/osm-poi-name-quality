"""The interval and effect-size machinery every reported proportion needs.

One home for the statistics the article prints, so that a proportion reported in
two places is computed one way. The choices here follow the study's reporting
rules rather than convenience: a Wilson score interval because the proportions
this study reports sit near zero and one, where the normal approximation gives
bounds outside the unit interval and coverage well below its nominal level.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest, norm

from poi_audit.config import get

DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True)
class Agreement:
    """Agreement between two readers over the same items.

    Attributes:
        items: How many items both readers labelled.
        observed: The share they labelled identically.
        expected: The share they would match on by chance, from their own
            marginal rates.
        kappa: Cohen's kappa, the observed agreement corrected for chance.
        standard_error: The asymptotic standard error of kappa.
        low: The lower bound of the interval.
        high: The upper bound of the interval.
        confidence: The nominal coverage of the interval.
    """

    items: int
    observed: float
    expected: float
    kappa: float
    standard_error: float
    low: float
    high: float
    confidence: float


def cohen_kappa(
    first: list[str],
    second: list[str],
    categories: tuple[str, ...] | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
) -> Agreement:
    """Return Cohen's kappa for two readings of the same items.

    Raw agreement flatters any task whose labels are unbalanced, and this one is
    expected to be: most names belong to the place carrying them. Kappa asks how
    much of the agreement survives once the rate each reader labels at is
    accounted for, which is why the protocol reports it rather than the raw
    share.

    Args:
        first: One reader's labels, item by item.
        second: The other reader's labels, in the same item order.
        categories: The label set, or None to take the labels that occur. Given
            explicitly, a category neither reader used still enters the table,
            which keeps kappa comparable across subsets.
        confidence: The nominal coverage of the interval.

    Returns:
        The agreement, its chance-corrected statistic and the interval.

    Raises:
        ValueError: If the two readings are of different lengths or are empty.
    """
    if len(first) != len(second):
        raise ValueError(
            f"the two readings cover different items: {len(first)} and {len(second)}"
        )
    if not first:
        raise ValueError("no item was labelled by both readers")
    labels = (
        tuple(categories) if categories else tuple(sorted(set(first) | set(second)))
    )
    total = len(first)
    observed = (
        sum(1 for left, right in zip(first, second, strict=True) if left == right)
        / total
    )
    expected = sum(
        (first.count(label) / total) * (second.count(label) / total) for label in labels
    )
    if expected == 1.0:
        # Both readers used one label for everything: chance alone explains the
        # agreement and kappa is undefined. Reporting zero says that plainly.
        return Agreement(total, observed, expected, 0.0, 0.0, 0.0, 0.0, confidence)
    kappa = (observed - expected) / (1 - expected)
    standard_error = (
        (observed * (1 - observed)) / (total * (1 - expected) ** 2)
    ) ** 0.5
    half = float(norm.ppf(1 - (1 - confidence) / 2)) * standard_error
    return Agreement(
        items=total,
        observed=observed,
        expected=expected,
        kappa=kappa,
        standard_error=standard_error,
        low=max(-1.0, kappa - half),
        high=min(1.0, kappa + half),
        confidence=confidence,
    )


@dataclass(frozen=True)
class Proportion:
    """One proportion with its interval.

    Attributes:
        successes: How many of the trials counted.
        trials: How many were measured.
        estimate: The point estimate, zero when nothing was measured.
        low: The interval's lower bound.
        high: The interval's upper bound.
        confidence: The nominal coverage of the interval.
    """

    successes: int
    trials: int
    estimate: float
    low: float
    high: float
    confidence: float


def wilson(
    successes: int, trials: int, confidence: float = DEFAULT_CONFIDENCE
) -> Proportion:
    """Return a proportion and its Wilson score interval.

    The interval stays inside the unit interval and keeps its coverage when the
    estimate is zero or one, which is what a normal approximation does not, and
    several of the proportions this study reports sit at exactly zero.

    Args:
        successes: How many of the trials counted.
        trials: How many were measured.
        confidence: The nominal coverage, 0.95 unless a caller says otherwise.

    Returns:
        The proportion with its bounds. A measurement of nothing returns zero
        with the whole unit interval, which is the honest answer rather than a
        division by zero.

    Raises:
        ValueError: If the counts are negative, or more successes than trials.
    """
    if successes < 0 or trials < 0:
        raise ValueError(f"negative counts: {successes} of {trials}")
    if successes > trials:
        raise ValueError(f"more successes than trials: {successes} of {trials}")
    if trials == 0:
        return Proportion(0, 0, 0.0, 0.0, 1.0, confidence)
    z = float(norm.ppf(1 - (1 - confidence) / 2))
    estimate = successes / trials
    denominator = 1 + z**2 / trials
    centre = (estimate + z**2 / (2 * trials)) / denominator
    spread = (
        z
        / denominator
        * ((estimate * (1 - estimate) / trials + z**2 / (4 * trials**2)) ** 0.5)
    )
    # At zero and at one the two terms cancel exactly, so the bound is written
    # as the algebra gives it rather than as the floating-point residue of the
    # subtraction, which is otherwise reported as a coverage of 9e-19.
    low = 0.0 if successes == 0 else max(0.0, centre - spread)
    high = 1.0 if successes == trials else min(1.0, centre + spread)
    return Proportion(
        successes=successes,
        trials=trials,
        estimate=estimate,
        low=low,
        high=high,
        confidence=confidence,
    )


@dataclass(frozen=True)
class Confusion:
    """One model's answers on one run, counted against the truth.

    A detection score is summarized by Youden's J rather than by accuracy,
    because accuracy on a set whose corrupted share is not one half rewards a
    model for answering with the majority class, and several comparisons in this
    study run across groups whose corrupted share differs.

    Attributes:
        true_positive: Corrupted records the model flagged.
        false_negative: Corrupted records the model passed.
        true_negative: Clean records the model passed.
        false_positive: Clean records the model flagged.
    """

    true_positive: int
    false_negative: int
    true_negative: int
    false_positive: int

    @property
    def positives(self) -> int:
        """Return how many records were corrupted."""
        return self.true_positive + self.false_negative

    @property
    def negatives(self) -> int:
        """Return how many records were clean."""
        return self.true_negative + self.false_positive

    @property
    def total(self) -> int:
        """Return how many records were answered."""
        return self.positives + self.negatives

    def sensitivity(self, confidence: float = DEFAULT_CONFIDENCE) -> Proportion:
        """Return recall on the corrupted records, with its interval.

        Args:
            confidence: The nominal coverage.

        Returns:
            The proportion of corrupted records the model flagged.
        """
        return wilson(self.true_positive, self.positives, confidence)

    def specificity(self, confidence: float = DEFAULT_CONFIDENCE) -> Proportion:
        """Return the share of clean records the model passed, with its interval.

        Args:
            confidence: The nominal coverage.

        Returns:
            The proportion of clean records the model left alone.
        """
        return wilson(self.true_negative, self.negatives, confidence)

    def flag_rate(self, confidence: float = DEFAULT_CONFIDENCE) -> Proportion:
        """Return how often the model flagged at all, with its interval.

        A detection score is read beside this number and never without it: a
        model that flags everything reaches perfect recall and has decided
        nothing, which is what the flag-rate correction in the audit gates
        exists to catch.

        Args:
            confidence: The nominal coverage.

        Returns:
            The proportion of all records the model flagged.
        """
        flagged = self.true_positive + self.false_positive
        return wilson(flagged, self.total, confidence)

    @property
    def balanced_accuracy(self) -> float:
        """Return the mean of sensitivity and specificity.

        Returns:
            The balanced accuracy, or zero when nothing was measured. A class
            with no members contributes nothing rather than raising, so that a
            run answered on one class alone still reports the other's absence.
        """
        if not self.positives or not self.negatives:
            return 0.0
        return (
            self.true_positive / self.positives + self.true_negative / self.negatives
        ) / 2

    @property
    def youdens_j(self) -> float:
        """Return Youden's J, which is sensitivity plus specificity minus one.

        Returns:
            The statistic, zero at chance and one at a perfect separation. It is
            twice the balanced accuracy minus one, and both are reported because
            the first is the study's detection summary and the second is what a
            comparison across groups of different composition is made on.
        """
        if not self.positives or not self.negatives:
            return 0.0
        return (
            self.true_positive / self.positives
            + self.true_negative / self.negatives
            - 1
        )

    def _sum_interval(self, confidence: float) -> tuple[float, float]:
        """Return the interval for sensitivity plus specificity.

        Both summaries this study reports are linear in that sum: balanced
        accuracy is half of it and Youden's J is one less than it. The bounds
        come from combining the two Wilson intervals by the square-and-add
        method, which keeps the coverage the Wilson intervals were chosen for
        instead of falling back on a normal approximation the counts do not
        support at the ends of the range.

        Args:
            confidence: The nominal coverage.

        Returns:
            The lower and upper bound of the sum, in the interval from zero to
            two, or the whole of it when a class has no members.
        """
        if not self.positives or not self.negatives:
            return 0.0, 2.0
        sensitivity = self.sensitivity(confidence)
        specificity = self.specificity(confidence)
        total = sensitivity.estimate + specificity.estimate
        below = (
            (sensitivity.estimate - sensitivity.low) ** 2
            + (specificity.estimate - specificity.low) ** 2
        ) ** 0.5
        above = (
            (sensitivity.high - sensitivity.estimate) ** 2
            + (specificity.high - specificity.estimate) ** 2
        ) ** 0.5
        return max(0.0, total - below), min(2.0, total + above)

    def balanced_accuracy_interval(
        self, confidence: float = DEFAULT_CONFIDENCE
    ) -> tuple[float, float]:
        """Return the interval around the balanced accuracy.

        Args:
            confidence: The nominal coverage.

        Returns:
            The lower and upper bound, in the interval from zero to one.
        """
        low, high = self._sum_interval(confidence)
        return low / 2, high / 2

    def youdens_j_interval(
        self, confidence: float = DEFAULT_CONFIDENCE
    ) -> tuple[float, float]:
        """Return the interval around Youden's J.

        An interval that covers zero is the arithmetic form of the flag-rate
        correction: a model whose bounds straddle zero has not been told apart
        from one that flags at random.

        Args:
            confidence: The nominal coverage.

        Returns:
            The lower and upper bound, in the interval from minus one to one.
        """
        low, high = self._sum_interval(confidence)
        return low - 1, high - 1


@dataclass(frozen=True)
class Paired:
    """The result of comparing two systems on the same records.

    Attributes:
        first_only: Records the first system got right and the second did not.
        second_only: Records the second system got right and the first did not.
        p_value: The exact two-sided p-value on the discordant pairs.
        odds_ratio: first_only over second_only, the effect size the study
            reports beside the test.
        odds_ratio_low: The lower end of the conditional interval, or None
            where the ratio itself is unbounded.
        odds_ratio_high: The upper end, or None on the same condition.
    """

    first_only: int
    second_only: int
    p_value: float
    odds_ratio: float | None
    odds_ratio_low: float | None = None
    odds_ratio_high: float | None = None


def exact_mcnemar(first_only: int, second_only: int) -> Paired:
    """Compare two systems on the records where they disagreed.

    The test is exact rather than asymptotic because several of these
    comparisons have few discordant pairs, where the chi-squared approximation
    is not trustworthy. Only the discordant pairs carry information: a record
    both systems answered the same way says nothing about which is better.

    Args:
        first_only: Records only the first system got right.
        second_only: Records only the second system got right.

    Returns:
        The counts, the exact two-sided p-value, and the odds ratio of the
        discordant pairs. The ratio is None when the second system was never
        alone correct, since the odds are then unbounded and reporting a large
        finite number would misstate what was measured.

    Raises:
        ValueError: If either count is negative.
    """
    if first_only < 0 or second_only < 0:
        raise ValueError(f"negative counts: {first_only}, {second_only}")
    discordant = first_only + second_only
    if discordant == 0:
        return Paired(0, 0, 1.0, None)
    p_value = float(binomtest(first_only, discordant, 0.5).pvalue)
    ratio = None if second_only == 0 else first_only / second_only
    # Conditioning on the discordant pairs turns the odds ratio into a
    # function of one proportion, so the interval the study already uses for a
    # proportion carries over: a Wilson interval on the share of discordant
    # pairs the first system won, mapped through p / (1 - p). Reporting the
    # ratio without it states an effect size with no uncertainty beside it,
    # which is what the reporting rule asks against.
    share = wilson(first_only, discordant)
    low = None if share.low >= 1.0 else share.low / (1.0 - share.low)
    high = None if share.high >= 1.0 else share.high / (1.0 - share.high)
    return Paired(
        first_only,
        second_only,
        p_value,
        ratio,
        None if low is None else round(low, 4),
        None if high is None else round(high, 4),
    )


@dataclass(frozen=True)
class ClusteredPaired:
    """A paired comparison whose uncertainty is read on the clustering unit.

    Attributes:
        difference: The first system's accuracy less the second's, on the
            records both answered.
        low: The lower end of the interval.
        high: The upper end of the interval.
        p_value: The share of resamples falling on the other side of zero,
            doubled, and never reported below one draw's worth.
        clusters: How many clusters the comparison had to resample.
        items: How many records both systems answered.
    """

    difference: float
    low: float
    high: float
    p_value: float
    clusters: int
    items: int


def clustered_paired_difference(
    first: dict[str, bool],
    second: dict[str, bool],
    cluster: dict[str, str],
    draws: int | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 42,
) -> ClusteredPaired:
    """Compare two systems with the cluster, not the record, as the unit.

    A record is not an independent draw here. The corpus is stratified by city
    and difficulty travels with the city, so treating 1,800 records as 1,800
    draws states an uncertainty the design did not buy. The cities are
    resampled with replacement instead, which is what the method says the
    inference does and what the exact McNemar test on pooled records cannot do.

    The test is not a replacement for the paired exact test on the discordant
    pairs: that one asks whether the two systems differ at all, and this one
    asks how much, with the uncertainty the clustering leaves. Both are
    reported.

    Args:
        first: Whether the first system answered each record correctly.
        second: Whether the second system answered each record correctly.
        cluster: The cluster each record belongs to.
        draws: How many resamples to take, or None to read the configuration.
        confidence: The nominal coverage.
        seed: The seed, fixed so that a reported interval is the same on every
            run.

    Returns:
        The difference, its interval, and the resampled two-sided p-value.

    Raises:
        ValueError: If no record was answered by both systems, or if a record
            carries no cluster.
    """
    shared = sorted(set(first) & set(second))
    if not shared:
        raise ValueError("the two systems answered no record in common")
    missing = [one for one in shared if one not in cluster]
    if missing:
        raise ValueError(f"{len(missing)} record(s) carry no cluster")

    if draws is None:
        draws = int(get("analysis.cluster_bootstrap_draws"))

    grouped: dict[str, list[float]] = defaultdict(list)
    for item in shared:
        grouped[cluster[item]].append(float(first[item]) - float(second[item]))
    units = sorted(grouped)
    sums = np.array([sum(grouped[one]) for one in units], dtype=float)
    sizes = np.array([len(grouped[one]) for one in units], dtype=float)
    difference = float(sums.sum() / sizes.sum())

    generator = np.random.default_rng(seed)
    picks = generator.integers(0, len(units), size=(draws, len(units)))
    resampled = sums[picks].sum(axis=1) / sizes[picks].sum(axis=1)

    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(resampled, [tail, 1.0 - tail])
    below = float(np.mean(resampled <= 0.0))
    above = float(np.mean(resampled >= 0.0))
    p_value = max(2.0 * min(below, above), 1.0 / draws)
    return ClusteredPaired(
        difference=round(difference, 4),
        low=round(float(low), 4),
        high=round(float(high), 4),
        p_value=min(1.0, round(p_value, 6)),
        clusters=len(units),
        items=len(shared),
    )


# ---------------------------------------------------------------------------
# Agreement across any number of readers
# ---------------------------------------------------------------------------

AgreementMeasure = Callable[[list[list[str]], tuple[str, ...]], float]


def _category_counts(
    ratings: list[list[str]], categories: tuple[str, ...]
) -> tuple[np.ndarray, int]:
    """Return an items-by-categories count matrix and the readers per item.

    Args:
        ratings: One list per item holding every reader's label.
        categories: Every label a reader may give.

    Returns:
        The count matrix and the number of readers each item carries.

    Raises:
        ValueError: If there are no items, if items carry different numbers of
            readings or fewer than two, or if a label is outside the categories.
    """
    if not ratings:
        raise ValueError("no items")
    readers = len(ratings[0])
    if readers < 2 or any(len(row) != readers for row in ratings):
        raise ValueError("every item needs the same number of readings, at least two")
    index = {category: position for position, category in enumerate(categories)}
    counts = np.zeros((len(ratings), len(categories)), dtype=float)
    for row, labels in enumerate(ratings):
        for label in labels:
            if label not in index:
                raise ValueError(f"label outside the categories: {label}")
            counts[row, index[label]] += 1.0
    return counts, readers


def _observed_agreement(counts: np.ndarray, readers: int) -> float:
    """Return the mean share of agreeing reader pairs per item."""
    agreeing = (np.sum(counts * counts, axis=1) - readers) / (readers * (readers - 1))
    return float(np.mean(agreeing))


def fleiss_kappa(ratings: list[list[str]], categories: tuple[str, ...]) -> float:
    """Return Fleiss' kappa for a fixed number of readers per item.

    With two readers it reduces to Scott's pi, whose chance term uses the rate
    pooled over the readers rather than each reader's own. That is the right
    chance model when the readers are interchangeable, which is what a gold
    standard taken by majority assumes of them.

    Args:
        ratings: One list per item holding every reader's label.
        categories: Every label a reader may give.

    Returns:
        Kappa; one when every item is unanimous.
    """
    counts, readers = _category_counts(ratings, categories)
    shares = counts.sum(axis=0) / counts.sum()
    expected = float(np.sum(shares * shares))
    observed = _observed_agreement(counts, readers)
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def gwet_ac1(ratings: list[list[str]], categories: tuple[str, ...]) -> float:
    """Return Gwet's AC1, which keeps its meaning when one label is rare.

    Kappa's chance term approaches one as a label becomes rare, so readers who
    agree on nearly every record can still score near zero. AC1 takes the
    chance term from the probability that a rating is given at random instead,
    which is why it is reported beside kappa for a class carried by about one
    record in eight.

    Args:
        ratings: One list per item holding every reader's label.
        categories: Every label a reader may give, at least two.

    Returns:
        AC1; one when every item is unanimous.
    """
    counts, readers = _category_counts(ratings, categories)
    shares = counts.sum(axis=0) / counts.sum()
    expected = float(np.sum(shares * (1.0 - shares)) / (len(categories) - 1))
    observed = _observed_agreement(counts, readers)
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def bootstrap_interval(
    ratings: list[list[str]],
    categories: tuple[str, ...],
    measure: AgreementMeasure,
    replicates: int,
    seed: int,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[float, float]:
    """Return a percentile bootstrap interval for an agreement measure.

    Items are resampled with replacement and every reader's label travels with
    its item, so the dependence between readers is kept.

    Args:
        ratings: One list per item holding every reader's label.
        categories: Every label a reader may give.
        measure: `fleiss_kappa` or `gwet_ac1`.
        replicates: How many resamples.
        seed: The generator seed.
        confidence: The nominal coverage.

    Returns:
        The lower and upper bound.
    """
    generator = np.random.default_rng(seed)
    size = len(ratings)
    values = [
        measure([ratings[i] for i in generator.integers(0, size, size)], categories)
        for _ in range(replicates)
    ]
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail])
    return float(low), float(high)


# ---------------------------------------------------------------------------
# What a paired comparison can separate at all
# ---------------------------------------------------------------------------


def min_discordant_for_significance(alpha: float) -> int:
    """Return the fewest discordant pairs an exact McNemar test can call significant.

    The most extreme split of k discordant pairs is k to none, whose exact
    two-sided p-value is 2 * 0.5 ** k. No smaller k reaches alpha however the
    pairs fall, so this is a floor on every comparison the set can support.

    Args:
        alpha: The significance threshold, in (0, 1).

    Returns:
        The smallest k with 2 * 0.5 ** k below alpha.

    Raises:
        ValueError: If alpha is outside (0, 1).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha outside (0, 1): {alpha}")
    k = 1
    while 2.0 * 0.5**k >= alpha:
        k += 1
    return k


def min_detectable_recall_difference(positives: int, alpha: float) -> float:
    """Return the smallest recall difference two systems can be separated by.

    Each discordant pair on the positives moves one system's recall by one over
    the positives, so the fewest significant discordant pairs sets the floor.

    Args:
        positives: The positives both systems were scored on.
        alpha: The significance threshold.

    Returns:
        The floor as a share of the positives; above one means no separation
        is possible at all.

    Raises:
        ValueError: If there are no positives.
    """
    if positives <= 0:
        raise ValueError("no positives")
    return min_discordant_for_significance(alpha) / positives
