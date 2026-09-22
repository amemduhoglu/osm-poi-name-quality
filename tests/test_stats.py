"""Tests for the detection summaries and the paired comparison.

The properties checked here are the ones the reporting rules depend on: that a
measure ignores how the classes are mixed where it claims to, that a flag rate
is reported beside recall, and that a test with no discordant pair says nothing
rather than something.
"""

from __future__ import annotations

import numpy as np
import pytest

from poi_audit import stats
from poi_audit.stats import Confusion, exact_mcnemar, wilson


def test_a_perfect_separation_reaches_one() -> None:
    perfect = Confusion(
        true_positive=40, false_negative=0, true_negative=60, false_positive=0
    )
    assert perfect.youdens_j == pytest.approx(1.0)
    assert perfect.balanced_accuracy == pytest.approx(1.0)


def test_flagging_everything_scores_at_chance() -> None:
    """The gate this measure exists for: perfect recall, nothing decided."""
    always = Confusion(
        true_positive=40, false_negative=0, true_negative=0, false_positive=60
    )
    assert always.sensitivity().estimate == pytest.approx(1.0)
    assert always.youdens_j == pytest.approx(0.0)
    assert always.flag_rate().estimate == pytest.approx(1.0)


def test_balanced_accuracy_does_not_move_with_the_corrupted_share() -> None:
    """Why comparisons across groups of different composition are made on it."""
    half = Confusion(
        true_positive=45, false_negative=5, true_negative=40, false_positive=10
    )
    tenth = Confusion(
        true_positive=9, false_negative=1, true_negative=360, false_positive=90
    )
    assert half.balanced_accuracy == pytest.approx(tenth.balanced_accuracy)
    assert half.youdens_j == pytest.approx(tenth.youdens_j)


def test_a_class_with_no_members_reports_zero_rather_than_raising() -> None:
    one_sided = Confusion(
        true_positive=10, false_negative=2, true_negative=0, false_positive=0
    )
    assert one_sided.balanced_accuracy == 0.0
    assert one_sided.youdens_j == 0.0


def test_youdens_j_is_twice_the_balanced_accuracy_minus_one() -> None:
    counts = Confusion(
        true_positive=30, false_negative=20, true_negative=45, false_positive=5
    )
    assert counts.youdens_j == pytest.approx(2 * counts.balanced_accuracy - 1)


def test_a_concordant_comparison_carries_no_evidence() -> None:
    result = exact_mcnemar(0, 0)
    assert result.p_value == 1.0
    assert result.odds_ratio is None


def test_an_unbounded_odds_ratio_is_reported_as_absent() -> None:
    """A large finite number here would misstate what was measured."""
    result = exact_mcnemar(12, 0)
    assert result.odds_ratio is None
    assert result.p_value < 0.001


def test_a_symmetric_disagreement_is_not_evidence_of_a_difference() -> None:
    result = exact_mcnemar(15, 15)
    assert result.p_value == pytest.approx(1.0)
    assert result.odds_ratio == pytest.approx(1.0)


def test_negative_counts_are_refused() -> None:
    with pytest.raises(ValueError):
        exact_mcnemar(-1, 3)


def test_a_proportion_at_zero_keeps_its_bounds_inside_the_unit_interval() -> None:
    at_zero = wilson(0, 250)
    assert at_zero.low == 0.0
    assert 0.0 < at_zero.high < 0.05


def test_a_clustered_difference_is_zero_when_two_systems_agree() -> None:
    """Two systems answering alike differ by nothing, whatever the clusters."""
    answers = {f"i{n}": n % 2 == 0 for n in range(40)}
    cities = {f"i{n}": f"city{n % 4}" for n in range(40)}
    result = stats.clustered_paired_difference(answers, answers, cities, draws=200)
    assert result.difference == 0.0
    assert result.low <= 0.0 <= result.high


def test_a_clustered_difference_reads_the_direction_and_the_size() -> None:
    """A system right everywhere leads one wrong everywhere by the whole rate."""
    right = {f"i{n}": True for n in range(40)}
    wrong = {f"i{n}": False for n in range(40)}
    cities = {f"i{n}": f"city{n % 4}" for n in range(40)}
    result = stats.clustered_paired_difference(right, wrong, cities, draws=200)
    assert result.difference == 1.0
    assert result.low == 1.0


def test_a_difference_inside_one_city_is_less_certain_than_one_spread_over_all() -> (
    None
):
    """This is what clustering is for: twelve cities are not 1,800 draws.

    The same number of records answered differently gives a much wider interval
    when the difference sits in a single city, because the city is the unit
    that could have come out otherwise.
    """
    cities = {f"i{n}": f"city{n // 10}" for n in range(40)}
    second = {f"i{n}": False for n in range(40)}

    concentrated = {f"i{n}": n < 10 for n in range(40)}
    spread = {f"i{n}": n % 4 == 0 for n in range(40)}
    assert sum(concentrated.values()) == sum(spread.values())

    one_city = stats.clustered_paired_difference(
        concentrated, second, cities, draws=2000
    )
    every_city = stats.clustered_paired_difference(spread, second, cities, draws=2000)
    assert one_city.difference == every_city.difference
    assert (one_city.high - one_city.low) > (every_city.high - every_city.low)


def test_the_clustered_comparison_is_the_same_on_every_run() -> None:
    """A resampled interval that moves between runs is not a reported number."""
    first = {f"i{n}": n % 3 != 0 for n in range(40)}
    second = {f"i{n}": n % 5 != 0 for n in range(40)}
    cities = {f"i{n}": f"city{n % 4}" for n in range(40)}
    one = stats.clustered_paired_difference(first, second, cities, draws=500)
    two = stats.clustered_paired_difference(first, second, cities, draws=500)
    assert (one.low, one.high, one.p_value) == (two.low, two.high, two.p_value)


def test_the_summed_interval_keeps_its_nominal_coverage() -> None:
    """The balanced-accuracy interval must cover at the rate it claims.

    Sensitivity and specificity are measured on disjoint records, so their
    Wilson intervals are combined by squaring and adding rather than by a
    normal approximation the counts at the ends of the range do not support.
    That combination is not a standard interval, and every gate in this study
    is decided on it, so its coverage is measured rather than assumed. The
    cases below are the operating points the article reports on: balanced
    halves of the injected class, the unbalanced natural set, and a model
    separating almost perfectly.
    """
    generator = np.random.default_rng(20260809)
    cases = (
        (300, 300, 0.80, 0.80),
        (300, 300, 0.60, 0.60),
        (300, 300, 0.99, 0.99),
        (29, 195, 0.70, 0.50),
        (21, 185, 0.70, 0.50),
        (29, 195, 0.50, 0.50),
    )
    for positives, negatives, sensitivity, specificity in cases:
        truth = (sensitivity + specificity) / 2
        covered = 0
        draws = 4000
        for _ in range(draws):
            true_positive = int(generator.binomial(positives, sensitivity))
            true_negative = int(generator.binomial(negatives, specificity))
            confusion = stats.Confusion(
                true_positive=true_positive,
                false_negative=positives - true_positive,
                true_negative=true_negative,
                false_positive=negatives - true_negative,
            )
            low, high = confusion.balanced_accuracy_interval()
            covered += low <= truth <= high
        # A nominal 95 per cent interval measured on 4,000 draws carries a
        # standard error of about 0.0035, so the band below is roughly four of
        # them wide: it catches a method that undercovers, not the noise.
        assert 0.935 <= covered / draws <= 0.965, (
            f"{positives}/{negatives} at {sensitivity}/{specificity}: "
            f"{covered / draws}"
        )


def test_the_odds_ratio_carries_an_interval_containing_it() -> None:
    """The effect size reported beside the test must carry its uncertainty."""
    result = stats.exact_mcnemar(79, 31)
    assert result.odds_ratio is not None
    assert result.odds_ratio_low is not None
    assert result.odds_ratio_high is not None
    assert result.odds_ratio_low < result.odds_ratio < result.odds_ratio_high
    # A discordant split this lopsided is a real difference, so the interval
    # clears the ratio of one that would mean the two systems are the same.
    assert result.odds_ratio_low > 1.0


def test_an_unbounded_odds_ratio_carries_no_upper_bound() -> None:
    """A ratio with nothing in its denominator reports no upper end."""
    result = stats.exact_mcnemar(12, 0)
    assert result.odds_ratio is None
    assert result.odds_ratio_high is None


# ---------------------------------------------------------------------------
# Agreement across any number of readers, and the detectability floor
# ---------------------------------------------------------------------------

TWO_READER_TABLE = (
    [["wrong", "wrong"]] * 40
    + [["wrong", "belongs"]] * 10
    + [["belongs", "wrong"]] * 5
    + [["belongs", "belongs"]] * 45
)


def test_fleiss_kappa_equals_scotts_pi_for_two_readers() -> None:
    """Pooled share wrong is 95/200, so chance agreement is 0.475^2 + 0.525^2."""
    expected = 0.475**2 + 0.525**2
    assert stats.fleiss_kappa(TWO_READER_TABLE, ("wrong", "belongs")) == pytest.approx(
        (0.85 - expected) / (1 - expected)
    )


def test_gwet_ac1_on_the_same_table() -> None:
    expected = 2 * 0.475 * 0.525
    assert stats.gwet_ac1(TWO_READER_TABLE, ("wrong", "belongs")) == pytest.approx(
        (0.85 - expected) / (1 - expected)
    )


def test_ac1_stays_high_where_kappa_collapses_at_low_prevalence() -> None:
    """The reason AC1 is reported beside kappa for a class near one in eight."""
    ratings = (
        [["wrong", "wrong"]] * 2
        + [["wrong", "belongs"]] * 4
        + [["belongs", "wrong"]] * 4
        + [["belongs", "belongs"]] * 90
    )
    categories = ("wrong", "belongs")
    assert stats.gwet_ac1(ratings, categories) > 0.85
    assert stats.fleiss_kappa(ratings, categories) < 0.30


def test_three_unanimous_readers_agree_perfectly() -> None:
    ratings = [["wrong"] * 3] * 10 + [["belongs"] * 3] * 30
    categories = ("wrong", "belongs", "cannot_say")
    assert stats.fleiss_kappa(ratings, categories) == pytest.approx(1.0)
    assert stats.gwet_ac1(ratings, categories) == pytest.approx(1.0)


def test_unequal_reader_counts_are_refused() -> None:
    with pytest.raises(ValueError):
        stats.fleiss_kappa([["wrong", "wrong"], ["wrong"]], ("wrong", "belongs"))


def test_a_label_outside_the_categories_is_refused() -> None:
    with pytest.raises(ValueError):
        stats.gwet_ac1([["wrong", "maybe"]], ("wrong", "belongs"))


def test_eleven_to_nothing_is_the_smallest_split_below_one_in_a_thousand() -> None:
    assert stats.min_discordant_for_significance(0.001) == 11
    assert stats.min_discordant_for_significance(0.05) == 6


def test_the_floor_agrees_with_the_exact_test() -> None:
    k = stats.min_discordant_for_significance(0.001)
    assert exact_mcnemar(k, 0).p_value < 0.001
    assert exact_mcnemar(k - 1, 0).p_value >= 0.001


def test_detectable_difference_shrinks_with_positives() -> None:
    assert stats.min_detectable_recall_difference(29, 0.001) == pytest.approx(11 / 29)
    assert stats.min_detectable_recall_difference(120, 0.001) == pytest.approx(11 / 120)


def test_bootstrap_interval_contains_the_estimate_and_is_reproducible() -> None:
    ratings = (
        [["wrong", "wrong", "wrong"]] * 20
        + [["wrong", "belongs", "belongs"]] * 10
        + [["belongs", "belongs", "belongs"]] * 70
    )
    categories = ("wrong", "belongs")
    first = stats.bootstrap_interval(
        ratings, categories, stats.fleiss_kappa, replicates=300, seed=42
    )
    again = stats.bootstrap_interval(
        ratings, categories, stats.fleiss_kappa, replicates=300, seed=42
    )
    assert first == again
    assert first[0] <= stats.fleiss_kappa(ratings, categories) <= first[1]
