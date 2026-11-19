"""The risk percentile of units (outbreak_size.ranking)."""

import numpy as np
import pytest
from scipy import stats

from outbreak_size.ranking import percentile_of_scores, risk_percentile


def test_the_lowest_score_gets_0_and_the_highest_100():
    assert percentile_of_scores([1.0, 9.0, 5.0]) == pytest.approx(
        [0.0, 100.0, 50.0]
    )


def test_tied_scores_share_the_average_rank():
    # ranks 1.5, 1.5, 3 shifted to 0.5, 0.5, 2 and scaled by the highest
    assert percentile_of_scores([2.0, 2.0, 7.0]) == pytest.approx(
        [25.0, 25.0, 100.0]
    )


def test_a_unit_that_is_always_lowest_gets_0_and_always_highest_gets_100():
    draws = np.tile([1.0, 5.0, 9.0], (20, 1))
    assert risk_percentile(draws) == pytest.approx([0.0, 50.0, 100.0])


def test_the_draws_are_averaged_before_the_units_are_ranked():
    # unit 0 is lowest in one draw and highest in the other, and its mean (2)
    # is below that of unit 1 (2.25): ranking each draw and averaging the
    # percentiles would give 50 and 50
    draws = np.array([[1.0, 2.0], [3.0, 2.5]])
    assert risk_percentile(draws) == pytest.approx([0.0, 100.0])
    assert risk_percentile(draws[:, ::-1]) == pytest.approx([100.0, 0.0])


def test_percentiles_are_between_0_and_100_and_follow_the_means():
    rng = np.random.default_rng(0)
    draws = rng.gamma(2.0, 1.0, size=(300, 6)) * np.arange(1, 7)
    ranks = risk_percentile(draws)
    assert ((0 <= ranks) & (ranks <= 100)).all()
    assert (np.diff(ranks) > 0).all()  # a bigger mean ranks higher
    expected = stats.rankdata(draws.mean(axis=0)) - 1
    assert ranks == pytest.approx(expected / expected.max() * 100)


def test_the_percentiles_of_many_places_spread_evenly_over_0_to_100():
    # a percentile of x means the top (100 - x)% of the places: the ranks of
    # distinct means are equally spaced (a mean of per-draw ranks is not)
    rng = np.random.default_rng(1)
    draws = rng.gamma(2.0, 1.0, size=(50, 101))
    ranks = np.sort(risk_percentile(draws))
    assert ranks == pytest.approx(np.arange(101.0))
