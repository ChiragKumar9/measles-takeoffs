"""CRPS and weighted interval score of the predictive draws (outbreak_size.scoring)."""

import numpy as np
import pytest

from outbreak_size.scoring import (
    crps,
    crps_above_threshold,
    weighted_interval_score,
)


@pytest.mark.parametrize(
    "outcome, forecast, expected",
    [
        (12.0, 8.0, 2.0),  # 8 counts as 0, 12 as 2 cases above 10
        (300.0, 0.0, 290.0),  # a big miss is almost its full size
        (3.0, 0.0, 0.0),  # both below the threshold: no cost
        (0.0, 20.0, 10.0),  # a forecast of 20 is 10 cases above the threshold
        (10.0, 9.9, 0.0),  # no cliff at the threshold itself
    ],
)
def test_crps_above_a_threshold_of_a_point_forecast_is_the_miss_above_it(
    outcome, forecast, expected
):
    draws = np.full((20, 1), forecast)
    assert crps_above_threshold([outcome], draws, 10)[0] == pytest.approx(
        expected
    )


def test_crps_above_a_threshold_of_zero_is_the_plain_crps():
    rng = np.random.default_rng(3)
    draws = rng.poisson(6, size=(300, 4)).astype(float)
    outcome = rng.poisson(6, size=4)
    assert crps_above_threshold(outcome, draws, 0) == pytest.approx(
        crps(outcome, draws)
    )


def test_crps_above_a_threshold_equals_the_crps_with_values_set_to_the_threshold():
    # the same numbers whether values below the threshold become 0 (and the
    # excess is counted) or become the threshold: CRPS depends on differences
    rng = np.random.default_rng(4)
    draws = rng.negative_binomial(1, 1 / 16, size=(500, 5)).astype(float)
    outcome = np.array([0.0, 3.0, 12.0, 50.0, 300.0])
    assert crps_above_threshold(outcome, draws, 10) == pytest.approx(
        crps(np.maximum(outcome, 10), np.maximum(draws, 10))
    )


def test_crps_above_a_threshold_is_never_more_than_the_plain_crps():
    rng = np.random.default_rng(5)
    draws = rng.negative_binomial(1, 1 / 16, size=(500, 8)).astype(float)
    outcome = rng.negative_binomial(1, 1 / 16, size=8).astype(float)
    assert (
        crps_above_threshold(outcome, draws, 10) <= crps(outcome, draws) + 1e-9
    ).all()


def test_crps_of_two_draws_matches_the_hand_calculation():
    # draws 0 and 2, observed 1: E|X-y| = 1, E|X-X'| = 1, so CRPS = 1 - 0.5
    assert crps([1.0], np.array([[0.0], [2.0]]))[0] == pytest.approx(0.5)


def test_crps_of_a_point_mass_is_the_absolute_error():
    draws = np.full((30, 2), 4.0)
    assert crps([4.0, 7.0], draws) == pytest.approx([0.0, 3.0])


def test_crps_is_per_location_and_not_negative():
    rng = np.random.default_rng(0)
    draws = rng.poisson(5, size=(500, 6)).astype(float)
    scores = crps(rng.poisson(5, size=6), draws)
    assert scores.shape == (6,)
    assert (scores >= 0).all()


def test_wis_of_a_point_mass_is_the_absolute_error():
    draws = np.full((100, 2), 3.0)
    assert weighted_interval_score([3.0, 8.0], draws) == pytest.approx(
        [0.0, 5.0]
    )


def test_wis_is_lower_for_draws_centered_on_the_truth():
    rng = np.random.default_rng(2)
    centered = rng.normal(10, 2, size=(2000, 1))
    shifted = rng.normal(20, 2, size=(2000, 1))
    assert (
        weighted_interval_score([10.0], centered)[0]
        < weighted_interval_score([10.0], shifted)[0]
    )
