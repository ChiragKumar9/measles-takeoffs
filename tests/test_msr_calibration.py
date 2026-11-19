"""The pieces of the calibration check (outbreak_size.msr_calibration)."""

import numpy as np
import pytest

from outbreak_size.msr_calibration import (
    between_county_share,
    calibration_verdict,
    independent_covariate,
    logistic_null_outcome,
    match_noise_scale,
    morans_i,
    rejection_rate,
    residual_smoothness,
    smooth_field,
    smoothness_matching,
)


def grid_adjacency(side=8):
    n = side * side
    adjacency = np.zeros((n, n))
    for i in range(side):
        for j in range(side):
            k = i * side + j
            if i + 1 < side:
                adjacency[k, k + side] = adjacency[k + side, k] = 1
            if j + 1 < side:
                adjacency[k, k + 1] = adjacency[k + 1, k] = 1
    return adjacency


def test_morans_i_is_high_for_a_smooth_field_and_negative_for_a_checkerboard():
    adjacency = grid_adjacency(6)
    smooth = np.repeat(np.arange(6.0), 6)  # rows of a grid, one value per row
    checkerboard = np.array([(i + j) % 2 for i in range(6) for j in range(6)])
    assert morans_i(smooth, adjacency) > 0.5
    assert morans_i(checkerboard.astype(float), adjacency) == pytest.approx(
        -1.0
    )


def test_a_smooth_field_is_standardized():
    adjacency = grid_adjacency()
    field = smooth_field(adjacency, 0.8, np.random.default_rng(0))
    assert field.mean() == pytest.approx(0.0, abs=1e-12)
    assert field.std() == pytest.approx(1.0)


def test_smoother_fields_have_a_higher_morans_i():
    adjacency = grid_adjacency()

    def mean_i(smoothness):
        rng = np.random.default_rng(1)
        return np.mean(
            [
                morans_i(smooth_field(adjacency, smoothness, rng), adjacency)
                for _ in range(30)
            ]
        )

    assert mean_i(0.0) < 0.1
    assert mean_i(0.5) < mean_i(0.9)
    assert mean_i(0.9) > 0.4


def test_the_matched_smoothness_gives_the_target_morans_i():
    adjacency = grid_adjacency()
    smoothness = smoothness_matching(adjacency, 0.3)
    rng = np.random.default_rng(0)  # the matching's own fields
    got = np.mean(
        [
            morans_i(smooth_field(adjacency, smoothness, rng), adjacency)
            for _ in range(20)
        ]
    )
    assert got == pytest.approx(0.3, abs=0.01)
    assert 0.0 < smoothness < 0.99


def test_a_target_out_of_reach_is_refused():
    with pytest.raises(ValueError, match="out of the reach"):
        smoothness_matching(grid_adjacency(), 0.99)


def test_the_observations_of_a_county_share_its_field_value():
    adjacency = grid_adjacency(4)
    counties = np.repeat(np.arange(16), 3)
    x = independent_covariate(
        adjacency, counties, 0.5, 1.0, np.random.default_rng(2)
    )
    by_county = x.reshape(16, 3)
    assert (by_county == by_county[:, :1]).all()  # share 1: no noise


def test_with_no_between_county_share_the_covariate_is_noise():
    adjacency = grid_adjacency(4)
    counties = np.repeat(np.arange(16), 3)
    x = independent_covariate(
        adjacency, counties, 0.9, 0.0, np.random.default_rng(2)
    )
    assert between_county_share(x, counties) < 0.3


def test_the_between_county_share_is_recovered():
    adjacency = grid_adjacency(10)
    counties = np.repeat(np.arange(100), 4)
    shares = [
        between_county_share(
            independent_covariate(
                adjacency, counties, 0.0, 0.75, np.random.default_rng(seed)
            ),
            counties,
        )
        for seed in range(20)
    ]
    assert np.mean(shares) == pytest.approx(0.75, abs=0.06)


def test_the_between_county_share_is_one_without_repeated_observations():
    assert between_county_share(np.arange(5.0), np.arange(5)) == 1.0


def test_between_county_share_by_hand():
    values = np.array([0.0, 2.0, 10.0, 14.0])
    counties = np.array([0, 0, 1, 1])
    # county means 1 and 12 (variance 30.25); within variances 2 and 8 (mean 5)
    assert between_county_share(values, counties) == pytest.approx(
        30.25 / 35.25
    )


def test_the_rejection_rate_and_its_standard_error_under_a_calibrated_test():
    p_values = np.array([0.01, 0.2, 0.04, 0.5, 0.9, 0.05, 0.3, 0.6, 0.7, 0.8])
    rate, se = rejection_rate(p_values, 0.05)
    assert rate == pytest.approx(0.2)  # 0.05 itself is not below the level
    assert se == pytest.approx(np.sqrt(0.05 * 0.95 / 10))


def test_a_smooth_field_by_hand_on_a_path_of_three_counties():
    # the neighbours' weights are row-standardized: the middle county, with
    # two neighbours, weighs each by a half
    adjacency = np.array([[0.0, 1, 0], [1, 0, 1], [0, 1, 0]])
    weights = np.array([[0.0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]])
    draws = np.random.default_rng(0).standard_normal(3)
    expected = np.linalg.solve(np.eye(3) - 0.5 * weights, draws)
    expected = (expected - expected.mean()) / expected.std()
    got = smooth_field(adjacency, 0.5, np.random.default_rng(0))
    assert got == pytest.approx(expected)


@pytest.mark.parametrize(
    "rate, expected",
    [
        (0.05, "calibrated"),
        (0.065, "calibrated"),  # within two standard errors (2 * 0.011)
        (0.08, "over-rejects"),
        (0.02, "conservative"),
    ],
)
def test_calibration_verdict(rate, expected):
    assert calibration_verdict(rate, 0.05, 0.011) == expected


def test_a_smooth_residual_gets_a_smooth_field():
    adjacency = grid_adjacency()
    n = adjacency.shape[0]
    counties = np.arange(n)
    rng = np.random.default_rng(0)
    smooth = smooth_field(adjacency, 0.9, rng)
    rough = rng.standard_normal(n)
    assert residual_smoothness(adjacency, smooth, counties) > 0.6
    assert residual_smoothness(adjacency, rough, counties) < 0.3


def test_a_residual_beyond_the_reach_of_the_fields_is_clamped():
    adjacency = grid_adjacency()
    counties = np.arange(adjacency.shape[0])
    checkerboard = np.array(
        [(-1.0) ** (i + j) for i in range(8) for j in range(8)]
    )
    assert residual_smoothness(adjacency, checkerboard, counties) == 0.0
    ramp = np.repeat(np.arange(8.0), 8)  # smoother than a random field
    assert residual_smoothness(adjacency, ramp, counties) == 0.99


def test_the_logistic_null_outcome_follows_the_logit_and_shares_the_noise_within_a_county():
    adjacency = grid_adjacency(6)
    n_counties = adjacency.shape[0]
    counties = np.repeat(np.arange(n_counties), 200)
    # without noise the share of ones is the probability of the logit
    logit = np.full(len(counties), np.log(0.3 / 0.7))
    out = logistic_null_outcome(
        adjacency, counties, logit, 0.0, 0.5, np.random.default_rng(2)
    )
    assert set(np.unique(out)) <= {0.0, 1.0}
    assert out.mean() == pytest.approx(0.3, abs=0.02)
    # with noise the probability moves with the county's field: counties differ in their share of ones, and the share
    # of a county follows the field that all its observations share
    noisy = logistic_null_outcome(
        adjacency, counties, logit, 2.0, 0.5, np.random.default_rng(2)
    )
    shares = noisy.reshape(n_counties, 200).mean(axis=1)
    field = smooth_field(adjacency, 0.5, np.random.default_rng(2))
    assert np.corrcoef(shares, field)[0, 1] > 0.9
    assert shares.std() > 0.1


def test_the_noise_scale_is_recovered_from_the_real_outcomes_county_variance():
    adjacency = grid_adjacency(7)
    n_counties = adjacency.shape[0]
    counties = np.repeat(np.arange(n_counties), 20)
    logit = np.full(len(counties), -0.4)
    for true_scale in (0.0, 1.0, 2.0):
        y = logistic_null_outcome(
            adjacency,
            counties,
            logit,
            true_scale,
            0.5,
            np.random.default_rng(11),
        )
        found = match_noise_scale(adjacency, counties, logit, y, 0.5)
        assert abs(found - true_scale) <= 0.5
