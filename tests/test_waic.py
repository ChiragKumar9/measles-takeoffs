"""WAIC of the Poisson and negative binomial models (outbreak_size.waic)."""

import math

import numpy as np
import pytest

from outbreak_size.waic import (
    log_likelihood,
    waic_point_estimate,
    waic_posterior,
)


def test_point_estimate_is_minus_two_times_loglik_minus_k():
    # mean 1 everywhere: log p(y) = -1 - log(y!)
    loglik = (-1 - 0) + (-1 - 0) + (-1 - math.log(2))
    waic = waic_point_estimate([0, 1, 2], [1.0, 1.0, 1.0], k=2)
    assert waic == pytest.approx(-2 * (loglik - 2))


def test_point_estimate_penalizes_parameters():
    cases, mu = [0, 1, 2], [1.0, 1.0, 1.0]
    assert waic_point_estimate(cases, mu, k=5) > waic_point_estimate(
        cases, mu, k=2
    )


def test_a_mean_of_zero_for_zero_cases_costs_nothing():
    # states left out of the mechanistic fit are predicted at a mean of 0
    assert waic_point_estimate([0, 0], [0.0, 0.0], k=0) == pytest.approx(0.0)


def test_posterior_waic_without_posterior_spread_is_minus_two_loglik():
    cases = np.array([0, 1, 2])
    draws = np.ones((50, 3))  # every draw has the same mean
    waic, p_waic = waic_posterior(cases, draws)
    loglik = (-1) + (-1) + (-1 - math.log(2))
    assert p_waic == pytest.approx(0.0)
    assert waic == pytest.approx(-2 * loglik)


def test_posterior_spread_adds_effective_parameters():
    rng = np.random.default_rng(0)
    cases = np.array([3, 5, 2, 8])
    spread = rng.gamma(20, 0.25, size=(400, 4))
    _, p_waic = waic_posterior(cases, spread)
    assert p_waic > 0


def test_a_better_fitting_model_has_a_lower_waic():
    rng = np.random.default_rng(1)
    cases = np.array([1, 10, 3, 20])
    good = np.tile(cases + 0.5, (200, 1)) * rng.uniform(0.95, 1.05, (200, 4))
    bad = np.full((200, 4), 40.0)
    assert waic_posterior(cases, good)[0] < waic_posterior(cases, bad)[0]


# --- the negative binomial -----------------------------------------------------


def test_the_negative_binomial_log_likelihood_is_worked_out_by_hand():
    # a concentration of 1 and a mean of 2 make a geometric distribution with
    # success probability 1/3: P(k) = (1/3) * (2/3)**k
    log_lik = log_likelihood([0, 2], np.full((1, 2), 2.0), [1.0])
    assert log_lik[0] == pytest.approx([math.log(1 / 3), math.log(4 / 27)])


def test_the_negative_binomial_matches_the_models_own_log_probability():
    # the model's likelihood is numpyro's NegativeBinomial2(mean, concentration)
    import numpyro.distributions as dist

    cases = np.array([0, 1, 5, 40])
    mu = np.array([[0.5, 2.0, 4.0, 30.0], [1.0, 1.5, 6.0, 55.0]])
    concentration = np.array([0.8, 3.0])
    expected = np.stack(
        [
            np.asarray(
                dist.NegativeBinomial2(mu[d], concentration[d]).log_prob(cases)
            )
            for d in range(2)
        ]
    )
    # numpyro computes in float32 here (the tests do not enable x64)
    assert log_likelihood(cases, mu, concentration) == pytest.approx(
        expected, rel=1e-5
    )


def test_a_huge_concentration_is_the_poisson():
    cases = np.array([0, 3, 7])
    mu = np.full((1, 3), 4.0)
    assert log_likelihood(cases, mu, [1e9]) == pytest.approx(
        log_likelihood(cases, mu), abs=1e-5
    )


def test_the_negative_binomial_waic_without_posterior_spread_is_minus_two_loglik():
    cases = np.array([0, 2])
    waic, p_waic = waic_posterior(
        cases, np.full((50, 2), 2.0), np.full(50, 1.0)
    )
    assert p_waic == pytest.approx(0.0)
    assert waic == pytest.approx(-2 * (math.log(1 / 3) + math.log(4 / 27)))


def test_overdispersed_counts_have_a_lower_waic_under_the_negative_binomial():
    # counts far more spread out than a Poisson allows (a mean of 20 with
    # values from 0 to 90) are explained better with a small concentration
    cases = np.array([0, 0, 1, 90, 2, 0, 60, 5])
    mu = np.full((100, 8), 20.0)
    poisson_waic = waic_posterior(cases, mu)[0]
    nb_waic = waic_posterior(cases, mu, np.full(100, 0.5))[0]
    assert nb_waic < poisson_waic
