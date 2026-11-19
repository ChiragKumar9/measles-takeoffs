"""Does the hierarchical Poisson model recover the coefficients that made the
data it is fitted to? (outbreak_size.poisson). Case counts are simulated from
the model's own likelihood with known coefficients and a fixed seed, and the
posterior of a short NUTS run is compared with the truth."""

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest
from jax.random import PRNGKey
from numpyro.infer import MCMC, NUTS

from outbreak_size.poisson import PartialPooledPoisson

BASELINE = -6.0  # log cases per person when every covariate is 0
EXPOSURE = 2000.0  # people per row, so the average row has about 5 cases


def simulate_cases(linear_predictor, seed):
    rng = np.random.default_rng(seed)
    return rng.poisson(EXPOSURE * np.exp(linear_predictor))


def fit(model, cases, seed=0, num_warmup=300, num_samples=300):
    mcmc = MCMC(
        NUTS(model.fit),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(
        PRNGKey(seed),
        is_observed=jnp.ones(len(cases)),
        cases=jnp.asarray(cases),
    )
    return mcmc.get_samples()


def no_neighbors(n):
    return np.zeros((n, n))


def test_the_model_recovers_the_coefficients_it_generated_the_data_with():
    """One pooled-free coefficient on the variable of interest (the "iv") and
    one ordinary covariate."""
    n = 1500
    rng = np.random.default_rng(1)
    iv = rng.normal(size=(n, 1))
    covariate = rng.normal(size=(n, 1))
    true_iv, true_covariate = 0.5, -0.4
    cases = simulate_cases(
        BASELINE + true_iv * iv[:, 0] + true_covariate * covariate[:, 0],
        seed=2,
    )
    model = PartialPooledPoisson(
        iv=jnp.asarray(iv),
        pool=False,
        regression_covariates=jnp.asarray(covariate),
        exposure=jnp.full(n, EXPOSURE),
        groups=pl.Series(["a"] * n),
        spatial_units=pl.Series(np.arange(n)),
        adjacency_matrix=no_neighbors(n),
        group_fixed_effect=False,
        spatial=False,
    )
    samples = fit(model, cases)
    # each estimate is within 0.08 of the truth (their standard errors are
    # about 0.01 to 0.02 with this much data)
    assert float(samples["baseline_rate"].mean()) == pytest.approx(
        BASELINE, abs=0.08
    )
    assert float(samples["iv_global"][:, 0].mean()) == pytest.approx(
        true_iv, abs=0.08
    )
    assert float(samples["regression_betas"][:, 0].mean()) == pytest.approx(
        true_covariate, abs=0.08
    )


def test_pooling_recovers_each_groups_own_coefficient():
    """Three states whose true coefficients on the variable of interest are
    0.2, 0.5 and 0.8: the pooled model's coefficient of each state (the
    global value plus the state's offset) comes out near its own."""
    per_group = 1200
    true_effects = np.array([0.2, 0.5, 0.8])
    groups = np.repeat(["a", "b", "c"], per_group)
    n = len(groups)
    iv = np.random.default_rng(3).normal(size=(n, 1))
    effect_of_row = np.repeat(true_effects, per_group)
    cases = simulate_cases(BASELINE + effect_of_row * iv[:, 0], seed=4)
    model = PartialPooledPoisson(
        iv=jnp.asarray(iv),
        pool=True,
        regression_covariates=jnp.zeros((n, 1)),
        exposure=jnp.full(n, EXPOSURE),
        groups=pl.Series(groups),
        spatial_units=pl.Series(np.arange(n)),
        adjacency_matrix=no_neighbors(n),
        group_fixed_effect=False,
        spatial=False,
    )
    samples = fit(model, cases)
    offsets = samples["iv_group_offsets"].mean(axis=0)[:, 0]  # [group]
    sigma = samples["iv_group_variabilities"].mean(axis=0)[0]
    estimates = float(samples["iv_global"][:, 0].mean()) + sigma * offsets
    assert np.asarray(estimates) == pytest.approx(true_effects, abs=0.1)
    # the groups stay in the order they first appear
    assert estimates[0] < estimates[1] < estimates[2]


def test_the_spatial_model_still_recovers_the_coefficient_without_spatial_structure():
    """Counties on a 6 x 6 grid, each seen 15 times, with no spatial effect in
    the data: the CAR terms must not move the coefficient away from the
    truth."""
    side, repeats = 6, 15
    n_units = side * side
    adjacency = np.zeros((n_units, n_units))
    for i in range(side):
        for j in range(side):
            k = i * side + j
            if i + 1 < side:
                adjacency[k, k + side] = adjacency[k + side, k] = 1
            if j + 1 < side:
                adjacency[k, k + 1] = adjacency[k + 1, k] = 1
    units = np.repeat(np.arange(n_units), repeats)
    n = len(units)
    iv = np.random.default_rng(5).normal(size=(n, 1))
    true_iv = 0.5
    cases = simulate_cases(BASELINE + true_iv * iv[:, 0], seed=6)
    model = PartialPooledPoisson(
        iv=jnp.asarray(iv),
        pool=False,
        regression_covariates=jnp.zeros((n, 1)),
        exposure=jnp.full(n, EXPOSURE),
        groups=pl.Series(["a"] * n),
        spatial_units=pl.Series(units),
        adjacency_matrix=adjacency,
        group_fixed_effect=False,
        spatial=True,
    )
    samples = fit(model, cases, num_warmup=200, num_samples=200)
    assert float(samples["iv_global"][:, 0].mean()) == pytest.approx(
        true_iv, abs=0.1
    )
    # one spatial effect per county
    assert samples["spatial_random_effects"].shape[1] == n_units


def test_the_model_recovers_the_population_exponent_and_the_baseline():
    """Counties of very different sizes whose expected cases scale as
    population ** 0.3: the posterior of the exponent comes out near 0.3 (a
    model that fixed it at 1 could not), and the baseline is the log rate of a
    county of the median population."""
    n = 1500
    rng = np.random.default_rng(7)
    population = np.exp(rng.normal(np.log(5000.0), 1.0, size=n))
    reference = np.median(population)
    true_exponent, true_baseline = 0.3, -5.0
    mean = (
        reference
        * np.exp(true_baseline)
        * (population / reference) ** true_exponent
    )
    cases = np.random.default_rng(8).poisson(mean)
    model = PartialPooledPoisson(
        iv=jnp.zeros((n, 1)),
        pool=False,
        regression_covariates=jnp.zeros((n, 1)),
        exposure=jnp.asarray(population),
        groups=pl.Series(["a"] * n),
        spatial_units=pl.Series(np.arange(n)),
        adjacency_matrix=no_neighbors(n),
        group_fixed_effect=False,
        spatial=False,
    )
    samples = fit(model, cases)
    assert float(samples["pop_exponent"].mean()) == pytest.approx(
        true_exponent, abs=0.08
    )
    assert float(samples["baseline_rate"].mean()) == pytest.approx(
        true_baseline, abs=0.1
    )


def test_the_negative_binomial_model_recovers_the_concentration_and_the_coefficient():
    """Counts drawn from a negative binomial with concentration 1.5 around a
    mean that depends on one covariate: the posterior of the concentration
    comes out near 1.5 (a Poisson model has no such parameter and would put
    all the extra variability into the coefficients' uncertainty), and the
    coefficient near its true value."""
    n = 2000
    rng = np.random.default_rng(9)
    iv = rng.normal(size=(n, 1))
    true_iv, true_concentration = 0.5, 1.5
    mean = EXPOSURE * np.exp(BASELINE + true_iv * iv[:, 0])
    # numpy's negative binomial (number of failures before `n` successes, with
    # success probability p) has mean n * (1 - p) / p: n = concentration and
    # p = concentration / (concentration + mean) give this mean
    cases = np.random.default_rng(10).negative_binomial(
        true_concentration, true_concentration / (true_concentration + mean)
    )
    model = PartialPooledPoisson(
        iv=jnp.asarray(iv),
        pool=False,
        regression_covariates=jnp.zeros((n, 1)),
        exposure=jnp.full(n, EXPOSURE),
        groups=pl.Series(["a"] * n),
        spatial_units=pl.Series(np.arange(n)),
        adjacency_matrix=no_neighbors(n),
        group_fixed_effect=False,
        spatial=False,
        likelihood="negative_binomial",
    )
    samples = fit(model, cases)
    assert float(samples["nb_concentration"].mean()) == pytest.approx(
        true_concentration, abs=0.25
    )
    assert float(samples["iv_global"][:, 0].mean()) == pytest.approx(
        true_iv, abs=0.1
    )
