"""The spatial-unit guard of the hierarchical Poisson model (outbreak_size.poisson)."""

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import polars as pl
import pytest
from numpyro import handlers

from outbreak_size import poisson


def build(units, n_adjacency_rows, spatial=True):
    n = len(units)
    adjacency = np.ones((n_adjacency_rows, n_adjacency_rows)) - np.eye(
        n_adjacency_rows
    )
    return poisson.PartialPooledPoisson(
        iv=jnp.zeros((n, 3)),
        pool=True,
        regression_covariates=jnp.zeros((n, 1)),
        exposure=jnp.ones(n),
        groups=pl.Series(["a"] * n),
        spatial_units=pl.Series(units),
        adjacency_matrix=adjacency,
        group_fixed_effect=False,
        spatial=spatial,
    )


def test_units_and_adjacency_rows_that_agree_are_accepted():
    build(["x", "y", "z"], 3)


def test_repeated_rows_of_a_unit_are_one_unit():
    # several years of the same county are rows of one spatial unit
    build(["x", "x", "y", "y", "z"], 3)


def test_a_county_spelled_two_ways_is_refused():
    # 4 unit names but 3 adjacency rows: every later unit would be shifted
    # onto another county's row without any error
    with pytest.raises(ValueError, match="4 spatial units"):
        build(["x", "y", "z", "z2"], 3)


def test_fewer_units_than_adjacency_rows_is_refused():
    with pytest.raises(ValueError, match="2 spatial units"):
        build(["x", "y"], 3)


def test_the_check_only_applies_to_spatial_models():
    build(["x", "y", "z", "z2"], 3, spatial=False)


def test_count_ivs_counts_the_leading_pooled_variables():
    """The pooled variables come first in a model's variable list; counting
    stops at the first ordinary covariate."""
    assert poisson.count_ivs(["estimated_mmr_cov", "log_income"]) == 1
    assert (
        poisson.count_ivs(
            ["estimated_mmr_cov", "nonmed_exempt_pct", "med_exempt_pct", "pop"]
        )
        == 3
    )
    assert poisson.count_ivs(["log_income", "estimated_mmr_cov"]) == 0
    assert poisson.count_ivs([]) == 0
    assert (
        poisson.count_ivs(
            [
                "estimated_mmr_cov_5yr",
                "nonmed_exempt_pct_5yr",
                "med_exempt_pct_5yr",
            ]
        )
        == 3
    )
    # the school SD is pooled when it follows the vaccination ivs
    assert (
        poisson.count_ivs(
            [
                "estimated_mmr_cov",
                "nonmed_exempt_pct",
                "med_exempt_pct",
                "school_sd",
                "log_income",
            ]
        )
        == 4
    )


# --- the population exponent -----------------------------------------------------


def expected_cases(exposure, is_observed, **values):
    """The mean `mu` of the model for the given exposures, with the sampled
    coefficients set to `values` (the others to 0) and no covariates."""
    from numpyro.handlers import seed, substitute, trace

    n = len(exposure)
    model = poisson.PartialPooledPoisson(
        iv=jnp.zeros((n, 1)),
        pool=False,
        regression_covariates=jnp.zeros((n, 1)),
        exposure=jnp.asarray(exposure, dtype=float),
        groups=pl.Series(["a"] * n),
        spatial_units=pl.Series(np.arange(n)),
        adjacency_matrix=np.zeros((n, n)),
        group_fixed_effect=False,
        spatial=False,
    )
    data = {
        "regression_betas": jnp.zeros(1),
        "iv_global": jnp.zeros(1),
        **{k: jnp.asarray(v) for k, v in values.items()},
    }
    traced = trace(seed(substitute(model.fit, data=data), 0)).get_trace(
        jnp.asarray(is_observed)
    )
    return np.asarray(traced["mu"]["value"])


EXPOSURES = [1000.0, 2000.0, 4000.0, 8000.0]
ALL_OBSERVED = [1, 1, 1, 1]


def test_an_exponent_of_one_makes_the_expected_cases_proportional_to_population():
    mu = expected_cases(
        EXPOSURES, ALL_OBSERVED, baseline_rate=-8.0, pop_exponent=1.0
    )
    assert mu == pytest.approx(np.array(EXPOSURES) * np.exp(-8.0))


def test_an_exponent_of_zero_gives_every_county_the_reference_population_size():
    # the reference is the median population (3000): every county's expected
    # cases are those of a county of 3000 people
    mu = expected_cases(
        EXPOSURES, ALL_OBSERVED, baseline_rate=-8.0, pop_exponent=0.0
    )
    assert mu == pytest.approx(np.full(4, 3000.0 * np.exp(-8.0)))


def test_the_exponent_scales_the_expected_cases_by_population_to_that_power():
    mu = expected_cases(
        EXPOSURES, ALL_OBSERVED, baseline_rate=-8.0, pop_exponent=0.5
    )
    expected = 3000.0 * np.exp(-8.0) * (np.array(EXPOSURES) / 3000.0) ** 0.5
    assert mu == pytest.approx(expected)


def test_a_fixed_exponent_is_not_sampled_and_scales_the_expected_cases():
    from numpyro.handlers import seed, substitute, trace

    n = len(EXPOSURES)
    model = poisson.PartialPooledPoisson(
        iv=jnp.zeros((n, 1)),
        pool=False,
        regression_covariates=jnp.zeros((n, 1)),
        exposure=jnp.asarray(EXPOSURES, dtype=float),
        groups=pl.Series(["a"] * n),
        spatial_units=pl.Series(np.arange(n)),
        adjacency_matrix=np.zeros((n, n)),
        group_fixed_effect=False,
        spatial=False,
        pop_exponent=1,
    )
    data = {
        "regression_betas": jnp.zeros(1),
        "iv_global": jnp.zeros(1),
        "baseline_rate": jnp.asarray(-8.0),
    }
    traced = trace(seed(substitute(model.fit, data=data), 0)).get_trace(
        jnp.ones(n)
    )
    assert "pop_exponent" not in traced
    assert np.asarray(traced["mu"]["value"]) == pytest.approx(
        np.array(EXPOSURES) * np.exp(-8.0)
    )


def test_the_reference_population_is_the_median_of_the_observed_rows_only():
    # the last row is not observed (a row to predict): its huge population
    # must not move the reference, so the first three rows give the same mean
    # as when the model has only them
    mu_with = expected_cases(
        EXPOSURES[:3] + [1e9],
        [1, 1, 1, 0],
        baseline_rate=-8.0,
        pop_exponent=0.5,
    )
    mu_without = expected_cases(
        EXPOSURES[:3], [1, 1, 1], baseline_rate=-8.0, pop_exponent=0.5
    )
    assert mu_with[:3] == pytest.approx(mu_without)


# --- the negative binomial likelihood --------------------------------------------


def one_county_trace(likelihood, cases, concentration=None):
    """The trace of a one-county model whose mean is exactly 2 (an exponent of
    1, a population of 1000 and a baseline of log(2 / 1000), no covariates)."""
    from numpyro.handlers import seed, substitute, trace

    model = poisson.PartialPooledPoisson(
        iv=jnp.zeros((1, 1)),
        pool=False,
        regression_covariates=jnp.zeros((1, 1)),
        exposure=jnp.asarray([1000.0]),
        groups=pl.Series(["a"]),
        spatial_units=pl.Series([0]),
        adjacency_matrix=np.zeros((1, 1)),
        group_fixed_effect=False,
        spatial=False,
        likelihood=likelihood,
    )
    data = {
        "regression_betas": jnp.zeros(1),
        "iv_global": jnp.zeros(1),
        "baseline_rate": jnp.asarray(np.log(2.0 / 1000.0)),
        "pop_exponent": jnp.asarray(1.0),
    }
    if concentration is not None:
        data["nb_concentration"] = jnp.asarray(concentration)
    return trace(seed(substitute(model.fit, data=data), 0)).get_trace(
        jnp.ones(1), cases
    )


def test_an_unknown_likelihood_is_refused():
    with pytest.raises(ValueError, match="likelihood must be one of"):
        one_county_trace("gaussian", None)


def test_the_poisson_likelihood_has_no_concentration():
    traced = one_county_trace("poisson", None)
    assert "nb_concentration" not in traced


def test_the_negative_binomial_log_probability_of_the_cases_is_worked_out_by_hand():
    # a concentration of 1 makes the negative binomial a geometric
    # distribution with mean 2: P(k) = (1/3) * (2/3)**k, so
    # P(0) = 1/3 and P(2) = 4/27
    for cases, probability in ((0.0, 1 / 3), (2.0, 4 / 27)):
        traced = one_county_trace(
            "negative_binomial", jnp.asarray([cases]), concentration=1.0
        )
        log_prob = float(traced["poisson_log_prob"]["fn"].log_factor.sum())
        assert log_prob == pytest.approx(np.log(probability))


def test_the_poisson_log_probability_of_the_same_cases_is_the_poisson_one():
    # Poisson with mean 2: P(0) = exp(-2), P(2) = 2 * exp(-2)
    for cases, probability in ((0.0, np.exp(-2)), (2.0, 2 * np.exp(-2))):
        traced = one_county_trace("poisson", jnp.asarray([cases]))
        log_prob = float(traced["poisson_log_prob"]["fn"].log_factor.sum())
        assert log_prob == pytest.approx(np.log(probability))


def test_the_negative_binomial_has_the_same_mean_and_a_larger_variance():
    # variance = mean + mean**2 / concentration = 2 + 4 / 1 = 6 (Poisson: 2)
    nb = one_county_trace("negative_binomial", None, concentration=1.0)
    assert float(nb["cases_uncapped"]["fn"].mean[0]) == pytest.approx(2.0)
    assert float(nb["cases_uncapped"]["fn"].variance[0]) == pytest.approx(6.0)
    po = one_county_trace("poisson", None)
    assert float(po["cases"]["fn"].variance[0]) == pytest.approx(2.0)


def test_predicted_negative_binomial_cases_never_exceed_the_population():
    # a county of 5 people with a mean of 5 and a very small concentration (a
    # huge tail): the raw draws often exceed 5, the predicted cases never do
    from numpyro.handlers import seed, substitute, trace

    model = poisson.PartialPooledPoisson(
        iv=jnp.zeros((1, 1)),
        pool=False,
        regression_covariates=jnp.zeros((1, 1)),
        exposure=jnp.asarray([5.0]),
        groups=pl.Series(["a"]),
        spatial_units=pl.Series([0]),
        adjacency_matrix=np.zeros((1, 1)),
        group_fixed_effect=False,
        spatial=False,
        likelihood="negative_binomial",
    )
    data = {
        "regression_betas": jnp.zeros(1),
        "iv_global": jnp.zeros(1),
        "baseline_rate": jnp.asarray(0.0),  # a mean of 5 * exp(0) = 5
        "pop_exponent": jnp.asarray(1.0),
        "nb_concentration": jnp.asarray(0.05),
    }
    raw, capped = [], []
    for s in range(200):
        traced = trace(seed(substitute(model.fit, data=data), s)).get_trace(
            jnp.ones(1), None
        )
        raw.append(float(traced["cases_uncapped"]["value"][0]))
        capped.append(float(traced["cases"]["value"][0]))
    assert max(raw) > 5  # the uncapped draws can exceed the population
    assert max(capped) == 5.0  # the capped ones stop at it, and reach it
    assert capped == [min(r, 5.0) for r in raw]


def test_the_poisson_predictions_are_drawn_as_before_and_not_capped():
    traced = one_county_trace("poisson", None)
    assert "cases_uncapped" not in traced
    assert traced["cases"]["type"] == "sample"


# --- the effectiveness of the MMR vaccine as a parameter --------------------

COVERAGE = np.array([90.0, 95.0, 99.0, 80.0])


def one_county_per_row_model(**kwargs):
    n = len(COVERAGE)
    return poisson.PartialPooledPoisson(
        pool=True,
        regression_covariates=jnp.zeros((n, 1)),
        exposure=jnp.full(n, 1000.0),
        groups=pl.Series(["a"] * n),
        spatial_units=pl.Series(list("wxyz")),
        adjacency_matrix=np.zeros((n, n)),
        group_fixed_effect=False,
        spatial=False,
        **kwargs,
    )


def sites_of(model, substitutions, seed):
    trace = handlers.trace(
        handlers.substitute(handlers.seed(model.fit, seed), substitutions)
    ).get_trace(jnp.ones(len(COVERAGE)))
    return trace


def test_the_model_rebuilds_the_natural_scale_coverage_from_the_effectiveness():
    # the coverage column as 02 builds it for the fixed effectiveness 0.97:
    # log(1 - 0.97 * coverage / 100), standardized
    log_susceptible = np.log(1 - 0.97 * COVERAGE / 100)
    mean, sd = log_susceptible.mean(), log_susceptible.std()
    fixed = one_county_per_row_model(
        iv=jnp.array(((log_susceptible - mean) / sd)[:, None])
    )
    estimated = one_county_per_row_model(
        iv=jnp.zeros((len(COVERAGE), 1)),  # not used
        mmr_effectiveness=dist.Uniform(0.8, 0.999),
        coverage_percent=jnp.array(COVERAGE),
        coverage_mean_sd=(mean, sd),
    )
    reference = sites_of(fixed, {}, seed=0)
    values = {
        name: site["value"]
        for name, site in reference.items()
        if site["type"] == "sample" and name != "cases"
    }
    rebuilt = sites_of(estimated, {**values, "mmr_effectiveness": 0.97}, 1)
    # (float32 here, and the sizes are large: a tolerance of 1e-3)
    np.testing.assert_allclose(
        rebuilt["mu"]["value"], reference["mu"]["value"], rtol=1e-3
    )


def test_a_sampled_effectiveness_moves_the_expected_size_as_computed_by_hand():
    # Substitute: baseline 0, population exponent 0 (the size is that of the
    # reference population, 1000), no group variability, a coefficient of 1 on
    # the coverage. The expected size is then 1000 * exp(x), x = (log(1 - e *
    # coverage / 100) - mean) / sd.
    # By hand for e = 0.97, coverage 90%, mean -2, sd 0.5:
    # log(1 - 0.873) = log(0.127) = -2.0636, x = (-2.0636 + 2) / 0.5 = -0.1271,
    # 1000 * exp(x) = 880.6
    model = one_county_per_row_model(
        iv=jnp.zeros((len(COVERAGE), 1)),
        mmr_effectiveness=dist.Uniform(0.8, 0.999),
        coverage_percent=jnp.array(COVERAGE),
        coverage_mean_sd=(-2.0, 0.5),
    )
    fixed_values = {
        "baseline_rate": 0.0,
        "regression_betas": jnp.zeros(1),
        "iv_global": jnp.ones(1),
        "iv_group_variabilities": jnp.zeros(1),
        "iv_group_offsets": jnp.zeros((1, 1)),
        "pop_exponent": 0.0,
    }
    trace = sites_of(model, {**fixed_values, "mmr_effectiveness": 0.97}, 0)
    np.testing.assert_allclose(trace["mu"]["value"][0], 880.6, atol=0.1)
    # a higher effectiveness leaves fewer susceptibles, so a smaller size (the
    # last county, at 80% coverage, is capped at its population either way)
    higher = sites_of(model, {**fixed_values, "mmr_effectiveness": 0.99}, 0)
    assert (higher["mu"]["value"][:3] < trace["mu"]["value"][:3]).all()


def test_an_effectiveness_prior_without_the_coverage_is_refused():
    with pytest.raises(ValueError, match="coverage_percent"):
        one_county_per_row_model(
            iv=jnp.zeros((len(COVERAGE), 1)),
            mmr_effectiveness=dist.Uniform(0.8, 0.999),
        )
