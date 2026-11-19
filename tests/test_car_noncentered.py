"""The non-centred CAR field (outbreak_size.car_noncentered) and the
`non_centered_car` option of the outbreak models."""

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl
import pytest
from jax.random import PRNGKey
from numpyro.handlers import reparam, seed, trace
from numpyro.infer import MCMC, NUTS, Predictive

from outbreak_size import poisson, random_walk
from outbreak_size.car_noncentered import CARNonCentered


def grid(side):
    """Adjacency of side x side counties that touch along an edge."""
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


# the tests run in float32 (the pipeline scripts turn on float64), so the
# exact identities below hold to about 1e-6 relative, not to 1e-12


def covariance_of_the_transform(car, correlation):
    """Column j is the field of the white noise that is 1 at unit j: the
    covariance of the field is T T^T."""
    n = car.num_units
    columns = [
        np.asarray(car.transform(jnp.eye(n)[j], correlation)) for j in range(n)
    ]
    transform = np.stack(columns, axis=1)
    return transform @ transform.T


def test_two_counties_have_the_hand_computed_covariance():
    # D = identity, so inverse(D - cW) = 1 / (1 - c^2) * [[1, c], [c, 1]]
    car = CARNonCentered(np.array([[0.0, 1.0], [1.0, 0.0]]))
    c = 0.7
    expected = np.array([[1, c], [c, 1]]) / (1 - c**2)
    assert covariance_of_the_transform(car, c) == pytest.approx(
        expected, rel=1e-5, abs=1e-5
    )
    assert expected[0, 0] == pytest.approx(1.960784, abs=1e-6)  # by hand


@pytest.mark.parametrize("correlation", [-0.5, 0.0, 0.7, 0.95])
def test_a_grid_has_exactly_the_covariance_of_the_car_distribution(
    correlation,
):
    adjacency = grid(5)
    car = CARNonCentered(adjacency)
    precision = np.diag(adjacency.sum(axis=1)) - correlation * adjacency
    assert covariance_of_the_transform(car, correlation) == pytest.approx(
        np.linalg.inv(precision), rel=1e-4, abs=1e-4
    )


def test_a_sparse_adjacency_gives_the_same_transform():
    from scipy import sparse

    dense, sp = (
        CARNonCentered(grid(4)),
        CARNonCentered(sparse.csr_matrix(grid(4))),
    )
    z = jnp.arange(16.0)
    assert np.asarray(dense.transform(z, 0.4)) == pytest.approx(
        np.asarray(sp.transform(z, 0.4))
    )


def test_a_unit_without_a_neighbour_is_refused():
    adjacency = np.zeros((3, 3))
    adjacency[0, 1] = adjacency[1, 0] = 1
    with pytest.raises(ValueError, match="needs a neighbour"):
        CARNonCentered(adjacency)


def test_the_draws_of_the_reparametrized_site_have_the_cars_covariance():
    adjacency = grid(4)
    c = 0.8

    def prior():
        numpyro.sample(
            "raw",
            dist.CAR(
                loc=0.0,
                correlation=c,  # type: ignore
                conditional_precision=1.0,  # type: ignore
                adj_matrix=adjacency,  # type: ignore
                is_sparse=True,
            ),
        )

    model = reparam(prior, config={"raw": CARNonCentered(adjacency)})
    draws = Predictive(model, num_samples=40000, return_sites=["raw"])(
        PRNGKey(0)
    )["raw"]
    target = np.linalg.inv(np.diag(adjacency.sum(axis=1)) - c * adjacency)
    relative_error = np.abs(np.cov(np.asarray(draws).T) - target).max() / (
        np.abs(target).max()
    )
    assert relative_error < 0.03  # Monte Carlo noise is about 0.01


def build_outbreak_model(non_centered_car, years=None, side=4, repeats=3):
    n_units = side * side
    units = np.repeat(np.arange(n_units), repeats)
    n = len(units)
    return poisson.PartialPooledPoisson(
        iv=jnp.zeros((n, 1)),
        pool=False,
        regression_covariates=jnp.zeros((n, 1)),
        exposure=jnp.full(n, 1000.0),
        groups=pl.Series(["a"] * n),
        spatial_units=pl.Series(units),
        adjacency_matrix=grid(side),
        years=years,  # type: ignore
        group_fixed_effect=False,
        spatial=True,
        non_centered_car=non_centered_car,
    )


def test_the_option_samples_white_noise_and_stores_the_field_as_before():
    model = build_outbreak_model(True)
    traced = trace(seed(model.fit, 3)).get_trace(jnp.ones(48))
    assert traced["raw_spatial_effect_decentered"]["type"] == "sample"
    assert traced["raw_spatial_effect"]["type"] == "deterministic"
    c = 2 * float(traced["spatial_correlation"]["value"]) - 1
    car = CARNonCentered(grid(4))
    expected = car.transform(
        traced["raw_spatial_effect_decentered"]["value"], c
    )
    assert np.asarray(traced["raw_spatial_effect"]["value"]) == pytest.approx(
        np.asarray(expected)
    )


def test_without_the_option_the_model_is_unchanged():
    traced = trace(seed(build_outbreak_model(False).fit, 3)).get_trace(
        jnp.ones(48)
    )
    assert traced["raw_spatial_effect"]["type"] == "sample"
    assert "raw_spatial_effect_decentered" not in traced


def test_a_model_without_a_car_term_ignores_the_option():
    n = 6
    model = poisson.PartialPooledPoisson(
        iv=jnp.zeros((n, 1)),
        pool=False,
        regression_covariates=jnp.zeros((n, 1)),
        exposure=jnp.full(n, 1000.0),
        groups=pl.Series(["a"] * n),
        spatial_units=pl.Series(np.arange(n)),
        adjacency_matrix=np.zeros((n, n)),
        group_fixed_effect=False,
        spatial=False,
        non_centered_car=True,
    )
    traced = trace(seed(model.fit, 0)).get_trace(jnp.ones(n))
    assert "raw_spatial_effect" not in traced


def test_every_year_of_a_per_year_car_is_reparametrized():
    years = pl.Series(np.tile([2020, 2021], 16 * 3 // 2))
    model = build_outbreak_model(True, years=years)
    traced = trace(seed(model.fit, 1)).get_trace(jnp.ones(48))
    # the stored field keeps the name the CAR figure reads
    for year in (0, 1):
        assert (
            traced[f"year_{year}/raw_spatial_effect"]["type"]
            == "deterministic"
        )
    # one white-noise site per year (numpyro puts the year's prefix on the
    # name twice, which does not matter)
    white_noise = [
        k for k in traced if k.endswith("raw_spatial_effect_decentered")
    ]
    assert len(white_noise) == 2
    assert all(traced[k]["type"] == "sample" for k in white_noise)


def test_the_random_walk_model_takes_the_option_too():
    units = np.repeat(np.arange(4), 3)
    n = len(units)
    adjacency = np.zeros((4, 4))
    for i in range(4):
        adjacency[i, (i + 1) % 4] = adjacency[(i + 1) % 4, i] = 1
    model = random_walk.PartialPooledPoisson(
        time_and_group_varying=jnp.zeros((n, 1)),
        state_vary_rw=False,
        group_varying_only=None,
        time_varying_only=None,
        controls=jnp.zeros((n, 1)),
        population=jnp.full(n, 100.0),
        groups=pl.Series(["a"] * n),
        years=pl.Series(np.tile([2011, 2012, 2013], 4)),
        spatial_units=pl.Series(units),
        adjacency_matrix=adjacency,
        non_centered_car=True,
    )
    traced = trace(seed(model.fit, 0)).get_trace(None)
    assert traced["raw_spatial_effect_decentered"]["type"] == "sample"
    assert traced["raw_spatial_effect"]["type"] == "deterministic"


def test_both_forms_give_the_same_posterior_and_predictions_follow_the_draws():
    """Counties on a grid with a known spatial field: the two forms agree on
    the spatial scale, the correlation and the coefficient, and
    Predictive recomputes the stored linear predictor from the draws."""
    side, repeats = 6, 5
    n_units = side * side
    units = np.repeat(np.arange(n_units), repeats)
    n = len(units)
    rng = np.random.default_rng(7)
    car = CARNonCentered(grid(side))
    field = np.asarray(
        car.transform(jnp.asarray(rng.standard_normal(n_units)), 0.7)
    )
    iv = rng.normal(size=(n, 1))
    mu = 4000.0 * np.exp(-6.0 + 0.4 * iv[:, 0] + 0.6 * field[units])
    cases = rng.negative_binomial(3.0, 3.0 / (3.0 + mu))

    def fit(non_centered_car):
        model = poisson.PartialPooledPoisson(
            iv=jnp.asarray(iv),
            pool=False,
            regression_covariates=jnp.zeros((n, 1)),
            exposure=jnp.full(n, 4000.0),
            groups=pl.Series(["a"] * n),
            spatial_units=pl.Series(units),
            adjacency_matrix=grid(side),
            group_fixed_effect=False,
            spatial=True,
            likelihood="negative_binomial",
            non_centered_car=non_centered_car,
        )
        mcmc = MCMC(
            NUTS(model.fit, target_accept_prob=0.85),
            num_warmup=400,
            num_samples=400,
            num_chains=1,
            progress_bar=False,
        )
        mcmc.run(PRNGKey(1), is_observed=jnp.ones(n), cases=jnp.asarray(cases))
        return model, mcmc.get_samples()

    centred_model, centred = fit(False)
    model, non_centred = fit(True)
    # each posterior mean within 0.1 of the other (their standard
    # deviations are about 0.03 to 0.15)
    for site in ("spatial_sigma", "spatial_correlation", "iv_global"):
        a = float(np.asarray(centred[site]).reshape(400, -1)[:, 0].mean())
        b = float(np.asarray(non_centred[site]).reshape(400, -1)[:, 0].mean())
        assert a == pytest.approx(b, abs=0.1), site
    assert non_centred["raw_spatial_effect"].shape == (400, n_units)
    predicted = Predictive(model.fit, non_centred, return_sites=["mu"])(
        PRNGKey(2), is_observed=jnp.ones(n), cases=None
    )["mu"]
    assert np.asarray(predicted) == pytest.approx(
        np.asarray(non_centred["mu"]), rel=1e-6
    )
