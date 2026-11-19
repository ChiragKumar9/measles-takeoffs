"""The likelihood options of the random-walk outbreak model
(outbreak_size.random_walk)."""

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import polars as pl
import pytest
from numpyro.handlers import seed, trace

from outbreak_size import random_walk


def build(likelihood, population, **kwargs):
    """Four counties in a ring, each seen in 3 years, one coefficient that
    varies over time and by state, and no neighbours' data needed. `kwargs`
    are further arguments of the model."""
    units = np.repeat(np.arange(4), 3)
    n = len(units)
    adjacency = np.zeros((4, 4))
    for i in range(4):
        adjacency[i, (i + 1) % 4] = adjacency[(i + 1) % 4, i] = 1
    return random_walk.PartialPooledPoisson(
        time_and_group_varying=jnp.zeros((n, 1)),
        state_vary_rw=False,
        group_varying_only=None,
        time_varying_only=None,
        controls=jnp.zeros((n, 1)),
        population=jnp.full(n, population),
        groups=pl.Series(["a"] * n),
        years=pl.Series(np.tile([2011, 2012, 2013], 4)),
        spatial_units=pl.Series(units),
        adjacency_matrix=adjacency,
        likelihood=likelihood,
        **kwargs,
    )


def test_the_spatial_sigma_prior_applies_to_both_spatial_scales():
    # one prior for the scale of the CAR field ("spatial_sigma") and for that of the
    # independent spatial effects ("spatial_random_sigma"), HalfNormal(0.25) by default
    model = build("poisson", 100.0, spatial_sigma=dist.HalfNormal(5.0))
    traced = trace(seed(model.fit, 0)).get_trace(None)
    assert float(traced["spatial_sigma"]["fn"].scale) == 5.0
    assert float(traced["spatial_random_sigma"]["fn"].scale) == 5.0
    default = trace(seed(build("poisson", 100.0).fit, 0)).get_trace(None)
    assert float(default["spatial_sigma"]["fn"].scale) == 0.25
    assert float(default["spatial_random_sigma"]["fn"].scale) == 0.25


def test_an_unknown_likelihood_is_refused():
    with pytest.raises(ValueError, match="likelihood must be one of"):
        build("gaussian", 100.0)


def test_negative_binomial_predictions_stop_at_the_population():
    # counties of 3 people drawn from the prior: the raw negative binomial
    # draws often exceed 3, the predicted cases never do
    model = build("negative_binomial", 3.0)
    raw, capped = [], []
    for s in range(30):
        traced = trace(seed(model.fit, s)).get_trace(None)
        raw.append(np.asarray(traced["cases_uncapped"]["value"]))
        capped.append(np.asarray(traced["cases"]["value"]))
        assert "nb_concentration" in traced
    raw, capped = np.concatenate(raw), np.concatenate(capped)
    assert raw.max() > 3
    assert capped.max() <= 3
    assert capped == pytest.approx(np.minimum(raw, 3.0))


def test_poisson_predictions_are_ordinary_samples():
    traced = trace(seed(build("poisson", 3.0).fit, 0)).get_trace(None)
    assert "cases_uncapped" not in traced
    assert "nb_concentration" not in traced
    assert traced["cases"]["type"] == "sample"
