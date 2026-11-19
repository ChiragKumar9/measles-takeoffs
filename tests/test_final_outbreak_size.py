"""Expected outbreak size from the reproduction number (outbreak_size.final_outbreak_size)."""

import numpy as np
import pytest

from outbreak_size.final_outbreak_size import final_outbreak_size


def test_subcritical_size_is_the_stuttering_chain_mean_and_variance():
    # Re = 2 * 0.25 = 0.5: mean 1 / (1 - Re) = 2, variance Re / (1 - Re)^3 = 4
    size, variance = final_outbreak_size(R0=2.0, s0=0.25, N=1_000_000)
    assert size == pytest.approx(2.0)
    assert variance == pytest.approx(4.0)


def test_subcritical_size_is_capped_at_the_population():
    size, _ = final_outbreak_size(R0=1.0, s0=0.999, N=10)
    assert size == 10


def test_negative_binomial_offspring_widen_the_variance():
    _, poisson_variance = final_outbreak_size(R0=2.0, s0=0.25, N=10**6)
    _, clumped_variance = final_outbreak_size(R0=2.0, s0=0.25, N=10**6, k=0.5)
    assert poisson_variance is not None and clumped_variance is not None
    assert clumped_variance > poisson_variance


def test_supercritical_size_solves_the_final_size_equation():
    # R0 = 2, everyone susceptible: z = 1 - exp(-2 z), z = 0.79681213
    z, variance = final_outbreak_size(R0=2.0, s0=1.0, N=1000)
    assert variance is None
    assert z == pytest.approx(0.79681213, abs=1e-6)
    assert z == pytest.approx(1 - np.exp(-2.0 * z), abs=1e-9)


def test_supercritical_fraction_grows_with_the_susceptible_fraction():
    sizes = [
        final_outbreak_size(R0=12.0, s0=s0, N=1000)[0]
        for s0 in (0.1, 0.2, 0.5, 1.0)
    ]
    assert sizes == sorted(sizes)
    assert all(0 < z <= 1 for z in sizes)
