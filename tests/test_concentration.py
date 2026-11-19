"""The case-concentration curves (outbreak_size.concentration)."""

import numpy as np
import pytest

from outbreak_size.concentration import (
    COVERAGE,
    GRID,
    IMPORTS,
    MECHANISTIC,
    OBSERVED,
    POPULATION,
    RANKINGS,
    STATISTICAL,
    cumulative_share,
    mean_curve,
    rankings,
    scenario_curve,
    validation_curves,
)
from outbreak_size.importation import expected_case_draws
from outbreak_size.ranking import risk_percentile


def test_the_share_of_the_cases_at_or_below_a_percentile():
    percentiles = [0.0, 50.0, 100.0]
    cases = [0, 0, 10]
    grid = np.array([-1.0, 0.0, 49.0, 50.0, 99.9, 100.0])
    assert cumulative_share(percentiles, cases, grid).tolist() == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        100.0,
    ]
    cases = [2, 3, 5]
    assert cumulative_share(percentiles, cases, grid).tolist() == [
        0.0,
        20.0,
        20.0,
        50.0,
        50.0,
        100.0,
    ]


def test_places_with_the_same_percentile_are_all_in_or_all_out():
    # the places 0 and 1 are tied: the order they come in changes nothing
    percentiles = np.array([10.0, 10.0, 90.0])
    grid = np.array([9.0, 10.0, 89.0, 90.0])
    one = cumulative_share(percentiles, [5, 1, 4], grid)
    other = cumulative_share(percentiles[::-1], [4, 1, 5], grid)
    assert one.tolist() == [0.0, 60.0, 60.0, 100.0]
    assert other.tolist() == one.tolist()


def test_no_cases_gives_a_curve_of_zeros():
    assert cumulative_share([1.0, 2.0], [0, 0]).tolist() == [0.0] * len(GRID)


def test_the_years_are_averaged_with_the_same_weight():
    # a year with 1000 cases and one with 10 cases, curves of 100% and 0% at x
    big = np.array([0.0, 100.0])
    small = np.array([50.0, 100.0])
    # pooling the cases would be dominated by the big year; the mean is not
    assert mean_curve([big, small]).tolist() == [25.0, 100.0]


def toy_year(seed=0, places=60, draws=40):
    rng = np.random.default_rng(seed)
    rate = rng.gamma(0.5, 0.4, size=(draws, places))
    probability = 1 - np.exp(-rate)
    size = rng.gamma(2.0, 5.0, size=(draws, places)) * np.arange(1, places + 1)
    other = rng.gamma(2.0, 5.0, size=(draws, places))
    coverage = rng.uniform(80, 99, places)
    population = rng.integers(1_000, 1_000_000, places).astype(float)
    cases = rng.poisson(rate.mean(axis=0) * 2)
    return size, other, probability, rate, coverage, population, cases


def test_every_ranking_is_a_percentile_of_every_place():
    size, other, probability, rate, coverage, population, cases = toy_year()
    out = rankings(size, other, probability, rate, coverage, population, cases)
    assert list(out) == RANKINGS
    for percentiles in out.values():
        assert percentiles.shape == (60,)
        assert percentiles.min() >= 0 and percentiles.max() <= 100


def test_each_ranking_orders_the_places_the_way_it_says():
    size, other, probability, rate, coverage, population, cases = toy_year(1)
    out = rankings(size, other, probability, rate, coverage, population, cases)
    lowest_coverage = int(np.argmin(coverage))
    assert out[COVERAGE][lowest_coverage] == 100.0  # the lower, the riskier
    assert out[POPULATION][int(np.argmax(population))] == 100.0
    assert out[IMPORTS][int(np.argmax(rate.mean(axis=0)))] == 100.0
    # a place with more cases is always above one with fewer, and the ranking
    # has no ties (places with the same cases are in any order)
    observed = out[OBSERVED]
    for fewer in (0, 1):
        for more in range(fewer + 1, 3):
            above, below = observed[cases == more], observed[cases == fewer]
            if len(above) and len(below):
                assert above.min() > below.max()
    assert len(np.unique(observed)) == len(cases)
    # the models rank by the average of the expected cases of the draws
    expected = risk_percentile(expected_case_draws(size, probability, rate))
    assert out[STATISTICAL].tolist() == pytest.approx(expected.tolist())
    assert out[MECHANISTIC].tolist() != out[STATISTICAL].tolist()


def test_the_observed_cases_are_the_lowest_curve_and_the_diagonal_the_highest():
    # no ranking can put more of the cases at the high percentiles than the
    # ranking by the cases themselves
    size, other, probability, rate, coverage, population, cases = toy_year(2)
    cases = cases + np.arange(60) % 3
    curves = validation_curves(
        rankings(size, other, probability, rate, coverage, population, cases),
        cases,
    )
    best = curves[OBSERVED]
    for name, curve in curves.items():
        if name == COVERAGE:
            continue  # the ties of a tied block are all in or all out
        assert (curve >= best - 1e-9).all(), name
    # the places are a continuous ranking: the share of the cases at the
    # percentile x is at most x for the best ranking
    assert (best <= GRID + 1e-9).all()
    assert best[-1] == pytest.approx(100.0)


def test_a_scenario_curve_ranks_by_the_mean_and_counts_the_mean_cases():
    rng = np.random.default_rng(3)
    draws = rng.gamma(1.0, 3.0, size=(50, 12)) * np.arange(1, 13)
    curve = scenario_curve(draws)
    manual = cumulative_share(risk_percentile(draws), draws.mean(axis=0), GRID)
    assert curve.tolist() == pytest.approx(manual.tolist())
    # the place with the highest mean is the last one: the curve reaches 100%
    # only at the top percentile
    assert curve[-1] == pytest.approx(100.0)
    assert curve[-2] < 100.0
