import numpy as np
import pytest
import statsmodels.api as sm
from scipy import optimize, stats

from outbreak_size import mechanistic as m

R0 = 12.0
EFFECTIVENESS = 0.97


def make_data(
    n=300, log_threshold=np.log(3.0), slope=4.0, seed=0, effects=False
):
    """
    Places with random coverage, and counts of cases drawn from the model
    with the given threshold size and slope (and, with effects, three years
    and four states whose effects are known and add up to 0).
    """
    rng = np.random.default_rng(seed)
    coverage = rng.uniform(85, 99.5, size=n)
    population = np.full(n, 50_000.0)
    lower, size = m.theory_sizes(coverage, population, R0, EFFECTIVENESS)
    years = states = None
    fit = dict(
        log_threshold=log_threshold,
        slope=slope,
        year_effects={},
        state_effects={},
        dropped_states=[],
    )
    if effects:
        years = rng.choice([2020, 2021, 2022], size=n)
        states = rng.choice(["A", "B", "C", "D"], size=n)
        fit["year_effects"] = {2020: -0.3, 2021: 0.1, 2022: 0.2}
        fit["state_effects"] = {"A": 0.5, "B": -0.2, "C": 0.1, "D": -0.4}
    mean = m.predict_mean(fit, lower, size, years, states)
    cases = rng.poisson(mean).astype(float)
    return dict(
        coverage=coverage,
        population=population,
        lower=lower,
        size=size,
        cases=cases,
        years=years,
        states=states,
    )


def test_theory_sizes_of_a_worked_example():
    # Hand calculation for coverage 98%: susceptible = 1 - 0.97 * 0.98 =
    # 0.0494, Re = 12 * 0.0494 = 0.5928, size = 1 / (1 - 0.5928) = 2.4558.
    # Coverage 96%: susceptible 0.0688, Re 0.8256, size 1 / 0.1744 = 5.7339.
    # Coverage 94.8%: susceptible 0.08044, Re 0.96528, size 28.80. Coverage
    # 94% and 90% have Re of 1.0584 and 1.524, so they are lower-coverage.
    coverage = np.array([98.0, 96.0, 94.8, 94.0, 90.0])
    lower, size = m.theory_sizes(
        coverage, np.full(5, 10_000.0), R0, EFFECTIVENESS
    )
    assert list(lower) == [False, False, False, True, True]
    np.testing.assert_allclose(size[:3], [2.4558, 5.7339, 28.80], rtol=1e-3)
    # the final size solves s * exp(-r0 * z) + z - s = 0
    susceptible = 1 - coverage[3:] * EFFECTIVENESS / 100
    residual = susceptible * np.exp(-R0 * size[3:]) + size[3:] - susceptible
    np.testing.assert_allclose(residual, 0, atol=1e-9)


def test_predicted_means_of_a_worked_example():
    # Threshold size 9 cases, slope 2: the higher-coverage places get their
    # theory size up to the ceiling of 9, and the lower-coverage places
    # exp(ln 9 + 2 z), with z = 0.00955 and 0.07595 (see the test above).
    coverage = np.array([98.0, 96.0, 94.8, 94.0, 90.0])
    lower, size = m.theory_sizes(
        coverage, np.full(5, 10_000.0), R0, EFFECTIVENESS
    )
    fit = dict(
        log_threshold=np.log(9.0),
        slope=2.0,
        year_effects={},
        state_effects={},
        dropped_states=[],
    )
    mean = m.predict_mean(fit, lower, size)
    np.testing.assert_allclose(
        mean, [2.4558, 5.7339, 9.0, 9.1736, 10.4764], rtol=1e-3
    )


def test_poisson_log_likelihood_matches_scipy():
    cases = np.array([0, 1, 4, 7])
    mean = np.array([0.5, 1.2, 3.0, 9.0])
    np.testing.assert_allclose(
        m.poisson_log_likelihood(cases, mean),
        stats.poisson.logpmf(cases, mean).sum(),
    )
    assert m.poisson_log_likelihood([0, 2], [0.0, 1.0]) == pytest.approx(
        stats.poisson.logpmf(2, 1.0)
    )
    assert m.poisson_log_likelihood([1], [0.0]) == -np.inf


def test_limit_to_population():
    np.testing.assert_array_equal(
        m.limit_to_population(np.array([5.0, 500.0]), np.array([100, 100])),
        [5.0, 100.0],
    )


def test_ranking_order_breaks_ties_by_coverage_and_keeps_equal_ties():
    mean = np.array([5.0, 5.0, 3.0, 3.0, 3.0, 9.0])
    coverage = np.array([90.0, 95.0, 96.0, 96.0, 92.0, 80.0])
    # highest mean first; equal means by lower coverage first; places with
    # equal mean and equal coverage share a position
    np.testing.assert_array_equal(
        m.ranking_order(mean, coverage), [1, 2, 4, 4, 3, 0]
    )


def test_the_fit_at_a_threshold_matches_statsmodels():
    data = make_data(effects=True, seed=1)
    effects, kept, _, _, _ = m._effects_design(
        data["years"], data["states"], data["cases"]
    )
    log_threshold = np.log(3.0)
    result = m._fit_at_threshold(
        data["lower"], data["size"], data["cases"], effects, log_threshold
    )
    # the same regression written for statsmodels: the threshold size is an
    # offset (the log of the smaller of the theory size and the threshold
    # size for higher-coverage places)
    offset = np.where(
        data["lower"],
        log_threshold,
        np.log(
            np.minimum(
                np.where(data["lower"], 1.0, data["size"]),
                np.exp(log_threshold),
            )
        ),
    )
    design = np.column_stack(
        [np.where(data["lower"], data["size"], 0.0), effects]
    )
    reference = sm.GLM(
        data["cases"], design, family=sm.families.Poisson(), offset=offset
    ).fit()
    assert result["slope"] >= 0
    np.testing.assert_allclose(
        np.concatenate([[result["slope"]], result["effect_coefficients"]]),
        reference.params,
        atol=1e-6,
    )


def test_a_negative_slope_is_replaced_by_0_and_the_rest_is_refitted():
    data = make_data(effects=True, seed=2, slope=-3.0)
    effects, *_ = m._effects_design(
        data["years"], data["states"], data["cases"]
    )
    log_threshold = np.log(3.0)
    result = m._fit_at_threshold(
        data["lower"], data["size"], data["cases"], effects, log_threshold
    )
    assert result["slope"] == 0.0
    # the other coefficients are those of the regression without the slope,
    # not those of the regression with a negative slope
    offset = np.where(
        data["lower"],
        log_threshold,
        np.log(
            np.minimum(
                np.where(data["lower"], 1.0, data["size"]),
                np.exp(log_threshold),
            )
        ),
    )
    reference = sm.GLM(
        data["cases"], effects, family=sm.families.Poisson(), offset=offset
    ).fit()
    np.testing.assert_allclose(
        result["effect_coefficients"], reference.params, atol=1e-6
    )


def test_the_search_finds_what_a_brute_force_search_finds():
    data = make_data(seed=3)
    fit = m.fit_mechanistic(data["lower"], data["size"], data["cases"])
    low, high = np.log(m.LOWEST_THRESHOLD_SIZE), np.log(data["cases"].max())

    def negative_log_likelihood(parameters):
        trial = dict(
            log_threshold=parameters[0],
            slope=parameters[1],
            year_effects={},
            state_effects={},
            dropped_states=[],
        )
        mean = m.predict_mean(trial, data["lower"], data["size"])
        return -m.poisson_log_likelihood(data["cases"], mean)

    rng = np.random.default_rng(0)
    best = np.inf
    # many starts of a bounded optimizer on both parameters at once, with the
    # slope allowed anywhere from 0 to 60
    for _ in range(40):
        start = [rng.uniform(low, high), rng.uniform(0, 20)]
        result = optimize.minimize(
            negative_log_likelihood,
            start,
            bounds=[(low, high), (0, 60)],
            method="Powell",
        )
        best = min(best, result.fun)
    assert fit["log_likelihood"] >= -best - 1e-3
    assert fit["slope"] > 0


def test_without_effects_the_mean_never_rises_with_coverage():
    for slope in (0.0, 4.0):
        data = make_data(seed=4, slope=slope)
        fit = m.fit_mechanistic(data["lower"], data["size"], data["cases"])
        order = np.argsort(data["coverage"], kind="stable")
        assert np.all(np.diff(fit["mean"][order]) <= 0)


def test_with_a_slope_of_0_places_at_the_ceiling_tie_exactly():
    data = make_data(seed=5, slope=-2.0)
    fit = m.fit_mechanistic(data["lower"], data["size"], data["cases"])
    assert fit["slope"] == 0.0
    ceiling = np.exp(fit["log_threshold"])
    # every lower-coverage place, and every higher-coverage place whose
    # theory size is above the ceiling, has exactly the ceiling as its mean
    at_ceiling = data["lower"] | (data["size"] >= ceiling)
    assert np.all(fit["mean"][at_ceiling] == ceiling)
    # so the ranking is by coverage, with equal coverage tied
    order = m.ranking_order(fit["mean"], data["coverage"])
    expected = np.unique(data["coverage"], return_inverse=True)[1]
    np.testing.assert_array_equal(order, expected)


def test_states_without_cases_are_left_out_and_predicted_0():
    data = make_data(effects=True, seed=6)
    data["cases"][data["states"] == "C"] = 0
    fit = m.fit_mechanistic(
        data["lower"],
        data["size"],
        data["cases"],
        data["years"],
        data["states"],
    )
    assert fit["dropped_states"] == ["C"]
    assert np.all(fit["mean"][data["states"] == "C"] == 0)
    assert np.all(fit["mean"][data["states"] != "C"] > 0)
    assert "C" not in fit["state_effects"]
    # a place with mean 0 gets 0 as its quantiles
    assert stats.poisson.ppf(0.5, 0.0) == 0
    # the log-likelihood is finite: the places left out have no cases
    assert np.isfinite(fit["log_likelihood"])


def test_the_effects_add_up_to_zero():
    data = make_data(effects=True, seed=7)
    fit = m.fit_mechanistic(
        data["lower"],
        data["size"],
        data["cases"],
        data["years"],
        data["states"],
    )
    assert sum(fit["year_effects"].values()) == pytest.approx(0, abs=1e-9)
    assert sum(fit["state_effects"].values()) == pytest.approx(0, abs=1e-9)
    assert set(fit["year_effects"]) == {2020, 2021, 2022}
    assert set(fit["state_effects"]) == {"A", "B", "C", "D"}


def test_the_fit_does_not_depend_on_the_names_or_the_order_of_the_states():
    data = make_data(effects=True, seed=8)
    fit = m.fit_mechanistic(
        data["lower"],
        data["size"],
        data["cases"],
        data["years"],
        data["states"],
    )
    # the states under other names (which also changes which one is last in
    # sorted order, and so which one has no column of its own) and the places
    # in another order
    renamed = {"A": "Z", "B": "Y", "C": "X", "D": "W"}
    states = np.array([renamed[s] for s in data["states"]])
    order = np.random.default_rng(0).permutation(len(states))
    other = m.fit_mechanistic(
        data["lower"][order],
        data["size"][order],
        data["cases"][order],
        data["years"][order],
        states[order],
    )
    assert other["log_likelihood"] == pytest.approx(
        fit["log_likelihood"], abs=1e-6
    )
    assert other["log_threshold"] == pytest.approx(
        fit["log_threshold"], abs=1e-4
    )
    np.testing.assert_allclose(other["mean"], fit["mean"][order], rtol=1e-4)


def test_the_number_of_parameters():
    # no effects: the threshold size, and the slope only if it is above 0
    data = make_data(seed=9, slope=4.0)
    fit = m.fit_mechanistic(data["lower"], data["size"], data["cases"])
    assert fit["slope"] > 0 and fit["n_parameters"] == 2
    data = make_data(seed=5, slope=-2.0)
    fit = m.fit_mechanistic(data["lower"], data["size"], data["cases"])
    assert fit["slope"] == 0 and fit["n_parameters"] == 1
    # effects: 2 of the 3 years and 3 of the 4 states have a column, and a
    # state left out counts as one
    data = make_data(effects=True, seed=10)
    data["cases"][data["states"] == "C"] = 0
    fit = m.fit_mechanistic(
        data["lower"],
        data["size"],
        data["cases"],
        data["years"],
        data["states"],
    )
    # 3 years and 3 kept states: 2 + 2 columns, the threshold size, the slope
    # if above 0, and the state left out
    assert fit["n_parameters"] == 2 + 2 + 1 + (fit["slope"] > 0) + 1


def test_a_best_value_at_a_bound_stops_the_fit():
    # every place has 500 cases and every place is lower-coverage, so the
    # best threshold size is 500, the highest bound of the search
    n = 50
    lower = np.ones(n, dtype=bool)
    size = np.linspace(0.01, 0.5, n)
    cases = np.full(n, 500.0)
    with pytest.raises(RuntimeError, match="bound"):
        m.fit_mechanistic(lower, size, cases)


def test_predicting_with_effects_needs_them_and_knows_its_levels():
    data = make_data(effects=True, seed=11)
    fit = m.fit_mechanistic(
        data["lower"],
        data["size"],
        data["cases"],
        data["years"],
        data["states"],
    )
    with pytest.raises(ValueError, match="years and states"):
        m.predict_mean(fit, data["lower"], data["size"])
    with pytest.raises(ValueError, match="not in the fit"):
        m.predict_mean(
            fit,
            data["lower"][:1],
            data["size"][:1],
            np.array([1999]),
            np.array(["A"]),
        )
    with pytest.raises(ValueError, match="both years and states"):
        m.fit_mechanistic(
            data["lower"], data["size"], data["cases"], data["years"]
        )


# ---------------------------------------------------------------------------
# Without effects, the mechanistic model ranks places exactly as their MMR
# coverage does (lower coverage = higher risk, equal coverage tied), whatever
# their populations. The tables rank the model's `mean` this way
# (`ranking_order(mean, coverage)`), so these tests use populations that
# differ by orders of magnitude and coverages that repeat.
# ---------------------------------------------------------------------------

# coverage at which Re = R0 * (1 - EFFECTIVENESS * coverage / 100) is exactly 1
BOUNDARY = 100 * (1 - 1 / R0) / EFFECTIVENESS


def no_effects_fit(threshold, slope):
    return dict(
        log_threshold=np.log(threshold),
        slope=slope,
        year_effects={},
        state_effects={},
        dropped_states=[],
    )


def coverage_rank(coverage):
    """0 for the lowest coverage; equal coverages share a position."""
    return np.unique(coverage, return_inverse=True)[1].ravel()


def places(n, seed, min_population=500.0, max_population=1e7, rounded=False):
    """
    Places with populations spread evenly on a log scale and coverages from 80
    to 100 (rounded to 0.1, so that many repeat, if `rounded`).
    """
    rng = np.random.default_rng(seed)
    coverage = rng.uniform(80, 100, size=n)
    if rounded:
        coverage = np.round(coverage, 1)
    population = np.exp(
        rng.uniform(np.log(min_population), np.log(max_population), size=n)
    )
    return coverage, population


def assert_ranks_like_coverage(mean, coverage):
    order = np.argsort(coverage, kind="stable")
    assert np.all(np.diff(mean[order]) <= 0), "the mean rises with coverage"
    np.testing.assert_array_equal(
        m.ranking_order(mean, coverage), coverage_rank(coverage)
    )


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("rounded", [False, True])
@pytest.mark.parametrize(
    "threshold, slope",
    [(3.0, 0.0), (3.0, 4.0), (20.0, 6.0), (150.0, 2.0), (3.0, 60.0)],
)
def test_the_mean_ranks_places_like_coverage_when_populations_differ(
    seed, rounded, threshold, slope
):
    # every population (at least 500) is above the threshold size
    coverage, population = places(400, seed, rounded=rounded)
    lower, size = m.theory_sizes(coverage, population, R0, EFFECTIVENESS)
    mean = m.predict_mean(no_effects_fit(threshold, slope), lower, size)
    assert_ranks_like_coverage(mean, coverage)


@pytest.mark.parametrize(
    "threshold, slope", [(3.0, 0.0), (3.0, 4.0), (200.0, 5.0)]
)
def test_places_with_equal_coverage_tie_whatever_their_population(
    threshold, slope
):
    # one coverage after another, each in places of 500 to 100 million people
    coverage = np.repeat(
        [
            0.0,
            50.0,
            80.0,
            90.0,
            94.0,
            94.5,
            94.6,
            95.0,
            97.0,
            99.0,
            99.9,
            100.0,
        ],
        6,
    )
    population = np.tile([500.0, 3e3, 5e4, 1e6, 3e7, 1e8], 12)
    lower, size = m.theory_sizes(coverage, population, R0, EFFECTIVENESS)
    mean = m.predict_mean(no_effects_fit(threshold, slope), lower, size)
    for value in np.unique(coverage):
        assert np.ptp(mean[coverage == value]) == 0.0, value
    assert_ranks_like_coverage(mean, coverage)


def test_the_ranking_is_continuous_across_the_re_1_boundary():
    # coverages from just below to just above the point where Re = 1, in
    # places of very different populations. A lower-coverage place (Re >= 1)
    # must never rank below a higher-coverage one.
    assert BOUNDARY == pytest.approx(94.5017, abs=1e-4)
    coverage = BOUNDARY + np.array(
        [-1.0, -0.1, -1e-3, -1e-9, 0.0, 1e-9, 1e-3, 0.1, 1.0]
    )
    for population in (500.0, 1e4, 1e6, 1e8):
        for threshold, slope in [(3.0, 0.0), (3.0, 4.0), (200.0, 5.0)]:
            lower, size = m.theory_sizes(
                coverage, np.full(len(coverage), population), R0, EFFECTIVENESS
            )
            mean = m.predict_mean(
                no_effects_fit(threshold, slope), lower, size
            )
            assert_ranks_like_coverage(mean, coverage)
    # at the boundary itself Re = 1 is a lower-coverage place, whose size
    # (a fraction of the population) is 0, so its mean is the threshold size
    lower, size = m.theory_sizes(
        np.array([BOUNDARY]), np.array([1e6]), R0, EFFECTIVENESS
    )
    assert lower[0] and size[0] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("seed", range(5))
def test_the_ranking_does_not_depend_on_the_order_of_the_places(seed):
    coverage, population = places(300, seed, rounded=True)
    lower, size = m.theory_sizes(coverage, population, R0, EFFECTIVENESS)
    mean = m.predict_mean(no_effects_fit(5.0, 4.0), lower, size)
    ranking = m.ranking_order(mean, coverage)
    order = np.random.default_rng(seed).permutation(len(coverage))
    np.testing.assert_array_equal(
        m.ranking_order(mean[order], coverage[order]), ranking[order]
    )


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("slope", [4.0, 0.0, -3.0])
def test_a_fitted_model_ranks_places_like_coverage(seed, slope):
    # the whole path: places with different populations, cases drawn from the
    # model (limited by population, as the pipeline does), a fit, and the
    # ranking of the fit's mean. A negative true slope is fitted as 0.
    coverage, population = places(500, seed, rounded=True)
    lower, size = m.theory_sizes(coverage, population, R0, EFFECTIVENESS)
    truth = no_effects_fit(3.0, slope)
    mu = m.limit_to_population(m.predict_mean(truth, lower, size), population)
    cases = np.random.default_rng(seed).poisson(mu).astype(float)
    fit = m.fit_mechanistic(lower, size, cases)
    assert fit["slope"] >= 0
    assert np.exp(fit["log_threshold"]) <= population.min()
    assert_ranks_like_coverage(fit["mean"], coverage)


def test_the_mean_limited_to_the_population_does_not_rank_like_coverage():
    # Why the tables must rank `mean` and not `mean_limited_to_population`:
    # two lower-coverage places with the same coverage but 500 and 1e6 people
    # have the same mean, but not the same mean limited to the population.
    coverage = np.array([85.0, 85.0, 90.0])
    population = np.array([500.0, 1e6, 1e6])
    lower, size = m.theory_sizes(coverage, population, R0, EFFECTIVENESS)
    mean = m.predict_mean(no_effects_fit(3.0, 40.0), lower, size)
    assert_ranks_like_coverage(mean, coverage)
    limited = m.limit_to_population(mean, population)
    assert limited[0] < limited[1]  # the 500 people cap the first place
    assert not np.array_equal(
        m.ranking_order(limited, coverage), coverage_rank(coverage)
    )


def test_the_known_limit_a_place_smaller_than_the_threshold_size():
    # The ranking is exactly the coverage ranking only while every place has
    # at least as many people as the threshold size: a higher-coverage place
    # with 10 people has a theory size of at most 10, below a ceiling of 100,
    # so it ranks below a place of a million people with higher coverage.
    # (The real data has places of at least 700 people and a threshold size of
    # about 6, so this does not happen there.)
    coverage = np.array([94.6, 95.0])
    population = np.array([10.0, 1e6])
    lower, size = m.theory_sizes(coverage, population, R0, EFFECTIVENESS)
    assert not lower.any()
    mean = m.predict_mean(no_effects_fit(100.0, 4.0), lower, size)
    assert mean[0] == pytest.approx(10.0) and mean[1] > mean[0]


def test_a_place_smaller_than_the_threshold_size_stops_the_ranking():
    fit = no_effects_fit(100.0, 4.0)
    # equal to the threshold size is fine; below it is not
    m.require_populations_above_threshold(fit, np.array([100.0, 1e6]))
    with pytest.raises(ValueError, match="fewer than the threshold"):
        m.require_populations_above_threshold(fit, np.array([99.0, 1e6]))


@pytest.mark.parametrize("seed", range(5))
def test_the_check_passes_exactly_when_the_ranking_is_the_coverage_ranking(
    seed,
):
    # populations from 5 to 1e6 and a threshold size of 20 or 100: whenever
    # the check passes the ranking is the coverage ranking, and whenever the
    # check stops it, the ranking is not (with this many small places)
    coverage, population = places(
        2000, seed, min_population=5.0, max_population=1e6
    )
    lower, size = m.theory_sizes(coverage, population, R0, EFFECTIVENESS)
    for threshold in (3.0, 20.0, 100.0):
        fit = no_effects_fit(threshold, 4.0)
        mean = m.predict_mean(fit, lower, size)
        is_coverage_ranking = np.array_equal(
            m.ranking_order(mean, coverage), coverage_rank(coverage)
        )
        try:
            m.require_populations_above_threshold(fit, population)
            passes = True
        except ValueError:
            passes = False
        if passes:
            assert is_coverage_ranking
        else:
            assert not is_coverage_ranking
