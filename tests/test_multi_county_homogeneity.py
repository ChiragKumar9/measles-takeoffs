"""
The panel d code of analysis/figure_predicting_county_level_outbreaks.py: the
multi-county flag at another case threshold, and the test and interval of the
SD term of the models of the supplemental table.
"""

import importlib.util
import os
import sys

import numpy as np
import polars as pl
import pytest

# the figure script needs the plotting stack; skip where it is not installed
for module in ("matplotlib", "geopandas", "jax", "addfips"):
    pytest.importorskip(module)

SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "analysis",
    "figure_predicting_county_level_outbreaks.py",
)


def load_script():
    import jax

    # the script calls enable_x64() when it is loaded: put the setting back
    argv, sys.argv = sys.argv, ["x"]
    x64 = jax.config.jax_enable_x64
    try:
        spec = importlib.util.spec_from_file_location("outbreak_fig", SCRIPT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.argv = argv
        jax.config.update("jax_enable_x64", x64)
    return module


def test_the_multi_county_flag_needs_min_cases_in_the_county_and_its_neighbor():
    figure = load_script()
    # a - b - c in a row, d alone; cases 6, 3, 5, 9
    df_spread = pl.DataFrame(
        {
            "year": [2025] * 4,
            "fips": ["a", "b", "c", "d"],
            "cases_outbreak": [6, 3, 5, 9],
        }
    )
    matrix = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
    no_matrix = np.zeros((0, 0))

    def flags(**kwargs):
        return figure.multi_county_flags(
            df_spread, no_matrix, matrix, **kwargs
        ).sort("fips")

    # the default is panel c's: 3 or more cases
    default = flags()
    assert default["fips"].to_list() == ["a", "b", "c", "d"]
    assert default["multi_county"].to_list() == [True, True, True, False]
    assert len(set(default["group"].to_list()[:3])) == 1  # one outbreak
    assert default["group"][3] != default["group"][0]
    # with 5 or more, b is not an outbreak county, so a and c have no
    # outbreak neighbor and every county is its own outbreak
    five = flags(min_cases=5)
    assert five["fips"].to_list() == ["a", "c", "d"]
    assert five["multi_county"].to_list() == [False, False, False]
    assert five["group"].n_unique() == 3


def toy_samples(rng, n=14):
    """Counties 0..n-1 in a row (neighbors 1 apart) with an SD, cases, a
    multi-county label and an outbreak (group) for each."""
    fips = [f"01{i:03d}" for i in range(1, n + 1)]
    adjacency_data = pl.DataFrame(
        {
            "County GEOID": fips[:-1] + fips[1:],
            "Neighbor GEOID": fips[1:] + fips[:-1],
        }
    )
    rows = pl.DataFrame(
        {
            "fips": fips,
            "sd": rng.normal(9, 3, n).round(2),
            "cases": rng.integers(5, 60, n).astype(float),
            "multi": rng.random(n) < 0.45,
            "group": [f"2025_{i // 3}" for i in range(n)],
        }
    )
    return {5: rows}, adjacency_data


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6])
def test_the_interval_excludes_1_exactly_when_the_permutation_p_is_at_most_05(
    seed,
):
    figure = load_script()
    figure._HOMOGENEITY_TESTS.clear()
    samples, adjacency_data = toy_samples(np.random.default_rng(seed))
    model = figure.HOMOGENEITY_BASE
    test = figure.homogeneity_test(
        samples, adjacency_data, model, n_permutations=199
    )
    assert 0 < test["p_perm"] <= 1
    assert test["odds_low"] < test["odds_ratio"] < test["odds_high"]
    assert (test["odds_low"] > 1 or test["odds_high"] < 1) == (
        test["p_perm"] <= 0.05
    )
    # the same test again (a new run) gives the same numbers
    figure._HOMOGENEITY_TESTS.clear()
    again = figure.homogeneity_test(
        samples, adjacency_data, model, n_permutations=199
    )
    np.testing.assert_equal(again, test)  # equal, nan included


def test_a_strong_sd_effect_has_a_small_p_and_an_interval_that_excludes_1():
    figure = load_script()
    figure._HOMOGENEITY_TESTS.clear()
    rng = np.random.default_rng(0)
    samples, adjacency_data = toy_samples(rng, n=40)
    rows = samples[5]
    # the odds of a multi-county outbreak fall with the SD (a logistic curve,
    # not a separation); the SDs are in no spatial order, as a smooth trend
    # along the row would be the spatial structure that the test keeps
    sd = rng.permutation(np.linspace(2, 20, 40))
    rows = rows.with_columns(
        sd=pl.Series(sd),
        multi=pl.Series(rng.random(40) < 1 / (1 + np.exp((sd - 11) / 2.5))),
    )
    test = figure.homogeneity_test(
        {5: rows}, adjacency_data, figure.HOMOGENEITY_BASE, n_permutations=499
    )
    assert not test["separated"]
    assert test["odds_ratio"] < 1
    assert test["odds_high"] < 1
    assert test["p_perm"] <= 0.05


def test_the_interval_needs_at_least_19_permutations():
    figure = load_script()
    figure._HOMOGENEITY_TESTS.clear()
    samples, adjacency_data = toy_samples(np.random.default_rng(1))
    with pytest.raises(ValueError, match="19 permutations"):
        figure.homogeneity_test(
            samples,
            adjacency_data,
            figure.HOMOGENEITY_BASE,
            n_permutations=10,
        )


def test_every_model_name_says_what_it_adds_and_the_base_is_the_first():
    figure = load_script()
    names = list(figure.HOMOGENEITY_MODELS)
    assert names[0] == figure.HOMOGENEITY_BASE
    assert len(names) == len(set(names)) == 11
    assert all(name.startswith("Base") for name in names)
