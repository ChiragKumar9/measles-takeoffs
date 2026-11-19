"""Projection results of the counties and states (outbreak_size.results)."""

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from outbreak_size.results import (
    COUNTY_COLUMNS,
    STATE_COLUMNS,
    county_results,
    join_importation,
    state_results,
)

# --- joining the importation samples ----------------------------------------


def projection_rows():
    """Three counties as the 2026 projection rows: `year` is the year of their
    vaccination data."""
    return pl.DataFrame(
        {
            "fips": ["01", "02", "03"],
            "state": ["A", "A", "B"],
            "county": ["x", "y", "z"],
            "year": [2024, 2024, 2023],
            "type": ["projection"] * 3,
            "pop": [2, 5, 100],
            "estimated_mmr_cov_raw": [90.0, 80.0, 70.0],
        }
    )


def import_rows():
    """What 08 writes: the 2025 validation rows and the 2026 projection rows
    of each county (a row per county, validation year and type)."""
    fips = ["01", "02", "03"] * 2
    years = [2025] * 3 + [2026] * 3
    n = len(fips)
    return pl.DataFrame(
        {
            "fips": fips,
            "year": [2024] * n,
            "validation_year": years,
            "type": ["projection"] * n,
            "importation_probability_mean": [0.1] * 3 + [0.5] * 3,
            "importation_probability_lower": [0.0] * n,
            "importation_probability_upper": [1.0] * n,
            "importation_probability_median": [0.2] * 3 + [0.6] * 3,
        }
    )


def sample_frames(draws_2025=(9.0, 9.0), draws_2026=(0.5, 0.25)):
    """The samples of the rows of import_rows (2 draws each); the 2026 rows
    are told apart from the 2025 rows by their values."""
    rows_2025 = [list(draws_2025)] * 3
    rows_2026 = [[d + i for d in draws_2026] for i in range(3)]
    probabilities = pl.DataFrame(
        {"importation_probability_samples": rows_2025 + rows_2026}
    )
    counts = pl.DataFrame(
        {"expected_importations_samples": [[7.0, 7.0]] * 3 + rows_2026}
    )
    return probabilities, counts


def test_projection_rows_take_the_2026_importation_rows():
    probabilities, counts = sample_frames()
    out = join_importation(
        projection_rows(), import_rows(), probabilities, counts
    )
    # the samples of the 2026 rows (the 2025 rows have 9s and 7s)
    probability = np.vstack(
        out["importation_probability_samples"].to_numpy()  # type: ignore
    )
    count = np.vstack(
        out["expected_importations_samples"].to_numpy()  # type: ignore
    )
    assert probability == pytest.approx(
        np.array([[0.5, 0.25], [1.5, 1.25], [2.5, 2.25]])
    )
    assert count[0] == pytest.approx([0.5, 0.25])
    assert out["importation_probability_mean"].to_list() == [0.5] * 3
    assert out["importation_probability_median"].to_list() == [0.6] * 3


def test_the_year_of_the_vaccination_data_is_left_as_it_is():
    probabilities, counts = sample_frames()
    out = join_importation(
        projection_rows(), import_rows(), probabilities, counts
    )
    assert out["year"].to_list() == [2024, 2024, 2023]
    assert "validation_year" not in out.columns
    assert out["fips"].to_list() == ["01", "02", "03"]  # the order is kept


def test_a_county_without_importation_samples_is_refused():
    probabilities, counts = sample_frames()
    import_df = import_rows().with_columns(
        fips=pl.when(
            (pl.col("fips") == "03") & (pl.col("validation_year") == 2026)
        )
        .then(pl.lit("99"))
        .otherwise(pl.col("fips"))
    )
    with pytest.raises(ValueError, match="Missing importation probability"):
        join_importation(projection_rows(), import_df, probabilities, counts)


def test_a_county_without_count_samples_is_refused():
    probabilities, counts = sample_frames()
    rows = counts["expected_importations_samples"].to_list()
    rows[4] = None  # county 02 in 2026
    counts = pl.DataFrame(
        {
            "expected_importations_samples": pl.Series(
                rows, dtype=pl.List(pl.Float64)
            )
        }
    )
    with pytest.raises(ValueError, match="Missing importation count"):
        join_importation(
            projection_rows(), import_rows(), probabilities, counts
        )


# --- the counts of the counties ----------------------------------------------


def with_samples(probability, counts, projections_pop=(2, 5, 100)):
    """probability and counts: a draws-by-counties array each."""
    return projection_rows().with_columns(
        pop=pl.Series(list(projections_pop)),
        importation_probability_samples=pl.Series(
            np.asarray(probability).T.tolist()
        ),
        expected_importations_samples=pl.Series(np.asarray(counts).T.tolist()),
    )


def constant_draws(per_county, n_draws=4):
    return np.tile(np.asarray(per_county, dtype=float), (n_draws, 1))


def test_projections_are_clipped_at_the_population_and_ranked():
    """The projected cases of the three counties (10, 4 and 3, with
    populations 2, 5 and 100) are clipped to 2, 4 and 3, times an importation
    probability of 0.5, plus 1 imported case: counts 2, 3 and 2.5 (the
    incidences 1, 0.6 and 0.025). The ranks are percentiles: lowest 0,
    highest 100."""
    df = with_samples(constant_draws([0.5] * 3), constant_draws([1.0] * 3))
    projections = jnp.asarray(constant_draws([10.0, 4.0, 3.0]))
    out = county_results(df, projections)
    assert out["mean_unconditioned_projection"].to_list() == pytest.approx(
        [2.0, 3.0, 2.5]
    )
    assert out["unconditioned_count_rank"].to_list() == pytest.approx(
        [0.0, 100.0, 50.0]
    )
    assert out["unconditioned_incidence_rank"].to_list() == pytest.approx(
        [100.0, 50.0, 0.0]
    )


def test_summaries_are_taken_over_the_draws():
    """No importation probability, so the counts are the expected imports:
    county x has 1, 2, 3, 9 (mean 3.75, median 2.5, the 2.5% and 97.5%
    percentiles 1.075 and 3 + 0.925 * 6 = 8.55), the others a constant."""
    counts = np.array(
        [
            [1.0, 10.0, 5.0],
            [2.0, 10.0, 5.0],
            [3.0, 10.0, 5.0],
            [9.0, 10.0, 5.0],
        ]
    )
    df = with_samples(constant_draws([0.0] * 3), counts, (100, 100, 100))
    out = county_results(df, jnp.asarray(constant_draws([1.0, 1.0, 1.0])))
    x = out.row(0, named=True)
    assert x["mean_unconditioned_projection"] == pytest.approx(3.75)
    assert x["median_unconditioned_projection"] == pytest.approx(2.5)
    assert x["lower_unconditioned_projection"] == pytest.approx(
        1.075, rel=1e-5
    )
    assert x["upper_unconditioned_projection"] == pytest.approx(8.55, rel=1e-5)
    assert x["unconditioned_count_replicates"] == pytest.approx([1, 2, 3, 9])
    # the counties are ranked by their mean counts: x (3.75) is below z (5),
    # which is below y (10)
    assert out["unconditioned_count_rank"].to_list() == pytest.approx(
        [0.0, 100.0, 50.0]
    )


def test_a_rank_is_the_percentile_of_the_mean_over_the_draws():
    """County x has 1, 1, 1 and 3 cases against county z's constant 2 and
    county y's 10. Ranking each draw and averaging the percentiles would give
    x 12.5 (it is the middle one in a single draw) and z 37.5; the draws are
    averaged first instead, so x (mean 1.5) is below z (2) and below y (10):
    0, 50 and 100."""
    counts = np.array(
        [
            [1.0, 10.0, 2.0],
            [1.0, 10.0, 2.0],
            [1.0, 10.0, 2.0],
            [3.0, 10.0, 2.0],
        ]
    )
    df = with_samples(constant_draws([0.0] * 3), counts, (100, 100, 100))
    out = county_results(df, jnp.asarray(constant_draws([1.0, 1.0, 1.0])))
    assert out["unconditioned_count_rank"].to_list() == pytest.approx(
        [0.0, 100.0, 50.0]
    )


# --- the states ---------------------------------------------------------------


def results_with_two_states():
    """County x: 1, 2, 3, 9 imports; y: 10; z: 5. State A has x and y."""
    counts = np.array(
        [
            [1.0, 10.0, 5.0],
            [2.0, 10.0, 5.0],
            [3.0, 10.0, 5.0],
            [9.0, 10.0, 5.0],
        ]
    )
    df = with_samples(constant_draws([0.0] * 3), counts, (100, 300, 100))
    return county_results(df, jnp.asarray(constant_draws([1.0] * 3)))


def test_a_states_count_is_the_sum_of_its_counties_in_each_draw():
    out = state_results(results_with_two_states())
    a = out.filter(pl.col("state") == "A").row(0, named=True)
    b = out.filter(pl.col("state") == "B").row(0, named=True)
    # A is x + y: 11, 12, 13, 19; B is z: 5 every draw
    assert np.array(a["unconditioned_count_replicates"]).ravel() == (
        pytest.approx([11, 12, 13, 19])
    )
    assert a["mean_unconditioned_projection"] == pytest.approx(13.75)
    assert a["median_unconditioned_projection"] == pytest.approx(12.5)
    assert a["lower_unconditioned_projection"] == pytest.approx(11.075)
    assert a["upper_unconditioned_projection"] == pytest.approx(18.55)
    assert b["mean_unconditioned_projection"] == pytest.approx(5.0)


def test_states_are_ranked_by_their_counts():
    out = state_results(results_with_two_states())
    ranks = dict(zip(out["state"], out["unconditioned_risk_rank"]))
    assert ranks["B"] == pytest.approx(0.0)  # always the lower
    assert ranks["A"] == pytest.approx(100.0)


def test_a_states_coverage_is_weighted_by_population():
    """A has 90% in a county of 100 people and 80% in one of 300: (90 * 100 +
    80 * 300) / 400 = 82.5."""
    out = state_results(results_with_two_states())
    coverage = dict(zip(out["state"], out["weighted_mmr_coverage"]))
    assert coverage["A"] == pytest.approx(82.5)
    assert coverage["B"] == pytest.approx(70.0)
    assert out.filter(pl.col("state") == "A")[
        "total_population"
    ].to_list() == [400]


def test_the_output_columns_the_script_writes_exist():
    counties = results_with_two_states()
    # the importation summaries come from the join; add them as it does
    counties = counties.with_columns(
        **{
            c: pl.lit(0.0)
            for c in COUNTY_COLUMNS
            if c.startswith("importation_probability_")
        }
    )
    assert counties.select(COUNTY_COLUMNS).columns == COUNTY_COLUMNS
    assert (
        state_results(counties).select(STATE_COLUMNS).columns == STATE_COLUMNS
    )
