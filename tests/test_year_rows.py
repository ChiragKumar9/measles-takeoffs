"""The places of one validation year (outbreak_size.year_rows)."""

import numpy as np
import polars as pl
import pytest

from outbreak_size.year_rows import validation_year_rows

N_DRAWS = 6


def import_row(fips, kind, mean_imports, year=2026):
    counts = np.full(N_DRAWS, mean_imports)
    # plain lists: polars turns numpy arrays inside row dicts into nulls
    return dict(
        fips=fips,
        validation_year=year,
        type=kind,
        expected_importations_samples=counts.tolist(),
        importation_probability_samples=(1 - np.exp(-counts)).tolist(),
    )


@pytest.fixture
def files(tmp_path):
    """A projection dataset of 2026 with an observed county (A: a train row
    and its projection twin), a county without cases (B) and a county (C)
    that is a member of a region (R) that has its own row."""
    rows = [
        # fips, type, county, cases_outbreak
        ("01001", "train", "a", 4),
        ("01001", "projection", "a", 0),
        ("01003", "projection", "b", 0),
        ("01005", "projection", "c", 0),
        ("Region R", "train", "Region R", 2),
    ]
    pl.DataFrame(
        {
            "fips": [r[0] for r in rows],
            "type": [r[1] for r in rows],
            "county": [r[2] for r in rows],
            "state": ["S"] * len(rows),
            "year": [2025] * len(rows),
            "cases_outbreak": [r[3] for r in rows],
            "pop": [1000] * len(rows),
            "x": [1.0] * len(rows),
        }
    ).write_csv(tmp_path / "projection_dataset_2026.csv")
    import_df = pl.DataFrame(
        [import_row(r[0], r[1], 0.5) for r in rows],
        schema_overrides={
            "importation_probability_samples": pl.List(pl.Float64),
            "expected_importations_samples": pl.List(pl.Float64),
        },
    )
    analysis = pl.DataFrame(
        {
            "state_county_year": ["01001_2026", "S_Region R_2026"],
            "cases_outbreak": [4, 2],
            "cases": [6, 3],
        }
    )
    return tmp_path, import_df, analysis


def test_the_twin_and_the_region_member_are_dropped_and_the_cases_come_from_the_key(
    files, monkeypatch
):
    tmp_path, import_df, analysis = files
    monkeypatch.setattr(
        "outbreak_size.year_rows.higher_level_regions",
        lambda: {"Region R": ["01005"]},
    )
    places, keep, draws = validation_year_rows(
        2026, str(tmp_path), import_df, analysis, seed=1, covariates=["x"]
    )
    # the projection twin of A and the member C of region R are dropped
    assert keep.tolist() == [True, False, True, False, True]
    assert places["fips"].to_list() == ["01001", "01003", "Region R"]
    # the observed places are train rows with the cases of their key (a place
    # without a county fips by name), the others have none
    assert places["type"].to_list() == ["train", "projection", "train"]
    assert places["cases"].to_list() == [6, 0, 3]
    # (a string column, as in the original table script: callers use int())
    assert [int(c) for c in places["cases_outbreak"]] == [4, 0, 2]
    assert places["year"].to_list() == [2026, 2025, 2026]
    assert "x" in places.columns
    # imports are drawn for the three places, the same draws for every model
    assert draws.shape == (N_DRAWS, 3)


def test_the_imports_drawn_are_reproducible_from_the_seed_and_the_year(
    files, monkeypatch
):
    tmp_path, import_df, analysis = files
    monkeypatch.setattr(
        "outbreak_size.year_rows.higher_level_regions", lambda: {}
    )
    first = validation_year_rows(
        2026, str(tmp_path), import_df, analysis, seed=7, covariates=["x"]
    )[2]
    again = validation_year_rows(
        2026, str(tmp_path), import_df, analysis, seed=7, covariates=["x"]
    )[2]
    other_seed = validation_year_rows(
        2026, str(tmp_path), import_df, analysis, seed=8, covariates=["x"]
    )[2]
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other_seed)


def test_a_probability_that_is_not_one_minus_exp_of_the_expected_imports_is_refused(
    files, monkeypatch
):
    tmp_path, import_df, analysis = files
    monkeypatch.setattr(
        "outbreak_size.year_rows.higher_level_regions", lambda: {}
    )
    wrong = import_df.with_columns(
        importation_probability_samples=pl.Series(
            [np.full(N_DRAWS, 0.9)] * import_df.height,
            dtype=pl.List(pl.Float64),
        )
    )
    with pytest.raises(AssertionError):
        validation_year_rows(
            2026, str(tmp_path), wrong, analysis, seed=1, covariates=["x"]
        )


def test_the_observed_county_years_are_keyed_like_the_validation_places():
    import polars as pl

    from outbreak_size.year_rows import observed_county_years

    analysis = pl.DataFrame(
        {
            "fips": ["01001", "region"],
            "state": ["A", "B"],
            "county": ["x", "y"],
            "year": [2021, 2022],
            "cases": [3, 5],
        }
    )
    missing = pl.DataFrame(
        {
            "fips": ["02002", "region"],
            "state": ["C", "B"],
            "county": ["z", "y"],
            "year": [2020, 2023],  # 2020 is before the validation years
            "cases": [9, 2],
        }
    )
    out = observed_county_years(analysis, missing)
    assert out["state_county_year"].to_list() == [
        "01001_2021",  # a county fips and its year
        "B_y_2022",  # a place without a county fips: state, county, year
        "B_y_2023",
    ]
    assert out["cases"].to_list() == [3, 5, 2]
