"""The 2026 train rows that take their county's earlier school year
(outbreak_size.analysis_dataset.fill_from_earlier_school_year)."""

import polars as pl
import pytest

from outbreak_size.analysis_dataset import (
    fill_from_earlier_school_year,
    merge_outbreaks_vaccinations,
)

REGIONS = ["Some District, State"]


def merged(rows):
    """Merged case rows: (year, state, county, fips, mmr, nme, med, source)."""
    return pl.DataFrame(
        rows,
        schema={
            "year": pl.Int64,
            "state": pl.String,
            "county": pl.String,
            "fips": pl.String,
            "estimated_mmr_cov": pl.Float64,
            "nonmed_exempt_pct": pl.Float64,
            "med_exempt_pct": pl.Float64,
            "coverage_source": pl.String,
            "cases": pl.Int64,
        },
        orient="row",
    )


def vaccines(rows):
    return pl.DataFrame(
        rows,
        schema={
            "year": pl.Int64,
            "state": pl.String,
            "county": pl.String,
            "fips": pl.String,
            "estimated_mmr_cov": pl.Float64,
            "nonmed_exempt_pct": pl.Float64,
            "med_exempt_pct": pl.Float64,
            "coverage_source": pl.String,
        },
        orient="row",
    )


EARLIER = vaccines(
    [
        (2024, "S", "x", "01001", 91.0, 4.0, 0.5, "mmr"),
        (2024, "S", "Fairfax", "51059", 95.0, 2.0, 0.3, "all_vaccines"),
        (2024, "S", "Some District, State", None, 88.0, 6.0, 0.4, "mmr"),
        (2024, "S", "z", "01005", None, 1.0, 0.1, None),
    ]
)


def fill(rows):
    return fill_from_earlier_school_year(merged(rows), EARLIER, 2026, REGIONS)


def test_a_row_without_data_takes_the_earlier_year_of_its_county():
    out = fill([(2026, "S", "x", "01001", None, None, None, None, 7)])
    row = out.row(0, named=True)
    assert (
        row["estimated_mmr_cov"],
        row["nonmed_exempt_pct"],
        row["med_exempt_pct"],
    ) == (
        91.0,
        4.0,
        0.5,
    )
    assert row["coverage_source"] == "mmr"
    assert row["vaccine_from_earlier_year"] is True
    assert row["cases"] == 7


def test_only_the_missing_fields_are_taken():
    out = fill([(2026, "S", "x", "01001", 80.0, None, 0.9, "mmr", 1)])
    row = out.row(0, named=True)
    assert row["estimated_mmr_cov"] == 80.0  # its own value stays
    assert row["med_exempt_pct"] == 0.9
    assert row["nonmed_exempt_pct"] == 4.0  # the earlier year's
    assert row["vaccine_from_earlier_year"] is True


def test_a_row_with_all_three_values_is_left_alone_and_unflagged():
    out = fill([(2026, "S", "x", "01001", 80.0, 3.0, 0.9, "mmr", 1)])
    row = out.row(0, named=True)
    assert (row["estimated_mmr_cov"], row["nonmed_exempt_pct"]) == (80.0, 3.0)
    assert row["vaccine_from_earlier_year"] is False


def test_only_the_outbreak_year_is_filled():
    out = fill(
        [
            (2025, "S", "x", "01001", None, None, None, None, 7),
            (2026, "S", "x", "01001", None, None, None, None, 8),
        ]
    )
    assert out["estimated_mmr_cov"].to_list() == [None, 91.0]
    assert out["vaccine_from_earlier_year"].to_list() == [False, True]


def test_a_place_without_an_earlier_value_stays_missing():
    # z has no coverage in the earlier year either
    out = fill([(2026, "S", "z", "01005", None, None, None, None, 2)])
    row = out.row(0, named=True)
    assert row["estimated_mmr_cov"] is None
    assert row["nonmed_exempt_pct"] == 1.0  # the fields it has are taken
    assert row["vaccine_from_earlier_year"] is True


def test_a_name_shared_with_a_city_gets_its_fips_from_the_earlier_row():
    # Fairfax VA has no fips from the case file, and no row of 2025-26
    out = fill([(2026, "S", "Fairfax", None, None, None, None, None, 2)])
    row = out.row(0, named=True)
    assert row["fips"] == "51059"
    assert row["coverage_source"] == "all_vaccines"


def test_a_region_keeps_a_null_fips():
    out = fill(
        [(2026, "S", "Some District, State", None, None, None, None, None, 4)]
    )
    row = out.row(0, named=True)
    assert row["fips"] is None
    assert row["estimated_mmr_cov"] == 88.0


def test_the_rows_are_not_added_or_reordered_and_the_columns_are_the_same_plus_the_flag():
    rows = [
        (2026, "S", "x", "01001", None, None, None, None, 7),
        (2025, "S", "y", "01002", 90.0, 2.0, 0.2, "mmr", 3),
        (2026, "S", "Fairfax", None, None, None, None, None, 2),
    ]
    base = merged(rows)
    out = fill_from_earlier_school_year(base, EARLIER, 2026, REGIONS)
    assert out.height == base.height
    assert out["county"].to_list() == ["x", "y", "Fairfax"]
    assert out.columns == base.columns + ["vaccine_from_earlier_year"]
    assert out["vaccine_from_earlier_year"].null_count() == 0
    assert "imputation_source" not in out.columns


def test_a_place_with_two_earlier_rows_cannot_be_matched():
    earlier = pl.concat([EARLIER, EARLIER.filter(pl.col("county") == "x")])
    with pytest.raises(ValueError, match="repeat a place"):
        fill_from_earlier_school_year(
            merged([(2026, "S", "x", "01001", None, None, None, None, 7)]),
            earlier,
            2026,
            REGIONS,
        )


def test_a_place_with_two_case_rows_in_the_year_is_refused():
    with pytest.raises(ValueError, match="two rows"):
        fill(
            [
                (2026, "S", "x", "01001", None, None, None, None, 7),
                (2026, "S", "x", "01001", None, None, None, None, 1),
            ]
        )


def test_the_filled_rows_come_out_of_the_real_merge_the_way_the_pipeline_uses_them():
    cases = pl.DataFrame(
        {
            "year": [2026],
            "state": ["S"],
            "county": ["x"],
            "cases": [7],
            "fips": [None],
        },
        schema={
            "year": pl.Int64,
            "state": pl.String,
            "county": pl.String,
            "cases": pl.Int64,
            "fips": pl.String,
        },
    )
    all_vaccines = pl.concat(
        [
            EARLIER,
            vaccines([(2025, "S", "other", "01009", 90.0, 1.0, 0.1, "mmr")]),
        ]
    )
    raw = merge_outbreaks_vaccinations(cases, all_vaccines)
    assert raw["estimated_mmr_cov"].to_list() == [None]  # no 2025 row for x
    out = fill_from_earlier_school_year(raw, all_vaccines, 2026, REGIONS)
    assert out["estimated_mmr_cov"].to_list() == [91.0]
    assert out["fips"].to_list() == ["01001"]


def test_the_earlier_value_is_trended_one_school_year_along_the_slope():
    # x's earlier values (91.0, 4.0, 0.5) move one school year along the
    # place's slopes; the clip keeps them within 0-100
    slopes = pl.DataFrame(
        {
            "state": ["S"],
            "county": ["x"],
            "estimated_mmr_cov": [None],
            "estimated_mmr_cov_slope": [-1.5],
            "nonmed_exempt_pct_slope": [0.5],
            "med_exempt_pct_slope": [-2.0],
        }
    ).drop("estimated_mmr_cov")
    out = fill_from_earlier_school_year(
        merged([(2026, "S", "x", "01001", None, None, None, None, 7)]),
        EARLIER,
        2026,
        REGIONS,
        slopes=slopes,
    )
    row = out.row(0, named=True)
    assert row["estimated_mmr_cov"] == pytest.approx(89.5)
    assert row["nonmed_exempt_pct"] == pytest.approx(4.5)
    assert row["med_exempt_pct"] == 0.0  # 0.5 - 2.0, clipped at 0
    assert row["vaccine_from_earlier_year"]


def test_a_value_the_row_has_is_not_trended_and_a_blank_stays_blank():
    slopes = pl.DataFrame(
        {
            "state": ["S", "S"],
            "county": ["x", "z"],
            "estimated_mmr_cov_slope": [1.0, 1.0],
            "nonmed_exempt_pct_slope": [1.0, 1.0],
            "med_exempt_pct_slope": [1.0, 1.0],
        }
    )
    out = fill_from_earlier_school_year(
        merged(
            [
                # x has its own coverage; z's earlier coverage is blank
                (2026, "S", "x", "01001", 80.0, None, None, None, 1),
                (2026, "S", "z", "01005", None, None, None, None, 1),
            ]
        ),
        EARLIER,
        2026,
        REGIONS,
        slopes=slopes,
    ).sort("county")
    x, z = out.row(0, named=True), out.row(1, named=True)
    assert x["estimated_mmr_cov"] == 80.0
    assert x["nonmed_exempt_pct"] == pytest.approx(5.0)
    assert z["estimated_mmr_cov"] is None
    assert z["nonmed_exempt_pct"] == pytest.approx(2.0)
