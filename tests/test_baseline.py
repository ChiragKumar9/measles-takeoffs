"""The vaccination baseline of each county (outbreak_size.baseline)."""

import numpy as np
import polars as pl
import pytest

from outbreak_size.baseline import as_of_values, panel_c_coverage, slope_by

C = "estimated_mmr_cov"
N = "nonmed_exempt_pct"
M = "med_exempt_pct"


def reported(state, county, fips, rows, county_reported=True):
    """rows: (year, coverage, nonmedical, medical) as reported."""
    return [
        dict(
            state=state,
            county=county,
            fips=fips,
            year=y,
            estimated_mmr_cov=c,
            nonmed_exempt_pct=n,
            med_exempt_pct=md,
            county_reported=county_reported,
        )
        for y, c, n, md in rows
    ]


@pytest.fixture(scope="module")
def inputs():
    reported_rows = pl.DataFrame(
        # A: its own values every year; the exemption is missing from 2022
        reported(
            "S1",
            "A",
            "00001",
            [
                (2020, 90.0, 2.0, 1.0),
                (2021, 91.0, 3.0, 1.0),
                (2022, 92.0, None, 1.0),
                (2023, 93.0, None, 1.0),
            ],
        )
        # D: one coverage point and nothing else
        + reported("S1", "D", "00004", [(2024, 95.0, None, None)])
        # E: a coverage of 0 is a missing value
        + reported(
            "S1", "E", "00005", [(2023, 90.0, 4.0, 1.0), (2024, 0.0, 5.0, 1.0)]
        )
        # F: county data only before 2020
        + reported(
            "S1",
            "F",
            "00006",
            [(2018, 70.0, 7.0, 1.0), (2019, 71.0, 7.0, 1.0)],
        )
        # G: a high exemption that grows past 100 when brought forward
        + reported(
            "S1",
            "G",
            "00007",
            [
                (2020, 90.0, 94.0, 1.0),
                (2021, 90.0, 99.0, 1.0),
                (2022, 90.0, None, 1.0),
                (2023, 90.0, None, 1.0),
            ],
        ),
        schema_overrides={
            "nonmed_exempt_pct": pl.Float64,
            "med_exempt_pct": pl.Float64,
        },
    )
    filled = []
    # B: a state row copied to the county, 2019-2024
    for y, v in zip(range(2019, 2025), [80, 81, 82, 83, 84, 85]):
        filled.append(
            dict(
                fips="00002",
                state="S1",
                county="State",
                year=y,
                estimated_mmr_cov=float(v),
                nonmed_exempt_pct=str(v / 10),
                med_exempt_pct="1.0",
            )
        )
    filled.append(
        dict(
            fips="00006",
            state="S1",
            county="F",
            year=2023,
            estimated_mmr_cov=60.0,
            nonmed_exempt_pct="6.0",
            med_exempt_pct="1.0",
        )
    )
    # J: real values, but only before 2020
    for y, v in zip(range(2015, 2020), [60, 61, 62, 63, 64]):
        filled.append(
            dict(
                fips="00010",
                state="S3",
                county="J",
                year=y,
                estimated_mmr_cov=float(v),
                nonmed_exempt_pct="5.0",
                med_exempt_pct="1.0",
            )
        )
    # I: a jump before 2020, then +2 a year
    for y, v in zip(range(2015, 2025), [50] * 5 + [70, 72, 74, 76, 78]):
        filled.append(
            dict(
                fips="00009",
                state="S1",
                county="I",
                year=y,
                estimated_mmr_cov=float(v),
                nonmed_exempt_pct="1.0",
                med_exempt_pct="1.0",
            )
        )
    s2 = pl.DataFrame(
        {
            "state": ["S2"] * 3 + ["S1"],
            "year": [2022, 2023, 2024, 2024],
            "mmr_cov_state": [70.0, 72.0, 74.0, 88.0],
            "nonmed_exempt_pct_state": [3.0, 3.5, 4.0, 6.0],
            "med_exempt_pct_state": [1.0, 1.0, 1.0, 2.0],
            "series_source": ["cdc"] * 4,
        }
    )
    s3_rows = [("cdc", y, 60.0 + i) for i, y in enumerate(range(2019, 2025))]
    s3 = pl.DataFrame(
        {
            "state": ["S3"] * len(s3_rows),
            "year": [r[1] for r in s3_rows],
            "mmr_cov_state": [r[2] for r in s3_rows],
            "nonmed_exempt_pct_state": [1.0] * len(s3_rows),
            "med_exempt_pct_state": [1.0] * len(s3_rows),
            "series_source": [r[0] for r in s3_rows],
        }
    )
    universe = pl.DataFrame(
        {
            "fips": [f"{i:05d}" for i in range(1, 11)],
            "county": list("abcdefghij"),
            "state": [
                "S1",
                "S1",
                "S2",
                "S1",
                "S1",
                "S1",
                "S1",
                "S3",
                "S1",
                "S3",
            ],
        }
    )
    return reported_rows, pl.DataFrame(filled), pl.concat([s2, s3]), universe


def values(inputs, before_year):
    reported_rows, filled, state_series, universe = inputs
    return as_of_values(
        reported_rows, filled, state_series, universe, before_year
    ).sort("fips")


@pytest.fixture(scope="module")
def v2026(inputs):
    return values(inputs, 2026)


def get(v, fips, col):
    return v.filter(pl.col("fips") == fips)[col][0]


def test_every_county_of_the_universe_has_every_field(v2026):
    assert v2026.height == 10
    assert all(v2026[c].null_count() == 0 for c in (C, N, M))


def test_a_county_with_its_own_values_uses_its_latest_and_their_slope(v2026):
    assert get(v2026, "00001", C) == 93.0
    assert get(v2026, "00001", C + "_year") == 2023
    assert get(v2026, "00001", C + "_source") == "county"
    assert get(v2026, "00001", C + "_slope") == pytest.approx(1.0)


def test_fields_are_brought_to_the_latest_year_any_of_them_has(v2026):
    # A's exemption is last observed in 2021 (3.0), slope +1; the reference
    # year is 2023, so it is brought forward two years
    assert get(v2026, "00001", N + "_observed") == 3.0
    assert get(v2026, "00001", N + "_year") == 2021
    assert get(v2026, "00001", "reference_year") == 2023
    assert get(v2026, "00001", N) == pytest.approx(5.0)
    assert get(v2026, "00001", C) == 93.0 and get(v2026, "00001", M) == 1.0


def test_brought_forward_values_are_clipped_at_100(v2026):
    assert get(v2026, "00007", N + "_observed") == 99.0
    assert get(v2026, "00007", N) == 100.0


def test_equal_points_have_a_slope_of_zero(v2026):
    assert get(v2026, "00001", M + "_slope") == 0.0


def test_a_county_without_reported_rows_uses_the_filled_table_from_2020_on(
    v2026,
):
    assert get(v2026, "00002", C) == 85.0
    assert get(v2026, "00002", C + "_year") == 2024
    assert get(v2026, "00002", C + "_source") == "region_or_state_rows"
    assert get(v2026, "00002", C + "_slope") == pytest.approx(1.0)
    # string-typed exemptions are read as numbers, and the data's name stays
    assert get(v2026, "00002", N) == pytest.approx(8.5)
    assert get(v2026, "00002", "county") == "State"


def test_a_county_missing_from_the_data_gets_its_states_series(v2026):
    assert get(v2026, "00003", C) == 74.0
    assert get(v2026, "00003", C + "_year") == 2024
    assert get(v2026, "00003", C + "_source") == "state_series"
    assert get(v2026, "00003", C + "_slope") == pytest.approx(2.0)
    assert get(v2026, "00003", "county") == "c"  # the shapefile's name


def test_a_single_point_takes_the_slope_of_the_state_else_the_national_one(
    v2026,
):
    # D has one point; its state S1 has one point too (2024), so no slope:
    # the national average, the mean of the slopes of the counties with two
    # own points since 2020 (A: +1, G: 0)
    assert get(v2026, "00004", C) == 95.0
    assert get(v2026, "00004", C + "_slope") == pytest.approx(0.5)
    assert get(v2026, "00004", N) == 6.0  # no exemption anywhere: its state
    assert get(v2026, "00004", N + "_source") == "state_series"


def test_a_coverage_of_zero_is_missing_but_an_exemption_of_zero_is_not(v2026):
    # one coverage point (2023), so the national slope (+0.5) brings it to 2024
    assert get(v2026, "00005", C) == pytest.approx(90.5)
    assert get(v2026, "00005", C + "_observed") == 90.0
    assert get(v2026, "00005", C + "_year") == 2023
    assert get(v2026, "00005", N) == 5.0  # used in the year the coverage was 0
    assert get(v2026, "00005", N + "_year") == 2024
    assert get(v2026, "00005", "reference_year") == 2024


def test_county_data_from_before_2020_is_not_used(inputs, v2026):
    assert get(v2026, "00006", C) == 60.0
    assert get(v2026, "00006", C + "_source") == "region_or_state_rows"
    # real values, all before 2020, but in the filled table only: the state
    # series is used instead (the rule for older county values is for
    # reported rows)
    assert get(v2026, "00010", C) == 65.0
    assert get(v2026, "00010", C + "_source") == "state_series"
    assert get(v2026, "00010", "county") == "J"


def test_before_year_is_exclusive(inputs):
    v = values(inputs, 2022)
    assert get(v, "00001", C) == 91.0 and get(v, "00001", C + "_year") == 2021
    assert get(v, "00002", C) == 82.0


def test_the_state_series_is_the_cdc_s_and_its_slope_is_since_2020(
    v2026, inputs
):
    # S3: 60 in 2019, then +1 a year; the slope uses 2020 on
    assert get(v2026, "00008", C) == 65.0
    assert get(v2026, "00008", C + "_year") == 2024
    assert get(v2026, "00008", C + "_slope") == pytest.approx(1.0)
    # as of 2021 the rule since 2020 spans one school year (2020), so the
    # series reaches back over that one year only: no slope, and no national
    # slope either (no county has two points): 0.0
    v2021 = values(inputs, 2021)
    assert get(v2021, "00008", C) == 61.0
    assert get(v2021, "00008", C + "_slope") == 0.0


def test_the_slope_of_a_filled_series_uses_2020_on_only(v2026):
    # +2 a year since 2020, not the long-run slope with the jump before 2020
    assert get(v2026, "00009", C + "_slope") == pytest.approx(2.0)


def test_slope_by_equals_scipy():
    x = np.array([2019, 2020, 2021, 2022, 2023, 2024, 2025])
    y = np.array([3.0, 3.4, 3.1, 4.0, 4.2, 4.0, 5.1])
    points = pl.DataFrame({"fips": ["z"] * 7, "year": x, "v": y}).filter(
        pl.col("year") >= 2020
    )
    # ordinary least squares slope of the points from 2020 on
    assert slope_by(points, "fips", "v")["slope"][0] == pytest.approx(
        np.polyfit(x[1:], y[1:], 1)[0]
    )


def test_slope_by_is_zero_for_a_single_year():
    points = pl.DataFrame({"fips": ["z"], "year": [2024], "v": [3.0]})
    assert slope_by(points, "fips", "v")["slope"][0] == 0.0


# --- the coverage map (panel c) ---------------------------------------------


def coverage_rows(rows):
    return pl.DataFrame(
        [
            dict(
                fips=f,
                state="S",
                county="c" + f,
                year=y,
                estimated_mmr_cov=cov,
                county_reported=cr,
            )
            for f, y, cov, cr in rows
        ],
        schema={
            "fips": pl.String,
            "state": pl.String,
            "county": pl.String,
            "year": pl.Int64,
            "estimated_mmr_cov": pl.Float64,
            "county_reported": pl.Boolean,
        },
    )


def test_panel_c_takes_2024_then_2023_and_leaves_the_rest_grey():
    rep = coverage_rows(
        [
            ("00001", 2024, 90.0, True),  # county's own 2024 value
            ("00002", 2024, 80.0, False),  # a region value copied to it
            ("00003", 2023, 70.0, True),  # nothing in 2024: carried forward
            ("00004", 2023, 60.0, False),
        ]
    )
    out = panel_c_coverage(rep).sort("fips")
    assert out["category"].to_list() == [
        "county_2024",
        "region_or_state_2024",
        "county_2023_carried_forward",
        "region_or_state_2023_carried_forward",
    ]
    assert out["estimated_mmr_cov"].to_list() == [90, 80, 70, 60]
    assert out["source_year"].to_list() == [2024, 2024, 2023, 2023]


def test_panel_c_ignores_zero_and_nan_coverage_and_undrawable_fips():
    rep = coverage_rows(
        [
            ("00001", 2024, 0.0, True),
            ("00001", 2023, 88.0, True),
            ("00002", 2024, float("nan"), True),
            ("123", 2024, 90.0, True),  # a school district, not drawable
            ("12", 2024, 91.0, True),  # a state value: not drawn
        ]
    )
    out = panel_c_coverage(rep).sort("fips")
    assert out["fips"].to_list() == ["00001"]
    assert out["category"].to_list() == ["county_2023_carried_forward"]


def test_panel_c_has_one_row_per_fips_and_prefers_the_counties_own_row():
    rep = coverage_rows(
        [
            ("00001", 2024, 70.0, False),  # a copied region value first
            ("00001", 2024, 90.0, True),  # the county's own value
        ]
    )
    out = panel_c_coverage(rep)
    assert out.height == 1
    assert out["estimated_mmr_cov"][0] == 90.0
    assert out["category"][0] == "county_2024"
