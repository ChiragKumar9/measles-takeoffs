"""The trended baselines (outbreak_size.baseline): a county without any value
since 2020 uses its own older values, a series without a slope takes the
state's and then the national one, values are trended to a school year, and
the places' slopes for the 2026 fill."""

import polars as pl
import pytest

from outbreak_size.baseline import (
    as_of_values,
    place_slopes,
    trend_to_year,
)

C = "estimated_mmr_cov"
N = "nonmed_exempt_pct"
M = "med_exempt_pct"


def reported(state, county, fips, rows):
    """rows: (year, coverage, nonmedical, medical)."""
    return [
        dict(
            state=state,
            county=county,
            fips=fips,
            year=y,
            estimated_mmr_cov=c,
            nonmed_exempt_pct=n,
            med_exempt_pct=m,
            county_reported=True,
        )
        for y, c, n, m in rows
    ]


def state_series(state, mmr, nonmed=None, med=None, first=2015):
    n = len(mmr)
    return pl.DataFrame(
        {
            "state": [state] * n,
            "year": list(range(first, first + n)),
            "mmr_cov_state": mmr,
            "nonmed_exempt_pct_state": nonmed or [2.0] * n,
            "med_exempt_pct_state": med or [1.0] * n,
            "series_source": ["cdc"] * n,
        }
    )


def universe(*counties):
    """counties: (fips, county, state)"""
    return pl.DataFrame(
        {
            "fips": [c[0] for c in counties],
            "county": [c[1] for c in counties],
            "state": [c[2] for c in counties],
        }
    )


NO_FILLED = pl.DataFrame(
    schema={
        "fips": pl.String,
        "state": pl.String,
        "county": pl.String,
        "year": pl.Int64,
        "estimated_mmr_cov": pl.Float64,
        "nonmed_exempt_pct": pl.Float64,
        "med_exempt_pct": pl.Float64,
    }
)


def run(rows, series, counties, before_year=2026, bans=()):
    return as_of_values(
        pl.DataFrame(
            rows,
            schema_overrides={
                "estimated_mmr_cov": pl.Float64,
                "nonmed_exempt_pct": pl.Float64,
                "med_exempt_pct": pl.Float64,
            },
        ),
        NO_FILLED,
        series,
        universe(*counties),
        before_year,
        states_with_bans=bans,
    ).sort("fips")


def get(v, fips, col):
    return v.filter(pl.col("fips") == fips)[col][0]


# an anchor county with own points every year 2020-2025, so that the fold of
# 2026 spans six school years and has a national slope of +1
ANCHOR = reported(
    "S",
    "anchor",
    "00099",
    [(y, 80.0 + i, 3.0 + i, 1.0) for i, y in enumerate(range(2020, 2026))],
)


def test_a_county_with_nothing_since_2020_uses_its_own_older_values():
    # M: 2016-2019, +2 a year: the window is the six school years back from
    # 2019 (2014-2019), so all four points count
    older = reported(
        "S",
        "m",
        "00001",
        [(2016, 70.0, 1.0, 0.5), (2017, 72.0, 1.0, 0.5)]
        + [(2018, 74.0, 1.0, 0.5), (2019, 76.0, 1.0, 0.5)],
    )
    v = run(
        ANCHOR + older,
        state_series("S", [90.0] * 11),
        [("00099", "anchor", "S"), ("00001", "m", "S")],
    )
    assert get(v, "00001", C + "_source") == "county_older"
    assert get(v, "00001", C + "_observed") == 76.0
    assert get(v, "00001", C + "_year") == 2019
    assert get(v, "00001", C + "_slope") == pytest.approx(2.0)
    # the state series (90) is not used for it
    assert get(v, "00001", C) == pytest.approx(76.0 + 2.0 * 0)  # own year
    assert get(v, "00001", "reference_year") == 2019


def test_the_older_values_window_has_the_length_of_the_rule_since_2020():
    # in the fold of 2022 the rule since 2020 spans two school years (2020,
    # 2021), so the older values are looked at over two school years (2018
    # and 2019) only: the 2017 value is left out of the slope
    rows = (
        reported(
            "S",
            "m",
            "00001",
            [(2017, 0.0, 1.0, 0.5), (2018, 70.0, 1.0, 0.5)]
            + [(2019, 72.0, 1.0, 0.5)],
        )
        + reported("S", "anchor", "00099", [(2020, 80.0, 3.0, 1.0)])
        + reported("S", "anchor", "00099", [(2021, 81.0, 4.0, 1.0)])
    )
    v = run(
        rows,
        state_series("S", [90.0] * 11),
        [("00099", "anchor", "S"), ("00001", "m", "S")],
        before_year=2022,
    )
    # coverage 0.0 in 2017 is a missing value; the slope is 70 -> 72
    assert get(v, "00001", C + "_slope") == pytest.approx(2.0)
    rows[0]["estimated_mmr_cov"] = (
        50.0  # a real 2017 value, outside the window
    )
    v = run(
        rows,
        state_series("S", [90.0] * 11),
        [("00099", "anchor", "S"), ("00001", "m", "S")],
        before_year=2022,
    )
    assert get(v, "00001", C + "_slope") == pytest.approx(2.0)


def test_a_county_with_values_since_2020_ignores_its_older_ones():
    rows = ANCHOR + reported(
        "S", "k", "00002", [(2018, 10.0, 9.0, 9.0), (2019, 20.0, 9.0, 9.0)]
    )
    rows += reported("S", "k", "00002", [(2024, 91.0, 1.0, 1.0)])
    v = run(
        rows,
        state_series("S", [90.0] * 11),
        [("00099", "anchor", "S"), ("00002", "k", "S")],
    )
    assert get(v, "00002", C + "_source") == "county"
    assert get(v, "00002", C + "_observed") == 91.0


def test_a_single_own_point_takes_the_state_slope_then_the_national_slope():
    one = reported("S", "one", "00003", [(2024, 91.0, 1.0, 1.0)])
    series = state_series(
        "S", [80.0, 81.0, 82.0, 83.0, 84.0, 85.0], first=2020
    )
    v = run(
        ANCHOR + one,
        series,
        [("00099", "anchor", "S"), ("00003", "one", "S")],
    )
    assert get(v, "00003", C + "_slope") == pytest.approx(1.0)  # the state's
    # a state series with one point has no slope: the national one (the
    # anchor county's +1)
    series = state_series("S", [80.0], first=2024)
    v = run(
        ANCHOR + one,
        series,
        [("00099", "anchor", "S"), ("00003", "one", "S")],
    )
    assert get(v, "00003", C + "_slope") == pytest.approx(1.0)
    # no national slope either: 0.0
    v = run(one, series, [("00003", "one", "S")])
    assert get(v, "00003", C + "_slope") == 0.0


def test_a_state_series_that_ends_in_2020_reaches_back_six_school_years():
    # Montana's CDC series: 2015-2020, falling 0.5 a year
    series = state_series(
        "S", [95.0, 94.5, 94.0, 93.5, 93.0, 92.5], first=2015
    )
    v = run(
        ANCHOR,
        series,
        [("00099", "anchor", "S"), ("00004", "none", "S")],
    )
    assert get(v, "00004", C + "_source") == "state_series"
    assert get(v, "00004", C + "_year") == 2020
    assert get(v, "00004", C + "_slope") == pytest.approx(-0.5)


def test_the_national_slope_uses_only_data_before_the_year():
    one = reported("S", "one", "00003", [(2024, 91.0, 1.0, 1.0)])
    series = state_series("S", [80.0], first=2024)
    before = run(
        ANCHOR + one, series, [("00099", "anchor", "S"), ("00003", "one", "S")]
    )
    # a steep series from 2026 on, after the year
    future = reported(
        "S",
        "anchor",
        "00099",
        [(2026, 200.0, 3.0, 1.0), (2027, 300.0, 3.0, 1.0)],
    )
    after = run(
        ANCHOR + future + one,
        series,
        [("00099", "anchor", "S"), ("00003", "one", "S")],
    )
    assert get(after, "00003", C + "_slope") == pytest.approx(
        get(before, "00003", C + "_slope")
    )


def test_a_rising_nonmedical_slope_is_zero_in_a_ban_state():
    series = state_series("S", [90.0] * 11)
    counties = [("00099", "anchor", "S")]
    free = run(ANCHOR, series, counties)
    banned = run(ANCHOR, series, counties, bans=["S"])
    assert get(free, "00099", N + "_slope") == pytest.approx(1.0)
    assert get(banned, "00099", N + "_slope") == 0.0
    # only the nonmedical exemption, and only a rising one
    assert get(banned, "00099", C + "_slope") == pytest.approx(1.0)


def test_trend_to_year_moves_each_field_from_its_own_year_never_backwards():
    rows = reported(
        "S",
        "a",
        "00005",
        [(2022, 90.0, 5.0, 1.0), (2023, 92.0, 4.0, 1.0)],
    ) + reported("S", "a", "00005", [(2024, 94.0, None, None)])
    v = run(rows, state_series("S", [90.0] * 11), [("00005", "a", "S")])
    assert get(v, "00005", C + "_year") == 2024
    out = trend_to_year(v, 2026)
    # coverage: 94 in 2024, slope +2: two more years
    assert get(out, "00005", C) == pytest.approx(98.0)
    # nonmedical: 4.0 observed in 2023, slope -1: three years
    assert get(out, "00005", N) == pytest.approx(1.0)
    assert get(out, "00005", N + "_observed") == 4.0
    assert get(out, "00005", "reference_year") == 2026
    # a target before the value's year leaves it
    same = trend_to_year(v, 2023)
    assert get(same, "00005", C) == get(v, "00005", C)
    assert get(same, "00005", "reference_year") == get(
        v, "00005", "reference_year"
    )
    # clipped at 100 and at 0
    far = trend_to_year(v, 2040)
    assert get(far, "00005", C) == 100.0 and get(far, "00005", N) == 0.0


def test_place_slopes_use_the_place_then_its_state_then_the_national_one():
    rows = []
    # own slope +2 (2020-2023)
    rows += [
        dict(
            state="S",
            county="own",
            year=y,
            estimated_mmr_cov=80.0 + 2 * i,
            nonmed_exempt_pct=2.0,
            med_exempt_pct=1.0,
        )
        for i, y in enumerate(range(2020, 2024))
    ]
    # one point only: the state's slope (+1)
    rows += [
        dict(
            state="S",
            county="one",
            year=2023,
            estimated_mmr_cov=85.0,
            nonmed_exempt_pct=2.0,
            med_exempt_pct=1.0,
        )
    ]
    # one point, in a state with no series: the national slope (the mean of
    # the places with an own slope: +2)
    rows += [
        dict(
            state="T",
            county="lone",
            year=2023,
            estimated_mmr_cov=85.0,
            nonmed_exempt_pct=2.0,
            med_exempt_pct=1.0,
        )
    ]
    series = state_series("S", [70.0, 71.0, 72.0, 73.0], first=2020)
    out = place_slopes(pl.DataFrame(rows), series, 2024)
    slope = {c: s for c, s in zip(out["county"], out[C + "_slope"])}
    assert slope["own"] == pytest.approx(2.0)
    assert slope["one"] == pytest.approx(1.0)
    assert slope["lone"] == pytest.approx(2.0)
    # nothing at or after the year is used
    assert place_slopes(pl.DataFrame(rows), series, 2022)[C + "_slope"][0] == (
        pytest.approx(2.0)
    )
