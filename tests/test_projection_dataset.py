"""Projection datasets of the validation years and scenarios
(outbreak_size.projection_dataset)."""

import polars as pl
import pytest

from outbreak_size.projection_dataset import (
    CENSUS_COLUMNS,
    PROJECTION_COLUMNS,
    TRAIN_COLUMNS,
    assemble_projection_year,
    clean_year_vaccination_data,
    require_consistent_places,
    require_same_census_values,
    scenario_projection_rows,
)

# --- one year's vaccination data ---------------------------------------------


def vaccination_rows(rows):
    """rows: (fips, county, estimated_mmr_cov, med_exempt, nonmed_exempt)"""
    return pl.DataFrame(
        rows,
        schema={
            "fips": pl.String,
            "county": pl.String,
            "estimated_mmr_cov": pl.Float64,
            "med_exempt_pct": pl.Float64,
            "nonmed_exempt_pct": pl.Float64,
        },
        orient="row",
    )


def test_county_names_are_spelled_as_in_the_training_data():
    df = vaccination_rows(
        [
            ("17109", "Mcdonough", 90.0, 1.0, 1.0),
            ("17043", "Dupage County", 90.0, 1.0, 1.0),
            ("48307", "Mcculloch", 90.0, 1.0, 1.0),
            ("01001", "Autauga", 90.0, 1.0, 1.0),
        ]
    )
    assert clean_year_vaccination_data(df)["county"].to_list() == [
        "McDonough",
        "DuPage County",
        "McCulloch",
        "Autauga",
    ]


@pytest.mark.parametrize(
    "bad_row",
    [
        ("01001", "a", None, 1.0, 1.0),  # no coverage
        ("01001", "a", 90.0, None, 1.0),  # no medical exemption
        ("01001", "a", 90.0, 1.0, None),  # no nonmedical exemption
        ("1001", "a", 90.0, 1.0, 1.0),  # a fips of 4 characters
        ("010011", "a", 90.0, 1.0, 1.0),  # more than a county's 5
        (None, "a", 90.0, 1.0, 1.0),  # no fips (a region)
    ],
)
def test_places_that_cannot_be_projected_are_left_out(bad_row):
    df = vaccination_rows([("01003", "good", 90.0, 1.0, 1.0), bad_row])
    assert clean_year_vaccination_data(df)["county"].to_list() == ["good"]


# --- the rows of one validation year -----------------------------------------


def census_row(fips, county, **overrides):
    row: dict = {c: 1.0 for c in PROJECTION_COLUMNS}
    row.update(
        fips=fips, state="A", county=county, year=2021, estimated_mmr_cov=90.0
    )
    row.update(overrides)
    return row


def projection_input():
    """Three counties with vaccination and census data; the third has a NaN
    census value (too few people for the census)."""
    return pl.DataFrame(
        [
            census_row("01001", "one"),
            census_row("01003", "two"),
            census_row("01005", "three", fraction_white=float("nan")),
        ]
    )


def analysis_rows():
    """County one in 2020 and 2021 with observed cases (5 in 2021), and county
    four in 2021 with a missing value."""
    rows = []
    for fips, county, year, cases in (
        ("01001", "one", 2020, 1),
        ("01001", "one", 2021, 5),
        ("01007", "four", 2021, 2),
    ):
        row: dict = {c: 1.0 for c in TRAIN_COLUMNS}
        row.update(
            fips=fips,
            state="A",
            county=county,
            year=year,
            estimated_mmr_cov=80.0,
            cases_outbreak=cases,
            cases=cases,
            geoid=county + fips,
        )
        rows.append(row)
    rows[2]["log_income"] = None
    return pl.DataFrame(rows)


def assembled():
    return assemble_projection_year(projection_input(), analysis_rows(), 2021)


def test_an_observed_county_appears_as_a_train_row_and_a_projection_row():
    out, *_ = assembled()
    one = out.filter(pl.col("fips") == "01001")
    assert sorted(one["type"].to_list()) == ["projection", "train"]
    train = one.filter(pl.col("type") == "train").row(0, named=True)
    projection = one.filter(pl.col("type") == "projection").row(0, named=True)
    assert (train["in_train"], projection["in_train"]) == (1, 0)
    # the train row has the observed cases and the projection row has
    # placeholders, which are not used
    assert (train["cases_outbreak"], projection["cases_outbreak"]) == (5, 0)
    assert projection["cases"] == 0
    # the train row comes from the analysis dataset (coverage 80), the
    # projection row from the year's vaccination data (coverage 90)
    assert (train["estimated_mmr_cov"], projection["estimated_mmr_cov"]) == (
        80.0,
        90.0,
    )


def test_only_the_train_rows_of_the_year_are_used():
    out, *_ = assembled()
    train = out.filter(pl.col("type") == "train")
    assert train["year"].to_list() == [2021]  # not county one's 2020 row


def test_the_raw_coverage_starts_as_the_coverage():
    out, *_ = assembled()
    assert out["estimated_mmr_cov_raw"].to_list() == (
        out["estimated_mmr_cov"].to_list()
    )


def test_a_projection_row_gets_a_geoid_from_county_and_fips():
    out, *_ = assembled()
    two = out.filter(pl.col("fips") == "01003")
    assert two["geoid"].to_list() == ["two01003"]
    # a train row keeps the geoid of the analysis dataset
    one_train = out.filter(
        (pl.col("fips") == "01001") & (pl.col("type") == "train")
    )
    assert one_train["geoid"].to_list() == ["one01001"]


def test_rows_with_missing_values_are_dropped_and_reported():
    out, dropped, dropped_train, n_projection, n_train = assembled()
    # county three (NaN census value) is not projected; county four (a null in
    # its observed row) is not a train row
    assert dropped == ["01005"]
    assert sorted(out["fips"].unique().to_list()) == ["01001", "01003"]
    assert dropped_train.rows() == [("01007", "A", "four")]
    assert (n_projection, n_train) == (3, 2)
    assert out.null_count().sum_horizontal().item() == 0


def test_the_train_and_projection_rows_have_the_same_columns():
    out, *_ = assembled()
    assert set(out.columns) == set(TRAIN_COLUMNS) | {
        "in_train",
        "type",
        "estimated_mmr_cov_raw",
    }


# --- the rows of a future scenario -------------------------------------------


def base_rows():
    return pl.DataFrame(
        {
            "fips": ["02", "01", "03"],  # not in order of fips
            "estimated_mmr_cov": [90.0, 91.0, 92.0],
            "estimated_mmr_cov_raw": [90.0, 91.0, 92.0],
            "nonmed_exempt_pct": [2.0, 3.0, 4.0],
            "med_exempt_pct": [1.0, 1.0, 1.0],
            "year": [2026, 2026, 2026],
            "pop": [100, 200, 300],
        }
    )


def projections():
    return pl.DataFrame(
        {
            "fips": ["01", "02", "01", "02", "03", "01"],
            "scenario": ["s1", "s1", "s1", "s1", "s1", "s2"],
            "year": [2027, 2027, 2028, 2028, 2028, 2027],
            "projected_coverage": [80.0, 81.0, 70.0, 71.0, 72.0, 60.0],
            "projected_nonmed_exempt": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "projected_med_exempt": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )


def test_a_scenario_year_takes_that_scenarios_vaccination_values():
    out, lost = scenario_projection_rows(
        base_rows(), projections(), "s1", 2027
    )
    # the rows keep the order of the base rows
    assert out["fips"].to_list() == ["02", "01"]
    assert out["estimated_mmr_cov"].to_list() == [81.0, 80.0]
    assert out["estimated_mmr_cov_raw"].to_list() == [81.0, 80.0]
    assert out["nonmed_exempt_pct"].to_list() == [6.0, 5.0]
    assert out["med_exempt_pct"].to_list() == [0.2, 0.1]
    assert out["year"].to_list() == [2027, 2027]
    assert out["scenario"].to_list() == ["s1", "s1"]
    # the census and demographic data are the base rows'
    assert out["pop"].to_list() == [100, 200]
    assert not any(c.startswith("projected_") for c in out.columns)


def test_a_county_without_a_projection_is_left_out_and_reported():
    out, lost = scenario_projection_rows(
        base_rows(), projections(), "s1", 2027
    )
    assert lost == {"03"}
    assert "03" not in out["fips"].to_list()
    out, lost = scenario_projection_rows(
        base_rows(), projections(), "s1", 2028
    )
    assert lost == set() and out.height == 3


def test_other_scenarios_and_years_are_not_mixed_in():
    out, _ = scenario_projection_rows(base_rows(), projections(), "s2", 2027)
    assert out["fips"].to_list() == ["01"]
    assert out["estimated_mmr_cov"].to_list() == [60.0]
    out, _ = scenario_projection_rows(base_rows(), projections(), "s1", 2028)
    assert out["estimated_mmr_cov"].to_list() == [71.0, 70.0, 72.0]


# --- the census values of a place that is a train and a projection row -------


def census_rows(rows):
    """rows: (fips, type, pop, log_income)"""
    base = {c: 0.5 for c in CENSUS_COLUMNS}
    return pl.DataFrame(
        [
            {**base, "fips": f, "type": t, "pop": p, "log_income": i}
            for f, t, p, i in rows
        ]
    )


def test_the_same_census_values_in_the_train_and_projection_rows_pass():
    df = census_rows(
        [
            ("01001", "train", 1000.0, 4.5),
            ("01001", "projection", 1000.0, 4.5),
            ("01003", "projection", 7.0, 4.0),
        ]
    )
    require_same_census_values(df)


def test_census_values_of_different_years_in_the_two_rows_raise():
    df = census_rows(
        [("01001", "train", 1000.0, 4.5), ("01001", "projection", 1010.0, 4.5)]
    )
    with pytest.raises(ValueError, match="pop"):
        require_same_census_values(df)


def test_a_region_labelled_county_takes_its_name_from_the_analysis_dataset():
    # the year file labels a county with its region's name ("Panhandle"); the
    # county's train row (any year) has its own name, and the geoids must agree
    projection = pl.DataFrame(
        [census_row("01001", "Region"), census_row("01003", "Region")]
    )
    out, *_ = assemble_projection_year(projection, analysis_rows(), 2021)
    rows = out.filter(pl.col("type") == "projection")
    assert dict(zip(rows["fips"], rows["geoid"])) == {
        "01001": "one01001",  # in the analysis dataset: its name
        "01003": "Region01003",  # not in it: the label stays
    }
    one = out.filter(pl.col("fips") == "01001")
    assert one["geoid"].n_unique() == 1


def places(rows):
    return pl.DataFrame(rows, schema=["fips", "geoid"], orient="row")


def test_consistent_places_pass_and_a_county_with_two_geoids_raises():
    analysis = places([("01001", "one01001")])
    require_consistent_places(
        places([("01001", "one01001"), ("01003", "two01003")]),
        analysis,
        "toy",
    )
    with pytest.raises(ValueError, match="two geoids"):
        require_consistent_places(
            places([("01001", "one01001"), ("01001", "Region01001")]),
            analysis,
            "toy",
        )


def test_a_geoid_of_two_fips_and_a_geoid_other_than_the_analysis_one_raise():
    analysis = places([("01001", "one01001")])
    with pytest.raises(ValueError, match="two fips"):
        require_consistent_places(
            places([("01001", "x"), ("01003", "x")]), analysis, "toy"
        )
    with pytest.raises(ValueError, match="other than that of the analysis"):
        require_consistent_places(
            places([("01001", "Region01001")]), analysis, "toy"
        )
    # region and state places (fips that are not five characters) are not counties
    require_consistent_places(
        places([("North, Alabama", "North, Alabama"), ("10", "Delaware")]),
        analysis,
        "toy",
    )
