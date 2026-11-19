"""State series and FIPS settling of the vaccination data (outbreak_size.vaccination_wrangling)."""

import geopandas as gpd
import polars as pl
import pytest
from shapely.geometry import Point

from outbreak_size import vaccination_wrangling as vw

CDC_FILE = (
    "Vaccination_Coverage_and_Exemptions_among_Kindergartners_20260526.csv"
)


def cdc_rows(rows):
    """rows: (geography type, geography, school year, vaccine, dose,
    estimate)."""
    return pl.DataFrame(
        [
            {
                "Geography Type": t,
                "Geography": g,
                "School Year": y,
                "Vaccine/Exemption": v,
                "Dose": d,
                "Estimate (%)": e,
            }
            for t, g, y, v, d, e in rows
        ]
    )


@pytest.fixture
def cdc(tmp_path):
    rows = [
        # X: has a Non-Medical row: it is used, whatever Any says
        ("States", "X", "2020-21", "MMR", "2 doses", "94.0"),
        (
            "States",
            "X",
            "2020-21",
            "Exemption",
            "Non-Medical Exemption",
            "3.0",
        ),
        ("States", "X", "2020-21", "Exemption", "Medical Exemption", "1.0"),
        ("States", "X", "2020-21", "Exemption", "Any Exemption", "9.0"),
        # Y: only Any: Any minus the medical exemption
        ("States", "Y", "2020-21", "MMR", "2 doses", "90.0"),
        ("States", "Y", "2020-21", "Exemption", "Any Exemption", "5.0"),
        ("States", "Y", "2020-21", "Exemption", "Medical Exemption", "1.5"),
        # Z: Any below the medical exemption: floored at 0
        ("States", "Z", "2020-21", "Exemption", "Any Exemption", "1.0"),
        ("States", "Z", "2020-21", "Exemption", "Medical Exemption", "2.0"),
        # W: only Any, no medical exemption: Any itself
        ("States", "W", "2020-21", "Exemption", "Any Exemption", "4.0"),
        # a second medical row of one state-year is not a second state row
        (
            "States",
            "X",
            "2020-21",
            "Exemption",
            "Medical Exemption (a)",
            "7.0",
        ),
        # not reported, and not a state
        ("States", "X", "2021-22", "MMR", "2 doses", "NR"),
        ("Counties", "X County", "2020-21", "MMR", "2 doses", "50.0"),
    ]
    path = tmp_path / "cdc.csv"
    cdc_rows(rows).write_csv(path)
    return vw.cdc_state_level_data(str(path))


def row(df, state):
    return df.filter(pl.col("state") == state).row(0, named=True)


def test_cdc_non_medical_rows_are_used_when_they_exist(cdc):
    x = row(cdc, "X")
    assert x["nonmed_exempt_pct_state"] == 3.0
    assert x["nonmed_source"] == "Non-Medical Exemption"
    assert x["mmr_cov_state"] == 94.0
    assert x["year"] == 2020  # the first year of the school year 2020-21


def test_cdc_any_exemption_is_reduced_by_the_medical_exemption(cdc):
    y = row(cdc, "Y")
    assert y["nonmed_exempt_pct_state"] == pytest.approx(3.5)
    assert y["nonmed_source"] == "Any Exemption"
    assert y["med_exempt_pct_state"] == 1.5


def test_cdc_any_minus_medical_is_floored_at_zero(cdc):
    assert row(cdc, "Z")["nonmed_exempt_pct_state"] == 0.0


def test_cdc_any_exemption_alone_is_used_as_it_is(cdc):
    w = row(cdc, "W")
    assert w["nonmed_exempt_pct_state"] == 4.0
    assert w["med_exempt_pct_state"] is None


def test_cdc_has_one_row_per_state_and_year_and_only_states(cdc):
    assert cdc.select("state", "year").is_duplicated().sum() == 0
    assert "X County" not in cdc["state"].to_list()
    assert 2021 not in cdc["year"].to_list()  # the only 2021 row was "NR"


def test_state_level_series_is_the_cdc_series_only(tmp_path):
    rows = [("States", "X", "2020-21", "MMR", "2 doses", "94.0")]
    cdc_rows(rows).write_csv(tmp_path / CDC_FILE)
    # the curated file of 13 states is no longer read, if it is still there
    pl.DataFrame(
        {
            "state": ["X"],
            "year": [2021],
            "coverage": [91.0],
            "nme": [2.0],
            "me": [1.0],
        }
    ).write_csv(tmp_path / "state_level_coverages.csv")
    series = vw.state_level_series(str(tmp_path))
    assert series["series_source"].to_list() == ["cdc"]
    assert series["mmr_cov_state"].to_list() == [94.0]


# --- settling the FIPS of the compiled vaccination data ---------------------

COUNTIES = [
    # (GEOID, NAME)
    ("24005", "Baltimore"),  # a county and an independent city share a name
    ("24510", "Baltimore"),
    ("17099", "LaSalle"),
    ("51059", "Fairfax"),
    ("11001", "District of Columbia"),
    ("06067", "Sacramento"),
    ("06069", "San Benito"),
    ("31153", "Sarpy"),
    ("10001", "Kent"),
    ("24901", "Testville"),  # two counties with a name addfips does not know
    ("24902", "Testville"),
    ("24903", "Uniqueville"),
]


@pytest.fixture
def data_dir(tmp_path):
    gpd.GeoDataFrame(
        {
            "GEOID": [g for g, _ in COUNTIES],
            "NAME": [n for _, n in COUNTIES],
            "geometry": [Point(i, i) for i in range(len(COUNTIES))],
        },
        crs="EPSG:4326",
    ).to_file(tmp_path / "cb_2020_us_county_20m.shp")
    return str(tmp_path)


def resolve(data_dir, rows):
    """rows: (state, county, fips); returns the settled fips in order."""
    df = pl.DataFrame(
        [dict(state=s, county=c, fips=f) for s, c, f in rows],
        schema={"state": pl.String, "county": pl.String, "fips": pl.String},
    )
    return vw.resolve_vaccination_fips(df, data_dir)["fips"].to_list()


def test_a_valid_fips_is_kept_not_replaced_by_a_name_lookup(data_dir):
    # looking "Baltimore" up by name could return the independent city
    assert resolve(data_dir, [("Maryland", "Baltimore", "24005")]) == ["24005"]
    assert resolve(data_dir, [("Maryland", "Baltimore", "24510")]) == ["24510"]


def test_a_missing_fips_is_looked_up_by_name_ignoring_case(data_dir):
    assert resolve(data_dir, [("Illinois", "Lasalle", None)]) == ["17099"]


def test_known_malformed_and_wrong_codes_are_fixed(data_dir):
    got = resolve(
        data_dir,
        [
            ("Virginia", "Fairfax", "510591"),
            ("District of Columbia", "District of Columbia", "11"),
            ("California", "San Benito", "06067"),
        ],
    )
    assert got == ["51059", "11001", "06069"]


def test_fix_wrong_fips_changes_only_the_rows_with_the_wrong_code():
    df = pl.DataFrame(
        {
            "state": ["California", "California", "Montana", "Montana"]
            + ["Montana", "Nebraska"],
            "county": ["San Benito", "Sacramento", "Lincoln", "Madison"]
            + ["Lincoln", "Lincoln"],
            "fips": ["06067", "06067", "30057", "30057", "30053", "31111"],
        }
    )
    assert vw.fix_wrong_fips(df)["fips"].to_list() == [
        "06069",  # San Benito carried Sacramento's code
        "06067",  # Sacramento keeps it
        "30053",  # Lincoln MT carried Madison's code
        "30057",  # Madison keeps it
        "30053",  # a Lincoln that already has its own code
        "31111",  # another state's Lincoln
    ]


def test_a_row_pooling_counties_has_no_fips(data_dir):
    assert resolve(data_dir, [("Nebraska", "Sarpy/Cass", "31153")]) == [None]


def test_places_that_are_not_counties_stay_null(data_dir):
    assert resolve(
        data_dir,
        [("Oregon", "North Central", None), ("Oregon", "State", "41")],
    ) == [None, None]


def test_delaware_is_left_as_it_is(data_dir):
    assert resolve(
        data_dir, [("Delaware", "State", "10"), ("Delaware", "Kent", None)]
    ) == ["10", None]


def test_an_ambiguous_name_is_not_guessed(data_dir):
    # two counties named Testville, a name addfips cannot resolve: neither
    # may be picked
    assert resolve(data_dir, [("Maryland", "Testville", None)]) == [None]
    assert resolve(data_dir, [("Maryland", "Uniqueville", None)]) == ["24903"]


def test_a_code_that_is_not_a_county_is_replaced_by_the_name_lookup(data_dir):
    assert resolve(data_dir, [("Illinois", "LaSalle", "99999")]) == ["17099"]


# --- regional averages of the vaccination data ------------------------------


def region_rows(rows):
    """rows: (fips, year, coverage, nonmedical, medical)."""
    return pl.DataFrame(
        [
            dict(
                fips=f,
                year=y,
                estimated_mmr_cov=c,
                nonmed_exempt_pct=n,
                med_exempt_pct=m,
            )
            for f, y, c, n, m in rows
        ],
        schema={
            "fips": pl.String,
            "year": pl.Int64,
            "estimated_mmr_cov": pl.Float64,
            "nonmed_exempt_pct": pl.Float64,
            "med_exempt_pct": pl.Float64,
        },
    )


def fixed_populations(pop_5_9_by_fips):
    """A stand-in for the census: the 5-9 population of each county."""

    def query(years, fips):
        return pl.DataFrame({"pop_5_9": [pop_5_9_by_fips[f] for f in fips]})

    return query


def test_regional_average_is_weighted_by_the_population():
    rows = region_rows(
        [("a", 2020, 90.0, 4.0, 1.0), ("b", 2020, 80.0, 2.0, 3.0)]
    )
    out = vw.query_avg_vaccinations_region_years(
        rows,
        [2020],
        ["a", "b"],
        population_query=fixed_populations({"a": 300.0, "b": 100.0}),
    )
    assert out["estimated_mmr_cov"] == [pytest.approx(87.5)]
    assert out["nonmed_exempt_pct"] == [pytest.approx(3.5)]
    assert out["med_exempt_pct"] == [pytest.approx(1.5)]


def test_a_county_without_a_value_does_not_pull_the_average_to_zero():
    rows = region_rows(
        [("a", 2020, 90.0, None, 1.0), ("b", 2020, None, 2.0, 3.0)]
    )
    out = vw.query_avg_vaccinations_region_years(
        rows,
        [2020],
        ["a", "b"],
        population_query=fixed_populations({"a": 100.0, "b": 100.0}),
    )
    assert out["estimated_mmr_cov"] == [90.0]  # only a reports it
    assert out["nonmed_exempt_pct"] == [2.0]
    assert out["med_exempt_pct"] == [pytest.approx(2.0)]


def test_regional_average_weights_each_year_by_its_population():
    rows = region_rows(
        [("a", 2019, 80.0, 1.0, 1.0), ("a", 2020, 90.0, 1.0, 1.0)]
    )

    def query(years, fips):  # the population is 3 times larger in 2020
        return pl.DataFrame(
            {"pop_5_9": [100.0 if y == 2019 else 300.0 for y in years]}
        )

    out = vw.query_avg_vaccinations_region_years(
        rows, [2019, 2020], ["a"], population_query=query
    )
    assert out["estimated_mmr_cov"] == [pytest.approx(87.5)]


def test_a_field_that_nobody_reports_has_no_average_and_no_rows_no_values():
    rows = region_rows([("a", 2020, 90.0, None, None)])
    out = vw.query_avg_vaccinations_region_years(
        rows,
        [2020],
        ["a"],
        population_query=fixed_populations({"a": 100.0}),
    )
    assert out["nonmed_exempt_pct"] == [None]
    none = vw.query_avg_vaccinations_region_years(
        rows, [2021], ["a"], population_query=fixed_populations({})
    )
    assert none == {
        "estimated_mmr_cov": None,
        "nonmed_exempt_pct": None,
        "med_exempt_pct": None,
    }


def test_no_population_means_no_average():
    rows = region_rows([("a", 2020, 90.0, 1.0, 1.0)])
    out = vw.query_avg_vaccinations_region_years(
        rows,
        [2020],
        ["a"],
        population_query=fixed_populations({"a": 0.0}),
    )
    assert out["estimated_mmr_cov"] == [None]


# --- the compiled vaccination data ------------------------------------------


def vaccinations(rows):
    """rows: (state, county, fips, year, mmr_cov, all_vacc_cov)."""
    return pl.DataFrame(
        [
            dict(
                state=s,
                county=c,
                fips=f,
                year=y,
                mmr_cov=m,
                all_vacc_cov=a,
                nonmed_exempt_pct=1.0,
                med_exempt_pct=0.5,
            )
            for s, c, f, y, m, a in rows
        ],
        schema={
            "state": pl.String,
            "county": pl.String,
            "fips": pl.String,
            "year": pl.Int64,
            "mmr_cov": pl.Float64,
            "all_vacc_cov": pl.Float64,
            "nonmed_exempt_pct": pl.Float64,
            "med_exempt_pct": pl.Float64,
        },
    )


def no_regions():
    return {}


def test_coverage_falls_back_to_all_vaccines_when_mmr_is_missing_or_zero():
    out = vw.process_vaccinations_df(
        vaccinations(
            [
                ("S", "a", "01001", 2020, 90.0, 80.0),
                ("S", "b", "01003", 2020, None, 85.0),
                ("S", "c", "01005", 2020, 0.0, 70.0),
            ]
        ),
        regions=no_regions(),
    ).sort("fips")
    assert out["estimated_mmr_cov"].to_list() == [90.0, 85.0, 70.0]


def test_coverage_source_says_which_coverage_a_row_uses():
    out = vw.process_vaccinations_df(
        vaccinations(
            [
                ("S", "a", "01001", 2020, 90.0, 80.0),
                ("S", "b", "01003", 2020, None, 85.0),
                ("S", "c", "01005", 2020, 0.0, 70.0),
                ("S", "d", "01007", 2020, None, None),
            ]
        ),
        regions=no_regions(),
    ).sort("fips")
    assert out["coverage_source"].to_list() == [
        "mmr",
        "all_vaccine",
        "all_vaccine",
        None,
    ]


def test_a_region_has_one_coverage_source_only_if_all_its_counties_have_it():
    rows = vaccinations(
        [
            ("T", "a", "47001", 2020, 90.0, 80.0),  # mmr
            ("T", "b", "47003", 2020, 80.0, 70.0),  # mmr
            ("T", "c", "47005", 2020, None, 70.0),  # all vaccines
            ("T", "a", "47001", 2021, 88.0, 70.0),  # mmr
            ("T", "c", "47005", 2021, None, 70.0),  # all vaccines
            ("T", "b", "47003", 2022, 85.0, 70.0),  # mmr
        ]
    )

    def region_averages(df, years, fips):
        return {
            "estimated_mmr_cov": [1.0],
            "nonmed_exempt_pct": [1.0],
            "med_exempt_pct": [0.5],
        }

    out = vw.process_vaccinations_df(
        rows,
        regions={"R": ["47001", "47003", "47005"], "Gulf Coast": ["47001"]},
        region_averages=region_averages,
    )
    region = out.filter(pl.col("county") == "R").sort("year")
    assert region["year"].to_list() == [2020, 2021, 2022]
    assert region["coverage_source"].to_list() == ["mixed", "mixed", "mmr"]


def test_county_names_are_spelled_as_in_the_case_data():
    out = vw.process_vaccinations_df(
        vaccinations(
            [
                ("Illinois", "Mcdonough", "17109", 2020, 90.0, 80.0),
                ("Illinois", "Dupage", "17043", 2020, 90.0, 80.0),
                ("Texas", "Mcculloch", "48307", 2020, 90.0, 80.0),
            ]
        ),
        regions=no_regions(),
    )
    assert sorted(out["county"]) == ["DuPage", "McCulloch", "McDonough"]


def test_the_district_of_columbia_gets_its_county_fips():
    out = vw.process_vaccinations_df(
        vaccinations(
            [
                ("District of Columbia", "DC", "00011", 2020, 90.0, 80.0),
                ("District of Columbia", "DC", "11", 2021, 90.0, 80.0),
            ]
        ),
        regions=no_regions(),
    )
    assert out["fips"].to_list() == ["11001", "11001"]


def test_old_nebraska_district_codes_are_expanded_to_their_counties():
    out = vw.process_vaccinations_df(
        vaccinations([("Nebraska", "District", "310012", 2020, 90.0, 80.0)]),
        regions=no_regions(),
    )
    # the district 310012 stands for three counties of the new codes
    assert out.height == 3
    assert out["new_nebraska"].null_count() == 0
    assert all(len(f) == 5 for f in out["new_nebraska"])


def test_a_region_is_added_for_every_year_with_the_averages_of_its_counties():
    rows = vaccinations(
        [
            ("Tennessee", "a", "47001", 2020, 90.0, 80.0),
            ("Tennessee", "b", "47003", 2020, 80.0, 70.0),
            ("Tennessee", "a", "47001", 2021, 88.0, 70.0),
            ("Tennessee", "b", "47003", 2021, 78.0, 70.0),
        ]
    )
    asked = []

    def region_averages(df, years, fips):
        asked.append((years, fips))
        return {
            "estimated_mmr_cov": [float(years[0])],
            "nonmed_exempt_pct": [1.0],
            "med_exempt_pct": [0.5],
        }

    out = vw.process_vaccinations_df(
        rows,
        regions={"Middle Region": ["47001", "47003"], "Gulf Coast": ["47001"]},
        region_averages=region_averages,
    )
    region = out.filter(pl.col("county") == "Middle Region").sort("year")
    assert region["year"].to_list() == [2020, 2021]
    assert region["state"].to_list() == ["Tennessee"] * 2
    assert region["fips"].null_count() == 2  # a region has no county fips
    assert region["estimated_mmr_cov"].to_list() == [2020.0, 2021.0]
    # the Gulf Coast is not averaged
    assert out.filter(pl.col("county") == "Gulf Coast").height == 0
    assert [f for _, f in asked] == [["47001", "47003"]] * 2


def test_long_fips_are_cut_to_five_characters_except_in_kansas():
    out = vw.process_vaccinations_df(
        vaccinations(
            [
                ("Ohio", "a", "390011", 2020, 90.0, 80.0),
                ("Kansas", "b", "201234567", 2020, 90.0, 80.0),
            ]
        ),
        regions=no_regions(),
    ).sort("state")
    assert out["fips"].to_list() == ["20123" + "4567", "39001"]


# --- imputing the vaccination data of counties that have none ---------------


def missing_row(county="a", year=2024, mmr=None, nonmed=None, med=None):
    return dict(
        state="X",
        county=county,
        year=year,
        year_relevant_vaccination=year - 1,
        estimated_mmr_cov=mmr,
        nonmed_exempt_pct=nonmed,
        med_exempt_pct=med,
    )


def missing_frame(rows):
    return pl.DataFrame(
        rows,
        schema={
            "state": pl.String,
            "county": pl.String,
            "year": pl.Int64,
            "year_relevant_vaccination": pl.Int64,
            "estimated_mmr_cov": pl.Float64,
            "nonmed_exempt_pct": pl.Float64,
            "med_exempt_pct": pl.Float64,
        },
    )


def county_values(rows):
    """rows: (county, year, coverage, nonmedical, medical) of state X."""
    return pl.DataFrame(
        [
            dict(
                state="X",
                county=c,
                year=y,
                estimated_mmr_cov=m,
                nonmed_exempt_pct=n,
                med_exempt_pct=md,
            )
            for c, y, m, n, md in rows
        ],
        schema={
            "state": pl.String,
            "county": pl.String,
            "year": pl.Int64,
            "estimated_mmr_cov": pl.Float64,
            "nonmed_exempt_pct": pl.Float64,
            "med_exempt_pct": pl.Float64,
        },
    )


@pytest.fixture
def cdc_file(tmp_path):
    """State X in 2023 (the vaccination year of the 2024 outbreak year):
    coverage 91 and a Non-Medical Exemption of 3 with a medical one of 1;
    state Y has only an "Any Exemption" (5) and no medical figure."""
    rows = [
        ("States", "X", "2023-24", "MMR", "2 doses", "91.0"),
        (
            "States",
            "X",
            "2023-24",
            "Exemption",
            "Non-Medical Exemption",
            "3.0",
        ),
        ("States", "X", "2023-24", "Exemption", "Medical Exemption", "1.0"),
        ("States", "Y", "2023-24", "MMR", "2 doses", "88.0"),
        ("States", "Y", "2023-24", "Exemption", "Any Exemption", "5.0"),
    ]
    path = tmp_path / "cdc.csv"
    cdc_rows(rows).write_csv(path)
    return str(path)


def impute(missing, values, cdc_file, regions=None):
    return vw.impute_from_bordering_years(
        missing_frame(missing),
        county_values(values),
        cdc_file,
        regions=regions if regions is not None else {},
    )


def only_row(out):
    assert out.height == 1
    return out.row(0, named=True)


def test_the_same_year_of_the_county_comes_first(cdc_file):
    out = impute(
        [missing_row()],
        [("a", 2024, 85.0, 2.0, 0.4), ("a", 2022, 70.0, 9.0, 9.0)],
        cdc_file,
    )
    row = only_row(out)
    assert (row["estimated_mmr_cov"], row["nonmed_exempt_pct"]) == (85.0, 2.0)
    assert row["med_exempt_pct"] == 0.4
    assert row["imputation_source"] == "mmr:year+0; nonmed:year+0; med:year+0"


def test_two_years_back_fills_what_the_same_year_lacks(cdc_file):
    out = impute(
        [missing_row()],
        [("a", 2024, 85.0, None, None), ("a", 2022, 70.0, 6.0, 0.9)],
        cdc_file,
    )
    row = only_row(out)
    assert row["estimated_mmr_cov"] == 85.0  # not overwritten
    assert (row["nonmed_exempt_pct"], row["med_exempt_pct"]) == (6.0, 0.9)
    assert row["imputation_source"] == "mmr:year+0; nonmed:year-2; med:year-2"


def test_the_states_cdc_values_are_the_last_resort(cdc_file):
    row = only_row(impute([missing_row()], [], cdc_file))
    assert row["estimated_mmr_cov"] == 91.0
    assert row["nonmed_exempt_pct"] == 3.0
    assert row["med_exempt_pct"] == 1.0
    assert (
        row["imputation_source"]
        == "mmr:state_cdc; nonmed:state_cdc; med:state_cdc"
    )


def test_any_exemption_without_a_medical_figure_sets_medical_to_zero(
    cdc_file,
):
    missing = missing_row()
    missing["state"] = "Y"
    out = vw.impute_from_bordering_years(
        missing_frame([missing]), county_values([]), cdc_file, regions={}
    )
    row = only_row(out)
    assert row["nonmed_exempt_pct"] == 5.0  # Any, as there is no medical
    assert row["med_exempt_pct"] == 0.0
    assert "med:state_cdc_zero" in row["imputation_source"]


def test_only_the_missing_fields_are_imputed(cdc_file):
    out = impute([missing_row(mmr=77.0, nonmed=1.5)], [], cdc_file)
    row = only_row(out)
    assert (row["estimated_mmr_cov"], row["nonmed_exempt_pct"]) == (77.0, 1.5)
    assert row["med_exempt_pct"] == 1.0
    assert row["imputation_source"] == "med:state_cdc"


def test_nothing_is_returned_for_complete_rows_or_unimputable_ones(cdc_file):
    complete = missing_row(mmr=80.0, nonmed=1.0, med=1.0)
    # state Z has no CDC row and the county has no values
    unimputable = dict(missing_row(), state="Z")
    out = vw.impute_from_bordering_years(
        missing_frame([complete, unimputable]),
        county_values([]),
        cdc_file,
        regions={},
    )
    assert out.height == 0


def test_regions_are_not_imputed(cdc_file):
    out = impute(
        [missing_row(county="Big Region"), missing_row(county="a")],
        [],
        cdc_file,
        regions={"Big Region": ["01001", "01003"]},
    )
    assert only_row(out)["county"] == "a"


# --- filling the target states from state-level values -----------------------

# county FIPS of the target states Oregon (41) and Nevada (32), of Hawaii (15,
# a target state that takes the CDC's values), of Texas (48, a target state
# with no state-level values) and of Ohio (39, not a target)
SLC_COUNTIES = [
    ("41001", "Baker", "OR"),
    ("41003", "Benton", "OR"),
    ("41005", "Clackamas", "OR"),
    ("32001", "Churchill", "NV"),
    ("15001", "Hawaii", "HI"),
    ("15005", "Kalawao", "HI"),
    ("48001", "Anderson", "TX"),
    ("39001", "Adams", "OH"),
]


@pytest.fixture
def slc_dir(tmp_path):
    """A shapefile of SLC_COUNTIES and the state-level values: Oregon for 2020
    and 2022, Nevada for 2022 only, nothing for Texas. The CDC's file has
    Hawaii for 2020-21 (no MMR value) and 2021-22."""
    cdc_rows(
        [
            (
                "States",
                "Hawaii",
                "2020-21",
                "Exemption",
                "Non-Medical Exemption",
                "3.0",
            ),
            (
                "States",
                "Hawaii",
                "2020-21",
                "Exemption",
                "Medical Exemption",
                "0.5",
            ),
            ("States", "Hawaii", "2021-22", "MMR", "2 doses", "88.0"),
            (
                "States",
                "Hawaii",
                "2021-22",
                "Exemption",
                "Non-Medical Exemption",
                "4.0",
            ),
            (
                "States",
                "Hawaii",
                "2021-22",
                "Exemption",
                "Medical Exemption",
                "0.6",
            ),
            # Oregon: 2019 (before the file's years) and 2020 (which the
            # file's own values replace)
            ("States", "Oregon", "2019-20", "MMR", "2 doses", "97.0"),
            (
                "States",
                "Oregon",
                "2019-20",
                "Exemption",
                "Non-Medical Exemption",
                "2.0",
            ),
            (
                "States",
                "Oregon",
                "2019-20",
                "Exemption",
                "Medical Exemption",
                "0.5",
            ),
            ("States", "Oregon", "2020-21", "MMR", "2 doses", "80.0"),
            (
                "States",
                "Oregon",
                "2020-21",
                "Exemption",
                "Non-Medical Exemption",
                "9.0",
            ),
            (
                "States",
                "Oregon",
                "2020-21",
                "Exemption",
                "Medical Exemption",
                "9.0",
            ),
        ]
    ).write_csv(tmp_path / CDC_FILE)
    gpd.GeoDataFrame(
        {
            "STATEFP": [fips[:2] for fips, _, _ in SLC_COUNTIES],
            "GEOID": [fips for fips, _, _ in SLC_COUNTIES],
            "NAME": [name for _, name, _ in SLC_COUNTIES],
            "geometry": [Point(i, i) for i in range(len(SLC_COUNTIES))],
        },
        crs="EPSG:4326",
    ).to_file(tmp_path / "cb_2020_us_county_20m.shp")
    return str(tmp_path)


def slc_vaccinations():
    """Real rows: Baker (41001) in 2020 and in 2022 (with no MMR value), a
    region (no fips) in 2021 and an Ohio county in 2020."""
    return pl.DataFrame(
        [
            ("41001", "Oregon", "Baker", 2020, 70.0, 8.0, 3.0, "a"),
            ("41001", "Oregon", "Baker", 2022, None, 2.0, 0.5, "b"),
            (None, "Oregon", "Big Region", 2021, 60.0, 1.0, 1.0, "c"),
            ("39001", "Ohio", "Adams", 2020, 95.0, 0.5, 0.5, "d"),
        ],
        schema={
            "fips": pl.String,
            "state": pl.String,
            "county": pl.String,
            "year": pl.Int64,
            "estimated_mmr_cov": pl.Float64,
            "nonmed_exempt_pct": pl.Float64,
            "med_exempt_pct": pl.Float64,
            "extra": pl.String,
        },
        orient="row",
    )


def slc_row(df, fips, year):
    out = df.filter((pl.col("fips") == fips) & (pl.col("year") == year))
    assert out.height == 1, (fips, year, out.height)
    return out.row(0, named=True)


def test_every_target_county_gets_every_year(slc_dir):
    out = vw.impute_from_state_level_coverages(slc_vaccinations(), slc_dir)
    # years 2020-2022 (2021 comes only from the region's row)
    for fips in ("41001", "41003", "41005"):
        years = out.filter(pl.col("fips") == fips)["year"].sort().to_list()
        assert years == [2020, 2021, 2022], fips
    assert "_is_imputed" not in out.columns


def test_a_year_with_state_values_uses_them_and_other_years_the_nearest(
    slc_dir,
):
    out = vw.impute_from_state_level_coverages(slc_vaccinations(), slc_dir)
    cols = ["estimated_mmr_cov", "nonmed_exempt_pct", "med_exempt_pct"]
    # the CDC has Oregon for 2019 and 2020 only: 2020 takes its own values,
    # and 2021 and 2022 the nearest year's (2020)
    assert [slc_row(out, "41003", 2020)[c] for c in cols] == [80.0, 9.0, 9.0]
    assert [slc_row(out, "41003", 2021)[c] for c in cols] == [80.0, 9.0, 9.0]
    assert [slc_row(out, "41005", 2022)[c] for c in cols] == [80.0, 9.0, 9.0]
    # Nevada is not in the CDC file here: its counties are skipped
    assert out.filter(pl.col("state") == "Nevada").height == 0


def test_years_before_the_file_take_the_cdc_values_of_that_year(slc_dir):
    # an Ohio row in 2019 puts 2019 among the years of the data
    earlier = pl.concat(
        [
            slc_vaccinations(),
            slc_vaccinations()
            .filter(pl.col("state") == "Ohio")
            .with_columns(year=pl.lit(2019, dtype=pl.Int64)),
        ]
    )
    out = vw.impute_from_state_level_coverages(earlier, slc_dir)
    cols = ["estimated_mmr_cov", "nonmed_exempt_pct", "med_exempt_pct"]
    # Oregon 2019 and 2020 are the CDC's own years
    assert [slc_row(out, "41003", 2019)[c] for c in cols] == [97.0, 2.0, 0.5]
    assert [slc_row(out, "41003", 2020)[c] for c in cols] == [80.0, 9.0, 9.0]


def test_a_target_state_not_in_the_file_takes_the_cdc_values(slc_dir):
    out = vw.impute_from_state_level_coverages(slc_vaccinations(), slc_dir)
    cols = ["estimated_mmr_cov", "nonmed_exempt_pct", "med_exempt_pct"]
    # Kalawao has no real row at all, so it gets every year
    kalawao = out.filter(pl.col("fips") == "15005")
    assert kalawao["year"].sort().to_list() == [2020, 2021, 2022]
    # 2020 has no MMR value in the CDC file: the nearest year's (2021) fills it
    assert [slc_row(out, "15005", 2020)[c] for c in cols] == [88.0, 3.0, 0.5]
    assert [slc_row(out, "15005", 2021)[c] for c in cols] == [88.0, 4.0, 0.6]
    # 2022 is not in the file: the latest year's values
    assert [slc_row(out, "15005", 2022)[c] for c in cols] == [88.0, 4.0, 0.6]


def test_real_values_are_kept_and_only_their_gaps_are_filled(slc_dir):
    out = vw.impute_from_state_level_coverages(slc_vaccinations(), slc_dir)
    kept = slc_row(out, "41001", 2020)
    assert (kept["estimated_mmr_cov"], kept["nonmed_exempt_pct"]) == (
        70.0,
        8.0,
    )
    assert kept["extra"] == "a"  # the real row, not the skeleton row
    # the real 2022 row has no MMR value: the state's (nearest year, 2020)
    # fills it, nothing else
    gap = slc_row(out, "41001", 2022)
    assert gap["estimated_mmr_cov"] == 80.0
    assert (gap["nonmed_exempt_pct"], gap["med_exempt_pct"]) == (2.0, 0.5)
    assert gap["extra"] == "b"


def test_added_rows_have_no_values_for_other_columns(slc_dir):
    out = vw.impute_from_state_level_coverages(slc_vaccinations(), slc_dir)
    assert slc_row(out, "41003", 2020)["extra"] is None


def test_rows_without_a_fips_are_kept_as_they_are(slc_dir):
    out = vw.impute_from_state_level_coverages(slc_vaccinations(), slc_dir)
    region = out.filter(pl.col("fips").is_null())
    assert region["county"].to_list() == ["Big Region"]
    assert region["estimated_mmr_cov"].to_list() == [60.0]


def test_other_states_are_untouched_and_target_states_without_values_skipped(
    slc_dir, capsys
):
    out = vw.impute_from_state_level_coverages(slc_vaccinations(), slc_dir)
    assert (
        out.filter(pl.col("state") == "Ohio").height == 1
    )  # its one real row
    assert out.filter(pl.col("state") == "Texas").height == 0
    assert "No state-level coverage for 'Texas'" in capsys.readouterr().out


def test_the_input_is_not_modified(slc_dir):
    original = slc_vaccinations()
    vw.impute_from_state_level_coverages(original, slc_dir)
    assert original.equals(slc_vaccinations())


# --- the data QC exclusions ---------------------------------------------------


def qc_rows():
    return pl.DataFrame(
        {
            "state": ["Georgia", "Georgia", "Georgia", "Ohio"],
            "county": ["Richmond", "Richmond", "Columbia", "Richmond"],
            "year": [2022, 2023, 2023, 2023],
            "nonmed_exempt_pct": [1.4, 1.2, 0.5, 2.0],
            "MMR_cov": [None, None, 93.0, 95.0],
            "all_vacc_cov": ["0", "0.04", "92.0", "94.0"],
        },
        schema_overrides={"MMR_cov": pl.Float64},
    )


def test_a_qc_exclusion_without_fields_removes_the_whole_row():
    out = vw.apply_qc_exclusions(
        qc_rows(), [{"state": "Georgia", "county": "Richmond", "year": 2023}]
    )
    assert out.height == 3
    # the same county in another year, another county and another state stay
    assert out.select("state", "county", "year").rows() == [
        ("Georgia", "Richmond", 2022),
        ("Georgia", "Columbia", 2023),
        ("Ohio", "Richmond", 2023),
    ]


def test_a_qc_exclusion_with_fields_blanks_only_those_fields():
    out = vw.apply_qc_exclusions(
        qc_rows(),
        [
            {
                "state": "Georgia",
                "county": "Richmond",
                "year": 2023,
                "fields": ["coverage"],
            }
        ],
    )
    expected = qc_rows().with_columns(
        all_vacc_cov=pl.Series(["0", None, "92.0", "94.0"])
    )
    assert out.equals(expected)  # also the dtypes
    # the exemption of the row is kept
    assert out["nonmed_exempt_pct"].to_list() == [1.4, 1.2, 0.5, 2.0]


def test_a_qc_exclusion_that_matches_no_row_is_an_error():
    with pytest.raises(ValueError, match="matches no row"):
        vw.apply_qc_exclusions(
            qc_rows(),
            [{"state": "Georgia", "county": "Richmnd", "year": 2023}],
        )


# New York.xlsx: a row of column codes and three description rows (the last
# wrapped onto two Excel rows), then one row per county and year; every cell
# is text in the sheet
def new_york_sheet(rows):
    """rows: (year, county, nme %, me %, mmr %, utd %)."""
    top = [
        "year state county race nme_pct nme_num me_pct mmr_pct utd_pct cases",
        "Year State County Race/ethnicity NME NME-N ME MMR UTD Cases",
        "#### Name Name Definitions % N % % % N",
        "(2010-2025) None None None None None None None None None",
    ]
    cells = [[None if c == "None" else c for c in r.split()] for r in top]
    cells = [c + [None] * (11 - len(c)) for c in cells]
    data = [
        [str(y), "New York", county, None, nme, None, me, mmr, utd, None, None]
        for y, county, nme, me, mmr, utd in rows
    ]
    return pl.DataFrame(
        cells + data,
        schema=[f"column_{i}" for i in range(1, 12)],
        orient="row",
        infer_schema_length=None,
    )


@pytest.fixture
def new_york(monkeypatch):
    sheet = new_york_sheet(
        [
            (2018, "ST. LAWRENCE", "0.45", "0.06", "99.1", "98.9"),
            (2018, "NEW YORK", "0.37", "0.01", "98.6", "98.0"),
            (2019, "ST. LAWRENCE", None, "0.06", "99.2", "98.9"),
            (2019, "NEW YORK", "0", "0.01", "0", "97.0"),
        ]
    )
    monkeypatch.setattr(vw.pl, "read_excel", lambda *a, **k: sheet)
    return vw.read_new_york_data("unused")


def test_the_new_york_reader_drops_the_description_rows(new_york):
    assert new_york.height == 4
    assert new_york["year"].dtype == pl.Int64
    assert sorted(new_york["year"].unique().to_list()) == [2018, 2019]


def test_new_york_counties_are_in_title_case_with_their_fips(new_york):
    got = dict(zip(new_york["county"], new_york["fips"]))
    assert got == {"St. Lawrence": "36089", "New York": "36061"}


def test_new_york_exemptions_are_zero_from_2019_and_other_values_are_kept(
    new_york,
):
    out = new_york.sort("county", "year")
    # St. Lawrence's blank 2019 rate is the ban's 0; 2018's rate is kept
    assert out["nonmed_exempt_pct"].to_list() == [0.37, 0.0, 0.45, 0.0]
    assert out["med_exempt_pct"].to_list() == [0.01, 0.01, 0.06, 0.06]


def test_new_york_estimated_coverage_falls_back_to_all_vaccine_when_mmr_is_0(
    new_york,
):
    out = new_york.sort("county", "year")
    assert out["estimated_mmr_cov"].to_list() == [98.6, 97.0, 99.1, 99.2]


def test_replacing_a_state_keeps_nothing_of_its_old_rows():
    df = pl.DataFrame(
        {
            "state": ["New York", "New York", "Ohio"],
            "county": ["Albany", "Kings", "Erie"],
            "year": [2012, 2012, 2012],
            "mmr_cov": [90.0, 91.0, 92.0],
            "old_only": ["a", "b", "c"],
        }
    )
    new = pl.DataFrame(
        {
            "state": ["New York"],
            "county": ["Albany"],
            "year": [2013],
            "mmr_cov": [99.0],
            "new_only": [1.0],
        }
    )
    out = vw.replace_state_rows(df, "New York", new)
    assert out.sort("state", "year").rows() == [
        ("New York", "Albany", 2013, 99.0, None),
        ("Ohio", "Erie", 2012, 92.0, "c"),
    ]


def test_flag_imputed_rows_marks_new_rows_and_filled_values_only():
    columns = ["estimated_mmr_cov", "nonmed_exempt_pct", "med_exempt_pct"]
    original = pl.DataFrame(
        {
            "fips": ["01001", "01003", "01005", None],
            "year": [2020] * 4,
            "estimated_mmr_cov": [90.0, None, None, 80.0],
            "nonmed_exempt_pct": [1.0, 2.0, None, 1.0],
            "med_exempt_pct": [0.1, 0.2, None, 0.1],
        }
    )
    imputed = pl.DataFrame(
        {
            "fips": ["01001", "01003", "01005", "01007", None],
            "year": [2020] * 5,
            "estimated_mmr_cov": [90.0, 95.0, None, 95.0, 80.0],
            "nonmed_exempt_pct": [1.0, 2.0, None, 3.0, 1.0],
            "med_exempt_pct": [0.1, 0.2, None, 0.3, 0.1],
        }
    )
    out = vw.flag_imputed_rows(imputed, original)
    # reported; coverage filled; blank and not filled; a county that was missing; a null-FIPS region
    assert out["_is_imputed"].to_list() == [False, True, False, True, False]
    assert out.select(["fips", *columns]).equals(
        imputed.select(["fips", *columns])
    )


# --- a region stands for the counties without a case row of their own ---------


def mean_coverage(df, years, fips):
    """The plain mean of the coverage of the counties `fips` (None for none):
    a stand-in for the population-weighted average that shows which counties
    were averaged."""
    rows = df.filter(pl.col("year").is_in(years) & pl.col("fips").is_in(fips))
    return {
        "estimated_mmr_cov": [rows["estimated_mmr_cov"].mean()],
        "nonmed_exempt_pct": [1.0],
        "med_exempt_pct": [0.5],
    }


REGION_COUNTIES = {"R": ["47001", "47003", "47005"]}


def county_vaccinations():
    # three counties in 2024 and 2025: coverage 90 80 70, then 60 50 40
    return vaccinations(
        [
            ("T", "a", "47001", 2024, 90.0, None),
            ("T", "b", "47003", 2024, 80.0, None),
            ("T", "c", "47005", 2024, 70.0, None),
            ("T", "a", "47001", 2025, 60.0, None),
            ("T", "b", "47003", 2025, 50.0, None),
            ("T", "c", "47005", 2025, 40.0, None),
        ]
    )


def case_places(rows):
    """rows: (year, county, fips)."""
    return pl.DataFrame(
        [dict(year=y, state="T", county=c, fips=f) for y, c, f in rows],
        schema={
            "year": pl.Int64,
            "state": pl.String,
            "county": pl.String,
            "fips": pl.String,
        },
    )


def test_a_regions_row_averages_the_members_without_a_row_of_their_own():
    # county a has a case row in 2025, so the region of the cases of 2025 does
    # not stand for it. Those cases take the vaccination row of 2024: the
    # region's row of 2024 averages b and c; its row of 2025 (for 2026) all
    out = vw.process_vaccinations_df(
        county_vaccinations(),
        regions=REGION_COUNTIES,
        region_averages=mean_coverage,
        case_places=case_places(
            [(2025, "R", None), (2025, "a", "47001"), (2026, "R", None)]
        ),
    )
    region = out.filter(pl.col("county") == "R").sort("year")
    assert region["year"].to_list() == [2024, 2025]
    assert region["estimated_mmr_cov"].to_list() == [75.0, 50.0]


def test_without_case_places_a_region_averages_every_member():
    out = vw.process_vaccinations_df(
        county_vaccinations(),
        regions=REGION_COUNTIES,
        region_averages=mean_coverage,
    )
    region = out.filter(pl.col("county") == "R").sort("year")
    assert region["estimated_mmr_cov"].to_list() == [80.0, 50.0]


def test_a_region_whose_members_all_have_a_row_has_no_coverage():
    # the row is for nobody: no county is left to average (the case rows
    # then fail the check of 00 if the region has cases that year)
    out = vw.process_vaccinations_df(
        county_vaccinations(),
        regions=REGION_COUNTIES,
        region_averages=mean_coverage,
        case_places=case_places(
            [
                (2025, c, f)
                for c, f in [("a", "47001"), ("b", "47003"), ("c", "47005")]
            ]
        ),
    )
    region = out.filter(pl.col("county") == "R").sort("year")
    assert region["estimated_mmr_cov"].to_list() == [None, 50.0]


def test_the_cases_of_a_year_take_the_row_built_for_their_own_members():
    # end to end with the real merge: a case row of year Y takes the
    # vaccination row of Y - 1, and that row averages the members that have
    # no case row of their own in Y
    from outbreak_size.analysis_dataset import merge_outbreaks_vaccinations

    cases = case_places(
        [(2025, "R", None), (2025, "a", "47001"), (2026, "R", None)]
    ).with_columns(cases=pl.lit(5))
    out = vw.process_vaccinations_df(
        county_vaccinations(),
        regions=REGION_COUNTIES,
        region_averages=mean_coverage,
        case_places=cases,
    )
    merged = merge_outbreaks_vaccinations(cases, out)
    coverage = {
        r["year"]: r["estimated_mmr_cov"]
        for r in merged.filter(pl.col("county") == "R").iter_rows(named=True)
    }
    # 2025: the members b and c of school year 2024; 2026: all three of 2025
    assert coverage == {2025: 75.0, 2026: 50.0}


# --- Nebraska districts, pooled rows, Wyoming, Hawaii, Montana --------------


def test_an_old_nebraska_district_code_is_kept_even_for_a_pooled_row(data_dir):
    got = resolve(
        data_dir,
        [
            ("Nebraska", "Sarpy/Cass", "310006"),
            ("Nebraska", "Central", "310012"),
            ("Mississippi", "Issaquena/Sharkey", "28055"),
        ],
    )
    # the districts keep their code (they are given to their counties later);
    # any other pooled row has none
    assert got == ["310006", "310012", None]


def test_every_old_nebraska_district_code_expands_to_counties():
    for code in sorted(vw.NEBRASKA_DISTRICT_CODES):
        out = vw.process_vaccinations_df(
            vaccinations([("Nebraska", "District", code, 2020, 90.0, 80.0)]),
            regions=no_regions(),
        )
        assert out.height >= 1, code
        assert out["new_nebraska"].null_count() == 0, code


def test_sarpy_cass_goes_to_sarpy_and_cass():
    out = vw.process_vaccinations_df(
        vaccinations([("Nebraska", "Sarpy/Cass", "310006", 2020, 97.0, 90.0)]),
        regions=no_regions(),
    )
    assert sorted(out["fips"].to_list()) == ["31025", "31153"]
    assert out["estimated_mmr_cov"].to_list() == [97.0, 97.0]
    # each row is named for its county, as the case data name it
    assert sorted(out["county"].to_list()) == ["Cass", "Sarpy"]


def test_expanded_nebraska_rows_are_named_for_their_counties_once_a_year():
    out = vw.process_vaccinations_df(
        vaccinations(
            [
                ("Nebraska", "Central", "310012", 2020, 90.0, 80.0),
                ("Nebraska", "Other", "310005", 2020, 85.0, 70.0),
            ]
        ),
        regions=no_regions(),
    )
    assert out.height == out["county"].n_unique()
    assert "Central" not in out["county"].to_list()
    assert out["county"].null_count() == 0


def pooled_frame(rows):
    """rows: (county, fips, year, nonmed, med, mmr, all-vaccine, estimated)."""
    return pl.DataFrame(
        [
            dict(
                state="Mississippi",
                county=c,
                fips=f,
                year=y,
                nonmed_exempt_pct=n,
                med_exempt_pct=m,
                mmr_cov=mmr,
                all_vacc_cov=a,
                estimated_mmr_cov=e,
                coverage_source=None if e is None else "all_vaccine",
                county_type="County",
            )
            for c, f, y, n, m, mmr, a, e in rows
        ],
        schema={
            "state": pl.String,
            "county": pl.String,
            "fips": pl.String,
            "year": pl.Int64,
            "nonmed_exempt_pct": pl.Float64,
            "med_exempt_pct": pl.Float64,
            "mmr_cov": pl.Float64,
            "all_vacc_cov": pl.Float64,
            "estimated_mmr_cov": pl.Float64,
            "coverage_source": pl.String,
            "county_type": pl.String,
        },
    )


VALUES = [
    "nonmed_exempt_pct",
    "med_exempt_pct",
    "mmr_cov",
    "all_vacc_cov",
    "estimated_mmr_cov",
    "coverage_source",
]


def test_a_pooled_row_goes_to_each_member_and_the_members_own_values_win():
    df = pooled_frame(
        [
            # the pooled row: a nonmedical exemption in 2022, coverage 2023
            ("Issaquena/Sharkey", None, 2022, 0.0, None, None, None, None),
            ("Issaquena/Sharkey", None, 2023, None, None, None, 100.0, 100.0),
            # Sharkey's own rows: its own exemption in 2022, nothing in 2023
            ("Sharkey", "28125", 2022, 0.2, None, None, None, None),
            ("Sharkey", "28125", 2023, None, None, None, None, None),
            ("Yazoo", "28163", 2023, 1.0, 0.5, 90.0, 85.0, 90.0),
        ]
    )
    out = vw.expand_pooled_rows(df, VALUES)
    assert "Issaquena/Sharkey" not in out["county"].to_list()

    def row(fips, year):
        r = out.filter((pl.col("fips") == fips) & (pl.col("year") == year))
        assert r.height == 1, (fips, year)
        return r.row(0, named=True)

    # Issaquena has no row of its own: the pooled values, its own name
    assert row("28055", 2022)["county"] == "Issaquena"
    assert row("28055", 2022)["nonmed_exempt_pct"] == 0.0
    assert row("28055", 2023)["estimated_mmr_cov"] == 100.0
    # Sharkey: its own 2022 value wins; the pooled one fills what it lacks
    assert row("28125", 2022)["nonmed_exempt_pct"] == 0.2
    assert row("28125", 2023)["estimated_mmr_cov"] == 100.0
    assert row("28125", 2023)["county"] == "Sharkey"
    # other rows are untouched
    assert row("28163", 2023)["mmr_cov"] == 90.0
    assert out.height == 5


def test_a_pooled_row_with_a_valid_fips_is_not_taken_for_an_own_row():
    # in the burden figure's data the pooled row carries Issaquena's fips
    df = pooled_frame(
        [
            ("Issaquena/Sharkey", "28055", 2022, 0.0, None, None, None, None),
            ("Sharkey", "28125", 2022, None, None, None, None, None),
        ]
    )
    out = vw.expand_pooled_rows(df, VALUES)
    assert sorted(out["county"].to_list()) == ["Issaquena", "Sharkey"]
    assert out["nonmed_exempt_pct"].to_list() == [0.0, 0.0]


NOV9_COLUMNS = [
    "Year",
    "State",
    "County",
    "Non-medical exemptions (%)",
    "Medical exemptions (%)",
    "MMR vaccine coverage",
    "Up-to-date vaccination (all vaccines)",
    "FIPS",
    "county_type",
]


def write_nov9(tmp_path, rows):
    pl.DataFrame(rows, schema=NOV9_COLUMNS, orient="row").write_csv(
        tmp_path / vw.VACCINATION_NOV9_FILE
    )
    return str(tmp_path)


def compiled(rows):
    """rows: (state, county, year, nonmed, med, mmr)"""
    return pl.DataFrame(
        rows,
        schema={
            "state": pl.String,
            "county": pl.String,
            "year": pl.Int64,
            "nonmed_exempt_pct": pl.Float64,
            "med_exempt_pct": pl.Float64,
            "mmr_cov": pl.Float64,
        },
        orient="row",
    )


def test_wyoming_nonmedical_moves_one_year_earlier_and_nov9_fills_mmr(
    tmp_path,
):
    data = write_nov9(
        tmp_path,
        [
            # labelled by the year of recording: the school year is the year
            # before. 2017 and 2018 repeat the spreadsheet's 2016 and 2017
            (
                2017,
                "Wyoming",
                "Albany",
                None,
                None,
                80.0,
                None,
                "56001",
                "County",
            ),
            (
                2018,
                "Wyoming",
                "Albany",
                None,
                None,
                87.0,
                None,
                "56001",
                "County",
            ),
            (
                2019,
                "Wyoming",
                "Albany",
                None,
                None,
                90.0,
                None,
                "56001",
                "County",
            ),
        ],
    )
    df = compiled(
        [
            ("Wyoming", "Albany", 2016, None, None, 80.0),
            ("Wyoming", "Albany", 2017, 1.0, None, 87.0),
            ("Wyoming", "Albany", 2018, 2.0, None, None),
            ("Wyoming", "Albany", 2019, None, None, None),
            ("Ohio", "Adams", 2017, 3.0, None, 95.0),
        ]
    )
    out = vw.fix_wyoming_year_labels(df, data).sort("state", "county", "year")
    wy = out.filter(pl.col("state") == "Wyoming")
    # nonmedical: 2017's 1.0 is school year 2016's, 2018's 2.0 is 2017's
    assert wy["nonmed_exempt_pct"].to_list() == [1.0, 2.0, None, None]
    # coverage: the spreadsheet's values are kept; the school year 2018 (the
    # file's 2019) fills the blank
    assert wy["mmr_cov"].to_list() == [80.0, 87.0, 90.0, None]
    # other states are untouched
    ohio = out.filter(pl.col("state") == "Ohio")
    assert ohio["nonmed_exempt_pct"].to_list() == [3.0]
    assert ohio["mmr_cov"].to_list() == [95.0]


def test_hawaii_2024_exemptions_fill_only_the_blanks(tmp_path):
    data = write_nov9(
        tmp_path,
        [
            (
                2024,
                "Hawaii",
                "Honolulu",
                2.559,
                0.113,
                None,
                None,
                "15003",
                "County",
            ),
            (
                2024,
                "Hawaii",
                "Maui",
                6.859,
                0.162,
                None,
                None,
                "15009",
                "County",
            ),
            (
                2023,
                "Hawaii",
                "Honolulu",
                9.9,
                9.9,
                None,
                None,
                "15003",
                "County",
            ),
            (2024, "Hawaii", None, None, None, 90.1, None, "15", "State"),
        ],
    )
    df = compiled(
        [
            ("Hawaii", "Honolulu", 2024, None, None, 95.31),
            ("Hawaii", "Maui", 2024, 1.0, None, 93.98),
            ("Hawaii", "Honolulu", 2023, None, None, None),
        ]
    )
    out = vw.fill_hawaii_2024_exemptions(df, data)
    assert out["nonmed_exempt_pct"].to_list() == [2.559, 1.0, None]
    assert out["med_exempt_pct"].to_list() == [0.113, 0.162, None]
    assert out["mmr_cov"].to_list() == [95.31, 93.98, None]


def test_montana_data_fill_only_what_the_compiled_data_lack(monkeypatch):
    montana = pl.DataFrame(
        {
            "year": [2019, 2019],
            "state": ["Montana", "Montana"],
            "county": ["Gallatin", "Hill"],
            "nonmed_exempt_pct": [None, None],
            "med_exempt_pct": [None, None],
            "MMR_cov": [83.2, 96.1],
            "all_vacc_cov": [61.2, 81.5],
            "estimated_mmr_cov": [83.2, 96.1],
        },
        schema_overrides={
            "nonmed_exempt_pct": pl.Float64,
            "med_exempt_pct": pl.Float64,
        },
    )
    monkeypatch.setattr(vw, "read_montana_data", lambda data_dir: montana)
    df = pl.DataFrame(
        {
            "state": ["Montana", "Montana", "Montana", "Ohio"],
            "county": ["Gallatin", "Hill", "Gallatin", "Adams"],
            "year": [2019, 2019, 2018, 2019],
            "nonmed_exempt_pct": [None, 1.0, 4.0, 2.0],
            "med_exempt_pct": [None, None, 0.3, 1.0],
            "mmr_cov": [None, 90.0, None, 95.0],
            "all_vacc_cov": [None, None, None, 80.0],
        },
        schema_overrides={
            "nonmed_exempt_pct": pl.Float64,
            "med_exempt_pct": pl.Float64,
        },
    )
    out = vw.add_montana_data(df, "unused")
    # Gallatin 2019 takes the coverage; the exemptions stay blank (the 2019
    # rows of Montana.xlsx have none: they are not copies of 2018)
    assert out["mmr_cov"].to_list() == [83.2, 90.0, None, 95.0]
    assert out["all_vacc_cov"].to_list() == [61.2, 81.5, None, 80.0]
    assert out["nonmed_exempt_pct"].to_list() == [None, 1.0, 4.0, 2.0]
    assert out.height == 4


def test_wyoming_rows_without_a_fips_take_their_countys(tmp_path):
    data = write_nov9(
        tmp_path,
        [
            (
                2024,
                "Wyoming",
                "Albany",
                None,
                None,
                83.0,
                None,
                "56001",
                "County",
            )
        ],
    )
    df = compiled(
        [
            ("Wyoming", "Albany", 2021, None, None, 77.0),
            ("Wyoming", "Albany", 2023, None, None, None),
        ]
    ).with_columns(fips=pl.Series(["56001", None], dtype=pl.String))
    out = vw.fix_wyoming_year_labels(df, data).sort("year")
    # the file's 2024 is the school year 2023: the blank row gets the value
    # and, to be used, its county's fips
    assert out["mmr_cov"].to_list() == [77.0, 83.0]
    assert out["fips"].to_list() == ["56001", "56001"]


def test_the_default_population_query_uses_the_census_cache(monkeypatch):
    """Script 00 asks for the five school years of every row: the requests must
    go through the cache (and the threads), not one by one to the Census."""
    from outbreak_size import utils

    seen = {}

    class StubQuery:
        def __init__(self, key, cache_path=None, workers=1):
            seen["cache_path"], seen["workers"] = cache_path, workers

        def query_5_9_pop(self, years, fips):
            return pl.DataFrame({"pop_5_9": [100.0] * len(fips)})

    monkeypatch.setattr(utils, "CensusQuery", StubQuery)
    rows = region_rows([("a", 2020, 90.0, 4.0, 1.0)])
    out = vw.query_avg_vaccinations_region_years(rows, [2020], ["a"])
    assert out["estimated_mmr_cov"] == [90.0]
    assert seen == {
        "cache_path": utils.CENSUS_CACHE_PATH,
        "workers": utils.CENSUS_WORKERS,
    }
