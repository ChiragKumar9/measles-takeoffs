"""Reading the case-assignment files (outbreak_size.case_assignments)."""

import os

import polars as pl
import pytest

from outbreak_size.case_assignments import (
    ARIZONA_2026_COUNTIES_WITHOUT_ROWS,
    read_case_assignments,
    shapefile_counties,
)

COLUMNS = [
    "row_type",
    "state",
    "county",
    "geo_unit",
    "month",
    "case_count",
    "marginal",
]


def write(tmp_path, rows, name="cases.csv"):
    path = tmp_path / name
    pl.DataFrame(rows, schema=COLUMNS, orient="row").write_csv(path)
    return str(path)


def A(state, county, geo, month, n, marginal="primary"):
    return ("assignment", state, county, geo, month, float(n), marginal)


# a county of each kind: unique names, and Fairfax (a county and a city)
COUNTIES = pl.DataFrame(
    {
        "fips": ["01001", "01003", "51059", "51600", "18037", "18073"],
        "county": [
            "Autauga",
            "Baldwin",
            "Fairfax",
            "Fairfax",
            "Dubois",
            "Jasper",
        ],
        "state": [
            "Alabama",
            "Alabama",
            "Virginia",
            "Virginia",
            "Indiana",
            "Indiana",
        ],
    }
)


def read(tmp_path, rows, **kw):
    return read_case_assignments(write(tmp_path, rows), 2026, COUNTIES, **kw)


def test_only_assignment_rows_are_cases_and_months_are_summed(tmp_path):
    rows = [
        A("Alabama", "Autauga", "county", "January", 2),
        A("Alabama", "Autauga", "county", None, 1),  # no month: still a case
        A("Alabama", "Autauga", "county", "March", -1),  # a downward revision
        # not cases: the state's count and the same cases on another basis
        (
            "state_control_total",
            "Alabama",
            None,
            "state",
            None,
            99.0,
            "primary",
        ),
        (
            "alternate_marginal",
            "Alabama",
            "Autauga",
            "county",
            "May",
            50.0,
            "onset",
        ),
    ]
    out, report = read(tmp_path, rows)
    assert out.rows() == [(2026, "Alabama", "Autauga", 2, None, None, "01001")]
    # the report keeps the state's own count next to the file's rows
    row = report.filter(pl.col("state") == "Alabama").row(0, named=True)
    assert (row["control_total"], row["assigned"], row["cases"]) == (
        99.0,
        2.0,
        2.0,
    )


def test_the_output_layout(tmp_path):
    out, _ = read(tmp_path, [A("Alabama", "Baldwin", "county", "May", 4)])
    assert out.columns == [
        "year",
        "state",
        "county",
        "cases",
        "indigenous",
        "imported",
        "fips",
    ]
    assert out.schema["cases"] == pl.Int64
    assert out.schema["imported"] == pl.Int64
    assert out.schema["indigenous"] == pl.Int64
    assert out.schema["fips"] == pl.String
    assert out.schema["year"] == pl.Int64


def test_a_place_that_nets_below_zero_raises(tmp_path):
    rows = [
        A("Alabama", "Autauga", "county", "May", 1),
        A("Alabama", "Autauga", "county", "June", -3),
    ]
    with pytest.raises(ValueError, match="net below zero"):
        read(tmp_path, rows)


def test_a_row_that_is_not_on_the_primary_basis_raises(tmp_path):
    with pytest.raises(ValueError, match="primary"):
        read(tmp_path, [A("Alabama", "Autauga", "county", "May", 1, "onset")])


def test_state_only_rows_and_dropped_units_are_counted_not_placed(tmp_path):
    rows = [
        A("Alabama", "Autauga", "county", "May", 3),
        A("Alabama", None, "state", "July", 1),  # known to the state only
        A("Georgia", "Metro Atlanta", "region", "May", 3),  # undefined unit
        A("South Carolina", "Upstate", "region", "July", 1),  # withheld
    ]
    out, report = read(tmp_path, rows)
    assert out["county"].to_list() == ["Autauga"]
    by = {r["state"]: r for r in report.iter_rows(named=True)}
    assert by["Alabama"]["state_only"] == 1.0
    assert by["Georgia"]["dropped_unit"] == 3.0
    assert by["South Carolina"]["dropped_unit"] == 1.0
    # kept + dropped = the file: asserted inside the reader, shown here
    assert out["cases"].sum() + 1 + 3 + 1 == 8


def test_an_unknown_multi_county_unit_raises_instead_of_being_lost(tmp_path):
    with pytest.raises(ValueError, match="not a known unit"):
        read(
            tmp_path,
            [A("Utah", "Some New District", "health_district", "May", 4)],
        )


def test_units_get_the_names_of_the_case_data(tmp_path):
    rows = [
        A("Utah", "Southwest Utah", "health_district", "May", 5),
        A("Utah", "Southwest Utah", "health_district", "June", 2),
        A("Minnesota", "Twin Cities metro area", "region", "May", 10),
        A("Tennessee", "East Tennessee Region", "health_district", "May", 6),
        A("Iowa", "HHS District 5", "health_district", "May", 7),
    ]
    out, _ = read(tmp_path, rows)
    got = {
        (r["state"], r["county"]): r["cases"]
        for r in out.iter_rows(named=True)
    }
    assert got == {
        ("Utah", "Southwest Health District, Utah"): 7,
        ("Minnesota", "Twin Cities Metro Area, Minnesota"): 10,
        ("Tennessee", "East tennessee?"): 6,
        ("Iowa", "HHS District 5, Iowa"): 7,
    }
    # a unit has no county fips
    assert out["fips"].null_count() == 4


def test_new_york_city_is_a_state_in_the_file(tmp_path):
    rows = [
        A("New York City", "New York City", "health_district", "May", 5),
        A("New York City", "New York", "county", "May", 2),  # Manhattan
    ]
    out, _ = read(tmp_path, rows)
    got = {r["county"]: r["cases"] for r in out.iter_rows(named=True)}
    # process_outbreaks_df turns "New York City" into Kings
    assert got == {"New York City": 5, "New York": 2}
    assert set(out["state"]) == {"New York"}


def test_spellings_are_those_of_the_vaccination_data(tmp_path):
    rows = [
        A("Indiana", "DeKalb", "county", "May", 1),
        A("Indiana", "LaGrange", "county", "May", 1),
        A("Pennsylvania", "McKean", "county", "May", 4),
        A("New Mexico", "Doña Ana", "county", "May", 3),
        A("Missouri", "St. Louis County", "county", "May", 1),
        A("Illinois", "DeKalb", "county", "May", 1),
    ]
    out, _ = read(tmp_path, rows)
    assert sorted(zip(out["state"], out["county"])) == [
        ("Illinois", "Dekalb"),
        ("Indiana", "Dekalb"),
        ("Indiana", "Lagrange"),
        ("Missouri", "St. Louis"),
        ("New Mexico", "Dona Ana"),
        ("Pennsylvania", "Mckean"),
    ]


def test_a_county_gets_a_fips_only_if_its_name_is_unique_in_its_state(
    tmp_path,
):
    rows = [
        A("Alabama", "Autauga", "county", "May", 1),
        A("Virginia", "Fairfax", "county", None, 2),  # a county and a city
    ]
    out, _ = read(tmp_path, rows)
    fips = {r["county"]: r["fips"] for r in out.iter_rows(named=True)}
    assert fips == {"Autauga": "01001", "Fairfax": None}
    # no row is doubled by the shared name (a plain join would double it)
    assert out["cases"].sum() == 3 and out.height == 2


def test_a_name_that_is_rewritten_keeps_the_fips_of_its_own_spelling(tmp_path):
    counties = COUNTIES.vstack(
        pl.DataFrame(
            {"fips": ["18033"], "county": ["DeKalb"], "state": ["Indiana"]}
        )
    )
    path = write(tmp_path, [A("Indiana", "DeKalb", "county", "May", 1)])
    out, _ = read_case_assignments(path, 2026, counties)
    assert out.row(0, named=True)["county"] == "Dekalb"
    assert out.row(0, named=True)["fips"] == "18033"  # looked up as "DeKalb"


def test_extra_counties_are_added_but_never_on_top_of_a_row(tmp_path):
    rows = [A("Arizona", "Maricopa", "county", "May", 25)]
    counties = pl.DataFrame(
        {
            "fips": ["04013", "04015"],
            "county": ["Maricopa", "Mohave"],
            "state": ["Arizona", "Arizona"],
        }
    )
    path = write(tmp_path, rows)
    out, report = read_case_assignments(
        path, 2026, counties, extra_counties=[("Arizona", "Mohave", 69)]
    )
    assert dict(zip(out["county"], out["cases"])) == {
        "Maricopa": 25,
        "Mohave": 69,
    }
    assert dict(zip(out["county"], out["fips"])) == {
        "Maricopa": "04013",
        "Mohave": "04015",
    }
    assert report.row(0, named=True)["extra"] == 69.0
    with pytest.raises(ValueError, match="count them twice"):
        read_case_assignments(
            path, 2026, counties, extra_counties=[("Arizona", "Maricopa", 3)]
        )


def test_the_arizona_counties_are_the_ones_without_rows_in_the_note():
    # the note: Coconino 4, Maricopa 28, Mohave 69, Pima 3, Pinal 18, Yavapai 1
    assert {c: n for _, c, n in ARIZONA_2026_COUNTIES_WITHOUT_ROWS} == {
        "Mohave": 69,
        "Pinal": 18,
        "Coconino": 4,
    }


# the real files: the numbers measured when the plan was made
DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
FILE_2026 = os.path.join(DATA, "measles_2026_case_assignments.csv")
FILE_SC = os.path.join(DATA, "measles_2025_case_assignments_SC.csv")
needs_data = pytest.mark.skipif(
    not (os.path.exists(FILE_2026) and os.path.exists(FILE_SC)),
    reason="the case files are not here",
)


@needs_data
def test_the_2026_file_reconciles_to_the_cases_of_the_file():
    counties = shapefile_counties(DATA)
    out, report = read_case_assignments(
        FILE_2026,
        2026,
        counties,
        extra_counties=ARIZONA_2026_COUNTIES_WITHOUT_ROWS,
    )
    # 3,485 assigned - 53 only known to a state - 5 in units no department
    # defines + Arizona's 91 cases in counties without a row
    assert report["assigned"].sum() == 3485
    assert report["state_only"].sum() == 53
    assert report["dropped_unit"].sum() == 5
    assert out["cases"].sum() == 3485 - 53 - 5 + 91 == 3518
    # nothing is doubled: a county of the file is one row
    assert not out.select("state", "county").is_duplicated().any()
    # only the multi-county units have no fips, and Fairfax (shared name)
    no_fips = out.filter(pl.col("fips").is_null())
    assert set(no_fips["county"]) >= {
        "Central Utah, Utah",
        "HHS District 3, Iowa",
    }
    assert "Fairfax" in set(no_fips["county"])
    assert out.filter(pl.col("state") == "Virginia")["cases"].sum() == 179
    # Maryland's 20 state-only cases are not placed; Baltimore has no fips
    assert out.filter(pl.col("state") == "Maryland")["cases"].sum() == 31


@needs_data
def test_the_south_carolina_2025_file_has_no_upstate_and_no_arizona():
    out, report = read_case_assignments(
        FILE_SC, 2025, shapefile_counties(DATA)
    )
    assert dict(zip(out["county"], out["cases"])) == {
        "Spartanburg": 204,
        "Greenville": 5,
        "Anderson": 1,
    }
    assert set(out["state"]) == {"South Carolina"}
    assert report["dropped_unit"].sum() == 3  # the withheld "Upstate" cases
    assert out["year"].unique().to_list() == [2025]
