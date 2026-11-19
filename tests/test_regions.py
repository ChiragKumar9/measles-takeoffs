"""The multi-county regions (outbreak_size.regions)."""

import os
from collections import Counter

import polars as pl
import pytest

from outbreak_size.regions import (
    expand_to_member_counties,
    expand_to_residual_members,
    higher_level_regions,
    members_with_own_row,
    members_without_own_row,
    require_filled_rows_use_same_members,
    require_no_own_row_members,
    require_own_row_members_in_main_rows,
)

# Fulton County (13121) is in the two Georgia places of the case data: the
# 2024 row "Fulton, Gwinett" and the 2019 row "Forsyth, Fulton, Paulding".
# They are places of different years, so no year has a county in both.
ALLOWED_SHARED_COUNTIES = {
    "13121": {"Fulton, Gwinett", "Forsyth, Fulton, Paulding"}
}


def test_every_region_is_a_list_of_county_fips_of_five_characters():
    regions = higher_level_regions()
    assert regions
    for name, fips in regions.items():
        assert fips, name
        assert all(isinstance(f, str) and len(f) == 5 for f in fips), name
        assert all(f.isdigit() for f in fips), name


def test_a_county_is_in_at_most_one_region():
    regions = higher_level_regions()
    counts = Counter(f for v in regions.values() for f in v)
    shared = {f for f, n in counts.items() if n > 1}
    # the only shared county is the documented one, in exactly those regions
    assert shared == set(ALLOWED_SHARED_COUNTIES)
    for fips, names in ALLOWED_SHARED_COUNTIES.items():
        assert {k for k, v in regions.items() if fips in v} == names


def test_a_region_is_inside_one_state():
    # the first two characters of a county FIPS are its state's
    for name, fips in higher_level_regions().items():
        states = {f[:2] for f in fips}
        # the Gulf Coast is not one region of one state; every other region
        # is within one state
        if name == "Gulf Coast":
            continue
        assert len(states) == 1, (name, states)


# the counties (by name, as the Census names them) of the regions that were
# corrected or added for the 2026 case data, from the sources in regions.py
EXPECTED_COUNTIES = {
    "Mid-Cumberland Region, Tennessee": [
        "Cheatham",
        "Dickson",
        "Houston",
        "Humphreys",
        "Montgomery",
        "Robertson",
        "Rutherford",
        "Stewart",
        "Sumner",
        "Trousdale",
        "Williamson",
        "Wilson",
    ],
    "Upper Cumberland Region, Tennessee": [
        "Cannon",
        "Clay",
        "Cumberland",
        "DeKalb",
        "Fentress",
        "Jackson",
        "Macon",
        "Overton",
        "Pickett",
        "Putnam",
        "Smith",
        "Van Buren",
        "Warren",
        "White",
    ],
    "East tennessee?": [
        "Anderson",
        "Blount",
        "Campbell",
        "Claiborne",
        "Cocke",
        "Grainger",
        "Hamblen",
        "Jefferson",
        "Loudon",
        "Monroe",
        "Morgan",
        "Roane",
        "Scott",
        "Sevier",
        "Union",
    ],
    "Twin Cities Metro Area, Minnesota": [
        "Anoka",
        "Carver",
        "Dakota",
        "Hennepin",
        "Ramsey",
        "Scott",
        "Washington",
    ],
    "Region 9, Louisiana": [
        "Livingston",
        "St. Helena",
        "St. Tammany",
        "Tangipahoa",
        "Washington",
    ],
    "North, Alabama": [
        "Colbert",
        "Cullman",
        "Franklin",
        "Jackson",
        "Lauderdale",
        "Lawrence",
        "Limestone",
        "Madison",
        "Marion",
        "Marshall",
        "Morgan",
        "Winston",
    ],
    "Fulton, Gwinett": ["Fulton", "Gwinnett"],
    "Forsyth, Fulton, Paulding": ["Forsyth", "Fulton", "Paulding"],
    "Central Utah, Utah": [
        "Juab",
        "Millard",
        "Piute",
        "Sanpete",
        "Sevier",
        "Wayne",
    ],
    "TriCounty, Utah": ["Daggett", "Duchesne", "Uintah"],
    "Weber-Morgan, Utah": ["Weber", "Morgan"],
    "HHS District 3, Iowa": [
        "Mitchell",
        "Howard",
        "Winneshiek",
        "Allamakee",
        "Cerro Gordo",
        "Floyd",
        "Chickasaw",
        "Fayette",
        "Clayton",
        "Franklin",
        "Butler",
        "Bremer",
        "Hardin",
        "Grundy",
        "Marshall",
        "Tama",
    ],
    "HHS District 4, Iowa": [
        "Shelby",
        "Audubon",
        "Guthrie",
        "Pottawattamie",
        "Cass",
        "Adair",
        "Mills",
        "Montgomery",
        "Adams",
        "Union",
        "Fremont",
        "Page",
        "Taylor",
        "Ringgold",
    ],
    "HHS District 5, Iowa": [
        "Hamilton",
        "Greene",
        "Boone",
        "Story",
        "Dallas",
        "Polk",
        "Jasper",
        "Madison",
        "Warren",
        "Marion",
        "Clarke",
        "Lucas",
        "Decatur",
        "Wayne",
    ],
    "HHS District 6, Iowa": [
        "Benton",
        "Poweshiek",
        "Iowa",
        "Mahaska",
        "Keokuk",
        "Washington",
        "Monroe",
        "Wapello",
        "Jefferson",
        "Henry",
        "Appanoose",
        "Davis",
        "Van Buren",
        "Lee",
    ],
    "Coastal Health District, Georgia": [
        "Bryan",
        "Camden",
        "Chatham",
        "Effingham",
        "Glynn",
        "Liberty",
        "Long",
        "McIntosh",
    ],
    "Northeast HCC Region, Kansas": [
        "Anderson",
        "Atchison",
        "Brown",
        "Chase",
        "Coffey",
        "Doniphan",
        "Douglas",
        "Franklin",
        "Geary",
        "Jackson",
        "Jefferson",
        "Linn",
        "Lyon",
        "Marshall",
        "Miami",
        "Morris",
        "Nemaha",
        "Osage",
        "Pottawatomie",
        "Riley",
        "Shawnee",
        "Wabaunsee",
        "Washington",
    ],
}
SHAPEFILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "cb_2020_us_county_20m.shp",
)


def test_the_counts_of_the_corrected_and_new_regions():
    regions = higher_level_regions()
    for name, counties in EXPECTED_COUNTIES.items():
        assert len(regions[name]) == len(counties), name
        assert len(set(regions[name])) == len(counties), name  # none twice


@pytest.mark.skipif(
    not os.path.exists(SHAPEFILE), reason="the county shapefile is not here"
)
def test_the_counties_of_the_corrected_and_new_regions_are_the_named_ones():
    import geopandas as gpd

    shapes = gpd.read_file(SHAPEFILE)
    name_of = dict(zip(shapes["GEOID"], shapes["NAME"]))
    regions = higher_level_regions()
    for name, counties in EXPECTED_COUNTIES.items():
        got = {name_of[f] for f in regions[name]}  # every fips is a county
        assert got == set(counties), name


SCHEMA = {
    "year": pl.Int64,
    "state": pl.String,
    "county": pl.String,
    "cases": pl.Int64,
    "fips": pl.String,
}


def test_a_place_with_one_row_is_that_row_for_every_county():
    # the shape the old code gave: the row repeated, with each fips in turn
    # (pl.concat([row] * n).with_columns(fips=pl.Series(fips)))
    row = pl.DataFrame(
        {
            "year": [2019],
            "state": ["T"],
            "county": ["East"],
            "cases": [5],
            "fips": [None],
        },
        schema=SCHEMA,
    )
    fips = ["47001", "47009", "47013"]
    old = pl.concat([row] * len(fips), how="diagonal_relaxed").with_columns(
        fips=pl.Series(fips)
    )
    new = expand_to_member_counties(row, fips).select(old.columns)
    assert new.equals(old)


def test_a_place_with_rows_in_two_years_gets_every_county_in_each_year():
    rows = pl.DataFrame(
        {
            "year": [2025, 2026],
            "state": ["T", "T"],
            "county": ["East", "East"],
            "cases": [5, 6],
            "fips": [None, None],
        },
        schema=SCHEMA,
    )
    fips = ["47001", "47009", "47013"]
    out = expand_to_member_counties(rows, fips)
    assert out.height == 6
    for year in (2025, 2026):
        assert sorted(out.filter(pl.col("year") == year)["fips"]) == fips


# --- the members that have a case row of their own ----------------------------

OWN_REGIONS = {"R": ["01001", "01003", "01005"]}


def case_rows(rows):
    """rows: (year, county, fips) of the case data of one state."""
    return pl.DataFrame(
        [dict(year=y, state="S", county=c, fips=f) for y, c, f in rows],
        schema={
            "year": pl.Int64,
            "state": pl.String,
            "county": pl.String,
            "fips": pl.String,
        },
    )


def county_names(rows):
    """rows: (county, fips) of the vaccination data of one state."""
    return pl.DataFrame(
        [dict(state="S", county=c, fips=f) for c, f in rows],
        schema={"state": pl.String, "county": pl.String, "fips": pl.String},
    )


NAMES = county_names([("a", "01001"), ("b", "01003"), ("c", "01005")])


def test_a_member_with_a_case_row_is_found_by_its_fips_or_by_its_name():
    cases = case_rows(
        [
            (2025, "R", None),  # the region itself
            (2025, "a", "01001"),  # a member, by the fips of the case row
            (2025, "b", None),  # a member, by its name
            (2025, "z", "01099"),  # not a member
            (2026, "c", None),
        ]
    )
    out = members_with_own_row(cases, OWN_REGIONS, NAMES)
    assert out.rows() == [
        ("R", 2025, "01001"),
        ("R", 2025, "01003"),
        ("R", 2026, "01005"),
    ]


def test_a_name_with_two_fips_places_nothing_and_the_region_row_is_skipped():
    names = county_names([("b", "01003"), ("b", "01099")])
    cases = case_rows([(2025, "b", None), (2025, "R", "01001")])
    assert members_with_own_row(cases, OWN_REGIONS, names).height == 0


def test_the_vaccination_datas_fips_wins_over_the_case_sheets_as_in_the_merge():
    # the merge gives a case row the vaccination row's fips; the sheet's own
    # (here a wrong one) is only the fallback for a name the vaccination data
    # does not place
    wrong = case_rows([(2025, "b", "01099")])
    assert members_with_own_row(wrong, OWN_REGIONS, NAMES).rows() == [
        ("R", 2025, "01003")
    ]
    # a name with two fips places nothing: the sheet's fips is the fallback
    ambiguous = county_names([("b", "01003"), ("b", "01099")])
    sheet = case_rows([(2025, "b", "01003")])
    assert members_with_own_row(sheet, OWN_REGIONS, ambiguous).rows() == [
        ("R", 2025, "01003")
    ]


def test_a_region_stands_for_the_members_without_a_case_row_of_their_own():
    own = members_with_own_row(
        case_rows([(2025, "a", "01001"), (2025, "c", "01005")]),
        OWN_REGIONS,
        NAMES,
    )
    assert members_without_own_row(OWN_REGIONS, own, "R", 2025) == ["01003"]
    # another year: every member
    assert members_without_own_row(OWN_REGIONS, own, "R", 2026) == [
        "01001",
        "01003",
        "01005",
    ]


def test_a_place_with_rows_in_two_years_expands_to_each_years_members():
    rows = pl.DataFrame(
        {
            "year": [2025, 2026],
            "state": ["S", "S"],
            "county": ["R", "R"],
            "cases": [5, 6],
            "fips": [None, None],
        },
        schema=SCHEMA,
    )
    own = members_with_own_row(
        case_rows([(2025, "a", "01001")]), OWN_REGIONS, NAMES
    )
    out = expand_to_residual_members(rows, "R", OWN_REGIONS, own)
    assert sorted(out.filter(pl.col("year") == 2025)["fips"]) == [
        "01003",
        "01005",
    ]
    assert sorted(out.filter(pl.col("year") == 2026)["fips"]) == [
        "01001",
        "01003",
        "01005",
    ]


def test_a_place_whose_members_all_have_a_row_of_their_own_is_refused():
    rows = pl.DataFrame(
        {
            "year": [2025],
            "state": ["S"],
            "county": ["R"],
            "cases": [5],
            "fips": [None],
        },
        schema=SCHEMA,
    )
    own = members_with_own_row(
        case_rows(
            [(2025, "a", "01001"), (2025, "b", "01003"), (2025, "c", "01005")]
        ),
        OWN_REGIONS,
        NAMES,
    )
    with pytest.raises(ValueError, match="every member county"):
        expand_to_residual_members(rows, "R", OWN_REGIONS, own)


def used(rows):
    return pl.DataFrame(
        [dict(place="R", year=y, fips=f) for y, f in rows],
        schema={"place": pl.String, "year": pl.Int64, "fips": pl.String},
    )


def test_a_county_in_a_region_and_in_a_row_of_its_own_is_refused():
    with pytest.raises(ValueError, match="case row of their own"):
        require_no_own_row_members(
            used([(2025, "01001")]),
            case_rows([(2025, "a", "01001")]).select("year", "fips"),
        )


def test_the_same_county_in_another_year_or_without_a_fips_is_no_overlap():
    require_no_own_row_members(
        used([(2025, "01001")]),
        case_rows([(2026, "a", "01001"), (2025, "b", None)]).select(
            "year", "fips"
        ),
    )


def filled(year, is_filled):
    return pl.DataFrame(
        {
            "county": ["R"],
            "year": [year],
            "vaccine_from_earlier_year": [is_filled],
        }
    )


def test_a_filled_row_built_for_other_counties_is_refused():
    # 2025: a has a row of its own; 2026: none has. A 2026 row that took the
    # school year 2024 (built for the cases of 2025) would describe only b, c
    own = members_with_own_row(
        case_rows([(2025, "a", "01001")]), OWN_REGIONS, NAMES
    )
    with pytest.raises(ValueError, match="built for other counties"):
        require_filled_rows_use_same_members(
            filled(2026, True), OWN_REGIONS, own
        )
    # not filled: nothing to check
    require_filled_rows_use_same_members(filled(2026, False), OWN_REGIONS, own)
    # the same members in both years: fine
    require_filled_rows_use_same_members(filled(2027, True), OWN_REGIONS, own)


def region_years(rows):
    return pl.DataFrame(
        [dict(place="R", year=y) for y in rows],
        schema={"place": pl.String, "year": pl.Int64},
    )


def main(rows):
    return pl.DataFrame(
        [dict(year=y, fips=f) for y, f in rows],
        schema={"year": pl.Int64, "fips": pl.String},
    )


def test_a_member_without_a_main_row_is_refused_in_a_year_of_the_region():
    own = members_with_own_row(
        case_rows([(2025, "a", "01001")]), OWN_REGIONS, NAMES
    )
    with pytest.raises(ValueError, match="no row with vaccine data"):
        require_own_row_members_in_main_rows(
            own, region_years([2025]), main([(2025, "01003")])
        )


def test_a_member_with_a_main_row_or_in_a_year_without_the_region_is_fine():
    own = members_with_own_row(
        case_rows([(2025, "a", "01001")]), OWN_REGIONS, NAMES
    )
    # it has a main row
    require_own_row_members_in_main_rows(
        own, region_years([2025]), main([(2025, "01001")])
    )
    # the region has no row that year, so nothing is left out of it
    require_own_row_members_in_main_rows(own, region_years([2026]), main([]))
