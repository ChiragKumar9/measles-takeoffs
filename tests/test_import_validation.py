"""Which county-years the importation model is validated on (outbreak_size.import_validation)."""

import polars as pl
import pytest

from outbreak_size import import_validation as iv

RESOLVED_SCHEMA = {
    "year": pl.Int64,
    "state": pl.String,
    "imported": pl.Int64,
    "place": pl.String,
    "members": pl.List(pl.String),
}
DROPPED_SCHEMA = {
    "year": pl.Int64,
    "state": pl.String,
    "imported": pl.Int64,
    "county": pl.String,
}
YEARS = [2021, 2022, 2023, 2024]


@pytest.fixture
def universe():
    # state A: counties a1, a2, a3 (populations 100, 200, 700);
    # state B: b1, b2 (300, 700)
    return pl.DataFrame(
        {
            "fips": ["01001", "01002", "01003", "02001", "02002"],
            "state": ["A", "A", "A", "B", "B"],
            "county": ["a1", "a2", "a3", "b1", "b2"],
        }
    )


@pytest.fixture
def populations(universe):
    pops = [100, 200, 700, 300, 700]
    return pl.DataFrame(
        [
            {"fips": f, "year": y, "pop": float(p)}
            for y in YEARS
            for f, p in zip(universe["fips"], pops)
        ]
    )


@pytest.fixture
def resolved():
    return pl.DataFrame(
        [
            # complete: the state's total is 2
            dict(
                year=2021, state="A", imported=2, place="a1", members=["01001"]
            ),
            # the state's total is 2: one import is unattributed
            dict(
                year=2022, state="A", imported=1, place="a1", members=["01001"]
            ),
            # a region unit of two counties, complete
            dict(
                year=2023,
                state="A",
                imported=1,
                place="r1",
                members=["01001", "01002"],
            ),
            # plus a dropped positive row (see `dropped`)
            dict(
                year=2024, state="A", imported=0, place="a1", members=["01001"]
            ),
        ],
        schema=RESOLVED_SCHEMA,
    )


@pytest.fixture
def dropped():
    return pl.DataFrame(
        [
            dict(year=2024, state="A", imported=1, county="zzz"),
            dict(year=2023, state="B", imported=1, county="yyy"),
        ],
        schema=DROPPED_SCHEMA,
    )


@pytest.fixture
def totals():
    # B has no row in 2024: no imports that year
    return pl.DataFrame(
        [
            dict(state="A", year=2021, total=2),
            dict(state="A", year=2022, total=2),
            dict(state="A", year=2023, total=1),
            dict(state="A", year=2024, total=1),
            dict(state="B", year=2021, total=0),
            dict(state="B", year=2022, total=3),
            dict(state="B", year=2023, total=1),
        ]
    )


@pytest.fixture
def out(resolved, dropped, totals, universe, populations):
    return iv.validation_county_years(
        resolved, dropped, totals, universe, populations, YEARS
    )


def rows(out, year, state):
    return out.filter((pl.col("year") == year) & (pl.col("state") == state))


def test_a_state_year_fully_attributed_to_named_counties_has_zeros_elsewhere(
    out,
):
    got = rows(out, 2021, "A")
    assert sorted(zip(got["place"], got["imported"])) == [
        ("01002", 0),
        ("01003", 0),
        ("a1", 2),
    ]


def test_a_state_year_without_imports_has_zeros_everywhere(out):
    got = rows(out, 2021, "B")
    assert sorted(got["place"]) == ["02001", "02002"]
    assert got["imported"].sum() == 0
    assert rows(out, 2024, "B").height == 2  # absent from the totals


def test_unattributed_imports_leave_the_other_counties_unknown(out):
    assert rows(out, 2022, "A")["place"].to_list() == ["a1"]
    assert rows(out, 2022, "B").height == 0  # imports, no county rows
    assert rows(out, 2024, "A")["place"].to_list() == [
        "a1"
    ]  # dropped positive


def test_a_dropped_positive_row_makes_the_state_unknown(out):
    assert rows(out, 2023, "B").height == 0


def test_a_region_is_one_place_with_the_population_of_its_counties(out):
    got = rows(out, 2023, "A")
    assert sorted(zip(got["place"], got["pop"], got["n_counties"])) == [
        ("01003", 700.0, 1),
        ("r1", 300.0, 2),
    ]


def test_shares_are_population_shares_of_the_state(out):
    a21 = rows(out, 2021, "A")
    assert a21.filter(pl.col("place") == "a1")["share"][0] == pytest.approx(
        0.1
    )
    assert rows(out, 2023, "A").filter(pl.col("place") == "r1")["share"][
        0
    ] == pytest.approx(0.3)
    assert sorted(rows(out, 2021, "B")["share"]) == pytest.approx([0.3, 0.7])
    assert a21["share_log"].sum() == pytest.approx(1.0)


def test_a_county_without_a_population_raises(
    resolved, dropped, totals, universe, populations
):
    with pytest.raises(ValueError, match="no population"):
        iv.validation_county_years(
            resolved,
            dropped,
            totals,
            universe,
            populations.filter(pl.col("fips") != "01003"),
            [2021],
        )


def test_a_state_spelled_differently_from_the_universe_raises(
    resolved, dropped, totals, universe, populations
):
    misspelled = resolved.with_columns(state=pl.lit("A "))
    with pytest.raises(ValueError, match="not in the county universe"):
        iv.validation_county_years(
            misspelled, dropped, totals, universe, populations, YEARS
        )


def test_overlapping_places_raise(
    resolved, dropped, totals, universe, populations
):
    overlapping = pl.concat(
        [
            resolved,
            pl.DataFrame(
                [
                    dict(
                        year=2021,
                        state="A",
                        imported=0,
                        place="r2",
                        members=["01001", "01003"],
                    )
                ],
                schema=RESOLVED_SCHEMA,
            ),
        ]
    )
    with pytest.raises(ValueError, match="overlaps"):
        iv.validation_county_years(
            overlapping, dropped, totals, universe, populations, YEARS
        )


def test_only_the_missing_populations_are_asked_for(universe, populations):
    have = populations.filter(
        (pl.col("fips") != "01003") & (pl.col("year") == 2021)
    )
    asked = []

    def query(years, fips):
        asked.append(list(fips))
        return [111.0] * len(fips)

    filled = iv.fill_missing_populations(universe, [2021], have, query)
    assert asked == [["01003"]]
    assert filled.height == 5
    assert filled.filter(pl.col("fips") == "01003")["pop"][0] == 111.0


def test_known_rows_are_mapped_to_places():
    universe = pl.DataFrame(
        {
            "fips": [
                "01001",
                "01003",
                "26161",
                "36005",
                "36047",
                "36061",
                "36081",
                "36085",
            ],
            "state": ["Alabama", "Alabama", "Michigan"] + ["New York"] * 5,
            "county": [
                "Autauga",
                "Baldwin",
                "Washtenaw",
                "Bronx",
                "Kings",
                "New York",
                "Queens",
                "Richmond",
            ],
        }
    )
    known = pl.DataFrame(
        [
            dict(year=2024, state="Michigan", county="Washentaw", imported=1),
            dict(
                year=2024, state="New York", county="New York City", imported=3
            ),
            dict(
                year=2024, state="Alabama", county="North, Alabama", imported=1
            ),
            dict(
                year=2024, state="Alabama", county="Baldwin County", imported=0
            ),
            dict(year=2024, state="Alabama", county="Nowhere", imported=1),
        ]
    )
    resolved, dropped = iv.resolve_known_rows(
        known, universe, {"North, Alabama": ["01001"]}
    )
    members = {
        r["place"]: r["members"] for r in resolved.iter_rows(named=True)
    }
    assert members["Washtenaw"] == ["26161"]  # misspelled in the data
    assert len(members["New York City"]) == 5
    assert members["North, Alabama"] == ["01001"]  # a region of counties
    assert members["Baldwin County"] == ["01003"]  # matched without "County"
    assert dropped["county"].to_list() == ["Nowhere"]
