"""Cleaning of the outbreak records (outbreak_size.outbreaks)."""

import polars as pl

from outbreak_size.outbreaks import process_outbreaks_df


def outbreaks(rows):
    """rows: (year, state, county, cases, indigenous, imported, fips)."""
    return pl.DataFrame(
        [
            dict(
                year=y,
                state=s,
                county=c,
                cases=n,
                indigenous=i,
                imported=m,
                fips=f,
                url="https://example.org",  # a source column that is dropped
            )
            for y, s, c, n, i, m, f in rows
        ],
        schema={
            "year": pl.Int64,
            "state": pl.String,
            "county": pl.String,
            "cases": pl.Int64,
            "indigenous": pl.Int64,
            "imported": pl.Int64,
            "fips": pl.String,
            "url": pl.String,
        },
    )


def clean(rows):
    return process_outbreaks_df(outbreaks(rows)).sort(
        "year", "state", "county"
    )


def test_years_before_2011_empty_years_and_zero_case_rows_are_dropped():
    out = clean(
        [
            (2010, "Texas", "Gaines", 5, 5, 0, "48165"),
            (None, "Texas", "TOTAL", 99, 90, 9, None),
            (2015, "Texas", "Gaines", 0, 0, 0, "48165"),
            (2015, "Texas", "Lubbock", 3, 3, 0, "48303"),
        ]
    )
    assert out["county"].to_list() == ["Lubbock"]


def test_fips_codes_are_padded_to_five_characters():
    out = clean(
        [
            (2015, "Alabama", "Autauga", 4, 4, 0, "1001"),
            (2015, "Alabama", "Statewide", 2, 2, 0, "01"),
        ]
    )
    assert dict(zip(out["county"], out["fips"])) == {
        "Autauga": "01001",
        "Statewide": "01",
    }


def test_delaware_is_one_place_for_the_state():
    out = clean(
        [
            (2015, "Delaware", "Kent", 2, 2, 0, "10001"),
            (2015, "Delaware", "Sussex", 3, 1, 2, "10005"),
        ]
    )
    assert out["county"].to_list() == ["State"]
    assert out["cases"].to_list() == [5]
    assert out["imported"].to_list() == [2]


def test_dc_cases_are_given_a_county_name():
    out = clean([(2015, "District of Columbia", None, 3, 3, 0, "11001")])
    assert out["county"].to_list() == ["District of Columbia"]


def test_rows_without_a_real_county_are_dropped():
    out = clean(
        [
            (2015, "Ohio", None, 5, 5, 0, None),
            (2015, "Ohio", "Unknown", 5, 5, 0, None),
            (2015, "Ohio", "TOTAL", 5, 5, 0, None),
            (2015, "Ohio", "Out of state resident", 5, 5, 0, None),
            (2015, "Ohio", "Holmes", 5, 5, 0, "39075"),
        ]
    )
    assert out["county"].to_list() == ["Holmes"]


def test_misspelled_places_are_corrected():
    out = clean(
        [
            (2015, "California", "San Bernadino", 1, 1, 0, None),
            (2015, "Michigan", "Washentaw", 2, 2, 0, None),
            (2015, "Minnesota", "LaSueur", 3, 3, 0, None),
            (2015, "Washington", "Spokane\xa0", 4, 4, 0, None),
            (2015, "Missouri", "St Louis City", 5, 5, 0, None),
        ]
    )
    assert sorted(out["county"]) == [
        "Le Sueur",
        "San Bernardino",
        "Spokane",
        "St. Louis City",
        "Washtenaw",
    ]


def test_new_york_city_cases_are_kings_county_cases():
    out = clean(
        [
            (2019, "New York", "New York City", 10, 10, 0, None),
            (2019, "New York", "Kings", 2, 2, 0, "36047"),
        ]
    )
    assert out["county"].to_list() == ["Kings"]
    assert out["cases"].to_list() == [12]


def test_detroit_is_added_to_wayne_county():
    out = clean(
        [
            (2015, "Michigan", "Wayne", 6, 5, 1, "26163"),
            (2015, "Michigan", "Detroit City", 4, 4, 0, None),
        ]
    )
    assert out["county"].to_list() == ["Wayne"]
    assert out["cases"].to_list() == [10]
    assert out["indigenous"].to_list() == [9]
    # the place keeps the fips of its first row, so Wayne's row comes first
    assert out["fips"].to_list() == ["26163"]


def test_missing_indigenous_and_imported_counts_are_summed_as_zero():
    # the first sum over a county-year turns counts that are missing in every
    # row into 0 (a polars sum of nothing)
    out = clean(
        [
            (2015, "Maine", "Cumberland", 4, None, None, "23005"),
            (2015, "Maine", "Cumberland", 3, None, None, "23005"),
        ]
    )
    assert out["cases"].to_list() == [7]
    assert out["indigenous"].to_list() == [0]
    assert out["imported"].to_list() == [0]


def test_imports_are_known_only_when_every_row_of_the_county_year_has_a_count():
    out = clean(
        [
            (2015, "Maine", "Cumberland", 4, None, None, "23005"),  # unknown
            (2015, "Maine", "Kennebec", 2, 2, 0, "23011"),  # recorded as 0
            (2015, "Maine", "York", 5, 3, 2, "23031"),  # recorded
            (2016, "Maine", "York", 3, 1, 2, "23031"),  # recorded
            (2016, "Maine", "York", 1, 1, None, "23031"),  # one row missing
        ]
    )
    assert out["imports_known"].to_list() == [False, True, True, False]
    # the imported column itself is as before
    assert out["imported"].to_list() == [0, 0, 2, 2]


def test_a_county_year_merged_from_two_places_is_known_only_if_both_are():
    out = clean(
        [
            (2015, "Michigan", "Wayne", 6, 5, 1, "26163"),
            (2015, "Michigan", "Detroit City", 4, 4, None, None),
            (2019, "New York", "New York City", 10, 10, 0, None),
            (2019, "New York", "Kings", 2, 2, 0, "36047"),
        ]
    )
    assert out["county"].to_list() == ["Wayne", "Kings"]
    assert out["imports_known"].to_list() == [False, True]


def test_outbreaks_among_specific_populations_are_removed():
    out = clean(
        [
            (2024, "Illinois", "Cook", 50, 50, 0, "17031"),
            (2023, "Illinois", "Cook", 5, 5, 0, "17031"),
            (2021, "Wisconsin", "Monroe", 8, 8, 0, "55081"),
            (2016, "Arizona", "Pinal", 6, 6, 0, "04021"),
            (2022, "Arizona", "Pinal", 2, 2, 0, "04021"),
        ]
    )
    assert sorted(zip(out["year"], out["county"])) == [
        (2022, "Pinal"),
        (2023, "Cook"),
    ]
