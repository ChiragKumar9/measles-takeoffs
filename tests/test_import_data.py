"""The import records of the importation model (outbreak_size.import_data)."""

import polars as pl
import pytest

from outbreak_size.import_data import (
    attach_state_posterior,
    clean_case_records,
    clean_state_year_imports,
    stack_county_years,
    tally_state_importations,
)

# --- case records ------------------------------------------------------------


def spreadsheet_rows():
    """The spreadsheet (with the capitalized column names of the file)."""
    return pl.DataFrame(
        [
            (2015, "A", "c1", 5, 2),  # a county with known imports
            (2015, "A", None, 10, None),  # a state total, imports unknown
            (2016, "A", None, 6, 3),  # a state total with its imports
            (2016, "A", "c2", 0, 0),  # no cases
            (2016, "B", "TOTAL", 7, 1),  # not a county
            (2016, "B", "Unknown", 7, 1),
            (2016, "B", "Out of state resident", 7, 1),
            (2017, "B", None, 4, 0),  # zero imports at the state level
            (None, "B", "c3", 4, 1),  # no year
            (2017, None, "c4", 4, 1),  # no state
            (2018, "B", "c5", 3, None),  # county, imports unknown
        ],
        schema=["Year", "State", "County", "Cases", "Imported"],
        orient="row",
    )


def sheet_2025_rows():
    return pl.DataFrame(
        [
            (2025, "A", "x", 9, 1, 3),
            (2925, "A", "y", 9, None, 4),  # a typo for 2025
            (2025, "South Carolina", "z", 9, 2, 5),
            (2025, "A", "w", 9, 0, None),  # no case count
        ],
        schema=["year", "state", "county", "cases", "imported", "Lauren Case"],
        orient="row",
    )


def test_known_imports_are_counties_with_an_imported_count():
    known, _ = clean_case_records(spreadsheet_rows(), sheet_2025_rows())
    assert known.columns == ["year", "state", "county", "imported"]
    assert known.rows() == [
        (2015, "A", "c1", 2),
        (2025, "A", "x", 1),
    ]  # c5 has no count, the typo row has none, and the rest are not counties


def test_the_2025_sheet_is_corrected_and_loses_south_carolina():
    _, sheet = clean_case_records(spreadsheet_rows(), sheet_2025_rows())
    # the row without a case count and South Carolina are gone, the year typo
    # is corrected, and the cases are the sheet's own count
    assert sheet["county"].to_list() == ["x", "y"]
    assert sheet["year"].to_list() == [2025, 2025]
    assert sheet["cases"].to_list() == [3, 4]


def test_state_totals_are_not_known_county_imports():
    """A row with no county (a state total) is not a county-year, whatever its
    imported count: (2016, A) has 3 imports and (2017, B) has 0."""
    known, _ = clean_case_records(spreadsheet_rows(), sheet_2025_rows())
    assert all(county is not None for _, _, county, _ in known.rows())
    assert known["imported"].sum() == 2 + 1


# --- state-level imports -----------------------------------------------------


def test_state_names_are_spelled_one_way_and_2025_is_added():
    df = pl.DataFrame(
        {
            "State": ["DC", "North Dakora", "idaho", "New York ", "Ohio"],
            "Year": [2020] * 5,
            "Imported": [1, 2, 3, 4, 5],
        }
    )
    sheet_2025 = pl.DataFrame(
        {"state": ["Ohio"], "year": [2025], "imported": [6], "cases": [9]}
    )
    out = clean_state_year_imports(df, sheet_2025)
    assert out.columns == ["state", "year", "imported"]
    assert out["state"].to_list() == [
        "District of Columbia",
        "North Dakota",
        "Idaho",
        "New York",
        "Ohio",
        "Ohio",
    ]
    assert out["year"].to_list() == [2020] * 5 + [2025]
    assert out["imported"].to_list() == [1, 2, 3, 4, 5, 6]


# --- tally by state ----------------------------------------------------------


def state_years(years, states=("A", "B")):
    """One import per state in each year, 10 more for A in 2022."""
    rows = [
        {
            "state": s,
            "year": y,
            "imported": 1 + (10 if (s, y) == ("A", 2022) else 0),
        }
        for s in states
        for y in years
    ]
    return pl.DataFrame(rows)


def test_imports_before_each_year_are_tallied_by_state():
    out, years_of_data = tally_state_importations(
        state_years(range(2019, 2026))
    )
    assert years_of_data == 7
    a = out.filter(pl.col("state") == "A").row(0, named=True)
    b = out.filter(pl.col("state") == "B").row(0, named=True)
    # 2019 and 2020 before 2021; plus 2021 (and A's 11 in 2022) before 2023
    assert (a["imported_pre_2021"], b["imported_pre_2021"]) == (2, 2)
    assert (a["imported_pre_2022"], b["imported_pre_2022"]) == (3, 3)
    assert (a["imported_pre_2023"], b["imported_pre_2023"]) == (14, 4)
    assert (a["imported_pre_2025"], b["imported_pre_2025"]) == (16, 6)
    assert (a["imported"], b["imported"]) == (17, 7)  # all years


@pytest.mark.parametrize(
    "years",
    [
        [2019, 2020, 2021, 2023, 2024, 2025],  # 2022 is missing
        list(range(2019, 2025)),  # does not end in 2025
    ],
)
def test_the_years_must_be_consecutive_and_end_in_2025(years):
    with pytest.raises(ValueError, match="every year from the first to 2025"):
        tally_state_importations(state_years(years))


def test_a_row_without_a_year_is_refused():
    df = state_years(range(2019, 2026)).vstack(
        pl.DataFrame(
            {"state": ["A"], "year": [None], "imported": [1]},
            schema_overrides={"year": pl.Int64},
        )
    )
    with pytest.raises(ValueError, match="rows without a year: 1"):
        tally_state_importations(df)


# --- the state's posterior of each county-year -------------------------------


def validation_rows():
    return pl.DataFrame(
        {
            "state": ["B", "A", "B"],
            "share": [0.5, 0.25, 0.1],
            "share_log": [0.4, 0.2, 0.05],
        }
    )


def posterior(states=("A", "B")):
    return pl.DataFrame(
        {
            "state": list(states),
            "posterior_alpha_pre_2021": [4.0, 8.0][: len(states)],
            "posterior_beta_pre_2021": [2.0, 3.0][: len(states)],
        }
    )


def test_each_county_gets_its_states_posterior_scaled_by_its_share():
    out = attach_state_posterior(validation_rows(), posterior())
    # the rows keep their order; B's alpha is 8 and A's is 4
    assert out["state"].to_list() == ["B", "A", "B"]
    assert out["posterior_alpha_pre_2021"].to_list() == pytest.approx(
        [8 * 0.5, 4 * 0.25, 8 * 0.1]
    )
    assert out["posterior_alpha_pre_2021_log"].to_list() == pytest.approx(
        [8 * 0.4, 4 * 0.2, 8 * 0.05]
    )
    # the rate parameter is the state's own
    assert out["posterior_beta_pre_2021"].to_list() == [3.0, 2.0, 3.0]


def test_a_state_without_a_posterior_is_refused():
    with pytest.raises(ValueError, match=r"\['B'\]"):
        attach_state_posterior(validation_rows(), posterior(states=("A",)))


# --- the county-years of the model ------------------------------------------


def county_frame(year_of_vaccination, kind="projection", fips="01001", pop=10):
    return pl.DataFrame(
        {
            "fips": [fips],
            "state": ["A"],
            "county": ["x"],
            "year": [year_of_vaccination],
            "pop": [pop],
            "type": [kind],
            "extra": [1],
        }
    )


def test_rows_are_marked_with_their_validation_year_and_only_the_model_columns_are_kept():
    out = stack_county_years(
        {2021: county_frame(2020), 2022: county_frame(2021)}
    )
    assert out.columns == [
        "fips",
        "state",
        "county",
        "year",
        "validation_year",
        "pop",
        "type",
    ]
    assert out["validation_year"].to_list() == [2021, 2022]
    assert out["year"].to_list() == [2020, 2021]


def test_the_2026_rows_keep_the_year_of_their_vaccination_data():
    """The 2025 and 2026 projection rows of a county both carry the year of
    its vaccination data (2024); they are told apart by validation year."""
    out = stack_county_years(
        {2025: county_frame(2024), 2026: county_frame(2024)}
    )
    assert out["year"].to_list() == [2024, 2024]
    assert out["validation_year"].to_list() == [2025, 2026]


def test_a_projection_row_may_not_use_vaccination_data_of_its_own_year():
    with pytest.raises(ValueError, match="2025 or later"):
        stack_county_years({2025: county_frame(2025)})
    # a train row of that year is not a projection row
    stack_county_years({2025: county_frame(2025, kind="train")})


def test_a_county_with_older_data_has_a_row_for_each_validation_year():
    # the same (fips, year of data, projection) in two datasets: two rows,
    # each with the population of its own dataset
    out = stack_county_years(
        {
            2022: county_frame(2020, pop=10),
            2023: county_frame(2020, pop=99),
        }
    )
    assert out["validation_year"].to_list() == [2022, 2023]
    assert out["pop"].to_list() == [10, 99]


def test_a_repeated_train_row_of_a_dataset_is_kept_once():
    frame = pl.concat(
        [county_frame(2021, kind="train"), county_frame(2021, kind="train")]
    )
    assert stack_county_years({2021: frame}).height == 1


def test_a_train_and_a_projection_row_of_a_county_are_both_kept():
    frame = pl.concat([county_frame(2021, kind="train"), county_frame(2020)])
    out = stack_county_years({2021: frame})
    assert sorted(out["type"].to_list()) == ["projection", "train"]


def test_a_county_twice_in_a_projection_dataset_is_refused():
    frame = pl.concat([county_frame(2020), county_frame(2020)])
    with pytest.raises(ValueError, match="county twice"):
        stack_county_years({2021: frame})
