"""The importation data and the year of its posterior in 2026 and after."""

import polars as pl
import pytest

from outbreak_size.import_data import (
    clean_state_year_imports,
    stack_county_years,
)
from outbreak_size.importation import IMPORT_CONDITIONING_VALIDATION_YEAR


def test_the_projections_after_2025_use_the_importation_rows_of_validation_year_2026():
    # imports are known through 2025, so 2026 is the posterior of all years
    assert IMPORT_CONDITIONING_VALIDATION_YEAR == 2026


def test_a_state_spreadsheet_with_2025_rows_is_refused():
    # the 2025 sheet is added on top of the spreadsheet: 2025 rows in both
    # would be counted twice
    sheet = pl.DataFrame({"State": ["Ohio"], "Year": [2025], "Imported": [3]})
    sheet_2025 = pl.DataFrame(
        {"state": ["Ohio"], "year": [2025], "imported": [6], "cases": [9]}
    )
    with pytest.raises(ValueError, match="years from 2025 on"):
        clean_state_year_imports(sheet, sheet_2025)


def county_frame(data_year, kind="projection"):
    return pl.DataFrame(
        {
            "fips": ["01001"],
            "state": ["A"],
            "county": ["x"],
            "year": [data_year],
            "pop": [10],
            "type": [kind],
        }
    )


def test_the_2026_projection_rows_may_not_use_vaccination_data_of_2026():
    # 2026 is a validation year now: its rows use data from before it
    stack_county_years({2026: county_frame(2025)})
    with pytest.raises(ValueError, match="2026 or later"):
        stack_county_years({2026: county_frame(2026)})
    # a train row of that year is not a projection row
    stack_county_years({2026: county_frame(2026, kind="train")})
