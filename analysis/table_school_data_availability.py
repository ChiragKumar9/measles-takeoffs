"""
School-level vaccination data available in each year, and the measles cases in
the counties that have it.

A school year is matched to the outbreak data by its end year (2023-2024 is
the year 2024), as in data/school_vaccination_counties.csv (year_end).

Run from the repo root, after analysis/assemble_school_vaccination.py:
    uv run analysis/table_school_data_availability.py
Writes outputs/school_vaccination/school_data_availability_by_year.csv
"""

import os

import polars as pl

root = os.path.dirname(os.path.dirname(__file__))
output_dir = os.path.join(root, "outputs", "school_vaccination")

counties = pl.read_csv(
    os.path.join(root, "data", "school_vaccination_counties.csv"),
    schema_overrides={"county_fips": pl.String},
)
cases = pl.read_csv(
    os.path.join(root, "data", "outbreaks_vaccinations_cleaned_imputed.csv"),
    infer_schema_length=50000,
).select("year", "fips", "cases")
cases = cases.with_columns(
    county_fips=pl.col("fips").cast(pl.String).str.zfill(5)
)

# the outbreak data has one row per county and year with cases
cases_by_year = cases.group_by("year").agg(
    counties_with_cases=pl.len(), cases_all_counties=pl.col("cases").sum()
)
# the join below must not repeat a school county: no (year, county) of the case
# data that matches a school county may appear twice. (Three rows of 2025, the
# unmapped Iowa districts, have the state's code and match no county.)
matchable = cases.join(
    counties.select(year="year_end", county_fips="county_fips").unique(),
    on=["year", "county_fips"],
    how="semi",
)
assert not matchable.select("year", "county_fips").is_duplicated().any(), (
    "a county has two rows of cases in a year"
)
with_cases = counties.join(
    cases.select(
        year_end="year", county_fips="county_fips", county_cases="cases"
    ),
    on=["year_end", "county_fips"],
    how="left",
)
by_year = (
    with_cases.group_by("year_end")
    .agg(
        school_year=pl.col("year").first(),
        n_states=pl.col("state").n_unique(),
        states=pl.col("state").unique().sort().str.join(", "),
        n_schools=pl.col("n_schools").sum(),
        n_counties_with_school_data=pl.len(),
        n_counties_with_school_data_and_cases=pl.col("county_cases")
        .is_not_null()
        .sum(),
        cases_in_those_counties=pl.col("county_cases").sum(),
    )
    .rename({"year_end": "year"})
)
table = (
    cases_by_year.join(by_year, on="year", how="full", coalesce=True)
    .with_columns(
        pl.col(
            "n_states",
            "n_schools",
            "n_counties_with_school_data",
            "n_counties_with_school_data_and_cases",
            "cases_in_those_counties",
        ).fill_null(0),
        pl.col("states").fill_null(""),
    )
    .select(
        "year",
        "school_year",
        "n_states",
        "states",
        "n_schools",
        "n_counties_with_school_data",
        "n_counties_with_school_data_and_cases",
        "cases_in_those_counties",
        "counties_with_cases",
        "cases_all_counties",
    )  # fmt: skip
    .sort("year")
)
# every county with cases in a year without school data is in counties_with_cases only
table.write_csv(
    os.path.join(output_dir, "school_data_availability_by_year.csv")
)
pl.Config.set_tbl_rows(30)
pl.Config.set_fmt_str_lengths(200)
print(table.drop("states"))
print(table.select("year", "n_states", "states"))
