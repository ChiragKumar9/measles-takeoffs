"""
Supplemental tables on the vaccination data behind the outbreak records. Every
outbreak county-year counts once, and the period is 2011-2020 or 2021-2025.

- vaccine_data_sources_by_period: for each period, how many county-years (and
  distinct places) take their MMR coverage from each source: the county's own
  MMR coverage, the all-vaccine coverage used where the MMR coverage is blank
  or 0, the average of the counties of a multi-county region, another year of
  the same county or the CDC state value (both imputed), or nothing (dropped).
- states_using_all_vaccine_coverage: the states whose county-years use the
  all-vaccine coverage, by period.
- vaccine_data_missingness_by_year, vaccine_data_missingness_by_state: how many
  county-years have all of their vaccine data observed (the rows of
  analysis_dataset.csv), how many have some of it imputed (the rows only
  analysis_dataset_imputed.csv has; the imputed sensitivity analysis uses
  them) and how many have none that could be filled in (dropped). The by-year
  table has a row for each period and one for all years; the by-state table
  has a row for each state in each period (and in all years), each with an
  "All" row for all states.

Reads data/raw_merged_outbreaks_vaccinations.csv, data/analysis_dataset.csv and
data/analysis_dataset_imputed.csv (written by 00_explore_data.py and
01_assemble_analysis_dataset.py) and writes the tables to figures/.
"""

import os

import polars as pl

repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(repo, "data")
fig_dir = os.path.join(repo, "figures")

KEY = ["year", "state", "county"]
PERIODS = ["2011-2020", "2021-2025"]
MMR = "MMR coverage (county, observed)"
ALL_VACCINE = "All-vaccine coverage (MMR blank or 0)"
REGION = "Multi-county region (average of its counties)"
ANOTHER_YEAR = "Another year of the same county (imputed)"
CDC_STATE = "CDC state MMR value (imputed)"
DROPPED = "Dropped (vaccine data could not be filled in)"
SOURCES = [MMR, ALL_VACCINE, REGION, ANOTHER_YEAR, CDC_STATE, DROPPED]


def read(name):
    return pl.read_csv(
        os.path.join(data_dir, name), schema_overrides={"fips": pl.String}
    )


# every county-year of the case data, with its vaccine data at the matched year
raw = read("raw_merged_outbreaks_vaccinations.csv")
primary = read("analysis_dataset.csv").select(KEY)
imputed = read("analysis_dataset_imputed.csv").select(
    *KEY, "imputation_source"
)
assert raw.height == raw.select(KEY).unique().height
# the joins below take one row of each file per county-year
assert primary.height == primary.select(KEY).unique().height
assert imputed.height == imputed.select(KEY).unique().height
# the imputed file has every row of the primary file and the imputed rows
assert primary.join(imputed, on=KEY, how="anti").height == 0
assert imputed.join(raw, on=KEY, how="anti").height == 0

county_years = (
    raw.join(primary.with_columns(in_primary=True), on=KEY, how="left")
    .join(imputed.with_columns(in_imputed_file=True), on=KEY, how="left")
    .with_columns(
        pl.col("in_primary", "in_imputed_file").fill_null(False),
        period=pl.when(pl.col("year") <= 2020)
        .then(pl.lit(PERIODS[0]))
        .otherwise(pl.lit(PERIODS[1])),
        place=pl.col("state") + " / " + pl.col("county"),
    )
    .with_columns(
        status=pl.when(pl.col("in_primary"))
        .then(pl.lit("observed"))
        .when(pl.col("in_imputed_file"))
        .then(pl.lit("imputed"))
        .otherwise(pl.lit("dropped"))
    )
)

# Where the MMR coverage of each county-year comes from. An imputation source
# such as "mmr:state_cdc; nonmed:year-2" says where each field was imputed
# from; a county-year without "mmr:" in it has its MMR coverage as observed.
# Observed coverage with neither mmr_cov nor all_vacc_cov is a region's average
imputation = pl.col("imputation_source").fill_null("")
county_years = county_years.with_columns(
    source=pl.when(pl.col("status") == "dropped")
    .then(pl.lit(DROPPED))
    .when(imputation.str.contains("mmr:state_cdc"))
    .then(pl.lit(CDC_STATE))
    .when(imputation.str.contains("mmr:year"))
    .then(pl.lit(ANOTHER_YEAR))
    .when(pl.col("mmr_cov").is_not_null() & (pl.col("mmr_cov") != 0))
    .then(pl.lit(MMR))
    .when(pl.col("all_vacc_cov").is_not_null())
    .then(pl.lit(ALL_VACCINE))
    .otherwise(pl.lit(REGION))
)


def write(table, name):
    path = os.path.join(fig_dir, f"table_supplemental_{name}.csv")
    table.write_csv(path)
    print(f"wrote {path} {table.shape}")


# one row per source, county-years and places for each period
by_source = pl.DataFrame({"coverage_source": SOURCES})
for period in PERIODS:
    counts = (
        county_years.filter(pl.col("period") == period)
        .group_by("source")
        .agg(
            pl.len().alias(f"county_years_{period}"),
            pl.col("place").n_unique().alias(f"places_{period}"),
        )
        .rename({"source": "coverage_source"})
    )
    by_source = by_source.join(
        counts, on="coverage_source", how="left", maintain_order="left"
    )
by_source = by_source.fill_null(0).with_columns(
    pl.exclude("coverage_source").cast(pl.Int64)
)
total: dict[str, str | int] = {"coverage_source": "Total"}
for period in PERIODS:
    period_rows = county_years.filter(pl.col("period") == period)
    total[f"county_years_{period}"] = period_rows.height
    total[f"places_{period}"] = period_rows["place"].n_unique()
assert by_source.select(
    pl.col(f"county_years_{p}").sum() for p in PERIODS
).row(0) == tuple(total[f"county_years_{p}"] for p in PERIODS)
write(
    pl.concat([by_source, pl.DataFrame(total)]),
    "vaccine_data_sources_by_period",
)

# the states that use the all-vaccine coverage
write(
    county_years.filter(pl.col("source") == ALL_VACCINE)
    .group_by("period", "state")
    .agg(
        pl.len().alias("county_years"),
        pl.col("place").n_unique().alias("places"),
    )
    .sort(
        ["period", "county_years", "state"], descending=[False, True, False]
    ),
    "states_using_all_vaccine_coverage",
)

# how much of the vaccine data is observed, imputed or dropped
status_counts = [
    pl.len().alias("county_years"),
    (pl.col("status") == "observed").sum().alias("observed"),
    (pl.col("status") == "imputed").sum().alias("imputed"),
    (pl.col("status") == "dropped").sum().alias("dropped"),
]


def in_period(period):
    """The county-years of a period, or all of them for the period "All"."""
    if period == "All":
        return county_years
    return county_years.filter(pl.col("period") == period)


def with_percent(table):
    table = table.with_columns(
        percent_not_fully_observed=(
            100
            * (pl.col("imputed") + pl.col("dropped"))
            / pl.col("county_years")
        ).round(1)
    )
    assert (
        table["observed"] + table["imputed"] + table["dropped"]
        == table["county_years"]
    ).all()
    return table


def missingness_by_year():
    """One row per year, then a row for each period and one for all years."""
    table = (
        county_years.group_by("year")
        .agg(status_counts)
        .sort("year")
        .with_columns(pl.col("year").cast(pl.String))
    )
    totals = [
        in_period(period)
        .select(status_counts)
        .with_columns(year=pl.lit(period))
        .select(table.columns)
        for period in [*PERIODS, "All"]
    ]
    return with_percent(pl.concat([table, *totals]))


def missingness_by_state():
    """
    For each period (and for all years, "All"): one row per state that has
    county-years in it, largest first, then an "All" row for all the states.
    """
    tables = []
    for period in [*PERIODS, "All"]:
        rows = in_period(period)
        by_state = (
            rows.group_by("state")
            .agg(status_counts)
            .sort(["county_years", "state"], descending=[True, False])
        )
        all_states = (
            rows.select(status_counts)
            .with_columns(state=pl.lit("All"))
            .select(by_state.columns)
        )
        tables.append(
            pl.concat([by_state, all_states]).with_columns(
                period=pl.lit(period)
            )
        )
    table = pl.concat(tables).select(
        "period", "state", pl.exclude("period", "state")
    )
    return with_percent(table)


write(missingness_by_year(), "vaccine_data_missingness_by_year")
write(missingness_by_state(), "vaccine_data_missingness_by_state")
