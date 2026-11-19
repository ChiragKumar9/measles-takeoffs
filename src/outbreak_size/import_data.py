"""
The import records the importation model is fitted to and validated on: the
cleaned case records, the state-level imports of each year, their tally by
state and the county-years stacked for the model (see
analysis/08_importation_model.py, which reads the files).
"""

import polars as pl


def clean_case_records(
    df: pl.DataFrame, df_2025: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    The case records of the outbreak dataset (`df`, as read from the
    spreadsheet) and of the 2025 sheet (`df_2025`), cleaned.

    Returns the county-years with a known number of imported cases (columns
    year, state, county and imported) and the cleaned 2025 records. South
    Carolina's 2025 records are left out: its data are analysed separately.
    """
    # to ensure consistent lower case column names
    df = df.rename({col: col.lower() for col in df.columns})

    # correct a year = 2925 to a 2025 (as 00_explore_data.py does); otherwise
    # that row counts as an extra year of data
    df_2025 = df_2025.with_columns(
        year=pl.when(pl.col("year") == 2925)
        .then(2025)
        .otherwise(pl.col("year"))
    )

    # set the fips for any county that is a higher level region to be the name
    # of the county
    df_2025 = df_2025.filter(pl.col("Lauren Case").is_not_null()).with_columns(
        cases=pl.col("Lauren Case")
    )

    # drop anything south carolina because we have our own south carolina data
    df_2025 = df_2025.filter(pl.col("state") != "South Carolina")

    # concatenate and return
    df = pl.concat([df, df_2025], how="diagonal_relaxed")

    # remove places where the year is null
    df = df.filter(pl.col("year").is_not_null())
    # if importeds is 0, it must be at the county level
    # (since we add in 0 importeds at the state level below when we join with
    # all state data)
    df = df.filter(
        pl.when(pl.col("imported") == 0)
        .then(pl.col("county").is_not_null())
        .otherwise(True)
    )
    # cases also can't be 0
    df = df.filter(pl.col("cases") > 0)
    # also remove counties with bogus county names
    df = df.filter(
        ~pl.col("county").is_in(["TOTAL", "Unknown", "Out of state resident"])
    )
    # state can't be null
    df = df.filter(pl.col("state").is_not_null())

    counties_with_known_imports = df.filter(
        pl.col("imported").is_not_null() & pl.col("county").is_not_null()
    ).select(["year", "state", "county", "imported"])

    return counties_with_known_imports, df_2025


def clean_state_year_imports(
    df: pl.DataFrame, df_2025: pl.DataFrame
) -> pl.DataFrame:
    """
    The state-level imports of each year (state, year, imported, ...): the
    spreadsheet of all years `df` with its state names spelled one way, and the
    imports of the cleaned 2025 records (`df_2025`) added.
    """
    # rename columns
    df = df.rename({col: col.lower() for col in df.columns})
    # the 2025 imports come from the 2025 sheet, added below: a spreadsheet
    # with 2025 rows too would count them twice
    if df["year"].max() >= 2025:  # type: ignore
        raise ValueError(
            "the state-level import spreadsheet has years from 2025 on: the "
            "2025 sheet is added on top of it"
        )
    # replace "DC" with "District of Columbia"
    df = (
        df.with_columns(
            state=pl.when(pl.col("state") == "DC")
            .then(pl.lit("District of Columbia"))
            .otherwise(pl.col("state"))
        )
        .with_columns(
            # fix North Dakora to be North Dakota
            state=pl.when(pl.col("state") == "North Dakora")
            .then(pl.lit("North Dakota"))
            .otherwise(pl.col("state"))
        )
        .with_columns(
            # idaho to be Idaho
            state=pl.when(pl.col("state") == "idaho")
            .then(pl.lit("Idaho"))
            .otherwise(pl.col("state"))
        )
        .with_columns(
            # "New York " to "New York"
            state=pl.when(pl.col("state") == "New York ")
            .then(pl.lit("New York"))
            .otherwise(pl.col("state"))
        )
    )
    # add in 2025 import data
    return pl.concat(
        [df, df_2025.select(["state", "year", "imported"])],
        how="diagonal_relaxed",
    )


def tally_state_importations(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, int]:
    """
    The imports of each state before each of the years 2021 to 2025
    (imported_pre_2021, ...) and in all years (imported), from the state-level
    imports of each year (`df`, see `clean_state_year_imports`), and the
    number of years of data. The years must be consecutive and end in 2025,
    as the importation posteriors count them.
    """
    # the posteriors below count the years of data as consecutive years that
    # end in 2025
    years = sorted(df["year"].drop_nulls().unique().to_list())
    if df["year"].null_count() > 0 or years != list(range(years[0], 2026)):
        raise ValueError(
            "the import data must have a year on every row and every year "
            f"from the first to 2025; years found: {years}, rows without a "
            f"year: {df['year'].null_count()}"
        )
    years_of_data = len(years)
    # group by state

    df = df.group_by("state").agg(
        [
            pl.col("imported")
            .filter(pl.col("year") < i)
            .sum()
            .alias(f"imported_pre_{i}")
            for i in [2021, 2022, 2023, 2024, 2025]
        ],
        imported=pl.sum("imported"),
    )

    return df, years_of_data


def attach_state_posterior(
    rows: pl.DataFrame, state_posterior: pl.DataFrame
) -> pl.DataFrame:
    """
    Give each county-year of the import validation its state's pre-2021
    posterior (posterior_alpha_pre_2021 and posterior_beta_pre_2021 of
    `state_posterior`), with the alpha scaled by the county's share of the
    state: posterior_alpha_pre_2021 by `share` and
    posterior_alpha_pre_2021_log by `share_log`, so that the county's rate is
    share * Gamma(alpha, beta), as for the projections. A state without a
    posterior is an error.
    """
    rows = rows.join(
        state_posterior.select(
            "state", "posterior_alpha_pre_2021", "posterior_beta_pre_2021"
        ),
        on="state",
        how="left",
        maintain_order="left",
    )
    no_posterior = rows.filter(pl.col("posterior_alpha_pre_2021").is_null())
    if no_posterior.height > 0:
        raise ValueError(
            "no import posterior for the states "
            f"{no_posterior['state'].unique().to_list()}"
        )
    return rows.with_columns(
        posterior_alpha_pre_2021=pl.col("posterior_alpha_pre_2021")
        * pl.col("share"),
        posterior_alpha_pre_2021_log=pl.col("posterior_alpha_pre_2021")
        * pl.col("share_log"),
    )


def stack_county_years(
    projection_frames: dict[int, pl.DataFrame],
) -> pl.DataFrame:
    """
    The county-years the importation model gives a probability to: the rows of
    the projection dataset of each validation year (`projection_frames`, by
    year), with the columns fips, state, county, year, validation_year, pop
    and type, once each. A dataset holds a projection row for every county of
    its year and a train row for each of its observed county-years, so the
    years before 2021 (which are not validated) have no rows.

    `year` is the year of the row's data: the year of the vaccination data of
    a projection row (2024 and earlier) or the year of a train row.
    `validation_year` is the year the dataset predicts (2026 for the
    projection of 2026). A county whose vaccination data is older than the
    rest of its dataset's has the same projection row in several datasets, so
    the rows are told apart by their validation year, and their population is
    that of the dataset's year. A projection row of a validation year (2025 or
    earlier) must use vaccination data from before that year, or its
    importation posterior would contain the imports of the year it predicts.
    """
    county_data_all_years = []
    for year, county_data in projection_frames.items():
        projection = county_data.filter(pl.col("type") == "projection")
        if projection["fips"].is_duplicated().any():
            raise ValueError(
                f"projection_dataset_{year}.csv has a county twice, so the "
                "population of its states cannot be summed"
            )
        if year <= 2026:
            too_late = projection.filter(pl.col("year") >= year)
            if too_late.height > 0:
                raise ValueError(
                    f"projection_dataset_{year}.csv has {too_late.height} "
                    f"projection rows with vaccination data from {year} or "
                    "later"
                )
        county_data_all_years.append(
            county_data.with_columns(
                validation_year=pl.lit(year, dtype=county_data.schema["year"])
            )
        )

    county_data = pl.concat(county_data_all_years, how="diagonal_relaxed")
    county_data = county_data.select(
        [
            "fips",
            "state",
            "county",
            "year",
            "validation_year",
            "pop",
            "type",
        ]
    )
    # drop duplicates
    return county_data.unique(
        ["fips", "validation_year", "type", "state", "county"],
        keep="first",
        maintain_order=True,
    )
