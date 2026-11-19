"""
The projection datasets: for each validation year, the rows the models are
trained on and the rows they predict, and for each future vaccination scenario
the rows of the latest year with the scenario's vaccination values (see
analysis/05_assemble_projection_dataset.py).

The rows of a year hold the county data of the year before (the vaccination
school year and the census year; see outbreak_size.data_year).
"""

import polars as pl

# the columns the projection rows have before they are given the columns
# that only the observed (train) rows carry
PROJECTION_COLUMNS = [
    "estimated_mmr_cov",
    "nonmed_exempt_pct",
    "med_exempt_pct",
    "fraction_under_5",
    "fraction_school_aged",
    "avg_house_size",
    "std_house_size",
    "fraction_public",
    "fraction_white",
    "fraction_graduate_degree",
    "log_population_density",
    "log_income",
    "fips",
    "state",
    "county",
    "year",
    "pop",
]
TRAIN_COLUMNS = PROJECTION_COLUMNS + [
    "cases_outbreak",
    "geoid",
    "cases",
]


def clean_year_vaccination_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    The counties of one year's vaccination data that can be projected: the
    county names are spelled as in the training data, and the places that miss
    a vaccination value or whose fips is not a county's (five characters) are
    left out.
    """
    # spell these counties as 00_explore_data.py does, so that a county's
    # geoid (county name + fips) is the same in the training data and in
    # this projection data
    df = df.with_columns(
        county=pl.col("county")
        .str.replace_all("Mcdonough", "McDonough")
        .str.replace_all("Dupage", "DuPage")
        .str.replace_all("Mcculloch", "McCulloch")
    )

    # remove any places that have missing data
    df = df.filter(
        (pl.col("estimated_mmr_cov").is_not_null())
        & (pl.col("med_exempt_pct").is_not_null())
        & (pl.col("nonmed_exempt_pct").is_not_null())
    )

    # remove any places with fips codes greater than five digits
    return df.filter(pl.col("fips").str.len_chars() == 5)


def require_consistent_places(
    df: pl.DataFrame, analysis_df: pl.DataFrame, label: str
) -> None:
    """
    Raise unless every county (a five-character fips) is one place in `df`
    and in the analysis dataset: one geoid per fips, one fips per geoid, and
    the geoid of the analysis dataset where the county has one. The models
    number their spatial units by geoid and their adjacency matrix by fips, so
    a county that has two geoids (a region's name in one data source, its own
    in another) would be two units or take another county's neighbours. Run
    here, the error stops script 05 and not a fit hours into script 06.
    """
    counties = df.filter(pl.col("fips").str.len_chars() == 5)
    two_geoids = (
        counties.group_by("fips")
        .agg(geoid=pl.col("geoid").unique().sort())
        .filter(pl.col("geoid").list.len() > 1)
    )
    two_fips = (
        counties.group_by("geoid")
        .agg(fips=pl.col("fips").unique().sort())
        .filter(pl.col("fips").list.len() > 1)
    )
    known = (
        analysis_df.filter(pl.col("fips").str.len_chars() == 5)
        .select("fips", "geoid")
        .unique()
    )
    against_analysis = (
        counties.select("fips", "geoid")
        .unique()
        .join(known, on="fips", suffix="_analysis")
        .filter(pl.col("geoid") != pl.col("geoid_analysis"))
    )
    problems = []
    if two_geoids.height:
        problems.append(f"fips with two geoids {two_geoids.head(5).rows()}")
    if two_fips.height:
        problems.append(f"geoids with two fips {two_fips.head(5).rows()}")
    if against_analysis.height:
        problems.append(
            "a geoid other than that of the analysis dataset "
            f"{against_analysis.head(5).rows()}"
        )
    if problems:
        raise ValueError(f"{label}: " + "; ".join(problems))


def assemble_projection_year(
    df: pl.DataFrame, analysis_df: pl.DataFrame, year
):
    """
    The rows of one validation year: the counties of `df` (the year's
    vaccination and census data) as projection rows, and the county-years of
    `analysis_df` (the analysis dataset) in that year as train rows.

    A county with observed outbreak data in this year is in both: as a train
    row (with its observed cases) and as a projection row like every other
    county, so it appears twice. The validation figure and table keep one row
    per county-year (the train row); script 06 predicts both.

    A row with a missing or NaN value is dropped (too few people to get census
    data, or no real fips in the synthetic population).

    Returns the rows, the projection fips that were dropped (sorted), the
    train places (state, county and fips) that were dropped, and the numbers
    of projection fips and of train places before dropping.
    """
    # only use train data that equals the validation year
    train_df = analysis_df.filter(pl.col("year") == year)

    train_df = train_df.select(TRAIN_COLUMNS).with_columns(
        in_train=pl.lit(1),
        type=pl.lit("train"),
        estimated_mmr_cov_raw=pl.col("estimated_mmr_cov"),
    )

    projection_df = (
        df.select(PROJECTION_COLUMNS)
        .with_columns(
            in_train=pl.lit(0),
            # we need to have some value but it is not used
            cases_outbreak=pl.lit(0),
            cases=pl.lit(0),
            estimated_mmr_cov_raw=df["estimated_mmr_cov"],
            type=pl.lit("projection"),
        )
        # a region's rows in the year file carry the region's name as the
        # county ("Panhandle" for the Nebraska counties of that health
        # district): a county that is in the analysis dataset (any year) takes
        # its name there, so that its geoid is the same in its train rows and
        # in its projection rows (the models number their spatial units by
        # geoid and the adjacency matrix by fips, and refuse places where the
        # two disagree)
        .join(
            analysis_df.filter(pl.col("fips").str.len_chars() == 5)
            .select("fips", county_in_analysis="county")
            .unique("fips", keep="first", maintain_order=True),
            on="fips",
            how="left",
            maintain_order="left",
        )
        .with_columns(county=pl.coalesce("county_in_analysis", "county"))
        .drop("county_in_analysis")
        .with_columns(
            geoid=pl.when(pl.col("fips").is_null())
            .then(pl.col("county"))
            .otherwise(pl.col("county") + pl.col("fips"))
        )
    )

    # join the two dataframes together
    df = pl.concat([train_df, projection_df], how="diagonal_relaxed")

    projection_fips = set(df.filter(pl.col("type") == "projection")["fips"])
    train_places = df.filter(pl.col("type") == "train").select(
        "fips", "state", "county"
    )
    # drop places with nas -- means there were too few people there to get census data
    df = df.drop_nans()
    # these arise from places that don't have a real fips code in the synth pop
    df = df.drop_nulls()
    dropped = sorted(
        projection_fips
        - set(df.filter(pl.col("type") == "projection")["fips"])
    )
    # observed outbreak rows that were dropped are missing from the
    # validation too
    kept_train = df.filter(pl.col("type") == "train")["fips"]
    # (a filter, not an anti join: it keeps the order of the places; a place
    # without a fips matches nothing, as in an anti join)
    dropped_train = train_places.filter(
        ~pl.col("fips").is_in(kept_train.implode()).fill_null(False)
    )
    require_consistent_places(
        df, analysis_df, f"the rows of validation year {year}"
    )
    return (
        df,
        dropped,
        dropped_train,
        len(projection_fips),
        train_places.height,
    )


# the columns that come from the census (and the land size): a county that is
# both a train row (from the analysis dataset, script 00) and a projection row
# (script 05) has the same values in both only if both asked the census of the
# same year
CENSUS_COLUMNS = [
    "pop",
    "log_income",
    "fraction_under_5",
    "fraction_school_aged",
    "fraction_public",
    "fraction_white",
    "fraction_graduate_degree",
    "log_population_density",
]


def require_same_census_values(df: pl.DataFrame) -> None:
    """Raise if a place that is a train row and a projection row of the same
    year has different census values in the two (see assemble_projection_year:
    scripts 00 and 05 must ask the census of the same year, the year before
    the outbreak year)."""
    train = df.filter(pl.col("type") == "train").select(
        "fips", *CENSUS_COLUMNS
    )
    projection = df.filter(pl.col("type") == "projection").select(
        "fips", *CENSUS_COLUMNS
    )
    both = train.join(projection, on="fips", suffix="_projection")
    for column in CENSUS_COLUMNS:
        different = both.filter(
            (pl.col(column) - pl.col(f"{column}_projection")).abs() > 1e-9
        )
        if different.height:
            raise ValueError(
                f"{different.height} places have another {column} as a train "
                f"row than as a projection row, e.g. "
                f"{different.select('fips', column, f'{column}_projection').head(3).rows()}: "
                "scripts 00 and 05 asked the census of different years"
            )


def scenario_projection_rows(
    base_df: pl.DataFrame,
    projections: pl.DataFrame,
    scenario: str,
    future_year: int,
):
    """
    The projection rows of one vaccination scenario in one future year: the
    rows of `base_df` (the latest year's projection rows, with their census and
    demographic data) with the vaccination values the scenario projects for
    that year. `projections` has one row per county, scenario and year (columns
    fips, scenario, year, projected_coverage, projected_nonmed_exempt and
    projected_med_exempt).

    A county with no projection in this scenario and year is left out. Returns
    the rows and the fips of the counties left out.
    """
    # Pull the projected values for this scenario and year
    proj_slice = projections.filter(
        (pl.col("scenario") == scenario) & (pl.col("year") == future_year)
    ).select(
        [
            "fips",
            "projected_coverage",
            "projected_nonmed_exempt",
            "projected_med_exempt",
        ]
    )

    # Join projected values onto the base census/demographic data
    # (the inner join drops a county without a projection: report it)
    lost = set(base_df["fips"]) - set(proj_slice["fips"])
    out_df = (
        base_df.join(proj_slice, on="fips", how="inner", maintain_order="left")
        .with_columns(
            [
                pl.col("projected_coverage").alias("estimated_mmr_cov"),
                pl.col("projected_coverage").alias("estimated_mmr_cov_raw"),
                pl.col("projected_nonmed_exempt").alias("nonmed_exempt_pct"),
                pl.col("projected_med_exempt").alias("med_exempt_pct"),
                pl.lit(future_year).alias("year"),
                pl.lit(scenario).alias("scenario"),
            ]
        )
        .drop(
            [
                "projected_coverage",
                "projected_nonmed_exempt",
                "projected_med_exempt",
            ]
        )
    )
    return out_df, lost
