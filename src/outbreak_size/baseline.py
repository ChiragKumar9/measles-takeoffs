"""
The vaccination baseline of every county: the value of each field (MMR
coverage, nonmedical and medical exemption), the school year it is from and
its slope, taken from the best source the county has. Used by
analysis/figure_measles_burden_overview.py to write the vaccination data the
projections start from.
"""

import polars as pl

FIELDS = {
    "estimated_mmr_cov": "mmr_cov_state",
    "nonmed_exempt_pct": "nonmed_exempt_pct_state",
    "med_exempt_pct": "med_exempt_pct_state",
}


def slope_by(points, key, value):
    """
    Least-squares slope of `value` on year within each `key`; 0.0 where there
    is a single year (the value is carried forward unchanged).
    """
    x = (
        pl.col("year").cast(pl.Float64)
        - pl.col("year").cast(pl.Float64).mean()
    )
    y = pl.col(value) - pl.col(value).mean()
    return points.group_by(key).agg(
        slope=pl.when((x**2).sum() > 0)
        .then((x * y).sum() / (x**2).sum())
        .otherwise(0.0)
    )


# School years before this are not used as a county's value or slope, except
# by the rule for a county without any value since then (see as_of_values).
FIRST_YEAR = 2020


def two_point_slopes(points, key):
    """
    Least-squares slope of `v` on year within each `key`, from the distinct
    school years of the points; only the keys with at least two years have a
    slope (a single year has none: the caller falls back to another slope).
    """
    points = points.unique([key, "year"], keep="first", maintain_order=True)
    two_or_more = (
        points.group_by(key)
        .agg(n=pl.len())
        .filter(pl.col("n") >= 2)
        .select(key)
    )
    return slope_by(points.join(two_or_more, on=key, how="semi"), key, "v")


def as_of_values(
    reported_rows,
    filled_rows,
    state_series,
    universe,
    before_year,
    states_with_bans=(),
):
    """
    Coverage, nonmedical and medical exemption of every county of `universe`
    as of the data before `before_year`. For each field the latest value, the
    school year it is from, the slope of the series it comes from and where it
    comes from are found on their own. In order of preference:
      1. the county's own reported values since 2020 ("county");
      2. otherwise the latest value since 2020 in the table before the state
         imputation, which is a state or region value copied to the county,
         or a state fill ("region_or_state_rows");
      3. otherwise the county's own latest value from before 2020
         ("county_older"), for a county that has nothing since 2020 (Montana's
         counties and Hamilton County, NY today);
      4. otherwise the value for the state in `state_series` (the CDC's)
         ("state_series").
    The slope is that of the series the value comes from since 2020, from at
    least two school years. Where the series has none, the slope is the
    state's, and where the state has none either, the national average (the
    unweighted mean of the slopes of the counties' own reported values since
    2020, from the data before `before_year` only); 0.0 only if there is no
    national slope either. A "county_older" value, and a state series with
    fewer than two school years since 2020 (Montana's), take their slope from
    the same number of school years as the rule since 2020 spans in this
    fold (the school years from 2020 to the latest before `before_year`),
    counted back from the series' latest school year. In the states of
    `states_with_bans` a rising nonmedical exemption slope is set to 0.
    The three fields are then brought to the latest school year any of them
    has (reference_year) along those slopes, clipped to 0-100; *_observed
    holds the values before that. The county and state names are those of the
    county's latest reported row, else of its latest row.

    Parameters:
    -----------
    reported_rows : pl.DataFrame
        Rows as reported (before interpolation and state fill), with
        county_reported, of every school year the data have
    filled_rows : pl.DataFrame
        The table after interpolation and the "State" row fill, before the
        state imputation (which adds copies of state values for years that
        have none)
    state_series : pl.DataFrame
        State values (vaccination_wrangling.state_level_series)
    universe : pl.DataFrame
        fips, county and state of every county to return
    states_with_bans : iterable of str
        The states with a ban on nonmedical exemptions
    """
    states_with_bans = list(states_with_bans)
    county_rows = reported_rows.filter(
        pl.col("county_reported")
        & (pl.col("year") >= FIRST_YEAR)
        & (pl.col("year") < before_year)
    )
    # the county's own values from before 2020 (the rule for a county with
    # nothing since)
    old_rows = reported_rows.filter(
        pl.col("county_reported")
        & (pl.col("fips").str.len_chars() == 5)
        & (pl.col("year") < FIRST_YEAR)
        & (pl.col("year") < before_year)
    )
    filled_all_years = filled_rows.filter(
        (pl.col("fips").str.len_chars() == 5) & (pl.col("year") < before_year)
    ).select(
        "fips",
        "state",
        "county",
        "year",
        *[pl.col(field).cast(pl.Float64, strict=False) for field in FIELDS],
    )
    filled = filled_all_years.filter(pl.col("year") >= FIRST_YEAR)
    state_points = state_series.filter(pl.col("year") < before_year)
    # the school years the rule since 2020 spans in this fold
    latest_data_year = reported_rows.filter(pl.col("year") < before_year)[
        "year"
    ].max()
    span = max(1, min(before_year - 1, latest_data_year) - FIRST_YEAR + 1)

    # names: the county's latest reported row, else its latest row
    names = (
        pl.concat(
            [
                county_rows.select(
                    "fips", "state", "county", "year"
                ).with_columns(rank=pl.lit(0)),
                filled_all_years.select(
                    "fips", "state", "county", "year"
                ).with_columns(rank=pl.lit(1)),
            ]
        )
        .sort(
            ["fips", "rank", "year"],
            descending=[False, False, True],
            maintain_order=True,
        )
        .unique("fips", keep="first", maintain_order=True)
        .select("fips", "state", "county")
    )
    out = universe.join(
        names, on="fips", how="left", suffix="_data", maintain_order="left"
    ).select(
        "fips",
        county=pl.coalesce("county_data", "county"),
        state=pl.coalesce("state_data", "state"),
    )
    shapefile_state = universe.select("fips", state_shapefile="state")
    out = out.join(
        shapefile_state, on="fips", how="left", maintain_order="left"
    )

    for field, state_column in FIELDS.items():

        def usable(points, column):
            if points.schema[column] in (pl.Float32, pl.Float64):
                points = points.with_columns(pl.col(column).fill_nan(None))
            # a coverage of 0 is a missing value, an exemption of 0 is not
            points = points.filter(pl.col(column).is_not_null())
            if field == "estimated_mmr_cov":
                points = points.filter(pl.col(column) != 0.0)
            return points

        def latest(points, column, key):
            return (
                usable(points, column)
                .sort([key, "year"], maintain_order=True)
                .group_by(key, maintain_order=True)
                .agg(value=pl.col(column).last(), year=pl.col("year").last())
            )

        def slopes_since_2020(points, column, key):
            points = usable(points, column).filter(
                pl.col("year") >= FIRST_YEAR
            )
            return two_point_slopes(
                points.select(key, "year", v=pl.col(column)), key
            )

        def slopes_counted_back(points, column, key, latest_points):
            """over `span` school years back from each key's latest year"""
            points = (
                usable(points, column)
                .select(key, "year", v=pl.col(column))
                .join(
                    latest_points.select(key, last_year="year"),
                    on=key,
                    how="inner",
                )
                .filter(pl.col("year") > pl.col("last_year") - span)
            )
            return two_point_slopes(points.select(key, "year", "v"), key)

        l1 = latest(county_rows, field, "fips")
        s1 = slopes_since_2020(county_rows, field, "fips")
        l2 = latest(filled, field, "fips")
        s2 = slopes_since_2020(filled, field, "fips")
        l_old = latest(old_rows, field, "fips")
        s_old = slopes_counted_back(old_rows, field, "fips", l_old)
        l3 = latest(state_points, state_column, "state")
        s3 = slopes_since_2020(state_points, state_column, "state")
        # a state series with fewer than two years since 2020 (Montana's CDC
        # series ends in 2020) reaches back, as the older county values do
        s3 = pl.concat(
            [
                s3,
                slopes_counted_back(
                    state_points, state_column, "state", l3
                ).join(s3.select("state"), on="state", how="anti"),
            ]
        )
        national = s1["slope"].mean()

        out = (
            out.join(
                l1.rename({"value": "v1", "year": "y1"}),
                on="fips",
                how="left",
                maintain_order="left",
            )
            .join(
                s1.rename({"slope": "s1"}),
                on="fips",
                how="left",
                maintain_order="left",
            )
            .join(
                l2.rename({"value": "v2", "year": "y2"}),
                on="fips",
                how="left",
                maintain_order="left",
            )
            .join(
                s2.rename({"slope": "s2"}),
                on="fips",
                how="left",
                maintain_order="left",
            )
            .join(
                l_old.rename({"value": "v_old", "year": "y_old"}),
                on="fips",
                how="left",
                maintain_order="left",
            )
            .join(
                s_old.rename({"slope": "s_old"}),
                on="fips",
                how="left",
                maintain_order="left",
            )
            .join(
                l3.rename(
                    {"value": "v3", "year": "y3", "state": "state_shapefile"}
                ),
                on="state_shapefile",
                how="left",
                maintain_order="left",
            )
            .join(
                s3.rename({"slope": "s3", "state": "state_shapefile"}),
                on="state_shapefile",
                how="left",
                maintain_order="left",
            )
            .with_columns(
                pl.coalesce("v1", "v2", "v_old", "v3").alias(field),
                pl.coalesce("y1", "y2", "y_old", "y3").alias(f"{field}_year"),
                pl.when(pl.col("v1").is_not_null())
                .then(pl.lit("county"))
                .when(pl.col("v2").is_not_null())
                .then(pl.lit("region_or_state_rows"))
                .when(pl.col("v_old").is_not_null())
                .then(pl.lit("county_older"))
                .when(pl.col("v3").is_not_null())
                .then(pl.lit("state_series"))
                .alias(f"{field}_source"),
                # the series' own slope, else the state's, else the national
                pl.coalesce(
                    pl.when(pl.col("v1").is_not_null()).then(pl.col("s1")),
                    pl.when(
                        pl.col("v1").is_null() & pl.col("v2").is_not_null()
                    ).then(pl.col("s2")),
                    pl.when(
                        pl.col("v1").is_null()
                        & pl.col("v2").is_null()
                        & pl.col("v_old").is_not_null()
                    ).then(pl.col("s_old")),
                    pl.col("s3"),
                    pl.lit(national, dtype=pl.Float64),
                    pl.lit(0.0),
                ).alias(f"{field}_slope"),
            )
            .drop("v1", "y1", "s1", "v2", "y2", "s2", "v_old", "y_old")
            .drop("s_old", "v3", "y3", "s3")
        )
        if field == "nonmed_exempt_pct":
            # a state with a ban does not get its exemptions back
            out = out.with_columns(
                pl.when(
                    pl.col("state_shapefile").is_in(states_with_bans)
                    & (pl.col(f"{field}_slope") > 0)
                )
                .then(pl.lit(0.0))
                .otherwise(pl.col(f"{field}_slope"))
                .alias(f"{field}_slope")
            )
    out = out.drop("state_shapefile")

    # bring the three fields to the same school year, the latest any of them
    # has, along the slope of their own series (never backwards), so that
    # each county has one baseline year; a value of that year is unchanged
    out = out.with_columns(
        reference_year=pl.max_horizontal(
            *[pl.col(f"{field}_year") for field in FIELDS]
        )
    ).with_columns(
        *[pl.col(field).alias(f"{field}_observed") for field in FIELDS]
    )
    return out.with_columns(
        *[
            pl.when(pl.col(f"{field}_year") < pl.col("reference_year"))
            .then(
                (
                    pl.col(field)
                    + pl.col(f"{field}_slope")
                    * (pl.col("reference_year") - pl.col(f"{field}_year"))
                ).clip(0.0, 100.0)
            )
            .otherwise(pl.col(field))
            .alias(field)
            for field in FIELDS
        ]
    )


def trend_to_year(values, target_year):
    """
    `values` (the result of as_of_values) with every field brought to school
    year `target_year` along its slope, from the school year of its observed
    value (never backwards), clipped to 0-100. reference_year becomes
    `target_year` (or stays, if later). The observed values and their years
    and sources are untouched.

    Used for the school year an outbreak year uses (Y - 1) in the vaccination
    data of year Y, and for the last known school year in the scenarios, so
    that a value is trended once, from its observation.
    """
    return values.with_columns(
        *[
            pl.when(pl.col(f"{field}_year") < target_year)
            .then(
                (
                    pl.col(f"{field}_observed")
                    + pl.col(f"{field}_slope")
                    * (target_year - pl.col(f"{field}_year"))
                ).clip(0.0, 100.0)
            )
            .otherwise(pl.col(field))
            .alias(field)
            for field in FIELDS
        ],
        reference_year=pl.max_horizontal(
            pl.col("reference_year"), pl.lit(target_year)
        ),
    )


def place_slopes(
    vaccinations_df, state_series, before_year, states_with_bans=()
):
    """
    The slope of each field (the keys of FIELDS) of every (state, county) place
    of `vaccinations_df`, from the data before `before_year`: the place's own
    since 2020 (at least two school years), else its state's in `state_series`
    (since 2020), else the national average (the unweighted mean of the
    places' own slopes), else 0.0. In the states of `states_with_bans` a
    rising nonmedical exemption slope is set to 0. Columns: state, county and
    one `<field>_slope` per field.

    Used to trend a value observed in an earlier school year (the 2026 rows
    that take the county's own 2024-25 values) to the school year they stand
    for.
    """
    states_with_bans = list(states_with_bans)
    places = vaccinations_df.select("state", "county").unique(
        maintain_order=True
    )
    rows = vaccinations_df.filter(
        (pl.col("year") >= FIRST_YEAR) & (pl.col("year") < before_year)
    ).with_columns(key=pl.col("state") + pl.lit("|") + pl.col("county"))
    state_points = state_series.filter(
        (pl.col("year") >= FIRST_YEAR) & (pl.col("year") < before_year)
    )
    out = places.with_columns(
        key=pl.col("state") + pl.lit("|") + pl.col("county")
    )
    for field, state_column in FIELDS.items():
        points = rows.select(
            "key", "year", v=pl.col(field).cast(pl.Float64, strict=False)
        ).filter(pl.col("v").is_not_null() & ~pl.col("v").is_nan())
        if field == "estimated_mmr_cov":
            points = points.filter(pl.col("v") != 0.0)
        own = two_point_slopes(points, "key")
        national = own["slope"].mean()
        own = own.rename({"slope": "own"})
        state = two_point_slopes(
            state_points.filter(pl.col(state_column).is_not_null()).select(
                "state", "year", v=pl.col(state_column)
            ),
            "state",
        ).rename({"slope": "state_slope"})
        out = (
            out.join(own, on="key", how="left", maintain_order="left")
            .join(state, on="state", how="left", maintain_order="left")
            .with_columns(
                pl.coalesce(
                    "own",
                    "state_slope",
                    pl.lit(national, dtype=pl.Float64),
                    pl.lit(0.0),
                ).alias(f"{field}_slope")
            )
            .drop("own", "state_slope")
        )
        if field == "nonmed_exempt_pct":
            out = out.with_columns(
                pl.when(
                    pl.col("state").is_in(states_with_bans)
                    & (pl.col(f"{field}_slope") > 0)
                )
                .then(pl.lit(0.0))
                .otherwise(pl.col(f"{field}_slope"))
                .alias(f"{field}_slope")
            )
    return out.drop("key")


def county_source_rows(reported_rows):
    """
    The rows the slopes are fitted on: the counties' own reported values since
    2020 (no interpolated values, no state or region value copied to its
    counties), for counties with at least two reported years.
    """
    return reported_rows.filter(
        pl.col("county_reported")
        & pl.col("estimated_mmr_cov").is_not_null()
        & (pl.col("estimated_mmr_cov") != 0.0)
        & (pl.col("year") >= 2020)
    ).filter(pl.col("year").n_unique().over("fips") >= 2)


def panel_c_coverage(reported_rows, year=2024, fallback_year=2023):
    """
    MMR coverage shown in panel c: the value reported for `year`, else the one
    reported for `fallback_year` (carried forward). One row per county FIPS
    (5 characters), with the school year it comes from and a category saying
    how it was obtained. A county with neither has no row (the map leaves it
    grey): a state value is not drawn for it, nor a value left by the
    interpolation and state fill.

    Parameters:
    -----------
    reported_rows : pl.DataFrame
        Rows as reported (before interpolation and state fill), with
        county_reported saying whether the row is the county's own value
    """

    def usable(df, y, *extra):
        # where a FIPS has several rows in the year, prefer the county's own
        if "county_reported" in df.columns:
            df = df.sort(
                "county_reported", descending=True, maintain_order=True
            )
        return (
            df.filter(
                (pl.col("year") == y)
                # only county FIPS are drawn: not a state (2) or a school
                # district
                & (pl.col("fips").str.len_chars() == 5)
                & pl.col("estimated_mmr_cov").is_not_null()
                & ~pl.col("estimated_mmr_cov").is_nan()
                & (pl.col("estimated_mmr_cov") != 0.0)
            )
            .unique("fips", keep="first", maintain_order=True)
            .select("fips", "state", "county", "estimated_mmr_cov", *extra)
        )

    # the county_reported flag comes from the same row as the value
    latest = usable(reported_rows, year, "county_reported")
    carried = usable(reported_rows, fallback_year, "county_reported").filter(
        ~pl.col("fips").is_in(latest["fips"].implode())
    )

    return pl.concat(
        [
            latest.with_columns(
                source_year=pl.lit(year),
                category=pl.when(pl.col("county_reported"))
                .then(pl.lit(f"county_{year}"))
                .otherwise(pl.lit(f"region_or_state_{year}")),
            ),
            carried.with_columns(
                source_year=pl.lit(fallback_year),
                category=pl.when(pl.col("county_reported"))
                .then(pl.lit(f"county_{fallback_year}_carried_forward"))
                .otherwise(
                    pl.lit(f"region_or_state_{fallback_year}_carried_forward")
                ),
            ),
        ],
        how="diagonal",
    )
