"""Cleaning of the county-level outbreak records."""

import polars as pl


def process_outbreaks_df(outbreaks_df: pl.DataFrame) -> pl.DataFrame:
    # remove anything pre 2011 because we declare study period as 2011-2025
    outbreaks_df = outbreaks_df.filter(pl.col("year") >= 2011)
    # the last row in outbreaks is a null row with the total number of cases over
    # the study period
    outbreaks_df = outbreaks_df.filter(pl.col("year").is_not_null())
    # let us drop outbreaks with no case count
    # because that doesn't tell us anything about what would have happened had
    # there been an importation
    outbreaks_df = outbreaks_df.filter(pl.col("cases") != 0)

    # when there are fips codes present, we need to do the usual cleaning
    outbreaks_df = outbreaks_df.with_columns(
        fips=pl.when(pl.col("fips").str.len_chars() > 2)
        .then(pl.col("fips").str.zfill(5))
        .otherwise(pl.col("fips"))
    )

    # last column seems to be urls of sources, drop it
    outbreaks_df = outbreaks_df.select(
        ["year", "state", "county", "cases", "indigenous", "imported", "fips"]
    )

    # Delaware reports vaccination coverage at the state only, so aggregate
    # the outbreak data up to that level as well
    outbreaks_df = outbreaks_df.with_columns(
        county=pl.when(pl.col("state") == "Delaware")
        .then(pl.lit("State"))
        .otherwise(pl.col("county"))
    )

    # Group by year, state, and county
    outbreaks_df = outbreaks_df.group_by(
        ["year", "state", "county"], maintain_order=True
    ).agg(
        cases=pl.sum("cases"),
        indigenous=pl.sum("indigenous"),
        imported=pl.sum("imported"),
        # the sum above counts a county-year with no recorded imports as 0
        # imports; this says whether every row of it had a recorded number
        imports_known=pl.col("imported").is_not_null().all(),
        fips=pl.first("fips"),
    )

    # District of Columbia cases don't have a county listed, so they're never going to merge
    # list the county as `District of Columbia` for merging purposes
    outbreaks_df = outbreaks_df.with_columns(
        county=pl.when(pl.col("state") == "District of Columbia")
        .then(pl.lit("District of Columbia"))
        .otherwise(pl.col("county"))
    )

    # for now, we don't want total importations by state, so drop rows with null county
    outbreaks_df = outbreaks_df.filter(pl.col("county").is_not_null())
    # also remove counties with bogus county names
    outbreaks_df = outbreaks_df.filter(
        ~pl.col("county").is_in(["Unknown", "TOTAL", "Out of state resident"])
    )

    # there are a bunch of mispellings of places
    # correct them
    outbreaks_df = outbreaks_df.with_columns(
        [
            pl.col("county")
            .str.replace_all("San Bernadino", "San Bernardino")
            .str.replace_all("Los Angleles", "Los Angeles")
            .str.replace_all("shasta", "Shasta")
            .str.replace_all("Alamakee", "Allamakee")
            .str.replace_all("Detroit City", "Detroit")
            .str.replace_all("St Clair", "St. Clair")
            .str.replace_all("Washentaw", "Washtenaw")
            .str.replace_all("LaSueur", "Le Sueur")
            .str.replace_all("LeSueur", "Le Sueur")
            .str.replace_all("St Louis City", "St. Louis City")
            .str.replace_all("St Charles", "St. Charles")
            # Knowing that the 2019 outbreak in New York City was in Brooklyn, for now
            # we assume that NYC cases are Kings County outbreaks
            .str.replace_all("New York City", "Kings")
            .str.replace_all("Salt lake", "Salt Lake")
            .str.replace_all("Albermarle", "Albemarle")
            .str.replace_all("Spokane\xa0", "Spokane"),
        ]
    )

    # Detroit by itself doesn't have a fips code, so aggregate detroit up to Wayne Co
    outbreaks_df = (
        outbreaks_df.with_columns(
            [pl.col("county").str.replace_all("Detroit", "Wayne")]
        )
        .group_by(["year", "state", "county"], maintain_order=True)
        .agg(
            cases=pl.when(pl.col("cases").is_null().all())
            .then(None)
            .otherwise(pl.col("cases").sum()),
            indigenous=pl.when(pl.col("indigenous").is_null().all())
            .then(None)
            .otherwise(pl.col("indigenous").sum()),
            imported=pl.when(pl.col("imported").is_null().all())
            .then(None)
            .otherwise(pl.col("imported").sum()),
            imports_known=pl.col("imports_known").all(),
            fips=pl.first("fips"),
        )
    )

    # Drop Chicago 2024 -- migrant shelter outbreak
    outbreaks_df = outbreaks_df.filter(
        ~(
            (pl.col("year") == 2024)
            & (pl.col("state") == "Illinois")
            & (pl.col("county") == "Cook")
        )
    )
    # Drop Wisconsin 2021 -- Operation Allies Welcome Afghan refugee outbreak
    outbreaks_df = outbreaks_df.filter(
        ~(
            (pl.col("year") == 2021)
            & (pl.col("state") == "Wisconsin")
            & (pl.col("county") == "Monroe")
        )
    )
    # Drop ICE detention facility 2016
    outbreaks_df = outbreaks_df.filter(
        ~(
            (pl.col("year") == 2016)
            & (pl.col("state") == "Arizona")
            & (pl.col("county") == "Pinal")
        )
    )

    return outbreaks_df
