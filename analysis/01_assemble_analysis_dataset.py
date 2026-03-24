import os

import numpy as np
import polars as pl

from outbreak_size.adjacency import assemble_adjacency_matrix

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)

if __name__ == "__main__":
    # Load dataset
    df = pl.read_csv(
        os.path.join(data_dir, "outbreaks_vaccinations_cleaned.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
    )

    # make year dummy variables
    df = df.sort("year", descending=True)
    year_data = df["year"]
    df = df.drop(["year_relevant_vaccination"])
    df = df.to_dummies(columns="year", separator="_", drop_first=False)
    df = df.with_columns(year=year_data)

    # make an "every two years" variable for easier plotting when random walking
    df = df.with_columns(
        year_every_two=pl.when(pl.col("year") % 2 == 1)
        .then(pl.col("year") - 1)
        .otherwise(pl.col("year"))
    )
    # now make these years contiguous
    df = df.with_columns(
        year_every_two_contiguous=(
            (pl.col("year_every_two") - pl.col("year_every_two").min()) / 2
        ).cast(pl.Int32)
    )
    df = df.drop("year_every_two")

    # spatial component
    adjacency_matrix = pl.read_csv(
        os.path.join(data_dir, "county_adjacencies.txt"),
        separator="|",
        schema_overrides={
            "County GEOID": pl.String,
            "Neighbor GEOID": pl.String,
        },
    )

    # assemble adjacency matrix
    # for any fips that are null, make the fips the county name
    # these are mega regions
    df = df.with_columns(
        fips=pl.when(pl.col("fips").is_null())
        .then(pl.col("county"))
        .otherwise(pl.col("fips"))
    )

    # whether a 2025 mega region
    df = df.with_columns(
        mega_region_2025=pl.col("fips").is_in(
            [
                "Upstate, South Carolina",
                "Eastern Region, Virginia",
                "Northwest Region, Virginia",
                "Region 9, Louisiana",
                "Twin Cities Metro Area, Minnesota",
                "Nashville-Davidson County Region, Tennessee",
                "Upper Cumberland Region, Tennessee",
                "North, Alabama",
                "Southeast Health District, Utah",
                "Bear River, Utah",
                "Mid-Cumberland Region, Tennessee",
                "Southwest Health District, Utah",
            ]
        )
    )

    df = df.with_columns(
        oddities_2025=pl.when(
            (pl.col("cases_outbreak") <= 10)
            # pl.when(
            #     (
            #         (pl.col("state") == "South Carolina")
            #         | (pl.col("state") == "Utah")
            #         | (pl.col("state") == "Virginia")
            #         | (pl.col("state") == "Louisiana")
            #         | (pl.col("state") == "Minnesota")
            #         | (pl.col("state") == "Tennessee")
            #         | (pl.col("state") == "Alabama")
            #     )
            & (pl.col("year") == 2025)
        )
        .then(pl.lit(True))
        .otherwise(pl.lit(False))
    )

    counties = df["fips"].unique(maintain_order=True).to_list()
    assembled_adjacency_matrix = assemble_adjacency_matrix(
        adjacency_matrix, counties
    )

    assembled_adjacency_matrix_no_2025_oddities = assemble_adjacency_matrix(
        adjacency_matrix,
        df.filter(pl.col("oddities_2025") == False)["fips"]
        .unique(maintain_order=True)
        .to_list(),
    )

    # what if we knew the neighbors vaccine coverage?
    # for each county, iterate through, get neighbors, and population-average
    # their vaccine coverage
    vaccinations = pl.read_csv(
        os.path.join(data_dir, "processed_vaccinations_data.csv"),
        schema_overrides={
            "fips": pl.String,
            "estimated_mmr_cov": pl.Float64,
            "nonmed_exempt_pct": pl.Float64,
            "med_exempt_pct": pl.Float64,
        },
    )
    df = df.with_columns(
        neighbors_coverage=pl.lit(None),
        neighbors_coverage_pop_weighted=pl.lit(None),
    )

    for fips, year in zip(df["fips"].to_list(), df["year"].to_list()):
        # what are the county's neighbors?
        neighbors = adjacency_matrix.filter(pl.col("County GEOID") == fips)
        if neighbors.height != 0:
            neighbor_coverages = []
            # neighbor_nme = []
            # neighbor_me = []
            # neighbor_populations = []
            for neighbor_fips in neighbors["Neighbor GEOID"].to_list():
                vac = vaccinations.filter(
                    (pl.col("fips") == neighbor_fips)
                    & (pl.col("year") == year - 1)
                )
                coverage = vac["estimated_mmr_cov"].to_list()
                # me = vac['med_exempt_pct'].to_list()
                # nme = vac["nonmed_exempt_pct"].to_list()
                if len(coverage) > 0:
                    neighbor_coverages.append(coverage[0])
                # neighbor_me.append(me)
                # neighbor_nme.append(nme)
            # drop Nones from the lists
            neighbor_coverages = [
                cov for cov in neighbor_coverages if cov is not None
            ]
            # average the coverages
            if len(neighbor_coverages) > 0:
                df = df.with_columns(
                    neighbors_coverage=pl.when(
                        (pl.col("fips") == fips) & (pl.col("year") == year)
                    )
                    .then(
                        np.nansum(neighbor_coverages) / len(neighbor_coverages)
                    )
                    .otherwise(pl.col("neighbors_coverage")),
                )
        else:
            print("No neighbors for county:", fips, "year:", year)

    # save dataset and adjacency matrix
    df.write_csv(os.path.join(data_dir, "analysis_dataset.csv"))
    np.savetxt(
        os.path.join(data_dir, "county_adjacency_matrix.csv"),
        assembled_adjacency_matrix,
        delimiter=",",
    )
    np.savetxt(
        os.path.join(data_dir, "county_adjacency_matrix_no_2025_oddities.csv"),
        assembled_adjacency_matrix_no_2025_oddities,
        delimiter=",",
    )
