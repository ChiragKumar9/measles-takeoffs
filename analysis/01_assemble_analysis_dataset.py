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
    counties = df["fips"].unique(maintain_order=True).to_list()
    assembled_adjacency_matrix = assemble_adjacency_matrix(
        adjacency_matrix, counties
    )

    # save dataset and adjacency matrix
    df.write_csv(os.path.join(data_dir, "analysis_dataset.csv"))
    np.savetxt(
        os.path.join(data_dir, "county_adjacency_matrix.csv"),
        assembled_adjacency_matrix,
        delimiter=",",
    )
