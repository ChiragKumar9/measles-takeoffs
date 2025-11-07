import numpy as np
import polars as pl


def assemble_adjacency_matrix(
    adjacency_data: pl.DataFrame, fips_list: list[str]
):
    # only include counties in our dataset
    adjacency_data = adjacency_data.filter(
        pl.col("County GEOID").is_in(fips_list)
    )
    # convert to adjacency matrix
    num_counties = len(fips_list)
    adjacency_matrix = np.zeros((num_counties, num_counties))
    for idx, fip in enumerate(fips_list):
        # check that county has some neighbors
        neighbors = adjacency_data.filter(pl.col("County GEOID") == fip)
        if neighbors.height != 0:
            for row in neighbors.iter_rows(named=True):
                # is this county in our dataset?
                if row["Neighbor GEOID"] in fips_list:
                    neighbor_idx = fips_list.index(row["Neighbor GEOID"])
                    adjacency_matrix[idx, neighbor_idx] = 1

    return adjacency_matrix
