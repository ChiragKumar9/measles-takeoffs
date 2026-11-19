import warnings
from collections import Counter

import numpy as np
import polars as pl

# Pairs of bordering counties that include one of Connecticut's eight counties
# (09001-09015), from the Census county adjacency file of 2010
# (https://www2.census.gov/geo/docs/reference/county_adjacency.txt; legal
# boundaries, so pairs across Long Island Sound count, as in the rest of our
# adjacency file). Our adjacency file lists Connecticut by its nine planning
# regions (2022 on, 09110-09190), but the case and vaccination data use the
# counties.
CONNECTICUT_COUNTY_PAIRS = [
    ("09001", "09005"), ("09001", "09009"), ("09001", "36027"),
    ("09001", "36059"), ("09001", "36079"), ("09001", "36103"),
    ("09001", "36119"), ("09003", "09005"), ("09003", "09007"),
    ("09003", "09009"), ("09003", "09011"), ("09003", "09013"),
    ("09003", "25013"), ("09005", "09009"), ("09005", "25003"),
    ("09005", "25013"), ("09005", "36027"), ("09007", "09009"),
    ("09007", "09011"), ("09007", "36103"), ("09009", "36103"),
    ("09011", "09013"), ("09011", "09015"), ("09011", "36103"),
    ("09011", "44003"), ("09011", "44009"), ("09013", "09015"),
    ("09013", "25013"), ("09013", "25027"), ("09015", "25027"),
    ("09015", "44003"), ("09015", "44007"),
]  # fmt: skip


def add_connecticut_counties(adjacency_data: pl.DataFrame) -> pl.DataFrame:
    """
    The adjacency data with the neighbors of Connecticut's eight counties
    added (each pair in both directions), unless the data already list them.
    The planning regions (09110-09190) are removed: they are no counties of
    our data, and left in they would link counties as stepping stones (two
    counties that border the same region count as neighbors of neighbors).
    """
    if "09001" in set(adjacency_data["County GEOID"]):
        return adjacency_data
    adjacency_data = adjacency_data.filter(
        ~(
            pl.col("County GEOID").str.starts_with("09")
            & (pl.col("County GEOID") >= "09100")
        ).fill_null(False)
        & ~(
            pl.col("Neighbor GEOID").str.starts_with("09")
            & (pl.col("Neighbor GEOID") >= "09100")
        ).fill_null(False)
    )
    pairs = CONNECTICUT_COUNTY_PAIRS + [
        (b, a) for a, b in CONNECTICUT_COUNTY_PAIRS
    ]
    added = pl.DataFrame(
        {
            "County GEOID": [a for a, _ in pairs],
            "Neighbor GEOID": [b for _, b in pairs],
        }
    )
    return pl.concat([adjacency_data, added], how="diagonal_relaxed")


def require_county_order(matrix_fips, row_fips, label):
    """
    Raise unless the rows of an adjacency matrix are the counties of row_fips
    in sorted order. spearman_msr_blocked numbers a county by its place in
    np.unique(row_fips), so a matrix built in another order (or for other
    counties) would give a county another county's neighbours without error.
    """
    unique_fips = np.unique(np.asarray(row_fips)).tolist()
    if list(matrix_fips) != unique_fips:
        raise ValueError(
            f"{label}: the adjacency matrix has {len(matrix_fips)} rows that "
            f"are not the {len(unique_fips)} counties of the data in sorted "
            "order"
        )


def assemble_adjacency_matrix(
    adjacency_data: pl.DataFrame, fips_list: list[str]
):
    # a county listed twice would get its neighbors in one row only (the
    # neighbor lookup below finds the first position); a fips that is no county
    # of the file (a region's name, a state's code) has no neighbors either way
    adjacency_data = add_connecticut_counties(adjacency_data)
    counties_of_file = set(adjacency_data["County GEOID"])
    # a county without a row in the file is silently given no neighbors
    unlisted = sorted(
        fip
        for fip in set(fips_list)
        if len(fip) == 5 and fip.isdigit() and fip not in counties_of_file
    )
    if unlisted:
        warnings.warn(
            f"counties without neighbors in the adjacency data: {unlisted}"
        )
    repeated = {
        fip
        for fip, n in Counter(fips_list).items()
        if n > 1 and fip in counties_of_file
    }
    if repeated:
        raise ValueError(
            f"counties listed more than once in the fips list: {sorted(repeated)}"
        )
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

    # the file lists each pair in both directions; the spatial models need a
    # symmetric matrix
    if not (adjacency_matrix == adjacency_matrix.T).all():
        raise ValueError(
            "the adjacency data list a pair of counties in one direction only"
        )
    return adjacency_matrix


def require_same_unit_order(fips, geoid, label):
    """
    Raise unless numbering the rows' counties by the first appearance of their
    fips gives the same numbers as by the first appearance of their geoid.
    The adjacency matrices of the models are built in the order the fips first
    appear, and the models number their spatial units by the order the geoids
    first appear, so a place sharing its fips with another (or two fips for
    one geoid) would give a unit the neighbours of another one without error.
    """

    def numbering(values):
        number = {}
        return [number.setdefault(value, len(number)) for value in values]

    if numbering(fips) != numbering(geoid):
        raise ValueError(
            f"{label}: the spatial units (geoid) and the rows of the "
            "adjacency matrix (fips) are not the same places in the same "
            "order"
        )


def state_adjacency_matrix(adjacency_data: pl.DataFrame, state_prefixes):
    """
    The adjacency matrix of states, in the order of state_prefixes (their two
    digit state FIPS): two states are adjacent when a county of one borders a
    county of the other in the county adjacency file (every county in the
    file counts, not only the counties of a dataset).
    """
    prefixes = list(state_prefixes)
    position = {prefix: i for i, prefix in enumerate(prefixes)}
    matrix = np.zeros((len(prefixes), len(prefixes)))
    pairs = (
        adjacency_data.select(
            pl.col("County GEOID").str.slice(0, 2).alias("a"),
            pl.col("Neighbor GEOID").str.slice(0, 2).alias("b"),
        )
        .filter(pl.col("a") != pl.col("b"))
        .unique()
    )
    for a, b in pairs.iter_rows():
        if a in position and b in position:
            matrix[position[a], position[b]] = 1
            matrix[position[b], position[a]] = 1
    return matrix
