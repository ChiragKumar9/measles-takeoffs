"""
The analysis dataset: the cleaned outbreak and vaccination records of each
county-year with year indicators, the 2025 flags and the neighbors' coverage,
and the county adjacency matrices that go with it (see
analysis/01_assemble_analysis_dataset.py), and the merge of the outbreak
records with the vaccination data that precedes it (see
analysis/00_explore_data.py).
"""

import numpy as np
import polars as pl

from outbreak_size.adjacency import (
    add_connecticut_counties,
    assemble_adjacency_matrix,
)
from outbreak_size.data_year import county_data_year_of_outbreak


def process_dataset(
    input_file: str,
    output_analysis_file: str,
    output_adjacency_file: str,
    output_adjacency_no_oddities_file: str,
    adjacency_matrix: pl.DataFrame,
    vaccinations: pl.DataFrame,
):
    """
    Process a dataset through the full analysis pipeline.

    Parameters:
    -----------
    input_file : str
        Path to input CSV (outbreaks_vaccinations_cleaned.csv or _imputed.csv)
    output_analysis_file : str
        Path to output analysis dataset CSV
    output_adjacency_file : str
        Path to output adjacency matrix CSV
    output_adjacency_no_oddities_file : str
        Path to output adjacency matrix (no 2025 oddities) CSV
    adjacency_matrix : pl.DataFrame
        County adjacency data
    vaccinations : pl.DataFrame
        Vaccination data for neighbor coverage calculation
    """
    print(f"\n{'=' * 80}")
    print(f"Processing: {input_file}")
    print(f"{'=' * 80}")

    # Load dataset
    df = pl.read_csv(
        input_file,
        schema_overrides={"fips": pl.String},
    )

    # make year dummy variables
    df = df.sort("year", descending=True, maintain_order=True)
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
                "Central Utah, Utah",
                "Region 1, Louisiana",
                "Central Region, Virginia",
                "Piedmont, Virginia",
            ]
        )
    )

    # a 2025 "oddity" is a 2025 county-year with fewer than 20 outbreak cases
    # (used to fit the models with and without these small 2025 outbreaks)
    df = df.with_columns(
        oddities_2025=pl.when(
            (pl.col("year").is_in([2025]))
            # & (pl.col("state").is_in(["Minnesota"]))
            & (pl.col("cases_outbreak") < 20)
        )
        .then(pl.lit(True))
        .otherwise(pl.lit(False))
    )
    # print out these places
    print("2025 oddities:")
    print(
        df.filter(pl.col("oddities_2025") == True)[
            ["state", "county", "cases_outbreak", "estimated_mmr_cov"]
        ]
    )

    # the neighbors of Connecticut's counties, for the matrices and for the
    # coverage of the neighbors below
    adjacency_matrix = add_connecticut_counties(adjacency_matrix)
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
    # for each county, iterate through, get neighbors, and average (without
    # weights) their vaccine coverage
    df = df.with_columns(neighbors_coverage=pl.lit(None))

    for fips, year in zip(df["fips"].to_list(), df["year"].to_list()):
        # what are the county's neighbors?
        neighbors = adjacency_matrix.filter(pl.col("County GEOID") == fips)
        if neighbors.height != 0:
            neighbor_coverages = []
            for neighbor_fips in neighbors["Neighbor GEOID"].to_list():
                vac = vaccinations.filter(
                    (pl.col("fips") == neighbor_fips)
                    & (pl.col("year") == county_data_year_of_outbreak(year))
                )
                coverage = vac["estimated_mmr_cov"].to_list()
                if len(coverage) > 0:
                    neighbor_coverages.append(coverage[0])
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

    # save dataset and adjacency matrices
    df.write_csv(output_analysis_file)
    np.savetxt(
        output_adjacency_file,
        assembled_adjacency_matrix,
        delimiter=",",
    )
    np.savetxt(
        output_adjacency_no_oddities_file,
        assembled_adjacency_matrix_no_2025_oddities,
        delimiter=",",
    )

    print(
        f"Saved analysis dataset: {output_analysis_file} ({df.shape[0]} rows)"
    )
    print(f"Saved adjacency matrix: {output_adjacency_file}")
    print(
        f"Saved adjacency matrix (no oddities): {output_adjacency_no_oddities_file}"
    )


def merge_outbreaks_vaccinations(
    outbreaks_df: pl.DataFrame, vaccinations_df: pl.DataFrame
) -> pl.DataFrame:
    outbreaks_df = outbreaks_df.with_columns(
        year_relevant_vaccination=county_data_year_of_outbreak(pl.col("year"))
    )
    print("Outbreaks df shape before merge:", outbreaks_df.shape)
    # each case row takes the vaccination row of its county from the year
    # before, and the case rows keep their order
    merged_df = outbreaks_df.join(
        vaccinations_df,
        left_on=["year_relevant_vaccination", "state", "county"],
        right_on=["year", "state", "county"],
        how="left",
        maintain_order="left",
    )
    # a county with two vaccination rows in a year would repeat its case row.
    # (Repeated rows of places that have no case row are harmless: the data
    # has one, a school district.)
    if merged_df.height != outbreaks_df.height:
        key = ["year", "state", "county"]
        repeated = vaccinations_df.filter(
            vaccinations_df.select(key).is_duplicated()
        )
        raise ValueError(
            f"the merge turned {outbreaks_df.height} case rows into "
            f"{merged_df.height}: the vaccination data repeat a county in a "
            f"year, e.g. {repeated.select(key).unique().head(5).rows()}"
        )
    merged_df = merged_df.with_columns(
        fips=pl.when(pl.col("fips_right").is_not_null())
        .then(pl.col("fips_right"))
        .otherwise(pl.col("fips"))
    )
    merged_df = merged_df.drop(["fips_right"])

    # "New York City" cases were assigned to Kings above, but the 2025 row
    # still gets the case sheet's own NYC fips (Manhattan, 36061) because
    # there's no 2024 NY vaccine data for any county to join against; fix it
    # to Kings' own fips so it isn't attributed to the wrong county
    merged_df = merged_df.with_columns(
        fips=pl.when(
            (pl.col("state") == "New York") & (pl.col("county") == "Kings")
        )
        .then(pl.lit("36047"))
        .otherwise(pl.col("fips"))
    )

    print("Merged df shape after merge:", merged_df.shape)
    return merged_df


VACCINE_FIELDS = ["estimated_mmr_cov", "nonmed_exempt_pct", "med_exempt_pct"]


def fill_from_earlier_school_year(
    merged_df: pl.DataFrame,
    vaccinations_df: pl.DataFrame,
    outbreak_year: int,
    regions: list[str],
    slopes: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """
    The rows of `outbreak_year` that lack a vaccine value (the school year
    before it, outbreak_year - 1, has none for the county) take the place's
    own value of the school year before that (outbreak_year - 2), field by
    field, with its fips if the row has none (a name shared by a county and
    an independent city, such as Fairfax, has none until a vaccination row
    gives it).

    `slopes` (state, county and a `<field>_slope` per field, as
    baseline.place_slopes gives them, from the data before the school year
    the outbreak year uses) trends the earlier value one school year forward
    to the school year the row stands for, clipped to 0-100; without it the
    value is taken as it is.

    This is the step of the imputation from bordering years that takes the
    county's own earlier year, without its CDC state fallback and without
    setting `imputation_source`: the filled rows are rows of the main data
    (analysis/00_explore_data.py writes the main file from the rows whose
    `imputation_source` is null). `vaccinations_df` is the original
    (non-imputed) table, so the values are the county's or the region's own.
    A boolean column `vaccine_from_earlier_year` marks the rows that took a
    value (False for every other row).

    `regions` are the names of the multi-county places, whose fips is null.
    """
    needs = pl.any_horizontal([pl.col(c).is_null() for c in VACCINE_FIELDS])
    is_target = (pl.col("year") == outbreak_year) & needs
    targets = merged_df.filter(is_target).select("state", "county")
    if targets.is_duplicated().any():
        raise ValueError(
            f"{outbreak_year}: a place has two rows, so its earlier year "
            f"cannot be matched: {targets.filter(targets.is_duplicated()).rows()}"
        )
    earlier = (
        # the school year before the one the outbreak year uses
        vaccinations_df.filter(
            pl.col("year") == county_data_year_of_outbreak(outbreak_year) - 1
        )
        .select(
            "state",
            "county",
            *[pl.col(c).alias(f"{c}_earlier") for c in VACCINE_FIELDS],
            pl.col("fips").alias("fips_earlier"),
            pl.col("coverage_source").alias("coverage_source_earlier"),
        )
        .join(targets, on=["state", "county"], how="semi")
    )
    if earlier.select("state", "county").is_duplicated().any():
        raise ValueError(
            f"{outbreak_year - 2}: the vaccination data repeat a place that "
            "needs its earlier year, so its values cannot be matched"
        )
    if slopes is not None:
        earlier = (
            earlier.join(
                slopes.select(
                    "state", "county", *[f"{c}_slope" for c in VACCINE_FIELDS]
                ),
                on=["state", "county"],
                how="left",
            )
            .with_columns(
                *[
                    (
                        pl.col(f"{c}_earlier")
                        + pl.col(f"{c}_slope").fill_null(0.0)
                    )
                    .clip(0.0, 100.0)
                    .alias(f"{c}_earlier")
                    for c in VACCINE_FIELDS
                ]
            )
            .drop([f"{c}_slope" for c in VACCINE_FIELDS])
        )
    n_before = merged_df.height
    out = merged_df.join(
        earlier, on=["state", "county"], how="left", maintain_order="left"
    )
    if out.height != n_before:
        raise ValueError(
            "filling from the earlier school year changed the rows"
        )

    takes = pl.col("year") == outbreak_year
    taken = [
        takes & pl.col(c).is_null() & pl.col(f"{c}_earlier").is_not_null()
        for c in VACCINE_FIELDS
    ]
    out = out.with_columns(
        vaccine_from_earlier_year=pl.any_horizontal(taken).fill_null(False),
        coverage_source=pl.when(taken[0])
        .then(pl.col("coverage_source_earlier"))
        .otherwise(pl.col("coverage_source")),
        fips=pl.when(
            takes & pl.col("fips").is_null() & ~pl.col("county").is_in(regions)
        )
        .then(pl.col("fips_earlier"))
        .otherwise(pl.col("fips")),
    ).with_columns(
        *[
            pl.when(takes)
            .then(pl.coalesce(pl.col(c), pl.col(f"{c}_earlier")))
            .otherwise(pl.col(c))
            .alias(c)
            for c in VACCINE_FIELDS
        ]
    )
    return out.drop(
        [f"{c}_earlier" for c in VACCINE_FIELDS]
        + ["fips_earlier", "coverage_source_earlier"]
    )


def require_vaccine_data(
    merged_df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    merged_df_return = merged_df.filter(
        pl.col("estimated_mmr_cov").is_not_null()
        & pl.col("nonmed_exempt_pct").is_not_null()
        & pl.col("med_exempt_pct").is_not_null()
    )
    # also get out the opposite
    merged_df_no_vaccine_data = merged_df.filter(
        pl.col("estimated_mmr_cov").is_null()
        | pl.col("nonmed_exempt_pct").is_null()
        | pl.col("med_exempt_pct").is_null()
    )
    return merged_df_return, merged_df_no_vaccine_data


def assemble_analysis_dataset(merged_df: pl.DataFrame) -> pl.DataFrame:
    # include = pl.col("cases") == 1
    # include = include | (pl.col("imported") == pl.col("cases"))
    # include = include | (pl.col("imported") < pl.col("cases"))
    # include = include | (pl.col("cases") >= params["inclusion_threshold"])
    # merged_df = merged_df.filter(include)

    merged_df = merged_df.with_columns(
        cases_outbreak=pl.when(
            pl.col("imported").is_null() | (pl.col("imported") == 0)
        )
        .then(pl.col("cases"))
        .otherwise(pl.col("cases") - pl.col("imported")),
        total_exempt_pct=pl.col("nonmed_exempt_pct")
        + pl.col("med_exempt_pct"),
    )

    return merged_df
