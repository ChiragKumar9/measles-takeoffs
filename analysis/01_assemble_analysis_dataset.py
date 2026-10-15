import os

import numpy as np
import polars as pl

from outbreak_size.adjacency import assemble_adjacency_matrix

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)


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
                "Region 1, Louisiana",
                "Central Region, Virginia",
                "Piedmont, Virginia",
            ]
        )
    )

    # whether a 2025 oddity: is a new state that appears for the first time in 2025
    # states pre 2025
    states_pre_2025 = (
        df.filter(pl.col("year") < 2025)
        .filter(pl.col("year") >= 2021)["state"]
        .unique()
        .to_list()
    )
    print("States pre-2025:", states_pre_2025)
    # Have tried:
    # New Mexico, Wisconson, Ohio, Colorado, Minnesota, South Dakota
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
    df = df.with_columns(
        neighbors_coverage=pl.lit(None),
        neighbors_coverage_pop_weighted=pl.lit(None),
    )

    for fips, year in zip(df["fips"].to_list(), df["year"].to_list()):
        # what are the county's neighbors?
        neighbors = adjacency_matrix.filter(pl.col("County GEOID") == fips)
        if neighbors.height != 0:
            neighbor_coverages = []
            for neighbor_fips in neighbors["Neighbor GEOID"].to_list():
                vac = vaccinations.filter(
                    (pl.col("fips") == neighbor_fips)
                    & (pl.col("year") == year - 1)
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


if __name__ == "__main__":
    # Load shared data
    adjacency_matrix = pl.read_csv(
        os.path.join(data_dir, "county_adjacencies.txt"),  # type: ignore
        separator="|",
        schema_overrides={
            "County GEOID": pl.String,
            "Neighbor GEOID": pl.String,
        },
    )

    vaccinations = pl.read_csv(
        os.path.join(data_dir, "processed_vaccinations_data.csv"),
        schema_overrides={
            "fips": pl.String,
            "estimated_mmr_cov": pl.Float64,
            "nonmed_exempt_pct": pl.Float64,
            "med_exempt_pct": pl.Float64,
        },
    )

    # Process original (non-imputed) dataset
    process_dataset(
        input_file=os.path.join(
            data_dir, "outbreaks_vaccinations_cleaned.csv"
        ),
        output_analysis_file=os.path.join(data_dir, "analysis_dataset.csv"),
        output_adjacency_file=os.path.join(
            data_dir, "county_adjacency_matrix.csv"
        ),
        output_adjacency_no_oddities_file=os.path.join(
            data_dir, "county_adjacency_matrix_no_2025_oddities.csv"
        ),
        adjacency_matrix=adjacency_matrix,
        vaccinations=vaccinations,
    )

    # Process imputed dataset
    if not os.path.exists(
        os.path.join(data_dir, "outbreaks_vaccinations_cleaned_imputed.csv")
    ):
        print(
            "skipped: outbreaks_vaccinations_cleaned_imputed.csv not provided, so the imputed dataset was not assembled."
        )
        raise SystemExit(0)
    process_dataset(
        input_file=os.path.join(
            data_dir, "outbreaks_vaccinations_cleaned_imputed.csv"
        ),
        output_analysis_file=os.path.join(
            data_dir, "analysis_dataset_imputed.csv"
        ),
        output_adjacency_file=os.path.join(
            data_dir, "county_adjacency_matrix_imputed.csv"
        ),
        output_adjacency_no_oddities_file=os.path.join(
            data_dir, "county_adjacency_matrix_no_2025_oddities_imputed.csv"
        ),
        adjacency_matrix=adjacency_matrix,
        vaccinations=vaccinations,
    )

    print("\n" + "=" * 80)
    print("Processing complete!")
    print("=" * 80)

    # # get out missing counties
    # raw = pl.read_csv(
    #     os.path.join(data_dir, "raw_merged_outbreaks_vaccinations.csv"),
    #     schema_overrides={"fips": pl.String},
    # )
    # joined = raw.join(df, on=["state", "county", "year"], how="left")
    # # pick an arbitrary column in df that is not in joined to get missing counties
    # missing = joined.filter(pl.col("mega_region_2025").is_null())
    # missing = missing.select([
    #     'year',
    #     'state',
    #     'county',
    #     'cases',
    #     'indigenous',
    #     'imported',
    #     'estimated_mmr_cov',
    #     'nonmed_exempt_pct',
    #     'med_exempt_pct',
    # ]).with_columns(
    #     cases_outbreak = pl.when(pl.col("imported").is_not_null())
    #     .then(pl.col("cases") - pl.col("imported"))
    #     .otherwise(pl.col("cases"))
    # )
    # missing.write_csv(os.path.join(data_dir, "missing_counties.csv"))
