import os

import polars as pl

from outbreak_size.analysis_dataset import process_dataset

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
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

    # The datasets of the years up to 2025 (the files of 00_explore_data.py
    # without a suffix) and those with the cases of 2026 (_with_2026), each
    # with its own adjacency matrices: the counties of the two differ.
    for suffix in ("", "_with_2026"):
        # Process original (non-imputed) dataset
        process_dataset(
            input_file=os.path.join(
                data_dir, f"outbreaks_vaccinations_cleaned{suffix}.csv"
            ),
            output_analysis_file=os.path.join(
                data_dir, f"analysis_dataset{suffix}.csv"
            ),
            output_adjacency_file=os.path.join(
                data_dir, f"county_adjacency_matrix{suffix}.csv"
            ),
            output_adjacency_no_oddities_file=os.path.join(
                data_dir,
                f"county_adjacency_matrix_no_2025_oddities{suffix}.csv",
            ),
            adjacency_matrix=adjacency_matrix,
            vaccinations=vaccinations,
        )

        # Process imputed dataset
        if not os.path.exists(
            os.path.join(
                data_dir, f"outbreaks_vaccinations_cleaned_imputed{suffix}.csv"
            )
        ):
            print(
                f"skipped: outbreaks_vaccinations_cleaned_imputed{suffix}.csv not provided, so the imputed dataset was not assembled."
            )
            continue
        process_dataset(
            input_file=os.path.join(
                data_dir, f"outbreaks_vaccinations_cleaned_imputed{suffix}.csv"
            ),
            output_analysis_file=os.path.join(
                data_dir, f"analysis_dataset_imputed{suffix}.csv"
            ),
            output_adjacency_file=os.path.join(
                data_dir, f"county_adjacency_matrix_imputed{suffix}.csv"
            ),
            output_adjacency_no_oddities_file=os.path.join(
                data_dir,
                f"county_adjacency_matrix_no_2025_oddities_imputed{suffix}.csv",
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
