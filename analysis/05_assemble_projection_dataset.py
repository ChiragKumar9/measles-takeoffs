import os

import polars as pl
import yaml
from dotenv import load_dotenv

from outbreak_size import synth_pop_clusters, utils

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)

# read analysis params yaml
with open("params.yml", "r") as f:
    params = yaml.safe_load(f)


def add_pop_variables(
    merged_df: pl.DataFrame, year, census_key: str
) -> pl.DataFrame:
    census_query = utils.CensusQuery(census_key)
    fips = merged_df["fips"].to_list()
    years_census = [year] * len(fips)
    pop_df = census_query.query_5_9_pop(years_census, fips)
    merged_df = merged_df.with_columns(
        pop_df[
            [
                "pop",
                "pop_under_5",
                "pop_5_9",
                "school_aged_population",
                "public_schooled",
                "private_schooled",
                "median_household_income",
                "number_graduate_degree",
                "pop_25_up",
                "pop_15_up",
                "fraction_white",
            ]
        ]
    )
    # add a land size variable
    land_df = census_query.query_land_size(fips)
    merged_df = merged_df.join(land_df, on="fips", how="left")

    merged_df = merged_df.with_columns(
        pop_density=(pl.col("pop") / pl.col("land_size_sq_km")),
        # these quantities are calculated only for people aged 5-9
        fraction_public=(pl.col("public_schooled") / pl.col("pop_5_9")),
        fraction_private=(pl.col("private_schooled") / pl.col("pop_5_9")),
        # I previously checked by hand that public+private+not_enrolled = total pop
        fraction_graduate_degree=(
            pl.col("number_graduate_degree") / pl.col("pop_25_up")
        ),
        fraction_school_aged=pl.col("school_aged_population") / pl.col("pop"),
        fraction_under_5=pl.col("pop_under_5") / pl.col("pop"),
        fraction_5_9=pl.col("pop_5_9") / pl.col("pop"),
        log_income=pl.col("median_household_income").log10(),
        log_population=pl.col("pop").log10(),
    ).with_columns(log_population_density=pl.col("pop_density").log10())
    # add household size and std variables
    avg_sizes = []
    std_sizes = []
    for fip in fips:
        avg_size, std_size = synth_pop_clusters.summarize_household_size([fip])
        avg_sizes.append(avg_size)
        std_sizes.append(std_size)
    merged_df = merged_df.with_columns(
        avg_house_size=pl.Series(avg_sizes),
        std_house_size=pl.Series(std_sizes),
    )

    return merged_df


if __name__ == "__main__":
    load_dotenv()

    validation_years = [2021, 2022, 2023, 2024, 2025, 2026]
    for year in validation_years:
        df = pl.read_csv(
            os.path.join(data_dir, f"{year}_vaccination_data.csv"),
            schema_overrides={"fips": pl.String},
        )

        # remove any places that have missing data
        df = df.filter(
            (pl.col("estimated_mmr_cov").is_not_null())
            & (pl.col("med_exempt_pct").is_not_null())
            & (pl.col("nonmed_exempt_pct").is_not_null())
        )

        # remove any places with fips codes greater than five digits
        df = df.filter(pl.col("fips").str.len_chars() == 5)

        # now add the relevant census data
        census_key = os.getenv("CENSUS_API_KEY")
        assert census_key is not None, (
            "CENSUS_API_KEY not found in environment"
        )
        df = add_pop_variables(df, year, census_key)

        # scale the data
        train_df = pl.read_csv(
            os.path.join(data_dir, "analysis_dataset.csv"),
            schema_overrides={"fips": pl.String},
        )

        # only use train data that equals the validation year
        train_df = train_df.filter(pl.col("year") == year)

        train_df = train_df.select(
            [
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
                "pop_density",
                "log_income",
                "fips",
                "state",
                "county",
                "year",
                "pop",
                "cases_outbreak",
                "geoid",
                "cases",
            ]
        ).with_columns(
            in_train=pl.lit(1),
            type=pl.lit("train"),
            estimated_mmr_cov_raw=pl.col("estimated_mmr_cov"),
        )

        projection_df = (
            df.select(
                [
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
                    "pop_density",
                    "log_income",
                    "fips",
                    "state",
                    "county",
                    "year",
                    "pop",
                ]
            )
            .with_columns(
                in_train=pl.lit(0),
                # we need to have some value but it is not used
                cases_outbreak=pl.lit(0),
                cases=pl.lit(0),
                estimated_mmr_cov_raw=df["estimated_mmr_cov"],
                type=pl.lit("projection"),
            )
            .with_columns(
                geoid=pl.when(pl.col("fips").is_null())
                .then(pl.col("county"))
                .otherwise(pl.col("county") + pl.col("fips"))
            )
        )

        # join the two dataframes together
        df = pl.concat([train_df, projection_df], how="diagonal_relaxed")

        # drop places with nas -- means there were too few people there to get census data
        df = df.drop_nans()
        # these arise from places that don't have a real fips code in the synth pop
        df = df.drop_nulls()

        df.write_csv(
            os.path.join(output_dir, f"projection_dataset_{year}.csv")
        )

    # Load the most recent projection dataset as the base
    base_year = max(validation_years)
    base_df = pl.read_csv(
        os.path.join(output_dir, f"projection_dataset_{base_year}.csv"),
        schema_overrides={"fips": pl.String},
    )

    # Keep only the projection rows (not train rows) since those are the
    # counties with current census data we want to reuse
    base_df = base_df.filter(pl.col("type") == "projection")

    # Load the projections file
    projections = pl.read_csv(
        os.path.join(
            data_dir, "vaccine_coverage_projections_all_scenarios.csv"
        ),
        schema_overrides={"fips": pl.String},
    )

    future_years = projections["year"].unique().sort().to_list()
    scenarios = projections["scenario"].unique().sort().to_list()

    for scenario in scenarios:
        for future_year in future_years:
            # Pull the projected values for this scenario and year
            proj_slice = projections.filter(
                (pl.col("scenario") == scenario)
                & (pl.col("year") == future_year)
            ).select(
                [
                    "fips",
                    "projected_coverage",
                    "projected_nonmed_exempt",
                    "projected_med_exempt",
                ]
            )

            # Join projected values onto the base census/demographic data
            out_df = (
                base_df.join(proj_slice, on="fips", how="inner")
                .with_columns(
                    [
                        pl.col("projected_coverage").alias(
                            "estimated_mmr_cov"
                        ),
                        pl.col("projected_coverage").alias(
                            "estimated_mmr_cov_raw"
                        ),
                        pl.col("projected_nonmed_exempt").alias(
                            "nonmed_exempt_pct"
                        ),
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

            out_df.write_csv(
                os.path.join(
                    output_dir,
                    f"projection_dataset_{scenario}_{future_year}.csv",
                )
            )
            print(
                f"Saved: projection_dataset_{scenario}_{future_year}.csv  ({out_df.height} counties)"
            )
