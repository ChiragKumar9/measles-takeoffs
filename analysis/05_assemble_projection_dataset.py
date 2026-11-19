import os

import polars as pl
import yaml
from dotenv import load_dotenv

from outbreak_size import synth_pop_clusters, utils
from outbreak_size.data_year import county_data_year_of_outbreak
from outbreak_size.population import add_census_ratios
from outbreak_size.projection_dataset import (
    assemble_projection_year,
    clean_year_vaccination_data,
    require_same_census_values,
    scenario_projection_rows,
)

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
    census_query = utils.CensusQuery(
        census_key,
        cache_path=utils.CENSUS_CACHE_PATH,
        workers=utils.CENSUS_WORKERS,
    )
    fips = merged_df["fips"].to_list()
    # the census of the year before the outbreak year, as the vaccination data
    years_census = [county_data_year_of_outbreak(year)] * len(fips)
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
    merged_df = merged_df.join(
        land_df, on="fips", how="left", maintain_order="left"
    )

    merged_df = add_census_ratios(merged_df)
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

    # the years with observed outbreaks (2021-2026; 2026 is the headline
    # model's validation year) and 2027, the base of the projections
    validation_years = [2021, 2022, 2023, 2024, 2025, 2026, 2027]
    # the cases of 2011-2026 (the _with_2026 files of 00 and 01)
    analysis_df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset_with_2026.csv"),
        schema_overrides={"fips": pl.String},
    )
    for year in validation_years:
        df = pl.read_csv(
            os.path.join(data_dir, f"{year}_vaccination_data.csv"),
            schema_overrides={"fips": pl.String},
        )

        df = clean_year_vaccination_data(df)

        # now add the relevant census data
        census_key = os.getenv("CENSUS_API_KEY")
        assert census_key is not None, (
            "CENSUS_API_KEY not found in environment"
        )
        df = add_pop_variables(df, year, census_key)

        df, dropped, dropped_train, n_projection, n_train = (
            assemble_projection_year(df, analysis_df, year)
        )
        print(
            f"{year}: {len(dropped)} of {n_projection} counties "
            f"dropped for missing census or household data: {dropped}"
        )
        print(
            f"{year}: {dropped_train.height} of {n_train} "
            f"observed (train) rows dropped: {dropped_train.rows()}"
        )
        # every observed county-year of the year is a train row, once: a
        # train row dropped for a missing census or household value would be
        # missing from the validation and the fit without a word
        if dropped_train.height:
            raise ValueError(
                f"{year}: observed rows dropped from the projection dataset: "
                f"{dropped_train.rows()}"
            )
        # a place that is a train row (script 00) and a projection row has
        # the census values of the same year in both
        require_same_census_values(df)
        train_fips = df.filter(pl.col("type") == "train")["fips"]
        if train_fips.is_duplicated().any():
            raise ValueError(
                f"{year}: a place has two train rows: "
                f"{train_fips.filter(train_fips.is_duplicated()).unique().to_list()}"
            )

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
    # the scenarios are for the outbreak years after the base year: a file
    # of an earlier run (2026-2030) would be projected from the wrong base
    assert future_years == list(range(base_year, 2031)), (
        f"vaccine_coverage_projections_all_scenarios.csv has the years "
        f"{future_years}, not {base_year}-2030: rerun the burden overview"
    )
    scenarios = projections["scenario"].unique().sort().to_list()

    for scenario in scenarios:
        for future_year in future_years:
            out_df, lost = scenario_projection_rows(
                base_df, projections, scenario, future_year
            )
            # a county without a projection would be left out of the rows,
            # and the rows are matched to the draws of 07 by position
            if lost:
                raise ValueError(
                    f"{scenario} {future_year}: {len(lost)} counties have no "
                    f"projected values: {sorted(lost)}"
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
