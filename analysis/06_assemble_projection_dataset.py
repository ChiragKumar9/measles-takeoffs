import os

import numpy as np
import polars as pl
import yaml
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler

from outbreak_size import adjacency, synth_pop_clusters, utils

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)

# read analysis params yaml
with open("params.yml", "r") as f:
    params = yaml.safe_load(f)


def add_pop_variables(
    merged_df: pl.DataFrame, census_key: str
) -> pl.DataFrame:
    census_query = utils.CensusQuery(census_key)
    fips = merged_df["fips"].to_list()
    years_census = [2025] * len(fips)
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

    df = pl.read_csv(
        os.path.join(data_dir, "latest_vaccination_data.csv"),
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

    # drop any years pre 2021
    df = df.filter(pl.col("year") >= 2021)

    # now add the relevant census data
    census_key = os.getenv("CENSUS_API_KEY")
    assert census_key is not None, "CENSUS_API_KEY not found in environment"
    df = add_pop_variables(df, census_key)

    # scale the data
    train_df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        schema_overrides={"fips": pl.String},
    )

    train_df = train_df.filter(
        pl.col("year").is_in([2021, 2022, 2023, 2024, 2025])
    )

    train_df_to_scale = train_df.select(
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
        ]
    )

    scaler = StandardScaler()
    print("Fitting scaler on training data...")
    scaled_train = scaler.fit_transform(train_df_to_scale.to_numpy())

    train_df = pl.DataFrame(
        scaled_train, train_df_to_scale.columns
    ).with_columns(
        state=train_df["state"],
        geoid=train_df["geoid"],
        fips=train_df["fips"],
        county=train_df["county"],
        year=train_df["year"],
        pop=train_df["pop"],
        in_train=pl.lit(1),
        cases_outbreak=train_df["cases_outbreak"],
        type=pl.lit("train"),
    )

    df.write_csv(os.path.join(output_dir, "projection_dataset.csv"))

    # now apply the same scaler to our df
    projection_df = df.select(
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
        ]
    )

    print("Scaling projection data...")
    scaled_array = scaler.transform(projection_df.to_numpy())
    scaled_df = pl.DataFrame(scaled_array, schema=projection_df.columns)

    # add back in fips, state, county, year
    scaled_df = scaled_df.with_columns(
        [
            df["fips"],
            df["state"],
            df["county"],
            df["year"],
            df["pop"],
        ],
        in_train=pl.lit(0),
        # we need to have some value but it is not used
        cases_outbreak=pl.lit(0),
        estimated_mmr_cov_raw=df["estimated_mmr_cov"],
        type=pl.lit("projection"),
    )
    scaled_df = scaled_df.with_columns(
        geoid=pl.when(pl.col("fips").is_null())
        .then(pl.col("county"))
        .otherwise(pl.col("county") + pl.col("fips"))
    )

    # join the two dataframes together
    scaled_df = pl.concat([train_df, scaled_df], how="diagonal_relaxed")

    # drop places with nas -- means there were too few people there to get census data
    scaled_df = scaled_df.drop_nans()

    # save scaled df
    print("Saving scaled projection dataset...")
    scaled_df.write_csv(
        os.path.join(output_dir, "scaled_projection_dataset.csv")
    )

    print("Making adjacency matrix for projection dataset...")
    adjacency_matrix = pl.read_csv(
        os.path.join(data_dir, "county_adjacencies.txt"),
        separator="|",
        schema_overrides={
            "County GEOID": pl.String,
            "Neighbor GEOID": pl.String,
        },
    )
    adj_matrix = adjacency.assemble_adjacency_matrix(
        adjacency_matrix,
        scaled_df["fips"].unique(maintain_order=True).to_list(),
    )
    np.savetxt(
        os.path.join(output_dir, "projection_adjacency_matrix.npy"),
        adj_matrix,
    )
