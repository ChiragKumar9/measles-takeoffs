import os
import pickle

import numpy as np
import polars as pl
import yaml
from scipy import stats

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)

# read analysis params yaml
with open("params.yml", "r") as f:
    params = yaml.safe_load(f)


def read_outbreaks(
    data_dir=data_dir,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    df = pl.read_excel(os.path.join(data_dir, "Combined Measles Dataset.xlsx"))

    # to ensure consistent lower case column names
    df = df.rename({col: col.lower() for col in df.columns})

    df_2025 = pl.read_csv(
        os.path.join(
            data_dir, "measles_data", "measles_county_all_updates_detailed.csv"
        ),
        schema_overrides={"location_id": pl.String},
    )
    # add 0 padding to 4 digit fips codes
    df_2025 = df_2025.with_columns(
        location_id=pl.when(pl.col("location_id").str.len_chars() >= 2)
        .then(pl.col("location_id").str.zfill(5))
        .otherwise(pl.col("location_id"))
    )
    # rename columns to match the main df
    df_2025 = df_2025.rename(
        {
            "location_id": "fips",
            "location_name": "county",
            "value": "cases",
        }
    )

    df_2025 = df_2025.filter(pl.col("date").str.starts_with("2025"))

    df_2025 = df_2025.with_columns(
        year=pl.lit(2025),
        state=pl.col("county").str.split(", ").list.get(1),
        county=pl.col("county").str.split(", ").list.get(0),
    )

    # map the outcome type to differentiate indigenous vs imported cases
    df_2025 = df_2025.with_columns(
        outcome_type=pl.when(pl.col("outcome_type") == "case_imported")
        .then(pl.lit("imported"))
        .otherwise(pl.lit("indigenous"))
    )
    # group by the case type to get imports vs indigenous cases
    df_2025 = df_2025.group_by(
        ["year", "state", "county", "fips", "outcome_type"]
    ).agg(pl.sum("cases"))
    # pivot the outcome type to columns
    df_2025 = df_2025.pivot(on="outcome_type", values="cases")

    # concatenate and return
    df = pl.concat([df, df_2025], how="diagonal_relaxed")

    # remove places where the year is null
    df = df.filter(pl.col("year").is_not_null())
    # if importeds is 0, it must be at the county level
    # (since we add in 0 importeds at the state level below when we join with
    # all state data)
    df = df.filter(
        pl.when(pl.col("imported") == 0)
        .then(pl.col("county").is_not_null())
        .otherwise(True)
    )
    # cases also can't be 0
    df = df.filter(pl.col("cases") > 0)
    # also remove counties with bogus county names
    df = df.filter(
        ~pl.col("county").is_in(["TOTAL", "Unknown", "Out of state resident"])
    )
    # state can't be null
    df = df.filter(pl.col("state").is_not_null())

    counties_with_known_imports = df.filter(
        pl.col("imported").is_not_null() & pl.col("county").is_not_null()
    ).select(["year", "state", "county", "imported"])

    return counties_with_known_imports, df_2025


def tally_state_importations(
    df_2025, data_dir=data_dir
) -> tuple[pl.DataFrame, int]:
    # group by state
    df = pl.read_excel(
        os.path.join(data_dir, "IndigenousImportedFullData.xlsx")
    )
    # rename columns
    df = df.rename({col: col.lower() for col in df.columns})
    # replace "DC" with "District of Columbia"
    df = (
        df.with_columns(
            state=pl.when(pl.col("state") == "DC")
            .then(pl.lit("District of Columbia"))
            .otherwise(pl.col("state"))
        )
        .with_columns(
            # fix North Dakora to be North Dakota
            state=pl.when(pl.col("state") == "North Dakora")
            .then(pl.lit("North Dakota"))
            .otherwise(pl.col("state"))
        )
        .with_columns(
            # idaho to be Idaho
            state=pl.when(pl.col("state") == "idaho")
            .then(pl.lit("Idaho"))
            .otherwise(pl.col("state"))
        )
        .with_columns(
            # "New York " to "New York"
            state=pl.when(pl.col("state") == "New York ")
            .then(pl.lit("New York"))
            .otherwise(pl.col("state"))
        )
    )
    # add in 2025 import data
    df = pl.concat(
        [df, df_2025.select(["state", "year", "imported"])],
        how="diagonal_relaxed",
    )

    years_of_data = df["year"].n_unique()

    df = df.group_by("state").agg(
        [
            pl.col("imported")
            .filter(pl.col("year") < i)
            .sum()
            .alias(f"imported_pre_{i}")
            for i in [2021, 2022, 2023, 2024, 2025]
        ],
        imported=pl.sum("imported"),
    )

    return df, years_of_data


def calculate_yearly_import_probability(
    df, years_of_data, prior_alpha=1, prior_beta=1
):
    df = df.with_columns(observed=pl.lit(True))
    # we now observe all states because we have cdc importations data
    # but we still want to flag them as unobserved because we did not observe
    # an import in those states
    df = df.with_columns(
        observed=pl.when(pl.col("imported") == 0)
        .then(pl.lit(False))
        .otherwise(pl.col("observed"))
    )

    # the number of imports is a Poisson process, and we want to fit the rate
    # parameter
    # using an exponential(1) / gamma(1, 1) prior on the rate parameter, which
    # yields a uniform(0, 1) probability of there being at least one import in
    # a year, we update the gamma parameters

    # use a distribution with uniform prior to calculate import probability
    df = df.with_columns(
        *[
            (pl.col(f"imported_pre_{i}") + prior_alpha).alias(
                f"posterior_alpha_pre_{i}"
            )
            for i in [2021, 2022, 2023, 2024, 2025]
        ],
        *[
            (pl.lit(years_of_data + (2026 - i)) + prior_beta).alias(
                f"posterior_beta_pre_{i}"
            )
            for i in [2021, 2022, 2023, 2024, 2025]
        ],
        posterior_alpha=pl.col("imported") + prior_alpha,
        posterior_beta=pl.lit(years_of_data) + prior_beta,
    )

    return df


def distribute_state_level_risk_to_county(
    df_state, county_data, counties_with_known_imports
) -> tuple[pl.DataFrame, pl.DataFrame]:
    # each state has a population-weighted value of the county's importation probability
    # join state level data to county data

    df = county_data.join(df_state, on="state", how="left")

    # get samples from the posterior
    df = df.with_columns(
        # calculate a bunch of samples for later use
        importation_probability_samples=pl.struct(
            [
                *["posterior_alpha", "posterior_beta", "year", "type"],
                *[
                    f"posterior_alpha_pre_{i}"
                    for i in [2021, 2022, 2023, 2024, 2025]
                ],
                *[
                    f"posterior_beta_pre_{i}"
                    for i in [2021, 2022, 2023, 2024, 2025]
                ],
            ]
        ).map_elements(
            lambda x: stats.gamma.rvs(
                a=x["posterior_alpha_pre_" + str(x["year"])],
                scale=1 / x["posterior_beta_pre_" + str(x["year"])],
                size=8000,
            ).tolist()  # type: ignore
            if x["type"] == "train" and x["year"] >= 2021
            else stats.gamma.rvs(
                a=x["posterior_alpha"],
                scale=1 / x["posterior_beta"],
                size=8000,
            ).tolist(),  # type: ignore
            pl.List(pl.Float64),
        ),
    )
    # calculate the population weight
    df = df.with_columns(
        pop_weight=pl.col("pop")
        / pl.col("pop").sum().over(["state", "year", "type"]),
        pop_weight_log=pl.col("pop").log10()
        / pl.col("pop").log10().sum().over(["state", "year", "type"]),
    )
    # distribute the state level importation probability to counties based on population weight
    df = df.with_columns(
        *[
            (pl.col(f"posterior_alpha_pre_{i}") * pl.col("pop_weight")).alias(
                f"posterior_alpha_pre_{i}"
            )
            for i in [2021, 2022, 2023, 2024, 2025]
        ],
        *[
            (
                pl.col(f"posterior_alpha_pre_{i}") * pl.col("pop_weight_log")
            ).alias(f"posterior_alpha_pre_{i}_log")
            for i in [2021, 2022, 2023, 2024, 2025]
        ],
        importation_probability_samples=pl.struct(
            ["pop_weight", "importation_probability_samples"]
        ).map_elements(
            lambda x: (
                x["pop_weight"]
                * np.array(x["importation_probability_samples"])
            ).tolist(),
            pl.List(pl.Float64),
        ),
        posterior_alpha=pl.col("posterior_alpha") * pl.col("pop_weight"),
        posterior_alpha_log=pl.col("posterior_alpha")
        * pl.col("pop_weight_log"),
    )

    # calculate summary statistics on the importation probabilities
    # first we need to convert the rate parameter to poisson probabilities of
    # at least one import

    # Save the rate samples (expected number of importations) before transforming
    df = df.with_columns(
        expected_importations_samples=pl.col("importation_probability_samples")
    )

    # Transform rate samples to probability of at least one import
    df = df.with_columns(
        importation_probability_samples=pl.col(
            "importation_probability_samples"
        ).map_elements(
            lambda x: (1 - np.exp(-np.array(x))).tolist(), pl.List(pl.Float64)
        )
    )

    # now we can calculate mean, median, 2.5th and 97.5th percentiles
    df = df.with_columns(
        importation_probability_mean=pl.col(
            "importation_probability_samples"
        ).map_elements(lambda x: np.mean(np.array(x)), pl.Float64),
        importation_probability_median=pl.col(
            "importation_probability_samples"
        ).map_elements(lambda x: np.median(np.array(x)), pl.Float64),
        importation_probability_lower=pl.col(
            "importation_probability_samples"
        ).map_elements(lambda x: np.percentile(np.array(x), 2.5), pl.Float64),
        importation_probability_upper=pl.col(
            "importation_probability_samples"
        ).map_elements(lambda x: np.percentile(np.array(x), 97.5), pl.Float64),
    )

    # take only the quantities we need
    df = df.select(
        [
            *[
                f"posterior_alpha_pre_{i}"
                for i in [2021, 2022, 2023, 2024, 2025]
            ],
            *[
                f"posterior_alpha_pre_{i}_log"
                for i in [2021, 2022, 2023, 2024, 2025]
            ],
            *[
                f"posterior_beta_pre_{i}"
                for i in [2021, 2022, 2023, 2024, 2025]
            ],
            *[
                "fips",
                "state",
                "county",
                "year",
                "type",
                "observed",
                "posterior_alpha",
                "posterior_beta",
                "posterior_alpha_log",
                "importation_probability_samples",
                "importation_probability_mean",
                "importation_probability_median",
                "importation_probability_lower",
                "importation_probability_upper",
                "expected_importations_samples",
            ],
        ]
    )

    # counties that we never observe must have no imports
    counties_with_known_imports_never_observed = df.filter(~pl.col("observed"))
    counties_with_known_imports_never_observed = (
        counties_with_known_imports_never_observed.with_columns(
            imported=pl.lit(0)
        )
    )
    # we know that other counties in state-years with known imports besides
    # the particular county-year with a known import also had 0 imports
    # we want to find those state-years
    counties_with_known_imports = counties_with_known_imports.with_columns(
        flag=pl.lit(True)
    )
    state_years_with_known_county_imports = df.join(
        counties_with_known_imports.select(["year", "state", "flag"]).unique(),
        on=["year", "state"],
        how="left",
    )
    # take states with the flag as true
    state_years_with_known_county_imports = (
        state_years_with_known_county_imports.filter(pl.col("flag") == True)
    )
    # drop the counties with known imports
    # make a dummy column of a county-state-year combination identifier
    counties_with_known_imports = counties_with_known_imports.with_columns(
        county_state_year=(
            pl.col("county") + pl.col("state") + pl.col("year").cast(pl.String)
        )
    )
    state_years_with_known_county_imports = (
        state_years_with_known_county_imports.with_columns(
            county_state_year=(
                pl.col("county")
                + pl.col("state")
                + pl.col("year").cast(pl.String)
            )
        )
    )
    state_years_with_known_county_imports = (
        state_years_with_known_county_imports.filter(
            ~pl.col("county_state_year").is_in(
                counties_with_known_imports.select(
                    "county_state_year"
                ).to_series()
            )
        )
    )
    # assign 0 imports to these counties
    state_years_with_known_county_imports = (
        state_years_with_known_county_imports.with_columns(imported=pl.lit(0))
    )

    counties_with_known_imports = pl.concat(
        [
            counties_with_known_imports,
            counties_with_known_imports_never_observed,
            state_years_with_known_county_imports,
        ],
        how="diagonal_relaxed",
    ).select(["year", "state", "county", "imported"])

    return df, counties_with_known_imports


if __name__ == "__main__":
    counties_with_known_imports, df_2025 = read_outbreaks()

    df, years_of_data = tally_state_importations(df_2025)
    df = calculate_yearly_import_probability(df, years_of_data)
    # read in our county data
    validation_years = [2021, 2022, 2023, 2024, 2025, 2026]
    county_data_all_years = []
    for year in validation_years:
        county_data = pl.read_csv(
            os.path.join(output_dir, f"projection_dataset_{year}.csv"),
            schema_overrides={"fips": pl.String},
        )
        county_data_all_years.append(county_data)

    county_data = pl.concat(county_data_all_years, how="diagonal_relaxed")

    # also join in our regular pre 2020 train data
    pre_df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        schema_overrides={"fips": pl.String},
    )
    # mark as train data
    pre_df = pre_df.with_columns(
        type=pl.lit("train"),
    )
    # concatenate with the county data
    county_data = pl.concat([pre_df, county_data], how="diagonal_relaxed")
    county_data = county_data.select(
        [
            "fips",
            "state",
            "county",
            "year",
            "pop",
            "type",
        ]
    )
    # drop duplicates
    county_data = county_data.unique(
        ["fips", "year", "type", "state", "county"]
    )
    df, counties_with_known_imports = distribute_state_level_risk_to_county(
        df, county_data, counties_with_known_imports
    )
    # save
    # save the values separately from the samples

    samples = df.select("importation_probability_samples")
    importation_count_samples = df.select(
        "expected_importations_samples"
    )  # <-- add
    df = df.drop(
        ["importation_probability_samples", "expected_importations_samples"]
    )  # <-- drop both

    with open(
        os.path.join(
            output_dir, "empirical_importation_probability_samples.pkl"
        ),
        "wb",
    ) as f:
        pickle.dump(samples, f)

    with open(  # <-- add
        os.path.join(output_dir, "empirical_importation_count_samples.pkl"),
        "wb",
    ) as f:
        pickle.dump(importation_count_samples, f)

    df.write_csv(
        os.path.join(output_dir, "empirical_importation_probability.csv")
    )
    counties_with_known_imports.write_csv(
        os.path.join(output_dir, "counties_with_known_imports.csv")
    )
