import pickle

import numpy as np
import polars as pl
from numpyro import enable_x64

enable_x64()
import os

import yaml

from outbreak_size import adjacency

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)
# read analysis params yaml
with open("params.yml", "r") as f:
    params = yaml.safe_load(f)

if __name__ == "__main__":
    for name in ["median", "lower", "upper"]:
        no_nme_df = pl.read_csv(
            os.path.join(
                output_dir, "nme_policy_analysis_broken_up_by_scenario.csv"
            ),
            schema_overrides={
                "fips": pl.String,
                "estimated_mmr_cov_raw": pl.Float64,
                "nonmed_exempt_pct_raw": pl.Float64,
            },
        )

        # read the df used to make the risk projections
        risk_df = pl.read_csv(
            os.path.join(output_dir, "scaled_projection_dataset_2026.csv"),
            # make sure we read fips codes as strings
            schema_overrides={
                "fips": pl.String,
                "estimated_mmr_cov_raw": pl.Float64,
            },
        )
        risk_df = risk_df.filter(pl.col("in_train") == 0)

        with open(
            os.path.join(
                output_dir,
                f"nme_policy_analysis_{name}.pkl",
            ),
            "rb",
        ) as f:
            no_nme_samples = pickle.load(f)

        with open(
            os.path.join(
                output_dir,
                "risk_projections.pkl",
            ),
            "rb",
        ) as f:
            risk_samples = pickle.load(f)

        # merge the two
        no_nme_df = no_nme_df.with_columns(
            no_nme_projections=pl.Series(no_nme_samples.T.__array__()),
        )

        risk_df = risk_df.with_columns(
            risk_projections=pl.Series(risk_samples.T.__array__()),
        )

        df = no_nme_df.join(
            risk_df.select(
                [
                    "fips",
                    "state",
                    "county",
                    "year",
                    "risk_projections",
                ]
            ),
            on=["fips", "state", "county", "year"],
            how="left",
        )

        # extract back out our samples
        no_nme_samples = df["no_nme_projections"].to_numpy().T
        risk_samples = df["risk_projections"].to_numpy().T

        # calculate the percent change in incidence
        percent_change_samples = np.where(
            risk_samples != 0,
            (no_nme_samples - risk_samples) / risk_samples * 100,
            np.nan,
        )
        no_outbreak_samples = np.mean(percent_change_samples == -100, axis=0)
        no_outbreak_samples_overall = no_outbreak_samples + np.mean(
            np.isnan(percent_change_samples), axis=0
        )

        percent_change_add_back_nmes = np.where(
            no_nme_samples != 0,
            (risk_samples - no_nme_samples) / no_nme_samples * 100,
            np.nan,
        )
        no_outbreak_samples_add_back_nmes = np.mean(
            percent_change_add_back_nmes == -100, axis=0
        )
        no_outbreak_samples_add_back_nmes_overall = (
            no_outbreak_samples_add_back_nmes
            + np.mean(np.isnan(percent_change_add_back_nmes), axis=0)
        )

        median_percent_change = np.nanmedian(percent_change_samples, axis=0)
        lower_percent_change = np.nanpercentile(
            percent_change_samples, 2.5, axis=0
        )
        upper_percent_change = np.nanpercentile(
            percent_change_samples, 97.5, axis=0
        )

        median_percent_change_add_back_nmes = np.nanmedian(
            percent_change_add_back_nmes, axis=0
        )
        lower_percent_change_add_back_nmes = np.nanpercentile(
            percent_change_add_back_nmes, 2.5, axis=0
        )
        upper_percent_change_add_back_nmes = np.nanpercentile(
            percent_change_add_back_nmes, 97.5, axis=0
        )

        # add these back to the dataframe
        df = df.with_columns(
            median_percent_change=pl.Series(median_percent_change),
            lower_percent_change=pl.Series(lower_percent_change),
            upper_percent_change=pl.Series(upper_percent_change),
            median_percent_change_add_back_nmes=pl.Series(
                median_percent_change_add_back_nmes
            ),
            lower_percent_change_add_back_nmes=pl.Series(
                lower_percent_change_add_back_nmes
            ),
            upper_percent_change_add_back_nmes=pl.Series(
                upper_percent_change_add_back_nmes
            ),
            no_outbreak_samples=pl.Series(no_outbreak_samples),
            no_outbreak_samples_add_back_nmes=pl.Series(
                no_outbreak_samples_add_back_nmes
            ),
            no_outbreak_samples_overall=pl.Series(no_outbreak_samples_overall),
            no_outbreak_samples_add_back_nmes_overall=pl.Series(
                no_outbreak_samples_add_back_nmes_overall
            ),
        ).with_columns(
            median_percent_change=pl.when(pl.col("add_nmes_back") == True)
            .then(pl.col("median_percent_change_add_back_nmes"))
            .otherwise(pl.col("median_percent_change")),
            lower_percent_change=pl.when(pl.col("add_nmes_back") == True)
            .then(pl.col("lower_percent_change_add_back_nmes"))
            .otherwise(pl.col("lower_percent_change")),
            upper_percent_change=pl.when(pl.col("add_nmes_back") == True)
            .then(pl.col("upper_percent_change_add_back_nmes"))
            .otherwise(pl.col("upper_percent_change")),
            no_outbreak_samples=pl.when(pl.col("add_nmes_back") == True)
            .then(pl.col("no_outbreak_samples_add_back_nmes"))
            .otherwise(pl.col("no_outbreak_samples")),
            no_outbreak_samples_overall=pl.when(
                pl.col("add_nmes_back") == True
            )
            .then(pl.col("no_outbreak_samples_add_back_nmes_overall"))
            .otherwise(pl.col("no_outbreak_samples_overall")),
        )

        df.drop(["no_nme_projections", "risk_projections"]).write_csv(
            os.path.join(
                output_dir, f"nme_policy_analysis_assembled_{name}.csv"
            )
        )

    # read the three assembled datasets and combine them
    df_median = pl.read_csv(
        os.path.join(output_dir, "nme_policy_analysis_assembled_median.csv"),
        schema_overrides={"fips": pl.String},
    )
    df_lower = pl.read_csv(
        os.path.join(output_dir, "nme_policy_analysis_assembled_lower.csv"),
        schema_overrides={"fips": pl.String},
    )
    df_upper = pl.read_csv(
        os.path.join(output_dir, "nme_policy_analysis_assembled_upper.csv"),
        schema_overrides={"fips": pl.String},
    )

    # from each df, we want to keep only the relevant columns
    columns = [
        "fips",
        "state",
        "county",
        "year",
        "no_outbreak_samples",
        "no_outbreak_samples_overall",
        "add_nmes_back",
        "median_percent_change",
        "lower_percent_change",
        "upper_percent_change",
    ]

    df_median = df_median.select(columns)
    df_lower = df_lower.select(columns)
    df_upper = df_upper.select(columns)

    df = df_median.join(
        df_lower,
        on=["fips", "state", "county", "year", "add_nmes_back"],
        suffix="_lower_nme_coverage_effect",
    ).join(
        df_upper,
        on=["fips", "state", "county", "year", "add_nmes_back"],
        suffix="_upper_nme_coverage_effect",
    )

    df.write_csv(
        os.path.join(output_dir, "nme_policy_analysis_assembled_output.csv")
    )
