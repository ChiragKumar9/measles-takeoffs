import os
import pickle

import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy import stats

from outbreak_size import final_outbreak_size

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)

if __name__ == "__main__":
    # load dataset
    df_all = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        schema_overrides={"fips": pl.String},
    )

    for years in [
        [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
        [2021, 2022, 2023, 2024, 2025],
    ]:
        df = df_all.filter(pl.col("year").is_in(years))

        # compute final outbreak size for each row
        population = df["pop_5_9"].to_numpy()
        # calculate immune fraction based on coverage and effectiveness
        mmr_effectiveness = 0.93
        immune = df["estimated_mmr_cov"].to_numpy() * mmr_effectiveness / 100  # type: ignore
        susceptible_fraction = 1 - immune
        # calculate final outbreak size
        R0 = 15.0  # measles basic reproduction number
        # final outbreak size calculation
        final_sizes = np.array(
            [
                final_outbreak_size.final_outbreak_size(R0, s0) * N
                for s0, N in zip(susceptible_fraction, population)
            ]
        )

        # now scale these values to match the scale of the observed outbreaks
        observed_outbreaks = df["cases_outbreak"].to_numpy()
        # fit a linear regression through the origin
        # include state and year fixed effects
        year_fixed_effects = df.select(
            [f"year_{y}" for y in years[1:]]
        ).to_numpy()
        df = df.to_dummies(columns="state", separator="_", drop_first=False)
        state_fixed_effects = [
            col
            for col in df.columns
            if col.startswith("state_") and col != "state_ap"
        ]
        state_fixed_effects = df.select(state_fixed_effects).to_numpy()
        X = np.hstack(
            [
                final_sizes.reshape(-1, 1),
                year_fixed_effects,
                state_fixed_effects,
            ]
        )
        model = sm.GLM(
            observed_outbreaks, X, family=sm.families.Poisson()
        ).fit()

        # get the predictions
        mu = model.fittedvalues

        # extract median and 95th percentile interval
        median = stats.poisson.ppf(0.5, mu)
        lower = stats.poisson.ppf(0.025, mu)
        upper = stats.poisson.ppf(0.975, mu)

        # pickle these three
        with open(
            os.path.join(
                output_dir, f"final_outbreak_size_predictions_{years}.pkl"
            ),
            "wb",
        ) as f:
            pickle.dump(
                {
                    "median": median,
                    "lower": lower,
                    "upper": upper,
                },
                f,
            )
