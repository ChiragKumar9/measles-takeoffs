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
        for five_year in [False, True]:
            df = df_all.filter(pl.col("year").is_in(years))

            # compute final outbreak size for each row
            population = df["pop"].to_numpy()
            # calculate immune fraction based on coverage and effectiveness
            mmr_effectiveness = 0.97
            coverage_col = (
                "estimated_mmr_cov_5yr" if five_year else "estimated_mmr_cov"
            )
            immune = df[coverage_col].to_numpy() * mmr_effectiveness / 100  # type: ignore
            susceptible_fraction = 1 - immune
            # calculate final outbreak size
            R0 = 12.0  # measles basic reproduction number
            Re = R0 * susceptible_fraction

            # compute per-row final sizes and variances
            # for Re >= 1, final_outbreak_size returns z_final (infected fraction)
            # for Re < 1, returns expected count in individuals
            results = [
                final_outbreak_size.final_outbreak_size(R0, s0, N)
                for s0, N in zip(susceptible_fraction, population)
            ]
            final_sizes = np.array([r[0] for r in results])
            final_variances = np.array(
                [r[1] if r[1] is not None else np.nan for r in results]
            )

            # split rows by regime
            subcritical_mask = Re < 1
            supercritical_mask = ~subcritical_mask

            observed_outbreaks = df["cases_outbreak"].to_numpy()

            # --- subcritical rows: use stuttering chains directly ---
            # final_sizes for subcritical rows are already expected counts
            # (capped at N), so use them directly as median predictions.
            # Approximate 95% interval via Poisson quantiles around the mean.
            sc_mu = final_sizes[subcritical_mask]
            sc_median = stats.poisson.ppf(0.5, sc_mu)
            sc_lower = stats.poisson.ppf(0.025, sc_mu)
            sc_upper = stats.poisson.ppf(0.975, sc_mu)

            # --- supercritical rows: Poisson GLM with intercept and
            # population offset. final_sizes here are z_final fractions,
            # so log(population) enters as an exposure term. ---
            df_dummies = df.to_dummies(
                columns="state", separator="_", drop_first=True
            )
            year_fixed_effects = df.select(
                [f"year_{y}" for y in years[1:]]
            ).to_numpy()
            state_fixed_effect_cols = [
                col
                for col in df_dummies.columns
                if col.startswith("state_") and col != "state_ap"
            ]
            state_fixed_effects = df_dummies.select(
                state_fixed_effect_cols
            ).to_numpy()

            X_super = np.hstack(
                [
                    final_sizes[supercritical_mask].reshape(-1, 1),
                    year_fixed_effects[supercritical_mask],
                    state_fixed_effects[supercritical_mask],
                ]
            )
            X_super = sm.add_constant(X_super)

            model = sm.GLM(
                observed_outbreaks[supercritical_mask],
                X_super,
                family=sm.families.Poisson(),
                # offset=np.log(population[supercritical_mask]),
            ).fit()

            print(model.summary())

            sp_mu = model.fittedvalues
            sp_median = stats.poisson.ppf(0.5, sp_mu)
            sp_lower = stats.poisson.ppf(0.025, sp_mu)
            sp_upper = stats.poisson.ppf(0.975, sp_mu)

            # --- recombine into full-length arrays in original row order ---
            n = len(observed_outbreaks)
            median = np.empty(n)
            lower = np.empty(n)
            upper = np.empty(n)

            median[subcritical_mask] = sc_median
            lower[subcritical_mask] = sc_lower
            upper[subcritical_mask] = sc_upper

            median[supercritical_mask] = sp_median
            lower[supercritical_mask] = sp_lower
            upper[supercritical_mask] = sp_upper

            # pickle results
            with open(
                os.path.join(
                    output_dir,
                    f"final_outbreak_size_predictions_{years}_five_year={five_year}.pkl",
                ),
                "wb",
            ) as f:
                pickle.dump(
                    {
                        "median": median,
                        "lower": lower,
                        "upper": upper,
                        "subcritical_mask": subcritical_mask,
                        "supercritical_mask": supercritical_mask,
                    },
                    f,
                )
