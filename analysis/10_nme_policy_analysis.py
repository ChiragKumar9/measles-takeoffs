import os
import pickle

import polars as pl
import yaml
from numpyro import enable_x64, set_host_device_count
from sklearn.preprocessing import StandardScaler

enable_x64()
set_host_device_count(4)  # use all cores for MCMC

import numpy as np
import numpyro.distributions as dist
from jax.random import PRNGKey
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median

from outbreak_size import poisson

# read analysis params yaml
with open("params.yml", "r") as f:
    params = yaml.safe_load(f)

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)
fig_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")

if __name__ == "__main__":
    nme_coverage_regression = pl.read_csv(
        os.path.join(
            fig_dir,
            "table_supplemental_mmr_coverage_change_nme_policy_interventions.csv",
        )
    )

    nme_coverage_regression = nme_coverage_regression.filter(
        (pl.col("State") == "All states") & (pl.col("Year") == "1 year after")
    )
    slope = nme_coverage_regression["Slope (95% CI)"][0]
    # get whatever is before the " ["
    slope_median = str(slope).split(" [")
    slope_lower = float(slope_median[1].split(", ")[0])
    slope_upper = float(slope_median[1].split(", ")[1].replace("]", ""))
    slope_median = float(slope_median[0])
    intercept = nme_coverage_regression["Intercept"][0]

    for slope, name in zip(
        [slope_median, slope_lower, slope_upper], ["median", "lower", "upper"]
    ):
        df = pl.read_csv(
            os.path.join(output_dir, "projection_dataset.csv"),
            schema_overrides={"fips": pl.String},
        )

        # we want to make different "versions" of df based on our scenarios
        # we want to run one scenario where we set NMEs to 0 for every county
        # and evaluate what happens
        # this is equivalent to setting NMEs to 0 in the top 5 highest risk states
        # and top 5 highest NME states and top 250 counties
        mmr = df["estimated_mmr_cov"]
        nme = df["nonmed_exempt_pct"]
        df_no_nmes = df.with_columns(
            nonmed_exempt_pct=pl.lit(0.0),
            estimated_mmr_cov=pl.col("estimated_mmr_cov")
            + slope * pl.col("nonmed_exempt_pct")
            + intercept,
            estimated_mmr_cov_raw=mmr,
            nonmed_exempt_pct_raw=nme,
        )
        # make sure that no values of mmr coverage are greater than 100
        df_no_nmes = df_no_nmes.with_columns(
            estimated_mmr_cov=pl.when(pl.col("estimated_mmr_cov") > 100)
            .then(100)
            .otherwise(pl.col("estimated_mmr_cov"))
        )
        # add label
        df_no_nmes = df_no_nmes.with_columns(remove_nmes=pl.lit(True))

        # we also want to do an analysis where we add NMEs back to the states that
        # have banned them
        states_with_policies = {
            "California": 2016,
            "Maine": 2021,
            "New York": 2019,
            "Connecticut": 2022,
        }

        vac_df = pl.read_csv(
            os.path.join(data_dir, "processed_vaccinations_data.csv"),
            schema_overrides={"nonmed_exempt_pct": pl.Float64},
        )

        for state, year in states_with_policies.items():
            vac_state_year = vac_df.filter(
                (pl.col("state") == state) & (pl.col("year") == year - 1)
            )
            if vac_state_year.height == 0:
                continue

            df = df.join(
                vac_state_year[["state", "county", "nonmed_exempt_pct"]],
                on=["state", "county"],
                how="left",
                suffix=f"_pre_policy_{state}",
            )

            # replace the nme values with these
            # recall that coverage change = slope * nme_pct + intercept
            df = df.with_columns(
                nonmed_exempt_pct=pl.when(
                    pl.col(
                        f"nonmed_exempt_pct_pre_policy_{state}"
                    ).is_not_null()
                    & (pl.col("state") == state)
                )
                .then(pl.col(f"nonmed_exempt_pct_pre_policy_{state}"))
                .otherwise(pl.col("nonmed_exempt_pct")),
            ).with_columns(
                # and reduce the coverage accordingly
                estimated_mmr_cov=pl.when(
                    pl.col(
                        f"nonmed_exempt_pct_pre_policy_{state}"
                    ).is_not_null()
                    & (pl.col("state") == state)
                )
                .then(
                    pl.col("estimated_mmr_cov")
                    - slope * pl.col(f"nonmed_exempt_pct_pre_policy_{state}")
                    - intercept
                )
                .otherwise(pl.col("estimated_mmr_cov")),
            )

        df = df.filter(
            pl.col("state").is_in(list(states_with_policies.keys()))
        )
        # add label
        df = df.with_columns(add_nmes_back=pl.lit(True))

        df = pl.concat([df, df_no_nmes], how="diagonal_relaxed")

        train_df = pl.read_csv(
            os.path.join(data_dir, "analysis_dataset.csv"),
            schema_overrides={"fips": pl.String},
        ).filter(pl.col("year").is_in([2021, 2022, 2023, 2024, 2025]))

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
        train_df_scaled = scaler.fit_transform(train_df_to_scale.to_numpy())
        train_df = train_df.with_columns(
            pl.DataFrame(
                train_df_scaled,
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
                ],
            )
        )

        print("Scaling projection data...")
        scaled_array = scaler.transform(
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
                ]
            ).to_numpy()
        )

        # save the mmr coverage values before scaling
        df = df.with_columns(
            pl.DataFrame(
                scaled_array,
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
                ],
            )
        )

        # concatenate the two arrays
        train_df = train_df.with_columns(in_train=pl.lit(1))
        df = df.with_columns(
            in_train=pl.lit(0),
            # we just need some value here
            cases_outbreak=pl.lit(0),
        )

        df = pl.concat([train_df, df], how="diagonal_relaxed")

        df = df.drop_nans(
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

        df.write_csv(os.path.join(output_dir, "nme_policy_analysis.csv"))

        model = poisson.PartialPooledPoisson(
            iv=df.select(
                ["estimated_mmr_cov", "nonmed_exempt_pct", "med_exempt_pct"]
            ).to_jax(),
            pool=True,
            regression_covariates=df.select(
                [
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
            ).to_jax(),
            exposure=df["pop"].to_jax(),
            groups=df["state"],
            group_fixed_effect=False,
            spatial=False,
            spatial_units=df["state"],
            adjacency_matrix=np.ones((1, 1)),
            spatial_sigma=dist.HalfNormal(1.0),
        )

        # fit model
        nuts_kernel = NUTS(
            model.fit,
            init_strategy=init_to_median(),
            target_accept_prob=0.99,
        )
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=4000,
            num_samples=2000,
            num_chains=4,
        )
        mcmc.run(
            PRNGKey(params["seed"]),
            is_observed=df["in_train"].to_jax(),
            cases=df["cases_outbreak"].to_jax(),
        )

        mcmc.print_summary()

        posterior_samples = mcmc.get_samples()

        # pass the posterior samples to the model to generate predictions
        predictive = Predictive(model.fit, posterior_samples)

        predictive_samples = predictive(
            PRNGKey(params["seed"]),
            is_observed=df["in_train"].to_jax(),
            cases=None,
        )

        # extract just the cases and take only those for the test year
        all_predictions = predictive_samples["cases"][
            :, (df["in_train"] == 0).to_jax()
        ]

        # pickle it
        with open(
            os.path.join(
                output_dir,
                f"nme_policy_analysis_{name}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(all_predictions, f)
