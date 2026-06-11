from numpyro import enable_x64, set_host_device_count

enable_x64()
set_host_device_count(4)  # use all cores for MCMC

import os
import pickle

import numpyro.distributions as dist
import polars as pl
import yaml
from jax.random import PRNGKey
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from sklearn.preprocessing import StandardScaler

from outbreak_size import adjacency, poisson

# read analysis params yaml
with open("params.yml", "r") as f:
    params = yaml.safe_load(f)

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)

GLOBALS_PRIOR_SD = 1

if __name__ == "__main__":
    # read the projection dataset
    df = pl.read_csv(
        os.path.join(output_dir, "projection_dataset_2026.csv"),
        schema_overrides={"fips": pl.String},
    )

    train = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        schema_overrides={"fips": pl.String},
    )

    train = train.filter(pl.col("year") >= 2019)

    # get the columns we need
    train = train.select(
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
        ]
    ).with_columns(
        in_train=pl.lit(1),
    )

    # concatenate with the projection dataset
    df = df.with_columns(
        in_train=pl.lit(0),
    )
    df = pl.concat([train, df], how="diagonal_relaxed")
    # any null cases_outbreak should be set to 0
    df = df.with_columns(
        cases_outbreak=pl.col("cases_outbreak").fill_null(0).cast(pl.Int64),
    )

    # scale the data
    scaler = StandardScaler()
    print("Fitting scaler on training data...")
    fit_scaler = scaler.fit(
        df.filter(pl.col("in_train") == 1)
        .select(
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
        .to_numpy()
    )

    # apply the scaler to the entire dataset
    scaled_df = pl.DataFrame(
        fit_scaler.transform(
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
        ),
        schema=[
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

    # make an adjacency matrix for the spatial model

    # spatial component
    adjacency_data = pl.read_csv(
        os.path.join(data_dir, "county_adjacencies.txt"),
        separator="|",
        schema_overrides={
            "County GEOID": pl.String,
            "Neighbor GEOID": pl.String,
        },
    )

    adjacency_matrix = adjacency.assemble_adjacency_matrix(
        adjacency_data, df["fips"].unique(maintain_order=True).to_list()
    )

    model = poisson.PartialPooledPoisson(
        iv=scaled_df.select(
            ["estimated_mmr_cov", "nonmed_exempt_pct", "med_exempt_pct"]
        ).to_jax(),
        pool=True,
        regression_covariates=scaled_df.select(
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
        spatial=True,
        spatial_units=df["geoid"],
        adjacency_matrix=adjacency_matrix,
        spatial_sigma=dist.HalfNormal(0.25),
        car_sigma=dist.HalfNormal(0.25),
        global_priors=dist.Normal(0, GLOBALS_PRIOR_SD),
        baseline=dist.Normal(0, GLOBALS_PRIOR_SD),
    )

    # fit model
    nuts_kernel = NUTS(
        model.fit,
        init_strategy=init_to_median(num_samples=50),
        target_accept_prob=0.7,
    )
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=2000,
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

    with open(
        os.path.join(
            output_dir,
            f"posterior_samples_globals_prior_sd={GLOBALS_PRIOR_SD}.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(posterior_samples, f)

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
            f"risk_projections_globals_prior_sd={GLOBALS_PRIOR_SD}.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(all_predictions, f)

    feature_cols = [
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

    future_years = [2026, 2027, 2028, 2029, 2030]
    scenarios = [
        "baseline",
        "targeted_worsening",
        "universal_worsening",
        "targeted_improvement",
        "universal_improvement",
        "no_change",
    ]

    for scenario in scenarios:
        for future_year in future_years:
            print(f"\nRunning scenario={scenario}, year={future_year}")

            proj_df = pl.read_csv(
                os.path.join(
                    output_dir,
                    f"projection_dataset_{scenario}_{future_year}.csv",
                ),
                schema_overrides={"fips": pl.String},
            ).with_columns(in_train=pl.lit(0))

            combined = pl.concat([train, proj_df], how="diagonal_relaxed")
            combined = combined.with_columns(
                cases_outbreak=pl.col("cases_outbreak")
                .fill_null(0)
                .cast(pl.Int64),
            )

            scaled = pl.DataFrame(
                fit_scaler.transform(combined.select(feature_cols).to_numpy()),
                schema=feature_cols,
            )

            scenario_model = poisson.PartialPooledPoisson(
                iv=scaled.select(
                    [
                        "estimated_mmr_cov",
                        "nonmed_exempt_pct",
                        "med_exempt_pct",
                    ]
                ).to_jax(),
                pool=True,
                regression_covariates=scaled.select(
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
                exposure=combined["pop"].to_jax(),
                groups=combined["state"],
                group_fixed_effect=False,
                spatial=True,
                spatial_units=combined["geoid"],
                adjacency_matrix=adjacency_matrix,  # reuse from original run
                spatial_sigma=dist.HalfNormal(0.25),
                car_sigma=dist.HalfNormal(0.25),
            )

            scenario_predictive = Predictive(
                scenario_model.fit, posterior_samples
            )
            scenario_samples = scenario_predictive(
                PRNGKey(params["seed"]),
                is_observed=combined["in_train"].to_jax(),
                cases=None,
            )

            future_predictions = scenario_samples["cases"][
                :, (combined["in_train"] == 0).to_jax()
            ]

            with open(
                os.path.join(
                    output_dir,
                    f"risk_projections_{scenario}_{future_year}_globals_prior_sd={GLOBALS_PRIOR_SD}.pkl",
                ),
                "wb",
            ) as f:
                pickle.dump(future_predictions, f)

            print(
                f"  Saved: risk_projections_{scenario}_{future_year}_globals_prior_sd={GLOBALS_PRIOR_SD}.pkl  "
                f"({future_predictions.shape[1]} counties)"
            )
