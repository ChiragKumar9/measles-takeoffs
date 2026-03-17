import os
import pickle

import jax.numpy as jnp
import numpy as np
import polars as pl
import statsmodels.api as sm
import yaml
from jax import clear_caches
from jax.random import PRNGKey
from numpyro import distributions as dist
from numpyro import enable_x64, set_host_device_count
from scipy import stats
from sklearn.preprocessing import StandardScaler

enable_x64()
set_host_device_count(4)  # use all cores for MCMC
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median

from outbreak_size import adjacency, final_outbreak_size, poisson

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)

# read analysis params yaml
with open("params.yml", "r") as f:
    params = yaml.safe_load(f)

if __name__ == "__main__":
    # Load dataset
    df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
    )

    # spatial component
    adjacency_data = pl.read_csv(
        os.path.join(data_dir, "county_adjacencies.txt"),
        separator="|",
        schema_overrides={
            "County GEOID": pl.String,
            "Neighbor GEOID": pl.String,
        },
    )
    models = {
        "MMR coverage": ["estimated_mmr_cov"],
        "MMR coverage + exemptions": [
            "estimated_mmr_cov",
            "nonmed_exempt_pct",
            "med_exempt_pct",
        ],
        "All covariates": [
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
    }
    # test the mechanistic model (separate)
    # test the three variable variations, with and without state fixed effects,
    # with and without the car term, with and without pooling
    for car in [True, False]:
        for pool in [True, False]:
            for model_name, vars in models.items():
                print(
                    f"Running rolling validation for model: {model_name}, "
                    f"CAR: {car}, Pooling: {pool}"
                )

                # train on progressing windows of time
                train_years = [
                    [
                        2011,
                        2012,
                        2013,
                        2014,
                        2015,
                        2016,
                        2017,
                        2018,
                        2019,
                        2020,
                    ],
                    [
                        2011,
                        2012,
                        2013,
                        2014,
                        2015,
                        2016,
                        2017,
                        2018,
                        2019,
                        2020,
                        2021,
                    ],
                    [
                        2011,
                        2012,
                        2013,
                        2014,
                        2015,
                        2016,
                        2017,
                        2018,
                        2019,
                        2020,
                        2021,
                        2022,
                    ],
                    [
                        2011,
                        2012,
                        2013,
                        2014,
                        2015,
                        2016,
                        2017,
                        2018,
                        2019,
                        2020,
                        2021,
                        2022,
                        2023,
                    ],
                    [
                        2011,
                        2012,
                        2013,
                        2014,
                        2015,
                        2016,
                        2017,
                        2018,
                        2019,
                        2020,
                        2021,
                        2022,
                        2023,
                        2024,
                    ],
                    [2019, 2020],
                    [2019, 2020, 2021],
                    [2019, 2020, 2021, 2022],
                    [2019, 2020, 2021, 2022, 2023],
                    [2019, 2020, 2021, 2022, 2023, 2024],
                    [2020, 2021],
                    [2021, 2022],
                    [2022, 2023],
                    [2023, 2024],
                ]

                for years in train_years:
                    first_through_years = False
                    print(f"Running rolling validation for years: {years}")
                    # filter dataset to only include the training years
                    train_df = df.filter(pl.col("year").is_in(years))
                    # denote as training
                    train_df = train_df.with_columns(
                        type=pl.lit("train"), cases_train=pl.lit(1)
                    )

                    # grab the test data as the entire projection dataset
                    test_df = pl.read_csv(
                        os.path.join(
                            output_dir,
                            f"projection_dataset_{years[-1] + 1}.csv",
                        ),
                        schema_overrides={"fips": pl.String},
                    )

                    # label this data with type and cases_train
                    test_df = test_df.with_columns(
                        type=pl.lit("projection"),
                        cases_train=pl.lit(0),
                    )

                    # concatenate the train and test data
                    train_df = pl.concat(
                        [train_df, test_df], how="diagonal_relaxed"
                    )
                    # any null cases_outbreak must be filled as 0s as a placeholder
                    train_df = train_df.with_columns(
                        cases_outbreak=pl.col("cases_outbreak").fill_null(0)
                    )

                    # assemble the adjacency matrix
                    adjacency_matrix = adjacency.assemble_adjacency_matrix(
                        adjacency_data,
                        train_df["fips"].unique(maintain_order=True).to_list(),
                    )

                    # scale the data
                    scaler = StandardScaler()

                    covariates = train_df.filter(
                        pl.col("cases_train") == 1
                    ).select(vars)
                    # scale covariates
                    scaler = StandardScaler()
                    covariates_scaled = scaler.fit_transform(
                        covariates.to_numpy()
                    )
                    # add the test data
                    covariates_scaled = np.vstack(
                        [
                            covariates_scaled,
                            scaler.transform(
                                train_df.filter(pl.col("cases_train") == 0)
                                .select(vars)
                                .to_numpy()
                            ),
                        ]
                    )
                    # append year variables
                    covariates_scaled = np.hstack(
                        [
                            covariates_scaled,
                        ]
                    )
                    covariates = pl.DataFrame(
                        covariates_scaled,
                        schema=[*covariates.columns],
                    )
                    # set up model
                    if model_name != "All covariates":
                        ivs = 1
                        # no regression covariates
                        # use zeros for the regression covariates since the model will ignore them
                        regression_covariates = jnp.zeros(
                            (covariates.shape[0], 1)
                        )
                    else:
                        ivs = 3
                        regression_covariates = jnp.array(
                            covariates[:, 3:].to_numpy()
                        )
                    model = poisson.PartialPooledPoisson(
                        iv=covariates[:, :ivs].to_jax(),
                        pool=pool,
                        regression_covariates=regression_covariates,
                        exposure=train_df["pop"].to_jax(),
                        groups=train_df["state"],
                        group_fixed_effect=False,
                        spatial=car,
                        spatial_units=train_df["geoid"],
                        adjacency_matrix=adjacency_matrix,
                        spatial_sigma=dist.HalfNormal(0.25),
                        car_sigma=dist.HalfNormal(0.25),
                    )

                    # fit model
                    nuts_kernel = NUTS(
                        model.fit,
                        init_strategy=init_to_median(),
                        target_accept_prob=0.95,
                    )
                    mcmc = MCMC(
                        nuts_kernel,
                        num_warmup=2000,
                        num_samples=2000,
                        num_chains=4,
                    )
                    mcmc.run(
                        PRNGKey(params["seed"]),
                        is_observed=train_df["cases_train"].to_jax(),
                        cases=train_df["cases_outbreak"].to_jax(),
                    )

                    mcmc.print_summary()

                    # save posterior samples
                    posterior_samples = mcmc.get_samples()

                    with open(
                        os.path.join(
                            output_dir,
                            f"samples_partial_pooled_poisson_validation_training_{years}_pool={pool}_CAR={car}_{model_name}.pkl",
                        ),
                        "wb",
                    ) as file:
                        pickle.dump(posterior_samples, file)

                    # pass the posterior samples to the model to generate predictions
                    predictive = Predictive(model.fit, posterior_samples)

                    predictive_samples = predictive(
                        PRNGKey(params["seed"]),
                        is_observed=train_df["cases_train"].to_jax(),
                        cases=None,
                    )

                    # extract just the cases and take only those for the test year
                    all_predictions = predictive_samples["cases"][
                        :, (train_df["cases_train"] == 0).to_jax()
                    ]

                    with open(
                        os.path.join(
                            output_dir,
                            f"samples_partial_pooled_poisson_validation_testing_{years}_pool={pool}_CAR={car}_{model_name}.pkl",
                        ),
                        "wb",
                    ) as file:
                        pickle.dump(all_predictions, file)  # type: ignore

                    # we also need to test the mechanistic model
                    # we only need to do this for our first time through the years
                    if not first_through_years:
                        first_through_years = True
                        # compute final outbreak size for each row
                        population = train_df.filter(
                            pl.col("cases_train") == 1
                        )["pop"].to_numpy()
                        # calculate immune fraction based on coverage and effectiveness
                        mmr_effectiveness = 0.97
                        immune = (
                            train_df.filter(pl.col("cases_train") == 1)[
                                "estimated_mmr_cov"
                            ].to_numpy()
                            * mmr_effectiveness
                            / 100
                        )  # type: ignore
                        susceptible_fraction = 1 - immune
                        # calculate final outbreak size
                        R0 = 12.0  # measles basic reproduction number
                        # final outbreak size calculation
                        final_sizes = np.array(
                            [
                                final_outbreak_size.final_outbreak_size(R0, s0)
                                * N
                                for s0, N in zip(
                                    susceptible_fraction, population
                                )
                            ]
                        )

                        # now scale these values to match the scale of the observed outbreaks
                        observed_outbreaks = train_df.filter(
                            pl.col("cases_train") == 1
                        )["cases_outbreak"].to_numpy()
                        # fit a poisson regression
                        X = final_sizes.reshape(-1, 1)
                        model = sm.GLM(
                            observed_outbreaks,
                            X,
                            family=sm.families.Poisson(),
                        ).fit()

                        # get the predictions
                        # now apply this model to the test data
                        population_test = train_df.filter(
                            pl.col("cases_train") == 0
                        )["pop"].to_numpy()
                        susceptible_fraction_test = (
                            1.0
                            - train_df.filter(pl.col("cases_train") == 0)[
                                "estimated_mmr_cov"
                            ].to_numpy()
                            * mmr_effectiveness
                            / 100
                        )
                        final_sizes_test = np.array(
                            [
                                final_outbreak_size.final_outbreak_size(R0, s0)
                                * N
                                for s0, N in zip(
                                    susceptible_fraction_test, population_test
                                )
                            ]
                        )
                        X_test = final_sizes_test.reshape(-1, 1)
                        mu = model.predict(X_test)
                        # truncate mus at the population size
                        mu = np.minimum(mu, population_test)

                        # extract median and 95th percentile interval
                        samples = stats.poisson.rvs(
                            mu=mu, size=(8000, mu.shape[0])
                        )

                        # pickle these three
                        with open(
                            os.path.join(
                                output_dir,
                                f"predictive_final_outbreak_size_predictions_{years}.pkl",
                            ),
                            "wb",
                        ) as f:
                            pickle.dump(
                                {
                                    "cases": samples,
                                },
                                f,
                            )

                    del (
                        model,
                        mcmc,
                        nuts_kernel,
                        posterior_samples,
                        predictive,
                        predictive_samples,
                        all_predictions,
                        final_sizes,  # type: ignore
                        final_sizes_test,  # type: ignore
                        mu,  # type: ignore
                        samples,  # type: ignore
                    )
                    clear_caches()
