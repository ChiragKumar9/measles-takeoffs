import os
import pickle

import jax.numpy as jnp
import numpy as np
import polars as pl
import yaml
from jax.random import PRNGKey
from numpyro import enable_x64, handlers, set_host_device_count
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

enable_x64()
set_host_device_count(4)  # use all cores for MCMC
from numpyro.infer import MCMC, NUTS, init_to_median

from outbreak_size import adjacency, poisson

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)

# read analysis params yaml
with open("params.yml", "r") as f:
    params = yaml.safe_load(f)


def predict(posterior_samples, model, *args, **kwargs):
    model = handlers.seed(
        handlers.condition(model.fit, posterior_samples),
        PRNGKey(params["seed"]),
    )
    model_trace = handlers.trace(model).get_trace(*args, **kwargs)
    return model_trace["cases"]["value"]


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

    # train on two year periods at a time
    train_years = [[2021, 2022], [2021, 2022, 2023], [2021, 2022, 2023, 2024]]

    for years in train_years:
        print(f"Running rolling validation for years: {years}")
        # filter dataset to only include the training years
        train_df = df.filter(pl.col("year").is_in(years))

        # assemble the adjacency matrix
        adjacency_matrix = adjacency.assemble_adjacency_matrix(
            adjacency_data,
            train_df["fips"].unique(maintain_order=True).to_list(),
        )

        # scale the data
        scaler = StandardScaler()
        vars = [
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

        covariates = train_df.select(vars)
        # scale covariates
        scaler = StandardScaler()
        covariates_scaled = scaler.fit_transform(covariates.to_numpy())
        # append year variables
        # take everything but the first year as the reference year
        year_fixed_effects = [f"year_{y}" for y in years[1:]]
        covariates_scaled = np.hstack(
            [covariates_scaled, train_df.select(year_fixed_effects).to_numpy()]
        )
        covariates = pl.DataFrame(
            covariates_scaled,
            schema=[*covariates.columns, *year_fixed_effects],
        )

        # set up model
        model = poisson.PartialPooledPoisson(
            iv=covariates[:, :3].to_jax(),
            pool=True,
            regression_covariates=covariates[:, 3:].to_jax(),
            exposure=train_df["pop"].to_jax(),
            groups=train_df["state"],
            group_fixed_effect=False,
            spatial=True,
            spatial_units=train_df["geoid"],
            adjacency_matrix=adjacency_matrix,
        )

        # fit model
        nuts_kernel = NUTS(
            model.fit,
            init_strategy=init_to_median(),
            target_accept_prob=0.99,
        )
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=2000,
            num_samples=2000,
            num_chains=4,
        )
        mcmc.run(
            PRNGKey(params["seed"]),
            cases=train_df["cases_outbreak"].to_jax(),
        )

        mcmc.print_summary()

        # save posterior samples
        posterior_samples = mcmc.get_samples()

        with open(
            os.path.join(
                output_dir,
                f"samples_partial_pooled_poisson_validation_training_{years}.pkl",
            ),
            "wb",
        ) as file:
            pickle.dump(posterior_samples, file)

        # test the model
        test_year = years[-1] + 1
        test_df = df.filter(pl.col("year") == test_year)

        # pass the test data through the same scaler
        scaled_test_df = test_df.select(vars)
        test_array = scaler.transform(scaled_test_df.to_numpy())
        scaled_test_df = pl.DataFrame(
            test_array, schema=scaled_test_df.columns
        )

        # add the state, county, fips back to this dataframe
        scaled_test_df = scaled_test_df.with_columns(
            [
                test_df["fips"],
                test_df["state"],
                test_df["county"],
                test_df["year"],
                test_df["pop"],
                test_df["geoid"],
            ]
        )

        # adjust the posterior to not include the data-specific quantities
        # adjust the posterior to not include the county-level effects
        posterior_samples.pop("raw_spatial_effect")
        posterior_samples.pop("spatial_random_effects")
        # adjust the regression betas to not include the year fixed effects
        betas = posterior_samples["regression_betas"]
        # the first 9 columns are safe -- they are the covariates
        # the remaining are the year fixed effects (in this case just one)
        betas = betas[:, :9]
        posterior_samples["regression_betas"] = betas

        # divide the data into those where the states are in the training data
        train_states = train_df["state"].unique().to_list()
        test_in_train = scaled_test_df.filter(
            pl.col("state").is_in(train_states)
        )
        test_not_in_train = scaled_test_df.filter(
            ~pl.col("state").is_in(train_states)
        )
        # order the test data with states in the training data the same as the training data
        test_in_train = test_in_train.with_columns(
            pl.col("state").cast(pl.Enum(train_states))
        ).sort("state")

        # pop out the indeces of the states that are only in the training data
        # from the posterior samples so we don't misallocate those state signals
        # to other states
        test_in_train_states = test_in_train["state"].unique().to_list()
        buffer = 0
        indexes_to_pop = []
        for idx, state in enumerate(test_in_train_states):
            if train_states[idx + buffer] != state:
                buffer += 1
                indexes_to_pop.append(idx)
        if len(indexes_to_pop) > 0:
            posterior_samples["iv_group_offsets"] = jnp.delete(
                posterior_samples["iv_group_offsets"],
                jnp.array(indexes_to_pop),
                axis=1,
            )

        # make a new adjacency matrix for the test data
        adjacency_matrix_in_train = adjacency.assemble_adjacency_matrix(
            adjacency_data,
            test_in_train["fips"].unique(maintain_order=True).to_list(),
        )

        adjacency_matrix_not_in_train = adjacency.assemble_adjacency_matrix(
            adjacency_data,
            test_not_in_train["fips"].unique(maintain_order=True).to_list(),
        )

        # set up a model instance with the new data
        model = poisson.PartialPooledPoisson(
            iv=test_in_train.select(
                ["estimated_mmr_cov", "nonmed_exempt_pct", "med_exempt_pct"]
            ).to_jax(),
            pool=True,
            regression_covariates=test_in_train.select(
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
            exposure=test_in_train["pop"].to_jax(),
            groups=test_in_train["state"],
            group_fixed_effect=False,
            spatial=True,
            spatial_units=test_in_train["geoid"],
            adjacency_matrix=adjacency_matrix_in_train,
        )

        # iterate through posterior samples and generate predictions
        num_samples = posterior_samples["regression_betas"].shape[0]
        for sample in tqdm(range(num_samples), desc="Generating predictions"):
            # extract a temp posterior sample dict
            posterior_sample = {
                k: v[sample] for k, v in posterior_samples.items()
            }
            predictions = predict(posterior_sample, model)
            if sample == 0:
                all_predictions = predictions[None, :]
            else:
                all_predictions = jnp.vstack(
                    [all_predictions, predictions[None, :]]  # type: ignore
                )

        # set up a model instance with the new data
        posterior_samples.pop("iv_group_offsets")
        model = poisson.PartialPooledPoisson(
            iv=test_not_in_train.select(
                ["estimated_mmr_cov", "nonmed_exempt_pct", "med_exempt_pct"]
            ).to_jax(),
            pool=True,
            regression_covariates=test_not_in_train.select(
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
            exposure=test_not_in_train["pop"].to_jax(),
            groups=test_not_in_train["state"],
            group_fixed_effect=False,
            spatial=True,
            spatial_units=test_not_in_train["geoid"],
            adjacency_matrix=adjacency_matrix_not_in_train,
        )

        # iterate through posterior samples and generate predictions
        num_samples = posterior_samples["regression_betas"].shape[0]
        for sample in tqdm(range(num_samples), desc="Generating predictions"):
            # extract a temp posterior sample dict
            posterior_sample = {
                k: v[sample] for k, v in posterior_samples.items()
            }
            predictions = predict(posterior_sample, model)
            if sample == 0:
                new_state_predictions = predictions[None, :]
            else:
                new_state_predictions = jnp.vstack(
                    [new_state_predictions, predictions[None, :]]  # type: ignore
                )

        # concatenate the predictions back together in the original test_df order
        all_predictions = jnp.hstack([all_predictions, new_state_predictions])  # type: ignore

        with open(
            os.path.join(
                output_dir,
                f"samples_partial_pooled_poisson_validation_testing_{test_year}.pkl",
            ),
            "wb",
        ) as file:
            pickle.dump(all_predictions, file)  # type: ignore
