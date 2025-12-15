from numpyro import enable_x64

enable_x64()

import os
import pickle

import jax.numpy as jnp
import numpy as np
import polars as pl
import yaml
from jax.random import PRNGKey
from numpyro import handlers
from tqdm.auto import tqdm

from outbreak_size import poisson

# read analysis params yaml
with open("params.yml", "r") as f:
    params = yaml.safe_load(f)

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)


def predict(posterior_samples, model, *args, **kwargs):
    model = handlers.seed(
        handlers.condition(model.fit, posterior_samples),
        PRNGKey(params["seed"]),
    )
    model_trace = handlers.trace(model).get_trace(*args, **kwargs)
    return model_trace["cases"]["value"]


if __name__ == "__main__":
    # load posterior
    with open(
        os.path.join(
            output_dir,
            "samples_partial_pooled_poisson_state_fixed_effect=False_pool=True_five_year=False_CAR=True_All covariates_[2021, 2022, 2023, 2024, 2025].pkl",
        ),
        "rb",
    ) as f:
        posterior = pickle.load(f)

    # load dataset
    df = pl.read_csv(
        os.path.join(output_dir, "scaled_projection_dataset.csv"),
        schema_overrides={"fips": pl.String},
    )
    df_states_in_train = df.filter(pl.col("in_train") == True)
    df_states_not_in_train = df.filter(pl.col("in_train") == False)

    adjacency_in_train = np.loadtxt(
        os.path.join(output_dir, "projection_adjacency_matrix_in_train.npy")
    )
    adjacency_not_in_train = np.loadtxt(
        os.path.join(
            output_dir, "projection_adjacency_matrix_not_in_train.npy"
        )
    )

    # adjust the posterior to not include the county-level effects
    posterior.pop("raw_spatial_effect")
    posterior.pop("spatial_random_effects")
    # adjust the regression betas to not include the year fixed effects
    betas = posterior["regression_betas"]
    # the first 9 columns are safe -- they are the covariates
    betas = betas[:, :9]
    posterior["regression_betas"] = betas

    # set up model
    model = poisson.PartialPooledPoisson(
        iv=df_states_in_train[
            ["estimated_mmr_cov", "nonmed_exempt_pct", "med_exempt_pct"]
        ].to_jax(),
        pool=True,
        regression_covariates=df_states_in_train[
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
        ].to_jax(),
        exposure=df_states_in_train["pop"].to_jax(),
        groups=df_states_in_train["state"],
        spatial_units=df_states_in_train["fips"],
        adjacency_matrix=adjacency_in_train,
        group_fixed_effect=False,
    )

    # iterate through posterior samples and generate predictions
    num_samples = posterior["regression_betas"].shape[0]
    for sample in tqdm(range(num_samples), desc="Generating predictions"):
        # extract a temp posterior sample dict
        posterior_sample = {k: v[sample] for k, v in posterior.items()}
        predictions = predict(posterior_sample, model)
        if sample == 0:
            all_predictions = predictions[None, :]
        else:
            all_predictions = jnp.vstack(
                [all_predictions, predictions[None, :]]
            )  # type: ignore

    # now do the states not in train
    posterior.pop("iv_group_offsets")
    # set up model
    model = poisson.PartialPooledPoisson(
        iv=df_states_not_in_train[
            ["estimated_mmr_cov", "nonmed_exempt_pct", "med_exempt_pct"]
        ].to_jax(),
        pool=True,
        regression_covariates=df_states_not_in_train[
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
        ].to_jax(),
        exposure=df_states_not_in_train["pop"].to_jax(),
        groups=df_states_not_in_train["state"],
        spatial_units=df_states_not_in_train["fips"],
        adjacency_matrix=adjacency_not_in_train,
        group_fixed_effect=False,
    )

    # iterate through posterior samples and generate predictions
    num_samples = posterior["regression_betas"].shape[0]
    for sample in tqdm(range(num_samples), desc="Generating predictions"):
        # extract a temp posterior sample dict
        posterior_sample = {k: v[sample] for k, v in posterior.items()}
        predictions = predict(posterior_sample, model)
        if sample == 0:
            new_state_predictions = predictions[None, :]
        else:
            new_state_predictions = jnp.vstack(
                [new_state_predictions, predictions[None, :]]
            )  # type: ignore

    # concatenate the predictions back together in the original test_df order
    all_predictions = jnp.hstack([all_predictions, new_state_predictions])  # type: ignore

    # pickle it
    with open(
        os.path.join(
            output_dir,
            "risk_projections.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(all_predictions, f)  # type: ignore
