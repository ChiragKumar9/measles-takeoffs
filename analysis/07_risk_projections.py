from numpyro import enable_x64, set_host_device_count

enable_x64()
set_host_device_count(4)  # use all cores for MCMC

import os
import pickle

import numpy as np
import numpyro.distributions as dist
import polars as pl
import yaml
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


if __name__ == "__main__":
    # read the projection dataset
    df = pl.read_csv(
        os.path.join(output_dir, "scaled_projection_dataset.csv"),
        schema_overrides={"fips": pl.String},
    )

    adjacency = np.loadtxt(
        os.path.join(output_dir, "projection_adjacency_matrix.npy")
    )

    # set up a model
    years_present_train = (
        df.filter(pl.col("in_train") == 1)["year"].unique().to_list()
    )
    # pseudo set the year for the test data to be the next year after the last training year
    df = df.with_columns(
        pl.when(pl.col("in_train") == 0)
        .then(pl.max("year") + 1)
        .otherwise(pl.col("year"))
    )
    # make fixed effects for each of the years present in train
    df = df.with_columns(
        [
            pl.when(pl.col("year") == year)
            .then(1)
            .otherwise(0)
            .alias(f"year_{year}")
            for year in years_present_train[1:]
        ]
    )
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
        spatial_units=df["geoid"],
        adjacency_matrix=adjacency,
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

    train_samples = predictive_samples["cases"][
        :, (df["in_train"] == 1).to_jax()
    ]

    # pickle it
    with open(
        os.path.join(
            output_dir,
            "risk_projections.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(all_predictions, f)

    # pickle it
    with open(
        os.path.join(
            output_dir,
            "risk_projections_train.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(train_samples, f)
