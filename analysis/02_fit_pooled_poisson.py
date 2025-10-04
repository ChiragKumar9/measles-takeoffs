import os
import pickle

import polars as pl
import yaml
from jax.random import PRNGKey
from numpyro import enable_x64, set_host_device_count
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median

from outbreak_size import poisson

enable_x64()
set_host_device_count(4)  # use all cores for MCMC

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
        os.path.join(data_dir, "outbreaks_vaccinations_cleaned.csv")
    )

    # make log of reasonably log scaled covariates
    df = df.with_columns(
        log_population=pl.col("pop_5_9").log(),
        log_income=pl.col("median_household_income").log(),
    )

    # make year dummy variables
    df = df.sort("year", descending=True)
    df = df.to_dummies(columns="year", separator="_", drop_first=True)
    years = [y for y in df.columns if y.startswith("year_")]

    # set up model
    model = poisson.PartialPooledPoisson(
        iv=df["estimated_mmr_cov"].to_jax(),
        covariates=df.select(
            [
                "fraction_public",
                "pop_density_5_9",
                "fraction_white",
                "fraction_graduate_degree",
                "log_population",
                "log_income",
                *years,
            ]
        ).to_jax(),
        counties=df["county"],
    )

    # fit model
    nuts_kernel = NUTS(model.fit, init_strategy=init_to_median())
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=params["mcmc_warmup"],
        num_samples=params["mcmc_samples"],
        num_chains=4,
    )
    mcmc.run(PRNGKey(params["seed"]), cases=df["cases_outbreak"].to_jax())

    mcmc.print_summary()

    # save posterior samples
    posterior_samples = mcmc.get_samples()

    with open(
        os.path.join(
            output_dir,
            "samples_partial_pooled_poisson_just_controls.pkl",
        ),
        "wb",
    ) as file:
        pickle.dump(posterior_samples, file)
