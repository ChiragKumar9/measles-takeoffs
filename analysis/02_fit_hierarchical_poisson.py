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
        os.path.join(data_dir, "outbreaks_vaccinations_cleaned.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
    )

    # make log of reasonably log scaled covariates
    df = df.with_columns(
        log_population=pl.col("pop_5_9").log(),
        log_income=pl.col("median_household_income").log(),
    )

    # make year dummy variables
    df = df.sort("year", descending=True)
    year_data = df["year"]
    df = df.to_dummies(columns="year", separator="_", drop_first=True)
    years = [y for y in df.columns if y.startswith("year_")]
    df = df.with_columns(year=year_data)

    # make new variables
    df = df.with_columns(
        fraction_nonmed_exempt=pl.col("nonmed_exempt_pct")
        / (pl.col("nonmed_exempt_pct") + pl.col("med_exempt_pct")),
    )
    df = df.with_columns(
        fraction_nonmed_exempt=pl.col("fraction_nonmed_exempt").fill_nan(0.0)
    )

    # set up model
    iv = df[
        ["estimated_mmr_cov", "total_exempt_pct", "fraction_nonmed_exempt"]
    ].to_jax()
    model = poisson.PartialPooledPoisson(
        iv=iv,
        regression_covariates=df.select(
            [
                "fraction_public",
                "pop_density_5_9",
                "fraction_white",
                "fraction_graduate_degree",
                "log_income",
                *years,
            ]
        ).to_jax(),
        exposure=df["pop_5_9"].to_jax(),
        groups=df["state"],
        group_fixed_effect=True,
    )

    # fit model
    nuts_kernel = NUTS(model.fit, init_strategy=init_to_median())
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=8000,
        num_samples=4000,
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

    predictive = Predictive(model.fit, posterior_samples)
    predictive_samples = predictive(PRNGKey(params["seed"]), cases=None)

    # save the predictive samples
    with open(
        os.path.join(
            output_dir,
            "predictive_partial_pooled_poisson_just_controls.pkl",
        ),
        "wb",
    ) as file:
        pickle.dump(predictive_samples, file)

    # save the analysis dataset
    df.write_csv(
        os.path.join(
            data_dir,
            "analysis_dataset.csv",
        )
    )
