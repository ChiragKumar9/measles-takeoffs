import os
import pickle

import polars as pl
import yaml
from jax.random import PRNGKey
from numpyro import enable_x64, set_host_device_count
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median

# We have to do this above importing our model to avoid issues with Jax for
# reasons I don't quite understand?
set_host_device_count(4)  # use all cores for MCMC
enable_x64()

from outbreak_size import variable_coverage

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
    years = [y for y in df.columns if y.startswith("year_")]

    # set up model
    model = variable_coverage.WithinCountyVariance(
        coverage_fraction=df["estimated_mmr_cov"].to_jax() / 100,
        population=df["pop_5_9"].to_jax(),
        variance_covariates=df.select(
            [
                "estimated_mmr_cov",
                "total_exempt_pct",
                "fraction_nonmed_exempt",
                "log_population",
                "fraction_public",
                "fraction_white",
                "fraction_graduate_degree",
                "log_income",
                *years,
            ]
        ).to_jax(),
        groups=df["state"],
        group_fixed_effect=True,
    )

    # fit model
    nuts_kernel = NUTS(
        model.fit, init_strategy=init_to_median(), dense_mass=True
    )
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=2000,
        num_samples=1000,
        num_chains=1,
    )
    mcmc.run(PRNGKey(params["seed"]), cases=df["cases_outbreak"].to_jax())

    mcmc.print_summary()

    # save posterior samples
    posterior_samples = mcmc.get_samples()

    with open(
        os.path.join(
            output_dir,
            "samples_within_county_coverage_varies.pkl",
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
            "predictive_within_county_coverage_varies.pkl",
        ),
        "wb",
    ) as file:
        pickle.dump(predictive_samples, file)
