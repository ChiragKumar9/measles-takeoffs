import os
import pickle

import numpy as np
import polars as pl
import yaml
from jax.random import PRNGKey
from numpyro import enable_x64, set_host_device_count
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median

# Not sure why this has to be set up above importation of model code?
set_host_device_count(4)
enable_x64()

from outbreak_size import heterogeneous_coverage

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

    adjacency_matrix = pl.read_csv(
        os.path.join(data_dir, "county_adjacencies.txt"),
        separator="|",
        schema_overrides={
            "County GEOID": pl.String,
            "Neighbor GEOID": pl.String,
        },
    )
    # only include counties in our dataset
    adjacency_matrix = adjacency_matrix.filter(
        pl.col("County GEOID").is_in(df["fips"])
    )
    # convert to adjacency matrix
    counties = df["fips"].unique().to_list()
    num_counties = len(counties)
    adj_matrix = np.zeros((num_counties, num_counties))
    for idx, fip in enumerate(counties):
        # check that county has some neighbors
        neighbors = adjacency_matrix.filter(pl.col("County GEOID") == fip)
        if neighbors.height != 0:
            for row in neighbors.iter_rows(named=True):
                # is this county in our dataset?
                if row["Neighbor GEOID"] in counties:
                    neighbor_idx = counties.index(row["Neighbor GEOID"])
                    adj_matrix[idx, neighbor_idx] = 1

    # set up model
    model = heterogeneous_coverage.WithinCountyVarianceRegression(
        coverage_fraction=df["estimated_mmr_cov"].to_jax() / 100,
        exemption_fraction=df["total_exempt_pct"].to_jax() / 100,
        fraction_nonmed_exempt=df["fraction_nonmed_exempt"].to_jax(),
        population=df["pop_5_9"].to_jax(),
        regression_controls=df.select(
            [
                "pop_density_5_9",
                "fraction_public",
                "fraction_white",
                "fraction_graduate_degree",
                "log_income",
                *years,
            ]
        ).to_jax(),
        groups=df["state"],
        spatial_units=df["fips"],
        adjacency_matrix=adj_matrix,
    )

    # fit model
    nuts_kernel = NUTS(
        model.fit,
        init_strategy=init_to_median(num_samples=50),
        target_accept_prob=0.8,
    )
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=8000,
        num_samples=2000,
        num_chains=4,
    )
    mcmc.run(PRNGKey(params["seed"]), cases=df["cases_outbreak"].to_jax())

    mcmc.print_summary()

    # save posterior samples
    posterior_samples = mcmc.get_samples()

    with open(
        os.path.join(
            output_dir,
            "samples_within_county_coverage_varies_CAR.pkl",
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
            "predictive_within_county_coverage_varies_CAR.pkl",
        ),
        "wb",
    ) as file:
        pickle.dump(predictive_samples, file)
