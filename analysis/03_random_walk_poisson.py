import os
import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml
from jax.random import PRNGKey
from matplotlib import rc
from numpyro import enable_x64, set_host_device_count
from sklearn.preprocessing import StandardScaler

enable_x64()
set_host_device_count(4)  # use all cores for MCMC
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median

from outbreak_size import random_walk

plt.rcParams["figure.figsize"] = [9, 9]
font = {"family": "Nimbus Roman", "size": 28}
rc("font", **font)

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
    years = [
        col
        for col in df.columns
        if col.startswith("year_")
        and not col.endswith("_2025")
        and col != "year_every_two_contiguous"
    ]

    assembled_adjacency_matrix = np.genfromtxt(
        os.path.join(data_dir, "county_adjacency_matrix.csv"), delimiter=","
    )

    covariates = df.select(
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
    # standardize covariates
    scaler = StandardScaler()
    covariates = pl.DataFrame(
        scaler.fit_transform(covariates.to_numpy()),
        schema=covariates.columns,
    )
    # set up model
    # number of covariates in each category
    model = random_walk.PartialPooledPoisson(
        time_and_group_varying=covariates.to_jax(),
        state_vary_rw=False,
        group_varying_only=None,
        time_varying_only=None,
        controls=jnp.hstack([df.select(years).to_jax()]),
        population=df["pop"].to_jax(),
        groups=df["state"],
        years=df["year_every_two_contiguous"],
        spatial_units=df["geoid"],
        adjacency_matrix=assembled_adjacency_matrix,
    )

    # fit model
    nuts_kernel = NUTS(
        model.fit,
        init_strategy=init_to_median(),
        target_accept_prob=0.80,
    )
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=2000,
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
            "samples_random_walk_poisson_CAR.pkl",
        ),
        "wb",
    ) as file:
        pickle.dump(posterior_samples, file)

    predictive = Predictive(model.fit, posterior_samples)
    predictive_samples = predictive(PRNGKey(params["seed"]), cases=None)

    # quick plot
    # plot
    median = jnp.median(predictive_samples["cases"], axis=0)
    lower = jnp.quantile(predictive_samples["cases"], 0.025, axis=0)
    upper = jnp.quantile(predictive_samples["cases"], 0.975, axis=0)
    fig, ax = plt.subplots(1, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis="x", direction="out")
    ax.tick_params(axis="y", direction="out")
    # offset the spines
    for spine in ax.spines.values():
        spine.set_position(("outward", 5))
    # put the grid behind
    ax.set_axisbelow(True)
    ax.errorbar(
        df["cases_outbreak"],
        median,
        yerr=[median - lower, upper - median],
        fmt="o",
        ecolor="blue",
        c="blue",
        capsize=0,
        alpha=0.4,
    )
    ax.set_xlabel("Observed outbreak size")
    ax.set_ylabel("Estimated outbreak size")
    ax.set_xscale("symlog")
    ax.set_yscale("symlog")
    ax.plot(
        [0, 1.05 * max(df["cases_outbreak"].max(), median.max())],  # type: ignore
        [0, 1.05 * max(df["cases_outbreak"].max(), median.max())],  # type: ignore
        ls="--",
        c="black",
    )
    ax.set_xlim(-0.3, 1.1 * max(df["cases_outbreak"].max(), median.max()))  # type: ignore
    ax.set_ylim(-0.3, 1.1 * max(df["cases_outbreak"].max(), median.max()))  # type: ignore

    plt.show()

    # print out some metrics
    rmse = jnp.sqrt(jnp.mean((median - df["cases_outbreak"].to_jax()) ** 2))
    mae = jnp.mean(jnp.abs(median - df["cases_outbreak"].to_jax()))
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    # R2
    print(jnp.corrcoef(df["cases_outbreak"].to_jax(), median)[0, 1] ** 2)
    # R2 in log scale
    log_r2 = (
        jnp.corrcoef(
            jnp.log1p(df["cases_outbreak"].to_jax()),
            jnp.log1p(median),
        )[0, 1]
        ** 2
    )
    print(f"Log R2: {log_r2}")

    # save the predictive samples
    with open(
        os.path.join(
            output_dir,
            "predictive_random_walk_poisson_CAR.pkl",
        ),
        "wb",
    ) as file:
        pickle.dump(predictive_samples, file)
