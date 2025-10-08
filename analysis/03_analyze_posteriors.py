import os
import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt
import polars as pl
import yaml
from jax.random import PRNGKey
from matplotlib import rc
from numpyro.infer import Predictive

from outbreak_size import poisson

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
    df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
    )

    with open(
        os.path.join(
            output_dir,
            "samples_partial_pooled_poisson_just_controls.pkl",
        ),
        "rb",
    ) as file:
        posterior_samples = pickle.load(file)

    with open(
        os.path.join(
            output_dir,
            "predictive_partial_pooled_poisson_just_controls.pkl",
        ),
        "rb",
    ) as file:
        predictive_samples = pickle.load(file)

    estimated_cases = predictive_samples["cases"]
    alpha = 0.95
    median = jnp.median(estimated_cases, axis=0)
    lower = jnp.quantile(estimated_cases, (1 - alpha) / 2, axis=0)
    upper = jnp.quantile(estimated_cases, 1 - ((1 - alpha) / 2), axis=0)

    # plot
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
    ax.set_xlabel("Observed cases")
    ax.set_ylabel("Fitted values")
    ax.set_xscale("symlog")
    ax.set_yscale("symlog")
    ax.plot(
        [0, max(df["cases_outbreak"].max(), median.max())],  # type: ignore
        [0, max(df["cases_outbreak"].max(), median.max())],  # type: ignore
        ls="--",
        c="black",
    )

    plt.show()

    # histograms of marginal effects
    iv_effect = (
        posterior_samples["iv_global"][:, jnp.newaxis]
        + posterior_samples["iv_group_variabilities"][:, jnp.newaxis]
        * posterior_samples["iv_group_offsets"]
    )

    df_grouped = df.group_by("state", maintain_order=True)
    cases = df_grouped.agg(pl.max("cases_outbreak"))
    outbreak_ever = cases["cases_outbreak"] >= params["inclusion_threshold"]
    outbreak_never = cases["cases_outbreak"] == 0

    for i in range(iv_effect.shape[-1]):
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
        ax.hist(
            iv_effect[:, outbreak_never.to_jax(), i].flatten(),
            bins=1000,
            density=True,
            color="blue",
            alpha=0.7,
            label="Never had outbreak",
        )

        ax.hist(
            iv_effect[:, outbreak_ever.to_jax(), i].flatten(),
            bins=1000,
            density=True,
            color="red",
            alpha=0.7,
            label="Had an outbreak",
        )

        ax.legend()
        ax.set_xlabel("Log marginal effect")
        ax.set_ylabel("Density")

        plt.show()
