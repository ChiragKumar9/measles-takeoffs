import os
import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml
from matplotlib import rc

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)
fig_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")

font = {"family": "Nimbus Roman", "size": 28}
rc("font", **font)

# read analysis params yaml
with open("params.yml", "r") as f:
    params = yaml.safe_load(f)


def setup_plot(nrows, ncols):
    fig, ax = plt.subplots(nrows, ncols, figsize=(20, 15))
    if nrows * ncols == 1:
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
        return ax
    for temp in ax.flatten():  # type: ignore
        temp.spines["top"].set_visible(False)
        temp.spines["right"].set_visible(False)
        temp.get_xaxis().tick_bottom()
        temp.get_yaxis().tick_left()
        temp.tick_params(axis="x", direction="out")
        temp.tick_params(axis="y", direction="out")
        # offset the spines
        for spine in temp.spines.values():
            spine.set_position(("outward", 5))
        # put the grid behind
        temp.set_axisbelow(True)
    return fig, ax


def plot_vaccine_effect_coverage(df, fig, ax1, ax2, ax3):
    std = df["estimated_mmr_cov"].to_numpy().std()

    # extract state-level vaccination effects
    with open(
        os.path.join(
            output_dir,
            "samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_[2021, 2022, 2023, 2024, 2025].pkl",
        ),
        "rb",
    ) as file:
        posterior_samples = pickle.load(file)

    cov_effects = (
        posterior_samples["iv_global"][:, 0][:, None]
        + posterior_samples["iv_group_variabilities"][:, 0][:, None]
        * posterior_samples["iv_group_offsets"][:, :, 0]
    )

    cov_effects = (jnp.exp(cov_effects / std) * 100) - 100

    cov_effects_median = jnp.median(cov_effects, axis=0)
    cov_effects_lower = jnp.quantile(cov_effects, 0.025, axis=0)
    cov_effects_upper = jnp.quantile(cov_effects, 0.975, axis=0)

    # extract state-level vaccine coverage
    df = df.filter(pl.col("year").is_in([2021, 2022, 2023, 2024, 2025]))
    df = df.with_columns(mu=posterior_samples["mu"].__array__().T)
    df = df.group_by("state", maintain_order=True).agg(
        mean_mmr_cov=pl.mean("estimated_mmr_cov"),
        median_mmr_cov=pl.median("estimated_mmr_cov"),
        outbreak_weighted_mean_mmr_cov=(
            pl.col("estimated_mmr_cov")
            * (pl.col("cases_outbreak") + pl.lit(1))
            / (pl.col("cases_outbreak") + pl.lit(1)).sum()
        ).sum(),
        mean_outbreak_weighted_mean_mmr_cov=(
            pl.col("estimated_mmr_cov") * (pl.col("mu")) / (pl.col("mu")).sum()
        ).sum(),
        num_points=pl.len(),
    )

    ax1.errorbar(
        df["outbreak_weighted_mean_mmr_cov"],
        cov_effects_median,
        [
            cov_effects_median - cov_effects_lower,
            cov_effects_upper - cov_effects_median,
        ],
        fmt="o",
        ecolor="k",
        c="k",
        alpha=0.7,
        capsize=0,
    )

    ax1.axhline(y=0, color="k", linestyle="--")

    ax1.set_xlabel("Average MMR coverage (%)")
    ax1.set_ylabel(
        "% change in outbreak size per 1%\nincrease in MMR vaccine coverage"
    )
    ax1.set_xlim(75, 101)

    # ax2 gets the credible interval width and number of occurences
    ax2.scatter(
        df["num_points"],
        cov_effects_upper - cov_effects_lower,
        c="k",
        alpha=0.7,
    )
    r2 = (
        np.corrcoef(
            (df["num_points"].to_numpy()),
            np.array((cov_effects_upper - cov_effects_lower)),
        )[0, 1]
        ** 2
    )
    ax2.text(
        0.7,
        0.9,
        f"$R^2$ = {r2:.2f}",
        transform=ax2.transAxes,
        fontsize=32,
        verticalalignment="top",
    )
    ax2.set_xlabel("Number of county-years in the state")
    ax2.set_ylabel(
        "95% credible interval width of\n% change in outbreak size per 1%\nincrease in MMR vaccine coverage"
    )

    ax3.scatter(
        (df["outbreak_weighted_mean_mmr_cov"].to_jax()),
        cov_effects_upper - cov_effects_lower,
        c="k",
        alpha=0.7,
    )
    # put a line of best fit through the data to highlight the trend
    z = jnp.polyfit(
        (df["outbreak_weighted_mean_mmr_cov"].to_jax()),
        (cov_effects_upper - cov_effects_lower),
        1,
    )
    ax3.plot(
        (df["outbreak_weighted_mean_mmr_cov"].to_jax()),
        jnp.polyval(
            z,  # type: ignore
            (df["outbreak_weighted_mean_mmr_cov"].to_jax()),
        ),
        color="red",
    )
    # get R2 and print out
    r2 = (
        jnp.corrcoef(
            (df["outbreak_weighted_mean_mmr_cov"].to_jax()),
            (cov_effects_upper - cov_effects_lower),
        )[0, 1]
        ** 2
    )
    ax3.text(
        0.7,
        0.15,
        f"$R^2$ = {r2:.2f}",
        transform=ax3.transAxes,
        fontsize=32,
        verticalalignment="top",
    )
    ax3.set_xlabel("Average MMR coverage (%)")
    ax3.set_ylabel(
        "95% credible interval width of\n% change in outbreak size per 1%\nincrease in MMR vaccine coverage"
    )
    ax3.set_xlim(75, 101)

    return fig, ax1, ax2, ax3


def plot_num_points_average_coverage(df, fig, ax):
    df = df.filter(pl.col("year").is_in([2021, 2022, 2023, 2024, 2025]))
    df = df.group_by("state", maintain_order=True).agg(
        mean_mmr_cov=pl.mean("estimated_mmr_cov"),
        num_points=pl.len(),
    )

    ax.scatter(df["num_points"], df["mean_mmr_cov"], c="k", alpha=0.7)

    # # put a line of best fit through the data to highlight the trend
    # z = np.polyfit(
    #     df["num_points"].to_numpy(), df["mean_mmr_cov"].to_numpy(), 1
    # )
    # ax.plot(
    #     df["num_points"].to_numpy(),
    #     np.polyval(
    #         z,  # type: ignore
    #         df["num_points"].to_numpy(),
    #     ),
    #     color="red",
    # )
    # get R2 and print out
    r2 = (
        np.corrcoef(
            df["num_points"].to_numpy(), df["mean_mmr_cov"].to_numpy()
        )[0, 1]
        ** 2
    )
    ax.text(
        0.7,
        0.15,
        f"$R^2$ = {r2:.2f}",
        transform=ax.transAxes,
        fontsize=32,
        verticalalignment="top",
    )
    ax.set_xlabel("Number of county-years in the state")
    ax.set_ylabel("Average MMR coverage (%)")

    return fig, ax


if __name__ == "__main__":
    df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
    )

    fig, ax = setup_plot(2, 2)  # type: ignore

    # why we need to pool
    ax[0, 0].text(  # type: ignore
        -0.25,
        1.08,
        "a",
        transform=ax[0, 0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    ax[0, 1].text(  # type: ignore
        -0.25,
        1.08,
        "b",
        transform=ax[0, 1].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    ax[1, 0].text(  # type: ignore
        -0.25,
        1.08,
        "c",
        transform=ax[1, 0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    ax[1, 1].text(  # type: ignore
        -0.25,
        1.08,
        "d",
        transform=ax[1, 1].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_vaccine_effect_coverage(
        df,
        fig,
        ax[0, 0],  # type: ignore
        ax[0, 1],  # type: ignore
        ax[1, 0],  # type: ignore
    )

    plot_num_points_average_coverage(df, fig, ax[1, 1])  # type: ignore

    fig.tight_layout()

    # save
    fig.savefig(
        os.path.join(
            fig_dir, "figure_supplemental_state_varying_coefficients.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(
            fig_dir, "figure_supplemental_state_varying_coefficients.pdf"
        ),
        dpi=300,
        bbox_inches="tight",
    )
