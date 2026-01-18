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
        return fig, ax
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


def plot_immunity_effects_on_vaccine_coverage(df, fig, ax):
    df = df.sort("year")

    df = df.with_columns(
        previous_outbreak_size=pl.cum_sum("cases_outbreak").over(
            ["county", "state"]
        )
    ).with_columns(
        # want to remove the current year's outbreak size
        previous_outbreak_size=pl.col("previous_outbreak_size")
        - pl.col("cases_outbreak")
    )

    df = df.filter(pl.col("year") >= 2021)

    medians = []
    lowers = []
    uppers = []

    with open(
        os.path.join(
            output_dir,
            f"samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_[2021, 2022, 2023, 2024, 2025].pkl",
        ),
        "rb",
    ) as file:
        posterior_samples = pickle.load(file)
    samples_of_interest = posterior_samples["iv_global"][:, 0]
    samples_of_interest = (
        samples_of_interest / df["estimated_mmr_cov"].to_numpy().std()
    )
    # make a percent change
    samples_of_interest = jnp.exp(samples_of_interest) * 100 - 100
    medians.append(jnp.median(samples_of_interest))
    lowers.append(jnp.percentile(samples_of_interest, 2.5))
    uppers.append(jnp.percentile(samples_of_interest, 97.5))

    with open(
        os.path.join(
            output_dir,
            f"samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=True_CAR=True_All covariates_[2021, 2022, 2023, 2024, 2025]_no_previous_outbreak.pkl",
        ),
        "rb",
    ) as file:
        posterior_samples = pickle.load(file)
    samples_of_interest = posterior_samples["iv_global"][:, 0]
    samples_of_interest = (
        samples_of_interest
        / df.filter(pl.col("previous_outbreak_size") == 0)["estimated_mmr_cov"]
        .to_numpy()
        .std()
    )
    # make a percent change
    samples_of_interest = jnp.exp(samples_of_interest) * 100 - 100
    medians.append(jnp.median(samples_of_interest))
    lowers.append(jnp.percentile(samples_of_interest, 5))
    uppers.append(jnp.percentile(samples_of_interest, 95))

    with open(
        os.path.join(
            output_dir,
            f"samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=True_CAR=True_All covariates_[2021, 2022, 2023, 2024, 2025]_previous_outbreak_size.pkl",
        ),
        "rb",
    ) as file:
        posterior_samples = pickle.load(file)
    samples_of_interest = posterior_samples["iv_global"][:, 0]
    samples_of_interest = (
        samples_of_interest / df["estimated_mmr_cov"].to_numpy().std()
    )
    # make a percent change
    samples_of_interest = jnp.exp(samples_of_interest) * 100 - 100
    medians.append(jnp.median(samples_of_interest))
    lowers.append(jnp.percentile(samples_of_interest, 2.5))
    uppers.append(jnp.percentile(samples_of_interest, 97.5))

    x = np.arange(3)
    ax.errorbar(
        x,
        medians,
        yerr=[
            np.array(medians) - np.array(lowers),
            np.array(uppers) - np.array(medians),
        ],
        fmt="o",
        color="black",
        capsize=0,
    )
    ax.axhline(y=0, color="black", linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "Baseline",
            "No prior\noutbreaks",
            "Covariate for\nprevious outbreak size",
        ]
    )
    ax.set_ylabel(
        "Percent change in outbreak size per\n1%  increase in MMR coverage"
    )


def plot_previous_outbreak_size_effects(df, fig, ax):
    df = df.sort("year")

    df = df.with_columns(
        previous_outbreak_size=pl.cum_sum("cases_outbreak").over(
            ["county", "state"]
        )
    ).with_columns(
        # want to remove the current year's outbreak size
        previous_outbreak_size=pl.col("previous_outbreak_size")
        - pl.col("cases_outbreak")
    )

    df = df.filter(pl.col("year") >= 2021)
    covariate = df["previous_outbreak_size"].drop_nulls().to_numpy()
    std = jnp.std(covariate)
    with open(
        os.path.join(
            output_dir,
            "samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=True_CAR=True_All covariates_[2021, 2022, 2023, 2024, 2025]_previous_outbreak_size.pkl",
        ),
        "rb",
    ) as file:
        posterior_samples = pickle.load(file)
    samples_of_interest = posterior_samples["regression_betas"][:, -1]
    samples_of_interest = samples_of_interest / std
    # make a percent change
    samples_of_interest = jnp.exp(samples_of_interest) * 100 - 100

    # plot the effect
    ax.hist(
        samples_of_interest,
        bins=100,
        density=True,
        color="gray",
        alpha=0.7,
    )

    ax.axvline(
        x=jnp.percentile(samples_of_interest, 2.5),
        color="red",
    )
    ax.axvline(
        x=jnp.percentile(samples_of_interest, 97.5),
        color="red",
    )
    ax.axvline(
        x=jnp.median(samples_of_interest),
        color="black",
    )
    ax.axvline(x=0, color="blue", linestyle="--")

    ax.set_xlabel(
        "Percent change in incidence per\n1%  increase in previous outbreak size"
    )
    ax.set_ylabel("Density")


def plot_medical_exemption_coefficients(df, fig, ax):
    df_old = df.filter(pl.col("year") <= 2020)
    df_new = df.filter(pl.col("year") >= 2021)
    # we want to plot the baseline me coefficient, the coefficient
    # when we subset to <10% and <5%
    medians_old = []
    lowers_old = []
    uppers_old = []
    medians_new = []
    lowers_new = []
    uppers_new = []
    for idx, years in enumerate(
        [
            [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
            [2021, 2022, 2023, 2024, 2025],
        ]
    ):
        if idx == 0:
            df = df_old
            medians = medians_old
            lowers = lowers_old
            uppers = uppers_old
        else:
            df = df_new
            medians = medians_new
            lowers = lowers_new
            uppers = uppers_new
        for me_cutoff in [None, 10.0, 5.0]:
            if me_cutoff is not None:
                additional_info = f"_me_cutoff={me_cutoff}"
                std = (
                    df.filter(pl.col("med_exempt_pct") <= me_cutoff)[
                        "med_exempt_pct"
                    ]
                    .drop_nulls()
                    .to_numpy()
                    .std()
                )
            else:
                additional_info = ""
                std = df["med_exempt_pct"].drop_nulls().to_numpy().std()
            with open(
                os.path.join(
                    output_dir,
                    f"samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_{years}{additional_info}.pkl",
                ),
                "rb",
            ) as file:
                posterior_samples = pickle.load(file)
            samples_of_interest = posterior_samples["iv_global"][:, -1]
            samples_of_interest = samples_of_interest / std
            # make a percent change
            samples_of_interest = jnp.exp(samples_of_interest) * 100 - 100
            medians.append(jnp.median(samples_of_interest))
            lowers.append(jnp.percentile(samples_of_interest, 5))
            uppers.append(jnp.percentile(samples_of_interest, 95))

    x = np.arange(3)
    ax.errorbar(
        x - 0.1,
        medians_old,
        yerr=[
            np.array(medians_old) - np.array(lowers_old),
            np.array(uppers_old) - np.array(medians_old),
        ],
        fmt="o",
        label="2011-2020",
        color="blue",
        capsize=0,
    )
    ax.errorbar(
        x + 0.1,
        medians_new,
        yerr=[
            np.array(medians_new) - np.array(lowers_new),
            np.array(uppers_new) - np.array(medians_new),
        ],
        fmt="o",
        label="2021-2025",
        color="green",
        capsize=0,
    )
    ax.axhline(y=0, color="black", linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(["All data", "ME ≤10%", "ME ≤5%"])
    ax.set_ylabel(
        "Percent change in outbreak size per\n1%  increase in medical exemptions"
    )
    ax.legend(loc=(0.03, 0.78))


def plot_neighbors_significance(df, fig, ax):
    df = df.filter(pl.col("year") >= 2021)
    neighbor_covariate = df["neighbors_coverage"].drop_nulls().to_numpy()
    std = jnp.std(neighbor_covariate)
    with open(
        os.path.join(
            output_dir,
            "samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_[2021, 2022, 2023, 2024, 2025]_use_neighbors.pkl",
        ),
        "rb",
    ) as file:
        posterior_samples = pickle.load(file)
    samples_of_interest = posterior_samples["regression_betas"][:, -1]
    samples_of_interest = samples_of_interest / std
    # make a percent change
    samples_of_interest = jnp.exp(samples_of_interest) * 100 - 100

    # plot the effect
    ax.hist(
        samples_of_interest,
        bins=100,
        density=True,
        color="gray",
        alpha=0.7,
    )

    ax.axvline(
        x=jnp.percentile(samples_of_interest, 2.5),
        color="red",
    )
    ax.axvline(
        x=jnp.percentile(samples_of_interest, 97.5),
        color="red",
    )
    ax.axvline(
        x=jnp.median(samples_of_interest),
        color="black",
    )
    ax.axvline(x=0, color="blue", linestyle="--")

    ax.set_xlabel(
        "Percent change in incidence per 1% increase\nin neighboring counties' MMR coverage"
    )
    ax.set_ylabel("Density")


if __name__ == "__main__":
    df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
    )

    fig, ax = setup_plot(2, 2)  # type: ignore

    # performance
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
    plot_immunity_effects_on_vaccine_coverage(
        df,
        fig,
        ax[0, 0],  # type: ignore
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
    plot_previous_outbreak_size_effects(
        df,
        fig,
        ax[0, 1],  # type: ignore
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
    plot_medical_exemption_coefficients(
        df,
        fig,
        ax[1, 0],  # type: ignore
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
    plot_neighbors_significance(
        df,
        fig,
        ax[1, 1],  # type: ignore
    )
    fig.tight_layout()

    # save
    fig.savefig(
        os.path.join(
            fig_dir,
            "figure_supplemental_sensitivity_analyses_vaccine_coverage_effect.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(
            fig_dir,
            "figure_supplemental_sensitivity_analyses_vaccine_coverage_effect.pdf",
        ),
        dpi=300,
        bbox_inches="tight",
    )
