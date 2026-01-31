import os
import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import polars as pl
import yaml
from matplotlib import rc

from outbreak_size import adjacency

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


def setup_plot(nrows, ncols, figsize=(30, 12)):
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
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


def plot_predictions(
    df,
    samples,
    naive_model,
    fig,
    ax,
    label_naive="Mechanistic: MMR coverage",
    label_statistical="Statistical: All covariates",
    show_r2=False,
):
    median = jnp.median(samples, axis=0)
    lower = jnp.quantile(samples, 0.025, axis=0)
    upper = jnp.quantile(samples, 0.975, axis=0)

    offset = 0.1
    ax.errorbar(
        df["cases_outbreak"] - offset,
        naive_model["median"],
        yerr=[
            naive_model["median"] - naive_model["lower"],
            naive_model["upper"] - naive_model["median"],
        ],
        fmt="o",
        ecolor="red",
        c="red",
        capsize=0,
        alpha=0.4,
        label=label_naive,
    )

    ax.errorbar(
        df["cases_outbreak"] + offset,
        median,
        yerr=[median - lower, upper - median],
        fmt="o",
        ecolor="green",
        c="green",
        capsize=0,
        alpha=0.8,
        label=label_statistical,
    )

    ax.set_xlabel("Reported cases (N)")
    ax.set_ylabel("Predicted cases (N)")
    ax.set_xscale("symlog")
    ax.set_yscale("symlog")
    ax.plot(
        [0, 1.05 * max(df["cases_outbreak"].max(), upper.max())],  # type: ignore
        [0, 1.05 * max(df["cases_outbreak"].max(), upper.max())],  # type: ignore
        ls="--",
        c="k",
    )
    ax.set_xlim(-0.3, 1.05 * max(df["cases_outbreak"].max(), upper.max()))  # type: ignore
    ax.set_ylim(-0.3, 1.05 * max(df["cases_outbreak"].max(), upper.max()))  # type: ignore

    ax.set_yticks([0, 1, 10, 100], labels=["0", "1", "10", "100"])
    ax.set_xticks([0, 1, 10, 100], labels=["0", "1", "10", "100"])

    ax.legend(loc=(0.0, 1.0))

    # Manually set minor tick locations on the x-axis
    minor_x_ticks = [
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        200,
        300,
        400,
    ]
    ax.xaxis.set_minor_locator(tck.FixedLocator(minor_x_ticks))

    # Manually set minor tick locations on the y-axis
    minor_y_ticks = [
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        200,
        300,
        400,
    ]
    ax.yaxis.set_minor_locator(tck.FixedLocator(minor_y_ticks))

    if show_r2:
        # add a text annotation of the R2
        R2 = jnp.corrcoef(df["cases_outbreak"].to_jax(), median)[0, 1] ** 2
        R2 = R2 * 100  # as a percentage
        ax.text(
            0.0001 * max(df["cases_outbreak"].max(), upper.max()),
            0.35 * max(df["cases_outbreak"].max(), upper.max()),
            f"$R^2$ = {R2:.2f}%",
            fontsize=26,
            color="green",
        )

        # add a text annotation of the R2
        R2 = (
            jnp.corrcoef(df["cases_outbreak"].to_jax(), naive_model["median"])[
                0, 1
            ]
            ** 2
        )
        R2 = R2 * 100  # as a percentage
        ax.text(
            0.0001 * max(df["cases_outbreak"].max(), upper.max()),
            0.63 * max(df["cases_outbreak"].max(), upper.max()),
            f"$R^2$ = {R2:.2f}%",
            fontsize=26,
            color="red",
        )
    ax.set_ylim(-0.05)

    return fig, ax


def plot_ecological_determinants(samples_pre, samples_post, df, fig, ax):
    pooled_covariates = [
        "MMR coverage",
        "Nonmedical exemptions",
        "Medical exemptions",
    ]
    covariates = [
        "% < 5yo",
        "% white",
        "Income (10,000 2024 USD)",
    ]
    covariate_name_map = {
        "MMR coverage": "estimated_mmr_cov",
        "Nonmedical exemptions": "nonmed_exempt_pct",
        "Medical exemptions": "med_exempt_pct",
        "% < 5yo": "fraction_under_5",
        "% school aged": "fraction_school_aged",
        "Average household size": "avg_house_size",
        "Standard deviation of household size": "std_house_size",
        "% public schooled": "fraction_public",
        "% white": "fraction_white",
        "% graduate degree": "fraction_graduate_degree",
        "Population density (per km²)": "pop_density",
        "Income (10,000 2024 USD)": "log_income",
    }

    offset = 0.1
    year_medians = []
    for year_idx, years in enumerate(
        [
            [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
            [2021, 2022, 2023, 2024, 2025],
        ]
    ):
        if year_idx == 0:
            samples = samples_pre
        else:
            samples = samples_post
        all_covariates = [*pooled_covariates, *covariates]
        median_effects = []
        lower_effects = []
        upper_effects = []

        for idx, covariate in enumerate(pooled_covariates):
            samples_of_interest = samples["iv_global"][:, idx]
            # scale by standard deviation
            std = (
                df.filter(pl.col("year").is_in(years))[
                    covariate_name_map[covariate]
                ]
                .to_numpy()
                .std()
            )
            samples_of_interest = samples_of_interest / std
            # exponentiate and multiply by 100
            samples_of_interest = jnp.exp(samples_of_interest) * 100
            # to express as percent change
            samples_of_interest = samples_of_interest - 100
            median_effects.append(jnp.median(samples_of_interest, axis=0))
            lower_effects.append(
                jnp.quantile(samples_of_interest, 0.05, axis=0)
            )
            upper_effects.append(
                jnp.quantile(samples_of_interest, 0.95, axis=0)
            )

        # same process for unpooled covariates
        for covariate in covariates:
            # index is the number of this thing in the dict
            idx = list(covariate_name_map.keys()).index(covariate)
            idx = idx - 3
            samples_of_interest = samples["regression_betas"][:, idx]
            # scale by standard deviation
            std = (
                df.filter(pl.col("year").is_in(years))[
                    covariate_name_map[covariate]
                ]
                .to_numpy()
                .std()
            )
            if "%" in covariate:
                std = std * 100  # because we do the modeling as a fraction
            samples_of_interest = samples_of_interest / std
            if "Income" in covariate:
                # try to convert this to an income instead of log income
                mean_log_income = (
                    df.filter(pl.col("year").is_in(years))[
                        covariate_name_map[covariate]
                    ]
                    .to_numpy()
                    .mean()
                )
                mean_income = jnp.pow(10, mean_log_income)
                samples_of_interest = samples_of_interest / mean_income
                # now convert from dollars to 10,000 USD
                samples_of_interest = samples_of_interest * 10000
            # exponentiate and multiply by 100
            samples_of_interest = jnp.exp(samples_of_interest) * 100
            # to express as percent change
            samples_of_interest = samples_of_interest - 100
            median_effects.append(jnp.median(samples_of_interest, axis=0))
            lower_effects.append(
                jnp.quantile(samples_of_interest, 0.025, axis=0)
            )
            upper_effects.append(
                jnp.quantile(samples_of_interest, 0.975, axis=0)
            )

        ax.errorbar(
            x=[
                i + (year_idx * 2 - 1) * offset
                for i in range(len(all_covariates))
            ],
            y=median_effects,
            yerr=[
                jnp.array(median_effects) - jnp.array(lower_effects),
                jnp.array(upper_effects) - jnp.array(median_effects),
            ],
            fmt="o",
            ecolor="blue" if year_idx == 0 else "green",
            c="blue" if year_idx == 0 else "green",
            capsize=0,
            alpha=0.7,
            label=f"{years[0]}-{years[-1]}",
        )
        year_medians.append(median_effects)

    # for i in range(len(year_medians[0])):
    #     ax.plot([i - offset, i + offset],
    #             [year_medians[0][i], year_medians[1][i]],
    #             linestyle = '-',
    #             color = 'grey')

    # line at 0%
    ax.axhline(0, ls="--", color="k")

    ax.legend(loc=(0.01, 0.78))

    ax.set_xticks(range(len(all_covariates)))  # type: ignore
    # at a 45 degree angle
    ax.set_xticklabels(
        all_covariates,  # type: ignore
        rotation=-60,
        ha="left",
    )
    ax.set_xlabel("Predictor")
    ax.set_ylabel(
        "% change in incidence rate per\n1 unit increase in predictor"
    )
    ax.set_yscale("symlog")
    ax.set_yticks(
        [-100, -10, -1, 0, 1, 10, 100],
        ["-100", "-10", "-1", "0", "1", "10", "100"],
    )

    minor_y_ticks = [
        -90,
        -80,
        -70,
        -60,
        -50,
        -40,
        -30,
        -20,
        -9,
        -8,
        -7,
        -6,
        -5,
        -4,
        -3,
        -2,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
    ]
    ax.yaxis.set_minor_locator(tck.FixedLocator(minor_y_ticks))

    return fig, ax


def plot_spatial_importance(samples, df, adjacency, fig, ax):
    raw_spatial_effect = samples["raw_spatial_effect"]
    # calculate the p-value for each spatial effect
    p_values = 2 * np.minimum(
        np.mean(raw_spatial_effect > 0, axis=0),
        np.mean(raw_spatial_effect < 0, axis=0),
    )

    # figure out who has a neighbor
    has_a_neighor = adjacency.sum(axis=0) > 0
    counties_with_neighbors = (
        df["fips"].unique(maintain_order=True).to_numpy()[has_a_neighor]
    )
    df = df.filter(pl.col("fips").is_in(counties_with_neighbors))
    df = df.group_by("fips", maintain_order=True).agg(
        pl.first("county"),
        pl.first("state"),
        pl.min("cases_outbreak"),
        estimated_mmr_cov=pl.mean("estimated_mmr_cov"),
        # pl.col("estimated_mmr_cov").gather(pl.col("cases_outbreak").arg_max()).first(),
    )

    df = df.with_columns(p_value=pl.Series(p_values))

    # violin plot the p values as a function of case count buckets
    df = df.with_columns(
        case_size_bucket=pl.when(pl.col("cases_outbreak") == 0)
        .then(pl.lit("0"))
        .when(
            (pl.col("cases_outbreak") >= 1) & (pl.col("cases_outbreak") <= 3)
        )
        .then(pl.lit("1"))
        .when(
            (pl.col("cases_outbreak") >= 4) & (pl.col("cases_outbreak") <= 9)
        )
        .then(pl.lit("2"))
        .otherwise(pl.lit("3")),
        coverage_above_95=pl.when(pl.col("estimated_mmr_cov") >= 95)
        .then(pl.lit(True))
        .otherwise(pl.lit(False)),
        vaccine_buckets=pl.when(pl.col("estimated_mmr_cov") <= 80)
        .then(pl.lit("0"))
        .when(pl.col("estimated_mmr_cov") <= 90)
        .then(pl.lit("1"))
        .when(pl.col("estimated_mmr_cov") <= 95)
        .then(pl.lit("2"))
        .otherwise(pl.lit("3")),
        cases_greater_than_10=pl.when(pl.col("cases_outbreak") < 10)
        .then(pl.lit(False))
        .otherwise(pl.lit(True)),
    )

    df = (
        df.group_by(["vaccine_buckets", "cases_greater_than_10"])
        .agg(
            p_value=pl.col("p_value"),
            median=pl.col("p_value").median(),
            lower=pl.col("p_value").quantile(0.25),
            upper=pl.col("p_value").quantile(0.75),
            avg=pl.col("p_value").mean(),
        )
        .sort(["vaccine_buckets", "cases_greater_than_10"])
    )

    # assemble the data for violin plotting

    data_below = (
        df.filter(pl.col("cases_greater_than_10") == False)
        .to_dict()["p_value"]
        .to_list()
    )
    data_above = (
        df.filter(pl.col("cases_greater_than_10") == True)
        .to_dict()["p_value"]
        .to_list()
    )

    medians_below = (
        df.filter(pl.col("cases_greater_than_10") == False)
        .to_dict()["median"]
        .to_numpy()
    )
    medians_above = (
        df.filter(pl.col("cases_greater_than_10") == True)
        .to_dict()["median"]
        .to_numpy()
    )

    lowers_below = (
        df.filter(pl.col("cases_greater_than_10") == False)
        .to_dict()["lower"]
        .to_numpy()
    )
    lowers_above = (
        df.filter(pl.col("cases_greater_than_10") == True)
        .to_dict()["lower"]
        .to_numpy()
    )

    uppers_below = (
        df.filter(pl.col("cases_greater_than_10") == False)
        .to_dict()["upper"]
        .to_numpy()
    )
    uppers_above = (
        df.filter(pl.col("cases_greater_than_10") == True)
        .to_dict()["upper"]
        .to_numpy()
    )

    means_below = (
        df.filter(pl.col("cases_greater_than_10") == False)
        .to_dict()["avg"]
        .to_list()
    )
    means_above = (
        df.filter(pl.col("cases_greater_than_10") == True)
        .to_dict()["avg"]
        .to_list()
    )

    # boxplot and swarmplot of coverage distributions
    offset = 0.2
    violinparts1 = ax.violinplot(
        data_below,
        positions=[i - offset for i in range(len(data_below))],
        widths=0.3,
        showextrema=False,
    )

    for pc in violinparts1["bodies"]:
        pc.set(facecolor="darkgoldenrod")

    ax.errorbar(
        x=[i - offset for i in range(len(medians_below))],
        y=medians_below,
        yerr=[medians_below - lowers_below, uppers_below - medians_below],
        fmt="o",
        capsize=5,
        capthick=2,
        color="darkgoldenrod",
    )

    # add means as stars
    ax.scatter(
        [i - offset for i in range(len(means_below))],
        means_below,
        marker="*",
        s=90,
        color="darkgoldenrod",
    )

    violinparts2 = ax.violinplot(
        data_above,
        positions=[i + offset for i in range(len(data_above))],
        widths=0.3,
        showextrema=False,
    )

    for pc in violinparts2["bodies"]:
        pc.set(facecolor="darkolivegreen")

    ax.errorbar(
        x=[i + offset for i in range(len(medians_above))],
        y=medians_above,
        yerr=[medians_above - lowers_above, uppers_above - medians_above],
        fmt="o",
        capsize=5,
        capthick=2,
        color="darkolivegreen",
    )

    # add means as stars
    ax.scatter(
        [i + offset for i in range(len(means_above))],
        means_above,
        marker="*",
        s=90,
        color="darkolivegreen",
    )

    ax.set_xticks(range(len(data_below)))
    ax.set_xticklabels(
        [
            "≤80%",
            "80% to 90%",
            "90% to 95%",
            ">95%",
        ],
        rotation=-60,
        ha="left",
    )

    ax.legend(
        [violinparts1["bodies"][0], violinparts2["bodies"][0]],
        ["Cases <10", "Cases ≥10"],
        loc=(0.6, 0.85),
    )

    ax.axhline(0.05, ls="--", color="k")

    ax.set_xlabel("MMR coverage (%)")
    ax.set_ylabel("Spatial effect p-value")
    ax.set_ylim([0, 1.0])

    return fig, ax


if __name__ == "__main__":
    df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
    )

    fig, ax = setup_plot(1, 3)  # type: ignore

    # read predictive samples
    with open(
        os.path.join(
            output_dir,
            "predictive_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=False_All covariates_[2021, 2022, 2023, 2024, 2025].pkl",
        ),
        "rb",
    ) as file:
        predictive_samples = pickle.load(file)

    with open(
        os.path.join(
            output_dir,
            "final_outbreak_size_predictions_[2021, 2022, 2023, 2024, 2025].pkl",
        ),
        "rb",
    ) as f:
        naive_model = pickle.load(f)

    # performance
    ax[0].text(  # type: ignore
        -0.25,
        1.08,
        "a",
        transform=ax[0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_predictions(
        df.filter(pl.col("year").is_in([2021, 2022, 2023, 2024, 2025])),
        predictive_samples["cases"],
        naive_model,
        fig,
        ax[0],  # type: ignore
    )

    # while we're here, make a plot of the performance for the R01
    fig_r01, ax_r01 = setup_plot(1, 1, figsize=(10, 7.5))  # type: ignore
    plot_predictions(
        df.filter(pl.col("year").is_in([2021, 2022, 2023, 2024, 2025])),
        predictive_samples["cases"],
        naive_model,
        fig_r01,
        ax_r01,
        label_naive="Naive mechanistic model (vaccine coverage only)",
        label_statistical="Statistical model",
        show_r2=True,
    )
    fig_r01.savefig(
        os.path.join(
            fig_dir, f"figure_county_level_predictions_r01_no_CAR.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )

    with open(
        os.path.join(
            output_dir,
            "samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_[2021, 2022, 2023, 2024, 2025].pkl",
        ),
        "rb",
    ) as file:
        posterior_samples = pickle.load(file)

    with open(
        os.path.join(
            output_dir,
            "samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_[2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020].pkl",
        ),
        "rb",
    ) as file:
        posterior_samples_pre = pickle.load(file)

    ax[1].text(  # type: ignore
        -0.25,
        1.08,
        "b",
        transform=ax[1].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_ecological_determinants(
        posterior_samples_pre,
        posterior_samples,
        df,
        fig,
        ax[1],  # type: ignore
    )

    # make adjacency matrix
    # read adjacency data
    adjacency_matrix = pl.read_csv(
        os.path.join(data_dir, "county_adjacencies.txt"),
        separator="|",
        schema_overrides={
            "County GEOID": pl.String,
            "Neighbor GEOID": pl.String,
        },
    )
    adjacency_matrix = adjacency.assemble_adjacency_matrix(
        adjacency_matrix,
        df.filter(pl.col("year") >= 2021)["fips"]
        .unique(maintain_order=True)
        .to_list(),
    )

    ax[2].text(  # type: ignore
        -0.25,
        1.08,
        "c",
        transform=ax[2].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_spatial_importance(
        posterior_samples,
        df.filter(pl.col("year") >= 2021),
        adjacency_matrix,
        fig,
        ax[2],  # type: ignore
    )

    fig.tight_layout()

    # save
    fig.savefig(
        os.path.join(
            fig_dir, "figure_county_level_predictions_determinants.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(
            fig_dir, "figure_county_level_predictions_determinants.pdf"
        ),
        dpi=300,
        bbox_inches="tight",
    )
