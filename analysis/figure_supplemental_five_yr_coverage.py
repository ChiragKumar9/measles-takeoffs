import os
import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt
import polars as pl
from matplotlib import rc

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)
fig_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")

font = {"family": "Nimbus Roman", "size": 28}
rc("font", **font)


def setup_plot(nrows, ncols):
    fig, ax = plt.subplots(nrows, ncols, figsize=(30, 10))
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


def plot_coverage_various_models(df, fig, ax):
    year_buckets = [
        [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
        [2021, 2022, 2023, 2024, 2025],
    ]
    models = ["MMR coverage", "MMR coverage + exemptions", "All covariates"]
    assembled_effects = pl.DataFrame({"years": year_buckets}).join(
        pl.DataFrame({"model": models}), how="cross"
    )
    assembled_effects = assembled_effects.with_columns(
        year_range=pl.col("years").map_elements(
            lambda x: f"{min(x)}-{max(x)}", return_dtype=pl.String
        ),
        median_coverage_effect=pl.lit(None),
        lower_coverage_effect=pl.lit(None),
        upper_coverage_effect=pl.lit(None),
        mean_coverage_effect=pl.lit(None),
    ).with_columns(
        year_index=pl.when(pl.col("year_range") == "2011-2020")
        .then(0)
        .otherwise(1)
    )

    for year in year_buckets:
        for model_name in models:
            with open(
                os.path.join(
                    output_dir,
                    f"samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=True_CAR=True_{model_name}_{year}.pkl",
                ),
                "rb",
            ) as file:
                posterior_samples = pickle.load(file)

            coverage_effects = posterior_samples["iv_global"][:, 0]
            # scale by the standard deviation of the coverage to get an
            # interpretable term
            std = (
                df.filter(pl.col("year").is_in(year))
                .select("estimated_mmr_cov")
                .to_numpy()
                .std()
            )
            coverage_effects = coverage_effects / std
            coverage_effects = jnp.exp(coverage_effects) * 100
            # to express as percent change
            coverage_effects = coverage_effects - 100
            median_effect = float(jnp.median(coverage_effects, axis=0))
            lower_effect = float(jnp.quantile(coverage_effects, 0.025, axis=0))
            upper_effect = float(jnp.quantile(coverage_effects, 0.975, axis=0))
            mean_effect = float(jnp.mean(coverage_effects, axis=0))

            assembled_effects = assembled_effects.with_columns(
                median_coverage_effect=pl.when(
                    (pl.col("years") == year) & (pl.col("model") == model_name)
                )
                .then(median_effect)
                .otherwise(pl.col("median_coverage_effect")),
                lower_coverage_effect=pl.when(
                    (pl.col("years") == year) & (pl.col("model") == model_name)
                )
                .then(lower_effect)
                .otherwise(pl.col("lower_coverage_effect")),
                upper_coverage_effect=pl.when(
                    (pl.col("years") == year) & (pl.col("model") == model_name)
                )
                .then(upper_effect)
                .otherwise(pl.col("upper_coverage_effect")),
                mean_coverage_effect=pl.when(
                    (pl.col("years") == year) & (pl.col("model") == model_name)
                )
                .then(mean_effect)
                .otherwise(pl.col("mean_coverage_effect")),
            )

    colors = {
        "MMR coverage": "red",
        "MMR coverage + exemptions": "purple",
        "All covariates": "orange",
    }

    for model_name, offset in zip(models, [-0.07, 0, 0.07]):
        model_effects = assembled_effects.filter(
            pl.col("model") == model_name
        ).sort("year_index")
        if model_name == "All covariates":
            ax.errorbar(
                [0 + offset, 1 + offset],
                model_effects["median_coverage_effect"].to_list(),
                yerr=[
                    model_effects["median_coverage_effect"].to_numpy()
                    - model_effects["lower_coverage_effect"].to_numpy(),
                    model_effects["upper_coverage_effect"].to_numpy()
                    - model_effects["median_coverage_effect"].to_numpy(),
                ],
                fmt="o",
                label=model_name,
                color=colors[model_name],
                zorder=10,
            )
        else:
            ax.errorbar(
                [0 + offset, 1 + offset],
                model_effects["median_coverage_effect"].to_list(),
                yerr=[
                    model_effects["median_coverage_effect"].to_numpy()
                    - model_effects["lower_coverage_effect"].to_numpy(),
                    model_effects["upper_coverage_effect"].to_numpy()
                    - model_effects["median_coverage_effect"].to_numpy(),
                ],
                fmt="o",
                label=model_name,
                color=colors[model_name],
            )

        # scatter plot the mean with a "*" as its marker
        # ax.scatter(
        #     [0 + offset, 1 + offset],
        #     model_effects["mean_coverage_effect"].to_list(),
        #     marker="*",
        #     s=12,
        #     color=colors[model_name],
        # )

    # dashed line at 1
    ax.axhline(y=0, color="k", linestyle="--")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["2011-2020", "2021-2025"])
    ax.set_xlim([-0.18, 1.18])

    ax.set_xlabel("Year")
    ax.set_ylabel(
        "% change in incidence rate per 1pp increase\nin 5 year MMR vaccine coverage"
    )
    ax.legend(loc=(0.2, 0.65), edgecolor="black")

    return fig, ax


def plot_5yr_performance_comparison(df, fig, ax):
    with open(
        os.path.join(
            output_dir,
            f"predictive_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_[2021, 2022, 2023, 2024, 2025].pkl",
        ),
        "rb",
    ) as file:
        one_year = pickle.load(file)

    with open(
        os.path.join(
            output_dir,
            f"predictive_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=True_CAR=True_All covariates_[2021, 2022, 2023, 2024, 2025].pkl",
        ),
        "rb",
    ) as file:
        five_year = pickle.load(file)

    df = df.filter(pl.col("year").is_in([2021, 2022, 2023, 2024, 2025]))
    # concatenate on the case estimates
    df = df.with_columns(
        one_year=pl.Series(jnp.median(one_year["cases"], axis=0).__array__()),
        five_year=pl.Series(
            jnp.median(five_year["cases"], axis=0).__array__()
        ),
    ).with_columns(
        one_year_residuals=pl.col("cases_outbreak") - pl.col("one_year"),
        five_year_residuals=pl.col("cases_outbreak") - pl.col("five_year"),
    )

    # join this up with the year-over-year vaccination change data
    if not os.path.exists(
        os.path.join(data_dir, "year_over_year_change_vaccination_data.csv")
    ):
        print(
            "skipped panel: year_over_year_change_vaccination_data.csv not provided."
        )
        return None
    vaccination_roc = pl.read_csv(
        os.path.join(data_dir, "year_over_year_change_vaccination_data.csv"),
        # read the fips codes as strings
        schema_overrides={"fips": pl.String},
    )
    vaccination_roc = vaccination_roc.with_columns(
        # because outbreaks are defined by coverage in the preceding year, adjust
        # in particular, 2017 vaccination coverage data tells us what happens in 2018
        # in terms of outbreaks
        # so if we want to join up, we need to increment year by 1
        year=pl.col("year") + 1
    )

    df = df.join(vaccination_roc, on=["fips", "year"])

    # remove any cases where vaccination_roc is null
    df = df.filter(pl.col("year_over_year_change").is_not_null())

    # We're going to violin plot the residuals by buckets
    df = df.with_columns(
        roc_buckets=pl.when(pl.col("year_over_year_change") <= -1.5)
        .then(pl.lit(0))
        .when(
            (pl.col("year_over_year_change") > -1.5)
            & (pl.col("year_over_year_change") <= 1.5)
        )
        .then(pl.lit(1))
        .when((pl.col("year_over_year_change") > 1.5))
        .then(pl.lit(2))
    )

    df = (
        df.group_by(["roc_buckets"])
        .agg(
            one_year_residuals=pl.col("one_year_residuals"),
            five_year_residuals=pl.col("five_year_residuals"),
            one_year_median=pl.col("one_year_residuals").median(),
            five_year_median=pl.col("five_year_residuals").median(),
            one_year_lower=pl.col("one_year_residuals").quantile(0.25),
            five_year_lower=pl.col("five_year_residuals").quantile(0.25),
            one_year_upper=pl.col("one_year_residuals").quantile(0.75),
            five_year_upper=pl.col("five_year_residuals").quantile(0.75),
            one_year_avg=pl.col("one_year_residuals").mean(),
            five_year_avg=pl.col("five_year_residuals").mean(),
        )
        .sort("roc_buckets")
    )

    # assemble the data for violin plotting

    data_one_year = df.to_dict()["one_year_residuals"].to_list()
    data_five_year = df.to_dict()["five_year_residuals"].to_list()

    medians_one_year = df.to_dict()["one_year_median"].to_numpy()
    medians_five_year = df.to_dict()["five_year_median"].to_numpy()
    lowers_one_year = df.to_dict()["one_year_lower"].to_numpy()
    lowers_five_year = df.to_dict()["five_year_lower"].to_numpy()
    uppers_one_year = df.to_dict()["one_year_upper"].to_numpy()
    uppers_five_year = df.to_dict()["five_year_upper"].to_numpy()
    means_one_year = df.to_dict()["one_year_avg"].to_list()
    means_five_year = df.to_dict()["five_year_avg"].to_list()

    ax.axhline(0, ls="--", color="k")

    offset = 0.2
    violinparts1 = ax.violinplot(
        data_one_year,
        positions=[i - offset for i in range(len(data_one_year))],
        widths=0.3,
        showextrema=False,
    )

    for pc in violinparts1["bodies"]:
        pc.set(facecolor="green")

    ax.errorbar(
        x=[i - offset for i in range(len(medians_one_year))],
        y=medians_one_year,
        yerr=[
            medians_one_year - lowers_one_year,
            uppers_one_year - medians_one_year,
        ],
        fmt="o",
        capsize=10,
        capthick=2,
        color="green",
    )

    # add means as stars
    ax.scatter(
        [i - offset for i in range(len(means_one_year))],
        means_one_year,
        marker="*",
        s=90,
        color="green",
    )

    violinparts2 = ax.violinplot(
        data_five_year,
        positions=[i + offset for i in range(len(data_five_year))],
        widths=0.3,
        showextrema=False,
    )

    for pc in violinparts2["bodies"]:
        pc.set(facecolor="yellowgreen")

    ax.errorbar(
        x=[i + offset for i in range(len(medians_five_year))],
        y=medians_five_year,
        yerr=[
            medians_five_year - lowers_five_year,
            uppers_five_year - medians_five_year,
        ],
        fmt="o",
        capsize=10,
        capthick=2,
        color="yellowgreen",
    )

    # add means as stars
    ax.scatter(
        [i + offset for i in range(len(means_five_year))],
        means_five_year,
        marker="*",
        s=90,
        color="yellowgreen",
    )

    ax.set_xticks(range(len(data_one_year)))
    ax.set_xticklabels(
        [
            "≤-1.5%",
            "-1.5% to 1.5%",
            ">1.5%",
        ],
        rotation=-60,
        ha="left",
    )

    ax.legend(
        [violinparts1["bodies"][0], violinparts2["bodies"][0]],
        ["One year", "Five year"],
        edgecolor="black",
        loc="lower left",
    )

    ax.set_xlabel("Year-to-year change in MMR coverage (pp)")
    ax.set_ylabel("Residuals (reported - predicted cases)")

    ax.set_yticks([-10, 0, 10, 20])

    return fig, ax


def plot_5yr_performance_comparison_mechanistic(df, fig, ax):
    with open(
        os.path.join(
            output_dir,
            f"final_outbreak_size_predictions_[2021, 2022, 2023, 2024, 2025]_five_year=False.pkl",
        ),
        "rb",
    ) as file:
        one_year = pickle.load(file)

    with open(
        os.path.join(
            output_dir,
            f"final_outbreak_size_predictions_[2021, 2022, 2023, 2024, 2025]_five_year=True.pkl",
        ),
        "rb",
    ) as file:
        five_year = pickle.load(file)

    df = df.filter(pl.col("year").is_in([2021, 2022, 2023, 2024, 2025]))
    df = df.with_columns(
        one_year=pl.Series(one_year["median"]),
        five_year=pl.Series(five_year["median"]),
    ).with_columns(
        one_year_residuals=pl.col("cases_outbreak") - pl.col("one_year"),
        five_year_residuals=pl.col("cases_outbreak") - pl.col("five_year"),
    )

    if not os.path.exists(
        os.path.join(data_dir, "year_over_year_change_vaccination_data.csv")
    ):
        print(
            "skipped panel: year_over_year_change_vaccination_data.csv not provided."
        )
        return None
    vaccination_roc = pl.read_csv(
        os.path.join(data_dir, "year_over_year_change_vaccination_data.csv"),
        schema_overrides={"fips": pl.String},
    )
    vaccination_roc = vaccination_roc.with_columns(year=pl.col("year") + 1)

    df = df.join(vaccination_roc, on=["fips", "year"])
    df = df.filter(pl.col("year_over_year_change").is_not_null())

    df = df.with_columns(
        roc_buckets=pl.when(pl.col("year_over_year_change") <= -1.5)
        .then(pl.lit(0))
        .when(
            (pl.col("year_over_year_change") > -1.5)
            & (pl.col("year_over_year_change") <= 1.5)
        )
        .then(pl.lit(1))
        .when((pl.col("year_over_year_change") > 1.5))
        .then(pl.lit(2))
    )

    df = (
        df.group_by(["roc_buckets"])
        .agg(
            one_year_residuals=pl.col("one_year_residuals"),
            five_year_residuals=pl.col("five_year_residuals"),
            one_year_median=pl.col("one_year_residuals").median(),
            five_year_median=pl.col("five_year_residuals").median(),
            one_year_lower=pl.col("one_year_residuals").quantile(0.25),
            five_year_lower=pl.col("five_year_residuals").quantile(0.25),
            one_year_upper=pl.col("one_year_residuals").quantile(0.75),
            five_year_upper=pl.col("five_year_residuals").quantile(0.75),
            one_year_avg=pl.col("one_year_residuals").mean(),
            five_year_avg=pl.col("five_year_residuals").mean(),
        )
        .sort("roc_buckets")
    )

    data_one_year = df.to_dict()["one_year_residuals"].to_list()
    data_five_year = df.to_dict()["five_year_residuals"].to_list()

    medians_one_year = df.to_dict()["one_year_median"].to_numpy()
    medians_five_year = df.to_dict()["five_year_median"].to_numpy()
    lowers_one_year = df.to_dict()["one_year_lower"].to_numpy()
    lowers_five_year = df.to_dict()["five_year_lower"].to_numpy()
    uppers_one_year = df.to_dict()["one_year_upper"].to_numpy()
    uppers_five_year = df.to_dict()["five_year_upper"].to_numpy()
    means_one_year = df.to_dict()["one_year_avg"].to_list()
    means_five_year = df.to_dict()["five_year_avg"].to_list()

    ax.axhline(0, ls="--", color="k")

    offset = 0.2
    violinparts1 = ax.violinplot(
        data_one_year,
        positions=[i - offset for i in range(len(data_one_year))],
        widths=0.3,
        showextrema=False,
    )
    for pc in violinparts1["bodies"]:
        pc.set(facecolor="green")

    ax.errorbar(
        x=[i - offset for i in range(len(medians_one_year))],
        y=medians_one_year,
        yerr=[
            medians_one_year - lowers_one_year,
            uppers_one_year - medians_one_year,
        ],
        fmt="o",
        capsize=10,
        capthick=2,
        color="green",
    )
    ax.scatter(
        [i - offset for i in range(len(means_one_year))],
        means_one_year,
        marker="*",
        s=90,
        color="green",
    )

    violinparts2 = ax.violinplot(
        data_five_year,
        positions=[i + offset for i in range(len(data_five_year))],
        widths=0.3,
        showextrema=False,
    )
    for pc in violinparts2["bodies"]:
        pc.set(facecolor="yellowgreen")

    ax.errorbar(
        x=[i + offset for i in range(len(medians_five_year))],
        y=medians_five_year,
        yerr=[
            medians_five_year - lowers_five_year,
            uppers_five_year - medians_five_year,
        ],
        fmt="o",
        capsize=10,
        capthick=2,
        color="yellowgreen",
    )
    ax.scatter(
        [i + offset for i in range(len(means_five_year))],
        means_five_year,
        marker="*",
        s=90,
        color="yellowgreen",
    )

    ax.set_xticks(range(len(data_one_year)))
    ax.set_xticklabels(
        ["≤-1.5%", "-1.5% to 1.5%", ">1.5%"],
        rotation=-60,
        ha="left",
    )
    ax.legend(
        [violinparts1["bodies"][0], violinparts2["bodies"][0]],
        ["One year", "Five year"],
        edgecolor="black",
        loc="lower left",
    )
    ax.set_xlabel("Year-to-year change in MMR coverage (pp)")
    ax.set_ylabel("Residuals (reported - predicted cases)")

    ax.set_yscale("symlog")

    return fig, ax


if __name__ == "__main__":
    fig, ax = setup_plot(1, 3)  # type: ignore

    df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        schema_overrides={"fips": pl.String},
    )

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
    plot_5yr_performance_comparison_mechanistic(df, fig, ax[0])  # type: ignore

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
    plot_5yr_performance_comparison(df, fig, ax[1])  # type: ignore

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
    plot_coverage_various_models(df, fig, ax[2])  # type: ignore

    fig.tight_layout()

    fig.savefig(
        os.path.join(fig_dir, "figure_supplemental_five_yr_coverage.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(fig_dir, "figure_supplemental_five_yr_coverage.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
