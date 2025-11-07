import os
import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy.stats as stats
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


# plot for the vaccination coverage in county-years with >= 10 cases
def plot_coverage_distribution(
    df,
    cases_threshold,
    vaccination_threshold,
    fig,
    ax,
    vaccination_metric="estimated_mmr_cov",
):
    # want to make violin plots by outbreak size category and then subfaceted by year
    df = df.with_columns(
        outbreak_size_buckets=pl.col("cases_outbreak").cut(
            [0, 1, 4, cases_threshold, 1000000000],
            labels=["-1", "0", "1", "2", "3", "4"],
            left_closed=True,
        ),
        year_of_interest=pl.col("year").is_in([2021, 2022, 2023, 2024, 2025]),
    )
    df = df.group_by(["outbreak_size_buckets", "year_of_interest"]).agg(
        pl.col(vaccination_metric)
    )
    df = df.sort(by=["outbreak_size_buckets", "year_of_interest"])
    df = df.with_columns(
        avg_cov=pl.col(vaccination_metric).list.mean(),
        median_cov=pl.col(vaccination_metric).list.median(),
        lower_cov=pl.col(vaccination_metric).map_elements(
            lambda x: np.percentile(x, 25)
        ),
        upper_cov=pl.col(vaccination_metric).map_elements(
            lambda x: np.percentile(x, 75)
        ),
    )

    data_pre = (
        df.filter(pl.col("year_of_interest") == False)
        .to_dict()[vaccination_metric]
        .to_list()
    )
    data_post = (
        df.filter(pl.col("year_of_interest") == True)
        .to_dict()[vaccination_metric]
        .to_list()
    )
    medians_pre = (
        df.filter(pl.col("year_of_interest") == False)
        .to_dict()["median_cov"]
        .to_numpy()
    )
    medians_post = (
        df.filter(pl.col("year_of_interest") == True)
        .to_dict()["median_cov"]
        .to_numpy()
    )
    lowers_pre = (
        df.filter(pl.col("year_of_interest") == False)
        .to_dict()["lower_cov"]
        .to_numpy()
    )
    lowers_post = (
        df.filter(pl.col("year_of_interest") == True)
        .to_dict()["lower_cov"]
        .to_numpy()
    )
    uppers_pre = (
        df.filter(pl.col("year_of_interest") == False)
        .to_dict()["upper_cov"]
        .to_numpy()
    )
    uppers_post = (
        df.filter(pl.col("year_of_interest") == True)
        .to_dict()["upper_cov"]
        .to_numpy()
    )
    means_pre = (
        df.filter(pl.col("year_of_interest") == False)
        .to_dict()["avg_cov"]
        .to_list()
    )
    means_post = (
        df.filter(pl.col("year_of_interest") == True)
        .to_dict()["avg_cov"]
        .to_list()
    )

    # KS test between the two distributions
    print(
        "KS test between pre COVID coverage across no outbreak and 10+ cases"
    )
    ks_stat, p_value = stats.ks_2samp(
        df.filter(pl.col("year_of_interest") == False).filter(
            pl.col("outbreak_size_buckets") == "3"
        )[vaccination_metric][0],
        df.filter(pl.col("year_of_interest") == False).filter(
            pl.col("outbreak_size_buckets") == "0"
        )[vaccination_metric][0],
        method="exact",
        alternative="greater",
    )
    print(f"KS statistic: {ks_stat}, p-value: {p_value}")

    print(
        "KS test between pre and post COVID coverage with greater than 10 cases"
    )
    ks_stat, p_value = stats.ks_2samp(
        df.filter(pl.col("year_of_interest") == False).filter(
            pl.col("outbreak_size_buckets") == "3"
        )[vaccination_metric][0],
        df.filter(pl.col("year_of_interest") == True).filter(
            pl.col("outbreak_size_buckets") == "3"
        )[vaccination_metric][0],
        method="exact",
        alternative="less",
    )
    print(f"KS statistic: {ks_stat}, p-value: {p_value}")

    print(
        "KS test between post COVID coverage across no outbreak and 10+ cases"
    )
    ks_stat, p_value = stats.ks_2samp(
        df.filter(pl.col("year_of_interest") == True).filter(
            pl.col("outbreak_size_buckets") == "3"
        )[vaccination_metric][0],
        df.filter(pl.col("year_of_interest") == True).filter(
            pl.col("outbreak_size_buckets") == "0"
        )[vaccination_metric][0],
        method="exact",
        alternative="greater",
    )
    print(f"KS statistic: {ks_stat}, p-value: {p_value}")

    # boxplot and swarmplot of coverage distributions
    offset = 0.2
    violinparts1 = ax.violinplot(
        data_pre,
        positions=[i - offset for i in range(len(data_pre))],
        widths=0.3,
        showextrema=False,
    )

    for pc in violinparts1["bodies"]:
        pc.set(facecolor="blue")

    ax.errorbar(
        x=[i - offset for i in range(len(medians_pre))],
        y=medians_pre,
        yerr=[medians_pre - lowers_pre, uppers_pre - medians_pre],
        fmt="o",
        capsize=5,
        color="blue",
    )

    # add means as stars
    ax.scatter(
        [i - offset for i in range(len(means_pre))],
        means_pre,
        marker="*",
        s=90,
        color="blue",
    )

    violinparts2 = ax.violinplot(
        data_post,
        positions=[i + offset for i in range(len(data_post))],
        widths=0.3,
        showextrema=False,
    )

    for pc in violinparts2["bodies"]:
        pc.set(facecolor="green")

    ax.errorbar(
        x=[i + offset for i in range(len(medians_post))],
        y=medians_post,
        yerr=[medians_post - lowers_post, uppers_post - medians_post],
        fmt="o",
        capsize=5,
        color="green",
    )

    # add means as stars
    ax.scatter(
        [i + offset for i in range(len(means_post))],
        means_post,
        marker="*",
        s=90,
        color="green",
    )

    ax.set_xticks(range(len(data_pre)))
    ax.set_xticklabels(
        [
            "0",
            "1-3",
            "4-9",
            f"≥{cases_threshold}",
        ]
    )

    ax.legend(
        [violinparts1["bodies"][0], violinparts2["bodies"][0]],
        ["2011-2020", "2021-2025"],
    )
    ax.axhline(y=vaccination_threshold, color="k", linestyle="--")
    ax.set_ylabel("Estimated MMR\nvaccine coverage (%)")
    ax.set_xlabel("Outbreak size (cases)")

    return fig, ax


def plot_outbreak_identification_rate(df, fig, ax):
    # calculate what percentage of outbreaks we would have captured had we said
    # all counties with coverage less than 90, 95, 98% were at risk
    df = df.with_columns(
        recent=pl.col("year").is_in([2021, 2022, 2023, 2024, 2025]),
        below_90=pl.col("estimated_mmr_cov") < 90,
        below_92=pl.col("estimated_mmr_cov") < 92,
        below_94=pl.col("estimated_mmr_cov") < 94,
        below_96=pl.col("estimated_mmr_cov") < 96,
        below_98=pl.col("estimated_mmr_cov") < 98,
        cases_10=pl.col("cases_outbreak") >= 10,
        cases_5=pl.col("cases_outbreak") >= 5,
        cases_3=pl.col("cases_outbreak") >= 3,
    )

    df = df.with_columns(
        total_10=pl.col("cases_10").sum().over("recent"),
        total_5=pl.col("cases_5").sum().over("recent"),
        total_3=pl.col("cases_3").sum().over("recent"),
    )

    df_90 = (
        df.group_by(["recent", "below_90"])
        .agg(
            identified_10=pl.sum("cases_10") / pl.col("total_10").first(),
            identified_5=pl.sum("cases_5") / pl.col("total_5").first(),
            identified_3=pl.sum("cases_3") / pl.col("total_3").first(),
        )
        .filter(pl.col("below_90") == True)
        .with_columns(coverage=pl.lit(90))
    )

    df_92 = (
        df.group_by(["recent", "below_92"])
        .agg(
            identified_10=pl.sum("cases_10") / pl.col("total_10").first(),
            identified_5=pl.sum("cases_5") / pl.col("total_5").first(),
            identified_3=pl.sum("cases_3") / pl.col("total_3").first(),
        )
        .filter(pl.col("below_92") == True)
        .with_columns(coverage=pl.lit(92))
    )

    df_94 = (
        df.group_by(["recent", "below_94"])
        .agg(
            identified_10=pl.sum("cases_10") / pl.col("total_10").first(),
            identified_5=pl.sum("cases_5") / pl.col("total_5").first(),
            identified_3=pl.sum("cases_3") / pl.col("total_3").first(),
        )
        .filter(pl.col("below_94") == True)
        .with_columns(coverage=pl.lit(94))
    )

    df_96 = (
        df.group_by(["recent", "below_96"])
        .agg(
            identified_10=pl.sum("cases_10") / pl.col("total_10").first(),
            identified_5=pl.sum("cases_5") / pl.col("total_5").first(),
            identified_3=pl.sum("cases_3") / pl.col("total_3").first(),
        )
        .filter(pl.col("below_96") == True)
        .with_columns(coverage=pl.lit(96))
    )

    df_98 = (
        df.group_by(["recent", "below_98"])
        .agg(
            identified_10=pl.sum("cases_10") / pl.col("total_10").first(),
            identified_5=pl.sum("cases_5") / pl.col("total_5").first(),
            identified_3=pl.sum("cases_3") / pl.col("total_3").first(),
        )
        .filter(pl.col("below_98") == True)
        .with_columns(coverage=pl.lit(98))
    )

    # stitch the values we need into one dataframe for analysis
    df_identification = pl.concat(
        [df_90, df_92, df_94, df_96, df_98], how="diagonal_relaxed"
    )

    ax.plot(
        df_identification.filter(pl.col("recent") == False)[
            "coverage"
        ].to_list(),
        100
        * df_identification.filter(pl.col("recent") == False)[
            "identified_10"
        ].to_numpy(),
        marker="P",
        label="10+, 2011-2020",
        color="blue",
        linestyle="--",
    )

    ax.plot(
        df_identification.filter(pl.col("recent") == True)[
            "coverage"
        ].to_list(),
        100
        * df_identification.filter(pl.col("recent") == True)[
            "identified_10"
        ].to_numpy(),
        marker="P",
        label="10+, 2021-2025",
        color="green",
        linestyle="--",
    )

    ax.plot(
        df_identification.filter(pl.col("recent") == False)[
            "coverage"
        ].to_list(),
        100
        * df_identification.filter(pl.col("recent") == False)[
            "identified_5"
        ].to_numpy(),
        marker="s",
        label="5+, 2011-2020",
        color="blue",
        linestyle="--",
    )

    ax.plot(
        df_identification.filter(pl.col("recent") == True)[
            "coverage"
        ].to_list(),
        100
        * df_identification.filter(pl.col("recent") == True)[
            "identified_5"
        ].to_numpy(),
        marker="s",
        label="5+, 2021-2025",
        color="green",
        linestyle="--",
    )

    ax.legend()
    ax.set_xlabel("Coverage threshold (%)")
    ax.set_ylabel("Outbreaks identified (%)")

    return df_identification, fig, ax


def plot_coverage_various_models(df, fig, ax):
    year_buckets = [
        [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
        [2021, 2022, 2023, 2024, 2025],
    ]
    models = ["Just coverage", "Just vaccination measures", "All covariates"]
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
                    f"samples_partial_pooled_poisson_five_year=False_CAR=True_{model_name}_{year}.pkl",
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
            median_effect = float(jnp.median(coverage_effects, axis=0))
            lower_effect = float(jnp.quantile(coverage_effects, 0.05, axis=0))
            upper_effect = float(jnp.quantile(coverage_effects, 0.95, axis=0))
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
        "Just coverage": "orange",
        "Just vaccination measures": "purple",
        "All covariates": "red",
    }

    for model_name in reversed(models):
        model_effects = assembled_effects.filter(
            pl.col("model") == model_name
        ).sort("year_index")
        if model_name == "Just vaccination measures":
            ax.errorbar(
                ["2011-2020", "2021-2025"],
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
                ["2011-2020", "2021-2025"],
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
        ax.plot(
            ["2011-2020", "2021-2025"],
            model_effects["mean_coverage_effect"].to_list(),
            marker="*",
            markersize=12,
            color=colors[model_name],
        )

    # dashed line at 1
    ax.axhline(y=100, color="k", linestyle="--")
    ax.set_xlim([-0.25, 1.25])

    ax.set_xlabel("Year")
    ax.set_ylabel("Estimated marginal IRR of\nvaccination coverage (%)")
    ax.legend()

    return fig, ax


def plot_effect_over_time(samples, df, years_in_order, fig, ax):
    # scale by standard deviation to make it interpretable
    std = df["estimated_mmr_cov"].to_numpy().std()
    samples = samples / std
    # exponentiate and multiply by 100
    samples = jnp.exp(samples) * 100

    median_effect = jnp.median(samples, axis=0)
    lower_effect = jnp.quantile(samples, 0.05, axis=0)
    upper_effect = jnp.quantile(samples, 0.95, axis=0)
    ax.fill_between(
        years_in_order,
        lower_effect,
        upper_effect,
        color="grey",
        alpha=0.3,
    )
    ax.plot(years_in_order, median_effect, color="grey")

    # histogram

    counts, bins = np.histogram(df["year"], bins=np.arange(2011, 2027) - 0.5)
    ax2 = ax.twinx()
    ax2.bar(bins[:-1], counts, width=np.diff(bins), alpha=0.2, align="edge")
    ax2.set_ylabel("County-year outbreaks (N)")
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.get_xaxis().tick_bottom()
    ax2.get_yaxis().tick_right()
    ax2.tick_params(axis="x", direction="out")
    ax2.tick_params(axis="y", direction="out")
    # offset the spines
    for spine in ax2.spines.values():
        spine.set_position(("outward", 5))
    # put the grid behind
    ax2.set_axisbelow(True)

    # horizontal line at y=100
    ax.axhline(100, ls="--", color="black")

    ax.set_xlabel("Year")
    ax.set_ylabel(f"Estimated marginal IRR of\nvaccination coverage")

    return fig, ax


if __name__ == "__main__":
    df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
    )

    fig, ax = setup_plot(2, 2)  # type: ignore

    # coverages and outbreak size relationship changes over time
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
    plot_coverage_distribution(
        df,
        10,
        95,
        fig,
        ax[0, 0],  # type: ignore
        vaccination_metric="estimated_mmr_cov",
    )

    # threshold analysis -- outbreak identification rate for thresholds
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
    plot_outbreak_identification_rate(
        df,
        fig,
        ax[0, 1],  # type: ignore
    )

    # significance of coverage across various models
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
    plot_coverage_various_models(
        df,
        fig,
        ax[1, 0],  # type: ignore
    )

    # read predictive samples
    with open(
        os.path.join(
            output_dir,
            "predictive_random_walk_poisson_CAR.pkl",
        ),
        "rb",
    ) as file:
        predictive_samples = pickle.load(file)

    # plot effect over time
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

    plot_effect_over_time(
        predictive_samples["effects_over_time"][:, :, 0],
        df,
        list(
            reversed(
                [2011, 2012.5, 2014.5, 2016.5, 2018.5, 2020.5, 2022.5, 2025]
            )
        ),
        fig,
        ax[1, 1],  # type: ignore
    )

    fig.tight_layout()

    # save
    fig.savefig(
        os.path.join(fig_dir, "figure_changing_county_level_predictors.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(fig_dir, "figure_changing_county_level_predictors.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
