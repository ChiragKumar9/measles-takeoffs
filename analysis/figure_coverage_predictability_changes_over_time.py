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
    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 10, nrows * 7.5))
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


# plot for the vaccination coverage in county-years with >= 10 cases
def plot_coverage_distribution(
    df,
    vaccination_threshold,
    fig,
    ax,
    vaccination_metric="estimated_mmr_cov",
):
    # want to make violin plots by outbreak size category and then subfaceted by year
    df = df.with_columns(
        outbreak_size_buckets=pl.col("cases_outbreak").cut(
            [0, 4, 9, 19],
            labels=["-1", "0", "1", "2", "3"],
            left_closed=False,
        ),
        year_of_interest=pl.col("year").is_in([2021, 2022, 2023, 2024, 2025]),
    )
    df_correlation = df.filter(pl.col("cases_outbreak") >= 1)
    print(
        stats.spearmanr(
            df_correlation.filter(pl.col("year_of_interest") == False)[
                vaccination_metric
            ],
            df_correlation.filter(pl.col("year_of_interest") == False)[
                "cases_outbreak"
            ],
        )
    )

    print(
        stats.spearmanr(
            df_correlation.filter(pl.col("year_of_interest") == True)[
                vaccination_metric
            ],
            df_correlation.filter(pl.col("year_of_interest") == True)[
                "cases_outbreak"
            ],
        )
    )
    df = df.group_by(["outbreak_size_buckets", "year_of_interest"]).agg(
        pl.col(vaccination_metric), pl.col("cases_outbreak")
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
            pl.col("outbreak_size_buckets") == "2"
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
            pl.col("outbreak_size_buckets") == "2"
        )[vaccination_metric][0],
        df.filter(pl.col("year_of_interest") == True).filter(
            pl.col("outbreak_size_buckets") == "2"
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
            pl.col("outbreak_size_buckets") == "2"
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
        capsize=10,
        capthick=2,
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
        capsize=10,
        capthick=2,
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

    # Add significance bar between post-COVID 0 cases and ≥20 cases
    # Position 0 corresponds to "0 cases", position 4 corresponds to "≥20 cases"
    x1 = 0 + offset  # Position for "0" cases (green/post)
    x2 = 4 + offset  # Position for "≥20" cases (green/post)

    # Get the max y-value from both distributions to place the bar above them
    y_max = max(
        max(data_post[0]) if len(data_post[0]) > 0 else 0,
        max(data_post[4]) if len(data_post[4]) > 0 else 0,
    )
    bar_height = y_max + 1

    # Draw the significance bar
    ax.plot(
        [x1, x1, x2, x2],
        [bar_height - 0.5, bar_height, bar_height, bar_height - 0.5],
        "k-",
        linewidth=1.5,
    )

    # Add the asterisk in the middle
    ax.text(
        (x1 + x2) / 2,
        bar_height + 0.5,
        "*",
        ha="center",
        va="bottom",
        fontsize=16,
        fontweight="bold",
    )

    ax.set_xticks(range(len(data_pre)))
    ax.set_xticklabels(
        [
            "0",
            "1-4",
            "5-9",
            "10-19",
            "≥20",
        ]
    )

    ax.legend(
        [violinparts1["bodies"][0], violinparts2["bodies"][0]],
        ["2011-2020", "2021-2025"],
        loc=(0.6, 0.01),
        edgecolor="black",
    )
    ax.axhline(y=vaccination_threshold, color="k", linestyle="--")
    ax.set_ylabel("MMR vaccine coverage (%)")
    ax.set_xlabel("Outbreak size (N)")
    # ax.set_ylim(top=100)

    return fig, ax


# plot for the vaccination coverage in county-years with >= 10 cases
def plot_coverage_distribution_all_years(
    df,
    vaccination_threshold,
    fig,
    ax,
    vaccination_metric="estimated_mmr_cov",
):
    # want to make violin plots by outbreak size category
    df = df.with_columns(
        outbreak_size_buckets=pl.col("cases_outbreak").cut(
            [0, 4, 9, 19],
            labels=["-1", "0", "1", "2", "3"],
            left_closed=False,
        ),
    )
    df_correlation = df.filter(pl.col("cases_outbreak") >= 1)
    print(
        stats.spearmanr(
            df_correlation[vaccination_metric],
            df_correlation["cases_outbreak"],
        )
    )

    df = df.group_by(["outbreak_size_buckets"]).agg(
        pl.col(vaccination_metric), pl.col("cases_outbreak")
    )
    df = df.sort(by=["outbreak_size_buckets"])
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

    data = df.to_dict()[vaccination_metric].to_list()
    medians = df.to_dict()["median_cov"].to_numpy()
    lowers = df.to_dict()["lower_cov"].to_numpy()
    uppers = df.to_dict()["upper_cov"].to_numpy()
    means = df.to_dict()["avg_cov"].to_list()

    # boxplot and swarmplot of coverage distributions
    violinparts = ax.violinplot(
        data,
        positions=range(len(data)),
        widths=0.3,
        showextrema=False,
    )

    for pc in violinparts["bodies"]:
        pc.set(facecolor="grey")

    ax.errorbar(
        x=range(len(medians)),
        y=medians,
        yerr=[medians - lowers, uppers - medians],
        fmt="o",
        capsize=10,
        capthick=2,
        color="grey",
    )

    # add means as stars
    ax.scatter(
        range(len(means)),
        means,
        marker="*",
        s=90,
        color="grey",
    )

    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(
        [
            "0",
            "1-4",
            "5-9",
            "10-19",
            "≥20",
        ]
    )

    ax.axhline(y=vaccination_threshold, color="k", linestyle="--")
    ax.set_ylabel("MMR vaccine coverage (%)")
    ax.set_xlabel("Outbreak size (N)")
    ax.set_ylim(top=100)

    return fig, ax


def plot_outbreak_identification_rate(df, fig, ax):
    # calculate what percentage of outbreaks we would have captured had we said
    # all counties with coverage less than threshold% were at risk
    df = df.with_columns(
        recent=pl.col("year").is_in([2021, 2022, 2023, 2024, 2025]),
        below_90=pl.col("estimated_mmr_cov") <= 90,
        below_91=pl.col("estimated_mmr_cov") <= 91,
        below_92=pl.col("estimated_mmr_cov") <= 92,
        below_93=pl.col("estimated_mmr_cov") <= 93,
        below_94=pl.col("estimated_mmr_cov") <= 94,
        below_95=pl.col("estimated_mmr_cov") <= 95,
        below_96=pl.col("estimated_mmr_cov") <= 96,
        below_97=pl.col("estimated_mmr_cov") <= 97,
        below_98=pl.col("estimated_mmr_cov") <= 98,
        below_99=pl.col("estimated_mmr_cov") <= 99,
        below_100=pl.col("estimated_mmr_cov") <= 100,
        cases_10=pl.col("cases_outbreak") >= 10,
        cases_5=pl.col("cases_outbreak") >= 5,
        cases_3=pl.col("cases_outbreak") >= 3,
    )

    # get the coverages across the two time periods
    coverage_old = (
        df.filter(pl.col("recent") == False)
        .select("estimated_mmr_cov")
        .to_numpy()
    )
    coverage_new = (
        df.filter(pl.col("recent") == True)
        .select("estimated_mmr_cov")
        .to_numpy()
    )

    df = df.with_columns(
        total_10=pl.col("cases_10").sum().over("recent"),
        total_5=pl.col("cases_5").sum().over("recent"),
        total_3=pl.col("cases_3").sum().over("recent"),
    )

    df_identification = pl.DataFrame()
    for coverage in range(90, 101):
        df_temp = (
            df.group_by(["recent", f"below_{coverage}"])
            .agg(
                identified_10=pl.sum("cases_10") / pl.col("total_10").first(),
                identified_5=pl.sum("cases_5") / pl.col("total_5").first(),
                identified_3=pl.sum("cases_3") / pl.col("total_3").first(),
            )
            .filter(pl.col(f"below_{coverage}") == True)
            .with_columns(coverage=pl.lit(coverage))
        )
        df_identification = pl.concat(
            [df_identification, df_temp], how="diagonal_relaxed"
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
        label="≥10 cases, 2011-2020",
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
        label="≥10 cases, 2021-2025",
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
        label="≥5 cases, 2011-2020",
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
        label="≥5 cases, 2021-2025",
        color="green",
        linestyle="--",
    )

    # add a cdf of the coverage distribution across all county-years in the two time periods
    # add a second axis
    # min_coverage = min(coverage_old.min(), coverage_new.min())
    # ax2 = ax.twinx()
    # coverage_old_hist = np.histogram(
    #     coverage_old, range=(min_coverage, 100), bins=200, density=True
    # )
    # coverage_new_hist = np.histogram(
    #     coverage_new, range=(min_coverage, 100), bins=200, density=True
    # )

    # ax2.bar(
    #     coverage_old_hist[1][:-1],
    #     np.cumsum(coverage_old_hist[0] / coverage_old_hist[0].sum()) * 100,
    #     width=np.diff(coverage_old_hist[1]),
    #     color="blue",
    #     alpha=0.2,
    # )
    # ax2.bar(
    #     coverage_new_hist[1][:-1],
    #     np.cumsum(coverage_new_hist[0] / coverage_new_hist[0].sum()) * 100,
    #     width=np.diff(coverage_new_hist[1]),
    #     color="green",
    #     alpha=0.2,
    # )

    # # set some x axis limits
    # ax2.set_xlim(89.7, 100.0)
    # ax2.set_ylim(0, 101)
    # # set ticks at every two integer coverage values
    # ax2.set_xticks(range(90, 102, 2))
    # # make ax2 look good
    # ax2.set_ylabel(
    #     "County-years with MMR vaccine\n" + r"coverage less than $x$ (%)",
    #     rotation=-90,
    #     labelpad=45,
    # )
    # ax2.spines["top"].set_visible(False)
    # ax2.spines["left"].set_visible(False)
    # ax2.get_xaxis().tick_bottom()
    # ax2.get_yaxis().tick_right()
    # ax2.tick_params(axis="x", direction="out")
    # ax2.tick_params(axis="y", direction="out")
    # # offset the spines
    # for spine in ax2.spines.values():
    #     spine.set_position(("outward", 5))
    # # put the grid behind
    # ax2.set_axisbelow(True)

    ax.legend(loc=(0.3, 0.0), labelspacing=0.2, edgecolor="black")
    ax.set_xlabel("MMR vaccine coverage (%)")
    ax.set_ylabel("Outbreaks below vaccine threshold (%)")
    ax.set_ylim(0, 101)
    ax.set_xlim(89.7, 100.2)
    ax.set_xticks(range(90, 101, 2))

    return df_identification, fig, ax


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
                    f"samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_{model_name}_{year}.pkl",
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
            print(
                f"{model_name} median effects: {model_effects['median_coverage_effect'].to_list()}"
            )
            print(
                f"{model_name} 95% CI: {list(zip(model_effects['lower_coverage_effect'].to_list(), model_effects['upper_coverage_effect'].to_list()))}"
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
        "% change in outbreak size per 1%\nincrease in MMR vaccine coverage"
    )
    ax.legend(
        loc=(0.25, 0.70),
        edgecolor="black",
        title="Model:                                         ",
    )

    return fig, ax


def plot_effect_over_time(
    samples_16_17, samples_2025, df_16_17, df_2025, years_in_order, fig, ax
):
    def compute_year_effects(samples, df):
        samples = samples["effects_over_time_and_group"][:, :, 0]
        std_wh = df["estimated_mmr_cov"].to_numpy().std()
        samples_wh = samples / std_wh
        # exponentiate and multiply by 100
        samples_wh = jnp.exp(samples_wh) * 100
        # to express as percent change
        samples_wh = samples_wh - 100

        # collect the values by years
        median_effect = np.array([])
        lower_effect = np.array([])
        upper_effect = np.array([])
        years_all_out = range(
            int(min(years_in_order)), int(max(years_in_order)) + 1
        )
        for year in years_all_out:
            mask = df["year"].is_in(year).to_jax()
            year_samples = samples_wh[:, mask]
            # simple average
            year_samples = year_samples.mean(
                axis=1
            )  # average across county-years in each sample
            median_effect = np.append(
                median_effect, float(jnp.median(year_samples, axis=0))
            )
            lower_effect = np.append(
                lower_effect, float(jnp.quantile(year_samples, 0.025, axis=0))
            )
            upper_effect = np.append(
                upper_effect, float(jnp.quantile(year_samples, 0.975, axis=0))
            )

        return median_effect, lower_effect, upper_effect

    years_all_out = range(
        int(min(years_in_order)), int(max(years_in_order)) + 1
    )

    median_16_17, lower_16_17, upper_16_17 = compute_year_effects(
        samples_16_17, df_16_17
    )
    median_2025, lower_2025, upper_2025 = compute_year_effects(
        samples_2025, df_2025
    )

    # average the year-level estimates across the two variants
    median_effect = (median_16_17 + median_2025) / 2
    lower_effect = (lower_16_17 + lower_2025) / 2
    upper_effect = (upper_16_17 + upper_2025) / 2

    n = 3  # smoothing window size
    median_effect_expanded = jnp.convolve(
        jnp.pad(median_effect, (n - 1, 0)), jnp.ones(n) / n, mode="valid"
    )
    lower_effect_expanded = jnp.convolve(
        jnp.pad(lower_effect, (n - 1, 0)), jnp.ones(n) / n, mode="valid"
    )
    upper_effect_expanded = jnp.convolve(
        jnp.pad(upper_effect, (n - 1, 0)), jnp.ones(n) / n, mode="valid"
    )

    ax.fill_between(
        years_all_out,
        lower_effect_expanded,
        upper_effect_expanded,
        color="grey",
        alpha=0.3,
    )
    ax.plot(years_all_out, median_effect_expanded, color="grey")
    # horizontal line at y=100
    ax.axhline(0, ls="--", color="black")

    ax.set_xlabel("Year")
    ax.set_ylabel(
        "% change in outbreak size per 1%\nincrease in MMR vaccine coverage"
    )
    ax.set_xticks([2011, 2015, 2020, 2025])

    return fig, ax


if __name__ == "__main__":
    df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
    )

    fig, ax = setup_plot(2, 2)  # type: ignore
    fig_cov, ax_cov = setup_plot(1, 1)  # type: ignore

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
        95,
        fig,
        ax[0, 0],  # type: ignore
        vaccination_metric="estimated_mmr_cov",
    )
    plot_coverage_distribution_all_years(
        df,
        95,
        fig_cov,
        ax_cov,  # type: ignore
        vaccination_metric="estimated_mmr_cov",
    )
    fig_cov.savefig(
        os.path.join(
            fig_dir, "figure_coverage_distribution_outbreak_size.png"
        ),
        dpi=300,
        bbox_inches="tight",
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
            "predictive_random_walk_poisson_CAR_time_and_group_varying_2025_oddities=True_exclude_16_17.pkl",
        ),
        "rb",
    ) as file:
        predictive_samples_1617 = pickle.load(file)

    with open(
        os.path.join(
            output_dir,
            "predictive_random_walk_poisson_CAR_time_and_group_varying_2025_oddities=False_no_2025_oddities.pkl",
        ),
        "rb",
    ) as file:
        predictive_samples_2025 = pickle.load(file)

    exclude_16_17 = pl.read_csv(
        os.path.join(output_dir, "exclude_16_17.csv"),
    )
    df = df.with_columns(
        year_county_state=pl.col("year").cast(pl.String)
        + pl.col("county")
        + pl.col("state")
    )

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
        predictive_samples_1617,
        predictive_samples_2025,
        df.filter(
            ~pl.col("year_county_state").is_in(
                exclude_16_17["year_county_state"].to_list()
            )
        ),
        df.filter(~pl.col("oddities_2025")),
        list([2011, 2012.5, 2014.5, 2016.5, 2018.5, 2020.5, 2022.5, 2025]),
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
