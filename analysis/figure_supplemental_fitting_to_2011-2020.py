import os
import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
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
    fig, ax = plt.subplots(nrows, ncols, figsize=(20, 20))
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


def plot_predictions(df, samples, naive_model, fig, ax):
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
        label="Mechanistic: MMR coverage",
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
        label="Statistical: All covariates",
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

    ax.legend(loc=(0.0, 0.83), edgecolor="black")

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

    # add a text annotation of the R2
    R2 = jnp.corrcoef(df["cases_outbreak"].to_jax(), median)[0, 1] ** 2
    R2 = R2 * 100  # as a percentage
    # ax.text(
    #     0.00005 * max(df["cases_outbreak"].max(), upper.max()),
    #     0.08 * max(df["cases_outbreak"].max(), upper.max()),
    #     f"$R^2$ = {R2:.2f}%",
    #     fontsize=26,
    #     color="green",
    # )
    ax.set_ylim(-0.05)

    return fig, ax


def plot_performance_comparison(df, fig, ax):
    df = df.with_columns(
        outbreak_size_bucket=pl.when(pl.col("cases_outbreak") == 0)
        .then(0)
        .when(pl.col("cases_outbreak") <= 3)
        .then(1)
        .when(pl.col("cases_outbreak") < 10)
        .then(2)
        .otherwise(3)
    )

    df_new = df.filter(pl.col("year") >= 2021)
    df_old = df.filter(pl.col("year") <= 2020)

    with open(
        os.path.join(
            output_dir,
            f"predictive_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_[2021, 2022, 2023, 2024, 2025].pkl",
        ),
        "rb",
    ) as file:
        new = pickle.load(file)

    with open(
        os.path.join(
            output_dir,
            f"predictive_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_[2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020].pkl",
        ),
        "rb",
    ) as file:
        old = pickle.load(file)

    df_new = df_new.with_columns(
        predicted=pl.col("cases_outbreak")
        - pl.Series(jnp.median(new["cases"], axis=0).__array__()),
    )
    df_old = df_old.with_columns(
        predicted=pl.col("cases_outbreak")
        - pl.Series(jnp.median(old["cases"], axis=0).__array__()),
    )

    df_new = df_new.group_by("outbreak_size_bucket", maintain_order=True).agg(
        values=pl.col("predicted"),
        median=pl.col("predicted").median(),
        lower=pl.col("predicted").quantile(0.25),
        upper=pl.col("predicted").quantile(0.75),
        mean=pl.col("predicted").mean(),
    )

    df_old = df_old.group_by("outbreak_size_bucket", maintain_order=True).agg(
        values=pl.col("predicted"),
        median=pl.col("predicted").median(),
        lower=pl.col("predicted").quantile(0.25),
        upper=pl.col("predicted").quantile(0.75),
        mean=pl.col("predicted").mean(),
    )

    data_old = df_old.to_dict()["values"].to_list()
    data_new = df_new.to_dict()["values"].to_list()

    medians_old = df_old.to_dict()["median"].to_numpy()
    medians_new = df_new.to_dict()["median"].to_numpy()
    lowers_old = df_old.to_dict()["lower"].to_numpy()
    lowers_new = df_new.to_dict()["lower"].to_numpy()
    uppers_old = df_old.to_dict()["upper"].to_numpy()
    uppers_new = df_new.to_dict()["upper"].to_numpy()
    means_old = df_old.to_dict()["mean"].to_list()
    means_new = df_new.to_dict()["mean"].to_list()

    ax.axhline(0, ls="--", color="k")

    offset = 0.2
    violinparts1 = ax.violinplot(
        data_old,
        positions=[i - offset for i in range(len(data_old))],
        widths=0.3,
        showextrema=False,
    )

    for pc in violinparts1["bodies"]:
        pc.set(facecolor="blue")

    ax.errorbar(
        x=[i - offset for i in range(len(medians_old))],
        y=medians_old,
        yerr=[
            medians_old - lowers_old,
            uppers_old - medians_old,
        ],
        fmt="o",
        capsize=10,
        capthick=2,
        color="blue",
    )

    # add means as stars
    ax.scatter(
        [i - offset for i in range(len(means_old))],
        means_old,
        marker="*",
        s=90,
        color="blue",
    )

    violinparts2 = ax.violinplot(
        data_new,
        positions=[i + offset for i in range(len(data_new))],
        widths=0.3,
        showextrema=False,
    )

    for pc in violinparts2["bodies"]:
        pc.set(facecolor="green")

    ax.errorbar(
        x=[i + offset for i in range(len(medians_new))],
        y=medians_new,
        yerr=[
            medians_new - lowers_new,
            uppers_new - medians_new,
        ],
        fmt="o",
        capsize=10,
        capthick=2,
        color="green",
    )

    # add means as stars
    ax.scatter(
        [i + offset for i in range(len(means_new))],
        means_new,
        marker="*",
        s=90,
        color="green",
    )

    ax.set_xticks(range(len(data_old)))
    ax.set_xticklabels(
        ["0", "1-3", "4-9", "10+"],
    )

    ax.legend(
        [violinparts1["bodies"][0], violinparts2["bodies"][0]],
        ["2011-2020", "2021-2025"],
        loc="upper right",
        edgecolor="black",
    )

    ax.set_xlabel("Outbreak size (cases)")
    ax.set_ylabel("Residuals (reported - predicted cases)")

    ax.set_yscale("symlog")
    ax.set_yticks(
        [-10, -1, 0, 1, 10, 100], ["-10", "-1", "0", "1", "10", "100"]
    )

    # set minor y ticks
    minor_y_ticks = [
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
    ]
    ax.yaxis.set_minor_locator(tck.FixedLocator(minor_y_ticks))


def plot_5yr_performance_comparison(df, fig, ax):
    with open(
        os.path.join(
            output_dir,
            f"predictive_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_[2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020].pkl",
        ),
        "rb",
    ) as file:
        one_year = pickle.load(file)

    with open(
        os.path.join(
            output_dir,
            f"predictive_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=True_CAR=True_All covariates_[2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020].pkl",
        ),
        "rb",
    ) as file:
        five_year = pickle.load(file)

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
    )

    ax.set_xlabel("Year-to-year change in MMR coverage (%)")
    ax.set_ylabel("Residuals (reported - predicted cases)")

    # ax.set_yticks([-10, 0, 10, 20])

    return fig, ax


def plot_ecological_determinants(samples_pre, samples_post, df, fig, ax):
    pooled_covariates = [
        "MMR coverage",
        "Nonmedical exemptions",
        "Medical exemptions",
    ]
    covariates = [
        "% < 5yo",
        "% school aged",
        "Average household size",
        "Standard deviation of household size",
        "% public schooled",
        "% white",
        "% graduate degree",
        "Population density (per km²)",
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
            if "Population density" in covariate:
                # convert from per km² to 1,000 per km²
                samples_of_interest = samples_of_interest * 1000
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
            # print the covariate and its coefficient
            print(
                f"{covariate} ({years[0]}-{years[-1]}): {median_effects[-1]:.2f}% change in incidence rate per 1 unit increase in predictor (95% CI: {lower_effects[-1]:.2f}%, {upper_effects[-1]:.2f}%)"
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

    ax.legend(loc=(0.01, 0.78), edgecolor="black")

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

    ax.set_yticks([-1000, -100, -10, 0, 10, 100, 1000, 10000, 100000])

    return fig, ax


if __name__ == "__main__":
    df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
    )

    fig, ax = setup_plot(2, 2)  # type: ignore

    # read predictive samples
    with open(
        os.path.join(
            output_dir,
            "predictive_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_[2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020].pkl",
        ),
        "rb",
    ) as file:
        predictive_samples = pickle.load(file)

    with open(
        os.path.join(
            output_dir,
            "final_outbreak_size_predictions_[2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020].pkl",
        ),
        "rb",
    ) as f:
        naive_model = pickle.load(f)

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
    plot_predictions(
        df.filter(
            pl.col("year").is_in(
                [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
            )
        ),
        predictive_samples["cases"],
        naive_model,
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
    plot_performance_comparison(
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
    plot_5yr_performance_comparison(
        df.filter(
            pl.col("year").is_in(
                [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
            )
        ),
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

    plot_ecological_determinants(
        posterior_samples_pre,
        posterior_samples,
        df,
        fig,
        ax[1, 1],  # type: ignore
    )

    fig.tight_layout()

    # save
    fig.savefig(
        os.path.join(fig_dir, "figure_supplemental_fitting_to_2011-2020.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(fig_dir, "figure_supplemental_fitting_to_2011-2020.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
