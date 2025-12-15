import os
import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import polars as pl
import yaml
from matplotlib import rc
from scipy.stats import ks_2samp

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


def plot_performance_over_time(df, fig, ax):
    year_buckets = [
        [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
        [2021, 2022, 2023, 2024, 2025],
    ]
    models = ["MMR coverage", "MMR coverage + exemptions", "All covariates"]
    assembled_performance = pl.DataFrame({"years": year_buckets}).join(
        pl.DataFrame({"model": models}), how="cross"
    )
    assembled_performance = assembled_performance.with_columns(
        year_range=pl.col("years").map_elements(
            lambda x: f"{min(x)}-{max(x)}", return_dtype=pl.String
        ),
        median_R2=pl.lit(None),
        lower_R2=pl.lit(None),
        upper_R2=pl.lit(None),
        mean_R2=pl.lit(None),
        median_mae=pl.lit(None),
        lower_mae=pl.lit(None),
        upper_mae=pl.lit(None),
        mean_mae=pl.lit(None),
        median_rmse=pl.lit(None),
        lower_rmse=pl.lit(None),
        upper_rmse=pl.lit(None),
        mean_rmse=pl.lit(None),
        median_fold_error=pl.lit(None),
        lower_fold_error=pl.lit(None),
        upper_fold_error=pl.lit(None),
        mean_fold_error=pl.lit(None),
        median_percent_outbreaks_5=pl.lit(None),
        lower_percent_outbreaks_5=pl.lit(None),
        upper_percent_outbreaks_5=pl.lit(None),
        mean_percent_outbreaks_5=pl.lit(None),
        median_percent_outbreaks_10=pl.lit(None),
        lower_percent_outbreaks_10=pl.lit(None),
        upper_percent_outbreaks_10=pl.lit(None),
        mean_percent_outbreaks_10=pl.lit(None),
        median_residual=pl.lit(None),
        lower_residual=pl.lit(None),
        upper_residual=pl.lit(None),
        mean_residual=pl.lit(None),
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
                    f"predictive_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_{model_name}_{year}.pkl",
                ),
                "rb",
            ) as file:
                predictive_samples = pickle.load(file)

            cases = predictive_samples["cases"]
            observed = df.filter(pl.col("year").is_in(year))[
                "cases_outbreak"
            ].to_jax()
            R2 = 100 * jnp.array(
                [jnp.corrcoef(observed, c)[0, 1] ** 2 for c in cases]
            )

            median_R2 = float(jnp.median(R2, axis=0))
            lower_R2 = float(jnp.quantile(R2, 0.025, axis=0))
            upper_R2 = float(jnp.quantile(R2, 0.975, axis=0))
            mean_R2 = float(jnp.mean(R2, axis=0))

            # while we're here, calculate MAE, RMSE, fold error, and percent of outbreaks (>5, >10 cases) predicted
            mae = jnp.median(jnp.abs(observed - cases), axis=1)
            median_mae = float(jnp.median(mae, axis=0))
            lower_mae = float(jnp.quantile(mae, 0.025, axis=0))
            upper_mae = float(jnp.quantile(mae, 0.975, axis=0))
            mean_mae = float(jnp.mean(mae, axis=0))

            rmse = jnp.sqrt(jnp.mean((observed - cases) ** 2, axis=1))
            median_rmse = float(jnp.median(rmse, axis=0))
            lower_rmse = float(jnp.quantile(rmse, 0.025, axis=0))
            upper_rmse = float(jnp.quantile(rmse, 0.975, axis=0))
            mean_rmse = float(jnp.mean(rmse, axis=0))

            fold_error = jnp.mean(
                jnp.exp(jnp.abs(jnp.log((observed + 1) / (cases + 1)))), axis=1
            )
            median_fold_error = float(jnp.median(fold_error, axis=0))
            lower_fold_error = float(jnp.quantile(fold_error, 0.025, axis=0))
            upper_fold_error = float(jnp.quantile(fold_error, 0.975, axis=0))
            mean_fold_error = float(jnp.mean(fold_error, axis=0))

            percent_outbreaks_5 = jnp.mean(
                ~jnp.logical_xor(cases >= 5, observed >= 5), axis=1
            )
            median_percent_outbreaks_5 = float(
                jnp.median(percent_outbreaks_5, axis=0)
            )
            lower_percent_outbreaks_5 = float(
                jnp.quantile(percent_outbreaks_5, 0.025, axis=0)
            )
            upper_percent_outbreaks_5 = float(
                jnp.quantile(percent_outbreaks_5, 0.975, axis=0)
            )
            mean_percent_outbreaks_5 = float(
                jnp.mean(percent_outbreaks_5, axis=0)
            )
            percent_outbreaks_10 = jnp.mean(
                ~jnp.logical_xor(cases >= 10, observed >= 10), axis=1
            )
            median_percent_outbreaks_10 = float(
                jnp.median(percent_outbreaks_10, axis=0)
            )
            lower_percent_outbreaks_10 = float(
                jnp.quantile(percent_outbreaks_10, 0.025, axis=0)
            )
            upper_percent_outbreaks_10 = float(
                jnp.quantile(percent_outbreaks_10, 0.95, axis=0)
            )
            mean_percent_outbreaks_10 = float(
                jnp.mean(percent_outbreaks_10, axis=0)
            )

            residuals = observed - jnp.median(cases, axis=0)
            median_residual = float(jnp.median(residuals))
            lower_residual = float(jnp.quantile(residuals, 0.0))
            upper_residual = float(jnp.quantile(residuals, 1.0))
            mean_residual = float(jnp.mean(residuals))

            assembled_performance = assembled_performance.with_columns(
                median_R2=pl.when(
                    (pl.col("years") == year) & (pl.col("model") == model_name)
                )
                .then(median_R2)
                .otherwise(pl.col("median_R2")),
                lower_R2=pl.when(
                    (pl.col("years") == year) & (pl.col("model") == model_name)
                )
                .then(lower_R2)
                .otherwise(pl.col("lower_R2")),
                upper_R2=pl.when(
                    (pl.col("years") == year) & (pl.col("model") == model_name)
                )
                .then(upper_R2)
                .otherwise(pl.col("upper_R2")),
                mean_R2=pl.when(
                    (pl.col("years") == year) & (pl.col("model") == model_name)
                )
                .then(mean_R2)
                .otherwise(pl.col("mean_R2")),
                median_mae=pl.when(
                    (pl.col("years") == year) & (pl.col("model") == model_name)
                )
                .then(median_mae)
                .otherwise(pl.col("median_mae")),
                lower_mae=pl.when(
                    (pl.col("years") == year) & (pl.col("model") == model_name)
                )
                .then(lower_mae)
                .otherwise(pl.col("lower_mae")),
                upper_mae=pl.when(
                    (pl.col("years") == year) & (pl.col("model") == model_name)
                )
                .then(upper_mae)
                .otherwise(pl.col("upper_mae")),
                mean_mae=pl.when(
                    (pl.col("years") == year) & (pl.col("model") == model_name)
                )
                .then(mean_mae)
                .otherwise(pl.col("mean_mae")),
                median_rmse=pl.when(
                    (pl.col("years") == year) & (pl.col("model") == model_name)
                )
                .then(median_rmse)
                .otherwise(pl.col("median_rmse")),
                lower_rmse=pl.when(
                    (pl.col("years") == year) & (pl.col("model") == model_name)
                )
                .then(lower_rmse)
                .otherwise(pl.col("lower_rmse")),
                upper_rmse=pl.when(
                    (pl.col("years") == year) & (pl.col("model") == model_name)
                )
                .then(upper_rmse)
                .otherwise(pl.col("upper_rmse")),
                mean_rmse=pl.when(
                    (pl.col("years") == year) & (pl.col("model") == model_name)
                )
                .then(mean_rmse)
                .otherwise(pl.col("mean_rmse")),
                median_fold_error=pl.when(
                    (pl.col("years") == year) & (pl.col("model") == model_name)
                )
                .then(median_fold_error)
                .otherwise(pl.col("median_fold_error")),
                lower_fold_error=pl.when(
                    (pl.col("years") == year) & (pl.col("model") == model_name)
                )
                .then(lower_fold_error)
                .otherwise(pl.col("lower_fold_error")),
                upper_fold_error=pl.when(
                    (pl.col("years") == year) & (pl.col("model") == model_name)
                )
                .then(upper_fold_error)
                .otherwise(pl.col("upper_fold_error")),
                mean_fold_error=pl.when(
                    (pl.col("years") == year) & (pl.col("model") == model_name)
                )
                .then(mean_fold_error)
                .otherwise(pl.col("mean_fold_error")),
                median_percent_outbreaks_5=pl.when(
                    (pl.col("years") == year) & (pl.col("model") == model_name)
                )
                .then(median_percent_outbreaks_5)
                .otherwise(pl.col("median_percent_outbreaks_5")),
                lower_percent_outbreaks_5=pl.when(
                    (pl.col("years") == year) & (pl.col("model") == model_name)
                )
                .then(lower_percent_outbreaks_5)
                .otherwise(pl.col("lower_percent_outbreaks_5")),
                upper_percent_outbreaks_5=pl.when(
                    (pl.col("years") == year) & (pl.col("model") == model_name)
                )
                .then(upper_percent_outbreaks_5)
                .otherwise(pl.col("upper_percent_outbreaks_5")),
                mean_percent_outbreaks_5=pl.when(
                    (pl.col("years") == year) & (pl.col("model") == model_name)
                )
                .then(mean_percent_outbreaks_5)
                .otherwise(pl.col("mean_percent_outbreaks_5")),
                median_percent_outbreaks_10=pl.when(
                    (pl.col("years") == year) & (pl.col("model") == model_name)
                )
                .then(median_percent_outbreaks_10)
                .otherwise(pl.col("median_percent_outbreaks_10")),
                lower_percent_outbreaks_10=pl.when(
                    (pl.col("years") == year) & (pl.col("model") == model_name)
                )
                .then(lower_percent_outbreaks_10)
                .otherwise(pl.col("lower_percent_outbreaks_10")),
                upper_percent_outbreaks_10=pl.when(
                    (pl.col("years") == year) & (pl.col("model") == model_name)
                )
                .then(upper_percent_outbreaks_10)
                .otherwise(pl.col("upper_percent_outbreaks_10")),
                mean_percent_outbreaks_10=pl.when(
                    (pl.col("years") == year) & (pl.col("model") == model_name)
                )
                .then(mean_percent_outbreaks_10)
                .otherwise(pl.col("mean_percent_outbreaks_10")),
                median_residual=pl.when(
                    (pl.col("years") == year) & (pl.col("model") == model_name)
                )
                .then(median_residual)
                .otherwise(pl.col("median_residual")),
                lower_residual=pl.when(
                    (pl.col("years") == year) & (pl.col("model") == model_name)
                )
                .then(lower_residual)
                .otherwise(pl.col("lower_residual")),
                upper_residual=pl.when(
                    (pl.col("years") == year) & (pl.col("model") == model_name)
                )
                .then(upper_residual)
                .otherwise(pl.col("upper_residual")),
                mean_residual=pl.when(
                    (pl.col("years") == year) & (pl.col("model") == model_name)
                )
                .then(mean_residual)
                .otherwise(pl.col("mean_residual")),
            )

    colors = {
        "MMR coverage": "red",
        "MMR coverage + exemptions": "purple",
        "All covariates": "orange",
    }

    for model_name in models:
        model_effects = assembled_performance.filter(
            pl.col("model") == model_name
        ).sort("year_index")
        ax.errorbar(
            ["2011-2020", "2021-2025"],
            model_effects["median_residual"].to_list(),
            yerr=[
                model_effects["median_residual"].to_numpy()
                - model_effects["lower_residual"].to_numpy(),
                model_effects["upper_residual"].to_numpy()
                - model_effects["median_residual"].to_numpy(),
            ],
            fmt="o",
            label=model_name,
            color=colors[model_name],
        )

        # scatter plot the mean with a "*" as its marker
        # ax.scatter(
        #     ["2011-2020", "2021-2025"],
        #     model_effects["mean_rmse"].to_list(),
        #     marker="*",
        #     s=12,
        #     color=colors[model_name],
        # )

    ax.set_xlim([-0.25, 1.25])

    ax.set_xlabel("Year")
    ax.set_ylabel("Residuals (observed - predicted cases)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc=(0.2, 0.7))

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
        alpha=0.3,
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
        alpha=0.3,
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

    ax.legend(loc="upper left")

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
    ax.text(
        0.00008 * max(df["cases_outbreak"].max(), upper.max()),
        0.05 * max(df["cases_outbreak"].max(), upper.max()),
        f"$R^2$ = {R2:.2f}%",
        fontsize=26,
        color="green",
    )

    return fig, ax


def plot_residuals(df1, fig, ax):
    with open(
        os.path.join(
            output_dir,
            f"predictive_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_[2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020].pkl",
        ),
        "rb",
    ) as file:
        predictive_samples_pre = pickle.load(file)

    with open(
        os.path.join(
            output_dir,
            f"predictive_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_[2021, 2022, 2023, 2024, 2025].pkl",
        ),
        "rb",
    ) as file:
        predictive_samples_post = pickle.load(file)

    # We're going to violin plot the residuals by buckets
    df1 = df1.with_columns(
        coverage_bucket=pl.when(pl.col("estimated_mmr_cov") < 80)
        .then(pl.lit("0"))
        .when(
            (pl.col("estimated_mmr_cov") >= 80)
            & (pl.col("estimated_mmr_cov") < 90)
        )
        .then(pl.lit("1"))
        .when(
            (pl.col("estimated_mmr_cov") >= 90)
            & (pl.col("estimated_mmr_cov") < 95)
        )
        .then(pl.lit("3"))
        .otherwise(pl.lit("4")),
        # case size buckets -- 0, 1-3, 4-9, 10+
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
    )

    # concatenate the predicted samples with df
    df_pre = df1.filter(
        pl.col("year").is_in(
            [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
        )
    )
    df_post = df1.filter(pl.col("year").is_in([2021, 2022, 2023, 2024, 2025]))

    df_pre = df_pre.with_columns(
        samples=pl.Series(predictive_samples_pre["cases"].__array__().T),
        predicted_cases=pl.Series(
            jnp.median(predictive_samples_pre["cases"], axis=0).__array__()
        ),
        period=pl.lit("0"),
    )

    df_post = df_post.with_columns(
        samples=pl.Series(predictive_samples_post["cases"].__array__().T),
        predicted_cases=pl.Series(
            jnp.median(predictive_samples_post["cases"], axis=0).__array__()
        ),
        period=pl.lit("1"),
    )

    df = pl.concat([df_pre, df_post])
    df = df.with_columns(
        residuals=pl.col("cases_outbreak") - pl.col("predicted_cases"),
        residuals_samples=pl.col("cases_outbreak") - pl.col("samples"),
    )

    # calculate rmse
    rmse_samples = []
    for case_size_bucket in ["0", "1", "2", "3"]:
        for period in ["0", "1"]:
            subset = df.filter(
                (pl.col("case_size_bucket") == case_size_bucket)
                & (pl.col("period") == period)
            )
            samples = jnp.array(subset["residuals_samples"])
            observed = jnp.array(subset["cases_outbreak"])
            rmse = jnp.sqrt(
                jnp.mean((observed[:, None] - samples) ** 2, axis=1)
            )
            rmse_samples.append(
                {
                    "case_size_bucket": case_size_bucket,
                    "period": period,
                    "rmse_samples": rmse,
                }
            )

    df = (
        df.group_by(["case_size_bucket", "period"])
        .agg(
            residuals=pl.col("residuals"),
            median=pl.col("residuals").median(),
            lower=pl.col("residuals").quantile(0.25),
            upper=pl.col("residuals").quantile(0.75),
            avg=pl.col("residuals").mean(),
            # aggregate all the residuals together
            residuals_samples=pl.col("residuals_samples"),
        )
        .sort(["case_size_bucket", "period"])
    )

    df = df.with_columns(
        residuals_samples_flattened=pl.col("residuals_samples").map_elements(
            lambda x: np.array(x).flatten(), return_dtype=pl.List(pl.Int64)
        )
    )

    df = df.with_columns(
        median_sample=pl.col("residuals_samples_flattened").map_elements(
            lambda x: np.median(x), return_dtype=pl.Float64
        ),
        lower_sample=pl.col("residuals_samples_flattened").map_elements(
            lambda x: np.quantile(x, 0.25), return_dtype=pl.Float64
        ),
        upper_sample=pl.col("residuals_samples_flattened").map_elements(
            lambda x: np.quantile(x, 0.75), return_dtype=pl.Float64
        ),
        avg_sample=pl.col("residuals_samples_flattened").map_elements(
            lambda x: x.mean(), return_dtype=pl.Float64
        ),
    )

    # concatenate on the rmse samples
    df = df.join(pl.DataFrame(rmse_samples), on=["case_size_bucket", "period"])

    # calculate rmse statistics
    df = df.with_columns(
        median_rmse=pl.col("rmse_samples").map_elements(
            lambda x: np.median(x), return_dtype=pl.Float64
        ),
        lower_rmse=pl.col("rmse_samples").map_elements(
            lambda x: np.quantile(x, 0.25), return_dtype=pl.Float64
        ),
        upper_rmse=pl.col("rmse_samples").map_elements(
            lambda x: np.quantile(x, 0.75), return_dtype=pl.Float64
        ),
        avg_rmse=pl.col("rmse_samples").map_elements(
            lambda x: x.mean(), return_dtype=pl.Float64
        ),
    )

    # assemble the data for violin plotting

    data_pre = (
        df.filter(pl.col("period") == "0").to_dict()["residuals"].to_list()
    )
    data_post = (
        df.filter(pl.col("period") == "1").to_dict()["residuals"].to_list()
    )
    medians_pre = (
        df.filter(pl.col("period") == "0").to_dict()["median"].to_numpy()
    )
    medians_post = (
        df.filter(pl.col("period") == "1").to_dict()["median"].to_numpy()
    )
    lowers_pre = (
        df.filter(pl.col("period") == "0").to_dict()["lower"].to_numpy()
    )
    lowers_post = (
        df.filter(pl.col("period") == "1").to_dict()["lower"].to_numpy()
    )
    uppers_pre = (
        df.filter(pl.col("period") == "0").to_dict()["upper"].to_numpy()
    )
    uppers_post = (
        df.filter(pl.col("period") == "1").to_dict()["upper"].to_numpy()
    )
    means_pre = df.filter(pl.col("period") == "0").to_dict()["avg"].to_list()
    means_post = df.filter(pl.col("period") == "1").to_dict()["avg"].to_list()

    ax.axhline(0, ls="--", color="k")

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
            "10+",
        ]
    )

    ax.legend(
        [violinparts1["bodies"][0], violinparts2["bodies"][0]],
        ["2011-2020", "2021-2025"],
    )

    ax.set_xlabel("Reported cases (N)")
    ax.set_ylabel("Residuals (reported - predicted cases)")

    # ax.set_yscale("symlog")

    return fig, ax


def plot_ecological_determinants(samples, df, significant_only, fig, ax):
    pooled_covariates = ["MMR coverage", "NME", "ME"]
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
        "NME": "nonmed_exempt_pct",
        "ME": "med_exempt_pct",
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

    if significant_only:
        pooled_covariates = ["MMR coverage", "NME", "ME"]
        covariates = [
            "% < 5yo",
            "% white",
            "% graduate degree",
            "Income (10,000 2024 USD)",
        ]

    all_covariates = [*pooled_covariates, *covariates]
    median_effects = []
    lower_effects = []
    upper_effects = []

    for idx, covariate in enumerate(pooled_covariates):
        samples_of_interest = samples["iv_global"][:, idx]
        # scale by standard deviation
        std = df[covariate_name_map[covariate]].to_numpy().std()
        samples_of_interest = samples_of_interest / std
        # exponentiate and multiply by 100
        samples_of_interest = jnp.exp(samples_of_interest) * 100
        # to express as percent change
        samples_of_interest = samples_of_interest - 100
        median_effects.append(jnp.median(samples_of_interest, axis=0))
        lower_effects.append(jnp.quantile(samples_of_interest, 0.025, axis=0))
        upper_effects.append(jnp.quantile(samples_of_interest, 0.975, axis=0))

    # same process for unpooled covariates
    for covariate in covariates:
        # index is the number of this thing in the dict
        idx = list(covariate_name_map.keys()).index(covariate)
        idx = idx - 3
        samples_of_interest = samples["regression_betas"][:, idx]
        # scale by standard deviation
        std = df[covariate_name_map[covariate]].to_numpy().std()
        if "%" in covariate:
            std = std * 100  # because we do the modeling as a fraction
        samples_of_interest = samples_of_interest / std
        if "Income" in covariate:
            # try to convert this to an income instead of log income
            mean_log_income = (
                df[covariate_name_map[covariate]].to_numpy().mean()
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
        lower_effects.append(jnp.quantile(samples_of_interest, 0.025, axis=0))
        upper_effects.append(jnp.quantile(samples_of_interest, 0.975, axis=0))

    ax.errorbar(
        x=range(len(all_covariates)),
        y=median_effects,
        yerr=[
            jnp.array(median_effects) - jnp.array(lower_effects),
            jnp.array(upper_effects) - jnp.array(median_effects),
        ],
        fmt="o",
        ecolor="k",
        c="k",
        capsize=0,
        alpha=0.7,
    )

    # line at 0%
    ax.axhline(0, ls="--", color="k")

    ax.set_xticks(range(len(all_covariates)))
    # at a 45 degree angle
    ax.set_xticklabels(all_covariates, rotation=-60, ha="left")
    ax.set_xlabel("Predictor")
    ax.set_ylabel(
        "% change in incidence rate per\n1 unit increase in predictor"
    )
    if not significant_only:
        ax.set_yscale("symlog")
    else:
        ax.set_ylim(-100)

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
        pl.mean("estimated_mmr_cov"),
        pl.min("cases_outbreak"),
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
    )

    df = (
        df.group_by(["case_size_bucket"])
        .agg(
            p_value=pl.col("p_value"),
            median=pl.col("p_value").median(),
            lower=pl.col("p_value").quantile(0.25),
            upper=pl.col("p_value").quantile(0.75),
            avg=pl.col("p_value").mean(),
        )
        .sort(["case_size_bucket"])
    )

    # assemble the data for violin plotting

    data = df.to_dict()["p_value"].to_list()

    medians = df.to_dict()["median"].to_numpy()

    lowers = df.to_dict()["lower"].to_numpy()

    uppers = df.to_dict()["upper"].to_numpy()

    means = df.to_dict()["avg"].to_list()

    # boxplot and swarmplot of coverage distributions
    violinparts1 = ax.violinplot(
        data,
        positions=[i for i in range(len(data))],
        widths=0.6,
        showextrema=False,
    )

    for pc in violinparts1["bodies"]:
        pc.set(facecolor="grey")

    ax.errorbar(
        x=[i for i in range(len(medians))],
        y=medians,
        yerr=[medians - lowers, uppers - medians],
        fmt="o",
        capsize=5,
        color="grey",
    )

    # add means as stars
    ax.scatter(
        [i for i in range(len(means))], means, marker="*", s=90, color="grey"
    )

    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(
        [
            "0",
            "1-3",
            "4-9",
            "10+",
        ]
    )

    ax.set_xlabel("Reported cases (N)")
    ax.set_ylabel("Spatial CAR effect empirical p-value")
    ax.set_ylim(0)

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
        roc_buckets=pl.when(pl.col("year_over_year_change") <= -2)
        .then(pl.lit(0))
        .when(
            (pl.col("year_over_year_change") > -2)
            & (pl.col("year_over_year_change") <= 2)
        )
        .then(pl.lit(1))
        .when((pl.col("year_over_year_change") > 2))
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

    # Print out ks test statistics for whether five year and one year differ
    for bucket in df["roc_buckets"].unique():
        one_year_samples = df.filter(pl.col("roc_buckets") == bucket)[
            "one_year_residuals"
        ].to_numpy()[0]
        five_year_samples = df.filter(pl.col("roc_buckets") == bucket)[
            "five_year_residuals"
        ].to_numpy()[0]

        # ks test assuming that the one year samples are smaller
        ks_result = ks_2samp(
            one_year_samples, five_year_samples, alternative="greater"
        )

        print(f"Vaccination ROC bucket {bucket} has KS result {ks_result}.")

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
        capsize=5,
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
        capsize=5,
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
            "≤-2%",
            "-2% to 2%",
            ">2%",
        ],
        rotation=-60,
        ha="left",
    )

    ax.legend(
        [violinparts1["bodies"][0], violinparts2["bodies"][0]],
        ["One year", "Five year"],
    )

    ax.set_xlabel("Year-to-year change in MMR coverage (%)")
    ax.set_ylabel("Residuals (reported - predicted cases)")

    ax.set_yticks([-10, -5, 0, 5, 10, 15])

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
            "predictive_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_[2021, 2022, 2023, 2024, 2025].pkl",
        ),
        "rb",
    ) as file:
        predictive_samples = pickle.load(file)

    with open(
        os.path.join(output_dir, "final_outbreak_size_predictions.pkl"),
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
        df.filter(pl.col("year").is_in([2021, 2022, 2023, 2024, 2025])),
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
    plot_residuals(
        df,
        fig,
        ax[0, 1],  # type: ignore
    )

    with open(
        os.path.join(
            output_dir,
            "samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_[2021, 2022, 2023, 2024, 2025].pkl",
        ),
        "rb",
    ) as file:
        posterior_samples = pickle.load(file)

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
    plot_spatial_importance(
        posterior_samples,
        df.filter(pl.col("year") >= 2021),
        adjacency_matrix,
        fig,
        ax[1, 0],
    )  # type: ignore

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
    plot_5yr_performance_comparison(df, fig, ax[1, 1])  # type: ignore

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
