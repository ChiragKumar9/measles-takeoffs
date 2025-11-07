import os
import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt
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
    fig, ax = plt.subplots(nrows, ncols, figsize=(18, 18))
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


def plot_performance_over_time(df, fig, ax):
    year_buckets = [
        [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
        [2021, 2022, 2023, 2024, 2025],
    ]
    models = ["Just coverage", "Just vaccination measures", "All covariates"]
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
                    f"predictive_partial_pooled_poisson_five_year=False_CAR=True_{model_name}_{year}.pkl",
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
            lower_R2 = float(jnp.quantile(R2, 0.05, axis=0))
            upper_R2 = float(jnp.quantile(R2, 0.95, axis=0))
            mean_R2 = float(jnp.mean(R2, axis=0))

            # while we're here, calculate MAE, RMSE, fold error, and percent of outbreaks (>5, >10 cases) predicted
            mae = jnp.median(jnp.abs(observed - cases), axis=1)
            median_mae = float(jnp.median(mae, axis=0))
            lower_mae = float(jnp.quantile(mae, 0.05, axis=0))
            upper_mae = float(jnp.quantile(mae, 0.95, axis=0))
            mean_mae = float(jnp.mean(mae, axis=0))

            rmse = jnp.sqrt(jnp.mean((observed - cases) ** 2, axis=1))
            median_rmse = float(jnp.median(rmse, axis=0))
            lower_rmse = float(jnp.quantile(rmse, 0.05, axis=0))
            upper_rmse = float(jnp.quantile(rmse, 0.95, axis=0))
            mean_rmse = float(jnp.mean(rmse, axis=0))

            fold_error = jnp.mean(
                jnp.exp(jnp.abs(jnp.log((observed + 1) / (cases + 1)))), axis=1
            )
            median_fold_error = float(jnp.median(fold_error, axis=0))
            lower_fold_error = float(jnp.quantile(fold_error, 0.05, axis=0))
            upper_fold_error = float(jnp.quantile(fold_error, 0.95, axis=0))
            mean_fold_error = float(jnp.mean(fold_error, axis=0))

            percent_outbreaks_5 = jnp.mean(
                ~jnp.logical_xor(cases >= 5, observed >= 5), axis=1
            )
            median_percent_outbreaks_5 = float(
                jnp.median(percent_outbreaks_5, axis=0)
            )
            lower_percent_outbreaks_5 = float(
                jnp.quantile(percent_outbreaks_5, 0.05, axis=0)
            )
            upper_percent_outbreaks_5 = float(
                jnp.quantile(percent_outbreaks_5, 0.95, axis=0)
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
                jnp.quantile(percent_outbreaks_10, 0.05, axis=0)
            )
            upper_percent_outbreaks_10 = float(
                jnp.quantile(percent_outbreaks_10, 0.95, axis=0)
            )
            mean_percent_outbreaks_10 = float(
                jnp.mean(percent_outbreaks_10, axis=0)
            )

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
            )

    colors = {
        "Just coverage": "orange",
        "Just vaccination measures": "purple",
        "All covariates": "red",
    }

    for model_name in reversed(models):
        model_effects = assembled_performance.filter(
            pl.col("model") == model_name
        ).sort("year_index")
        ax.errorbar(
            ["2011-2020", "2021-2025"],
            model_effects["median_fold_error"].to_list(),
            yerr=[
                model_effects["median_fold_error"].to_numpy()
                - model_effects["lower_fold_error"].to_numpy(),
                model_effects["upper_fold_error"].to_numpy()
                - model_effects["median_fold_error"].to_numpy(),
            ],
            fmt="o",
            label=model_name,
            color=colors[model_name],
        )

        # scatter plot the mean with a "*" as its marker
        ax.plot(
            ["2011-2020", "2021-2025"],
            model_effects["median_fold_error"].to_list(),
            marker="*",
            markersize=12,
            color=colors[model_name],
        )

    ax.set_xlim([-0.25, 1.25])

    ax.set_xlabel("Year")
    ax.set_ylabel("Fold error")
    ax.legend()

    return fig, ax


def plot_predictions(df, samples, fig, ax):
    median = jnp.median(samples, axis=0)
    lower = jnp.quantile(samples, 0.025, axis=0)
    upper = jnp.quantile(samples, 0.975, axis=0)

    ax.errorbar(
        df["cases_outbreak"],
        median,
        yerr=[median - lower, upper - median],
        fmt="o",
        ecolor="green",
        c="green",
        capsize=0,
        alpha=0.4,
    )
    ax.set_xlabel("Observed outbreak size (cases)")
    ax.set_ylabel("Estimated outbreak size (cases)")
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

    # add a text annotation of the R2
    R2 = jnp.corrcoef(df["cases_outbreak"].to_jax(), median)[0, 1] ** 2
    R2 = R2 * 100  # as a percentage
    ax.text(
        0.0005 * max(df["cases_outbreak"].max(), upper.max()),
        0.5 * max(df["cases_outbreak"].max(), upper.max()),
        f"$R^2$ = {R2:.2f}%",
        fontsize=28,
    )

    return fig, ax


def plot_ecological_determinants(samples, df, significant_only, fig, ax):
    pooled_covariates = ["NME", "ME"]
    covariates = [
        "% < 5yo",
        "% school aged",
        "Average household size",
        "Standard deviation of household size",
        "% public schooled",
        "% white",
        "% graduate degree",
        "Population density (per km²)",
        "Log income ($1,000 2024 USD)",
    ]
    covariate_name_map = {
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
        "Log income (2024 USD)": "log_income",
    }

    if significant_only:
        pooled_covariates = ["NME", "ME"]
        covariates = [
            "% public schooled",
            "% white",
            "% graduate degree",
            "Log income (2024 USD)",
        ]

    all_covariates = [*pooled_covariates, *covariates]
    median_effects = []
    lower_effects = []
    upper_effects = []

    for idx, covariate in enumerate(pooled_covariates):
        samples_of_interest = samples["iv_global"][
            :, idx + 1
        ]  # +1 to skip coverage
        # scale by standard deviation
        std = df[covariate_name_map[covariate]].to_numpy().std()
        if "%" in covariate:
            std = std * 100  # because we do the modeling as a fraction
        samples_of_interest = samples_of_interest / std
        # exponentiate and multiply by 100
        samples_of_interest = jnp.exp(samples_of_interest) * 100
        median_effects.append(jnp.median(samples_of_interest, axis=0))
        lower_effects.append(jnp.quantile(samples_of_interest, 0.05, axis=0))
        upper_effects.append(jnp.quantile(samples_of_interest, 0.95, axis=0))

    # same process for unpooled covariates
    for covariate in covariates:
        # index is the number of this thing in the dict
        idx = list(covariate_name_map.keys()).index(covariate)
        samples_of_interest = samples["regression_betas"][:, idx]
        # scale by standard deviation
        std = df[covariate_name_map[covariate]].to_numpy().std()
        if "%" in covariate:
            std = std * 100  # because we do the modeling as a fraction
        samples_of_interest = samples_of_interest / std
        # exponentiate and multiply by 100
        samples_of_interest = jnp.exp(samples_of_interest) * 100
        median_effects.append(jnp.median(samples_of_interest, axis=0))
        lower_effects.append(jnp.quantile(samples_of_interest, 0.05, axis=0))
        upper_effects.append(jnp.quantile(samples_of_interest, 0.95, axis=0))

    ax.errorbar(
        x=range(len(all_covariates)),
        y=median_effects,
        yerr=[
            jnp.array(median_effects) - jnp.array(lower_effects),
            jnp.array(upper_effects) - jnp.array(median_effects),
        ],
        fmt="o",
        ecolor="magenta",
        c="magenta",
        capsize=0,
        alpha=0.7,
    )

    # line at 100%
    ax.axhline(100, ls="--", color="k")

    ax.set_xticks(range(len(all_covariates)))
    # at a 45 degree angle
    ax.set_xticklabels(all_covariates, rotation=45, ha="right")
    ax.set_xlabel("Predictor")
    ax.set_ylabel("Estimated marginal IRR (%)")
    if not significant_only:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0)

    return fig, ax


def plot_5_yr_coverage_performance(df, fig, ax):
    five_year = [True, False]
    models = ["Just coverage", "Just vaccination measures", "All covariates"]
    assembled_performance = pl.DataFrame({"five_year": five_year}).join(
        pl.DataFrame({"model": models}), how="cross"
    )
    assembled_performance = assembled_performance.with_columns(
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
    )

    for fy in five_year:
        for model_name in models:
            with open(
                os.path.join(
                    output_dir,
                    f"predictive_partial_pooled_poisson_five_year={fy}_CAR=True_{model_name}_[2021, 2022, 2023, 2024, 2025].pkl",
                ),
                "rb",
            ) as file:
                predictive_samples = pickle.load(file)

            cases = predictive_samples["cases"]
            observed = df.filter(
                pl.col("year").is_in([2021, 2022, 2023, 2024, 2025])
            )["cases_outbreak"].to_jax()
            R2 = 100 * jnp.array(
                [jnp.corrcoef(observed, c)[0, 1] ** 2 for c in cases]
            )

            median_R2 = float(jnp.median(R2, axis=0))
            lower_R2 = float(jnp.quantile(R2, 0.05, axis=0))
            upper_R2 = float(jnp.quantile(R2, 0.95, axis=0))
            mean_R2 = float(jnp.mean(R2, axis=0))

            # while we're here, calculate MAE, RMSE, fold error, and percent of outbreaks (>5, >10 cases) predicted
            mae = jnp.median(jnp.abs(observed - cases), axis=1)
            median_mae = float(jnp.median(mae, axis=0))
            lower_mae = float(jnp.quantile(mae, 0.05, axis=0))
            upper_mae = float(jnp.quantile(mae, 0.95, axis=0))
            mean_mae = float(jnp.mean(mae, axis=0))

            rmse = jnp.sqrt(jnp.mean((observed - cases) ** 2, axis=1))
            median_rmse = float(jnp.median(rmse, axis=0))
            lower_rmse = float(jnp.quantile(rmse, 0.05, axis=0))
            upper_rmse = float(jnp.quantile(rmse, 0.95, axis=0))
            mean_rmse = float(jnp.mean(rmse, axis=0))

            fold_error = jnp.mean(
                jnp.exp(jnp.abs(jnp.log((observed + 1) / (cases + 1)))), axis=1
            )
            median_fold_error = float(jnp.median(fold_error, axis=0))
            lower_fold_error = float(jnp.quantile(fold_error, 0.05, axis=0))
            upper_fold_error = float(jnp.quantile(fold_error, 0.95, axis=0))
            mean_fold_error = float(jnp.mean(fold_error, axis=0))

            percent_outbreaks_5 = jnp.mean(
                ~jnp.logical_xor(cases >= 5, observed >= 5), axis=1
            )
            median_percent_outbreaks_5 = float(
                jnp.median(percent_outbreaks_5, axis=0)
            )
            lower_percent_outbreaks_5 = float(
                jnp.quantile(percent_outbreaks_5, 0.05, axis=0)
            )
            upper_percent_outbreaks_5 = float(
                jnp.quantile(percent_outbreaks_5, 0.95, axis=0)
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
                jnp.quantile(percent_outbreaks_10, 0.05, axis=0)
            )
            upper_percent_outbreaks_10 = float(
                jnp.quantile(percent_outbreaks_10, 0.95, axis=0)
            )
            mean_percent_outbreaks_10 = float(
                jnp.mean(percent_outbreaks_10, axis=0)
            )

            assembled_performance = assembled_performance.with_columns(
                median_R2=pl.when(
                    (pl.col("five_year") == fy)
                    & (pl.col("model") == model_name)
                )
                .then(median_R2)
                .otherwise(pl.col("median_R2")),
                lower_R2=pl.when(
                    (pl.col("five_year") == fy)
                    & (pl.col("model") == model_name)
                )
                .then(lower_R2)
                .otherwise(pl.col("lower_R2")),
                upper_R2=pl.when(
                    (pl.col("five_year") == fy)
                    & (pl.col("model") == model_name)
                )
                .then(upper_R2)
                .otherwise(pl.col("upper_R2")),
                mean_R2=pl.when(
                    (pl.col("five_year") == fy)
                    & (pl.col("model") == model_name)
                )
                .then(mean_R2)
                .otherwise(pl.col("mean_R2")),
                median_mae=pl.when(
                    (pl.col("five_year") == fy)
                    & (pl.col("model") == model_name)
                )
                .then(median_mae)
                .otherwise(pl.col("median_mae")),
                lower_mae=pl.when(
                    (pl.col("five_year") == fy)
                    & (pl.col("model") == model_name)
                )
                .then(lower_mae)
                .otherwise(pl.col("lower_mae")),
                upper_mae=pl.when(
                    (pl.col("five_year") == fy)
                    & (pl.col("model") == model_name)
                )
                .then(upper_mae)
                .otherwise(pl.col("upper_mae")),
                mean_mae=pl.when(
                    (pl.col("five_year") == fy)
                    & (pl.col("model") == model_name)
                )
                .then(mean_mae)
                .otherwise(pl.col("mean_mae")),
                median_rmse=pl.when(
                    (pl.col("five_year") == fy)
                    & (pl.col("model") == model_name)
                )
                .then(median_rmse)
                .otherwise(pl.col("median_rmse")),
                lower_rmse=pl.when(
                    (pl.col("five_year") == fy)
                    & (pl.col("model") == model_name)
                )
                .then(lower_rmse)
                .otherwise(pl.col("lower_rmse")),
                upper_rmse=pl.when(
                    (pl.col("five_year") == fy)
                    & (pl.col("model") == model_name)
                )
                .then(upper_rmse)
                .otherwise(pl.col("upper_rmse")),
                mean_rmse=pl.when(
                    (pl.col("five_year") == fy)
                    & (pl.col("model") == model_name)
                )
                .then(mean_rmse)
                .otherwise(pl.col("mean_rmse")),
                median_fold_error=pl.when(
                    (pl.col("five_year") == fy)
                    & (pl.col("model") == model_name)
                )
                .then(median_fold_error)
                .otherwise(pl.col("median_fold_error")),
                lower_fold_error=pl.when(
                    (pl.col("five_year") == fy)
                    & (pl.col("model") == model_name)
                )
                .then(lower_fold_error)
                .otherwise(pl.col("lower_fold_error")),
                upper_fold_error=pl.when(
                    (pl.col("five_year") == fy)
                    & (pl.col("model") == model_name)
                )
                .then(upper_fold_error)
                .otherwise(pl.col("upper_fold_error")),
                mean_fold_error=pl.when(
                    (pl.col("five_year") == fy)
                    & (pl.col("model") == model_name)
                )
                .then(mean_fold_error)
                .otherwise(pl.col("mean_fold_error")),
                median_percent_outbreaks_5=pl.when(
                    (pl.col("five_year") == fy)
                    & (pl.col("model") == model_name)
                )
                .then(median_percent_outbreaks_5)
                .otherwise(pl.col("median_percent_outbreaks_5")),
                lower_percent_outbreaks_5=pl.when(
                    (pl.col("five_year") == fy)
                    & (pl.col("model") == model_name)
                )
                .then(lower_percent_outbreaks_5)
                .otherwise(pl.col("lower_percent_outbreaks_5")),
                upper_percent_outbreaks_5=pl.when(
                    (pl.col("five_year") == fy)
                    & (pl.col("model") == model_name)
                )
                .then(upper_percent_outbreaks_5)
                .otherwise(pl.col("upper_percent_outbreaks_5")),
                mean_percent_outbreaks_5=pl.when(
                    (pl.col("five_year") == fy)
                    & (pl.col("model") == model_name)
                )
                .then(mean_percent_outbreaks_5)
                .otherwise(pl.col("mean_percent_outbreaks_5")),
                median_percent_outbreaks_10=pl.when(
                    (pl.col("five_year") == fy)
                    & (pl.col("model") == model_name)
                )
                .then(median_percent_outbreaks_10)
                .otherwise(pl.col("median_percent_outbreaks_10")),
                lower_percent_outbreaks_10=pl.when(
                    (pl.col("five_year") == fy)
                    & (pl.col("model") == model_name)
                )
                .then(lower_percent_outbreaks_10)
                .otherwise(pl.col("lower_percent_outbreaks_10")),
                upper_percent_outbreaks_10=pl.when(
                    (pl.col("five_year") == fy)
                    & (pl.col("model") == model_name)
                )
                .then(upper_percent_outbreaks_10)
                .otherwise(pl.col("upper_percent_outbreaks_10")),
                mean_percent_outbreaks_10=pl.when(
                    (pl.col("five_year") == fy)
                    & (pl.col("model") == model_name)
                )
                .then(mean_percent_outbreaks_10)
                .otherwise(pl.col("mean_percent_outbreaks_10")),
            )

    colors = {
        "Just coverage": "orange",
        "Just vaccination measures": "purple",
        "All covariates": "red",
    }

    for model_name in reversed(models):
        model_effects = assembled_performance.filter(
            pl.col("model") == model_name
        ).sort("five_year")
        ax.errorbar(
            ["One year", "Five year"],
            model_effects["median_fold_error"].to_list(),
            yerr=[
                model_effects["median_fold_error"].to_numpy()
                - model_effects["lower_fold_error"].to_numpy(),
                model_effects["upper_fold_error"].to_numpy()
                - model_effects["median_fold_error"].to_numpy(),
            ],
            fmt="o",
            label=model_name,
            color=colors[model_name],
        )

        # scatter plot the mean with a "*" as its marker
        ax.plot(
            ["One year", "Five year"],
            model_effects["median_fold_error"].to_list(),
            marker="*",
            markersize=12,
            color=colors[model_name],
        )

    ax.set_xlim([-0.25, 1.25])

    ax.set_xlabel("Vaccination aggregation period")
    ax.set_ylabel("Fold error")
    ax.legend()

    return fig, ax


if __name__ == "__main__":
    df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
    )

    fig, ax = setup_plot(2, 2)  # type: ignore

    # performance pre and post pandemic
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
    plot_performance_over_time(df, fig, ax[0, 0])  # type: ignore

    # read predictive samples
    with open(
        os.path.join(
            output_dir,
            "predictive_partial_pooled_poisson_five_year=False_CAR=True_All covariates_[2021, 2022, 2023, 2024, 2025].pkl",
        ),
        "rb",
    ) as file:
        predictive_samples = pickle.load(file)

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
    plot_predictions(
        df.filter(pl.col("year").is_in([2021, 2022, 2023, 2024, 2025])),
        predictive_samples["cases"],
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
    with open(
        os.path.join(
            output_dir,
            "samples_partial_pooled_poisson_five_year=False_CAR=True_All covariates_[2021, 2022, 2023, 2024, 2025].pkl",
        ),
        "rb",
    ) as file:
        posterior_samples = pickle.load(file)
    plot_ecological_determinants(
        posterior_samples,
        df.filter(pl.col("year").is_in([2021, 2022, 2023, 2024, 2025])),
        significant_only=True,
        fig=fig,
        ax=ax[1, 0],  # type: ignore
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
    plot_5_yr_coverage_performance(df, fig, ax[1, 1])  # type: ignore

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
