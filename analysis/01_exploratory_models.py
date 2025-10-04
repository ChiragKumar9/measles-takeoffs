import os

import matplotlib.pyplot as plt
import polars as pl
import statsmodels.api as sm
from matplotlib import rc

plt.rcParams["figure.figsize"] = [9, 9]
font = {"family": "Nimbus Roman", "size": 28}
rc("font", **font)

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

if __name__ == "__main__":
    # Load dataset
    df = pl.read_csv(
        os.path.join(data_dir, "outbreaks_vaccinations_cleaned.csv")
    )

    # Logistic regression
    df = df.with_columns(
        pl.when(pl.col("cases_outbreak") >= 3)
        .then(1)
        .otherwise(
            pl.when(pl.col("cases_outbreak") == 0)
            .then(0)
            .otherwise(pl.lit(None))
        )
        .alias("outbreak_occurred")
    )

    df = df.filter(pl.col("outbreak_occurred").is_not_null())

    # make log of some variables we expect to be log-linear
    df = df.with_columns(
        log_population=pl.col("pop_5_9").log(),
        log_income=pl.col("median_household_income").log(),
    )
    # Make a year fixed effect variable
    df = df.drop("year_mobility")
    year = df["year"]
    # sort by year to ensure that the first year seen is the most recent year
    # that will be the year that is dropped, which makes the coefficients all
    # relative to the recent past
    df = df.sort("year", descending=True)
    df = df.to_dummies(columns="year", separator="_", drop_first=True)
    years = [y for y in df.columns if y.startswith("year_")]
    df = df.with_columns(year=year)
    df = df.with_columns(
        not_vaccine_exempt=100.0
        - pl.col("med_exempt_pct")
        - pl.col("nonmed_exempt_pct"),
        fraction_nonmed_exempt=pl.col("nonmed_exempt_pct")
        / (pl.col("nonmed_exempt_pct") + pl.col("med_exempt_pct")),
    )
    df = df.with_columns(
        fraction_nonmed_exempt=pl.col("fraction_nonmed_exempt").fill_nan(0.0)
    )
    exog = sm.add_constant(
        df.select(
            [
                *[
                    "estimated_mmr_cov",
                    "nonmed_exempt_pct",
                    "med_exempt_pct",
                    "total_exempt_pct",
                    "fraction_public",
                    "vaccine_coverage_school_interaction",
                    "pop_density_5_9",
                    "fraction_white",
                    "fraction_graduate_degree",
                    "log_population",
                    "log_income",
                ],
                *years,
            ]
        ).to_pandas()
    )
    print(sm.Logit(df["outbreak_occurred"].to_pandas(), exog).fit().summary())

    fit = sm.GLM(
        df["cases_outbreak"].to_pandas(),
        exog,
        family=sm.families.Poisson(),
    ).fit()
    print(fit.summary())

    fit = fit.get_prediction(exog).summary_frame(alpha=0.05)
    mean_fitted = fit["mean"]
    lower_ci = fit["mean_ci_lower"]
    upper_ci = fit["mean_ci_upper"]

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
        mean_fitted,
        yerr=[mean_fitted - lower_ci, upper_ci - mean_fitted],
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
        [0, max(df["cases_outbreak"].max(), mean_fitted.max())],
        [0, max(df["cases_outbreak"].max(), mean_fitted.max())],
        ls="--",
        c="black",
    )

    plt.show()
