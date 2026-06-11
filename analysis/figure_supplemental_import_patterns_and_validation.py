import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml
from matplotlib import rc
from scipy.stats import gamma
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

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


def plot_imports_over_time(df, fig, ax):
    df = df.group_by("year").agg(pl.sum("imported"))
    df = df.sort(by="year")
    ax.plot(
        df["year"],
        df["imported"],
        marker="^",
        markersize=10,
        color="black",
        linestyle="--",
    )
    ax.set_ylim(0)
    ax.set_xlabel("Year")
    ax.set_ylabel("Total imported cases\nin the US (N)")


def plot_state_std_imports(df, fig, ax):
    df_all_years = df.group_by("state").agg(
        std=pl.std("imported"),
    )
    df_no_covid = (
        df.filter(~pl.col("year").is_in([2020, 2021]))
        .group_by("state")
        .agg(
            std=pl.std("imported"),
        )
    )
    ax.hist(
        df_all_years["std"],
        color="grey",
        bins=50,
        alpha=0.8,
        label="All years",
    )
    ax.hist(
        df_no_covid["std"],
        color="blue",
        bins=50,
        alpha=0.3,
        label="Excluding 2020-2021",
    )
    ax.set_xlabel("Standard deviation in yearly imported cases\nby state (N)")
    ax.set_xlim(0)
    ax.set_ylabel("Counts")
    ax.legend(edgecolor="black")


def plot_importation_probability_roc_validation(df, fig, ax):
    # roc curve
    tpr_log, fpr_log, _ = roc_curve(
        (df["imported"] >= 1).cast(pl.Int64).to_numpy(),
        df["probability_at_least_one_import_log"].to_numpy(),
    )
    score_log = roc_auc_score(
        (df["imported"] >= 1).cast(pl.Int64).to_numpy(),
        df["probability_at_least_one_import_log"].to_numpy(),
    )

    tpr, fpr, _ = roc_curve(
        (df["imported"] >= 1).cast(pl.Int64).to_numpy(),
        df["probability_at_least_one_import"].to_numpy(),
    )
    score = roc_auc_score(
        (df["imported"] >= 1).cast(pl.Int64).to_numpy(),
        df["probability_at_least_one_import"].to_numpy(),
    )

    ax.plot(
        tpr_log * 100,
        fpr_log * 100,
        color="purple",
        lw=2,
        label=f"Log10-pop weighted\nAUC: {score_log:.2f}",
    )
    ax.plot(
        tpr * 100,
        fpr * 100,
        color="salmon",
        lw=2,
        label=f"Pop weighted\nAUC: {score:.2f}",
    )
    ax.plot([0, 100], [0, 100], color="gray", lw=1, linestyle="--")
    ax.set_xlabel("True positive rate (%)")
    ax.set_ylabel("False positive rate (%)")
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.legend(loc="lower right", edgecolor="black")
    return fig, ax


def plot_importation_probability_prc_validation(df, fig, ax):
    # precision-recall curve
    precision_log, recall_log, _ = precision_recall_curve(
        (df["imported"] >= 1).cast(pl.Int64).to_numpy(),
        df["probability_at_least_one_import_log"].to_numpy(),
    )
    score_log = average_precision_score(
        (df["imported"] >= 1).cast(pl.Int64).to_numpy(),
        df["probability_at_least_one_import_log"].to_numpy(),
    )

    precision, recall, _ = precision_recall_curve(
        (df["imported"] >= 1).cast(pl.Int64).to_numpy(),
        df["probability_at_least_one_import"].to_numpy(),
    )
    score = average_precision_score(
        (df["imported"] >= 1).cast(pl.Int64).to_numpy(),
        df["probability_at_least_one_import"].to_numpy(),
    )

    ax.plot(
        recall_log * 100,
        precision_log * 100,
        color="purple",
        lw=2,
        label=f"Log10-pop weighted\nAverage precision: {score_log:.2f}\n(Fold increase: {score_log / (df['imported'] >= 1).mean():.2f}x)",
    )
    ax.plot(
        recall * 100,
        precision * 100,
        color="salmon",
        lw=2,
        label=f"Pop weighted\nAverage precision: {score:.2f}\n(Fold increase: {score / (df['imported'] >= 1).mean():.2f}x)",
    )
    ax.set_xlabel("Recall (%)")
    ax.set_ylabel("Precision (%)")
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    # add a grey dashed line for the baseline precision
    ax.plot(
        [0, 100],
        [(df["imported"] >= 1).mean() * 100] * 2,
        color="gray",
        lw=1,
        linestyle="--",
        label="Baseline precision\n({:.5f})".format(
            (df["imported"] >= 1).mean()
        ),
    )
    ax.legend(loc=(0.3, 0.4), edgecolor="black")
    return fig, ax


if __name__ == "__main__":
    df = pl.read_excel(
        os.path.join(data_dir, "IndigenousImportedFullData.xlsx")
    )

    df = df.rename({col: col.lower() for col in df.columns})
    # replace "DC" with "District of Columbia"
    df = (
        df.with_columns(
            state=pl.when(pl.col("state") == "DC")
            .then(pl.lit("District of Columbia"))
            .otherwise(pl.col("state"))
        )
        .with_columns(
            # fix North Dakora to be North Dakota
            state=pl.when(pl.col("state") == "North Dakora")
            .then(pl.lit("North Dakota"))
            .otherwise(pl.col("state"))
        )
        .with_columns(
            # idaho to be Idaho
            state=pl.when(pl.col("state") == "idaho")
            .then(pl.lit("Idaho"))
            .otherwise(pl.col("state"))
        )
    )

    # concatenate 2025 import data
    df_2025 = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        schema_overrides={"fips": pl.String},
    )
    df_2025 = df_2025.filter(pl.col("year") == 2025).select(
        ["state", "year", "imported"]
    )

    df = pl.concat([df, df_2025], how="diagonal_relaxed")
    # for any states not present in 2025, fill in with 0 imported cases
    states_2025 = df_2025.select("state").unique().to_series().to_list()
    all_states = df.select("state").unique().to_series().to_list()
    missing_states = set(all_states) - set(states_2025)
    for state in missing_states:
        df = pl.concat(
            [
                df,
                pl.DataFrame(
                    {"state": [state], "year": [2025], "imported": [0]}
                ),
            ],
            how="diagonal_relaxed",
        )

    fig, ax = setup_plot(2, 2)  # type: ignore

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
    plot_imports_over_time(df, fig, ax[0, 0])  # type: ignore

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
    plot_state_std_imports(df, fig, ax[0, 1])  # type: ignore

    # we want to plot empirical import probability against recorded number
    # of imports for the counties that have known import numbers
    counties_with_known_imports = pl.read_csv(
        os.path.join(output_dir, "counties_with_known_imports.csv"),
    )
    import_df = pl.read_csv(
        os.path.join(output_dir, "empirical_importation_probability.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
    )
    import_df = import_df.with_columns(
        mu=pl.col("posterior_alpha") / pl.col("posterior_beta")
    )
    count_imports = import_df.filter(pl.col("type") == "projection")
    count_imports = count_imports.sort(by="year", descending=True)
    count_imports = count_imports.group_by(
        ["state", "county"], maintain_order=True
    ).agg(
        # for each county, take the most recent projection of import counts as the "count" of imports for that county
        pl.first("importation_probability_mean"),
        pl.first("importation_probability_median"),
        pl.first("importation_probability_lower"),
        pl.first("importation_probability_upper"),
        pl.first("mu"),
        pl.first("posterior_alpha"),
        pl.first("posterior_beta"),
    )
    # sum
    print(count_imports.sum())

    rng = np.random.default_rng(42)
    N_SAMPLES = 100_000

    # Sample lambda_i ~ Gamma(alpha_i, beta_i) for each county, then
    # sample counts ~ Poisson(lambda_i). Sum across counties per sample.
    alphas = count_imports["posterior_alpha"].to_numpy()
    betas = count_imports["posterior_beta"].to_numpy()

    # Shape: (N_SAMPLES, n_counties)
    lambda_samples = rng.gamma(
        shape=alphas, scale=1.0 / betas, size=(N_SAMPLES, len(alphas))
    )
    count_samples = rng.poisson(
        lam=lambda_samples
    )  # Poisson draw per county per sample
    national_totals = count_samples.sum(axis=1)  # Sum across counties

    print(f"Median national imports:          {np.median(national_totals)}")
    print(
        f"2.5th percentile national imports: {np.percentile(national_totals, 2.5)}"
    )
    print(
        f"97.5th percentile national imports: {np.percentile(national_totals, 97.5)}"
    )

    print((df_2025["imported"].sum() / count_imports.sum()["mu"])[0])
    with open(
        os.path.join(
            output_dir,
            "import_scaler_2025.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(
            (df_2025["imported"].sum() / count_imports.sum()["mu"])[0], f
        )
    counties_with_known_imports = counties_with_known_imports.join(
        import_df, on=["year", "state", "county"], how="left"
    )

    counties_with_known_imports = counties_with_known_imports.filter(
        pl.col("posterior_alpha_pre_2021").is_not_null()
        & pl.col("posterior_beta_pre_2021").is_not_null()
        & pl.col("posterior_alpha_pre_2021_log").is_not_null()
    )
    # filter the year to be either the latest year for that county or at least
    # be 2020
    counties_with_known_imports = counties_with_known_imports.with_columns(
        latest_year=pl.col("year").max().over(["state", "county"])
    )
    counties_with_known_imports = counties_with_known_imports.filter(
        (pl.col("year") == pl.col("latest_year")) | (pl.col("year") >= 2020)
    )
    counties_with_known_imports = counties_with_known_imports.with_columns(
        # recall that gamma-poisson mixture is a negative binomial
        # so P(X >= 1) can be calculated using a neg binomial PMF
        probability_at_least_one_import_log=pl.struct(
            ["posterior_alpha_pre_2021_log", "posterior_beta_pre_2021"]
        ).map_elements(
            lambda x: 1
            - (
                x["posterior_beta_pre_2021"]
                / (
                    x["posterior_alpha_pre_2021_log"]
                    + x["posterior_beta_pre_2021"]
                )
            )
            ** x["posterior_alpha_pre_2021_log"],
            return_dtype=pl.Float64,
        ),
        probability_at_least_one_import=pl.struct(
            ["posterior_alpha_pre_2021", "posterior_beta_pre_2021"]
        ).map_elements(
            lambda x: 1
            - (
                x["posterior_beta_pre_2021"]
                / (
                    x["posterior_alpha_pre_2021"]
                    + x["posterior_beta_pre_2021"]
                )
            )
            ** x["posterior_alpha_pre_2021"],
            return_dtype=pl.Float64,
        ),
    )
    # tell us about the distribution of counties with known imports
    print(counties_with_known_imports["imported"].sort().value_counts())

    ax[1, 0].text(  # type: ignore
        -0.05,
        1.08,
        "c",
        transform=ax[1, 0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )

    plot_importation_probability_roc_validation(
        counties_with_known_imports,
        fig,
        ax[1, 0],  # type: ignore
    )

    ax[1, 1].text(  # type: ignore
        -0.05,
        1.08,
        "d",
        transform=ax[1, 1].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )

    plot_importation_probability_prc_validation(
        counties_with_known_imports,
        fig,
        ax[1, 1],  # type: ignore
    )

    fig.tight_layout()

    # save
    fig.savefig(
        os.path.join(fig_dir, "figure_supplemental_import_patterns.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(fig_dir, "figure_supplemental_import_patterns.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
