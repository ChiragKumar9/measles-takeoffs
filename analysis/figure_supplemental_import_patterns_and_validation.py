import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml
from matplotlib import rc
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from outbreak_size.importation import (
    IMPORT_CONDITIONING_VALIDATION_YEAR,
    probability_at_least_one_import,
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
    # one value per state and year (the 2025 data has a row for each place)
    df = df.group_by(["state", "year"]).agg(pl.sum("imported"))
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
    fpr_log, tpr_log, _ = roc_curve(
        (df["imported"] >= 1).cast(pl.Int64).to_numpy(),
        df["probability_at_least_one_import_log"].to_numpy(),
    )
    score_log = roc_auc_score(
        (df["imported"] >= 1).cast(pl.Int64).to_numpy(),
        df["probability_at_least_one_import_log"].to_numpy(),
    )

    fpr, tpr, _ = roc_curve(
        (df["imported"] >= 1).cast(pl.Int64).to_numpy(),
        df["probability_at_least_one_import"].to_numpy(),
    )
    score = roc_auc_score(
        (df["imported"] >= 1).cast(pl.Int64).to_numpy(),
        df["probability_at_least_one_import"].to_numpy(),
    )

    ax.plot(
        fpr_log * 100,
        tpr_log * 100,
        color="purple",
        lw=2,
        label=f"Log10-pop weighted\nAUC: {score_log:.2f}",
    )
    ax.plot(
        fpr * 100,
        tpr * 100,
        color="salmon",
        lw=2,
        label=f"Pop weighted\nAUC: {score:.2f}",
    )
    ax.plot([0, 100], [0, 100], color="gray", lw=1, linestyle="--")
    ax.set_xlabel("False positive rate (%)")
    ax.set_ylabel("True positive rate (%)")
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

    # concatenate 2025 import data: the file and filters of
    # 08_importation_model.py (year typo corrected, rows with a case count, no
    # South Carolina), so that panels a and b and the scaler below count the
    # imports the importation model used
    df_2025 = pl.read_csv(
        os.path.join(data_dir, "2025CasesImports QCed clean.csv"),
        schema_overrides={"fips": pl.String},
    )
    df_2025 = df_2025.with_columns(
        year=pl.when(pl.col("year") == 2925)
        .then(2025)
        .otherwise(pl.col("year"))
    )
    df_2025 = (
        df_2025.filter(
            pl.col("Lauren Case").is_not_null()
            & (pl.col("state") != "South Carolina")
        )
        .filter(pl.col("year") == 2025)
        .select(["state", "year", "imported"])
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
    # every county-year with known imports (08), also the counties that are
    # not in the analysis or projection data, with the pre-2021 posterior times
    # the county's share of its state
    counties_with_known_imports = pl.read_csv(
        os.path.join(output_dir, "import_validation_county_years.csv"),
        schema_overrides={"fips": pl.String, "place": pl.String},
    )
    import_df = pl.read_csv(
        os.path.join(output_dir, "empirical_importation_probability.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
    )
    import_df = import_df.with_columns(
        mu=pl.col("posterior_alpha") / pl.col("posterior_beta")
    )
    # South Carolina is left out of the observed imports above (08 uses its
    # own South Carolina data), so it is left out of the expected imports too
    # The expected annual imports are those of the posterior that serves the
    # projections: the projection rows of one validation year (the posterior
    # of all years of data, through 2025), one row per county. (This used to
    # sort the projection rows of every validation year by the year of their
    # vaccination data and take the first of each county; rows of equal year
    # have no defined order, so a county could get the posterior of an earlier
    # validation year, which uses fewer years of data.)
    count_imports = import_df.filter(
        (pl.col("type") == "projection")
        & (pl.col("validation_year") == IMPORT_CONDITIONING_VALIDATION_YEAR)
        & (pl.col("state") != "South Carolina")
    ).select(
        "fips",
        "importation_probability_mean",
        "importation_probability_median",
        "importation_probability_lower",
        "importation_probability_upper",
        "mu",
        "posterior_alpha",
        "posterior_beta",
    )
    # by fips: county names are not unique within a state (e.g. Delaware's
    # "State" or Alaska's regions)
    assert not count_imports["fips"].is_duplicated().any(), (
        "a county has two projection rows of the validation year "
        f"{IMPORT_CONDITIONING_VALIDATION_YEAR}"
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

    # the scaler conditions the imports of the projections on the level of
    # 2025: the imports of 2025 over the expected annual imports of the
    # posterior of the projections
    observed_imports_2025 = df_2025["imported"].sum()
    expected_annual_imports = count_imports["mu"].sum()
    import_scaler_2025 = observed_imports_2025 / expected_annual_imports
    print(
        f"imports of 2025: {observed_imports_2025}; expected annual imports "
        f"of the posterior: {expected_annual_imports:.3f}; scaler: "
        f"{import_scaler_2025:.4f}"
    )
    with open(
        os.path.join(
            output_dir,
            "import_scaler_2025.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(float(import_scaler_2025), f)
    # the probabilities below use the posterior from the years before 2021, so
    # only years from 2021 on are out-of-sample
    counties_with_known_imports = counties_with_known_imports.filter(
        pl.col("year") >= 2021
    )
    # the probability of at least one import in a year, as 08 distributes the
    # state's rate to the county (see probability_at_least_one_import)
    counties_with_known_imports = counties_with_known_imports.with_columns(
        probability_at_least_one_import_log=pl.struct(
            [
                "posterior_alpha_pre_2021_log",
                "posterior_beta_pre_2021",
                "share_log",
            ]
        ).map_elements(
            lambda x: probability_at_least_one_import(
                x["posterior_alpha_pre_2021_log"],
                x["posterior_beta_pre_2021"],
                x["share_log"],
            ),
            return_dtype=pl.Float64,
        ),
        probability_at_least_one_import=pl.struct(
            ["posterior_alpha_pre_2021", "posterior_beta_pre_2021", "share"]
        ).map_elements(
            lambda x: probability_at_least_one_import(
                x["posterior_alpha_pre_2021"],
                x["posterior_beta_pre_2021"],
                x["share"],
            ),
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
