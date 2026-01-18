import os

import matplotlib.pyplot as plt
import polars as pl
import yaml
from matplotlib import rc
from sklearn.metrics import roc_auc_score, roc_curve

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
    fig, ax = plt.subplots(nrows, ncols, figsize=(30, 7.5))
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
    ax.set_xlabel("Standard deviation in yearly imported cases by state (N)")
    ax.set_ylabel("Counts")
    ax.legend()


def plot_importation_probability_roc_validation(df, fig, ax):
    # precision-recall curve
    tpr, fpr, thresholds = roc_curve(
        (df["imported"] >= 1).cast(pl.Int64).to_numpy(),
        df["probability_at_least_one_import"].to_numpy(),
    )
    score = roc_auc_score(
        (df["imported"] >= 1).cast(pl.Int64).to_numpy(),
        df["probability_at_least_one_import"].to_numpy(),
    )
    ax.plot(tpr * 100, fpr * 100, color="black", lw=2)
    ax.plot([0, 100], [0, 100], color="gray", lw=1, linestyle="--")
    ax.set_xlabel("True positive rate (%)")
    ax.set_ylabel("False positive rate (%)")
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.text(60, 90, f"AUC = {score:.2f}", fontsize=32)
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

    fig, ax = setup_plot(1, 3)  # type: ignore

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
    plot_imports_over_time(df, fig, ax[0])  # type: ignore

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
    plot_state_std_imports(df, fig, ax[1])  # type: ignore

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
    counties_with_known_imports = counties_with_known_imports.join(
        import_df, on=["year", "state", "county"], how="left"
    )

    counties_with_known_imports = counties_with_known_imports.filter(
        pl.col("posterior_alpha").is_not_null()
        & pl.col("posterior_beta").is_not_null()
    )
    counties_with_known_imports = counties_with_known_imports.with_columns(
        # recall that gamma-poisson mixture is a negative binomial
        # so P(X >= 1) can be calculated using a neg binomial PMF
        probability_at_least_one_import=pl.struct(
            ["posterior_alpha", "posterior_beta"]
        ).map_elements(
            lambda x: 1
            - (
                x["posterior_beta"]
                / (x["posterior_alpha"] + x["posterior_beta"])
            )
            ** x["posterior_alpha"],
            return_dtype=pl.Float64,
        ),
    )

    ax[2].text(  # type: ignore
        -0.05,
        1.08,
        "c",
        transform=ax[2].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )

    plot_importation_probability_roc_validation(
        counties_with_known_imports,
        fig,
        ax[2],  # type: ignore
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
