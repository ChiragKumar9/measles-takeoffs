from numpyro import enable_x64

enable_x64()
import os
import pickle

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import polars as pl
import yaml
from matplotlib import rc
from scipy.stats import rankdata

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
    fig, ax = plt.subplots(nrows, ncols, figsize=(20, 7.5))
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


def plot_risk_validation(df, fig, ax):
    # we have validation data for 2023, 2024, and 2025
    projection_years = [2023, 2024, 2025]
    dfs_by_year = {}
    method = {
        2023: "dense",
        2024: "dense",
        2025: "dense",
    }
    for year in projection_years:
        df_year = df.filter(pl.col("year") == year)
        # calculate the rank order of the samples
        with open(
            os.path.join(
                output_dir,
                f"samples_partial_pooled_poisson_validation_testing_{year}.pkl",
            ),
            "rb",
        ) as file:
            projections = pickle.load(file)

        ranked_projections = (
            rankdata(projections, method=method[year], axis=1) - 1
        )
        percentile_projections = (
            100
            * ranked_projections
            / (ranked_projections.max(axis=1, keepdims=True) + 1)
        )

        df_year = df_year.with_columns(
            median_risk_rank=pl.Series(
                np.median(percentile_projections, axis=0).__array__()
            ),
            lower_risk_rank=pl.Series(
                np.percentile(
                    percentile_projections, 2.5, method="lower", axis=0
                ).__array__()
            ),
            upper_risk_rank=pl.Series(
                np.percentile(
                    percentile_projections, 97.5, method="higher", axis=0
                ).__array__()
            ),
        )

        df_year = df_year.with_columns(
            year_specific_rank_order=pl.col("cases_outbreak").rank(
                method[year],  # type: ignore
                descending=False,
            )
            - 1
        ).with_columns(
            year_specific_percentile=pl.col("year_specific_rank_order")
            / (pl.max("year_specific_rank_order") + 1)
            * 100
        )

        # ask whether the 2.5th to 97.5th percentile contains the observed rank
        df_year = df_year.with_columns(
            contains_observed_rank=pl.when(
                (
                    pl.col("lower_risk_rank")
                    <= pl.col("year_specific_percentile")
                )
                & (
                    pl.col("upper_risk_rank")
                    >= pl.col("year_specific_percentile")
                )
            )
            .then(True)
            .otherwise(False),
            observed_rank_above=pl.when(
                pl.col("year_specific_percentile") > pl.col("upper_risk_rank")
            )
            .then(True)
            .otherwise(False),
            observed_rank_below=pl.when(
                pl.col("year_specific_percentile") < pl.col("lower_risk_rank")
            )
            .then(True)
            .otherwise(False),
        )
        # # if this is true, then we have a successful prediction
        # # we want to circumvent the issue of rank ordering biasing towards the
        # # center for values that are tied
        df_year = df_year.with_columns(
            adjusted_risk_rank=pl.when("contains_observed_rank")
            .then(pl.col("year_specific_percentile"))
            .otherwise(
                pl.when("observed_rank_above")
                .then(pl.col("upper_risk_rank"))
                .otherwise(pl.col("lower_risk_rank"))
            )
        )

        dfs_by_year[year] = df_year
        print(
            df_year.filter(~pl.col("contains_observed_rank"))[
                [
                    "county",
                    "state",
                    "cases_outbreak",
                    "median_risk_rank",
                    "lower_risk_rank",
                    "upper_risk_rank",
                    "year_specific_percentile",
                ]
            ]
        )

    # boxplots by chunk of risk quintiles
    num_chunks = 5
    outbreaks = {2023: [], 2024: [], 2025: []}
    for tile in range(num_chunks):
        # want the outbreak size data for each chunk
        percentile_lower = tile * (100 / num_chunks)
        percentile_upper = (tile + 1) * (100 / num_chunks)

        # take the data for each year from that chunk
        for year in projection_years:
            outbreaks[year].append(
                dfs_by_year[year]
                .filter(
                    (pl.col("adjusted_risk_rank") >= percentile_lower)
                    & (pl.col("adjusted_risk_rank") < percentile_upper)
                )["cases_outbreak"]
                .to_list()
            )

    offsets = np.linspace(-0.35, 0.35, num_chunks)
    cmap = plt.get_cmap("Reds")
    cmap = [cmap(i) for i in np.linspace(0.3, 1.0, num_chunks)]
    for tile in range(num_chunks):
        data = [outbreaks[year][tile] for year in outbreaks.keys()]
        ax.boxplot(
            data,
            positions=np.array([2023, 2024, 2025]) + offsets[tile],
            orientation="vertical",
            whis=(2.5, 97.5),
            widths=0.15,
            showmeans=True,
            boxprops=dict(color=cmap[tile], linewidth=2),
            medianprops=dict(color="black", linewidth=2),
            meanprops=dict(
                markeredgecolor=cmap[tile],
                markerfacecolor=cmap[tile],
                marker="*",
                markersize=12,
            ),
            whiskerprops=dict(color=cmap[tile]),
            capprops=dict(color=cmap[tile]),
            flierprops=dict(markeredgecolor=cmap[tile], markersize=8),
        )

    ax.set_yscale("symlog")
    ax.set_ylim(0)

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
    ax.xaxis.set_minor_locator(
        tck.FixedLocator(
            [year + offset for year in projection_years for offset in offsets]
        )
    )

    ax.set_yticks([0, 1, 10, 100], labels=[0, 1, 10, 100])
    ax.set_ylabel("Reported cases (N)")
    ax.set_xlabel("Year")

    patches = [
        mpatches.Patch(edgecolor=color, linewidth=2, fill=False, label=label)
        for color, label in zip(
            cmap,
            [
                f"{i * 100 / num_chunks:.0f}%-{(i + 1) * 100 / num_chunks:.0f}%"
                for i in range(num_chunks)
            ],
        )
    ]

    ax.legend(handles=patches, loc=(0.008, 0.53))
    ax.set_xticks([2023, 2024, 2025], labels=[2023, 2024, 2025])

    return fig, ax


if __name__ == "__main__":
    df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
    )

    fig, ax = setup_plot(1, 2)  # type: ignore

    ax[0].text(  # type: ignore
        -0.05,
        1.08,
        "a",
        transform=ax[0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_risk_validation(df, fig, ax[0])  # type: ignore

    ax[1].text(  # type: ignore
        -0.05,
        1.08,
        "b",
        transform=ax[1].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )

    fig.tight_layout()

    # save
    fig.savefig(
        os.path.join(fig_dir, "figure_supplemental_validation.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(fig_dir, "figure_supplemental_validation.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
