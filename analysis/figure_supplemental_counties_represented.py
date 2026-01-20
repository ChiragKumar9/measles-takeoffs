import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import rc

font = {"family": "Nimbus Roman", "size": 28}
rc("font", **font)

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
fig_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")


def setup_plot(nrows, ncols):
    fig, ax = plt.subplots(nrows, ncols, figsize=(20, 22.5))
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


def plot_num_times_county_represented(df, fig, ax):
    counts = df.group_by("fips").agg(pl.len())["len"]
    # histogram with bins for every integer
    ax.hist(counts, bins=np.arange(1, 16, 1), color="grey", alpha=0.6)
    ax.set_xlabel("Number of years with cases")
    ax.set_ylabel("Number of counties")
    # shift the ticks because these are integer data
    ax.set_xticks([1.5, 5.5, 10.5, 15.5], ["1", "5", "10", "15"])
    ax.set_xlim(0.5)


def plot_std_cases(df, fig, ax):
    df = df.with_columns(county_occurences=pl.len().over("fips"))
    df = df.filter(pl.col("county_occurences") > 1)

    df = df.group_by("fips").agg(
        std_cases=pl.std("cases"), std_cases_outbreak=pl.std("cases_outbreak")
    )
    # ax.hist(df['std_cases_outbreak'], label = 'Total cases', alpha = 0.6, color = 'grey')
    ax.hist(
        df["std_cases_outbreak"],
        label="Indigenous cases",
        alpha=0.6,
        color="grey",
        bins=50,
    )
    # ax.legend()
    ax.set_yscale("log")
    ax.set_yticks([1, 10, 100], ["1", "10", "100"])
    ax.set_xlabel("Standard deviation of reported cases (N)")
    ax.set_ylabel("Number of\nrepeatedly-observed counties (N)")
    ax.set_xlim(0.9)
    ax.set_xticks([0, 40, 80, 120, 160, 200])


def plot_num_times_state_represented(df, fig, ax):
    counts = df.group_by("state").agg(pl.len())["len"]
    # histogram with bins for every integer
    ax.hist(counts, bins=np.arange(1, 16, 1), color="grey", alpha=0.6)
    ax.set_xlabel("Number of years with cases")
    ax.set_ylabel("Number of states")
    # shift the ticks because these are integer data
    ax.set_xticks([1.5, 5.5, 10.5, 15.5], ["1", "5", "10", "15"])
    ax.set_xlim(0.5)


def plot_std_cases_state(df, fig, ax):
    df = df.with_columns(state_occurences=pl.len().over("state"))
    df = df.filter(pl.col("state_occurences") > 1)

    df = df.group_by("state").agg(
        std_cases=pl.std("cases"), std_cases_outbreak=pl.std("cases_outbreak")
    )
    # ax.hist(df['std_cases'], label = 'Total cases', alpha = 0.6, color = 'grey')
    ax.hist(
        df["std_cases_outbreak"],
        label="Indigenous cases",
        alpha=0.6,
        color="grey",
        bins=50,
    )
    # ax.legend()
    ax.set_yscale("log")
    ax.set_yticks([1, 10, 100], ["1", "10", "100"])
    ax.set_xlabel("Standard deviation of reported cases (N)")
    ax.set_ylabel("Number of\nrepeatedly-observed states (N)")
    ax.set_xlim(0.9)
    ax.set_xticks([0, 40, 80, 120])


def plot_std_vaccine_coverage(df, fig, ax):
    # remove nulls
    df = df.select(["year", "state", "estimated_mmr_cov"])
    df = df.filter(pl.col("estimated_mmr_cov").is_not_null())
    # make sure the mmr cov column is a float
    df = df.with_columns(pl.col("estimated_mmr_cov").cast(pl.Float64))
    df_state = df.group_by(["state", "year"]).agg(
        std_coverage=pl.std("estimated_mmr_cov"),
    )
    # replace nulls with 0 because the state has only one county present
    df_state = df_state.with_columns(
        std_coverage=pl.col("std_coverage").fill_null(0)
    )
    ax.hist(
        df_state["std_coverage"],
        color="blue",
        alpha=0.6,
        label="Within-state",
        density=True,
        bins=np.arange(0, max(df_state["std_coverage"]) + 0.5, 0.5),
    )
    df_year = df.group_by(["year"]).agg(
        std_coverage=pl.std("estimated_mmr_cov")
    )
    df_year = df_year.with_columns(
        std_coverage=pl.col("std_coverage").fill_null(0)
    )
    ax.hist(
        df_year["std_coverage"],
        color="grey",
        alpha=0.6,
        label="Nationwide",
        density=True,
        bins=np.arange(0, max(df_year["std_coverage"]) + 0.5, 0.5),
    )
    ax.legend()
    ax.set_xlabel("Standard deviation of vaccine coverage (%)")
    ax.set_ylabel("Density")
    ax.set_xlim(0)


if __name__ == "__main__":
    df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        schema_overrides={"fips": pl.String},
    )

    fig, ax = setup_plot(3, 2)  # type: ignore

    ax[0, 0].text(  # type: ignore
        -0.05,
        1.08,
        "a",
        transform=ax[0, 0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )

    plot_num_times_county_represented(df, fig, ax[0, 0])  # type: ignore

    ax[0, 1].text(  # type: ignore
        -0.05,
        1.08,
        "b",
        transform=ax[0, 1].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )

    plot_std_cases(df, fig, ax[0, 1])  # type: ignore

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

    plot_num_times_state_represented(df, fig, ax[1, 0])  # type: ignore

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

    plot_std_cases_state(df, fig, ax[1, 1])  # type: ignore

    ax[2, 0].text(  # type: ignore
        -0.05,
        1.08,
        "e",
        transform=ax[2, 0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )

    plot_std_vaccine_coverage(df, fig, ax[2, 0])  # type: ignore

    df = pl.read_csv(os.path.join(data_dir, "processed_vaccinations_data.csv"))

    ax[2, 1].text(  # type: ignore
        -0.05,
        1.08,
        "f",
        transform=ax[2, 1].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )

    plot_std_vaccine_coverage(df, fig, ax[2, 1])  # type: ignore

    fig.tight_layout()

    # save
    fig.savefig(
        os.path.join(fig_dir, "figure_supplemental_counties_represented.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(fig_dir, "figure_supplemental_counties_represented.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
