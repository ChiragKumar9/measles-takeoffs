import os

import matplotlib.pyplot as plt
import polars as pl
import yaml
from matplotlib import rc
from numpyro import enable_x64

enable_x64()

font = {"family": "Nimbus Roman", "size": 28}
rc("font", **font)

output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)
fig_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")

# read analysis params yaml
with open("params.yml", "r") as f:
    params = yaml.safe_load(f)

state_abbreviations = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}


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


def plot_county_risk_fcn_of_coverage(df, median, upper, lower, label, fig, ax):
    # plot rank vs MMR coverage
    ax.errorbar(
        df["estimated_mmr_cov_raw"].to_numpy(),
        df[median].to_numpy(),
        yerr=0,
        # [[df[median].to_numpy() - df[lower].to_numpy()],
        #         [df[upper].to_numpy() - df[median].to_numpy()]],
        fmt="o",
        alpha=0.7,
        ecolor="black",
        color="black",
    )

    ax.set_ylabel(label)
    ax.set_xlabel("MMR coverage (%)")

    return fig, ax


def plot_state_risk_fcn_of_coverage(state_df, column_prefix, label, fig, ax):
    # get state abbreviations
    state_df = state_df.with_columns(
        state_abbr=pl.col("state").replace(state_abbreviations)
    )

    # now plot rank vs MMR coverage
    for cov, median_risk, lower_risk, upper_risk, state in zip(
        state_df["weighted_mmr_coverage"],
        state_df[f"median_{column_prefix}_risk_rank"],
        state_df[f"lower_{column_prefix}_risk_rank"],
        state_df[f"upper_{column_prefix}_risk_rank"],
        state_df["state_abbr"],
    ):
        # ax.errorbar(
        #     cov,
        #     median_risk,
        #     yerr = [[median_risk - lower_risk],
        #             [upper_risk - median_risk]],
        #     fmt='none',
        #     alpha=0.7,
        #     ecolor='black',
        # )

        ax.scatter(cov, median_risk, s=0, alpha=0.0)

        ax.text(
            cov,
            median_risk,
            state,
            fontsize=18,
            va="center",
            ha="center",
        )

    ax.set_ylabel(label)
    ax.set_xlabel("MMR coverage (%)")

    return fig, ax


if __name__ == "__main__":
    fig, ax = setup_plot(2, 2)  # type: ignore

    df = pl.read_csv(
        os.path.join(output_dir, "assembled_projection_results.csv"),
        # make sure we read fips codes as strings
    )
    df = df.filter(pl.col("type") == "projection")

    state_df = pl.read_csv(
        os.path.join(output_dir, "state_assembled_projection_results.csv"),
    )
    state_df = state_df.filter(pl.col("type") == "projection")

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
    plot_county_risk_fcn_of_coverage(
        df,
        "median_unconditioned_risk_rank",
        "upper_unconditioned_risk_rank",
        "lower_unconditioned_risk_rank",
        "Risk percentile",
        fig,
        ax[0, 0],  # type: ignore
    )

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
    plot_state_risk_fcn_of_coverage(
        state_df,
        "unconditioned",
        "Risk percentile",
        fig,
        ax[0, 1],  # type: ignore
    )

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
    plot_county_risk_fcn_of_coverage(
        df,
        "median_conditioned_risk_rank",
        "upper_conditioned_risk_rank",
        "lower_conditioned_risk_rank",
        "Import-conditioned risk percentile",
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
    plot_state_risk_fcn_of_coverage(
        state_df,
        "conditioned",
        "Import-conditioned risk percentile",
        fig,
        ax[1, 1],  # type: ignore
    )

    fig.tight_layout()

    # save
    fig.savefig(
        os.path.join(fig_dir, "figure_supplemental_risk_fcn_coverage.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(fig_dir, "figure_supplemental_risk_fcn_coverage.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
