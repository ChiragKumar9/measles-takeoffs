import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import rc

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)
fig_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")

font = {"family": "Nimbus Roman", "size": 28}
rc("font", **font)


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


def plot_scatter_and_regression(x, y, xlabel, ylabel, idealized_line, fig, ax):
    ax.scatter(x, y, c="k", alpha=0.7)
    # put a line of best fit through the data to highlight the trend
    # get R2 and print out
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    if idealized_line:
        ax.plot([60, 100], [40, 0], linestyle="--", c="red")
    ax.text(
        0.65,
        0.95,
        f"$R^2$ = {r2:.2f}",
        transform=ax.transAxes,
        fontsize=32,
        verticalalignment="top",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0)


if __name__ == "__main__":
    df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        schema_overrides={"fips": pl.String},
    )

    fig, ax = setup_plot(1, 3)

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

    plot_scatter_and_regression(
        df["estimated_mmr_cov"],
        df["nonmed_exempt_pct"],
        "MMR coverage (%)",
        "Nonmedical exemptions (%)",
        False,
        fig,
        ax[0],  # type: ignore
    )

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

    plot_scatter_and_regression(
        df["estimated_mmr_cov"],
        df["med_exempt_pct"],
        "MMR coverage (%)",
        "Medical exemptions (%)",
        False,
        fig,
        ax[1],  # type: ignore
    )

    ax[2].text(  # type: ignore
        -0.25,
        1.08,
        "c",
        transform=ax[2].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )

    plot_scatter_and_regression(
        df["estimated_mmr_cov"],
        df["nonmed_exempt_pct"].to_numpy() + df["med_exempt_pct"].to_numpy(),
        "MMR coverage (%)",
        "Exemptions, nonmedical + medical (%)",
        True,
        fig,
        ax[2],  # type: ignore
    )

    plt.tight_layout()

    # save
    fig.savefig(
        os.path.join(
            fig_dir, "figure_supplemental_coverage_exemptions_correlation.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(
            fig_dir, "figure_supplemental_coverage_exemptions_correlation.pdf"
        ),
        dpi=300,
        bbox_inches="tight",
    )
