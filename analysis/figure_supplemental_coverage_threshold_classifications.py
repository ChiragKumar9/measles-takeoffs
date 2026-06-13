import os

import matplotlib.pyplot as plt
import polars as pl
import yaml
from matplotlib import rc
from sklearn.metrics import roc_curve

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
    fig, ax = plt.subplots(nrows, ncols, figsize=(30, 15))
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


def plot_outbreak_identification_rate(
    df, measure, fig, ax_sensitivity, ax_specificity, ax_roc
):
    # calculate what percentage of outbreaks we would have captured had we said
    # all counties with coverage less than threshold% were at risk
    df = df.with_columns(
        recent=pl.col("year").is_in([2021, 2022, 2023, 2024, 2025]),
        below_90=pl.col(measure) <= 90,
        below_91=pl.col(measure) <= 91,
        below_92=pl.col(measure) <= 92,
        below_93=pl.col(measure) <= 93,
        below_94=pl.col(measure) <= 94,
        below_95=pl.col(measure) <= 95,
        below_96=pl.col(measure) <= 96,
        below_97=pl.col(measure) <= 97,
        below_98=pl.col(measure) <= 98,
        cases_5=pl.col("cases_outbreak") >= 5,
        cases_not_5=pl.col("cases_outbreak") < 5,
    )

    df = df.with_columns(
        total_5=pl.col("cases_5").sum().over("recent"),
        total_not_5=pl.col("cases_not_5").sum().over("recent"),
    )

    df_identification = pl.DataFrame()
    for coverage in range(90, 99):
        df_temp = (
            df.group_by(["recent", f"below_{coverage}"])
            .agg(
                identified_5=pl.sum("cases_5") / pl.col("total_5").first(),
                not_identified_5=pl.sum("cases_not_5")
                / pl.col("total_not_5").first(),
            )
            .filter(pl.col(f"below_{coverage}") == True)
            .with_columns(coverage=pl.lit(coverage))
        )
        df_identification = pl.concat(
            [df_identification, df_temp], how="diagonal_relaxed"
        )

    ax_sensitivity.plot(
        df_identification.filter(pl.col("recent") == False)[
            "coverage"
        ].to_list(),
        100
        * df_identification.filter(pl.col("recent") == False)[
            "identified_5"
        ].to_numpy(),
        marker="s",
        label="≥5 cases, 2011-2020",
        color="blue",
        linestyle="--",
    )

    ax_sensitivity.plot(
        df_identification.filter(pl.col("recent") == True)[
            "coverage"
        ].to_list(),
        100
        * df_identification.filter(pl.col("recent") == True)[
            "identified_5"
        ].to_numpy(),
        marker="s",
        label="≥5 cases, 2021-2025",
        color="green",
        linestyle="--",
    )

    ax_sensitivity.legend(loc=(0.40, 0.0), edgecolor="black")
    ax_sensitivity.set_xlabel("MMR vaccine coverage (%)")
    ax_sensitivity.set_ylabel("True positive rate (%)")
    ax_sensitivity.set_ylim(0, 100)

    ax_specificity.plot(
        df_identification.filter(pl.col("recent") == False)[
            "coverage"
        ].to_list(),
        100
        * df_identification.filter(pl.col("recent") == False)[
            "not_identified_5"
        ].to_numpy(),
        marker="s",
        label="≥5 cases, 2011-2020",
        color="blue",
        linestyle="--",
    )

    ax_specificity.plot(
        df_identification.filter(pl.col("recent") == True)[
            "coverage"
        ].to_list(),
        100
        * df_identification.filter(pl.col("recent") == True)[
            "not_identified_5"
        ].to_numpy(),
        marker="s",
        label="≥5 cases, 2021-2025",
        color="green",
        linestyle="--",
    )

    ax_specificity.set_xlabel("MMR vaccine coverage (%)")
    ax_specificity.set_ylabel("False positive rate (%)")
    ax_specificity.set_ylim(0, 100)

    # Calculate ROC curves using sklearn
    # For older period (2011-2020)
    df_old = df.filter(pl.col("recent") == False)
    y_true_old = df_old["cases_5"].to_numpy().astype(int)
    # Use negative coverage as scores (lower coverage = higher risk)
    y_scores_old = -df_old[measure].to_numpy()

    fpr_old, tpr_old, _ = roc_curve(y_true_old, y_scores_old)

    # For recent period (2021-2025)
    df_recent = df.filter(pl.col("recent") == True)
    y_true_recent = df_recent["cases_5"].to_numpy().astype(int)
    y_scores_recent = -df_recent[measure].to_numpy()

    fpr_recent, tpr_recent, _ = roc_curve(y_true_recent, y_scores_recent)

    ax_roc.plot(
        100 * fpr_old,
        100 * tpr_old,
        label="≥5 cases, 2011-2020",
        color="blue",
        linestyle="--",
        linewidth=2,
    )

    ax_roc.plot(
        100 * fpr_recent,
        100 * tpr_recent,
        label="≥5 cases, 2021-2025",
        color="green",
        linestyle="--",
        linewidth=2,
    )

    ax_roc.plot([0, 100], [0, 100], linestyle="--", c="k", alpha=0.5)
    ax_roc.set_xlabel("False positive rate (%)")
    ax_roc.set_ylabel("True positive rate (%)")
    ax_roc.set_ylim(0, 100)
    ax_roc.set_xlim(0, 100)


if __name__ == "__main__":
    df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        schema_overrides={"fips": pl.String},
    )

    fig, ax = setup_plot(2, 3)  # type: ignore

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

    ax[0, 2].text(  # type: ignore
        -0.25,
        1.08,
        "c",
        transform=ax[0, 2].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_outbreak_identification_rate(
        df,
        "estimated_mmr_cov",
        fig,
        ax[0, 0],  # type: ignore
        ax[0, 1],  # type: ignore
        ax[0, 2],  # type: ignore
    )

    ax[1, 0].text(  # type: ignore
        -0.25,
        1.08,
        "d",
        transform=ax[1, 0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )

    ax[1, 1].text(  # type: ignore
        -0.25,
        1.08,
        "e",
        transform=ax[1, 1].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )

    ax[1, 2].text(  # type: ignore
        -0.25,
        1.08,
        "f",
        transform=ax[1, 2].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_outbreak_identification_rate(
        df,
        "estimated_mmr_cov_5yr",
        fig,
        ax[1, 0],  # type: ignore
        ax[1, 1],  # type: ignore
        ax[1, 2],  # type: ignore
    )

    fig.tight_layout()

    # save
    fig.savefig(
        os.path.join(
            fig_dir,
            "figure_supplemental_coverage_threshold_classifications.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(
            fig_dir,
            "figure_supplemental_coverage_threshold_classifications.pdf",
        ),
        dpi=300,
        bbox_inches="tight",
    )
