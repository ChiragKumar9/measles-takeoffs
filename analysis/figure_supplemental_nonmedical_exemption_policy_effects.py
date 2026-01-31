import os

import matplotlib.pyplot as plt
import numpy as np
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
    fig, ax = plt.subplots(nrows, ncols, figsize=(30, 10))
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


def median_percent_change(df, fig, ax):
    # make violin plots of the percent change in outbreak risk
    # the scenarios are
    scenarios = {
        "Existing NME bans": "add_nmes_back",
        "Highest risk states": "top_risk_state_unconditioned",
        "Neighboring counties": "bordering_high_risk_state",
        "Highest NME states": "top_nme_state",
        "Highest risk counties": "top_ten_pct_counties",
    }

    for idx, (scenario, column) in enumerate(scenarios.items()):
        scenario_df = df.filter(pl.col(column) == True)
        if scenario == "Highest NME states":
            scenario_df = scenario_df.filter(
                ~(pl.col("top_ten_pct_counties") == True)
            )
            scenario_df = scenario_df.filter(
                ~(pl.col("top_risk_state_unconditioned") == True)
            )
        violinparts = ax.violinplot(
            scenario_df["median_percent_change"].drop_nans(),
            positions=[idx],
            showextrema=False,
        )

        for pc in violinparts["bodies"]:
            pc.set(facecolor="grey", alpha=0.4)

        median = np.nanmedian(scenario_df["median_percent_change"].to_numpy())
        lower = np.nanpercentile(
            scenario_df["median_percent_change"].to_numpy(), 25
        )
        upper = np.nanpercentile(
            scenario_df["median_percent_change"].to_numpy(), 75
        )

        ax.errorbar(
            idx,
            median,
            yerr=[[median - lower], [upper - median]],
            fmt="o",
            color="grey",
            capsize=5,
            capthick=2,
        )

    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(list(scenarios.keys()), rotation=-45, ha="left")
    ax.set_ylabel("Percent change in outbreak size (%)")
    ax.set_ylim(-100)
    ax.axhline(0, color="black", linestyle="--")


def scenario_results(df, fig, ax, plot_total, remove_overlap):
    # make violin plots of the percent change in outbreak risk
    # the scenarios are
    scenarios = {
        "Existing NME bans": "add_nmes_back",
        "Highest risk states": "top_risk_state_unconditioned",
        "Neighboring counties": "bordering_high_risk_state",
        "Highest NME states": "top_nme_state",
        "Highest risk counties": "top_ten_pct_counties",
    }

    for idx, (scenario, column) in enumerate(scenarios.items()):
        scenario_df = df.filter(pl.col(column) == True)
        if remove_overlap:
            if scenario == "Highest NME states":
                scenario_df = scenario_df.filter(
                    ~(pl.col("top_ten_pct_counties") == True)
                )
                scenario_df = scenario_df.filter(
                    ~(pl.col("top_risk_state_unconditioned") == True)
                )
        if plot_total:
            metric = "no_outbreak_samples_overall"
        else:
            metric = "no_outbreak_samples"
        violinparts = ax.violinplot(
            scenario_df[metric].drop_nans() * 100,
            positions=[idx],
            showextrema=False,
        )

        for pc in violinparts["bodies"]:
            pc.set(facecolor="grey", alpha=0.4)

        ax.errorbar(
            x=[idx],
            y=scenario_df[metric].drop_nans().median() * 100,
            yerr=100
            * np.array(
                [
                    scenario_df[metric].drop_nans().median()
                    - scenario_df[metric].drop_nans().quantile(0.25),
                    scenario_df[metric].drop_nans().quantile(0.75)
                    - scenario_df[metric].drop_nans().median(),
                ]
            )[:, None],
            fmt="o",
            capsize=5,
            capthick=2,
            color="grey",
        )

    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(list(scenarios.keys()), rotation=-45, ha="left")
    ax.set_ylabel("Probability of no outbreak (%)")
    ax.set_ylim(0)


if __name__ == "__main__":
    # map of risk
    fig, ax = setup_plot(1, 3)  # type: ignore

    df = pl.read_csv(
        os.path.join(output_dir, "nme_policy_analysis_assembled_median.csv"),
        schema_overrides={
            "estimated_mmr_cov_raw": pl.Float64,
            "nonmed_exempt_pct_raw": pl.Float64,
        },
    )

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
    plot_data = median_percent_change(
        df,
        fig,
        ax[0],  # type: ignore
    )

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
    plot_data = scenario_results(
        df,
        fig,
        ax[1],  # type: ignore
        plot_total=False,
        remove_overlap=True,
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
    plot_data = scenario_results(
        df,
        fig,
        ax[2],  # type: ignore
        plot_total=True,
        remove_overlap=False,
    )

    fig.tight_layout()
    # save
    fig.savefig(
        os.path.join(
            fig_dir,
            "figure_supplemental_nonmedical_exemption_policy_effects.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(
            fig_dir,
            "figure_supplemental_nonmedical_exemption_policy_effects.pdf",
        ),
        dpi=300,
        bbox_inches="tight",
    )
