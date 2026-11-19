import os

from numpyro import enable_x64

enable_x64()

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy.stats as stats
import yaml
from matplotlib import rc

from outbreak_size import adjacency
from outbreak_size.coverage_effect import (
    effect_label,
    percent_change_per_pp,
)
from outbreak_size.model_colors import EARLIER_PERIOD_COLOR, LATER_PERIOD_COLOR
from outbreak_size.msr import spearman_msr_blocked
from outbreak_size.samples_io import load_samples

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
# the scale of the vaccination covariates of the fits (see params.yml)
COVARIATE_SCALE = params["covariate_scale"]
MMR_EFFECTIVENESS = params.get("mmr_effectiveness")


def setup_plot(nrows, ncols):
    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 10, nrows * 8.5))
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


def in_sorted_fips_order(
    adjacency_matrix, fips_in_matrix_order, county_ids, label
):
    """
    The adjacency matrix with its rows and columns in sorted-fips order, the
    order of the np.unique county indices that spearman_msr_blocked uses (it
    raises if the counties of the matrix are not those of county_ids). The
    matrices are built in the order the counties first appear in the data,
    which the neighbor counting below relies on, so only the MSR test reorders.
    """
    adjacency.require_county_order(
        sorted(fips_in_matrix_order), county_ids, label
    )
    order = sorted(
        range(len(fips_in_matrix_order)), key=lambda i: fips_in_matrix_order[i]
    )
    return adjacency_matrix[np.ix_(order, order)]


def plot_coverage_distribution(
    df,
    vaccination_threshold,
    fig,
    ax,
    vaccination_metric="estimated_mmr_cov",
    adjacency_matrix_pre=None,
    adjacency_matrix_post=None,
):
    # want to make violin plots by outbreak size category and then subfaceted by year
    df = df.with_columns(
        outbreak_size_buckets=pl.col("cases_outbreak").cut(
            [0, 9],
            labels=["-1", "0", "1"],
            left_closed=False,
        ),
        year_of_interest=pl.col("year").is_in([2021, 2022, 2023, 2024, 2025]),
    )
    df_correlation = df
    print("Spearman correlation (2011-2020):")
    print(
        stats.spearmanr(
            df_correlation.filter(pl.col("year_of_interest") == False)[
                vaccination_metric
            ],
            df_correlation.filter(pl.col("year_of_interest") == False)[
                "cases_outbreak"
            ],
        )
    )

    print("Spearman correlation (2021-2025):")
    print(
        stats.spearmanr(
            df_correlation.filter(pl.col("year_of_interest") == True)[
                vaccination_metric
            ],
            df_correlation.filter(pl.col("year_of_interest") == True)[
                "cases_outbreak"
            ],
        )
    )

    if adjacency_matrix_pre is not None:
        county_ids = df_correlation.filter(
            pl.col("year_of_interest") == False
        )["fips"].to_numpy()
        unique_ids = np.unique(county_ids)
        county_id_to_index = {cid: i for i, cid in enumerate(unique_ids)}
        county_indices = np.array(
            [county_id_to_index[cid] for cid in county_ids]
        )

        print("MSR blocked (2011-2020):")
        print(
            spearman_msr_blocked(
                df_correlation.filter(pl.col("year_of_interest") == False)[
                    vaccination_metric
                ],
                df_correlation.filter(pl.col("year_of_interest") == False)[
                    "cases_outbreak"
                ],
                in_sorted_fips_order(
                    adjacency_matrix_pre,
                    df_correlation.filter(~pl.col("year_of_interest"))["fips"]
                    .unique(maintain_order=True)
                    .to_list(),
                    county_ids,
                    "imputed 2011-2020",
                ),
                county_indices=county_indices,
            )
        )

    # the 2021-2025 p-value of the MSR-blocked Spearman test sets the stars of
    # the bracket below (no bracket without it or when it is above 0.05)
    p_post = None
    if adjacency_matrix_post is not None:
        county_ids = df_correlation.filter(pl.col("year_of_interest") == True)[
            "fips"
        ].to_numpy()
        unique_ids = np.unique(county_ids)
        county_id_to_index = {cid: i for i, cid in enumerate(unique_ids)}
        county_indices = np.array(
            [county_id_to_index[cid] for cid in county_ids]
        )

        print("MSR blocked (2021-2025):")
        msr_post = spearman_msr_blocked(
            df_correlation.filter(pl.col("year_of_interest") == True)[
                vaccination_metric
            ],
            df_correlation.filter(pl.col("year_of_interest") == True)[
                "cases_outbreak"
            ],
            in_sorted_fips_order(
                adjacency_matrix_post,
                df_correlation.filter(pl.col("year_of_interest"))["fips"]
                .unique(maintain_order=True)
                .to_list(),
                county_ids,
                "imputed 2021-2025",
            ),
            county_indices=county_indices,
        )
        print(msr_post)
        p_post = msr_post[1]

    df = df.group_by(["outbreak_size_buckets", "year_of_interest"]).agg(
        pl.col(vaccination_metric), pl.col("cases_outbreak")
    )
    df = df.sort(by=["outbreak_size_buckets", "year_of_interest"])
    df = df.with_columns(
        avg_cov=pl.col(vaccination_metric).list.mean(),
        median_cov=pl.col(vaccination_metric).list.median(),
        lower_cov=pl.col(vaccination_metric).map_elements(
            lambda x: np.percentile(x, 25), return_dtype=pl.Float64
        ),
        upper_cov=pl.col(vaccination_metric).map_elements(
            lambda x: np.percentile(x, 75), return_dtype=pl.Float64
        ),
    )

    data_pre = (
        df.filter(pl.col("year_of_interest") == False)
        .to_dict()[vaccination_metric]
        .to_list()
    )
    data_post = (
        df.filter(pl.col("year_of_interest") == True)
        .to_dict()[vaccination_metric]
        .to_list()
    )
    medians_pre = (
        df.filter(pl.col("year_of_interest") == False)
        .to_dict()["median_cov"]
        .to_numpy()
    )
    medians_post = (
        df.filter(pl.col("year_of_interest") == True)
        .to_dict()["median_cov"]
        .to_numpy()
    )
    lowers_pre = (
        df.filter(pl.col("year_of_interest") == False)
        .to_dict()["lower_cov"]
        .to_numpy()
    )
    lowers_post = (
        df.filter(pl.col("year_of_interest") == True)
        .to_dict()["lower_cov"]
        .to_numpy()
    )
    uppers_pre = (
        df.filter(pl.col("year_of_interest") == False)
        .to_dict()["upper_cov"]
        .to_numpy()
    )
    uppers_post = (
        df.filter(pl.col("year_of_interest") == True)
        .to_dict()["upper_cov"]
        .to_numpy()
    )
    means_pre = (
        df.filter(pl.col("year_of_interest") == False)
        .to_dict()["avg_cov"]
        .to_list()
    )
    means_post = (
        df.filter(pl.col("year_of_interest") == True)
        .to_dict()["avg_cov"]
        .to_list()
    )

    # boxplot and swarmplot of coverage distributions
    # SWAPPED: Now plotting horizontally with bigger outbreaks higher up
    offset = 0.2

    # Reverse the order so bigger outbreaks are higher up (at higher y values)
    data_pre_reversed = data_pre[::-1]
    data_post_reversed = data_post[::-1]
    medians_pre_reversed = medians_pre[::-1]
    medians_post_reversed = medians_post[::-1]
    lowers_pre_reversed = lowers_pre[::-1]
    lowers_post_reversed = lowers_post[::-1]
    uppers_pre_reversed = uppers_pre[::-1]
    uppers_post_reversed = uppers_post[::-1]
    means_pre_reversed = list(reversed(means_pre))
    means_post_reversed = list(reversed(means_post))

    n_categories = len(data_pre_reversed)
    # Map positions: we want position 4 at the top of the plot
    # So we use (n_categories - 1 - i) for positioning
    positions_reversed = [n_categories - 1 - i for i in range(n_categories)]

    # 2011-2020 (green) goes ABOVE (+offset), 2021-2025 (blue) goes BELOW (-offset)
    violinparts1 = ax.violinplot(
        data_pre_reversed,
        positions=[p + offset for p in positions_reversed],
        widths=0.35,
        showextrema=False,
        vert=False,
    )

    for pc in violinparts1["bodies"]:
        pc.set(facecolor=EARLIER_PERIOD_COLOR)

    ax.axvline(x=vaccination_threshold, color="k", linestyle="--")

    ax.errorbar(
        y=[p + offset for p in positions_reversed],
        x=medians_pre_reversed,
        xerr=[
            medians_pre_reversed - lowers_pre_reversed,
            uppers_pre_reversed - medians_pre_reversed,
        ],
        fmt="o",
        capsize=10,
        capthick=2,
        color=EARLIER_PERIOD_COLOR,
    )

    # add means as stars
    ax.scatter(
        means_pre_reversed,
        [p + offset for p in positions_reversed],
        marker="*",
        s=90,
        color=EARLIER_PERIOD_COLOR,
    )

    violinparts2 = ax.violinplot(
        data_post_reversed,
        positions=[p - offset for p in positions_reversed],
        widths=0.35,
        showextrema=False,
        vert=False,
    )

    for pc in violinparts2["bodies"]:
        pc.set(facecolor=LATER_PERIOD_COLOR)

    ax.errorbar(
        y=[p - offset for p in positions_reversed],
        x=medians_post_reversed,
        xerr=[
            medians_post_reversed - lowers_post_reversed,
            uppers_post_reversed - medians_post_reversed,
        ],
        fmt="o",
        capsize=10,
        capthick=2,
        color=LATER_PERIOD_COLOR,
    )

    # add means as stars
    ax.scatter(
        means_post_reversed,
        [p - offset for p in positions_reversed],
        marker="*",
        s=90,
        color=LATER_PERIOD_COLOR,
    )

    # Significance bar over the 2021-2025 outbreak sizes, with the stars of
    # the p-value of the MSR-blocked Spearman test of coverage against
    # outbreak size (computed above): * < 0.05, ** < 0.01, *** < 0.001; no bar
    # when it is not below 0.05
    stars = (
        ""
        if p_post is None
        else "***"
        if p_post < 0.001
        else "**"
        if p_post < 0.01
        else "*"
        if p_post < 0.05
        else ""
    )
    if stars:
        # ≥10 is at index 0 of reversed arrays, 0 cases at index 2
        y1 = (
            positions_reversed[0] - offset
        )  # Position for "≥10" cases (blue/post) - at top
        y2 = (
            positions_reversed[2] - offset
        )  # Position for "0" cases (blue/post) - at bottom

        # Get the max x-value from both distributions to place the bar to the right
        x_max = max(
            max(data_post_reversed[0])
            if len(data_post_reversed[0]) > 0
            else 0,  # ≥10 cases
            max(data_post_reversed[2])
            if len(data_post_reversed[2]) > 0
            else 0,  # 0 cases
        )
        bar_position = x_max + 2

        # Draw the significance bar (now vertical)
        ax.plot(
            [
                bar_position - 0.5,
                bar_position,
                bar_position,
                bar_position - 0.5,
            ],
            [y1, y1, y2, y2],
            "k-",
            linewidth=1.5,
        )

        y_middle = (
            positions_reversed[1] - offset
        )  # Position for "1-9" cases (blue/post)
        ax.plot(
            [bar_position - 0.5, bar_position],
            [y_middle, y_middle],
            "k-",
            linewidth=1.5,
        )

        # Add the stars in the middle
        ax.text(
            bar_position + 0.5,
            (y1 + y2) / 2,
            stars,
            ha="left",
            va="center",
            fontsize=16,
            fontweight="bold",
            # rotate the text to be vertical
            rotation=90,
        )

    ax.set_yticks(positions_reversed)
    ax.set_yticklabels(
        [
            "≥10",
            "1-9",
            "0",
        ]
    )

    ax.legend(
        [violinparts1["bodies"][0], violinparts2["bodies"][0]],
        ["2011-2020", "2021-2025"],
        loc=(0.01, 0.75),
        edgecolor="black",
    )
    ax.set_xlabel("MMR vaccine coverage (%)")
    ax.set_ylabel("Outbreak size (N)")

    return fig, ax


def plot_coverage_various_models(
    df,
    fig,
    ax,
    year_buckets,
    model_name="All covariates",
    file_tag="_IMPUTED",
    reference_rows=None,
):
    """
    Plot coverage effects from models, specifically for imputed data.
    `df` holds the rows the fits were made on, `file_tag` ends the names of
    their files, and the % change per 1pp is averaged over `reference_rows`
    (all of `df` unless given).
    """
    if reference_rows is None:
        reference_rows = df
    assembled_effects = pl.DataFrame({"years": year_buckets})
    assembled_effects = assembled_effects.with_columns(
        year_range=pl.col("years").map_elements(
            lambda x: f"{min(x)}-{max(x)}", return_dtype=pl.String
        ),
        median_coverage_effect=pl.lit(None),
        lower_coverage_effect=pl.lit(None),
        upper_coverage_effect=pl.lit(None),
        mean_coverage_effect=pl.lit(None),
    ).with_columns(
        year_index=pl.when(pl.col("year_range") == "2011-2020")
        .then(0)
        .otherwise(1)
    )

    for year in year_buckets:
        posterior_samples = load_samples(
            os.path.join(
                output_dir,
                f"samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_{model_name}_{year}{file_tag}.npz",
            )
        )

        coverage_effects = posterior_samples["iv_global"][:, 0]
        # scale by the standard deviation of the coverage to get an
        # interpretable term
        # % change per 1pp, averaged over the reference county-years
        coverage_effects = percent_change_per_pp(
            coverage_effects,
            df.filter(pl.col("year").is_in(year)),
            "estimated_mmr_cov",
            reference_rows,
            COVARIATE_SCALE,
            MMR_EFFECTIVENESS,
        )
        median_effect = float(jnp.median(coverage_effects, axis=0))
        lower_effect = float(jnp.quantile(coverage_effects, 0.025, axis=0))
        upper_effect = float(jnp.quantile(coverage_effects, 0.975, axis=0))
        mean_effect = float(jnp.mean(coverage_effects, axis=0))

        assembled_effects = assembled_effects.with_columns(
            median_coverage_effect=pl.when(pl.col("years") == year)
            .then(median_effect)
            .otherwise(pl.col("median_coverage_effect")),
            lower_coverage_effect=pl.when(pl.col("years") == year)
            .then(lower_effect)
            .otherwise(pl.col("lower_coverage_effect")),
            upper_coverage_effect=pl.when(pl.col("years") == year)
            .then(upper_effect)
            .otherwise(pl.col("upper_coverage_effect")),
            mean_coverage_effect=pl.when(pl.col("years") == year)
            .then(mean_effect)
            .otherwise(pl.col("mean_coverage_effect")),
        )

    # dashed line at 0
    ax.axhline(y=0, color="k", linestyle="--")

    model_effects = assembled_effects.sort("year_index")
    ax.errorbar(
        [0, 1],
        model_effects["median_coverage_effect"].to_list(),
        yerr=[
            model_effects["median_coverage_effect"].to_numpy()
            - model_effects["lower_coverage_effect"].to_numpy(),
            model_effects["upper_coverage_effect"].to_numpy()
            - model_effects["median_coverage_effect"].to_numpy(),
        ],
        fmt="o",
        color="black",
        zorder=10,
    )
    print(
        f"{model_name} median effects: {model_effects['median_coverage_effect'].to_list()}"
    )
    print(
        f"{model_name} 95% CI: {list(zip(model_effects['lower_coverage_effect'].to_list(), model_effects['upper_coverage_effect'].to_list()))}"
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["2011-2020", "2021-2025"])
    ax.set_xlim([-0.18, 1.18])

    ax.set_xlabel("Year")
    ax.set_ylabel(effect_label())

    return fig, ax


def plot_car_effect_correlation_simple(samples, fig, ax):
    """
    Simplified version of plot_car_effect_correlation for use in multi-panel figures.
    Just plots the posterior histogram without the prior overlay.
    """
    # Transform correlation from [0,1] to [-1,1]
    spatial_correlation = 2 * samples["spatial_correlation"] - 1

    ax.hist(
        spatial_correlation,
        bins=100,
        density=True,
        color="green",
        alpha=0.7,
        label="Posterior",
    )

    ax.axvline(
        x=jnp.percentile(spatial_correlation, 2.5),
        color="red",
    )
    print(
        f"Spatial correlation 2.5%: {jnp.percentile(spatial_correlation, 2.5)}"
    )
    print(f"Spatial correlation median: {jnp.median(spatial_correlation)}")
    print(
        f"Spatial correlation 97.5%: {jnp.percentile(spatial_correlation, 97.5)}"
    )
    ax.axvline(
        x=jnp.percentile(spatial_correlation, 97.5),
        color="red",
    )
    ax.axvline(
        x=jnp.median(spatial_correlation),
        color="black",
    )
    ax.axvline(x=0, color="blue", linestyle="--")

    ax.set_xlabel("Spatial correlation")
    ax.set_ylabel("Probability density")
    x = np.linspace(-1, 1, 1000)
    prior = stats.beta.pdf((x + 1) / 2, 2, 2) / 2
    ax.plot(x, prior, color="grey", label="Prior")
    ax.fill_between(x, 0, prior, color="grey", alpha=0.3)
    ax.legend(edgecolor="black", loc="upper left")
    ax.set_xlim([-1, 1])

    return fig, ax


if __name__ == "__main__":
    if not os.path.exists(
        os.path.join(data_dir, "analysis_dataset_imputed.csv")
    ):
        print(
            "skipped: analysis_dataset_imputed.csv not provided, so the imputed-data models were not run."
        )
        raise SystemExit(0)
    # Load imputed dataset
    df_imputed = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset_imputed.csv"),
        schema_overrides={"fips": pl.String},
    )

    # Load adjacency matrix
    adjacency_data = pl.read_csv(
        os.path.join(data_dir, "county_adjacencies.txt"),
        separator="|",
        schema_overrides={
            "County GEOID": pl.String,
            "Neighbor GEOID": pl.String,
        },
    )

    # Create adjacency matrices for pre and post COVID periods
    adjacency_matrix_observed_post = adjacency.assemble_adjacency_matrix(
        adjacency_data,
        df_imputed.filter(pl.col("year") >= 2021)["fips"]
        .unique(maintain_order=True)
        .to_list(),
    )
    adjacency_matrix_observed_pre = adjacency.assemble_adjacency_matrix(
        adjacency_data,
        df_imputed.filter(pl.col("year") < 2021)["fips"]
        .unique(maintain_order=True)
        .to_list(),
    )

    # Load posterior samples for imputed data
    print("\n" + "=" * 80)
    print("LOADING POSTERIOR SAMPLES FOR IMPUTED DATA")
    print("=" * 80)

    # Post-COVID period (2021-2025)
    posterior_samples_post_imputed = load_samples(
        os.path.join(
            output_dir,
            "samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_[2021, 2022, 2023, 2024, 2025]_IMPUTED.npz",
        )
    )

    # Create figure with 1 row, 3 columns
    fig, ax = setup_plot(1, 3)  # type: ignore

    # Panel a: Coverage distribution violin plot
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
    plot_coverage_distribution(
        df_imputed,
        95,
        fig,
        ax[0],  # type: ignore
        vaccination_metric="estimated_mmr_cov",
        adjacency_matrix_pre=adjacency_matrix_observed_pre,
        adjacency_matrix_post=adjacency_matrix_observed_post,
    )

    # Panel b: Coverage effects from models
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

    year_buckets = [
        [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
        [2021, 2022, 2023, 2024, 2025],
    ]

    plot_coverage_various_models(
        df_imputed,
        fig,
        ax[1],  # type: ignore
        year_buckets=year_buckets,
        model_name="All covariates",
    )

    # Panel c: Spatial correlation posterior (post-COVID period)
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

    plot_car_effect_correlation_simple(
        posterior_samples_post_imputed,
        fig,
        ax[2],  # type: ignore
    )

    fig.tight_layout()

    # Save figure
    fig.savefig(
        os.path.join(
            fig_dir, "figure_supplemental_imputed_inferential_analysis.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(
            fig_dir, "figure_supplemental_imputed_inferential_analysis.pdf"
        ),
        dpi=300,
        bbox_inches="tight",
    )

    print("\n" + "=" * 80)
    print("FIGURE SAVED SUCCESSFULLY!")
    print("=" * 80)
    print(
        f"Location: {os.path.join(fig_dir, 'figure_supplemental_imputed_inferential_analysis.png')}"
    )
