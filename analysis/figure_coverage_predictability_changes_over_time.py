import os

from numpyro import enable_x64

enable_x64()

import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import polars as pl
import scipy.stats as stats
import yaml
from matplotlib import rc

from outbreak_size import adjacency
from outbreak_size.coverage_effect import (
    cases_change_per_pp,
    covariate_sd,
    effect_label,
    factor_per_pp,
    marginal_percent_change_per_pp,
    percent_change,
    percent_change_per_pp,
    posterior_contrast,
)
from outbreak_size.model_colors import (
    COVARIATE_SET_COLORS,
    EARLIER_PERIOD_COLOR,
    LATER_PERIOD_COLOR,
    POISSON_COLOR,
)
from outbreak_size.msr import spearman_msr_blocked, spearman_msr_interval
from outbreak_size.samples_io import load_samples
from outbreak_size.trend import trend_draws

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
# the scale of the coverage covariate of the main fits, and the other scale
# that 02 fits for the baseline models (see params.yml)
COVARIATE_SCALE = params["covariate_scale"]
OTHER_SCALE = "raw" if COVARIATE_SCALE == "natural" else "natural"
MMR_EFFECTIVENESS = params.get("mmr_effectiveness")
COVERAGE = "estimated_mmr_cov"
# The y axes are the % change in outbreak size per percentage point of
# coverage. On the natural scale that depends on the coverage it starts from
# (and on the assumed effectiveness of the vaccine), so it is averaged over the
# same county-years for every period, year and fit: all of 2011-2025, each at
# its own coverage (see outbreak_size.coverage_effect)
POISSON = "All covariates, Poisson likelihood"
POISSON_FIT_TAG = (
    "likelihood=poisson_covariate_scale=raw_pop_exponent=1_pop_density"
)

# make adjacency matrix
df = pl.read_csv(
    os.path.join(data_dir, "analysis_dataset.csv"),
    # make sure we read fips codes as strings
    schema_overrides={"fips": pl.String},
)
df = df  # .filter(pl.col("cases_outbreak") >= 1)
# read adjacency data
adjacency_matrix = pl.read_csv(
    os.path.join(data_dir, "county_adjacencies.txt"),
    separator="|",
    schema_overrides={
        "County GEOID": pl.String,
        "Neighbor GEOID": pl.String,
    },
)
# sorted so these line up with the np.unique-ordered county_indices used below
fips_post = sorted(
    df.filter(pl.col("year") >= 2021)["fips"].unique().to_list()
)
adjacency_matrix_observed_post = adjacency.assemble_adjacency_matrix(
    adjacency_matrix, fips_post
)
fips_pre = sorted(df.filter(pl.col("year") < 2021)["fips"].unique().to_list())
adjacency_matrix_observed_pre = adjacency.assemble_adjacency_matrix(
    adjacency_matrix, fips_pre
)
all_fips_sorted = sorted(df["fips"].unique().to_list())
adjacency_matrix_observed_all = adjacency.assemble_adjacency_matrix(
    adjacency_matrix, all_fips_sorted
)


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


def significance_stars(p):
    """* p < 0.05, ** p < 0.01, *** p < 0.001, n.s. otherwise."""
    for limit, stars in [(0.001, "***"), (0.01, "**"), (0.05, "*")]:
        if p < limit:
            return stars
    return "n.s."


def print_spearman_msr(period, result):
    """
    Print the (rho, p-value, null) that spearman_msr_blocked returns, with the
    95% interval of rho from the MSR null (see spearman_msr_interval).
    """
    rho, p, null = result
    low, high, critical = spearman_msr_interval(rho, null)
    print(
        f"Spearman rho, coverage vs outbreak size, {period}, spatially "
        f"constrained permutation test ({len(null)} permutations): "
        f"{rho:.3f}, p = {p:.5f}; null SD {np.std(null):.3f}, 95% interval "
        f"from the null [{low:.3f}, {high:.3f}] (rho +/- {critical:.3f}; "
        "excludes 0 exactly when p <= 0.05)"
    )


def print_spearman_difference(
    df, vaccination_metric, n_bootstrap=10000, seed=2026
):
    """
    Difference between the 2021-2025 and 2011-2020 Spearman correlations of
    coverage vs outbreak size (post minus pre), with a county bootstrap: whole
    counties are resampled with replacement, and the same resampled counties
    feed both periods, so counties seen in both periods stay paired and
    within-county repeated years stay together. df needs the columns fips,
    cases_outbreak, year_of_interest and vaccination_metric.
    """
    x = df[vaccination_metric].to_numpy()
    y = df["cases_outbreak"].to_numpy().astype(float)
    post = df["year_of_interest"].to_numpy()
    fips = df["fips"].to_numpy()
    counties = np.unique(fips)
    rows_by_county = {c: np.where(fips == c)[0] for c in counties}

    def rho_pre_post(rows) -> tuple[float, float]:
        m = post[rows]
        return (
            float(stats.spearmanr(x[rows][~m], y[rows][~m])[0]),  # type: ignore
            float(stats.spearmanr(x[rows][m], y[rows][m])[0]),  # type: ignore
        )

    rho_pre, rho_post = rho_pre_post(np.arange(len(x)))
    rng = np.random.default_rng(seed)
    boot = np.empty((n_bootstrap, 2))
    for b in range(n_bootstrap):
        draw = rng.choice(counties, size=len(counties), replace=True)
        boot[b] = rho_pre_post(
            np.concatenate([rows_by_county[c] for c in draw])
        )
    diff = boot[:, 1] - boot[:, 0]
    lo, hi = np.nanpercentile(diff, [2.5, 97.5])
    # two-sided bootstrap p-value: twice the smaller tail
    p = max(
        2 * min(np.mean(diff >= 0), np.mean(diff <= 0)),  # type: ignore
        2 / (n_bootstrap + 1),
    )
    ci = lambda v: np.nanpercentile(v, [2.5, 97.5])
    print(
        f"Spearman rho, coverage vs outbreak size, county bootstrap "
        f"({n_bootstrap} resamples, seed {seed}):\n"
        f"  2011-2020: {rho_pre:.3f} [{ci(boot[:, 0])[0]:.3f}, {ci(boot[:, 0])[1]:.3f}]\n"
        f"  2021-2025: {rho_post:.3f} [{ci(boot[:, 1])[0]:.3f}, {ci(boot[:, 1])[1]:.3f}]\n"
        f"  difference (2021-2025 minus 2011-2020): {rho_post - rho_pre:.3f} "
        f"[{lo:.3f}, {hi:.3f}], bootstrap SE {np.nanstd(diff):.3f}, "
        f"two-sided p = {p:.3f}"
    )


def plot_coverage_distribution(
    df,
    vaccination_threshold,
    fig,
    ax,
    vaccination_metric="estimated_mmr_cov",
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
    df_correlation = df  # .filter(pl.col("cases_outbreak") >= 1)
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

    county_ids = df_correlation.filter(pl.col("year_of_interest") == False)[
        "fips"
    ].to_numpy()
    unique_ids = np.unique(county_ids)
    county_id_to_index = {cid: i for i, cid in enumerate(unique_ids)}
    county_indices = np.array([county_id_to_index[cid] for cid in county_ids])
    adjacency.require_county_order(fips_pre, county_ids, "2011-2020")

    print_spearman_msr(
        "2011-2020",
        spearman_msr_blocked(
            df_correlation.filter(pl.col("year_of_interest") == False)[
                vaccination_metric
            ],
            df_correlation.filter(pl.col("year_of_interest") == False)[
                "cases_outbreak"
            ],
            adjacency_matrix_observed_pre,
            county_indices=county_indices,
        ),
    )

    county_ids = df_correlation.filter(pl.col("year_of_interest") == True)[
        "fips"
    ].to_numpy()
    unique_ids = np.unique(county_ids)
    county_id_to_index = {cid: i for i, cid in enumerate(unique_ids)}
    county_indices = np.array([county_id_to_index[cid] for cid in county_ids])
    adjacency.require_county_order(fips_post, county_ids, "2021-2025")

    result_post = spearman_msr_blocked(
        df_correlation.filter(pl.col("year_of_interest") == True)[
            vaccination_metric
        ],
        df_correlation.filter(pl.col("year_of_interest") == True)[
            "cases_outbreak"
        ],
        adjacency_matrix_observed_post,
        county_indices=county_indices,
    )
    print_spearman_msr("2021-2025", result_post)
    # the stars of the bracket: the spatially constrained p of this test
    stars = significance_stars(result_post[1])

    print_spearman_difference(df_correlation, vaccination_metric)

    df = df.group_by(["outbreak_size_buckets", "year_of_interest"]).agg(
        pl.col(vaccination_metric), pl.col("cases_outbreak")
    )
    df = df.sort(by=["outbreak_size_buckets", "year_of_interest"])
    df = df.with_columns(
        avg_cov=pl.col(vaccination_metric).list.mean(),
        median_cov=pl.col(vaccination_metric).list.median(),
        lower_cov=pl.col(vaccination_metric).map_elements(
            lambda x: np.percentile(x, 25)
        ),
        upper_cov=pl.col(vaccination_metric).map_elements(
            lambda x: np.percentile(x, 75)
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
    # Original order: 0, 1-4, 5-9, 10-19, ≥20 (indices 0-4)
    # We want: ≥20, 10-19, 5-9, 1-4, 0 (so index 0 is at top)
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

    # Add significance bar between post-COVID 0 cases and ≥20 cases
    # ≥20 is at index 0 of reversed arrays, which maps to position 3
    # 0 cases is at index 2 of reversed arrays, which maps to position 0
    y1 = (
        positions_reversed[0] - offset
    )  # Position for "≥20" cases (blue/post) - at top
    y2 = (
        positions_reversed[2] - offset
    )  # Position for "0" cases (blue/post) - at bottom

    # Get the max x-value from both distributions to place the bar to the right
    x_max = max(
        max(data_post_reversed[0])
        if len(data_post_reversed[0]) > 0
        else 0,  # ≥20 cases
        max(data_post_reversed[2])
        if len(data_post_reversed[2]) > 0
        else 0,  # 0 cases (changed from [4])
    )
    bar_position = x_max + 2

    # Draw the significance bar (now vertical)
    ax.plot(
        [bar_position - 0.5, bar_position, bar_position, bar_position - 0.5],
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

    # Add the asterisk in the middle
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
    # ax.set_xlim(right=100)

    return fig, ax


def plot_coverage_distribution_all_yrs(
    df,
    vaccination_threshold,
    fig,
    ax,
    vaccination_metric="estimated_mmr_cov",
):
    """
    Same as plot_coverage_distribution's panel, but pooling all years
    (no pre/post-COVID split).
    """
    df = df.with_columns(
        outbreak_size_buckets=pl.col("cases_outbreak").cut(
            [0, 9],
            labels=["-1", "0", "1"],
            left_closed=False,
        ),
    )
    df_correlation = df
    print(
        stats.spearmanr(
            df_correlation[vaccination_metric],
            df_correlation["cases_outbreak"],
        )
    )

    county_ids = df_correlation["fips"].to_numpy()
    unique_ids = np.unique(county_ids)
    county_id_to_index = {cid: i for i, cid in enumerate(unique_ids)}
    county_indices = np.array([county_id_to_index[cid] for cid in county_ids])
    adjacency.require_county_order(all_fips_sorted, county_ids, "all years")

    print_spearman_msr(
        "all years",
        spearman_msr_blocked(
            df_correlation[vaccination_metric],
            df_correlation["cases_outbreak"],
            adjacency_matrix_observed_all,
            county_indices=county_indices,
        ),
    )

    df = df.group_by(["outbreak_size_buckets"]).agg(
        pl.col(vaccination_metric), pl.col("cases_outbreak")
    )
    df = df.sort(by=["outbreak_size_buckets"])
    df = df.with_columns(
        avg_cov=pl.col(vaccination_metric).list.mean(),
        median_cov=pl.col(vaccination_metric).list.median(),
        lower_cov=pl.col(vaccination_metric).map_elements(
            lambda x: np.percentile(x, 25)
        ),
        upper_cov=pl.col(vaccination_metric).map_elements(
            lambda x: np.percentile(x, 75)
        ),
    )

    data = df.to_dict()[vaccination_metric].to_list()
    medians = df.to_dict()["median_cov"].to_numpy()
    lowers = df.to_dict()["lower_cov"].to_numpy()
    uppers = df.to_dict()["upper_cov"].to_numpy()
    means = df.to_dict()["avg_cov"].to_list()

    # reverse order so bigger outbreaks are higher up, matching panel a
    data_reversed = data[::-1]
    medians_reversed = medians[::-1]
    lowers_reversed = lowers[::-1]
    uppers_reversed = uppers[::-1]
    means_reversed = list(reversed(means))

    n_categories = len(data_reversed)
    positions_reversed = [n_categories - 1 - i for i in range(n_categories)]

    violinparts = ax.violinplot(
        data_reversed,
        positions=positions_reversed,
        widths=0.35,
        showextrema=False,
        vert=False,
    )
    for pc in violinparts["bodies"]:
        pc.set(facecolor="grey")

    ax.axvline(x=vaccination_threshold, color="k", linestyle="--")

    ax.errorbar(
        y=positions_reversed,
        x=medians_reversed,
        xerr=[
            medians_reversed - lowers_reversed,
            uppers_reversed - medians_reversed,
        ],
        fmt="o",
        capsize=10,
        capthick=2,
        color="grey",
    )

    ax.scatter(
        means_reversed,
        positions_reversed,
        marker="*",
        s=90,
        color="grey",
    )

    ax.set_yticks(positions_reversed)
    ax.set_yticklabels(
        [
            "≥10",
            "1-9",
            "0",
        ]
    )
    ax.set_xlabel("MMR vaccine coverage (%)")
    ax.set_ylabel("Outbreak size (N)")

    return fig, ax


def plot_coverage_distribution_r01(
    df,
    vaccination_threshold,
    fig,
    ax,
    vaccination_metric="estimated_mmr_cov",
):
    """
    R01-only standalone figure, replicating the pre-refactor style of
    `plot_coverage_distribution` (commit a5b9c36): vertical violins, buckets
    0 / 1-3 / 4-9 / >=10 cases, 2011-2020 vs 2021-2025 side by side. Doesn't
    print the Spearman/KS stats that version printed.
    """
    cases_threshold = 10
    df = df.with_columns(
        outbreak_size_buckets=pl.col("cases_outbreak").cut(
            [0, 1, 4, cases_threshold, 1000000000],
            labels=["-1", "0", "1", "2", "3", "4"],
            left_closed=True,
        ),
        year_of_interest=pl.col("year").is_in([2021, 2022, 2023, 2024, 2025]),
    )
    df = df.group_by(
        ["outbreak_size_buckets", "year_of_interest"], maintain_order=True
    ).agg(pl.col(vaccination_metric), pl.col("cases_outbreak"))
    df = df.sort(by=["outbreak_size_buckets", "year_of_interest"])
    df = df.with_columns(
        avg_cov=pl.col(vaccination_metric).list.mean(),
        median_cov=pl.col(vaccination_metric).list.median(),
        lower_cov=pl.col(vaccination_metric).map_elements(
            lambda x: np.percentile(x, 25)
        ),
        upper_cov=pl.col(vaccination_metric).map_elements(
            lambda x: np.percentile(x, 75)
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

    offset = 0.2
    violinparts1 = ax.violinplot(
        data_pre,
        positions=[i - offset for i in range(len(data_pre))],
        widths=0.3,
        showextrema=False,
    )
    for pc in violinparts1["bodies"]:
        pc.set(facecolor=EARLIER_PERIOD_COLOR)

    ax.errorbar(
        x=[i - offset for i in range(len(medians_pre))],
        y=medians_pre,
        yerr=[medians_pre - lowers_pre, uppers_pre - medians_pre],
        fmt="o",
        capsize=10,
        capthick=2,
        color=EARLIER_PERIOD_COLOR,
    )

    ax.scatter(
        [i - offset for i in range(len(means_pre))],
        means_pre,
        marker="*",
        s=90,
        color=EARLIER_PERIOD_COLOR,
    )

    violinparts2 = ax.violinplot(
        data_post,
        positions=[i + offset for i in range(len(data_post))],
        widths=0.3,
        showextrema=False,
    )
    for pc in violinparts2["bodies"]:
        pc.set(facecolor=LATER_PERIOD_COLOR)

    ax.errorbar(
        x=[i + offset for i in range(len(medians_post))],
        y=medians_post,
        yerr=[medians_post - lowers_post, uppers_post - medians_post],
        fmt="o",
        capsize=10,
        capthick=2,
        color=LATER_PERIOD_COLOR,
    )

    ax.scatter(
        [i + offset for i in range(len(means_post))],
        means_post,
        marker="*",
        s=90,
        color=LATER_PERIOD_COLOR,
    )

    ax.set_xticks(range(len(data_pre)))
    ax.set_xticklabels(
        [
            "0",
            "1-3",
            "4-9",
            f"≥{cases_threshold}",
        ]
    )

    ax.legend(
        [violinparts1["bodies"][0], violinparts2["bodies"][0]],
        ["2011-2020", "2021-2025"],
    )
    ax.axhline(y=vaccination_threshold, color="k", linestyle="--")
    ax.set_ylabel("MMR vaccine coverage (%)")
    ax.set_xlabel("Outbreak size (cases)")

    return fig, ax


def plot_cases_vs_coverage_scatter(
    df,
    fig,
    ax,
    vaccination_metric="estimated_mmr_cov",
):
    """
    Scatter plot of outbreak cases vs coverage with symlog y-axis
    Points colored by time period (pre/post COVID)
    """
    df = df.with_columns(
        year_of_interest=pl.col("year").is_in([2021, 2022, 2023, 2024, 2025]),
    )

    df_plot = df  # .filter(pl.col("cases_outbreak") >= 1)

    # Separate data by time period
    df_pre = df_plot.filter(pl.col("year_of_interest") == False)
    df_post = df_plot.filter(pl.col("year_of_interest") == True)

    # Create scatter plots
    ax.scatter(
        df_pre[vaccination_metric].to_numpy(),
        df_pre["cases_outbreak"].to_numpy(),
        alpha=0.5,
        s=50,
        color=EARLIER_PERIOD_COLOR,
        label="2011-2020",
        edgecolors="none",
    )

    ax.scatter(
        df_post[vaccination_metric].to_numpy(),
        df_post["cases_outbreak"].to_numpy(),
        alpha=0.5,
        s=50,
        color=LATER_PERIOD_COLOR,
        label="2021-2025",
        edgecolors="none",
    )

    # Set symlog scale for y-axis
    ax.set_yscale("symlog", linthresh=1)

    # Add reference line at 95% coverage
    ax.axvline(x=95, color="k", linestyle="--", linewidth=1.5, alpha=0.7)

    ax.set_xlabel("MMR vaccine coverage (%)")
    ax.set_ylabel("Outbreak size (N)")
    ax.legend(loc="upper right", edgecolor="black")
    # show every county-year: the axis starts at the lowest coverage, rounded
    # down to a multiple of 5
    lowest = float(df_plot[vaccination_metric].min())  # type: ignore
    ax.set_xlim(left=5 * np.floor(lowest / 5), right=100)

    # Set y-axis to show meaningful ticks
    ax.set_yticks([0, 1, 5, 10, 50, 100, 500, 1000])
    ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())

    return fig, ax


# plot for the vaccination coverage in county-years with >= 10 cases
def plot_coverage_distribution_all_years(
    df,
    vaccination_threshold,
    fig,
    ax,
    vaccination_metric="estimated_mmr_cov",
):
    # want to make violin plots by outbreak size category
    df = df.with_columns(
        outbreak_size_buckets=pl.col("cases_outbreak").cut(
            [0, 4, 9, 19],
            labels=["-1", "0", "1", "2", "3"],
            left_closed=False,
        ),
    )
    df_correlation = df.filter(pl.col("cases_outbreak") >= 1)
    print(
        stats.spearmanr(
            df_correlation[vaccination_metric],
            df_correlation["cases_outbreak"],
        )
    )

    df = df.group_by(["outbreak_size_buckets"]).agg(
        pl.col(vaccination_metric), pl.col("cases_outbreak")
    )
    df = df.sort(by=["outbreak_size_buckets"])
    df = df.with_columns(
        avg_cov=pl.col(vaccination_metric).list.mean(),
        median_cov=pl.col(vaccination_metric).list.median(),
        lower_cov=pl.col(vaccination_metric).map_elements(
            lambda x: np.percentile(x, 25)
        ),
        upper_cov=pl.col(vaccination_metric).map_elements(
            lambda x: np.percentile(x, 75)
        ),
    )

    data = df.to_dict()[vaccination_metric].to_list()
    medians = df.to_dict()["median_cov"].to_numpy()
    lowers = df.to_dict()["lower_cov"].to_numpy()
    uppers = df.to_dict()["upper_cov"].to_numpy()
    means = df.to_dict()["avg_cov"].to_list()

    # boxplot and swarmplot of coverage distributions
    violinparts = ax.violinplot(
        data,
        positions=range(len(data)),
        widths=0.3,
        showextrema=False,
    )

    for pc in violinparts["bodies"]:
        pc.set(facecolor="grey")

    ax.errorbar(
        x=range(len(medians)),
        y=medians,
        yerr=[medians - lowers, uppers - medians],
        fmt="o",
        capsize=10,
        capthick=2,
        color="grey",
    )

    # add means as stars
    ax.scatter(
        range(len(means)),
        means,
        marker="*",
        s=90,
        color="grey",
    )

    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(
        [
            "0",
            "1-4",
            "5-9",
            "10-19",
            "≥20",
        ]
    )

    ax.axhline(y=vaccination_threshold, color="k", linestyle="--")
    ax.set_ylabel("MMR vaccine coverage (%)")
    ax.set_xlabel("Outbreak size (N)")
    ax.set_ylim(top=100)

    return fig, ax


def summary(values, digits=2):
    """median [2.5%, 97.5%] of the draws, as text"""
    low, median, high = np.percentile(values, [2.5, 50, 97.5])
    return f"{median:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def print_coverage_effect_summary(draws, reference_rows, covariate_scale):
    """
    Print what the coverage coefficients mean, for each model and period.
    `draws[(model, period)]` is (the coefficient / SD of each draw, the
    county-years of the period). On the natural scale that is the elasticity:
    the change in log outbreak size per change in log susceptibility.
    """
    natural = covariate_scale == "natural"
    effect_name = "elasticity" if natural else "log effect per pp"
    periods = ["2011-2020", "2021-2025"]
    reference = reference_rows[COVERAGE].to_numpy()
    print(
        f"\n===== coverage effects, {covariate_scale} scale: median "
        "[95% credible interval] ====="
    )
    print(
        "the plotted % change per 1pp is averaged over the same "
        f"{len(reference)} county-years for every period and model "
        f"(2011-2025, each at its own coverage; {int((reference >= 99).sum())} "
        "of them at 99% or more, where 1pp more goes past 100%)"
    )
    contrast_on = {}
    contrast_models = list(dict.fromkeys(model for model, _ in draws))
    for model in contrast_models:
        print(f"\n{model}")
        for period in periods:
            effect, rows = draws[(model, period)]
            coverage = rows[COVERAGE].to_numpy()
            cases = rows["cases_outbreak"].to_numpy()
            per_pp = marginal_percent_change_per_pp(
                effect, reference, covariate_scale, MMR_EFFECTIVENESS
            )
            per_pp_own = marginal_percent_change_per_pp(
                effect, coverage, covariate_scale, MMR_EFFECTIVENESS
            )
            more_cases = cases_change_per_pp(
                effect, cases, coverage, covariate_scale, MMR_EFFECTIVENESS
            )
            print(f"  {period} ({rows.height} county-years)")
            print(f"    {effect_name}: {summary(effect, 3)}")
            if natural:
                print(
                    "    % change in outbreak size per 1% more susceptibility "
                    f"(same at any coverage): {summary(percent_change(effect, 1.0, 'natural'))}"
                )
            print(
                "    % change in outbreak size per 1pp more coverage "
                f"(plotted; reference county-years): {summary(per_pp)}"
            )
            print(
                "    the same over this period's own county-years "
                f"(not comparable between periods): {summary(per_pp_own)}"
            )
            print(
                f"    average outbreak: {cases.mean():.2f} cases at "
                f"{coverage.mean():.1f}% coverage; 1pp more coverage changes "
                f"it by {summary(more_cases)} cases "
                f"({summary(100 * more_cases / cases.mean(), 1)}% of "
                "the average outbreak)"
            )
            contrast_on[(model, period)] = (
                effect,
                per_pp,
                per_pp_own,
                more_cases,
            )
    # do the two periods differ? the fits are independent, so compare every
    # draw of 2021-2025 with every draw of 2011-2020. The per-pp effect over
    # the reference county-years is one increasing function of the effect for
    # both periods, so its p-value is that of the effect; the per-pp effect over
    # each period's own county-years also differs by their coverage, and the
    # cases per pp also by the size of their outbreaks
    for model in [
        m for m in contrast_models if m.startswith("All covariates")
    ]:
        print(
            f"\n{model}, 2021-2025 compared with 2011-2020 (independent "
            "fits; every draw of one against every draw of the other):"
        )
        for position, name in enumerate(
            [
                effect_name,
                "% change per 1pp more coverage (reference county-years)",
                "% change per 1pp more coverage (each period's own "
                "county-years: also differs by their coverage)",
                "cases per 1pp more coverage (each period's own "
                "county-years: also differs by outbreak size)",
            ]
        ):
            before = contrast_on[(model, periods[0])][position]
            after = contrast_on[(model, periods[1])][position]
            out = posterior_contrast(before, after)
            print(
                f"  {name}: P(2021-2025 > 2011-2020) = "
                f"{out['P(after > before)']:.3f}, two-sided empirical p = "
                f"{out['two-sided p']:.4f}; difference "
                f"{out['difference median']:.3f} "
                f"[{out['difference 2.5%']:.3f}, {out['difference 97.5%']:.3f}]"
            )


def plot_coverage_various_models(
    df, fig, ax, covariate_scale=None, models=None
):
    """
    Panel c: the effect of the coverage in each of the two periods for the
    three covariate sets, from the 02 fits on `covariate_scale` (the scale of
    the main fits unless given; the fits of the other scale carry its tag).
    `models` are the fits drawn (the three covariate sets unless given).
    """
    covariate_scale = covariate_scale or COVARIATE_SCALE
    draws = {}
    file_tag = (
        ""
        if covariate_scale == COVARIATE_SCALE
        else f"_covariate_scale={covariate_scale}"
    )
    year_buckets = [
        [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
        [2021, 2022, 2023, 2024, 2025],
    ]
    # POISSON is the all-covariates fit with the Poisson likelihood (02,
    # fit_all_covariates_poisson_raw; its population exponent is fixed at 1
    # and its density is not logged)
    models = models or [
        "MMR coverage",
        "MMR coverage + exemptions",
        "All covariates",
    ]
    assembled_effects = pl.DataFrame({"years": year_buckets}).join(
        pl.DataFrame({"model": models}), how="cross"
    )
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
        for model_name in models:
            fit_name = (
                f"All covariates_{year}_{POISSON_FIT_TAG}"
                if model_name == POISSON
                else f"{model_name}_{year}{file_tag}"
            )
            posterior_samples = load_samples(
                os.path.join(
                    output_dir,
                    f"samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_{fit_name}.npz",
                )
            )

            coverage_effects = posterior_samples["iv_global"][:, 0]
            # the coefficient is per SD of the covariate the model was fit
            # with; this gives the % change in outbreak size per 1pp of
            # coverage, averaged over the county-years of all the years
            rows = df.filter(pl.col("year").is_in(year))
            std = covariate_sd(
                rows, COVERAGE, covariate_scale, MMR_EFFECTIVENESS
            )
            draws[(model_name, f"{year[0]}-{year[-1]}")] = (
                np.asarray(coverage_effects) / std,
                rows,
            )
            coverage_effects = percent_change_per_pp(
                coverage_effects,
                rows,
                COVERAGE,
                df,
                covariate_scale,
                MMR_EFFECTIVENESS,
            )
            median_effect = float(jnp.median(coverage_effects, axis=0))
            lower_effect = float(jnp.quantile(coverage_effects, 0.025, axis=0))
            upper_effect = float(jnp.quantile(coverage_effects, 0.975, axis=0))
            mean_effect = float(jnp.mean(coverage_effects, axis=0))

            assembled_effects = assembled_effects.with_columns(
                median_coverage_effect=pl.when(
                    (pl.col("years") == year) & (pl.col("model") == model_name)
                )
                .then(median_effect)
                .otherwise(pl.col("median_coverage_effect")),
                lower_coverage_effect=pl.when(
                    (pl.col("years") == year) & (pl.col("model") == model_name)
                )
                .then(lower_effect)
                .otherwise(pl.col("lower_coverage_effect")),
                upper_coverage_effect=pl.when(
                    (pl.col("years") == year) & (pl.col("model") == model_name)
                )
                .then(upper_effect)
                .otherwise(pl.col("upper_coverage_effect")),
                mean_coverage_effect=pl.when(
                    (pl.col("years") == year) & (pl.col("model") == model_name)
                )
                .then(mean_effect)
                .otherwise(pl.col("mean_coverage_effect")),
            )

    colors = {**COVARIATE_SET_COLORS, POISSON: POISSON_COLOR}

    # dashed line at 0
    ax.axhline(y=0, color="k", linestyle="--")

    for model_name, offset in zip(
        models,
        np.linspace(-0.07, 0.07, len(models)) if len(models) > 1 else [0.0],
    ):
        model_effects = assembled_effects.filter(
            pl.col("model") == model_name
        ).sort("year_index")
        if model_name.startswith("All covariates"):
            print(
                f"{model_name} median effects: {model_effects['median_coverage_effect'].to_list()}"
            )
            print(
                f"{model_name} 95% CI: {list(zip(model_effects['lower_coverage_effect'].to_list(), model_effects['upper_coverage_effect'].to_list()))}"
            )
        ax.errorbar(
            [0 + offset, 1 + offset],
            model_effects["median_coverage_effect"].to_list(),
            yerr=[
                model_effects["median_coverage_effect"].to_numpy()
                - model_effects["lower_coverage_effect"].to_numpy(),
                model_effects["upper_coverage_effect"].to_numpy()
                - model_effects["median_coverage_effect"].to_numpy(),
            ],
            fmt="o",
            label=model_name,
            color=colors[model_name],
            linewidth=3,
            markersize=10,
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["2011-2020", "2021-2025"])
    ax.set_xlim([-0.18, 1.18])
    # leave the top 38% of the axes to the legend, so it covers no estimate
    low, high = ax.get_ylim()
    ax.set_ylim(low, low + (high - low) / 0.62)

    print_coverage_effect_summary(draws, df, covariate_scale)

    ax.set_xlabel("Year")
    ax.set_ylabel(effect_label())
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),  # a little above the top of the axes
        edgecolor="black",
        title="Model's covariate set:                         ",
    )

    return fig, ax


def plot_nationwide_effect_over_time(
    samples_16_17,
    samples_2025,
    df_16_17,
    df_2025,
    years_in_order,
    fig,
    ax,
    weight_by_mu=False,
):
    """
    Plot the nationwide (population-level) effect of MMR coverage over time.

    If weight_by_mu=False: Extracts the upper-level hierarchical parameter (effects_over_time)
    If weight_by_mu=True: Uses observation-level effects weighted by expected cases (mu)
    """

    def compute_nationwide_effects(samples, df, weight_by_mu):
        std_wh = covariate_sd(df, COVERAGE, COVARIATE_SCALE, MMR_EFFECTIVENESS)

        if not weight_by_mu:
            # Use upper-level parameter
            effects = samples["effects_over_time"][:, :, 0]
            effects = effects[:, ::-1]  # Reverse to chronological order
            # % change per 1pp, averaged over the county-years of all the
            # years (df_2025), the same for every year
            effects_pct = percent_change_per_pp(
                effects,
                df,
                COVERAGE,
                df_2025,
                COVARIATE_SCALE,
                MMR_EFFECTIVENESS,
            )
            median_effect = jnp.median(effects_pct, axis=0)
            lower_effect = jnp.quantile(effects_pct, 0.025, axis=0)
            upper_effect = jnp.quantile(effects_pct, 0.975, axis=0)
            return median_effect, lower_effect, upper_effect

        else:
            # Use observation-level effects weighted by mu
            mu = samples["mu"]
            effects_obs = samples["effects_over_time_and_group"][
                :, :, 0
            ]  # [n_samples, n_observations]
            effects_pct = 100 * (
                factor_per_pp(
                    np.asarray(effects_obs) / std_wh,
                    df[COVERAGE].to_numpy(),
                    COVARIATE_SCALE,
                    MMR_EFFECTIVENESS,
                )
                - 1
            )

            years_calendar = df["year"].to_numpy()
            min_year = int(years_calendar.min())
            max_year = int(years_calendar.max())

            median_effects = []
            lower_effects = []
            upper_effects = []

            for year in range(min_year, max_year + 1):
                mask = df["year"].is_in([year]).to_jax()

                # Get effects and mu for this year
                year_effects = effects_pct[
                    :, mask
                ]  # [n_samples, n_obs_in_year]
                year_mu = mu[:, mask]  # [n_samples, n_obs_in_year]

                # Weighted average
                weighted_effects = (
                    year_effects * year_mu / year_mu.sum(axis=1, keepdims=True)
                ).sum(axis=1)  # [n_samples]

                median_effects.append(float(jnp.median(weighted_effects)))
                lower_effects.append(
                    float(jnp.quantile(weighted_effects, 0.025))
                )
                upper_effects.append(
                    float(jnp.quantile(weighted_effects, 0.975))
                )

            return (
                np.array(median_effects),
                np.array(lower_effects),
                np.array(upper_effects),
            )

    # Get effects from both model variants
    median_16_17, lower_16_17, upper_16_17 = compute_nationwide_effects(
        samples_16_17, df_16_17, weight_by_mu
    )
    median_2025, lower_2025, upper_2025 = compute_nationwide_effects(
        samples_2025, df_2025, weight_by_mu
    )

    # Average across the two variants
    median_effect = (median_16_17 + median_2025) / 2
    lower_effect = (lower_16_17 + lower_2025) / 2
    upper_effect = (upper_16_17 + upper_2025) / 2

    # Create year range for plotting
    years_all_out = range(
        int(min(years_in_order)), int(max(years_in_order)) + 1
    )

    # Verify alignment
    assert len(median_effect) == len(list(years_all_out)), (
        f"Mismatch: {len(median_effect)} effects vs {len(list(years_all_out))} years"
    )

    # Smooth the estimates: centered 3-year mean; at the ends, average only
    # the years available (no zero-padding, which shrinks the ends toward 0)
    def smooth(x, n=3):
        return jnp.convolve(x, jnp.ones(n), mode="same") / jnp.convolve(
            jnp.ones_like(x), jnp.ones(n), mode="same"
        )

    median_effect_smoothed = smooth(median_effect)
    lower_effect_smoothed = smooth(lower_effect)
    upper_effect_smoothed = smooth(upper_effect)

    # Plot
    ax.fill_between(
        years_all_out,
        lower_effect_smoothed,
        upper_effect_smoothed,
        color="grey",
        alpha=0.3,
    )
    ax.plot(years_all_out, median_effect_smoothed, color="grey")
    ax.axhline(0, ls="--", color="black")
    # where the 2021-2025 period begins (the break of panel c)
    ax.axvline(2021, ls="--", color="#737373", lw=2, zorder=0)

    ax.set_xlabel("Year")
    ax.set_ylabel(effect_label())
    ax.set_xticks([2011, 2015, 2020, 2025])

    return fig, ax


def print_effect_trend(samples_1617, samples_2025, df_1617, df_2025):
    years = np.arange(2011, 2026)
    print(
        "effects below are the % change in outbreak size per 1pp of coverage, "
        "averaged over the county-years of all the years (the same for every "
        "year and fit)"
    )
    per_fit = {}
    for name, samples, d in [
        ("without 2017 MN", samples_1617, df_1617),
        ("full data", samples_2025, df_2025),
    ]:
        # log of the factor per 1pp, so that exp(beta) - 1 is the effect
        # per 1pp of coverage
        beta = np.log1p(
            percent_change_per_pp(
                samples["effects_over_time"][:, ::-1, 0],
                d,
                COVERAGE,
                df_2025,
                COVARIATE_SCALE,
                MMR_EFFECTIVENESS,
            )
            / 100
        )
        mu = np.asarray(samples["mu"])
        assert mu.shape[1] == d.height, (mu.shape, d.height)
        row_years = d["year"].to_numpy()
        per_fit[name] = (beta, mu, row_years)
        print(f"\nraw (unsmoothed) median E [% per pp] by year, {name}:")
        E_med = np.median(100 * (np.exp(beta) - 1), axis=0)
        print(dict(zip(years.tolist(), np.round(E_med, 2).tolist())))

    for drop in [(), (2025,)]:
        tag = "all years" if not drop else f"dropping {drop}"
        results = {
            name: trend_draws(args[0], args[1], args[2], years, drop)
            for name, args in per_fit.items()
        }
        results["pooled"] = {
            k: np.concatenate([r[k] for r in results.values()])
            for k in next(iter(results.values()))
        }
        for name, res in results.items():
            print(f"\n=== {name}, {tag}: median [95% CrI] ===")
            for k, v in res.items():
                lo, med, hi = np.percentile(v, [2.5, 50, 97.5])
                # two-sided posterior p-value: twice the smaller tail
                p_two = 2 * min(np.mean(v < 0), np.mean(v > 0))
                if k.startswith("mubar"):
                    p_text = ""
                elif p_two > 0:
                    p_text = f"  p={p_two:.4f}"
                else:  # no draw on the other side of zero
                    p_text = f"  p<{2 / len(v):.5f}"
                print(
                    f"  {k:56s} {med:9.3f} [{lo:9.3f}, {hi:9.3f}]"
                    f"  P(<0)={np.mean(v < 0):.3f}{p_text}"
                )

        # Spearman rho of the yearly median effect vs year, with scipy's
        # p-value. Descriptive only: the random-walk prior makes neighboring
        # years dependent, so this p-value is anti-conservative (for driftless
        # random walks of length 15, p < 0.05 about 62% of the time vs 5%)
        keep = ~np.isin(years, drop)
        E_by_name = {
            name: 100 * (np.exp(args[0][:, keep]) - 1)
            for name, args in per_fit.items()
        }
        E_by_name["pooled"] = np.concatenate(list(E_by_name.values()))
        print(
            f"\nSpearman rho of the yearly median effect vs year, {tag} "
            "(descriptive only: the p-value is anti-conservative):"
        )
        for name, Ek in E_by_name.items():
            rho, p = stats.spearmanr(years[keep], np.median(Ek, axis=0))
            print(f"  {name:16s} rho = {rho:6.3f}, p = {p:.2g}")


if __name__ == "__main__":
    df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
    )

    fig_cov, ax_cov = setup_plot(1, 1)  # type: ignore

    # R01-only: standalone copy of the coverage violin plot (panel b of the
    # main figure below)
    fig_r01_a, ax_r01_a = setup_plot(1, 1)  # type: ignore
    plot_coverage_distribution(
        df,
        95,
        fig_r01_a,
        ax_r01_a,  # type: ignore
        vaccination_metric="estimated_mmr_cov",
    )
    fig_r01_a.savefig(
        os.path.join(fig_dir, "figure_r01_coverage_distribution_current.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig_r01_a)

    # R01-only: replica of the older (pre-refactor) panel a style
    fig_r01, ax_r01 = setup_plot(1, 1)  # type: ignore
    plot_coverage_distribution_r01(
        df,
        95,
        fig_r01,
        ax_r01,  # type: ignore
        vaccination_metric="estimated_mmr_cov",
    )
    fig_r01.savefig(
        os.path.join(
            fig_dir,
            "figure_coverage_outbreak_size_distribution_time_r01.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig_r01)

    plot_coverage_distribution_all_years(
        df,
        95,
        fig_cov,
        ax_cov,  # type: ignore
        vaccination_metric="estimated_mmr_cov",
    )
    fig_cov.savefig(
        os.path.join(
            fig_dir, "figure_coverage_distribution_outbreak_size.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )

    # panel a equivalent, pooling all years (no pre/post-COVID split)
    fig_all, ax_all = setup_plot(1, 1)  # type: ignore
    plot_coverage_distribution_all_yrs(
        df,
        95,
        fig_all,
        ax_all,  # type: ignore
        vaccination_metric="estimated_mmr_cov",
    )
    fig_all.savefig(
        os.path.join(
            fig_dir, "figure_supplemental_coverage_cases_all_yrs.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    fig_all.savefig(
        os.path.join(
            fig_dir, "figure_supplemental_coverage_cases_all_yrs.pdf"
        ),
        dpi=300,
        bbox_inches="tight",
    )

    # read predictive samples
    predictive_samples_1617 = load_samples(
        os.path.join(
            output_dir,
            "predictive_random_walk_poisson_CAR_time_and_group_varying_2025_oddities=True_exclude_16_17.npz",
        )
    )

    predictive_samples_2025 = load_samples(
        os.path.join(
            output_dir,
            "predictive_random_walk_poisson_CAR_time_and_group_varying_2025_oddities=True.npz",
        )
    )

    exclude_16_17 = pl.read_csv(
        os.path.join(output_dir, "exclude_16_17.csv"),
    )
    df = df.with_columns(
        year_county_state=pl.col("year").cast(pl.String)
        + pl.col("county")
        + pl.col("state")
    )

    # main figure: a = nationwide coverage effect over time, b = coverage by
    # outbreak size, c = significance of coverage across various models
    df_1617 = df.filter(
        ~pl.col("year_county_state").is_in(
            exclude_16_17["year_county_state"].to_list()
        )
    )
    print_effect_trend(
        predictive_samples_1617, predictive_samples_2025, df_1617, df
    )

    fig_r, ax_r = setup_plot(1, 3)  # type: ignore
    for panel_ax, label in zip(ax_r, "abc"):  # type: ignore
        panel_ax.text(
            -0.25,
            1.08,
            label,
            transform=panel_ax.transAxes,
            fontsize=24,
            fontweight="bold",
            va="top",
            ha="right",
        )
    plot_nationwide_effect_over_time(
        predictive_samples_1617,
        predictive_samples_2025,
        df_1617,
        df,
        list([2011, 2012.5, 2014.5, 2016.5, 2018.5, 2020.5, 2022.5, 2025]),
        fig_r,
        ax_r[0],  # type: ignore
    )
    plot_coverage_distribution(
        df,
        95,
        fig_r,
        ax_r[1],  # type: ignore
        vaccination_metric="estimated_mmr_cov",
    )
    plot_coverage_various_models(df, fig_r, ax_r[2])  # type: ignore
    fig_r.tight_layout()
    for ext in ["png", "pdf"]:
        fig_r.savefig(
            os.path.join(
                fig_dir, f"figure_changing_county_level_predictors.{ext}"
            ),
            dpi=300,
            bbox_inches="tight",
        )

    # supplemental: panel c on the linear scale of the coverage covariate (the
    # raw scale, which 02 fits for the baseline models next to the natural
    # one), a: with the negative binomial likelihood of the main fits, and
    # b: the all-covariates fit with the Poisson likelihood
    assert OTHER_SCALE == "raw", "the supplemental figure is of the raw scale"
    fig_other, ax_other = setup_plot(1, 2)  # type: ignore
    plot_coverage_various_models(
        df,
        fig_other,
        ax_other[0],  # type: ignore
        covariate_scale=OTHER_SCALE,
    )
    plot_coverage_various_models(
        df,
        fig_other,
        ax_other[1],  # type: ignore
        covariate_scale=OTHER_SCALE,
        models=[POISSON],
    )
    for panel_ax, label in zip(ax_other, "ab"):  # type: ignore
        panel_ax.text(
            -0.25,
            1.08,
            label,
            transform=panel_ax.transAxes,
            fontsize=24,
            fontweight="bold",
            va="top",
            ha="right",
        )
    fig_other.tight_layout()
    for ext in ["png", "pdf"]:
        fig_other.savefig(
            os.path.join(
                fig_dir,
                f"figure_supplemental_coverage_effect_linear_scale.{ext}",
            ),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig_other)

    fig_scatter, ax_scatter = setup_plot(1, 1)  # type: ignore
    plot_cases_vs_coverage_scatter(
        df,
        fig_scatter,
        ax_scatter,
        vaccination_metric="estimated_mmr_cov",
    )
    fig_scatter.savefig(
        os.path.join(fig_dir, "figure_supplemental_cases_coverage.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig_scatter.savefig(
        os.path.join(fig_dir, "figure_supplemental_cases_coverage.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
