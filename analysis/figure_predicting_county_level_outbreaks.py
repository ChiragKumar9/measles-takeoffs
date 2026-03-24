import os
import pickle

from numpyro import enable_x64
from scipy import stats

enable_x64()

import jax.numpy as jnp
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import polars as pl
import yaml
from matplotlib import rc

from outbreak_size import adjacency

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


def setup_plot(nrows, ncols, figsize=(30, 14)):
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
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


def plot_predictions(
    df,
    samples,
    naive_model,
    fig,
    ax,
    label_naive="Mechanistic: MMR coverage",
    label_statistical="Statistical: All covariates",
    show_r2=False,
):
    median = jnp.median(samples, axis=0)
    lower = jnp.quantile(samples, 0.025, axis=0)
    upper = jnp.quantile(samples, 0.975, axis=0)

    offset = 0.1
    ax.errorbar(
        df["cases_outbreak"] - offset,
        naive_model["median"],
        yerr=[
            naive_model["median"] - naive_model["lower"],
            naive_model["upper"] - naive_model["median"],
        ],
        fmt="o",
        ecolor="red",
        c="red",
        capsize=0,
        alpha=0.4,
        label=label_naive,
    )

    ax.errorbar(
        df["cases_outbreak"] + offset,
        median,
        yerr=[median - lower, upper - median],
        fmt="o",
        ecolor="green",
        c="green",
        capsize=0,
        alpha=0.8,
        label=label_statistical,
    )

    ax.set_xlabel("Reported outbreak size (N)")
    ax.set_ylabel("Predicted outbreak size (N)")
    ax.set_xscale("symlog")
    ax.set_yscale("symlog")
    ax.plot(
        [0, 1.05 * max(df["cases_outbreak"].max(), upper.max())],  # type: ignore
        [0, 1.05 * max(df["cases_outbreak"].max(), upper.max())],  # type: ignore
        ls="--",
        c="k",
    )
    ax.set_xlim(-0.3, 1.05 * max(df["cases_outbreak"].max(), upper.max()))  # type: ignore
    ax.set_ylim(-0.3, 1.05 * max(df["cases_outbreak"].max(), upper.max()))  # type: ignore

    ax.set_yticks([0, 1, 10, 100], labels=["0", "1", "10", "100"])
    ax.set_xticks([0, 1, 10, 100], labels=["0", "1", "10", "100"])

    ax.legend(loc=(0.0, 1.0))

    # Manually set minor tick locations on the x-axis
    minor_x_ticks = [
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
    ax.xaxis.set_minor_locator(tck.FixedLocator(minor_x_ticks))

    # Manually set minor tick locations on the y-axis
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

    if show_r2:
        # add a text annotation of the R2
        R2 = jnp.corrcoef(df["cases_outbreak"].to_jax(), median)[0, 1] ** 2
        R2 = R2 * 100  # as a percentage
        ax.text(
            0.0001 * max(df["cases_outbreak"].max(), upper.max()),
            0.35 * max(df["cases_outbreak"].max(), upper.max()),
            f"$R^2$ = {R2:.2f}%",
            fontsize=26,
            color="green",
        )

        # add a text annotation of the R2
        R2 = (
            jnp.corrcoef(df["cases_outbreak"].to_jax(), naive_model["median"])[
                0, 1
            ]
            ** 2
        )
        R2 = R2 * 100  # as a percentage
        ax.text(
            0.0001 * max(df["cases_outbreak"].max(), upper.max()),
            0.63 * max(df["cases_outbreak"].max(), upper.max()),
            f"$R^2$ = {R2:.2f}%",
            fontsize=26,
            color="red",
        )
    ax.set_ylim(-0.05)

    return fig, ax


def plot_ecological_determinants(samples_pre, samples_post, df, fig, ax):
    pooled_covariates = [
        "MMR coverage",
        "Nonmedical exemptions",
        "Medical exemptions",
    ]
    covariates = [
        "% < 5yo",
        "% white",
        "% graduate degree",
        "Population density (1,000 per km²)",
        "Income (10,000 2024 USD)",
    ]
    covariate_name_map = {
        "MMR coverage": "estimated_mmr_cov",
        "Nonmedical exemptions": "nonmed_exempt_pct",
        "Medical exemptions": "med_exempt_pct",
        "% < 5yo": "fraction_under_5",
        "% school aged": "fraction_school_aged",
        "Average household size": "avg_house_size",
        "Standard deviation of household size": "std_house_size",
        "% public schooled": "fraction_public",
        "% white": "fraction_white",
        "% graduate degree": "fraction_graduate_degree",
        "Population density (1,000 per km²)": "pop_density",
        "Income (10,000 2024 USD)": "log_income",
    }

    offset = 0.1
    year_medians = []
    for year_idx, years in enumerate(
        [
            [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
            [2021, 2022, 2023, 2024, 2025],
        ]
    ):
        if year_idx == 0:
            samples = samples_pre
        else:
            samples = samples_post
        all_covariates = [*pooled_covariates, *covariates]
        median_effects = []
        lower_effects = []
        upper_effects = []

        for idx, covariate in enumerate(pooled_covariates):
            samples_of_interest = samples["iv_global"][:, idx]
            # scale by standard deviation
            std = (
                df.filter(pl.col("year").is_in(years))[
                    covariate_name_map[covariate]
                ]
                .to_numpy()
                .std()
            )
            samples_of_interest = samples_of_interest / std
            # exponentiate and multiply by 100
            samples_of_interest = jnp.exp(samples_of_interest) * 100
            # to express as percent change
            samples_of_interest = samples_of_interest - 100
            median_effects.append(jnp.median(samples_of_interest, axis=0))
            lower_effects.append(
                jnp.quantile(samples_of_interest, 0.05, axis=0)
            )
            upper_effects.append(
                jnp.quantile(samples_of_interest, 0.95, axis=0)
            )

        # same process for unpooled covariates
        for covariate in covariates:
            # index is the number of this thing in the dict
            idx = list(covariate_name_map.keys()).index(covariate)
            idx = idx - 3
            samples_of_interest = samples["regression_betas"][:, idx]
            # scale by standard deviation
            std = (
                df.filter(pl.col("year").is_in(years))[
                    covariate_name_map[covariate]
                ]
                .to_numpy()
                .std()
            )
            if "%" in covariate:
                std = std * 100  # because we do the modeling as a fraction
            samples_of_interest = samples_of_interest / std
            if "Income" in covariate:
                # try to convert this to an income instead of log income
                mean_log_income = (
                    df.filter(pl.col("year").is_in(years))[
                        covariate_name_map[covariate]
                    ]
                    .to_numpy()
                    .mean()
                )
                mean_income = jnp.pow(10, mean_log_income)
                samples_of_interest = samples_of_interest / mean_income
                # now convert from dollars to 10,000 USD
                samples_of_interest = samples_of_interest * 10000
            # similar thing for population density -- want it to be 1,000 per km² instead of per person per km²
            if "Population density" in covariate:
                samples_of_interest = samples_of_interest * 1000
            # exponentiate and multiply by 100
            samples_of_interest = jnp.exp(samples_of_interest) * 100
            # to express as percent change
            samples_of_interest = samples_of_interest - 100
            median_effects.append(jnp.median(samples_of_interest, axis=0))
            lower_effects.append(
                jnp.quantile(samples_of_interest, 0.05, axis=0)
            )
            upper_effects.append(
                jnp.quantile(samples_of_interest, 0.95, axis=0)
            )

        ax.errorbar(
            x=[
                i + (year_idx * 2 - 1) * offset
                for i in range(len(all_covariates))
            ],
            y=median_effects,
            yerr=[
                jnp.array(median_effects) - jnp.array(lower_effects),
                jnp.array(upper_effects) - jnp.array(median_effects),
            ],
            fmt="o",
            ecolor="blue" if year_idx == 0 else "green",
            c="blue" if year_idx == 0 else "green",
            capsize=0,
            alpha=0.7,
            label=f"{years[0]}-{years[-1]}",
        )
        year_medians.append(median_effects)

    ax.axhline(0, ls="--", color="k")

    ax.legend(loc=(0.01, 1.0))

    ax.set_xticks(range(len(all_covariates)))  # type: ignore
    # at a 45 degree angle
    ax.set_xticklabels(
        all_covariates,  # type: ignore
        rotation=-60,
        ha="left",
    )
    ax.set_xlabel("Predictor")
    ax.set_ylabel("% change in incidence per\n1 unit increase in predictor")
    ax.set_yscale("symlog")
    ax.set_yticks(
        [-100, -10, -1, 0, 1, 10, 100],
        ["-100", "-10", "-1", "0", "1", "10", "100"],
    )

    minor_y_ticks = [
        -90,
        -80,
        -70,
        -60,
        -50,
        -40,
        -30,
        -20,
        -9,
        -8,
        -7,
        -6,
        -5,
        -4,
        -3,
        -2,
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
        500,
        600,
        700,
        800,
    ]
    ax.yaxis.set_minor_locator(tck.FixedLocator(minor_y_ticks))

    return fig, ax


def plot_increasing_spread_over_time(
    df,
    adjacency_matrix_pre,
    adjacency_matrix_post,
    posterior_samples_pre,
    posterior_samples_post,
    fig,
    ax,
    year_bins=None,
):
    """
    Plot number of county-years with neighboring outbreaks over time as grouped bar chart.

    Outbreak defined as >= 3 cases. Counts unique counties involved in neighboring
    outbreaks (where both the county and at least one neighbor have >= 3 cases).
    Shows three bars for different significance requirements:
    - Black: No significance requirement
    - Orangered: At least one county in cluster significant at 5% level
    - Darkmagenta: At least one county in cluster significant at 1% level

    Parameters:
    -----------
    df : polars.DataFrame
        Analysis dataset with columns: year, fips, county, state, cases_outbreak
    adjacency_matrix_pre : numpy.ndarray
        Adjacency matrix for counties in 2011-2020 period
    adjacency_matrix_post : numpy.ndarray
        Adjacency matrix for counties in 2021-2025 period
    posterior_samples_pre : dict
        Posterior samples for 2011-2020 period containing raw_spatial_effect and spatial_sigma
    posterior_samples_post : dict
        Posterior samples for 2021-2025 period containing raw_spatial_effect and spatial_sigma
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axis for plotting
    year_bins : list of tuples, optional
        Custom year bins as [(start1, end1), (start2, end2), ...].
        Years are inclusive and will be labeled as "start-end".
        If None, uses individual years.
        Example: [(2011, 2015), (2016, 2020), (2021, 2025)]
    """

    def get_significant_spatial_counties(
        posterior_samples, adjacency_matrix, all_fips, significance_threshold
    ):
        """
        Helper function to determine which counties have significant spatial effects.

        Returns a dictionary mapping FIPS codes to boolean (significant or not).
        Only includes counties that have neighbors (matching the spatial model structure).
        """
        raw_spatial_effect = posterior_samples["raw_spatial_effect"]
        spatial_sigma = posterior_samples["spatial_sigma"]
        spatial_effect = raw_spatial_effect * spatial_sigma[:, None]

        # Get counties with neighbors (matching plot_spatial_importance logic)
        has_a_neighbor = adjacency_matrix.sum(axis=0) > 0
        counties_with_neighbors = np.array(all_fips)[has_a_neighbor]

        # Verify dimensions match
        assert len(counties_with_neighbors) == spatial_effect.shape[1], (
            f"Counties with neighbors ({len(counties_with_neighbors)}) doesn't match spatial effects ({spatial_effect.shape[1]})"
        )

        # Calculate significance for each county with neighbors
        significant_spatial_dict = {}

        for county_idx, county_fips in enumerate(counties_with_neighbors):
            county_samples = spatial_effect[:, county_idx]

            # Calculate proportion on each side of zero
            prop_positive = np.mean(county_samples > 0)
            prop_negative = np.mean(county_samples < 0)

            # Significant if at least significance_threshold of mass is on one side
            if (
                prop_positive >= significance_threshold
                or prop_negative >= significance_threshold
            ):
                significant_spatial_dict[county_fips] = True
            else:
                significant_spatial_dict[county_fips] = False

        return significant_spatial_dict

    def count_valid_clusters(
        year_df,
        adjacency_matrix,
        significant_spatial_dict,
        all_fips,
        require_significance=True,
    ):
        """
        Count counties in valid clusters for a specific year.

        Parameters:
        -----------
        year_df : polars.DataFrame
            Data for a single year
        adjacency_matrix : numpy.ndarray
            Adjacency matrix for all counties in the period
        significant_spatial_dict : dict
            Dictionary mapping FIPS codes to whether they have significant spatial effects
            Can be None if require_significance is False
        all_fips : list
            All FIPS codes in the period (matching adjacency matrix order)
        require_significance : bool
            If True, only count clusters with at least one significant spatial effect
        """
        fips_to_idx = {fips: idx for idx, fips in enumerate(all_fips)}

        # Get counties with outbreaks this year (cases >= 3)
        outbreak_fips_list = year_df.filter(
            (pl.col("cases_outbreak") >= 3) & (pl.col("fips").is_in(all_fips))
        )["fips"].to_list()
        outbreak_fips_set = set(outbreak_fips_list)

        if len(outbreak_fips_list) == 0:
            return 0

        # Track clusters and whether they have significant spatial effects
        visited = set()
        counties_in_valid_clusters = set()

        # Find all clusters using BFS (Breadth-First Search)
        for county_fips in outbreak_fips_list:
            if county_fips in visited:
                continue

            # Start a new cluster search
            cluster = set()
            cluster_has_significant_spatial = False
            queue = [county_fips]  # Counties to explore

            while queue:
                current_fips = queue.pop(0)  # Get next county to explore

                if current_fips in visited:
                    continue

                visited.add(current_fips)
                cluster.add(current_fips)

                # Check if this county has significant spatial effect
                # (only counties with neighbors are in the dict)
                if (
                    require_significance
                    and significant_spatial_dict is not None
                ):
                    if (
                        current_fips in significant_spatial_dict
                        and significant_spatial_dict[current_fips]
                    ):
                        cluster_has_significant_spatial = True

                idx = fips_to_idx[current_fips]

                # Get neighbors of this county
                neighbors_mask = adjacency_matrix[idx] > 0
                neighbor_indices = np.where(neighbors_mask)[0]

                # Add outbreak neighbors to queue for exploration
                for neighbor_idx in neighbor_indices:
                    neighbor_fips = all_fips[neighbor_idx]

                    if (
                        neighbor_fips in outbreak_fips_set
                        and neighbor_fips not in visited
                    ):
                        queue.append(neighbor_fips)

            # Only count this cluster if it has more than 1 county (neighboring outbreak)
            # and (if required) at least one county has significant spatial effect
            if len(cluster) > 1:
                if not require_significance or cluster_has_significant_spatial:
                    counties_in_valid_clusters.update(cluster)

        return len(counties_in_valid_clusters)

    # Get FIPS lists for each period (ALL counties in the period, matching adjacency matrices)
    all_fips_pre = (
        df.filter(pl.col("year") < 2021)["fips"]
        .unique(maintain_order=True)
        .to_list()
    )
    all_fips_post = (
        df.filter(pl.col("year") >= 2021)["fips"]
        .unique(maintain_order=True)
        .to_list()
    )

    # Define significance thresholds and corresponding colors/labels
    thresholds_and_styles = [
        (None, "black", "All"),
        (0.95, "orangered", "$p < 0.05$"),
        (0.99, "darkmagenta", "$p < 0.01$"),
    ]

    # Set up year bins
    all_years = sorted(df["year"].unique().to_list())

    if year_bins is None:
        # Default: individual years
        bins = [(year, year) for year in all_years]
    else:
        bins = year_bins

    # Create labels for bins
    bin_labels = []
    for start_year, end_year in bins:
        if start_year == end_year:
            bin_labels.append(str(start_year))
        else:
            bin_labels.append(f"{start_year}-{end_year}")

    # Calculate counts for each bin and threshold
    data_by_bin = []

    for start_year, end_year in bins:
        bin_counts = []

        for threshold, color, threshold_label in thresholds_and_styles:
            # Get significant spatial effects for both time periods (if needed)
            if threshold is not None:
                significant_spatial_pre = get_significant_spatial_counties(
                    posterior_samples_pre,
                    adjacency_matrix_pre,
                    all_fips_pre,
                    threshold,
                )
                significant_spatial_post = get_significant_spatial_counties(
                    posterior_samples_post,
                    adjacency_matrix_post,
                    all_fips_post,
                    threshold,
                )
                require_significance = True
            else:
                significant_spatial_pre = None
                significant_spatial_post = None
                require_significance = False

            # Sum counts across all years in this bin
            total_count = 0

            for year in range(start_year, end_year + 1):
                if year not in all_years:
                    continue

                year_df = df.filter(pl.col("year") == year)

                if year < 2021:
                    # Use pre-2021 model and adjacency matrix
                    count = count_valid_clusters(
                        year_df,
                        adjacency_matrix_pre,
                        significant_spatial_pre,
                        all_fips_pre,
                        require_significance,
                    )
                else:
                    # Use post-2020 model and adjacency matrix
                    count = count_valid_clusters(
                        year_df,
                        adjacency_matrix_post,
                        significant_spatial_post,
                        all_fips_post,
                        require_significance,
                    )

                total_count += count

            bin_counts.append(total_count)

        data_by_bin.append(bin_counts)

    # Create grouped bar chart
    n_bins = len(bins)
    n_bars_per_group = len(thresholds_and_styles)

    # Bar width and positions
    bar_width = 0.25
    group_width = bar_width * n_bars_per_group
    group_spacing = 0.3  # Space between groups
    total_group_width = group_width + group_spacing

    x_positions = np.arange(n_bins) * total_group_width

    # Plot bars for each threshold
    for bar_idx, (threshold, color, threshold_label) in enumerate(
        thresholds_and_styles
    ):
        counts = [bin_data[bar_idx] for bin_data in data_by_bin]
        x_pos = x_positions + bar_idx * bar_width

        ax.bar(
            x_pos,
            counts,
            width=bar_width,
            color=color,
            label=threshold_label,
            alpha=0.8,
        )

    # Set x-axis labels centered under each group of 3 bars
    ax.set_xticks(x_positions + group_width / 2 - bar_width / 2)
    ax.set_xticklabels(bin_labels, rotation=-45, ha="left")

    ax.set_xlabel("Year")
    ax.set_ylabel("Counties in outbreak clusters (N)")
    ax.set_ylim(bottom=0)
    ax.legend(loc=(0.02, 0.79))

    return fig, ax


if __name__ == "__main__":
    df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
    )

    fig, ax = setup_plot(1, 3)  # type: ignore

    # read predictive samples
    with open(
        os.path.join(
            output_dir,
            "predictive_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_[2021, 2022, 2023, 2024, 2025].pkl",
        ),
        "rb",
    ) as file:
        predictive_samples = pickle.load(file)

    with open(
        os.path.join(
            output_dir,
            "final_outbreak_size_predictions_[2021, 2022, 2023, 2024, 2025].pkl",
        ),
        "rb",
    ) as f:
        naive_model = pickle.load(f)

    # performance
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
    plot_predictions(
        df.filter(pl.col("year").is_in([2021, 2022, 2023, 2024, 2025])),
        predictive_samples["cases"],
        naive_model,
        fig,
        ax[0],  # type: ignore
    )

    # while we're here, make a plot of the performance for the R01
    fig_r01, ax_r01 = setup_plot(1, 1, figsize=(10, 7.5))  # type: ignore
    plot_predictions(
        df.filter(pl.col("year").is_in([2021, 2022, 2023, 2024, 2025])),
        predictive_samples["cases"],
        naive_model,
        fig_r01,
        ax_r01,
        label_naive="Naive mechanistic model (vaccine coverage only)",
        label_statistical="Statistical model",
        show_r2=True,
    )
    fig_r01.savefig(
        os.path.join(
            fig_dir, f"figure_county_level_predictions_r01_no_CAR.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )

    with open(
        os.path.join(
            output_dir,
            "samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_[2021, 2022, 2023, 2024, 2025].pkl",
        ),
        "rb",
    ) as file:
        posterior_samples = pickle.load(file)

    with open(
        os.path.join(
            output_dir,
            "samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_[2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020].pkl",
        ),
        "rb",
    ) as file:
        posterior_samples_pre = pickle.load(file)

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
    plot_ecological_determinants(
        posterior_samples_pre,
        posterior_samples,
        df,
        fig,
        ax[1],  # type: ignore
    )

    # make adjacency matrix
    # read adjacency data
    adjacency_matrix = pl.read_csv(
        os.path.join(data_dir, "county_adjacencies.txt"),
        separator="|",
        schema_overrides={
            "County GEOID": pl.String,
            "Neighbor GEOID": pl.String,
        },
    )
    adjacency_matrix_observed = adjacency.assemble_adjacency_matrix(
        adjacency_matrix,
        df.filter(pl.col("year") >= 2021)["fips"]
        .unique(maintain_order=True)
        .to_list(),
    )
    adjacency_matrix_observed_pre = adjacency.assemble_adjacency_matrix(
        adjacency_matrix,
        df.filter(pl.col("year") < 2021)["fips"]
        .unique(maintain_order=True)
        .to_list(),
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
    plot_increasing_spread_over_time(
        df,
        adjacency_matrix_observed_pre,
        adjacency_matrix_observed,
        posterior_samples_pre,
        posterior_samples,
        fig,
        ax[2],  # type: ignore
        year_bins=[(2011, 2015), (2016, 2020), (2021, 2024), (2025, 2025)],
    )

    fig.tight_layout()

    # save
    fig.savefig(
        os.path.join(
            fig_dir, "figure_county_level_predictions_determinants.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(
            fig_dir, "figure_county_level_predictions_determinants.pdf"
        ),
        dpi=300,
        bbox_inches="tight",
    )
