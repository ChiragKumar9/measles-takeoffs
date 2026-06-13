import os
import pickle

from numpyro import enable_x64

enable_x64()

import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import pandas as pd
import polars as pl
import yaml
from addfips import AddFIPS
from matplotlib import rc
from scipy import stats

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


def setup_plot(nrows, ncols, figsize=(22, 18)):
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
    ax.set_ylabel("Model-calibrated outbreak size (N)")
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

    ax.set_yticks([0, 1, 10, 100, 500], labels=["0", "1", "10", "100", "500"])
    ax.set_xticks([0, 1, 10, 100, 500], labels=["0", "1", "10", "100", "500"])

    ax.legend(loc=(0.0, 1.05), edgecolor="black")

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
        500,
        600,
        700,
        800,
        900,
        1000,
    ]
    ax.yaxis.set_minor_locator(tck.FixedLocator(minor_y_ticks))

    if show_r2:
        # add a text annotation of the R2
        R2 = jnp.corrcoef(df["cases_outbreak"].to_jax(), median)[0, 1] ** 2
        R2 = R2 * 100  # as a percentage
        ax.text(
            0.00025 * max(df["cases_outbreak"].max(), upper.max()),
            0.35 * max(df["cases_outbreak"].max(), upper.max()),
            f"$R^2$ = {R2:.2f}%",
            fontsize=30,
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
            0.00025 * max(df["cases_outbreak"].max(), upper.max()),
            0.63 * max(df["cases_outbreak"].max(), upper.max()),
            f"$R^2$ = {R2:.2f}%",
            fontsize=30,
            color="red",
        )
    ax.set_ylim(-0.05)

    return fig, ax


def plot_ecological_determinants(samples_pre, samples_post, df, fig, ax):
    pooled_covariates = [
        "MMR coverage",
        "Nonmedical\nexemptions",
        "Medical\nexemptions",
    ]
    covariates = [
        "% school aged",
        "% public schooled",
        "% white",
        "Income\n(10,000 2024 USD)",
    ]
    covariate_name_map = {
        "MMR coverage": "estimated_mmr_cov",
        "Nonmedical\nexemptions": "nonmed_exempt_pct",
        "Medical\nexemptions": "med_exempt_pct",
        "% < 5yo": "fraction_under_5",
        "% school aged": "fraction_school_aged",
        "Average household size": "avg_house_size",
        "Standard deviation of household size": "std_house_size",
        "% public schooled": "fraction_public",
        "% white": "fraction_white",
        "% graduate degree": "fraction_graduate_degree",
        "Population density\n(1,000 people/km²)": "pop_density",
        "Income\n(10,000 2024 USD)": "log_income",
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
                jnp.quantile(samples_of_interest, 0.025, axis=0)
            )
            upper_effects.append(
                jnp.quantile(samples_of_interest, 0.975, axis=0)
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
                jnp.quantile(samples_of_interest, 0.025, axis=0)
            )
            upper_effects.append(
                jnp.quantile(samples_of_interest, 0.975, axis=0)
            )

        # Plot error bars
        color = "blue" if year_idx == 0 else "green"
        y_positions = [
            i + (year_idx * 2 - 1) * offset for i in range(len(all_covariates))
        ]

        # Calculate error bar lengths
        lower_errors = jnp.array(median_effects) - jnp.array(lower_effects)
        upper_errors = jnp.array(upper_effects) - jnp.array(median_effects)

        ax.errorbar(
            x=median_effects,
            y=y_positions,
            xerr=[lower_errors, upper_errors],
            fmt="o",
            ecolor=color,
            c=color,
            capsize=0,
            alpha=0.7,
            label=f"{years[0]}-{years[-1]}",
        )

        # Add arrows for CIs that extend beyond ±100
        for i, (y_pos, median, lower, upper) in enumerate(
            zip(y_positions, median_effects, lower_effects, upper_effects)
        ):
            # Check if lower CI extends below -100
            if lower < -100:
                ax.annotate(
                    "",
                    xy=(-100, y_pos),  # Arrow tip at plot edge
                    xytext=(
                        -98,
                        y_pos,
                    ),  # Arrow tail closer to tip for smaller arrow
                    arrowprops=dict(
                        arrowstyle="->, head_width=0.2, head_length=0.4",  # Smaller arrowhead
                        color=color,
                        lw=1.5,  # Thinner line
                        alpha=0.7,
                    ),
                )

            # Check if upper CI extends above 100
            if upper > 100:
                ax.annotate(
                    "",
                    xy=(100, y_pos),  # Arrow tip at plot edge
                    xytext=(
                        98,
                        y_pos,
                    ),  # Arrow tail closer to tip for smaller arrow
                    arrowprops=dict(
                        arrowstyle="->, head_width=0.2, head_length=0.4",  # Smaller arrowhead
                        color=color,
                        lw=1.5,  # Thinner line
                        alpha=0.7,
                    ),
                )

        year_medians.append(median_effects)

    ax.axvline(0, ls="--", color="k")

    ax.legend(loc=(0.01, 1.05), edgecolor="black")

    ax.set_yticks(range(len(all_covariates)))  # type: ignore
    ax.set_yticklabels(all_covariates)  # type: ignore
    ax.set_xlabel("% change in incidence per 1 unit\nincrease in predictor")
    ax.set_xlim(-100, 100)

    # Add "Lower risk" and "Higher risk" annotations at the top
    y_top = -0.3  # Position above the first data point
    ax.text(
        -60,
        y_top,
        "← Lower risk",
        ha="center",
        va="bottom",
        fontsize=22,
        style="italic",
        color="k",
    )
    ax.text(
        80,
        y_top,
        "Higher risk →",
        ha="center",
        va="bottom",
        fontsize=22,
        style="italic",
        color="k",
    )

    ax.invert_yaxis()

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
    aggregation_method="average",
):
    """
        Plot proportion of measles cases in counties with neighboring outbreaks over time as grouped bar chart.

        Outbreak defined as >= 3 cases. For each county with an outbreak, checks if at least one
        neighbor also has an outbreak, then sums cases in those counties.
        Shows two bars for different significance requirements:
        - Midnightblue: No significance requirement
        - Grey: County or at least one outbreak neighbor has significant spatial effect at 5% level

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
        aggregation_method : str, optional
            Method for aggregating across years in a bin:
            - "average": Average of yearly percentages (each year weighted equally)
            - "sum": Sum counts across years then calculate percentage (years with more cases weighted more)
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

    def count_cases_with_neighbors(
        year_df,
        adjacency_matrix,
        significant_spatial_dict,
        all_fips,
        require_significance=False,
    ):
        """
        Count cases in outbreak counties with at least one neighboring outbreak.

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
            If True, only count if the county OR at least one outbreak neighbor has significant spatial effect

        Returns:
        --------
        tuple : (cases_with_outbreak_neighbors, total_cases)
        """
        fips_to_idx = {fips: idx for idx, fips in enumerate(all_fips)}

        # Get counties with outbreaks this year (cases >= 3)
        outbreak_df = year_df.filter(
            (pl.col("cases_outbreak") >= 3) & (pl.col("fips").is_in(all_fips))
        )

        outbreak_fips_list = outbreak_df["fips"].to_list()
        outbreak_fips_set = set(outbreak_fips_list)

        # Create a dictionary mapping FIPS to cases
        fips_to_cases = dict(
            zip(
                outbreak_df["fips"].to_list(),
                outbreak_df["cases_outbreak"].to_list(),
            )
        )

        if len(outbreak_fips_list) == 0:
            return 0, 0

        cases_with_outbreak_neighbors = 0
        total_cases = sum(fips_to_cases.values())

        # For each outbreak county, check if it has outbreak neighbors
        for county_fips in outbreak_fips_list:
            idx = fips_to_idx[county_fips]

            # Get neighbors of this county
            neighbors_mask = adjacency_matrix[idx] > 0
            neighbor_indices = np.where(neighbors_mask)[0]

            # Find outbreak neighbors
            outbreak_neighbors = []
            for neighbor_idx in neighbor_indices:
                neighbor_fips = all_fips[neighbor_idx]
                if neighbor_fips in outbreak_fips_set:
                    outbreak_neighbors.append(neighbor_fips)

            # Check if this county has at least one outbreak neighbor
            if len(outbreak_neighbors) > 0:
                # If significance is required, check if county or any outbreak neighbor is significant
                if (
                    require_significance
                    and significant_spatial_dict is not None
                ):
                    # Check if the county itself is significant
                    county_is_significant = (
                        county_fips in significant_spatial_dict
                        and significant_spatial_dict[county_fips]
                    )

                    # Check if any outbreak neighbor is significant
                    neighbor_is_significant = any(
                        neighbor_fips in significant_spatial_dict
                        and significant_spatial_dict[neighbor_fips]
                        for neighbor_fips in outbreak_neighbors
                    )

                    # Count cases if either the county or a neighbor is significant
                    if county_is_significant or neighbor_is_significant:
                        cases_with_outbreak_neighbors += fips_to_cases[
                            county_fips
                        ]
                else:
                    # No significance requirement
                    cases_with_outbreak_neighbors += fips_to_cases[county_fips]

        return cases_with_outbreak_neighbors, total_cases

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
        (None, "#8b8c89", "All"),
        # (0.95, "#274c77", "Neighboring county\ncases are correlated"),
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
    error_bars_by_bin = []  # Store (lower_error, upper_error) for each bin

    for start_year, end_year in bins:
        bin_percentages = []
        bin_errors = []

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

            if aggregation_method == "average":
                # Collect data for each year in the bin
                year_data = []  # List of (clustered_cases, total_cases) tuples

                for year in range(start_year, end_year + 1):
                    if year not in all_years:
                        continue

                    year_df = df.filter(pl.col("year") == year)

                    if year < 2021:
                        cases_with_neighbors, total_cases = (
                            count_cases_with_neighbors(
                                year_df,
                                adjacency_matrix_pre,
                                significant_spatial_pre,
                                all_fips_pre,
                                require_significance,
                            )
                        )
                    else:
                        cases_with_neighbors, total_cases = (
                            count_cases_with_neighbors(
                                year_df,
                                adjacency_matrix_post,
                                significant_spatial_post,
                                all_fips_post,
                                require_significance,
                            )
                        )

                    # Only include years with cases
                    if total_cases > 0:
                        year_data.append((cases_with_neighbors, total_cases))

                # Calculate average percentage and uncertainty using Beta distributions
                if len(year_data) > 0:
                    n_samples = 10000
                    average_samples = []

                    for _ in range(n_samples):
                        yearly_proportions = []
                        for clustered, total in year_data:
                            isolated = total - clustered
                            alpha = clustered + 1
                            beta_param = isolated + 1
                            # Draw one sample from this year's Beta distribution
                            p_sample = stats.beta.rvs(alpha, beta_param)
                            yearly_proportions.append(p_sample)

                        # Average the proportions for this sample
                        avg_proportion = np.mean(yearly_proportions)
                        average_samples.append(
                            avg_proportion * 100
                        )  # Convert to percentage

                    # Get credible intervals
                    lower = np.percentile(average_samples, 2.5)
                    upper = np.percentile(average_samples, 97.5)
                    median = np.percentile(average_samples, 50)

                    avg_percentage = median
                    lower_error = median - lower
                    upper_error = upper - median
                else:
                    avg_percentage = 0
                    lower_error = 0
                    upper_error = 0

                bin_percentages.append(avg_percentage)
                bin_errors.append((lower_error, upper_error))

            elif aggregation_method == "sum":
                # Sum counts across all years, then calculate percentage
                total_cases_with_neighbors = 0
                total_cases_all = 0

                for year in range(start_year, end_year + 1):
                    if year not in all_years:
                        continue

                    year_df = df.filter(pl.col("year") == year)

                    if year < 2021:
                        cases_with_neighbors, total_cases = (
                            count_cases_with_neighbors(
                                year_df,
                                adjacency_matrix_pre,
                                significant_spatial_pre,
                                all_fips_pre,
                                require_significance,
                            )
                        )
                    else:
                        cases_with_neighbors, total_cases = (
                            count_cases_with_neighbors(
                                year_df,
                                adjacency_matrix_post,
                                significant_spatial_post,
                                all_fips_post,
                                require_significance,
                            )
                        )

                    total_cases_with_neighbors += cases_with_neighbors
                    total_cases_all += total_cases

                # Calculate percentage and uncertainty using Beta distribution
                if total_cases_all > 0:
                    isolated = total_cases_all - total_cases_with_neighbors
                    alpha = total_cases_with_neighbors + 1
                    beta_param = isolated + 1

                    # Draw samples from Beta distribution
                    n_samples = 10000
                    samples = (
                        stats.beta.rvs(alpha, beta_param, size=n_samples) * 100
                    )

                    lower = np.percentile(samples, 2.5)
                    upper = np.percentile(samples, 97.5)
                    median = np.percentile(samples, 50)

                    percentage = median
                    lower_error = median - lower
                    upper_error = upper - median
                else:
                    percentage = 0
                    lower_error = 0
                    upper_error = 0

                bin_percentages.append(percentage)
                bin_errors.append((lower_error, upper_error))

            else:
                raise ValueError(
                    f"Unknown aggregation_method: {aggregation_method}"
                )

        data_by_bin.append(bin_percentages)
        error_bars_by_bin.append(bin_errors)

    # Create grouped bar chart
    n_bins = len(bins)
    n_bars_per_group = len(thresholds_and_styles)

    # Bar width and positions
    bar_width = 0.4
    group_width = bar_width * n_bars_per_group
    group_spacing = 0.3  # Space between groups
    total_group_width = group_width + group_spacing

    x_positions = np.arange(n_bins) * total_group_width  # type: ignore

    # Plot bars for each threshold
    for bar_idx, (threshold, color, threshold_label) in enumerate(
        thresholds_and_styles
    ):
        percentages = [bin_data[bar_idx] for bin_data in data_by_bin]
        errors = [bin_errors[bar_idx] for bin_errors in error_bars_by_bin]

        # Extract lower and upper errors
        lower_errors = [e[0] for e in errors]
        upper_errors = [e[1] for e in errors]

        x_pos = x_positions + bar_idx * bar_width

        ax.bar(
            x_pos,
            percentages,
            width=bar_width,
            color=color,
            label=threshold_label,
            alpha=0.8,
            yerr=[lower_errors, upper_errors],
            capsize=5,
            error_kw={"elinewidth": 1.5, "ecolor": "black", "alpha": 0.8},
        )
        # print out what we're plotting so we can read
        # the statistics out
        for i in range(n_bins):
            print(
                f"Bin {bin_labels[i]}, Threshold '{threshold_label}': "
                f"{percentages[i]:.2f}% (95% CI: {percentages[i] - lower_errors[i]:.2f}% - {percentages[i] + upper_errors[i]:.2f}%)"
            )

    # Set x-axis labels centered under each group of 2 bars
    # ax.set_xticks(x_positions + group_width / 2 - bar_width / 2)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(bin_labels, rotation=-45, ha="left")

    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_ylabel("Cases in multi-county outbreaks (%)")
    ax.set_ylim(bottom=0)
    # ax.legend(loc = (0.01, 0.7), edgecolor="black",
    # title = "Multi-county outbreaks:       ")

    # Calculate ratio: Predictive relationship / All for both 2011-2024 and 2025
    print("\n" + "=" * 70)
    print("RATIO ANALYSIS: Predictive relationship / All")
    print("=" * 70)

    # Function to calculate statistics for a time period
    def calculate_period_stats(year_range, period_name):
        """Calculate statistics for a given time period"""
        # Collect data for both thresholds
        data_by_threshold = {}

        for threshold, _, threshold_label in thresholds_and_styles:
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

            # Collect data for each year in the range
            year_data = []

            for year in year_range:
                if year not in all_years:
                    continue

                year_df = df.filter(pl.col("year") == year)

                if year < 2021:
                    cases_with_neighbors, total_cases = (
                        count_cases_with_neighbors(
                            year_df,
                            adjacency_matrix_pre,
                            significant_spatial_pre,
                            all_fips_pre,
                            require_significance,
                        )
                    )
                else:
                    cases_with_neighbors, total_cases = (
                        count_cases_with_neighbors(
                            year_df,
                            adjacency_matrix_post,
                            significant_spatial_post,
                            all_fips_post,
                            require_significance,
                        )
                    )

                # Only include years with cases
                if total_cases > 0:
                    year_data.append((cases_with_neighbors, total_cases))

            data_by_threshold[threshold_label] = year_data

        # Check if we have data for both thresholds
        if (
            len(data_by_threshold["All"]) == 0
        ):  # or len(data_by_threshold["Neighboring county\ncases are correlated"]) == 0:
            print(f"\n{period_name}: Insufficient data")
            return None

        # Calculate statistics with proper uncertainty propagation
        n_samples = 10000
        ratio_samples = []
        all_samples = []
        pred_samples = []

        for _ in range(n_samples):
            # Sample from "All" threshold
            all_yearly_proportions = []
            for clustered, total in data_by_threshold["All"]:
                isolated = total - clustered
                alpha = clustered + 1
                beta_param = isolated + 1
                p_sample = stats.beta.rvs(alpha, beta_param)
                all_yearly_proportions.append(p_sample)
            avg_all = np.mean(all_yearly_proportions)
            all_samples.append(avg_all * 100)

            # Sample from "Predictive relationship" threshold
            pred_yearly_proportions = []
            # for clustered, total in data_by_threshold["Neighboring county\ncases are correlated"]:
            #     isolated = total - clustered
            #     alpha = clustered + 1
            #     beta_param = isolated + 1
            #     p_sample = stats.beta.rvs(alpha, beta_param)
            #     pred_yearly_proportions.append(p_sample)
            avg_pred = np.mean(pred_yearly_proportions)
            pred_samples.append(avg_pred * 100)

            # Calculate ratio (avoid division by zero)
            if avg_all > 0:
                ratio = avg_pred / avg_all
                ratio_samples.append(ratio)

        # Get credible intervals
        all_median = np.percentile(all_samples, 50)
        all_lower = np.percentile(all_samples, 2.5)
        all_upper = np.percentile(all_samples, 97.5)

        pred_median = np.percentile(pred_samples, 50)
        pred_lower = np.percentile(pred_samples, 2.5)
        pred_upper = np.percentile(pred_samples, 97.5)

        ratio_median = np.percentile(ratio_samples, 50)
        ratio_lower = np.percentile(ratio_samples, 2.5)
        ratio_upper = np.percentile(ratio_samples, 97.5)

        # Print results
        print(f"\n{period_name}:")
        print(f"  All multi-county outbreaks:")
        print(
            f"    {all_median:.2f}% (95% CI: {all_lower:.2f}% - {all_upper:.2f}%)"
        )
        print(f"  Predictive relationship between neighbors:")
        print(
            f"    {pred_median:.2f}% (95% CI: {pred_lower:.2f}% - {pred_upper:.2f}%)"
        )
        print(f"  Ratio (Predictive / All):")
        print(
            f"    {ratio_median:.3f} (95% CI: {ratio_lower:.3f} - {ratio_upper:.3f})"
        )
        print(
            f"    → {100 * ratio_median:.1f}% of multi-county outbreaks show predictive spatial relationships"
        )
        print(
            f"  Based on {len(data_by_threshold['All'])} years with outbreak cases"
        )

        return {
            "all_median": all_median,
            "all_lower": all_lower,
            "all_upper": all_upper,
            "pred_median": pred_median,
            "pred_lower": pred_lower,
            "pred_upper": pred_upper,
            "ratio_median": ratio_median,
            "ratio_lower": ratio_lower,
            "ratio_upper": ratio_upper,
            "ratio_samples": ratio_samples,
            "all_samples": all_samples,
            "pred_samples": pred_samples,
        }

    # Calculate for 2011-2024
    calculate_period_stats(range(2011, 2025), "2011-2024")

    # Calculate for 2025
    calculate_period_stats([2025], "2025")

    print("=" * 70 + "\n")

    return fig, ax


def calculate_moran_eigenvectors_adjacency(
    adjacency_matrix, n_vectors="positive"
):
    """
    Calculate Moran's eigenvector maps from adjacency matrix
    """
    W = np.asarray(adjacency_matrix, dtype=float)
    n = W.shape[0]

    # Row-standardize
    row_sums = W.sum(axis=1)
    row_sums[row_sums == 0] = 1
    W = W / row_sums[:, np.newaxis]

    # Center the matrix
    I = np.eye(n)
    ones = np.ones((n, 1))
    H = I - (ones @ ones.T) / n
    M = H @ W @ H

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(M)

    # Sort descending
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Keep positive eigenvalues
    if n_vectors == "positive":
        n_vectors = np.sum(eigenvalues > 1e-10)

    return eigenvectors[:, :n_vectors], eigenvalues[:n_vectors]


def regression_msr_multiple_blocked(
    X,
    y,
    county_indices,
    adjacency_matrix,
    n_permutations=99999,
    alpha=0.05,
    seed=108,
):
    """
    Multiple regression with MSR-based inference using BLOCK permutation

    For data with multiple observations per spatial unit (e.g., counties over time).
    Permutes entire counties together to preserve within-county structure.

    Parameters:
    -----------
    X : array, shape (n_obs, p) or (n_obs,)
        Predictor matrix (without intercept - will be added)
    y : array, shape (n_obs,)
        Outcome variable
    county_indices : array, shape (n_obs,)
        County identifier for each observation (0-indexed integers: 0, 1, 2, ..., n_counties-1)
        Example: [0, 0, 0, 1, 1, 2, 2, 2, ...] means first 3 obs are county 0, etc.
    adjacency_matrix : array, shape (n_counties, n_counties)
        County-level adjacency matrix
    n_permutations : int
        Number of permutations
    alpha : float
        Significance level for CIs
    seed : int, optional
        Random seed

    Returns:
    --------
    results : dict
        'coef': observed coefficients (including intercept)
        'ci_lower': lower CI bounds
        'ci_upper': upper CI bounds
        'p_values': MSR p-values for each coefficient
        'coef_null': null distributions (n_permutations x n_coef)
        'r_squared': R²
    """
    if seed is not None:
        np.random.seed(seed)

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    county_indices = np.asarray(county_indices, dtype=int)

    # Validation
    if len(X) != len(y) or len(X) != len(county_indices):
        raise ValueError(
            f"X ({len(X)}), y ({len(y)}), and county_indices ({len(county_indices)}) "
            f"must have same length"
        )

    # Handle 1D predictor
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Add intercept
    X_design = np.column_stack([np.ones(len(y)), X])
    n_coef = X_design.shape[1]

    # Get unique counties and validate
    unique_counties = np.unique(county_indices)
    n_counties = len(unique_counties)

    if n_counties != adjacency_matrix.shape[0]:
        raise ValueError(
            f"Number of unique counties ({n_counties}) doesn't match "
            f"adjacency matrix size ({adjacency_matrix.shape[0]})"
        )

    # Check that county_indices are 0-indexed and contiguous
    if not np.array_equal(unique_counties, np.arange(n_counties)):
        raise ValueError(
            f"county_indices must be 0-indexed integers from 0 to {n_counties - 1}. "
            f"Got: {unique_counties}"
        )

    # Observed coefficients
    coef_obs = np.linalg.lstsq(X_design, y, rcond=None)[0]

    # R²
    y_pred = X_design @ coef_obs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot

    # Get Moran eigenvectors at county level
    eigenvectors, eigenvalues = calculate_moran_eigenvectors_adjacency(
        adjacency_matrix
    )

    # Pre-compute masks for each county (for efficiency)
    county_masks = [county_indices == i for i in range(n_counties)]

    # Generate null distributions via block permutation
    coef_null = np.zeros((n_permutations, n_coef))

    for perm_i in range(n_permutations):
        # Generate county-level spatial permutation
        coeffs = np.random.randn(eigenvectors.shape[1])
        scores = eigenvectors @ coeffs
        county_perm = np.argsort(
            scores
        )  # New county assignment for each original county

        # Build permuted y by swapping county blocks
        y_perm = np.zeros_like(y)

        for original_county_idx in range(n_counties):
            # This original county's observations should get values from new_county
            new_county_idx = county_perm[original_county_idx]

            # Get observations from both counties
            original_mask = county_masks[original_county_idx]
            new_mask = county_masks[new_county_idx]

            original_obs = y[original_mask]
            new_obs = y[new_mask]

            # Handle different lengths
            n_original = len(original_obs)
            n_new = len(new_obs)

            if n_original == n_new:
                # Same length: direct assignment
                y_perm[original_mask] = new_obs
            elif n_original < n_new:
                # Fewer observations in original: sample from new
                sampled_indices = np.random.choice(
                    n_new, size=n_original, replace=False
                )
                y_perm[original_mask] = new_obs[sampled_indices]
            else:
                # More observations in original: sample with replacement from new
                sampled_indices = np.random.choice(
                    n_new, size=n_original, replace=True
                )
                y_perm[original_mask] = new_obs[sampled_indices]

        # Calculate coefficients for this permutation
        coef_null[perm_i, :] = np.linalg.lstsq(X_design, y_perm, rcond=None)[0]

    # P-values (two-tailed, Phipson-Smyth correction)
    p_values = np.zeros(n_coef)
    for j in range(n_coef):
        n_extreme = np.sum(np.abs(coef_null[:, j]) >= np.abs(coef_obs[j]))
        p_values[j] = (n_extreme + 1) / (n_permutations + 1)

    # Confidence intervals (percentile method)
    ci_lower = np.zeros(n_coef)
    ci_upper = np.zeros(n_coef)

    for j in range(n_coef):
        diffs = coef_obs[j] - coef_null[:, j]
        ci_lower[j] = coef_obs[j] - np.percentile(diffs, 100 * (1 - alpha / 2))
        ci_upper[j] = coef_obs[j] - np.percentile(diffs, 100 * alpha / 2)

    return {
        "coef": coef_obs,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_values": p_values,
        "coef_null": coef_null,
        "r_squared": r_squared,
    }


def plot_county_level_variation(
    df, posterior_samples_post, adjacency_matrix_post, fig, ax
):
    """
    Plot within-county vaccination coverage variability for large outbreaks,
    comparing independent vs correlated multi-county outbreaks.
    """
    school = pl.read_csv(
        os.path.join(data_dir, "vaxrates_schools.csv"),
        null_values="NA",
        encoding="utf8-lossy",
    )

    school = school.with_columns(
        year=pl.col("year").str.slice(5, 9).cast(pl.Int64)
    )

    school = school.with_columns(
        mmr_rate=pl.when(pl.col("mmr_rate").is_not_null())
        .then(pl.col("mmr_rate"))
        .otherwise(pl.col("overall_rate"))
    )

    school = school.group_by(
        ["year", "county", "state"], maintain_order=True
    ).agg(
        median_coverage=pl.col("mmr_rate").median(),
        mean_coverage=pl.col("mmr_rate").mean(),
        std_coverage=pl.col("mmr_rate").std(),
        IQR_coverage=pl.col("mmr_rate").quantile(0.75)
        - pl.col("mmr_rate").quantile(0.25),
        fraction_public=pl.col("school_type").eq("PUBLIC").sum()
        / pl.count("mmr_rate"),
        n=pl.count("mmr_rate"),
    )

    school = school.filter(pl.col("n") > 0)
    school = school.filter(
        pl.col("county").is_not_null() & pl.col("state").is_not_null()
    )
    school = school.with_columns(
        std_coverage=pl.when(
            pl.col("std_coverage").is_null() & (pl.col("n") == 1)
        )
        .then(0.0)
        .otherwise(pl.col("std_coverage"))
    )

    fip_getter = AddFIPS()
    school = school.with_columns(
        fips=pl.struct(["county", "state"]).map_elements(
            lambda x: fip_getter.get_county_fips(
                x["county"], state=x["state"]
            ),
            return_dtype=pl.String,
        )
    )
    school = school.filter(pl.col("fips").is_not_null())

    # Get all FIPS codes for post-2021 period
    all_fips_post = (
        df.filter(pl.col("year") >= 2021)["fips"]
        .unique(maintain_order=True)
        .to_list()
    )

    # Get spatial effects from posterior
    raw_spatial_effect = posterior_samples_post["raw_spatial_effect"]
    spatial_sigma = posterior_samples_post["spatial_sigma"]
    spatial_effect = raw_spatial_effect * spatial_sigma[:, None]

    has_a_neighbor = adjacency_matrix_post.sum(axis=0) > 0
    counties_with_neighbors = np.array(all_fips_post)[has_a_neighbor]

    # Calculate Bayesian p-values for spatial effects
    significance_records = []
    bayesian_pvalue_records = []
    for county_idx, county_fips in enumerate(counties_with_neighbors):
        county_samples = spatial_effect[:, county_idx]
        prop_positive = np.mean(county_samples > 0)
        prop_negative = np.mean(county_samples < 0)
        is_significant = prop_positive >= 0.95 or prop_negative >= 0.95
        bayesian_p = float(np.minimum(prop_positive, prop_negative))
        significance_records.append(
            {"fips": county_fips, "significant_car": is_significant}
        )
        bayesian_pvalue_records.append(
            {"fips": county_fips, "bayesian_p": bayesian_p}
        )

    significant_car_df = pl.DataFrame(significance_records)
    bayesian_p_df = pl.DataFrame(bayesian_pvalue_records)

    school_post = school.filter(pl.col("year") >= 2021)
    cases_post = df.filter(pl.col("year") >= 2021).select(
        ["fips", "year", "cases_outbreak", "estimated_mmr_cov"]
    )

    county_df = (
        school_post.join(cases_post, on=["fips", "year"], how="inner")
        .join(significant_car_df, on="fips", how="left")
        .join(bayesian_p_df, on="fips", how="left")
        .with_columns(
            significant_car=pl.col("significant_car").fill_null(False),
        )
    )

    print(f"\nSchool data after joining:")
    print(f"  Total observations: {len(county_df)}")
    print(
        f"  Year distribution:\n{county_df.group_by('year').len().sort('year')}"
    )
    # percent of measles cases represented over this time period
    total_cases = df.filter(pl.col("year") >= 2024)["cases_outbreak"].sum()
    cases_in_county_df = county_df.filter(pl.col("year") >= 2024)[
        "cases_outbreak"
    ].sum()
    print(f"  Total cases in df from 2024 onward: {total_cases}")
    print(
        f"  Cases represented in county_df: {cases_in_county_df} ({cases_in_county_df / total_cases * 100:.1f}%)"
    )

    cutoff = 10

    county_df = county_df.with_columns(
        std_coverage_pct=pl.col("std_coverage") * 100,
        mean_coverage_pct=pl.col("mean_coverage")
        * 100,  # Convert to percentage
        outbreak_category=pl.when(pl.col("cases_outbreak") < cutoff)
        .then(pl.lit(f"<{cutoff}"))
        .otherwise(pl.lit(f"≥{cutoff}")),
    )

    # Filter: mean_coverage <= 95% AND estimated_mmr_cov and mean_coverage are within 10pp
    county_df = county_df.filter(
        (
            # (pl.col("mean_coverage_pct") <= 95) &
            (pl.col("estimated_mmr_cov") - pl.col("mean_coverage_pct")).abs()
            <= 10
        )
    )

    # ============= NEW: Identify multi-county outbreaks =============
    # Create a mapping from FIPS to index in adjacency matrix
    fips_to_adj_idx = {fips: i for i, fips in enumerate(all_fips_post)}

    # Create a subset of the full df for post-2021 period for faster lookups
    df_post = df.filter(pl.col("year") >= 2021)

    # For each county-year, determine if it's part of a multi-county outbreak
    multi_county_outbreak_records = []

    for row in county_df.iter_rows(named=True):
        fips = row["fips"]
        year = row["year"]
        cases = row["cases_outbreak"]

        # Check if this county has 3+ cases
        if cases >= 3 and fips in fips_to_adj_idx:
            adj_idx = fips_to_adj_idx[fips]

            # Get neighbors from adjacency matrix
            neighbor_indices = np.where(adjacency_matrix_post[adj_idx, :] > 0)[
                0
            ]
            neighbor_fips = [all_fips_post[i] for i in neighbor_indices]

            # Check if any neighbor also has 3+ cases in the same year
            # Use the FULL dataset (df_post), not just counties with school data
            neighbor_cases = df_post.filter(
                (pl.col("fips").is_in(neighbor_fips))
                & (pl.col("year") == year)
            )["cases_outbreak"].to_list()

            has_neighbor_with_outbreak = any(nc >= 3 for nc in neighbor_cases)
            is_multi_county = has_neighbor_with_outbreak
        else:
            is_multi_county = False

        multi_county_outbreak_records.append(
            {
                "fips": fips,
                "year": year,
                "multi_county_outbreak": is_multi_county,
            }
        )

    multi_county_df = pl.DataFrame(multi_county_outbreak_records)

    # Join the multi-county outbreak flag
    county_df = county_df.join(
        multi_county_df, on=["fips", "year"], how="left"
    )

    # Build regression dataframe
    county_df_reg = (
        county_df.filter(pl.col("bayesian_p").is_not_null())
        .with_columns(cases_outbreak_log=(pl.col("cases_outbreak") + 1).log())
        .with_columns(
            interaction=pl.col("std_coverage_pct")
            * pl.col("cases_outbreak_log")
        )
        .to_pandas()
    )

    # Create county indices
    fips_to_county_idx = {fips: i for i, fips in enumerate(all_fips_post)}
    county_df_reg["county_idx"] = county_df_reg["fips"].map(fips_to_county_idx)

    unique_fips_in_regression = county_df_reg["fips"].unique()
    fips_to_new_idx = {
        fips: i for i, fips in enumerate(unique_fips_in_regression)
    }
    fips_to_original_idx = {fips: i for i, fips in enumerate(all_fips_post)}
    original_indices = [
        fips_to_original_idx[fips] for fips in unique_fips_in_regression
    ]
    adjacency_matrix_regression = adjacency_matrix_post[
        np.ix_(original_indices, original_indices)
    ]

    county_df_reg["county_idx"] = county_df_reg["fips"].map(fips_to_new_idx)

    # Prepare data for MSR
    county_df_msr = county_df_reg.copy()

    # ============= PLOTTING CODE =============

    # Filter to only large outbreaks
    outbreak_cat = f"≥{cutoff}"

    # Get data for the two categories
    # Grey: Everything else (not blue)
    print(county_df["significant_car"])
    print(county_df["multi_county_outbreak"])
    not_sig_data = county_df.filter(
        (pl.col("outbreak_category") == outbreak_cat)
        & (pl.col("significant_car") == 0.0)
    )["std_coverage_pct"].to_numpy()

    # Blue: Significant CAR
    sig_data = county_df.filter(
        (pl.col("outbreak_category") == outbreak_cat)
        & (pl.col("significant_car") == 1.0)
    )["std_coverage_pct"].to_numpy()

    c1 = "#8b8c89"  # Not significant CAR
    c2 = "#274c77"  # Significant CAR

    positions = [0, 1]
    data_groups = [not_sig_data, sig_data]
    colors = [c1, c2]

    # Plot violins
    for i, (data, pos, color) in enumerate(
        zip(data_groups, positions, colors)
    ):
        if len(data) > 0:
            violinparts = ax.violinplot(
                [data], positions=[pos], widths=0.5, showextrema=False
            )
            for pc in violinparts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.6)

            # Add median and IQR
            median = np.median(data)
            q25 = np.percentile(data, 25)
            q75 = np.percentile(data, 75)
            mean = np.mean(data)

            ax.errorbar(
                x=[pos],
                y=[median],
                yerr=[[median - q25], [q75 - median]],
                fmt="o",
                capsize=10,
                capthick=2,
                color=color,
            )
            ax.scatter([pos], [mean], marker="*", s=150, color=color, zorder=3)

    # Run MSR regression for large outbreak bucket
    subset = county_df_msr[county_df_msr["outbreak_category"] == outbreak_cat]

    # Get unique counties in this bucket
    bucket_fips = subset["fips"].unique()
    print(len(subset))
    print(len(bucket_fips))

    # Map bucket FIPS to indices in the regression adjacency matrix
    bucket_to_regression_idx = [fips_to_new_idx[fips] for fips in bucket_fips]

    # Subset the adjacency matrix for this bucket
    adjacency_bucket = adjacency_matrix_regression[
        np.ix_(bucket_to_regression_idx, bucket_to_regression_idx)
    ]

    # Create FRESH 0-indexed county IDs for this bucket
    bucket_fips_to_bucket_idx = {fips: i for i, fips in enumerate(bucket_fips)}
    bucket_county_indices = np.array(
        [bucket_fips_to_bucket_idx[fips] for fips in subset["fips"]]
    )

    # Run bucket-specific MSR regression
    X_bucket = subset[["std_coverage_pct", "cases_outbreak"]].values

    print(X_bucket.shape)

    y_bucket = subset["bayesian_p"].values

    print("X_bucket:", X_bucket)
    print("y_bucket:", y_bucket)
    print("bucket_county_indices:", bucket_county_indices)
    print("adjacency_bucket shape:", adjacency_bucket.shape)
    print("adjacency_bucket sum:", adjacency_bucket.sum())

    bucket_msr = regression_msr_multiple_blocked(
        X_bucket,
        y_bucket,
        bucket_county_indices,
        adjacency_bucket,
    )

    print(bucket_msr)

    p = bucket_msr["p_values"][1]

    # Determine significance stars
    if p < 0.001:
        sig_text = "***"
    elif p < 0.01:
        sig_text = "**"
    elif p < 0.05:
        sig_text = "*"
    else:
        sig_text = "ns"
    # Draw significance bracket
    max_y = max(
        np.max(not_sig_data) if len(not_sig_data) > 0 else 0,
        np.max(sig_data) if len(sig_data) > 0 else 0,
    )
    bar_height = max_y * 1.05
    ax.plot([0, 1], [bar_height, bar_height], "k-", linewidth=1.5)
    ax.plot(
        [0, 0], [bar_height, bar_height - max_y * 0.02], "k-", linewidth=1.5
    )
    ax.plot(
        [1, 1], [bar_height, bar_height - max_y * 0.02], "k-", linewidth=1.5
    )
    ax.text(
        0.5, bar_height + 0.2, sig_text, ha="center", va="bottom", fontsize=22
    )

    ax.set_ylabel(
        "Standard deviation of within-county\nMMR vaccine coverage (pp)"
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [
            "Independent single\ncounty outbreak",
            "Correlated multi-\ncounty outbreak",
        ]
    )
    ax.set_ylim(0)
    ax.set_yticks([0, 5, 10, 15])

    # Add heterogeneity/homogeneity annotations
    y_min, y_max = ax.get_ylim()
    ax.text(
        1.5,
        y_max * 0.85,
        "↑ Heterogeneous\ncoverage",
        ha="center",
        va="bottom",
        fontsize=22,
        style="italic",
        color="k",
    )
    ax.text(
        1.5,
        y_max * 0.05,
        "Homogeneous\ncoverage ↓",
        ha="center",
        va="bottom",
        fontsize=22,
        style="italic",
        color="k",
    )

    return fig, ax


def plot_supplemental_mmr_coverage_correlation(df, fig, ax):
    """
    Plot correlation between estimated MMR coverage and school-level mean coverage.

    Parameters:
    -----------
    df : polars.DataFrame
        Analysis dataset with estimated_mmr_cov column
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axis for plotting
    """
    school = pl.read_csv(
        os.path.join(data_dir, "vaxrates_schools.csv"),
        null_values="NA",
        encoding="utf8-lossy",
    )

    school = school.with_columns(
        year=pl.col("year").str.slice(5, 9).cast(pl.Int64)
    )

    school = school.with_columns(
        mmr_rate=pl.when(pl.col("mmr_rate").is_not_null())
        .then(pl.col("mmr_rate"))
        .otherwise(pl.col("overall_rate"))
    )

    school = school.group_by(
        ["year", "county", "state"], maintain_order=True
    ).agg(
        mean_coverage=pl.col("mmr_rate").mean(),
        n=pl.count("mmr_rate"),
    )

    school = school.filter(pl.col("n") > 0)
    school = school.filter(
        pl.col("county").is_not_null() & pl.col("state").is_not_null()
    )

    fip_getter = AddFIPS()
    school = school.with_columns(
        fips=pl.struct(["county", "state"]).map_elements(
            lambda x: fip_getter.get_county_fips(
                x["county"], state=x["state"]
            ),
            return_dtype=pl.String,
        )
    )
    school = school.filter(pl.col("fips").is_not_null())

    # Convert mean_coverage from 0-1 to 0-100
    school = school.with_columns(
        mean_coverage_pct=pl.col("mean_coverage") * 100
    )

    # Filter to post-2021
    school_post = school.filter(pl.col("year") >= 2021)
    df_post = df.filter(pl.col("year") >= 2021).select(
        ["fips", "year", "estimated_mmr_cov"]
    )

    # Join
    merged = school_post.join(df_post, on=["fips", "year"], how="inner")

    # Extract data for plotting
    x = merged["estimated_mmr_cov"].to_numpy()
    y = merged["mean_coverage_pct"].to_numpy()

    # Scatter plot
    ax.scatter(x, y, alpha=0.5, s=20, color="black")

    # OLS trend line
    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    r2 = r_value**2 * 100  # as a percentage  # type: ignore

    x_line = np.array([x.min(), x.max()])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, "r-", linewidth=2, label=f"OLS fit")

    # Annotate with R²
    ax.text(
        0.05,
        0.95,
        f"$R^2$ = {r2:.2f}%",
        transform=ax.transAxes,
        fontsize=24,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("County-level estimated MMR coverage (%)")
    ax.set_ylabel("School-level mean MMR coverage (%)")

    # Add diagonal line for reference
    ax.plot([0, 100], [0, 100], "k--", alpha=0.3, linewidth=1)

    ax.set_xlim(45, 103)
    ax.set_ylim(45, 103)

    return fig, ax


def plot_supplemental_sd_vs_mean(df, fig, ax):
    """
    Plot SD vs mean within-county MMR coverage (school-level data).
    """
    school = pl.read_csv(
        os.path.join(data_dir, "vaxrates_schools.csv"),
        null_values="NA",
        encoding="utf8-lossy",
    )
    school = school.with_columns(
        year=pl.col("year").str.slice(5, 9).cast(pl.Int64)
    )
    school = school.with_columns(
        mmr_rate=pl.when(pl.col("mmr_rate").is_not_null())
        .then(pl.col("mmr_rate"))
        .otherwise(pl.col("overall_rate"))
    )
    school = (
        school.group_by(["year", "county", "state"], maintain_order=True)
        .agg(
            mean_coverage=pl.col("mmr_rate").mean(),
            std_coverage=pl.col("mmr_rate").std(),
            n=pl.count("mmr_rate"),
        )
        .with_columns(pl.col("std_coverage").fill_null(0.0))
    )
    school = school.filter(
        pl.col("county").is_not_null()
        & pl.col("state").is_not_null()
        & pl.col("std_coverage").is_not_null()
    )

    fip_getter = AddFIPS()
    school = school.with_columns(
        fips=pl.struct(["county", "state"]).map_elements(
            lambda x: fip_getter.get_county_fips(
                x["county"], state=x["state"]
            ),
            return_dtype=pl.String,
        )
    ).filter(pl.col("fips").is_not_null())

    # Filter to post-2021 and join with analysis dataset
    school_post = school.filter(pl.col("year") >= 2021)
    df_post = df.filter(pl.col("year") >= 2021).select(["fips", "year"])

    # Join to keep only counties in our analysis
    merged = school_post.join(df_post, on=["fips", "year"], how="inner")

    x = merged["mean_coverage"].to_numpy() * 100  # → percent
    y = merged["std_coverage"].to_numpy() * 100  # → percentage points

    ax.scatter(x, y, alpha=0.4, s=15, color="black")

    ax.set_xlabel("Mean within-county MMR coverage (%)")
    ax.set_ylabel("SD of within-county MMR coverage (pp)")

    return fig, ax


def create_state_dummies(states):
    """
    Create state dummy variables (one-hot encoding)

    Parameters:
    -----------
    states : array-like
        State identifiers for each observation

    Returns:
    --------
    state_dummies : array, shape (n_obs, n_states - 1)
        Dummy variables for states (drops first state as reference)
    state_names : list
        Names of states corresponding to columns (excluding reference)
    """
    unique_states = np.unique(states)
    n_states = len(unique_states)
    n_obs = len(states)

    # Create mapping
    state_to_idx = {state: i for i, state in enumerate(unique_states)}

    # Create dummy matrix (exclude first state as reference)
    state_dummies = np.zeros((n_obs, n_states - 1))

    for i, state in enumerate(states):
        state_idx = state_to_idx[state]
        if state_idx > 0:  # Skip first state (reference category)
            state_dummies[i, state_idx - 1] = 1

    state_names = [f"state_{s}" for s in unique_states[1:]]

    return state_dummies, state_names


def create_regression_table(df, posterior_samples_post, adjacency_matrix_post):
    """
    Create supplemental regression table with MSR p-values and CIs.

    All rows use MSR inference with block permutation. Rows differ only in model specification.
    """
    # Prepare data (same as in plot_county_level_variation)
    school = pl.read_csv(
        os.path.join(data_dir, "vaxrates_schools.csv"),
        null_values="NA",
        encoding="utf8-lossy",
    )

    school = school.with_columns(
        year=pl.col("year").str.slice(5, 9).cast(pl.Int64)
    )

    school = school.with_columns(
        mmr_rate=pl.when(pl.col("mmr_rate").is_not_null())
        .then(pl.col("mmr_rate"))
        .otherwise(pl.col("overall_rate"))
    )

    school = school.group_by(
        ["year", "county", "state"], maintain_order=True
    ).agg(
        median_coverage=pl.col("mmr_rate").median(),
        mean_coverage=pl.col("mmr_rate").mean(),
        std_coverage=pl.col("mmr_rate").std(),
        IQR_coverage=pl.col("mmr_rate").quantile(0.75)
        - pl.col("mmr_rate").quantile(0.25),
        fraction_public=pl.col("school_type").eq("PUBLIC").sum()
        / pl.count("mmr_rate"),
        n=pl.count("mmr_rate"),
    )

    school = school.filter(pl.col("n") > 0)
    school = school.filter(
        pl.col("county").is_not_null() & pl.col("state").is_not_null()
    )
    school = school.with_columns(
        std_coverage=pl.when(
            pl.col("std_coverage").is_null() & (pl.col("n") == 1)
        )
        .then(0.0)
        .otherwise(pl.col("std_coverage"))
    )

    fip_getter = AddFIPS()
    school = school.with_columns(
        fips=pl.struct(["county", "state"]).map_elements(
            lambda x: fip_getter.get_county_fips(
                x["county"], state=x["state"]
            ),
            return_dtype=pl.String,
        )
    )
    school = school.filter(pl.col("fips").is_not_null())

    all_fips_post = (
        df.filter(pl.col("year") >= 2021)["fips"]
        .unique(maintain_order=True)
        .to_list()
    )

    raw_spatial_effect = posterior_samples_post["raw_spatial_effect"]
    spatial_sigma = posterior_samples_post["spatial_sigma"]
    spatial_effect = raw_spatial_effect * spatial_sigma[:, None]

    has_a_neighbor = adjacency_matrix_post.sum(axis=0) > 0
    counties_with_neighbors = np.array(all_fips_post)[has_a_neighbor]

    bayesian_pvalue_records = []
    for county_idx, county_fips in enumerate(counties_with_neighbors):
        county_samples = spatial_effect[:, county_idx]
        prop_positive = np.mean(county_samples > 0)
        prop_negative = np.mean(county_samples < 0)
        bayesian_p = float(np.minimum(prop_positive, prop_negative))
        bayesian_pvalue_records.append(
            {"fips": county_fips, "bayesian_p": bayesian_p}
        )

    bayesian_p_df = pl.DataFrame(bayesian_pvalue_records)

    school_post = school.filter(pl.col("year") >= 2021)
    cases_post = df.filter(pl.col("year") >= 2021).select(
        ["fips", "year", "cases_outbreak", "state"]
    )

    county_df = school_post.join(
        cases_post, on=["fips", "year"], how="inner"
    ).join(bayesian_p_df, on="fips", how="left")

    county_df = county_df.with_columns(
        std_coverage_pct=pl.col("std_coverage") * 100,
        cv_coverage=100 * pl.col("std_coverage") / pl.col("mean_coverage"),
    )

    county_df_reg = (
        county_df.filter(pl.col("bayesian_p").is_not_null())
        .with_columns(cases_outbreak_log=(pl.col("cases_outbreak") + 1).log())
        .to_pandas()
    )

    # ===== Subset adjacency matrix for full regression data =====
    unique_fips_in_regression = county_df_reg["fips"].unique()
    n_counties_in_regression = len(unique_fips_in_regression)

    print(f"\nRegression dataset:")
    print(f"  Total observations: {len(county_df_reg)}")
    print(f"  Unique counties: {n_counties_in_regression}")
    print(f"  Original adjacency matrix size: {adjacency_matrix_post.shape}")

    fips_to_new_idx = {
        fips: i for i, fips in enumerate(unique_fips_in_regression)
    }
    fips_to_original_idx = {fips: i for i, fips in enumerate(all_fips_post)}
    original_indices = [
        fips_to_original_idx[fips] for fips in unique_fips_in_regression
    ]
    adjacency_matrix_regression = adjacency_matrix_post[
        np.ix_(original_indices, original_indices)
    ]

    print(
        f"  Subsetted adjacency matrix size: {adjacency_matrix_regression.shape}"
    )

    county_df_reg["county_idx"] = county_df_reg["fips"].map(fips_to_new_idx)

    results = []

    # ===== Row 1: Base model (SD × outbreak size, NOT logged) =====
    print("\n=== Row 1: Base model: SD × outbreak size ===")

    county_df_reg["interaction_raw"] = (
        county_df_reg["std_coverage_pct"] * county_df_reg["cases_outbreak"]
    )

    X1 = county_df_reg[
        ["std_coverage_pct", "cases_outbreak", "interaction_raw"]
    ].values
    y1 = county_df_reg["bayesian_p"].values
    county_indices1 = county_df_reg["county_idx"].values

    msr1 = regression_msr_multiple_blocked(
        X1,
        y1,
        county_indices1,
        adjacency_matrix_regression,
    )

    # Extract interaction coefficient (index 3: intercept, std_cov, outbreak, interaction)
    interaction_coef = -msr1["coef"][3]
    interaction_p = msr1["p_values"][3]

    results.append(
        {
            "Model specification": "Base model: SD × outbreak size",
            "Interaction coefficient": f"{interaction_coef:.4e}",
            "p-value": f"{interaction_p:.4e}",
            "R2": f"{msr1['r_squared']:.4f}",
        }
    )

    # ===== Row 2: + fraction_public =====
    print("\n=== Row 2: + fraction_public ===")

    X2 = county_df_reg[
        [
            "std_coverage_pct",
            "cases_outbreak",
            "interaction_raw",
            "fraction_public",
        ]
    ].values
    y2 = county_df_reg["bayesian_p"].values
    county_indices2 = county_df_reg["county_idx"].values

    msr2 = regression_msr_multiple_blocked(
        X2,
        y2,
        county_indices2,
        adjacency_matrix_regression,
    )

    # Interaction is index 3
    interaction_coef = -msr2["coef"][3]
    interaction_p = msr2["p_values"][3]

    results.append(
        {
            "Model specification": "Controlling for fraction of schools that are public",
            "Interaction coefficient": f"{interaction_coef:.4e}",
            "p-value": f"{interaction_p:.4e}",
            "R2": f"{msr2['r_squared']:.4f}",
        }
    )

    # ===== Row 3: + State fixed effects (WITHOUT fraction_public) =====
    print("\n=== Row 3: + State fixed effects ===")

    # Create state dummy variables for regression data
    state_dummies_reg, state_names_reg = create_state_dummies(
        county_df_reg["state"].values
    )

    X3_base = county_df_reg[
        ["std_coverage_pct", "cases_outbreak", "interaction_raw"]
    ].values
    X3 = np.column_stack([X3_base, state_dummies_reg])
    y3 = county_df_reg["bayesian_p"].values
    county_indices3 = county_df_reg["county_idx"].values

    msr3 = regression_msr_multiple_blocked(
        X3,
        y3,
        county_indices3,
        adjacency_matrix_regression,
    )

    # Interaction is still index 3
    interaction_coef = -msr3["coef"][3]
    interaction_p = msr3["p_values"][3]

    n_states = len(np.unique(county_df_reg["state"]))

    results.append(
        {
            "Model specification": f"State fixed effects ({n_states} states)",
            "Interaction coefficient": f"{interaction_coef:.4e}",
            "p-value": f"{interaction_p:.4e}",
            "R2": f"{msr3['r_squared']:.4f}",
        }
    )

    # ===== Row 4: + mean MMR vaccine coverage =====
    print("\n=== Row 4: + mean MMR vaccine coverage ===")

    X4 = county_df_reg[
        [
            "std_coverage_pct",
            "cases_outbreak",
            "interaction_raw",
            "mean_coverage",
        ]
    ].values
    y4 = county_df_reg["bayesian_p"].values
    county_indices4 = county_df_reg["county_idx"].values

    msr4 = regression_msr_multiple_blocked(
        X4,
        y4,
        county_indices4,
        adjacency_matrix_regression,
    )

    interaction_coef = -msr4["coef"][3]
    interaction_p = msr4["p_values"][3]

    results.append(
        {
            "Model specification": "Controlling for mean MMR vaccine coverage",
            "Interaction coefficient": f"{interaction_coef:.4e}",
            "p-value": f"{interaction_p:.4e}",
            "R2": f"{msr4['r_squared']:.4f}",
        }
    )

    # ===== Row 5: CV instead of SD =====
    print("\n=== Row 5: CV × outbreak size ===")

    county_df_reg["interaction_cv"] = (
        county_df_reg["cv_coverage"] * county_df_reg["cases_outbreak"]
    )

    X5 = county_df_reg[
        ["cv_coverage", "cases_outbreak", "interaction_cv"]
    ].values
    y5 = county_df_reg["bayesian_p"].values
    county_indices5 = county_df_reg["county_idx"].values

    msr5 = regression_msr_multiple_blocked(
        X5,
        y5,
        county_indices5,
        adjacency_matrix_regression,
    )

    interaction_coef = -msr5["coef"][3]
    interaction_p = msr5["p_values"][3]

    results.append(
        {
            "Model specification": "CV × outbreak size",
            "Interaction coefficient": f"{interaction_coef:.4e}",
            "p-value": f"{interaction_p:.4e}",
            "R2": f"{msr5['r_squared']:.4f}",
        }
    )

    # ===== Row 6: Log outbreak size (with SD) =====
    print("\n=== Row 6: SD × log(outbreak size) ===")

    county_df_reg["interaction_log"] = (
        county_df_reg["std_coverage_pct"] * county_df_reg["cases_outbreak_log"]
    )

    X6 = county_df_reg[
        ["std_coverage_pct", "cases_outbreak_log", "interaction_log"]
    ].values
    y6 = county_df_reg["bayesian_p"].values
    county_indices6 = county_df_reg["county_idx"].values

    msr6 = regression_msr_multiple_blocked(
        X6,
        y6,
        county_indices6,
        adjacency_matrix_regression,
    )

    interaction_coef = -msr6["coef"][3]
    interaction_p = msr6["p_values"][3]

    results.append(
        {
            "Model specification": "SD × log(outbreak size + 1)",
            "Interaction coefficient": f"{interaction_coef:.4e}",
            "p-value": f"{interaction_p:.4e}",
            "R2": f"{msr6['r_squared']:.4f}",
        }
    )

    # ===== Row 7: Predicting multi-county outbreak status =====
    print("\n=== Row 7: Predicting multi-county outbreak indicator ===")

    # First, identify multi-county outbreaks (reusing logic from plot_county_level_variation)
    fips_to_adj_idx = {fips: i for i, fips in enumerate(all_fips_post)}
    df_post = df.filter(pl.col("year") >= 2021)

    multi_county_outbreak_records = []

    for row in county_df_reg.iterrows():
        row_data = row[1]
        fips = row_data["fips"]
        year = row_data["year"]
        cases = row_data["cases_outbreak"]

        # Check if this county has 3+ cases
        if cases >= 3 and fips in fips_to_adj_idx:
            adj_idx = fips_to_adj_idx[fips]

            # Get neighbors from adjacency matrix
            neighbor_indices = np.where(adjacency_matrix_post[adj_idx, :] > 0)[
                0
            ]
            neighbor_fips = [all_fips_post[i] for i in neighbor_indices]

            # Check if any neighbor also has 3+ cases in the same year
            neighbor_cases = df_post.filter(
                (pl.col("fips").is_in(neighbor_fips))
                & (pl.col("year") == year)
            )["cases_outbreak"].to_list()

            has_neighbor_with_outbreak = any(nc >= 3 for nc in neighbor_cases)
            is_multi_county = has_neighbor_with_outbreak
        else:
            is_multi_county = False

        multi_county_outbreak_records.append(
            {"index": row[0], "multi_county_outbreak": int(is_multi_county)}
        )

    multi_county_df = pd.DataFrame(multi_county_outbreak_records).set_index(
        "index"
    )
    county_df_reg["multi_county_outbreak"] = multi_county_df[
        "multi_county_outbreak"
    ]

    # Filter to only counties with outbreaks (>=3 cases)
    county_df_reg_outbreaks = county_df_reg[
        county_df_reg["cases_outbreak"] >= 3
    ].copy()

    # Create interaction term (using raw outbreak size, not logged)
    county_df_reg_outbreaks["interaction_raw"] = (
        county_df_reg_outbreaks["std_coverage_pct"]
        * county_df_reg_outbreaks["cases_outbreak"]
    )

    # Prepare X and y - NOW y is the multi-county outbreak indicator
    X7 = county_df_reg_outbreaks[
        ["std_coverage_pct", "cases_outbreak", "interaction_raw"]
    ].values
    y7 = county_df_reg_outbreaks["multi_county_outbreak"].values  # Binary 0/1

    # Need to subset adjacency matrix for outbreak-only counties
    outbreak_fips = county_df_reg_outbreaks["fips"].unique()
    outbreak_to_regression_idx = [
        fips_to_new_idx[fips]
        for fips in outbreak_fips
        if fips in fips_to_new_idx
    ]
    adjacency_outbreak = adjacency_matrix_regression[
        np.ix_(outbreak_to_regression_idx, outbreak_to_regression_idx)
    ]

    # Re-index counties for this subset
    outbreak_fips_to_outbreak_idx = {
        fips: i for i, fips in enumerate(outbreak_fips)
    }
    county_indices7_reindexed = np.array(
        [
            outbreak_fips_to_outbreak_idx[fips]
            for fips in county_df_reg_outbreaks["fips"]
        ]
    )

    msr7 = regression_msr_multiple_blocked(
        X7,
        y7,
        county_indices7_reindexed,
        adjacency_outbreak,
    )

    # Extract interaction coefficient (index 3: intercept, std_cov, outbreak, interaction)
    interaction_coef = msr7["coef"][3]
    interaction_p = msr7["p_values"][3]

    results.append(
        {
            "Model specification": "Predicting multi-county outbreak: SD × outbreak size",
            "Interaction coefficient": f"{interaction_coef:.4e}",
            "p-value": f"{interaction_p:.4e}",
            "R2": f"{msr7['r_squared']:.4f}",
        }
    )

    print(f"  Interaction coefficient: {interaction_coef:.4e}")
    print(f"  p-value: {interaction_p:.4e}")
    print(
        f"  N multi-county outbreaks: {county_df_reg_outbreaks['multi_county_outbreak'].sum()}"
    )
    print(
        f"  N single-county outbreaks: {(county_df_reg_outbreaks['multi_county_outbreak'] == 0).sum()}"
    )
    print(f"  Total outbreaks (>=3 cases): {len(county_df_reg_outbreaks)}")

    # Create DataFrame
    results_df = pd.DataFrame(results)

    print("\n=== Regression Table Summary ===")
    print(results_df.to_string(index=False))

    return results_df


if __name__ == "__main__":
    df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
    )

    fig, ax = setup_plot(2, 2)  # type: ignore

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
            "final_outbreak_size_predictions_[2021, 2022, 2023, 2024, 2025]_five_year=False.pkl",
        ),
        "rb",
    ) as f:
        naive_model = pickle.load(f)

    # performance
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
    plot_predictions(
        df.filter(pl.col("year").is_in([2021, 2022, 2023, 2024, 2025])),
        predictive_samples["cases"],
        naive_model,
        fig,
        ax[0, 0],  # type: ignore
        show_r2=True,
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
    plot_ecological_determinants(
        posterior_samples_pre,
        posterior_samples,
        df,
        fig,
        ax[0, 1],  # type: ignore
    )

    ax[1, 0].text(  # type: ignore
        -0.25,
        1.08,
        "c",
        transform=ax[1, 0].transAxes,  # type: ignore
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
        ax[1, 0],  # type: ignore
        year_bins=[(2011, 2015), (2016, 2020), (2021, 2024), (2025, 2025)],
    )

    ax[1, 1].text(  # type: ignore
        -0.25,
        1.08,
        "d",
        transform=ax[1, 1].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_county_level_variation(
        df,
        posterior_samples,
        adjacency_matrix_observed,
        fig,
        ax[1, 1],  # type: ignore
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

    # Create supplemental figure
    print("\n" + "=" * 70)
    print("Creating supplemental figure...")
    print("=" * 70)

    fig_supp, ax_supp = setup_plot(1, 2, figsize=(20, 10))  # type: ignore

    ax_supp[0].text(  # type: ignore
        -0.25,
        1.08,
        "a",
        transform=ax_supp[0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_supplemental_mmr_coverage_correlation(df, fig_supp, ax_supp[0])  # type: ignore

    ax_supp[1].text(  # type: ignore
        -0.25,
        1.08,
        "b",
        transform=ax_supp[1].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_supplemental_sd_vs_mean(
        df,
        fig_supp,
        ax_supp[1],  # type: ignore
    )

    fig_supp.tight_layout()
    fig_supp.savefig(
        os.path.join(
            fig_dir,
            "figure_supplemental_mmr_coverage_and_cluster_variability.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    fig_supp.savefig(
        os.path.join(
            fig_dir,
            "figure_supplemental_mmr_coverage_and_cluster_variability.pdf",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    # Create supplemental regression table
    print("\n" + "=" * 70)
    print("Creating supplemental regression table...")
    print("=" * 70)

    regression_table = create_regression_table(
        df, posterior_samples, adjacency_matrix_observed
    )

    # Save table to CSV
    regression_table.to_csv(
        path_or_buf=os.path.join(
            fig_dir,
            "table_supplemental_variability_spatially_related_regressions.csv",
        ),
        index=False,
    )

    print("\nRegression table saved!")
    print(regression_table.to_string(index=False))
