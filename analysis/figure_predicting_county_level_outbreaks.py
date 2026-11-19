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
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
from scipy import stats
from scipy.sparse.csgraph import connected_components

from outbreak_size import adjacency, waic
from outbreak_size.ccd_enrollment import county_sds, match_to_ccd
from outbreak_size.coverage_effect import (
    percent_change_per_pp,
)
from outbreak_size.model_colors import (
    EARLIER_PERIOD_COLOR,
    LATER_PERIOD_COLOR,
    MECHANISTIC_COLOR,
    STATISTICAL_COLOR,
)
from outbreak_size.msr import regression_msr_multiple_blocked
from outbreak_size.msr_logistic import logistic_msr_test
from outbreak_size.outbreak_clusters import (
    CLUSTER_NAMES,
    cluster_seeds_2025,
    clusters_holding,
    outbreak_cases_all_counties,
    outbreak_place_clusters,
    place_adjacency_matrix,
    region_members,
)
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

# the Beta draws below are seeded so the figure is reproducible
rng = np.random.default_rng(params["seed"])


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
    r2_x_frac=0.00025,
    r2_x_axes_frac=None,
    color_naive=MECHANISTIC_COLOR,
    color_statistical=STATISTICAL_COLOR,
    waic_values=None,  # (statistical WAIC, naive WAIC); replaces the R2 text
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
        ecolor=color_naive,
        c=color_naive,
        capsize=0,
        alpha=0.4,
        label=label_naive,
    )

    ax.errorbar(
        df["cases_outbreak"] + offset,
        median,
        yerr=[median - lower, upper - median],
        fmt="o",
        ecolor=color_statistical,
        c=color_statistical,
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

    if show_r2 or waic_values is not None:
        # r2_x_axes_frac, when given, places the text at a fixed fraction of
        # the axes width (independent of the symlog data scale) instead of
        # r2_x_frac's data-coordinate fraction; use this on any figure whose
        # aspect ratio differs from the one r2_x_frac was tuned for, since a
        # data-coordinate x is compressed almost flat near 0 on a symlog axis
        # and barely moves with small changes to r2_x_frac
        if r2_x_axes_frac is not None:
            text_kwargs = dict(
                transform=blended_transform_factory(ax.transAxes, ax.transData)
            )
            r2_x = r2_x_axes_frac
        else:
            text_kwargs = {}
            r2_x = r2_x_frac * max(df["cases_outbreak"].max(), upper.max())  # type: ignore

        # add a text annotation of the R2, or of the WAIC if given
        if waic_values is not None:
            statistical_text = f"WAIC = {waic_values[0]:.0f}"
        else:
            R2 = jnp.corrcoef(df["cases_outbreak"].to_jax(), median)[0, 1] ** 2
            R2 = R2 * 100  # as a percentage
            statistical_text = f"$R^2$ = {R2:.2f}%"
        ax.text(
            r2_x,
            0.35 * max(df["cases_outbreak"].max(), upper.max()),  # type: ignore
            statistical_text,
            fontsize=30,
            color=color_statistical,
            **text_kwargs,
        )

        if waic_values is not None:
            naive_text = f"WAIC = {waic_values[1]:.0f}"
        else:
            R2 = (
                jnp.corrcoef(
                    df["cases_outbreak"].to_jax(), naive_model["median"]
                )[0, 1]
                ** 2
            )
            R2 = R2 * 100  # as a percentage
            naive_text = f"$R^2$ = {R2:.2f}%"
        ax.text(
            r2_x,
            0.63 * max(df["cases_outbreak"].max(), upper.max()),  # type: ignore
            naive_text,
            fontsize=30,
            color=color_naive,
            **text_kwargs,
        )
    ax.set_ylim(-0.05)

    return fig, ax


def plot_ecological_determinants(samples_pre, samples_post, df, fig, ax):
    pooled_covariates = [
        "MMR coverage",
        "Nonmedical\nexemptions",
        "Medical\nexemptions",
    ]
    covariates = [  # all the covariates of the fit; see keep below
        "% < 5yo",
        "% school aged",
        "Average household size",
        "Standard deviation of household size",
        "% public schooled",
        "% white",
        "% graduate degree",
        "Population density\n(per 10-fold increase)",
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
        "Population density\n(per 10-fold increase)": "log_population_density",
        "Income\n(10,000 2024 USD)": "log_income",
    }

    offset = 0.1
    year_medians = []
    period_effects = []  # (medians, lowers, uppers) of each period
    period_90 = []  # (5th, 95th percentiles) of each period
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
        lower_90 = []
        upper_90 = []

        for idx, covariate in enumerate(pooled_covariates):
            samples_of_interest = samples["iv_global"][:, idx]
            # per SD of the covariate on the scale of the fit -> % change per
            # 1pp, averaged over the county-years of all the years (see
            # coverage_effect)
            samples_of_interest = percent_change_per_pp(
                samples_of_interest,
                df.filter(pl.col("year").is_in(years)),
                covariate_name_map[covariate],
                df,
                COVARIATE_SCALE,
                MMR_EFFECTIVENESS,
            )
            median_effects.append(jnp.median(samples_of_interest, axis=0))
            lower_effects.append(
                jnp.quantile(samples_of_interest, 0.025, axis=0)
            )
            upper_effects.append(
                jnp.quantile(samples_of_interest, 0.975, axis=0)
            )
            lower_90.append(jnp.quantile(samples_of_interest, 0.05, axis=0))
            upper_90.append(jnp.quantile(samples_of_interest, 0.95, axis=0))

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
                # log_income is log10, so d(log10 income)/d(income) =
                # 1 / (income * ln(10)); the coefficient is per log10 unit
                samples_of_interest = samples_of_interest / (
                    mean_income * jnp.log(10)
                )
                # now convert from dollars to 10,000 USD
                samples_of_interest = samples_of_interest * 10000
            # log_population_density is log10, so the coefficient is already
            # per 10-fold increase in density: no unit conversion
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
            lower_90.append(jnp.quantile(samples_of_interest, 0.05, axis=0))
            upper_90.append(jnp.quantile(samples_of_interest, 0.95, axis=0))

        period_effects.append((median_effects, lower_effects, upper_effects))
        period_90.append((lower_90, upper_90))
        year_medians.append(median_effects)

    # MMR coverage and the exemptions are always shown; the other covariates
    # are shown when they are significant at the 5% level (the 90% credible
    # interval, the 5th to the 95th percentile, excludes 0) in 2021-2025 or in
    # 2011-2020
    def significant(i, period):
        lower, upper = period_90[period]
        return bool(lower[i] > 0 or upper[i] < 0)

    keep = [
        i
        for i in range(len(all_covariates))
        if i < len(pooled_covariates) or significant(i, 0) or significant(i, 1)
    ]
    for i, name in enumerate(all_covariates):
        print(
            f"Panel b, {name!r}: 2011-2020 "
            f"{float(period_effects[0][0][i]):.1f}% "
            f"(90% interval [{float(period_90[0][0][i]):.1f}, "
            f"{float(period_90[0][1][i]):.1f}]), 2021-2025 "
            f"{float(period_effects[1][0][i]):.1f}% "
            f"(90% interval [{float(period_90[1][0][i]):.1f}, "
            f"{float(period_90[1][1][i]):.1f}]): "
            + (
                "shown"
                if i in keep
                else "left out (not significant at 5% in either period)"
            )
        )
    all_covariates = [all_covariates[i] for i in keep]

    for year_idx, years in enumerate(
        [
            [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
            [2021, 2022, 2023, 2024, 2025],
        ]
    ):
        median_effects, lower_effects, upper_effects = (
            [effects[i] for i in keep] for effects in period_effects[year_idx]
        )
        # Plot error bars
        color = EARLIER_PERIOD_COLOR if year_idx == 0 else LATER_PERIOD_COLOR
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

    ax.axvline(0, ls="--", color="k")

    ax.legend(loc=(0.01, 1.05), edgecolor="black")

    ax.set_yticks(range(len(all_covariates)))  # type: ignore
    ax.set_yticklabels(all_covariates)  # type: ignore
    ax.set_xlabel("% change in incidence per 1 unit\nincrease in predictor")
    ax.set_xlim(-100, 100)

    # Add "Lower risk" and "Higher risk" annotations at the top
    # (at the top edge of the axes, whatever the number of covariates shown)
    y_top = 1.0
    ax.text(
        -60,
        y_top,
        "← Lower risk",
        transform=ax.get_xaxis_transform(),  # x in data, y in the axes
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
        transform=ax.get_xaxis_transform(),
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
    n_series=1,
    series_index=0,
    connected_groups=False,
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
        n_series, series_index : int, optional
            To draw several calls of this function as bars side by side on
            one axis: the number of calls, and which one (from 0) this is.
        connected_groups : bool, optional
            If True, the counties with any cases in a year are grouped by
            bordering each other (directly or through other counties of the
            group), and the cases of the outbreak counties (3 or more cases)
            count as multi-county when their group has two outbreak counties
            that border each other and at least half of its counties are
            outbreak counties.
            df must then hold the counties with 1 or 2 cases as well.
    """

    def count_cases_with_neighbors(year_df, adjacency_matrix, all_fips):
        """
        Count cases in outbreak counties with at least one neighboring outbreak.

        Parameters:
        -----------
        year_df : polars.DataFrame
            Data for a single year
        adjacency_matrix : numpy.ndarray
            Adjacency matrix for all counties in the period
        all_fips : list
            All FIPS codes in the period (matching adjacency matrix order)

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

        if connected_groups:
            # all counties with any cases, grouped by bordering each other
            with_cases = year_df.filter(
                (pl.col("cases_outbreak") > 0)
                & (pl.col("fips").is_in(all_fips))
            )["fips"].to_list()
            at = [fips_to_idx[fips] for fips in with_cases]
            borders = adjacency_matrix[np.ix_(at, at)]
            _, group = connected_components(borders, directed=False)
            is_outbreak = np.array(
                [fips in fips_to_cases for fips in with_cases]
            )
            # outbreak counties that border another outbreak county
            has_outbreak_neighbor = is_outbreak & (
                (borders > 0) @ is_outbreak > 0
            )
            for g in np.unique(group):
                in_group = group == g
                n_outbreaks = int((in_group & is_outbreak).sum())
                if (
                    in_group & has_outbreak_neighbor
                ).any() and 2 * n_outbreaks >= in_group.sum():
                    cases_with_outbreak_neighbors += sum(
                        fips_to_cases[fips]
                        for fips, member in zip(with_cases, in_group)
                        if member and fips in fips_to_cases
                    )
            return cases_with_outbreak_neighbors, total_cases

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

    # The bar drawn: its color and label
    thresholds_and_styles = [
        (None, "#8b8c89", "All"),
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
                                all_fips_pre,
                            )
                        )
                    else:
                        cases_with_neighbors, total_cases = (
                            count_cases_with_neighbors(
                                year_df,
                                adjacency_matrix_post,
                                all_fips_post,
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
                            p_sample = stats.beta.rvs(
                                alpha, beta_param, random_state=rng
                            )
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
                                all_fips_pre,
                            )
                        )
                    else:
                        cases_with_neighbors, total_cases = (
                            count_cases_with_neighbors(
                                year_df,
                                adjacency_matrix_post,
                                all_fips_post,
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
                        stats.beta.rvs(
                            alpha, beta_param, size=n_samples, random_state=rng
                        )
                        * 100
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
    n_bars_per_group = len(thresholds_and_styles) * n_series

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

        x_pos = (
            x_positions
            + (series_index * len(thresholds_and_styles) + bar_idx) * bar_width
        )

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

    return fig, ax


def read_old_school_data():
    """
    The OLD school data: the raw Washington Post file (vaxrates_schools.csv)
    as one row per county and year (outbreak year): fips, year, the mean and
    SD of the MMR rates of the schools (fractions). It is NOT the data of the
    main figure's panel d (that is read_school_sd_data, from
    school_vaccination_counties.csv). The file is read as it is: FIPS come from
    the county label, and none of the checks of
    analysis/assemble_school_vaccination.py (point in county, fallback points,
    duplicates, placeholder rates, qc_gap_ok) are applied. It is used only for
    the school-data coverage printout (print_school_data_coverage) and as the
    default of plot_county_level_variation, which the supplemental figure calls
    with the new data (read_new_school_data).
    """
    if not os.path.exists(os.path.join(data_dir, "vaxrates_schools.csv")):
        print("skipped panel: vaxrates_schools.csv not provided.")
        return None
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
        # the schools of a county are all its schools: the population SD
        std_coverage=pl.col("mmr_rate").std(ddof=0),
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
    return school


def read_new_school_data():
    """
    The new school data (school_vaccination_counties.csv, every school year of
    the study window) in the form of read_old_school_data: the unweighted mean
    and SD of the MMR rates of the schools of a county (percent in the file,
    fractions here). The SD is the population SD over the schools, so a county
    with one school has an SD of 0, as in the old data.
    """
    if not os.path.exists(
        os.path.join(data_dir, "school_vaccination_counties.csv")
    ):
        print("skipped panel: school_vaccination_counties.csv not provided.")
        return None
    return pl.read_csv(
        os.path.join(data_dir, "school_vaccination_counties.csv"),
        schema_overrides={"county_fips": pl.String},
    ).select(
        fips="county_fips",
        year="year_end",
        mean_coverage=pl.col("mean_unweighted") / 100,
        std_coverage=pl.col("sd_unweighted") / 100,
    )


def multi_county_flags(
    df_spread,
    adjacency_matrix_spread_pre,
    adjacency_matrix_spread,
    min_cases=3,
):
    """
    The multi-county flag of panel c for the county-years of `min_cases` (3,
    as in panel c) or more cases: fips, year and multi_county, whether the
    county borders another county of `min_cases` or more cases the same year,
    and group, the outbreak of the county (counties of `min_cases` or more
    cases joined by borders, directly or through other such counties, in the
    same year; a county alone is its own group).
    Neighbors are found among the places of df_spread (regions left out) of
    the same period, 2011-2020 or 2021-2025, with that period's adjacency
    matrix.
    """
    flags = []
    for before_2021, matrix in (
        (True, adjacency_matrix_spread_pre),
        (False, adjacency_matrix_spread),
    ):
        period = df_spread.filter((pl.col("year") < 2021) == before_2021)
        position = {
            fips: i
            for i, fips in enumerate(
                period["fips"].unique(maintain_order=True).to_list()
            )
        }
        matrix = np.asarray(matrix) > 0
        for year in sorted(period["year"].unique()):
            big = period.filter(
                (pl.col("year") == year)
                & (pl.col("cases_outbreak") >= min_cases)
            )
            at = [position[fips] for fips in big["fips"]]
            among_big = matrix[np.ix_(at, at)]
            has_neighbor = among_big.sum(axis=1) > 0
            _, label = connected_components(among_big, directed=False)
            flags.append(
                big.select("fips", "year").with_columns(
                    multi_county=pl.Series(has_neighbor),
                    group=pl.Series(
                        [f"{year}_{k}" for k in label], dtype=pl.String
                    ),
                )
            )
    return pl.concat(flags)


def print_school_data_coverage(
    df, df_spread, adjacency_matrix_spread_pre, adjacency_matrix_spread
):
    """
    Prints the numbers of the text on the school-level data: how many schools
    and school districts there are, and how much of the measles cases of the
    analysis dataset (cases_outbreak of df, regions included) are in
    county-years that have school data. The multi-county cases are those of
    the multi-county flag of panel c (a county of 3 or more cases that borders
    another one the same year; neighbors among the places of df_spread of the
    same period, 2011-2020 or 2021-2025; regions left out). That flag differs
    from the one inside plot_county_level_variation, which exists from 2021
    on. Copied school years (a year repeated from another year) are left out.
    """
    schools_file = os.path.join(data_dir, "school_vaccination_schools.csv")
    counties_file = os.path.join(data_dir, "school_vaccination_counties.csv")
    if not (os.path.exists(schools_file) and os.path.exists(counties_file)):
        print("skipped school data coverage: school files not provided.")
        return
    keys = ["state", "school_name", "lon", "lat"]
    schools = pl.read_csv(
        schools_file,
        columns=[*keys, "year_end", "unit", "copied_year"],
        infer_schema_length=None,
    ).filter(pl.col("year_end").is_between(2014, 2025))
    own = schools.filter(~pl.col("copied_year"))
    counties = pl.read_csv(
        counties_file,
        schema_overrides={"county_fips": pl.String},
        columns=["county_fips", "year_end", "copied_year"],
    ).filter(
        pl.col("year_end").is_between(2014, 2025) & ~pl.col("copied_year")
    )
    print("\n=== School data coverage (numbers of the text) ===")
    print(
        "School-level file, school years 2014-2025: "
        f"{schools.select(keys).unique().height:,} distinct schools and "
        "districts (state, name and location; the file has no school ID), "
        f"{own.select(keys).unique().height:,} without the copied years"
    )
    for unit, n in (
        own.group_by("unit")
        .agg(n=pl.struct(keys).n_unique())
        .sort("unit")
        .iter_rows()
    ):
        print(f"  {unit}: {n:,}")
    print(
        f"County file: {counties['county_fips'].n_unique():,} counties, "
        f"{counties.height:,} county-years (copied years left out)"
    )

    def has_school(school):
        return (
            school.select("fips", "year")
            .unique()
            .with_columns(has_school=pl.lit(True))
        )

    new_has = has_school(counties.select(fips="county_fips", year="year_end"))

    def coverage(label, has, lo, hi):
        # cases: all measles cases; cases_outbreak: the same without the
        # imported cases (the cases of the models)
        rows = (
            df.filter(pl.col("year").is_between(lo, hi))
            .join(has, on=["fips", "year"], how="left")
            .with_columns(pl.col("has_school").fill_null(False))
        )
        measles = rows.filter(pl.col("cases") > 0)
        covered = measles.filter(pl.col("has_school"))
        local = measles.filter(pl.col("cases_outbreak") > 0)
        n_all, n_covered = measles["cases"].sum(), covered["cases"].sum()
        o_all = measles["cases_outbreak"].sum()
        o_covered = covered["cases_outbreak"].sum()
        print(
            f"{label}, {lo}-{hi}: {measles.height} county-years with "
            f"measles, {covered.height} with school data "
            f"({100 * covered.height / measles.height:.1f}%); "
            f"{local.filter(pl.col('has_school')).height} of the "
            f"{local.height} with outbreak cases (not only imported cases) "
            f"have school data. All measles cases: {n_covered:,} of "
            f"{n_all:,} = {100 * n_covered / n_all:.1f}%. Outbreak cases: "
            f"{o_covered:,} of {o_all:,} = {100 * o_covered / o_all:.1f}%"
        )

    old = read_old_school_data()
    if old is not None:
        coverage("Washington Post data", has_school(old), 2024, 2025)
    for lo, hi in ((2024, 2025), (2014, 2025), (2014, 2024), (2025, 2025)):
        coverage("New school data", new_has, lo, hi)

    flagged = df.join(
        multi_county_flags(
            df_spread, adjacency_matrix_spread_pre, adjacency_matrix_spread
        ),
        on=["fips", "year"],
        how="left",
    ).join(new_has, on=["fips", "year"], how="left")
    flagged = flagged.with_columns(pl.col("has_school").fill_null(False))
    for lo, hi in ((2014, 2025), (2014, 2024), (2025, 2025)):
        in_years = flagged.filter(pl.col("year").is_between(lo, hi))
        multi = in_years.filter(pl.col("multi_county") == True)  # noqa: E712
        covered = multi.filter(pl.col("has_school"))["cases_outbreak"].sum()
        total = multi["cases_outbreak"].sum()
        # places of 3+ cases that are several counties (a region) have no flag
        no_flag = in_years.filter(
            (pl.col("cases_outbreak") >= 3) & pl.col("multi_county").is_null()
        )
        extra = no_flag["cases_outbreak"].sum()
        print(
            f"Multi-county cases (panel c flag), {lo}-{hi}: {multi.height} "
            f"county-years, {total:,} outbreak cases "
            f"({100 * total / in_years['cases_outbreak'].sum():.1f}% of the "
            f"outbreak cases); with school data: {covered:,} of {total:,} = "
            f"{100 * covered / total:.1f}%. Regions with 3+ cases, left out "
            f"of the flag: {no_flag.height} places, {extra} cases; with "
            f"them all counted as multi-county and none with school data: "
            f"{covered:,} of {total + extra:,} = "
            f"{100 * covered / (total + extra):.1f}%"
        )
        for row in no_flag.sort("year", "fips", "cases_outbreak").iter_rows(
            named=True
        ):
            print(
                f"    region left out: {row['fips']}, {row['year']}, "
                f"{row['cases_outbreak']} outbreak cases"
            )


SD_TEST_PERMUTATIONS = 99999


def read_school_sd_data():
    """
    The new school data (school_vaccination_counties.csv) as one row per
    county and outbreak year for panel d: the unweighted SD and mean of the
    school MMR rates (percentage points), whether the county has only
    districts (is_district) or schools and districts (is_mixed), the log of the
    number of schools and districts with a rate (log_n_schools), and the QC
    columns of the file (county_mmr, mean_gap_pp, qc_gap_ok: the mean within
    school_mean_qc_max_gap_pp of the county MMR of the same school year,
    params.yml; empty when that county MMR is imputed). Copied school years (a
    year repeated from another year) are left out.
    """
    path = os.path.join(data_dir, "school_vaccination_counties.csv")
    if not os.path.exists(path):
        print("skipped panel: school_vaccination_counties.csv not provided.")
        return None
    school = pl.read_csv(path, schema_overrides={"county_fips": pl.String})
    assert "qc_gap_ok" in school.columns, (
        "the county file has no qc_gap_ok: run "
        "analysis/assemble_school_vaccination.py"
    )
    return school.filter(~pl.col("copied_year")).select(
        pl.col("county_fips").alias("fips"),
        pl.col("year_end").alias("year"),
        pl.col("sd_unweighted").alias("sd"),
        pl.col("mean_unweighted").alias("school_mean"),
        (pl.col("unit_type") == "district")
        .cast(pl.Float64)
        .alias("is_district"),
        (pl.col("unit_type") == "mixed").cast(pl.Float64).alias("is_mixed"),
        pl.col("n_schools"),
        pl.col("n_schools").log().alias("log_n_schools"),
        "county_mmr",
        "county_mmr_imputed",
        "mean_gap_pp",
        "qc_gap_ok",
    )


# Panel d and its supplemental figure and table: are the counties of
# multi-county outbreaks more homogeneous in their school MMR rates than those
# of single-county outbreaks? Outbreak year 2025 (school year 2024-25). The
# counties are those of panel c with HOMOGENEITY_MIN_CASES or more outbreak
# cases in 2025, whether or not they have vaccine data (regions left out); one
# is a multi-county county when it borders another such county (the flag of
# panel c, multi_county_flags). The test is the spatially constrained
# permutation test of the logistic regression of multi-county (1) vs
# single-county (0) outbreak on the SD of the school MMR rates
# (outbreak_size.msr_logistic). The base model has the SD, the outbreak size
# (cases) and their product; SD and size are centered, so the SD term is the
# effect at an average-size county. Every other model of HOMOGENEITY_MODELS is
# the base model with the change or the control that its name says. The SD of
# the base model, and so of every model but the last two, is weighted by CCD
# kindergarten counts for the counties where every school is matched to the
# CCD, and unweighted for the others (panel d shows the base model). The last
# two rows are the SD weighted school by school (the panel of the supplemental
# figure) and the unweighted SD of all the counties.
HOMOGENEITY_YEAR = 2025
HOMOGENEITY_MIN_CASES = 5
HOMOGENEITY_PERMUTATIONS = SD_TEST_PERMUTATIONS
# the weighted SD of the school MMR rates: by the kindergarten counts of the
# CCD (analysis/read_ccd_enrollment.py), for the counties where every school
# is matched to the CCD; and school by school, the CCD count, else our own
# enrollment, else the county is unweighted (outbreak_size.ccd_enrollment)
HOMOGENEITY_BASE = (
    "Base: SD (weighted by CCD kindergarten counts where the county has "
    "them), outbreak size, SD × outbreak size"
)
HOMOGENEITY_HIERARCHY_MODEL = (
    "Base with SD weighted school by school: CCD kindergarten counts, then "
    "our own enrollment (otherwise unweighted)"
)


def _homogeneity_model(**changes):
    """A model of HOMOGENEITY_MODELS: the base model with `changes`."""
    spec = dict(
        min_cases=HOMOGENEITY_MIN_CASES,
        size="cases",  # or "log_cases", log(cases + 1)
        interaction=True,  # SD x size
        x="sd",  # or "cv", SD / mean of the school rates (percent)
        controls=[],  # columns of the county table (homogeneity_samples)
        state_fe=False,  # a dummy for each state but the one with most counties
        # None: the SD of the base model (CCD-weighted where the county has the
        # counts, column sd); or "hierarchy" (school by school, column
        # sd_hierarchy) or "unweighted" (column sd_unweighted)
        weighting=None,
    )
    spec.update(changes)
    return spec


HOMOGENEITY_MODELS = {
    HOMOGENEITY_BASE: _homogeneity_model(),
    "Base with log(outbreak size + 1) in place of outbreak size": (
        _homogeneity_model(size="log_cases")
    ),
    "Base + county MMR coverage": _homogeneity_model(controls=["county_mmr"]),
    "Base + rural/urban code": _homogeneity_model(controls=["rucc"]),
    "Base + number of schools": _homogeneity_model(controls=["n_schools"]),
    "Base with CV (SD / mean) in place of SD (odds ratio per percentage point of CV)": (
        _homogeneity_model(x="cv")
    ),
    "Base + state fixed effects": _homogeneity_model(state_fe=True),
    "Base without SD × outbreak size": _homogeneity_model(interaction=False),
    "Base, counties with 3 or more outbreak cases": _homogeneity_model(
        min_cases=3
    ),
    HOMOGENEITY_HIERARCHY_MODEL: _homogeneity_model(weighting="hierarchy"),
    "Base with the unweighted SD of all the counties": (
        _homogeneity_model(weighting="unweighted")
    ),
}
# the model of panel d: the SD weighted by CCD kindergarten counts only (9 of
# the 36 counties; the others unweighted). Set it to the model of the
# unweighted SD for the SD of all the counties
HOMOGENEITY_PANEL_MODEL = HOMOGENEITY_BASE
# the width (pp of SD) of the kernel of the violins of panel d: a fixed width,
# so that a lone county far above the others is not stretched into a tail
HOMOGENEITY_VIOLIN_BANDWIDTH = 2.5
_HOMOGENEITY_TESTS = {}  # the tests done: the panel and the table share them


def read_ccd_county_sds(fips):
    """
    The enrollment-weighted SD of the school MMR rates of the counties `fips`
    in HOMOGENEITY_YEAR (outbreak_size.ccd_enrollment; the CCD files are
    written by analysis/read_ccd_enrollment.py): fips, sd_ccd (CCD
    kindergarten counts only, counties with every school matched) and
    sd_hierarchy (school by school: the CCD count, else our own enrollment).
    Empty where a county cannot be weighted.
    """
    paths = [
        os.path.join(data_dir, f"ccd_2024_25_{kind}.csv")
        for kind in ("schools", "districts")
    ]
    assert all(os.path.exists(path) for path in paths), (
        "no CCD files: run analysis/read_ccd_enrollment.py"
    )
    # the ids have leading zeros: read them as text
    ccd_schools = pl.read_csv(
        paths[0],
        schema_overrides={
            "ncessch": pl.String,
            "leaid": pl.String,
            "county_fips": pl.String,
        },
    )
    ccd_districts = pl.read_csv(
        paths[1], schema_overrides={"leaid": pl.String}
    )
    rows = (
        pl.read_csv(
            os.path.join(data_dir, "school_vaccination_schools.csv"),
            schema_overrides={"county_fips": pl.String},
            infer_schema_length=None,
        )
        .filter(
            (pl.col("year_end") == HOMOGENEITY_YEAR)
            & ~pl.col("copied_year")
            & pl.col("county_fips").is_in(list(fips))
            & pl.col("mmr_rate").is_not_null()
        )
        .select(
            "state",
            "county_fips",
            "school_name",
            "unit",
            "lon",
            "lat",
            "mmr_rate",
            "enrollment",
        )
    )
    matched = match_to_ccd(
        rows,
        ccd_schools,
        ccd_districts,
        radius_m=params["school_vaccination"]["match_radius_m"],
    )
    weights = county_sds(matched)
    print(
        f"CCD weights for {rows['county_fips'].n_unique()} counties: "
        f"{(matched['match'] != 'none').sum()} of {matched.height} school "
        "rows are matched to the CCD"
    )
    return weights.select(
        fips="county_fips", sd_ccd="sd_ccd", sd_hierarchy="sd_hierarchy"
    )


def homogeneity_samples(
    df, df_spread, adjacency_matrix_spread_pre, adjacency_matrix_spread
):
    """
    The county tables of panel d, by the fewest outbreak cases a county needs
    (HOMOGENEITY_MODELS has the values): the 2025 counties with that many
    outbreak cases and a school SD (read_school_sd_data), sorted by fips, with
    multi (borders another such county), group (its outbreak) and the columns
    of the models: cases, log_cases (log(cases + 1)), cv, rucc (rural/urban
    code, 1-3 metro, 4-9 nonmetro), state (the fips prefix).
    The columns sd and cv (SD / mean of the school rates; the mean is
    unweighted) use the SD weighted by CCD kindergarten counts where every
    school of the county is matched to the CCD and the unweighted SD elsewhere
    (weighted: whether it was weighted); sd_unweighted is the unweighted SD of
    all the counties and sd_hierarchy the SD weighted school by school (the
    unweighted SD where a school has no weight), with sd_hierarchy_weighted.
    """
    school = read_school_sd_data()
    if school is None:
        return None
    rucc = read_rucc(os.path.join(data_dir, "rural_urban_continuum_codes"))
    cases = df_spread.filter(pl.col("year") == HOMOGENEITY_YEAR).select(
        "fips", "cases_outbreak"
    )
    samples = {}
    for min_cases in sorted(
        {spec["min_cases"] for spec in HOMOGENEITY_MODELS.values()}
    ):
        flags = multi_county_flags(
            df_spread,
            adjacency_matrix_spread_pre,
            adjacency_matrix_spread,
            min_cases=min_cases,
        ).filter(pl.col("year") == HOMOGENEITY_YEAR)
        rows = (
            flags.join(cases, on="fips", how="left")
            .join(school, on=["fips", "year"], how="inner")
            .with_columns(
                multi=pl.col("multi_county"),
                cases=pl.col("cases_outbreak").cast(pl.Float64),
                log_cases=pl.col("cases_outbreak").log1p(),
                rucc=pl.col("fips")
                .map_elements(
                    lambda fips: rucc.get(fips), return_dtype=pl.Int64
                )
                .cast(pl.Float64),
                state=pl.col("fips").str.slice(0, 2),
            )
            .sort("fips")
        )
        assert not rows["fips"].is_duplicated().any()
        samples[min_cases] = rows
        print(
            f"\nPanel d, {HOMOGENEITY_YEAR}, {min_cases}+ outbreak cases: "
            f"{flags.height} counties, {rows.height} with a school SD "
            f"({int(rows['multi'].sum())} multi-county, "
            f"{int((~rows['multi']).sum())} single-county); "
            f"{int((rows['qc_gap_ok'] != True).sum())} of them fail the "  # noqa: E712
            "school-data QC or have no QC result"
        )
    # the weighted SDs: where the county has the weights, else the unweighted
    # SD (sd, until the columns are rebuilt here, is the unweighted SD)
    weights = read_ccd_county_sds(
        sorted({fips for rows in samples.values() for fips in rows["fips"]})
    )
    for min_cases, rows in samples.items():
        samples[min_cases] = (
            rows.join(weights, on="fips", how="left")
            .with_columns(
                sd_unweighted=pl.col("sd"),
                weighted=pl.col("sd_ccd").is_not_null(),
                sd_hierarchy_weighted=pl.col("sd_hierarchy").is_not_null(),
            )
            .with_columns(
                sd=pl.coalesce("sd_ccd", "sd_unweighted"),
                sd_hierarchy=pl.coalesce("sd_hierarchy", "sd_unweighted"),
            )
            .with_columns(cv=pl.col("sd") / pl.col("school_mean") * 100)
        )
    return samples


def homogeneity_rows(samples, model):
    """
    The counties of one model of HOMOGENEITY_MODELS, sorted by fips: those
    with its number of outbreak cases, a value for every column the model
    uses (no school-data QC filter).
    """
    spec = HOMOGENEITY_MODELS[model]
    rows = samples[spec["min_cases"]]
    # the SD of the weighting (the test, the plot and the subsets of the
    # figure all read the column sd), and whether each county is weighted
    if spec["weighting"] == "hierarchy":
        rows = rows.with_columns(
            sd=pl.col("sd_hierarchy"), weighted=pl.col("sd_hierarchy_weighted")
        )
    elif spec["weighting"] == "unweighted":
        rows = rows.with_columns(
            sd=pl.col("sd_unweighted"), weighted=pl.lit(False)
        )
    return rows.drop_nulls([spec["x"], spec["size"], *spec["controls"]]).sort(
        "fips"
    )


def homogeneity_test(
    samples,
    adjacency_data,
    model,
    n_permutations=HOMOGENEITY_PERMUTATIONS,
):
    """
    The test of the SD term (the CV term for the CV model) of one model of
    HOMOGENEITY_MODELS, by the spatially constrained permutation test of
    outbreak_size.msr_logistic, with the p value of a Wald test clustered by
    outbreak beside it. Each model is tested once per run.

    The odds ratio is per +1 percentage point of the term at an average-size
    county. Its interval is the coefficient +- c x its standard error, where c
    is the critical value of |z| in the permutation null of the test (the
    permutation null of the Wald z, not 1.96): the interval excludes 1 exactly
    when the permutation p is 0.05 or less. It is a permutation-calibrated
    Wald interval, not an inversion of the test over the values of the
    coefficient.
    Returns n, n_multi, coef, se, odds ratio and interval, p_perm,
    p_clustered, McFadden's R2 and whether the fit separates.
    """
    key = (model, n_permutations)
    if key in _HOMOGENEITY_TESTS:
        return _HOMOGENEITY_TESTS[key]
    spec = HOMOGENEITY_MODELS[model]
    rows = homogeneity_rows(samples, model)
    x, size = spec["x"], spec["size"]
    x_values = rows[x].to_numpy()
    size_values = rows[size].to_numpy().astype(float)
    assert x_values.std() > 0
    x_centered = x_values - x_values.mean()
    size_centered = size_values - size_values.mean()
    columns = [x_centered, size_centered]
    if spec["interaction"]:
        columns.append(x_centered * size_centered)
    columns += [
        rows[column].to_numpy().astype(float) for column in spec["controls"]
    ]
    if spec["state_fe"]:
        states = (
            rows["state"]
            .value_counts()
            .sort(["count", "state"], descending=[True, False])["state"]
            .to_list()[1:]
        )
        columns += [
            (rows["state"] == state).cast(pl.Float64).to_numpy()
            for state in states
        ]
    # a column that does not vary cannot be fitted (the SD is the first)
    columns = [column for column in columns if column.std() > 0]
    fips = rows["fips"].to_list()
    test = logistic_msr_test(
        np.column_stack(columns),
        rows["multi"].cast(pl.Float64).to_numpy(),
        np.arange(len(fips)),
        np.asarray(adjacency.assemble_adjacency_matrix(adjacency_data, fips))
        > 0,
        tested=0,
        n_permutations=n_permutations,
        seed=params["seed"],
        clusters=rows["group"].to_numpy(),
    )
    coef, se = float(test["coef"][1]), float(test["se"][1])
    # p <= 0.05 exactly when fewer than (n_permutations + 1) / 20 null values
    # of |z| reach the observed |z|, that is, when the observed |z| is above
    # the k-th largest null value
    k = (n_permutations + 1) // 20 - 1
    if k < 0:
        raise ValueError(
            "at least 19 permutations are needed for the 5% interval"
        )
    critical = float(np.sort(np.abs(test["z_null"]))[::-1][k])
    _HOMOGENEITY_TESTS[key] = dict(
        n=rows.height,
        n_multi=int(rows["multi"].sum()),
        coef=coef,
        se=se,
        odds_ratio=float(np.exp(coef)),
        odds_low=float(np.exp(coef - critical * se)),
        odds_high=float(np.exp(coef + critical * se)),
        p_perm=test["p_value"],
        p_clustered=test["p_clustered"],
        mcfadden_r2=test["mcfadden_r2"],
        separated=test["separated"],
    )
    return _HOMOGENEITY_TESTS[key]


def plot_multi_county_homogeneity(
    ax,
    rows,
    p=None,
    ymax=None,
    counts=False,
    notes=True,
    bandwidth=HOMOGENEITY_VIOLIN_BANDWIDTH,
):
    """
    The SD of the school MMR rates of the counties of single-county (grey) and
    multi-county (blue) outbreaks (violins with the median and IQR, and the
    mean as a star). With `p` (the permutation p of the SD term), a bracket
    with its stars. `counts` puts the numbers of counties under the labels.
    `bandwidth` (pp of SD) is the fixed width of the Gaussian kernel of the
    violins; None is Scott's rule, which scales with the spread of the values,
    so that a lone high value drags the density out to it.
    """
    c1 = "#8b8c89"  # single-county outbreaks
    c2 = "#274c77"  # multi-county outbreaks
    ymax = float(rows["sd"].max()) * 1.12 if ymax is None else ymax
    for position, multi in ((0, False), (1, True)):
        values = rows.filter(pl.col("multi") == multi)["sd"].to_numpy()
        color = c2 if multi else c1
        if len(values) >= 3:
            violinparts = ax.violinplot(
                [values],
                positions=[position],
                widths=0.5,
                showextrema=False,
                # a number is the kernel's SD as a multiple of the data's SD
                bw_method=None
                if bandwidth is None
                else bandwidth / values.std(ddof=1),
            )
            for pc in violinparts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.6)
            q25, median, q75 = np.percentile(values, [25, 50, 75])
            ax.errorbar(
                x=[position],
                y=[median],
                yerr=[[median - q25], [q75 - median]],
                fmt="o",
                capsize=10,
                capthick=2,
                color=color,
            )
        if len(values):
            ax.scatter(
                [position],
                [values.mean()],
                marker="*",
                s=150,
                color=color,
                zorder=3,
            )
    if p is not None:
        sig_text = (
            "***"
            if p < 0.001
            else "**"
            if p < 0.01
            else "*"
            if p < 0.05
            else "ns"
        )
        bar_height = ymax * 0.97
        ax.plot([0, 1], [bar_height, bar_height], "k-", linewidth=1.5)
        for position in (0, 1):
            ax.plot(
                [position, position],
                [bar_height, bar_height - ymax * 0.02],
                "k-",
                linewidth=1.5,
            )
        ax.text(
            0.5,
            bar_height + 0.2,
            sig_text,
            ha="center",
            va="bottom",
            fontsize=22,
        )
    n_single = rows.filter(~pl.col("multi")).height
    n_multi = rows.filter(pl.col("multi")).height
    ax.set_ylabel(
        "Standard deviation of within-county\nMMR vaccine coverage (pp)"
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [
            "Single-county\noutbreak"
            + (f"\n(n = {n_single})" if counts else ""),
            "Multi-county\noutbreak"
            + (f"\n(n = {n_multi})" if counts else ""),
        ]
    )
    ax.set_ylim(0, ymax)
    ax.set_yticks(range(0, int(ymax) + 1, 10))
    if notes:
        ax.text(
            1.5,
            ymax * 0.85,
            "↑ Heterogeneous\ncoverage",
            ha="center",
            va="bottom",
            fontsize=22,
            style="italic",
            color="k",
        )
        ax.text(
            1.5,
            ymax * 0.02,
            "Homogeneous\ncoverage ↓",
            ha="center",
            va="bottom",
            fontsize=22,
            style="italic",
            color="k",
        )
    return ax


def plot_sd_by_outbreak_type(samples, adjacency_data, ax):
    """
    Panel d: the SD of the school MMR rates of the counties of single-county
    and multi-county outbreaks in 2025 (HOMOGENEITY_MIN_CASES or more outbreak
    cases; see plot_multi_county_homogeneity), for the counties of the model
    HOMOGENEITY_PANEL_MODEL (for now the SD weighted by CCD kindergarten
    counts where a county has them, the unweighted SD of the other counties).
    The bracket is the permutation test of the SD term of that model.
    """
    model = HOMOGENEITY_PANEL_MODEL
    test = homogeneity_test(samples, adjacency_data, model)
    rows = homogeneity_rows(samples, model)
    print(
        f"Panel d, SD term of the model '{model}' ({test['n']} counties, "
        f"{test['n_multi']} multi-county, {int(rows['weighted'].sum())} of "
        f"them with a weighted SD): odds ratio per +1 pp "
        f"{test['odds_ratio']:.3f} ({test['odds_low']:.3f}, "
        f"{test['odds_high']:.3f}), permutation p {test['p_perm']:.4f} "
        f"({HOMOGENEITY_PERMUTATIONS} permutations), clustered p "
        f"{test['p_clustered']:.4f}"
    )
    return plot_multi_county_homogeneity(ax, rows, test["p_perm"])


def create_homogeneity_figure(samples, adjacency_data):
    """
    The supplemental figure of panel d: the panels for the rural and for the
    urban counties (rural/urban code 4-9 and 1-3) and for the counties with 2
    or more, 3 or more and 5 or more schools and districts, which are subsets
    of the counties of panel d with its SD (HOMOGENEITY_PANEL_MODEL: weighted
    by CCD kindergarten counts where a county has them), and the panel of the
    SD weighted school by school (HOMOGENEITY_HIERARCHY_MODEL): the CCD
    kindergarten count of each school, else our own enrollment, and a county
    with a school that has neither is unweighted. No panel has a bracket or
    stars (the tests are in the table). All the panels have the vertical axis
    of the counties of panel d.
    """
    rows = homogeneity_rows(samples, HOMOGENEITY_PANEL_MODEL)
    hierarchy_rows = homogeneity_rows(samples, HOMOGENEITY_HIERARCHY_MODEL)
    subsets = [
        (
            "Rural counties (rural/urban code 4-9)",
            rows.filter(pl.col("rucc") >= 4),
        ),
        (
            "Urban counties (rural/urban code 1-3)",
            rows.filter(pl.col("rucc") <= 3),
        ),
        (
            "SD weighted school by school: CCD kindergarten counts,\n"
            "then our own enrollment, otherwise unweighted",
            hierarchy_rows,
        ),
        ("2 or more schools", rows.filter(pl.col("n_schools") >= 2)),
        ("3 or more schools", rows.filter(pl.col("n_schools") >= 3)),
        ("5 or more schools", rows.filter(pl.col("n_schools") >= 5)),
    ]
    fig, ax = setup_plot(2, 3, figsize=(33, 21))
    ax = ax.flatten()  # type: ignore
    ymax = float(max(rows["sd"].max(), hierarchy_rows["sd"].max())) * 1.12
    for k, (title, subset) in enumerate(subsets):
        print(
            f"Supplemental panel d, {title.replace(chr(10), ' ')}: "
            f"{subset.height} counties "
            f"({int(subset['multi'].sum())} multi-county)"
        )
        plot_multi_county_homogeneity(
            ax[k],
            subset,
            ymax=ymax,
            counts=True,
            notes=(k % 3 == 2),  # the notes beside the right panels
        )
        ax[k].set_title(title)
        if k % 3:
            ax[k].set_ylabel("")
            ax[k].tick_params(labelleft=False)
    fig.tight_layout()
    return fig


def create_homogeneity_table(
    samples, adjacency_data, n_permutations=HOMOGENEITY_PERMUTATIONS
):
    """
    The supplemental table of panel d: for each model of HOMOGENEITY_MODELS,
    the odds ratio of a multi-county outbreak per percentage point of SD (of
    CV for the CV model) at an average-size county with its 95% interval (see
    homogeneity_test), the permutation and the clustered p, and McFadden's R2.
    """
    odds_column = (
        "Odds ratio of a multi-county outbreak per percentage point of SD "
        "(95% CI)"
    )
    table = []
    for model in HOMOGENEITY_MODELS:
        test = homogeneity_test(samples, adjacency_data, model, n_permutations)
        # the interval agrees with the permutation p
        assert (test["odds_low"] > 1 or test["odds_high"] < 1) == (
            test["p_perm"] <= 0.05
        ), model
        print(
            f"{model}: {test['n']} counties ({test['n_multi']} multi-county), "
            f"odds ratio {test['odds_ratio']:.3f}, permutation p "
            f"{test['p_perm']:.4f}, separated {test['separated']}"
        )
        table.append(
            {
                "Model specification": model,
                odds_column: (
                    f"{test['odds_ratio']:.2f} ({test['odds_low']:.2f}, "
                    f"{test['odds_high']:.2f})"
                ),
                "Permutation p": f"{test['p_perm']:.4f}",
                "Clustered p": f"{test['p_clustered']:.4f}",
                "McFadden R2": f"{test['mcfadden_r2']:.3f}",
            }
        )
    return pd.DataFrame(table)


def plot_county_level_variation(
    df, posterior_samples_post, adjacency_matrix_post, fig, ax, school=None
):
    """
    Plot within-county vaccination coverage variability for large outbreaks,
    comparing independent vs correlated multi-county outbreaks. `school` is
    the school data (read_old_school_data by default). All the county-years
    are used, whether or not they pass the school-data QC; the SD is the
    unweighted one (the CCD kindergarten counts are of 2024-25 only).
    """
    if school is None:
        school = read_old_school_data()
    if school is None:
        return None

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
        school_post.join(
            cases_post, on=["fips", "year"], how="inner", maintain_order="left"
        )
        .join(significant_car_df, on="fips", how="left", maintain_order="left")
        .join(bayesian_p_df, on="fips", how="left", maintain_order="left")
        .with_columns(
            significant_car=pl.col("significant_car").fill_null(False),
        )
    )

    print("\nSchool data after joining:")
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

    # no QC filter: the county-years whose school mean is far from the county
    # MMR (qc_gap_ok) are shown

    print(
        "  Rows with 10 or more cases, by outbreak year:"
        f"\n{county_df.filter(pl.col('outbreak_category') == f'≥{cutoff}').group_by('year').len().sort('year')}"
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
        multi_county_df, on=["fips", "year"], how="left", maintain_order="left"
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
    county_df_reg["county_idx"] = county_df_reg["fips"].map(fips_to_county_idx)  # type: ignore

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

    county_df_reg["county_idx"] = county_df_reg["fips"].map(fips_to_new_idx)  # type: ignore

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
    bucket_fips = subset["fips"].unique()  # type: ignore
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
    X_bucket = subset[["std_coverage_pct", "cases_outbreak"]].values  # type: ignore

    print(X_bucket.shape)

    y_bucket = subset["bayesian_p"].values  # type: ignore

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
        tested=(1,),
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

    The school data are the new ones (read_new_school_data, the unweighted
    mean of the school rates of a county and outbreak year), with no QC
    filter: this panel is the comparison that the QC flag (qc_gap_ok) is
    built on.
    """
    school = read_new_school_data()
    if school is None:
        return None

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
    merged = school_post.join(
        df_post, on=["fips", "year"], how="inner", maintain_order="left"
    )

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
    ax.plot(x_line, y_line, "r-", linewidth=2, label="OLS fit")

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

    The school data are the new ones (read_new_school_data: the unweighted mean
    and SD of the school rates of a county and outbreak year; a county of one
    school has an SD of 0). No QC filter: counties that fail qc_gap_ok are
    shown.
    """
    school = read_new_school_data()
    if school is None:
        return None

    # Filter to post-2021 and join with analysis dataset
    school_post = school.filter(pl.col("year") >= 2021)
    df_post = df.filter(pl.col("year") >= 2021).select(["fips", "year"])

    # Join to keep only counties in our analysis
    merged = school_post.join(
        df_post, on=["fips", "year"], how="inner", maintain_order="left"
    )

    x = merged["mean_coverage"].to_numpy() * 100  # → percent
    y = merged["std_coverage"].to_numpy() * 100  # → percentage points

    ax.scatter(x, y, alpha=0.4, s=15, color="black")

    ax.set_xlabel("Mean within-county MMR coverage (%)")
    ax.set_ylabel("SD of within-county MMR coverage (pp)")

    return fig, ax


def period_matrices(df, adjacency_data, regions, steps=1):
    """
    The neighbor matrices (place_adjacency_matrix) of the places of df in
    2011-2020 and in 2021-2025, in the order of their first appearance, as
    plot_increasing_spread_over_time takes them.
    """
    return tuple(
        place_adjacency_matrix(
            adjacency_data,
            df.filter(in_period)["fips"].unique(maintain_order=True).to_list(),
            regions,
            steps,
        )
        for in_period in (pl.col("year") < 2021, pl.col("year") >= 2021)
    )


def read_rucc(folder):
    """
    County fips -> Rural-Urban Continuum Code (1-9; 1-3 metro, 4-9 nonmetro).
    U.S. Department of Agriculture, Economic Research Service (January 2024).
    Rural-Urban Continuum Codes, https://www.ers.usda.gov/data-products/rural-urban-continuum-codes
    The 2023 codes, except in Connecticut, which they give by planning region:
    its old counties (the ones of the case data) take the 2013 codes. A county
    that has no 2023 code takes its 2013 code.
    """
    rucc_2023 = pl.read_csv(
        os.path.join(folder, "rucc_2023.csv"),
        schema_overrides={"FIPS": pl.String},
        encoding="utf8-lossy",  # the county names of Puerto Rico are not utf-8
    ).filter(
        (pl.col("Attribute") == "RUCC_2023")
        & ~pl.col("FIPS").str.starts_with("09")
    )
    rucc_2013 = (
        pl.read_excel(os.path.join(folder, "rucc_2013.xls"))
        .select(pl.col("FIPS").cast(pl.String).str.zfill(5), "RUCC_2013")
        .drop_nulls()
    )
    codes = {fips: int(code) for fips, code in rucc_2013.iter_rows()}
    codes.update(
        {
            fips: int(code)
            for fips, code in zip(rucc_2023["FIPS"], rucc_2023["Value"])
        }
    )
    return codes


def place_type(place, regions, rucc):
    """
    "metro" or "nonmetro" for a place: a county fips, or a region of `regions`
    (several counties). A region takes the type of the majority of its counties,
    one vote each (the case data have one row for a whole region, so its cases
    cannot be split by county). A tie, or a county without a code, is an error.
    """
    metro = [rucc[county] <= 3 for county in regions.get(place, [place])]
    assert 2 * sum(metro) != len(metro), (
        f"{place}: as many metro as nonmetro counties"
    )
    return "metro" if 2 * sum(metro) > len(metro) else "nonmetro"


def multi_county_cases_by_year_and_type(df, adjacency_data, regions, rucc):
    """
    By year and type of place (place_type), the cases in outbreak places (3 or
    more cases) that have an outbreak neighbor, and the cases in all outbreak
    places. Neighbors are found among the places of the same period (2011-2020
    or 2021-2025), as in plot_increasing_spread_over_time, and of any type, so
    an outbreak that spans both types adds its metro places' cases to the metro
    rows and its nonmetro places' cases to the nonmetro rows.
    """
    rows = []
    for before_2021 in (True, False):
        period = df.filter((pl.col("year") < 2021) == before_2021)
        places = period["fips"].unique(maintain_order=True).to_list()
        matrix = place_adjacency_matrix(adjacency_data, places, regions)
        position = {place: i for i, place in enumerate(places)}
        for year in sorted(period["year"].unique()):
            outbreaks = period.filter(
                (pl.col("year") == year) & (pl.col("cases_outbreak") >= 3)
            )
            at = [position[place] for place in outbreaks["fips"]]
            cases = outbreaks["cases_outbreak"].to_numpy()
            has_neighbor = matrix[np.ix_(at, at)].sum(axis=1) > 0
            types = np.array(
                [
                    place_type(place, regions, rucc)
                    for place in outbreaks["fips"]
                ]
            )
            for kind in ("metro", "nonmetro"):
                of_kind = types == kind
                rows.append(
                    dict(
                        year=year,
                        type=kind,
                        multi_county_cases=int(
                            cases[of_kind & has_neighbor].sum()
                        ),
                        outbreak_cases=int(cases[of_kind].sum()),
                    )
                )
    return pl.DataFrame(rows)


def multi_county_outbreaks_by_year(df, adjacency_data, regions):
    """
    By year, the number of multi-county outbreaks and the cases in them. An
    outbreak is a group of outbreak places (3 or more cases) joined by borders,
    directly or through other places of the group; it is multi-county when the
    group has two or more places. Neighbors are found among the places of the
    same period (2011-2020 or 2021-2025), as in plot_increasing_spread_over_time.
    """
    rows = []
    for before_2021 in (True, False):
        period = df.filter((pl.col("year") < 2021) == before_2021)
        places = period["fips"].unique(maintain_order=True).to_list()
        matrix = place_adjacency_matrix(adjacency_data, places, regions)
        position = {place: i for i, place in enumerate(places)}
        for year in sorted(period["year"].unique()):
            outbreaks = period.filter(
                (pl.col("year") == year) & (pl.col("cases_outbreak") >= 3)
            )
            cases = outbreaks["cases_outbreak"].to_numpy()
            n_outbreaks, n_cases = 0, 0
            if len(cases):
                at = [position[place] for place in outbreaks["fips"]]
                _, group = connected_components(
                    matrix[np.ix_(at, at)], directed=False
                )
                sizes = np.bincount(group)
                multi = sizes[group] >= 2
                n_outbreaks, n_cases = (
                    int((sizes >= 2).sum()),
                    int(cases[multi].sum()),
                )
            rows.append(
                dict(
                    year=year,
                    multi_county_outbreaks=n_outbreaks,
                    multi_county_cases=n_cases,
                )
            )
    return pl.DataFrame(rows)


def beta_share_interval(clustered, total, pooled, n_samples=10000):
    """
    The median and 95% interval (percent) of a share, as
    plot_increasing_spread_over_time computes them: each year has a
    Beta(clustered + 1, total - clustered + 1) distribution, and the share of
    the years is the mean of one draw from each (pooled=False) or the Beta
    of the years' sums (pooled=True).
    Returns (median, lower, upper).
    """
    clustered, total = np.asarray(clustered), np.asarray(total)
    if pooled:
        clustered, total = (
            clustered.sum(keepdims=True),
            total.sum(keepdims=True),
        )
    draws = stats.beta.rvs(
        clustered + 1,
        total - clustered + 1,
        size=(n_samples, len(total)),
        random_state=rng,
    ).mean(axis=1)  # type: ignore
    return tuple(np.percentile(100 * draws, [50, 2.5, 97.5]))


def share_without_clusters(
    df, adjacency_data, regions, year, seeds=(), names=None
):
    """
    The share (percent) of the outbreak cases of a year that are in multi-county
    outbreaks, after leaving out whole clusters, as (median, lower, upper) of
    the Beta interval of plot_increasing_spread_over_time. A cluster is a group
    of outbreak places (3 or more cases) joined by borders, so leaving one out
    changes no other cluster. The clusters left out are those holding one of
    `seeds` (a county fips or the name of a region). `names` (fips -> name) is
    used to print what is left out.
    """
    names = {} if names is None else names
    places, cases, cluster, matrix = outbreak_place_clusters(
        df, adjacency_data, regions, year
    )
    left_out = clusters_holding(places, cluster, seeds)
    for c in sorted(left_out):
        members = [
            f"{names.get(p, p)} {int(n)}"
            for p, n, k in zip(places, cases, cluster)
            if k == c
        ]
        print(f"    left out: {'; '.join(members)}")
    keep = ~np.isin(cluster, list(left_out))
    # the cases left out as a percent of all the outbreak cases of the year
    # (of every place, whatever its number of cases)
    cases_left_out = int(cases[~keep].sum())
    cases_of_year = int(
        df.filter(pl.col("year") == year)["cases_outbreak"].sum()
    )
    print(
        f"    cases left out: {cases_left_out} of the {cases_of_year} outbreak cases of {year} "
        f"({100 * cases_left_out / cases_of_year:.1f}%)"
    )
    has_neighbor = matrix.sum(axis=1) > 0
    return beta_share_interval(
        [cases[keep & has_neighbor].sum()], [cases[keep].sum()], pooled=True
    )


if __name__ == "__main__":
    df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
    )

    fig, ax = setup_plot(2, 2)  # type: ignore

    # read predictive samples
    predictive_file = "predictive_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_[2021, 2022, 2023, 2024, 2025].npz"
    predictive_samples = load_samples(
        os.path.join(output_dir, predictive_file)
    )
    # the concentration of the negative binomial (if the model has one) is in
    # the posterior draws of the same fit
    concentration = load_samples(
        os.path.join(
            output_dir, predictive_file.replace("predictive_", "samples_", 1)
        )
    ).get("nb_concentration")

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
    df_waic = df.filter(pl.col("year").is_in([2021, 2022, 2023, 2024, 2025]))
    waic_statistical, p_waic = waic.waic_posterior(
        df_waic["cases_outbreak"], predictive_samples["mu"], concentration
    )
    waic_naive = waic.waic_point_estimate(
        df_waic["cases_outbreak"], naive_model["mu"], naive_model["k"]
    )
    # lower is better; the naive model has no posterior, so its WAIC is the
    # AIC-equivalent (log-likelihood penalized by its k fitted parameters)
    print(
        f"WAIC 2021-2025: statistical {waic_statistical:.1f} "
        f"(effective parameters {p_waic:.1f}), naive mechanistic "
        f"{waic_naive:.1f} (k = {naive_model['k']})"
    )
    # the statistical points and bars are the expected outbreak size (mu): the
    # median and 95% interval of its posterior (the draws of the count itself,
    # "cases", add the negative binomial noise). The mechanistic bars are as
    # they were: the 95% Poisson interval of the count around its mu
    plot_predictions(
        df_waic,
        predictive_samples["mu"],
        naive_model,
        fig,
        ax[0, 0],  # type: ignore
        waic_values=(waic_statistical, waic_naive),
    )
    ax[0, 0].set_ylabel("Model-expected outbreak size (N)")  # type: ignore

    # while we're here, make a plot of the performance for the R01, for the
    # state_fixed_effect=True, pool=False variant we've fit at CAR=False
    r01_df = df.filter(pl.col("year").is_in([2021, 2022, 2023, 2024, 2025]))
    for state_fixed_effect_r01, pool_r01 in [(True, False)]:
        # 02_fit_hierarchical_poisson.py currently fits only state_fixed_effect=True, pool=False at CAR=False
        if not os.path.exists(
            os.path.join(
                output_dir,
                f"predictive_partial_pooled_poisson_state_fixed_effect={state_fixed_effect_r01}_pool={pool_r01}_five_year=False_CAR=False_All covariates_[2021, 2022, 2023, 2024, 2025].npz",
            )
        ):
            print(
                f"skipped: no CAR=False fit for state_fixed_effect={state_fixed_effect_r01}, pool={pool_r01}."
            )
            continue
        predictive_samples_r01 = load_samples(
            os.path.join(
                output_dir,
                f"predictive_partial_pooled_poisson_state_fixed_effect={state_fixed_effect_r01}_pool={pool_r01}_five_year=False_CAR=False_All covariates_[2021, 2022, 2023, 2024, 2025].npz",
            )
        )

        fig_r01, ax_r01 = setup_plot(1, 1, figsize=(10, 7.5))  # type: ignore
        plot_predictions(
            r01_df,
            predictive_samples_r01["cases"],
            naive_model,
            fig_r01,
            ax_r01,
            label_naive="Naive mechanistic model (vaccine coverage only)",
            label_statistical="Statistical model",
            show_r2=True,
            # a data-coordinate x (r2_x_frac) is compressed almost flat near
            # 0 on this symlog axis, so use an axes-fraction position
            # instead to actually move the text
            r2_x_axes_frac=0.05,
            # same colors as the mechanistic (MMR coverage) and statistical
            # bars in figure_r01_validation_comparison.py
            color_naive=MECHANISTIC_COLOR,
            color_statistical=STATISTICAL_COLOR,
        )
        fig_r01.savefig(
            os.path.join(
                fig_dir,
                f"figure_county_level_predictions_r01_state_fixed_effect={state_fixed_effect_r01}_pool={pool_r01}.png",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig_r01)

    posterior_samples = load_samples(
        os.path.join(
            output_dir,
            "samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_[2021, 2022, 2023, 2024, 2025].npz",
        )
    )

    posterior_samples_pre = load_samples(
        os.path.join(
            output_dir,
            "samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_[2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020].npz",
        )
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

    # panel c uses every county with outbreak cases, not only those with
    # vaccine data (which are the rows of df)
    raw_merged = pl.read_csv(
        os.path.join(data_dir, "raw_merged_outbreaks_vaccinations.csv"),
        schema_overrides={"fips": pl.String},
    )
    df_spread = outbreak_cases_all_counties(raw_merged)
    adjacency_matrix_spread = adjacency.assemble_adjacency_matrix(
        adjacency_matrix,
        df_spread.filter(pl.col("year") >= 2021)["fips"]
        .unique(maintain_order=True)
        .to_list(),
    )
    adjacency_matrix_spread_pre = adjacency.assemble_adjacency_matrix(
        adjacency_matrix,
        df_spread.filter(pl.col("year") < 2021)["fips"]
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
        df_spread,
        adjacency_matrix_spread_pre,
        adjacency_matrix_spread,
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
    print_school_data_coverage(
        df, df_spread, adjacency_matrix_spread_pre, adjacency_matrix_spread
    )
    # panel d: the school MMR rates of single-county and multi-county
    # outbreaks in 2025 (the data of its supplemental figure and table too)
    homogeneity = homogeneity_samples(
        df, df_spread, adjacency_matrix_spread_pre, adjacency_matrix_spread
    )
    if homogeneity is not None:
        plot_sd_by_outbreak_type(
            homogeneity,
            adjacency_matrix,
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

    # Supplemental figure: other ways of handling the places that are several
    # counties in the share of cases in multi-county outbreaks (panel c)
    regions = region_members()
    df_spread_regions = outbreak_cases_all_counties(raw_merged, regions)
    adjacency_regions_pre, adjacency_regions_post = period_matrices(
        df_spread_regions, adjacency_matrix, regions
    )

    fig_options, ax_options = setup_plot(2, 3, figsize=(42, 24))
    for ax_k, letter in zip(ax_options.flatten(), "abcdef"):  # type: ignore
        ax_k.text(
            -0.25,
            1.08,
            letter,
            transform=ax_k.transAxes,
            fontsize=24,
            fontweight="bold",
            va="top",
            ha="right",
        )
    grey, blue = "#8b8c89", "#274c77"
    year_bins = [(2011, 2015), (2016, 2020), (2021, 2024), (2025, 2025)]
    # a: panel c as it is calculated (average of the yearly shares of the
    # years of each bin), with the regions left out (as in panel c) and in
    for k, (df_k, matrix_pre, matrix_post) in enumerate(
        [
            (df_spread, adjacency_matrix_spread_pre, adjacency_matrix_spread),
            (df_spread_regions, adjacency_regions_pre, adjacency_regions_post),
        ]
    ):
        plot_increasing_spread_over_time(
            df_k,
            matrix_pre,
            matrix_post,
            posterior_samples_pre,
            posterior_samples,
            fig_options,
            ax_options[0, 0],  # type: ignore
            year_bins=year_bins,
            n_series=2,
            series_index=k,
        )
    bars = ax_options[0, 0].patches  # type: ignore
    for bar in bars[len(bars) // 2 :]:
        bar.set_color(blue)
    # one tick label for each pair of bars
    labels = [tick.get_text() for tick in ax_options[0, 0].get_xticklabels()]  # type: ignore
    ax_options[0, 0].set_xticks(  # type: ignore
        [
            (first.get_x() + second.get_x() + second.get_width()) / 2
            for first, second in zip(bars, bars[len(bars) // 2 :])
        ]
    )
    ax_options[0, 0].set_xticklabels(labels, rotation=-45, ha="left")  # type: ignore
    # the second call's bars are taller than the first call's limits
    ax_options[0, 0].set_ylim(0, 100)  # type: ignore
    ax_options[0, 0].set_yticks(range(0, 101, 20))  # type: ignore
    ax_options[0, 0].legend(  # type: ignore
        handles=[
            Patch(
                color=grey, alpha=0.8, label="Excludes multi-county regions"
            ),
            Patch(
                color=blue, alpha=0.8, label="Includes multi-county regions"
            ),
        ],
        loc="upper left",
        edgecolor="black",
    )
    # f: panel a's way of aggregating the years of a bin changed to pooling
    # their cases (a year counts by its cases), with the regions included
    plot_increasing_spread_over_time(
        df_spread_regions,
        adjacency_regions_pre,
        adjacency_regions_post,
        posterior_samples_pre,
        posterior_samples,
        fig_options,
        ax_options[1, 2],  # type: ignore
        year_bins=year_bins,
        aggregation_method="sum",
    )
    for bar in ax_options[1, 2].patches:  # type: ignore
        bar.set_color(blue)
    ax_options[1, 2].set_ylim(0, 100)  # type: ignore
    ax_options[1, 2].set_yticks(range(0, 101, 20))  # type: ignore
    # d: the share of the outbreak cases (places of 3+ cases) that are in
    # multi-county outbreaks, as panel a calculates it (the mean of the yearly
    # shares of the years of a bin, each year counting equally; a year without
    # outbreak cases of the type is left out), regions included, for metro and
    # nonmetro places: by the Rural-Urban Continuum Codes of the county (1-3
    # metro, 4-9 nonmetro; a region of several counties by the majority of its
    # counties, place_type). A place is in a multi-county outbreak if it borders
    # an outbreak place of either type, so an outbreak that spans both types
    # adds its metro places' cases to the metro bar and its nonmetro places'
    # cases to the nonmetro bar.
    rucc = read_rucc(os.path.join(data_dir, "rural_urban_continuum_codes"))
    by_type = multi_county_cases_by_year_and_type(
        df_spread_regions, adjacency_matrix, regions, rucc
    )
    types = [
        ("metro", "Metro", "#274c77"),
        ("nonmetro", "Nonmetro", "#eb6834"),
    ]
    for k, (kind, label, color) in enumerate(types):
        for j, (first, last) in enumerate(year_bins):
            rows = by_type.filter(
                (pl.col("type") == kind)
                & pl.col("year").is_between(first, last)
                & (pl.col("outbreak_cases") > 0)
            )
            if rows.height == 0:
                continue
            median, lower_k, upper_k = beta_share_interval(
                rows["multi_county_cases"],
                rows["outbreak_cases"],
                pooled=False,
            )
            print(
                f"  {kind}, {first}-{last}: {median:.1f} ({lower_k:.1f}, {upper_k:.1f}), "
                f"{rows['multi_county_cases'].sum()} of {rows['outbreak_cases'].sum()} cases"
            )
            x = j + (k - 0.5) * 0.4
            ax_options[1, 0].bar(  # type: ignore
                x,
                median,
                width=0.4,
                color=color,
                alpha=0.8,
                yerr=[[median - lower_k], [upper_k - median]],
                capsize=5,
                error_kw={"elinewidth": 1.5, "ecolor": "black", "alpha": 0.8},
            )
    ax_options[1, 0].set_xticks(range(len(year_bins)))  # type: ignore
    ax_options[1, 0].set_xticklabels(  # type: ignore
        [
            f"{first}-{last}" if first != last else str(first)
            for first, last in year_bins
        ],
        rotation=-45,
        ha="left",
    )
    ax_options[1, 0].set_ylabel("Cases in multi-county outbreaks (%)")  # type: ignore
    ax_options[1, 0].set_ylim(0, 100)  # type: ignore
    ax_options[1, 0].set_yticks(range(0, 101, 20))  # type: ignore
    ax_options[1, 0].legend(  # type: ignore
        handles=[
            Patch(color=color, alpha=0.8, label=label)
            for _, label, color in types
        ],
        loc="upper left",
        edgecolor="black",
    )
    # d: 2025 without whole clusters of outbreaks (regions included); the
    # clusters are left out of both the numerator and the denominator
    names = dict(
        zip(
            raw_merged["fips"],
            raw_merged["county"] + ", " + raw_merged["state"],
        )
    )
    # The seeds of each bar are ALL the counties that official sources (an MMWR,
    # with its PMID, or a report of a health department) name as part of the
    # outbreak, whatever their number of cases. Only the seeds that are outbreak
    # places (3+ cases that year) are kept, in the loop below, and
    # share_without_clusters leaves out the whole cluster of outbreak places that
    # holds one of them.
    seeds = cluster_seeds_2025(adjacency_matrix, regions)
    exclusions = [("All", {})] + [
        (f"Excluding {name}", dict(seeds=seeds[name]))
        for name in CLUSTER_NAMES
    ]
    # the seeds that are outbreak places: 3 or more cases in 2025
    outbreak_places = set(
        df_spread_regions.filter(
            (pl.col("year") == 2025) & (pl.col("cases_outbreak") >= 3)
        )["fips"]
    )
    shares = []
    for exclusion, kwargs in exclusions:
        print("2025, " + exclusion)
        if "seeds" in kwargs:
            kwargs = dict(
                seeds=[
                    seed for seed in kwargs["seeds"] if seed in outbreak_places
                ]
            )
            print(
                f"  seeds with 3+ cases ({len(kwargs['seeds'])}):",
                [names.get(seed, seed) for seed in kwargs["seeds"]],
            )
        shares.append(
            share_without_clusters(
                df_spread_regions,
                adjacency_matrix,
                regions,
                2025,
                names=names,
                **kwargs,
            )
        )
        print(
            "  share, 95% interval:", [round(float(x), 1) for x in shares[-1]]
        )
    share, lower, upper = np.array(shares).T
    ax_options[0, 1].bar(  # type: ignore
        range(len(exclusions)),
        share,
        width=0.6,
        color=blue,
        alpha=0.8,
        yerr=[share - lower, upper - share],
        capsize=5,
        error_kw={"elinewidth": 1.5, "ecolor": "black", "alpha": 0.8},
    )
    ax_options[0, 1].set_xticks(range(len(exclusions)))  # type: ignore
    ax_options[0, 1].set_xticklabels(  # type: ignore
        [label for label, _ in exclusions], rotation=-45, ha="left"
    )
    ax_options[0, 1].set_ylabel("2025 cases in multi-county outbreaks (%)")  # type: ignore
    ax_options[0, 1].set_ylim(0, 100)  # type: ignore
    ax_options[0, 1].set_yticks(range(0, 101, 20))  # type: ignore
    # e: other definitions of a multi-county outbreak (regions included), by bin
    # (label, color, fewest cases of a county that is kept in the data, steps
    # between neighbors, whether the outbreak counties are counted by their connected group)
    definitions = [
        (
            "Focal & neighbor county must have 3+ cases (current definition)",
            blue,
            3,
            1,
            False,
        ),
        ("Focal & neighbor county must have 5+ cases", "#eb6834", 5, 1, False),
        ("≥ half of counties have 3+ cases", "#8a5a9e", 1, 1, True),
        ("Neighbors of neighbors allowed", "#2a9d8f", 3, 2, False),
    ]
    for k, (_, _, min_cases, steps, groups) in enumerate(definitions):
        df_k = df_spread_regions.filter(pl.col("cases_outbreak") >= min_cases)
        matrix_pre, matrix_post = period_matrices(
            df_k, adjacency_matrix, regions, steps
        )
        plot_increasing_spread_over_time(
            df_k,
            matrix_pre,
            matrix_post,
            posterior_samples_pre,
            posterior_samples,
            fig_options,
            ax_options[0, 2],  # type: ignore
            year_bins=year_bins,
            n_series=len(definitions),
            series_index=k,
            connected_groups=groups,
        )
    bars = ax_options[0, 2].patches  # type: ignore
    n_bins = len(bars) // len(definitions)
    for k, (_, color, _, _, _) in enumerate(definitions):
        for bar in bars[k * n_bins : (k + 1) * n_bins]:
            bar.set_color(color)
    labels = [tick.get_text() for tick in ax_options[0, 2].get_xticklabels()]  # type: ignore
    ax_options[0, 2].set_xticks(  # type: ignore
        [
            (first.get_x() + last.get_x() + last.get_width()) / 2
            for first, last in zip(bars[:n_bins], bars[-n_bins:])
        ]
    )
    ax_options[0, 2].set_xticklabels(labels, rotation=-45, ha="left")  # type: ignore
    ax_options[0, 2].set_ylim(0, 100)  # type: ignore
    ax_options[0, 2].set_yticks(range(0, 101, 20))  # type: ignore
    ax_options[0, 2].legend(  # type: ignore
        handles=[
            Patch(color=color, alpha=0.8, label=label)
            for label, color, _, _, _ in definitions
        ],
        loc=(0.0, 1.05),
        edgecolor="black",
    )
    # e: every year, the cases in multi-county outbreaks (dashed line, left
    # axis) and the number of multi-county outbreaks (groups of two or more
    # outbreak places joined by borders, regions included; bars, right axis),
    # drawn as the cases and counties of plot_cases_over_time in
    # figure_measles_burden_overview.py are
    by_year = multi_county_outbreaks_by_year(
        df_spread_regions, adjacency_matrix, regions
    )
    print(
        "  multi-county outbreaks by year:",
        dict(zip(by_year["year"], by_year["multi_county_outbreaks"])),
    )
    print(
        "  cases in them by year:",
        dict(zip(by_year["year"], by_year["multi_county_cases"])),
    )
    ax_cases = ax_options[1, 1]  # type: ignore
    ax_cases.plot(
        by_year["year"],
        by_year["multi_county_cases"],
        marker="^",
        markersize=10,
        color="black",
        linestyle="--",
        label="Cases in multi-county outbreaks",
    )
    ax_cases.set_xlabel("Year")
    ax_cases.set_ylabel("Cases in multi-county outbreaks (N)")
    ax_cases.set_xticks([2013, 2016, 2019, 2022, 2025])
    # room above the highest point for the legend
    ax_cases.set_ylim(0, 1.3 * by_year["multi_county_cases"].max())  # type: ignore
    ax_outbreaks = ax_cases.twinx()
    ax_outbreaks.bar(
        by_year["year"].to_numpy() - 0.5,
        by_year["multi_county_outbreaks"].to_numpy(),
        width=1,
        alpha=0.2,
        align="edge",
        edgecolor="grey",
        label="Number of multi-county outbreaks",
    )
    ax_outbreaks.set_ylabel(
        "Multi-county outbreaks (N)", rotation=-90, labelpad=30
    )
    ax_outbreaks.spines["top"].set_visible(False)
    ax_outbreaks.spines["left"].set_visible(False)
    ax_outbreaks.get_xaxis().tick_bottom()
    ax_outbreaks.get_yaxis().tick_right()
    ax_outbreaks.tick_params(axis="x", direction="out")
    ax_outbreaks.tick_params(axis="y", direction="out")
    for spine in ax_outbreaks.spines.values():
        spine.set_position(("outward", 5))
    ax_outbreaks.set_axisbelow(True)
    ax_outbreaks.yaxis.set_major_locator(tck.MaxNLocator(integer=True))
    ax_outbreaks.set_ylim(0, 1.3 * by_year["multi_county_outbreaks"].max())  # type: ignore
    handles_cases, labels_cases = ax_cases.get_legend_handles_labels()
    handles_outbreaks, labels_outbreaks = (
        ax_outbreaks.get_legend_handles_labels()
    )
    ax_cases.legend(
        [*handles_cases, *handles_outbreaks],
        [*labels_cases, *labels_outbreaks],
        edgecolor="black",
    )
    fig_options.tight_layout()
    for extension in ("png", "pdf"):
        fig_options.savefig(
            os.path.join(
                fig_dir,
                f"figure_supplemental_multi_county_outbreak_share_options.{extension}",
            ),
            dpi=300,
            bbox_inches="tight",
        )

    # Supplemental figure: the share of cases in multi-county outbreaks (as
    # panel a above calculates it, the regions left out as in panel c of the
    # main figure) in the analysis dataset (a: the counties with vaccine data,
    # the rows of df) and in all the data (b: every county with outbreak cases)
    fig_data, ax_data = setup_plot(1, 2, figsize=(22, 10))
    datasets = [
        (
            "Analysis dataset",
            df,
            adjacency_matrix_observed_pre,
            adjacency_matrix_observed,
        ),
        (
            "All data",
            df_spread,
            adjacency_matrix_spread_pre,
            adjacency_matrix_spread,
        ),
    ]
    for ax_k, (title, df_k, matrix_pre, matrix_post) in zip(ax_data, datasets):  # type: ignore
        print(
            f"{title}: {df_k.height} county-years, {df_k['cases_outbreak'].sum()} outbreak cases"
        )
        plot_increasing_spread_over_time(
            df_k,
            matrix_pre,
            matrix_post,
            posterior_samples_pre,
            posterior_samples,
            fig_data,
            ax_k,
            year_bins=year_bins,
        )
        for bar in ax_k.patches:
            bar.set_color(blue)
        ax_k.set_ylim(0, 100)
        ax_k.set_yticks(range(0, 101, 20))
    fig_data.tight_layout()
    for extension in ("png", "pdf"):
        fig_data.savefig(
            os.path.join(
                fig_dir,
                f"figure_supplemental_multi_county_outbreak_share_analysis_dataset_vs_all_data.{extension}",
            ),
            dpi=300,
            bbox_inches="tight",
        )

    # Supplemental figure: panel d of the main figure with the new school data
    # (every school year of 2021-2025) in place of the Washington Post file.
    # The independent and correlated multi-county outbreaks are told apart by
    # the spatial (CAR) effect of the All covariates fit with the Poisson
    # likelihood, the raw covariates, the population exponent fixed at 1 and
    # pop_density (02, fit_all_covariates_poisson_raw)
    car_fit_file = os.path.join(
        output_dir,
        "samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_[2021, 2022, 2023, 2024, 2025]_likelihood=poisson_covariate_scale=raw_pop_exponent=1_pop_density.npz",
    )
    new_school = read_new_school_data()
    if not os.path.exists(car_fit_file):
        print("skipped supplemental figure: the Poisson raw fit is missing.")
    elif new_school is not None:
        fig_school, ax_school = setup_plot(1, 1, figsize=(11, 10))  # type: ignore
        plot_county_level_variation(
            df,
            load_samples(car_fit_file),
            adjacency_matrix_observed,
            fig_school,
            ax_school,  # type: ignore
            school=new_school,
        )
        fig_school.tight_layout()
        for extension in ("png", "pdf"):
            fig_school.savefig(
                os.path.join(
                    fig_dir,
                    f"figure_supplemental_within_county_coverage_variation_new_school_data.{extension}",
                ),
                dpi=300,
                bbox_inches="tight",
            )

    # Supplemental figure of panel d: rural and urban counties, and counties
    # with 2 or more, 3 or more and 5 or more schools
    if homogeneity is not None:
        print("\n" + "=" * 70)
        print("Creating supplemental figure of panel d...")
        print("=" * 70)
        fig_homogeneity = create_homogeneity_figure(
            homogeneity, adjacency_matrix
        )
        for extension in ("png", "pdf"):
            fig_homogeneity.savefig(
                os.path.join(
                    fig_dir,
                    f"figure_supplemental_multi_county_homogeneity.{extension}",
                ),
                dpi=300,
                bbox_inches="tight",
            )

        # Supplemental table of panel d: every model of HOMOGENEITY_MODELS
        print("\n" + "=" * 70)
        print("Creating supplemental table of panel d...")
        print("=" * 70)
        homogeneity_table = create_homogeneity_table(
            homogeneity, adjacency_matrix
        )
        homogeneity_table.to_csv(
            path_or_buf=os.path.join(
                fig_dir,
                "table_supplemental_multi_county_homogeneity_regressions.csv",
            ),
            index=False,
        )
        print("\nMulti-county homogeneity table saved!")
        print(homogeneity_table.to_string(index=False))
