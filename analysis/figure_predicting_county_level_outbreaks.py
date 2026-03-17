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


def setup_plot(nrows, ncols, figsize=(20, 22)):
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

    # for i in range(len(year_medians[0])):
    #     ax.plot([i - offset, i + offset],
    #             [year_medians[0][i], year_medians[1][i]],
    #             linestyle = '-',
    #             color = 'grey')

    # line at 0%
    ax.axhline(0, ls="--", color="k")

    ax.legend(loc=(0.01, 1.1))

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


def plot_spatial_importance(
    samples_pre, samples_post, df, adjacency_pre, adjacency_post, fig, ax
):
    """
    Plot spatial autocorrelation by outbreak case size bucket and time period.

    Parameters:
    -----------
    samples_pre : dict
        Posterior samples for 2011-2020 period
    samples_post : dict
        Posterior samples for 2021-2025 period
    df : polars.DataFrame
        Analysis dataset
    adjacency_pre : numpy.ndarray
        Adjacency matrix for counties in 2011-2020 period
    adjacency_post : numpy.ndarray
        Adjacency matrix for counties in 2021-2025 period
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axis for plotting
    """

    def calculate_correlations_for_period(samples, df_period, adjacency):
        """Helper function to calculate correlations for one time period."""
        raw_spatial_effect = samples["raw_spatial_effect"]
        spatial_sigma = samples["spatial_sigma"]
        spatial_effect = raw_spatial_effect * spatial_sigma[:, None]

        # Get counties with neighbors
        has_a_neighbor = adjacency.sum(axis=0) > 0
        counties_with_neighbors = (
            df_period["fips"]
            .unique(maintain_order=True)
            .to_numpy()[has_a_neighbor]
        )

        # Filter adjacency matrix to only counties with neighbors
        adjacency_filtered = adjacency[has_a_neighbor][:, has_a_neighbor]

        # Create county-level dataframe (one row per county)
        df_counties = (
            df_period.filter(pl.col("fips").is_in(counties_with_neighbors))
            .group_by("fips", maintain_order=True)
            .agg(
                pl.first("county"),
                pl.first("state"),
                pl.max("cases_outbreak"),
                pl.mean("pop"),
                estimated_mmr_cov=pl.mean("estimated_mmr_cov"),
            )
        )

        # Create case size buckets
        df_counties = df_counties.with_columns(
            case_buckets=pl.when(pl.col("cases_outbreak") == 0)
            .then(pl.lit("0"))
            .when(pl.col("cases_outbreak") <= 4)
            .then(pl.lit("1"))
            .when(pl.col("cases_outbreak") <= 9)
            .then(pl.lit("2"))
            .when(pl.col("cases_outbreak") <= 19)
            .then(pl.lit("3"))
            .otherwise(pl.lit("4"))
        )

        # Calculate neighbor correlations by bucket
        bucket_correlations = []

        for case_bucket in ["0", "1", "2", "3", "4"]:
            bucket_mask = (
                df_counties["case_buckets"] == case_bucket
            ).to_numpy()

            # Get indices of counties in this bucket
            bucket_county_indices = np.where(bucket_mask)[0]

            if len(bucket_county_indices) == 0:
                bucket_correlations.append([])
                continue

            # For each posterior sample, calculate correlation in this bucket
            correlations_per_sample = []
            for s in range(spatial_effect.shape[0]):
                county_values = []
                neighbor_means = []

                for county_idx in bucket_county_indices:
                    # Get ALL neighbors from filtered adjacency matrix
                    neighbors = adjacency_filtered[county_idx] > 0
                    if neighbors.sum() == 0:
                        continue

                    # County's spatial effect
                    county_values.append(spatial_effect[s, county_idx])

                    # Mean of ALL neighbors' spatial effects
                    neighbor_means.append(spatial_effect[s, neighbors].mean())

                # Calculate correlation across all counties in this bucket for this sample
                if len(county_values) > 1:
                    corr = np.corrcoef(county_values, neighbor_means)[0, 1]
                    correlations_per_sample.append(corr)

            bucket_correlations.append(correlations_per_sample)

        return bucket_correlations, df_counties

    # Calculate correlations for both time periods with their respective adjacency matrices
    df_pre = df.filter(pl.col("year") <= 2020)
    df_post = df.filter(pl.col("year") >= 2021)

    bucket_correlations_pre, df_counties_pre = (
        calculate_correlations_for_period(samples_pre, df_pre, adjacency_pre)
    )
    bucket_correlations_post, df_counties_post = (
        calculate_correlations_for_period(
            samples_post, df_post, adjacency_post
        )
    )

    # Plotting
    def get_stats(data_list):
        medians = [np.median(d) if len(d) > 0 else np.nan for d in data_list]
        lowers = [
            np.percentile(d, 25) if len(d) > 0 else np.nan for d in data_list
        ]
        uppers = [
            np.percentile(d, 75) if len(d) > 0 else np.nan for d in data_list
        ]
        means = [np.mean(d) if len(d) > 0 else np.nan for d in data_list]
        return medians, lowers, uppers, means

    def trim_to_percentiles(data_list):
        """Trim data to 2.5th to 97.5th percentiles for violin plot."""
        trimmed = []
        for d in data_list:
            if len(d) > 0:
                d_array = np.array(d)
                lower_bound = np.percentile(d_array, 2.5)
                upper_bound = np.percentile(d_array, 97.5)
                trimmed_d = d_array[
                    (d_array >= lower_bound) & (d_array <= upper_bound)
                ]
                trimmed.append(trimmed_d.tolist())
            else:
                trimmed.append([])
        return trimmed

    # Calculate stats on full data
    medians_pre, lowers_pre, uppers_pre, means_pre = get_stats(
        bucket_correlations_pre
    )
    medians_post, lowers_post, uppers_post, means_post = get_stats(
        bucket_correlations_post
    )

    # Trim data for violin plots
    bucket_correlations_pre_trimmed = trim_to_percentiles(
        bucket_correlations_pre
    )
    bucket_correlations_post_trimmed = trim_to_percentiles(
        bucket_correlations_post
    )

    offset = 0.2
    valid_indices = [
        i
        for i in range(5)
        if len(bucket_correlations_pre[i]) > 0
        or len(bucket_correlations_post[i]) > 0
    ]

    # Plot pre-2021 data
    if any(len(bucket_correlations_pre_trimmed[i]) > 0 for i in valid_indices):
        positions_pre = [
            i - offset
            for i in valid_indices
            if len(bucket_correlations_pre_trimmed[i]) > 0
        ]
        data_pre = [
            bucket_correlations_pre_trimmed[i]
            for i in valid_indices
            if len(bucket_correlations_pre_trimmed[i]) > 0
        ]

        violinparts1 = ax.violinplot(
            data_pre, positions=positions_pre, widths=0.3, showextrema=False
        )
        for pc in violinparts1["bodies"]:
            pc.set(facecolor="blue")

        valid_pre_indices = [
            i for i in valid_indices if len(bucket_correlations_pre[i]) > 0
        ]
        ax.errorbar(
            x=positions_pre,
            y=[medians_pre[i] for i in valid_pre_indices],
            yerr=[
                [medians_pre[i] - lowers_pre[i] for i in valid_pre_indices],
                [uppers_pre[i] - medians_pre[i] for i in valid_pre_indices],
            ],
            fmt="o",
            capsize=5,
            capthick=2,
            color="blue",
        )

    # Plot post-2020 data
    if any(
        len(bucket_correlations_post_trimmed[i]) > 0 for i in valid_indices
    ):
        positions_post = [
            i + offset
            for i in valid_indices
            if len(bucket_correlations_post_trimmed[i]) > 0
        ]
        data_post = [
            bucket_correlations_post_trimmed[i]
            for i in valid_indices
            if len(bucket_correlations_post_trimmed[i]) > 0
        ]

        violinparts2 = ax.violinplot(
            data_post, positions=positions_post, widths=0.3, showextrema=False
        )
        for pc in violinparts2["bodies"]:
            pc.set(facecolor="green")

        valid_post_indices = [
            i for i in valid_indices if len(bucket_correlations_post[i]) > 0
        ]
        ax.errorbar(
            x=positions_post,
            y=[medians_post[i] for i in valid_post_indices],
            yerr=[
                [medians_post[i] - lowers_post[i] for i in valid_post_indices],
                [uppers_post[i] - medians_post[i] for i in valid_post_indices],
            ],
            fmt="o",
            capsize=5,
            capthick=2,
            color="green",
        )

    # Bayesian comparison: probability that correlation is higher in post period
    for i in valid_indices:
        if (
            len(bucket_correlations_pre[i]) > 0
            and len(bucket_correlations_post[i]) > 0
        ):
            corr_pre = np.array(bucket_correlations_pre[i])
            corr_post = np.array(bucket_correlations_post[i])

            # Probability that post > pre
            prob_post_greater = np.mean(corr_post > corr_pre)

            # Two-sided: probability of difference in either direction
            ks_statistic, p_value = stats.ks_2samp(
                corr_pre, corr_post, alternative="greater"
            )

            # Bayesian interpretation: if prob_different is low, groups differ
            if p_value < 0.01:  # type: ignore
                sig_text = "**"
            elif p_value < 0.05:  # type: ignore
                sig_text = "*"
            elif p_value < 0.10:  # type: ignore
                sig_text = "†"
            else:
                sig_text = "n.s."

            max_height = np.nanmax([uppers_pre[i], uppers_post[i]])
            bracket_height = max_height + 0.05
            x1, x2 = i - offset, i + offset
            h = 0.02

            ax.plot(
                [x1, x2], [bracket_height, bracket_height], "k-", linewidth=1
            )
            ax.plot(
                [x1, x1],
                [bracket_height, bracket_height - h],
                "k-",
                linewidth=1,
            )
            ax.plot(
                [x2, x2],
                [bracket_height, bracket_height - h],
                "k-",
                linewidth=1,
            )
            ax.text(
                (x1 + x2) / 2,
                bracket_height + 0.02,
                sig_text,
                ha="center",
                va="bottom",
                fontsize=16,
                fontweight="bold",
            )

    ax.set_xticks(range(5))
    ax.set_xticklabels(
        ["0", "1-4", "5-9", "10-19", "20+"], rotation=-60, ha="left"
    )
    ax.legend(
        [
            violinparts1["bodies"][0],  # type: ignore
            violinparts2["bodies"][0],
        ],  # type: ignore
        ["2011-2020", "2021-2025"],
    )
    ax.set_xlabel("Outbreak size (N)")
    ax.set_ylabel("Residual spatial autocorrelation")

    return (
        fig,
        ax,
        {
            "bucket_correlations_pre": bucket_correlations_pre,
            "bucket_correlations_post": bucket_correlations_post,
            "df_counties_pre": df_counties_pre,
            "df_counties_post": df_counties_post,
        },
    )


def plot_increasing_spread_over_time(
    df,
    adjacency_matrix_pre,
    adjacency_matrix_post,
    posterior_samples_pre,
    posterior_samples_post,
    fig,
    ax,
    significance_threshold=0.95,
):
    """
    Plot number of county-years with neighboring outbreaks over time.

    Outbreak defined as >= 3 cases. Counts unique counties involved in neighboring
    outbreaks (where both the county and at least one neighbor have >= 3 cases),
    but only if at least one county in the cluster has a significant spatial term.

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
    significance_threshold : float
        Threshold for determining if spatial effect is significant (default: 0.95)
        A county is considered to have significant spatial effect if no more than
        5% (1 - threshold) of posterior mass is on the opposite side of zero.
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
        year_df, adjacency_matrix, significant_spatial_dict, all_fips
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
        all_fips : list
            All FIPS codes in the period (matching adjacency matrix order)
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

            # Only count this cluster if:
            # 1. It has more than 1 county (neighboring outbreak)
            # 2. At least one county has significant spatial effect
            if len(cluster) > 1 and cluster_has_significant_spatial:
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

    # Get significant spatial effects for both time periods
    # This returns dicts mapping FIPS -> bool (only for counties with neighbors)
    significant_spatial_pre = get_significant_spatial_counties(
        posterior_samples_pre,
        adjacency_matrix_pre,
        all_fips_pre,
        significance_threshold,
    )
    significant_spatial_post = get_significant_spatial_counties(
        posterior_samples_post,
        adjacency_matrix_post,
        all_fips_post,
        significance_threshold,
    )

    years = sorted(df["year"].unique().to_list())
    n_neighbor_outbreaks = []

    for year in years:
        year_df = df.filter(pl.col("year") == year)

        if year < 2021:
            # Use pre-2021 model and adjacency matrix
            count = count_valid_clusters(
                year_df,
                adjacency_matrix_pre,
                significant_spatial_pre,
                all_fips_pre,
            )
        else:
            # Use post-2020 model and adjacency matrix
            count = count_valid_clusters(
                year_df,
                adjacency_matrix_post,
                significant_spatial_post,
                all_fips_post,
            )

        n_neighbor_outbreaks.append(count)

    # Plot single black line
    ax.plot(
        years,
        n_neighbor_outbreaks,
        "o-",
        color="black",
        linewidth=2,
        markersize=8,
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Outbreaks spreading over multiple counties (N)")
    ax.set_ylim(0)
    ax.set_xticks([2015, 2020, 2025])

    return fig, ax


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
            "final_outbreak_size_predictions_[2021, 2022, 2023, 2024, 2025].pkl",
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
    plot_spatial_importance(
        posterior_samples_pre,
        posterior_samples,
        df,
        adjacency_matrix_observed_pre,
        adjacency_matrix_observed,
        fig,
        ax[1, 0],  # type: ignore
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
    plot_increasing_spread_over_time(
        df,
        adjacency_matrix_observed_pre,
        adjacency_matrix_observed,
        posterior_samples_pre,
        posterior_samples,
        fig,
        ax[1, 1],  # type: ignore
        significance_threshold=0.95,
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
