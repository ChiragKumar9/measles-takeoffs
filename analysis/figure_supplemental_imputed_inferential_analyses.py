import os
import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy.stats as stats
import yaml
from matplotlib import rc
from tqdm.auto import tqdm

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


def calculate_moran_eigenvectors_adjacency(
    adjacency_matrix, n_vectors="positive"
):
    """
    Calculate Moran's eigenvector maps from adjacency matrix

    Parameters:
    -----------
    adjacency_matrix : array-like, shape (n, n)
        Binary adjacency matrix (1 = neighbors, 0 = not neighbors)
    n_vectors : int or 'positive'
        Number of eigenvectors to retain

    Returns:
    --------
    eigenvectors : array, shape (n, n_vectors)
    eigenvalues : array, shape (n_vectors,)
    """
    W = np.asarray(adjacency_matrix, dtype=float)
    n = W.shape[0]

    # Row-standardize
    row_sums = W.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    W = W / row_sums[:, np.newaxis]

    # Center the matrix
    I = np.eye(n)
    ones = np.ones((n, 1))
    H = I - (ones @ ones.T) / n

    # Double-centered spatial weights
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


def spearman_msr_blocked(
    x, y, adjacency_matrix, county_indices, n_permutations=99999, seed=108
):
    """
    MSR with block permutation for repeated measures (time series per county)

    Permutes entire counties together, preserving within-county temporal structure.
    Handles unbalanced panels (different # of observations per county).

    Parameters:
    -----------
    x, y : array-like
        Data vectors (length = total observations across all counties/times)
    adjacency_matrix : array, shape (n_counties, n_counties)
        County-level adjacency matrix
    county_indices : array-like
        County identifier for each observation (0-indexed)
        Example: [0, 0, 0, 1, 1, 2, 2, 2, ...] means first 3 obs are county 0, etc.
    n_permutations : int
        Number of permutations
    seed : int, optional
        Random seed

    Returns:
    --------
    rho_obs : float
        Observed correlation
    p_value : float
        MSR p-value
    rho_null : array
        Null distribution
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    county_indices = np.asarray(county_indices)

    # Validation
    if len(x) != len(y) or len(x) != len(county_indices):
        raise ValueError(
            f"x ({len(x)}), y ({len(y)}), and county_indices ({len(county_indices)}) "
            f"must have same length"
        )

    # Get unique counties and validate
    unique_counties = np.unique(county_indices)
    n_counties = len(unique_counties)

    if n_counties != adjacency_matrix.shape[0]:
        raise ValueError(
            f"Number of unique counties ({n_counties}) doesn't match "
            f"adjacency matrix size ({adjacency_matrix.shape[0]})"
        )

    # Check that county_indices are 0-indexed and contiguous
    if not np.array_equal(unique_counties, np.arange(n_counties)):  # type: ignore
        raise ValueError(
            f"county_indices must be 0-indexed integers from 0 to {n_counties - 1}. "
            f"Got: {unique_counties}"
        )

    # Observed correlation
    rho_obs, _ = stats.spearmanr(x, y)

    # Get eigenvectors at county level
    eigenvectors, eigenvalues = calculate_moran_eigenvectors_adjacency(
        adjacency_matrix
    )

    # Pre-compute masks for each county (for efficiency)
    county_masks = [county_indices == i for i in range(n_counties)]
    county_obs_counts = [np.sum(mask) for mask in county_masks]

    # Check for issues
    if any(count == 0 for count in county_obs_counts):
        raise ValueError(
            "Some counties in adjacency matrix have no observations in data"
        )

    # Permutations
    rho_null = np.zeros(n_permutations)

    for perm_i in tqdm(range(n_permutations)):  # type: ignore
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

        # Calculate correlation for this permutation
        rho_null[perm_i], _ = stats.spearmanr(x, y_perm)  # type: ignore

    # P-value
    n_extreme = np.sum(np.abs(rho_null) >= np.abs(rho_obs))  # type: ignore
    p_value = (n_extreme + 1) / (n_permutations + 1)

    return rho_obs, p_value, rho_null


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
                adjacency_matrix_pre,
                county_indices=county_indices,
            )
        )

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
        print(
            spearman_msr_blocked(
                df_correlation.filter(pl.col("year_of_interest") == True)[
                    vaccination_metric
                ],
                df_correlation.filter(pl.col("year_of_interest") == True)[
                    "cases_outbreak"
                ],
                adjacency_matrix_post,
                county_indices=county_indices,
            )
        )

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

    # 2011-2020 (blue) goes ABOVE (+offset), 2021-2025 (green) goes BELOW (-offset)
    violinparts1 = ax.violinplot(
        data_pre_reversed,
        positions=[p + offset for p in positions_reversed],
        widths=0.35,
        showextrema=False,
        vert=False,
    )

    for pc in violinparts1["bodies"]:
        pc.set(facecolor="blue")

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
        color="blue",
    )

    # add means as stars
    ax.scatter(
        means_pre_reversed,
        [p + offset for p in positions_reversed],
        marker="*",
        s=90,
        color="blue",
    )

    violinparts2 = ax.violinplot(
        data_post_reversed,
        positions=[p - offset for p in positions_reversed],
        widths=0.35,
        showextrema=False,
        vert=False,
    )

    for pc in violinparts2["bodies"]:
        pc.set(facecolor="green")

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
        color="green",
    )

    # add means as stars
    ax.scatter(
        means_post_reversed,
        [p - offset for p in positions_reversed],
        marker="*",
        s=90,
        color="green",
    )

    # Add significance bar between post-COVID 0 cases and ≥20 cases
    # ≥20 is at index 0 of reversed arrays, which maps to position 3
    # 0 cases is at index 2 of reversed arrays, which maps to position 0
    y1 = (
        positions_reversed[0] - offset
    )  # Position for "≥20" cases (green/post) - at top
    y2 = (
        positions_reversed[2] - offset
    )  # Position for "0" cases (green/post) - at bottom

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
    )  # Position for "1-9" cases (green/post)
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
        "**",
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
    df, fig, ax, year_buckets, model_name="All covariates"
):
    """
    Plot coverage effects from models, specifically for imputed data
    """
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
        with open(
            os.path.join(
                output_dir,
                f"samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_{model_name}_{year}_IMPUTED.pkl",
            ),
            "rb",
        ) as file:
            posterior_samples = pickle.load(file)

        coverage_effects = posterior_samples["iv_global"][:, 0]
        # scale by the standard deviation of the coverage to get an
        # interpretable term
        std = (
            df.filter(pl.col("year").is_in(year))
            .select("estimated_mmr_cov")
            .to_numpy()
            .std()
        )
        coverage_effects = coverage_effects / std
        coverage_effects = jnp.exp(coverage_effects) * 100
        # to express as percent change
        coverage_effects = coverage_effects - 100
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
    ax.set_ylabel(
        "% change in outbreak size per 1pp\nincrease in MMR vaccine coverage"
    )

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


if __name__ == "__main__":
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

    # Pre-COVID period (2011-2020)
    with open(
        os.path.join(
            output_dir,
            "samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_[2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]_IMPUTED.pkl",
        ),
        "rb",
    ) as file:
        posterior_samples_pre_imputed = pickle.load(file)

    # Post-COVID period (2021-2025)
    with open(
        os.path.join(
            output_dir,
            "samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_[2021, 2022, 2023, 2024, 2025]_IMPUTED.pkl",
        ),
        "rb",
    ) as file:
        posterior_samples_post_imputed = pickle.load(file)

    # Create figure with 2 rows, 2 columns
    fig, ax = setup_plot(2, 2)  # type: ignore

    # ===== TOP ROW =====

    # Panel a: Coverage distribution violin plot
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
    plot_coverage_distribution(
        df_imputed,
        95,
        fig,
        ax[0, 0],  # type: ignore
        vaccination_metric="estimated_mmr_cov",
        adjacency_matrix_pre=adjacency_matrix_observed_pre,
        adjacency_matrix_post=adjacency_matrix_observed_post,
    )

    # Panel b: Coverage effects from models
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

    year_buckets = [
        [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
        [2021, 2022, 2023, 2024, 2025],
    ]

    plot_coverage_various_models(
        df_imputed,
        fig,
        ax[0, 1],  # type: ignore
        year_buckets=year_buckets,
        model_name="All covariates",
    )

    # ===== BOTTOM ROW =====

    # Panel c: Spatial correlation posterior (post-COVID period)
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

    plot_car_effect_correlation_simple(
        posterior_samples_post_imputed,
        fig,
        ax[1, 0],  # type: ignore
    )

    # Panel d: Increasing spread over time
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
        df_imputed,
        adjacency_matrix_observed_pre,
        adjacency_matrix_observed_post,
        posterior_samples_pre_imputed,
        posterior_samples_post_imputed,
        fig,
        ax[1, 1],  # type: ignore
        year_bins=[(2011, 2015), (2016, 2020), (2021, 2024), (2025, 2025)],
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

    # Print summary of imputed observations
    if "imputation_source" in df_imputed.columns:
        n_total = df_imputed.height
        n_imputed = df_imputed.filter(
            pl.col("imputation_source").is_not_null()
        ).height
        print(f"\nDataset composition:")
        print(f"  Total observations: {n_total}")
        print(
            f"  Original (non-imputed): {n_total - n_imputed} ({100 * (n_total - n_imputed) / n_total:.1f}%)"
        )
        print(f"  Imputed: {n_imputed} ({100 * n_imputed / n_total:.1f}%)")

        # Breakdown by time period
        n_pre = df_imputed.filter(pl.col("year") < 2021).height
        n_imputed_pre = df_imputed.filter(
            (pl.col("year") < 2021)
            & (pl.col("imputation_source").is_not_null())
        ).height

        n_post = df_imputed.filter(pl.col("year") >= 2021).height
        n_imputed_post = df_imputed.filter(
            (pl.col("year") >= 2021)
            & (pl.col("imputation_source").is_not_null())
        ).height

        print(f"\n  2011-2020:")
        print(f"    Total: {n_pre}")
        print(
            f"    Imputed: {n_imputed_pre} ({100 * n_imputed_pre / n_pre:.1f}%)"
        )

        print(f"  2021-2025:")
        print(f"    Total: {n_post}")
        print(
            f"    Imputed: {n_imputed_post} ({100 * n_imputed_post / n_post:.1f}%)"
        )
