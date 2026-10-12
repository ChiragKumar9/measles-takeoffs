import os
import pickle
from re import L

import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
adjacency_matrix_observed_post = adjacency.assemble_adjacency_matrix(
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
# sorted so this lines up with the np.unique-ordered county_indices used below
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
    if not np.array_equal(unique_counties, np.arange(n_counties)):
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

    for perm_i in tqdm(range(n_permutations)):
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

    def rho_pre_post(rows):
        m = post[rows]
        return (
            stats.spearmanr(x[rows][~m], y[rows][~m])[0],
            stats.spearmanr(x[rows][m], y[rows][m])[0],
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
        2 * min(np.mean(diff >= 0), np.mean(diff <= 0)), 2 / (n_bootstrap + 1)
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

    print(
        spearman_msr_blocked(
            df_correlation.filter(pl.col("year_of_interest") == False)[
                vaccination_metric
            ],
            df_correlation.filter(pl.col("year_of_interest") == False)[
                "cases_outbreak"
            ],
            adjacency_matrix_observed_pre,
            county_indices=county_indices,
        )
    )

    county_ids = df_correlation.filter(pl.col("year_of_interest") == True)[
        "fips"
    ].to_numpy()
    unique_ids = np.unique(county_ids)
    county_id_to_index = {cid: i for i, cid in enumerate(unique_ids)}
    county_indices = np.array([county_id_to_index[cid] for cid in county_ids])

    print(
        spearman_msr_blocked(
            df_correlation.filter(pl.col("year_of_interest") == True)[
                vaccination_metric
            ],
            df_correlation.filter(pl.col("year_of_interest") == True)[
                "cases_outbreak"
            ],
            adjacency_matrix_observed_post,
            county_indices=county_indices,
        )
    )

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

    # KS test between the two distributions
    print(
        "KS test between pre COVID coverage across no outbreak and 10+ cases"
    )
    ks_stat, p_value = stats.ks_2samp(
        df.filter(pl.col("year_of_interest") == False).filter(
            pl.col("outbreak_size_buckets") == "1"
        )[vaccination_metric][0],
        df.filter(pl.col("year_of_interest") == False).filter(
            pl.col("outbreak_size_buckets") == "0"
        )[vaccination_metric][0],
        method="exact",
        alternative="greater",
    )
    print(f"KS statistic: {ks_stat}, p-value: {p_value}")

    print(
        "KS test between pre and post COVID coverage with greater than 10 cases"
    )
    ks_stat, p_value = stats.ks_2samp(
        df.filter(pl.col("year_of_interest") == False).filter(
            pl.col("outbreak_size_buckets") == "0"
        )[vaccination_metric][0],
        df.filter(pl.col("year_of_interest") == True).filter(
            pl.col("outbreak_size_buckets") == "1"
        )[vaccination_metric][0],
        method="exact",
        alternative="less",
    )
    print(f"KS statistic: {ks_stat}, p-value: {p_value}")

    print(
        "KS test between post COVID coverage across no outbreak and 10+ cases"
    )
    ks_stat, p_value = stats.ks_2samp(
        df.filter(pl.col("year_of_interest") == True).filter(
            pl.col("outbreak_size_buckets") == "1"
        )[vaccination_metric][0],
        df.filter(pl.col("year_of_interest") == True).filter(
            pl.col("outbreak_size_buckets") == "0"
        )[vaccination_metric][0],
        method="exact",
        alternative="greater",
    )
    print(f"KS statistic: {ks_stat}, p-value: {p_value}")

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

    print(
        spearman_msr_blocked(
            df_correlation[vaccination_metric],
            df_correlation["cases_outbreak"],
            adjacency_matrix_observed_all,
            county_indices=county_indices,
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
        pc.set(facecolor="blue")

    ax.errorbar(
        x=[i - offset for i in range(len(medians_pre))],
        y=medians_pre,
        yerr=[medians_pre - lowers_pre, uppers_pre - medians_pre],
        fmt="o",
        capsize=10,
        capthick=2,
        color="blue",
    )

    ax.scatter(
        [i - offset for i in range(len(means_pre))],
        means_pre,
        marker="*",
        s=90,
        color="blue",
    )

    violinparts2 = ax.violinplot(
        data_post,
        positions=[i + offset for i in range(len(data_post))],
        widths=0.3,
        showextrema=False,
    )
    for pc in violinparts2["bodies"]:
        pc.set(facecolor="green")

    ax.errorbar(
        x=[i + offset for i in range(len(medians_post))],
        y=medians_post,
        yerr=[medians_post - lowers_post, uppers_post - medians_post],
        fmt="o",
        capsize=10,
        capthick=2,
        color="green",
    )

    ax.scatter(
        [i + offset for i in range(len(means_post))],
        means_post,
        marker="*",
        s=90,
        color="green",
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
        color="blue",
        label="2011-2020",
        edgecolors="none",
    )

    ax.scatter(
        df_post[vaccination_metric].to_numpy(),
        df_post["cases_outbreak"].to_numpy(),
        alpha=0.5,
        s=50,
        color="green",
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
    ax.set_xlim(left=85, right=100)

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


def plot_coverage_various_models(df, fig, ax):
    year_buckets = [
        [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
        [2021, 2022, 2023, 2024, 2025],
    ]
    models = ["MMR coverage", "MMR coverage + exemptions", "All covariates"]
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
            with open(
                os.path.join(
                    output_dir,
                    f"samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_{model_name}_{year}.pkl",
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

    colors = {
        "MMR coverage": "red",
        "MMR coverage + exemptions": "purple",
        "All covariates": "orange",
    }

    # dashed line at 0
    ax.axhline(y=0, color="k", linestyle="--")

    for model_name, offset in zip(models, [-0.07, 0, 0.07]):
        model_effects = assembled_effects.filter(
            pl.col("model") == model_name
        ).sort("year_index")
        if model_name == "All covariates":
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

    ax.set_xlabel("Year")
    ax.set_ylabel(
        "% change in outbreak size per 1pp\nincrease in MMR vaccine coverage"
    )
    ax.legend(
        loc=(0.2, 0.65),
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
        std_wh = df["estimated_mmr_cov"].to_numpy().std()

        if not weight_by_mu:
            # Use upper-level parameter
            effects = samples["effects_over_time"][:, :, 0]
            effects = effects[:, ::-1]  # Reverse to chronological order
            effects_scaled = effects / std_wh
            effects_pct = jnp.exp(effects_scaled) * 100 - 100
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
            effects_scaled = effects_obs / std_wh
            effects_pct = jnp.exp(effects_scaled) * 100 - 100

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

    ax.set_xlabel("Year")
    ax.set_ylabel(
        "% change in outbreak size per 1pp\nincrease in MMR vaccine coverage"
    )
    ax.set_xticks([2011, 2015, 2020, 2025])

    return fig, ax


def trend_draws(beta, mu, row_years, years, drop=()):
    """
    Per-draw trend statistics for the time-varying coverage effect.

    beta      [S, T]  log outbreak-size rate per +1pp coverage, chronological
    mu        [S, N]  model-expected cases for each county-year
    row_years [N]     calendar year of each county-year (columns of mu)
    years     [T]     calendar years matching beta's columns
    drop      years to leave out (sensitivity analysis)

    Returns a dict of [S] arrays (f, l = first and last kept year):
      1. slopes: of E_t = 100*(exp(beta_t)-1) [pp of %-effect / yr], and the
         log-scale version 100*(exp(slope of beta_t)-1) [% / yr in the rate
         ratio per +1pp]
      2. in cases: A = mubar*sA/100 (fixed average outbreak size) and
         sB = slope of AME_t = mean_i(mu_it)*(exp(beta_t)-1) (year-specific
         size), both [cases / pp coverage / yr]
      3. endpoints f and l, from the line fit on the %-scale, the line fit on
         the log scale, and the raw yearly estimates; effects in % per pp,
         cases per county-year per +1pp; "diff" = l minus f, per draw
      4. Spearman rho between the yearly effect E_t and the year (how
         monotone the trend is)
      mubar  mean of mu over all kept county-years [cases per county-year]
    """
    years = np.asarray(years)
    row_years = np.asarray(row_years)
    keep = ~np.isin(years, list(drop))
    row_keep = ~np.isin(row_years, list(drop))
    yk = years[keep]
    xc = yk - yk.mean()  # centered year, [yr]

    def slope_and_ends(y):  # y [S, Tk] -> slope [S], fitted first/last [S]
        ybar = y.mean(axis=1)
        s = (y - ybar[:, None]) @ xc / (xc @ xc)
        return s, ybar + s * xc[0], ybar + s * xc[-1]

    growth = np.exp(beta[:, keep]) - 1  # [S, Tk], fractional change per pp
    E = 100 * growth  # [S, Tk], % change per pp
    # AME_t: expected cases per county-year in year t associated with +1pp
    ame = np.stack(
        [
            (mu[:, row_years == yr] * g[:, None]).mean(axis=1)
            for yr, g in zip(yk, growth.T)
        ],
        axis=1,
    )  # [S, Tk], cases per pp
    mubar = mu[:, row_keep].mean(axis=1)  # [S], cases per county-year

    sA, E_first, E_last = slope_and_ends(E)
    sB, ame_first, ame_last = slope_and_ends(ame)
    # same trend fitted on the log rate-ratio scale: exp(slope)-1 is the
    # compounded % change per year in the rate ratio per +1pp
    b_log, lb_first, lb_last = slope_and_ends(beta[:, keep])
    Elog_first = 100 * (np.exp(lb_first) - 1)  # % per pp
    Elog_last = 100 * (np.exp(lb_last) - 1)
    f, l = yk[0], yk[-1]
    out = {
        "1a slope of %-effect (pp/yr)": sA,
        "1b % change per year in rate ratio (log scale)": 100
        * (np.exp(b_log) - 1),
        "2a A basis: sA*mubar/100 (cases/pp/yr)": mubar * sA / 100,
        "2b B basis: slope of AME (cases/pp/yr)": sB,
        "2c B minus A (cases/pp/yr)": sB - mubar * sA / 100,
    }
    # endpoints: (label, effect % per pp at f, at l, AME cases per pp at f, at l)
    for label, e0, e1, a0, a1 in [
        ("3 %-scale line", E_first, E_last, ame_first, ame_last),
        ("4 log-scale line", Elog_first, Elog_last, None, None),
        ("5 raw yearly", E[:, 0], E[:, -1], ame[:, 0], ame[:, -1]),
    ]:
        out[f"{label}: effect {f} (% per pp)"] = e0
        out[f"{label}: effect {l} (% per pp)"] = e1
        out[f"{label}: effect diff (pp)"] = e1 - e0
        out[f"{label}: A-basis cases {f}"] = mubar * e0 / 100
        out[f"{label}: A-basis cases {l}"] = mubar * e1 / 100
        out[f"{label}: A-basis cases diff"] = mubar * (e1 - e0) / 100
        if a0 is not None:
            out[f"{label}: B-basis cases {f}"] = a0
            out[f"{label}: B-basis cases {l}"] = a1
            out[f"{label}: B-basis cases diff"] = a1 - a0
    # Spearman rho between each draw's yearly effect and the calendar year:
    # how monotone the trend is (the posterior carries the uncertainty; a
    # p-value from the median curve is not used because the random-walk
    # prior makes neighboring years dependent)
    ranks = stats.rankdata(E, axis=1)
    ranks = ranks - ranks.mean(axis=1, keepdims=True)
    year_ranks = stats.rankdata(yk) - stats.rankdata(yk).mean()
    out["6 Spearman rho, effect vs year (per draw)"] = (
        ranks @ year_ranks
    ) / np.sqrt((ranks**2).sum(axis=1) * (year_ranks**2).sum())
    out["mubar (cases per county-year)"] = mubar
    return out


def _toy_checks():
    years = np.arange(2011, 2026)
    # 1. E falls linearly 0 -> -14% (1pp/yr); mu = 50 everywhere
    E = -(years - 2011.0)
    beta = np.log(1 + E / 100)[None, :]
    row_years = np.repeat(years, 2)
    out = trend_draws(beta, np.full((1, 30), 50.0), row_years, years)
    assert np.isclose(out["1a slope of %-effect (pp/yr)"][0], -1.0)
    assert np.isclose(out["2a A basis: sA*mubar/100 (cases/pp/yr)"][0], -0.5)
    assert np.isclose(out["3 %-scale line: effect 2025 (% per pp)"][0], -14.0)
    assert np.isclose(out["3 %-scale line: A-basis cases diff"][0], -7.0)
    assert np.isclose(
        out["6 Spearman rho, effect vs year (per draw)"][0], -1.0
    )
    # 1b. Spearman rho agrees with scipy on an arbitrary (non-monotone) curve
    E_rand = np.random.default_rng(0).uniform(-20, 20, (1, 15))
    out = trend_draws(
        np.log(1 + E_rand / 100), np.full((1, 30), 50.0), row_years, years
    )
    assert np.isclose(
        out["6 Spearman rho, effect vs year (per draw)"][0],
        stats.spearmanr(years, E_rand[0])[0],
    )
    # 2. confound: E fixed at -10%, expected outbreak size grows 10 -> 100
    beta = np.full((1, 15), np.log(0.9))
    mu_t = np.linspace(10, 100, 15)
    out = trend_draws(beta, np.repeat(mu_t, 2)[None, :], row_years, years)
    assert np.isclose(out["1a slope of %-effect (pp/yr)"][0], 0.0, atol=1e-9)
    assert np.isclose(
        out["2b B basis: slope of AME (cases/pp/yr)"][0], -0.1 * 90 / 14
    )
    # 3. log-scale: beta falls 0.02 per year; mu = 50 everywhere
    beta = (-0.02 * (years - 2011.0))[None, :]
    out = trend_draws(beta, np.full((1, 30), 50.0), row_years, years)
    assert np.isclose(
        out["1b % change per year in rate ratio (log scale)"][0],
        100 * (np.exp(-0.02) - 1),
    )
    assert np.isclose(
        out["4 log-scale line: A-basis cases 2025"][0],
        50 * (np.exp(-0.28) - 1),
    )
    print(
        "toy checks passed (-1pp/yr -> A=-0.5; confound sB=-0.643, sA=0; log slope -0.02 -> -1.98%/yr)"
    )


def print_effect_trend(samples_1617, samples_2025, df_1617, df_2025):
    _toy_checks()
    years = np.arange(2011, 2026)
    per_fit = {}
    for name, samples, d in [
        ("without 2017 MN", samples_1617, df_1617),
        ("full data", samples_2025, df_2025),
    ]:
        std = d["estimated_mmr_cov"].to_numpy().std()
        beta = np.asarray(samples["effects_over_time"][:, ::-1, 0]) / std
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
            name: trend_draws(*args[:2], args[2], years, drop)
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
    with open(
        os.path.join(
            output_dir,
            "predictive_random_walk_poisson_CAR_time_and_group_varying_2025_oddities=True_exclude_16_17.pkl",
        ),
        "rb",
    ) as file:
        predictive_samples_1617 = pickle.load(file)

    with open(
        os.path.join(
            output_dir,
            "predictive_random_walk_poisson_CAR_time_and_group_varying_2025_oddities=True.pkl",
        ),
        "rb",
    ) as file:
        predictive_samples_2025 = pickle.load(file)

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
