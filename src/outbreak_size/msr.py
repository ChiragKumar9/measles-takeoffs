"""
Spatially constrained permutation tests for the correlation between a county
outcome and a covariate, built on Moran spectral randomization (MSR).

The Moran eigenvectors of a symmetric adjacency matrix carry the spatial
patterns a field can have. A surrogate field has the outcome's own spectral
coefficients with random signs, so it has the same spatial autocorrelation
(the "singleton" procedure of Wagner and Dray 2015, Methods in Ecology and
Evolution 6:1169, doi: 10.1111/2041-210X.12407). Textbook MSR stops there: one value per site and the
continuous surrogate. The tests here build on it, and the two steps below are
ours, not the published method's:
  * the surrogate only decides which counties swap their outcomes: the county
    with the r-th lowest surrogate value takes the whole block of observations
    of the county with the r-th lowest observed value, so the outcomes the null
    sees are the real ones (a permutation of the outcome, ties and all);
  * counties have several observations (years), and whole blocks are
    exchangeable only if they have the same size, so counties swap only with
    counties that have the same number of observations.
See `msr_county_permutation`. The random draws use numpy's global state, so
the caller seeds it with `np.random.seed`.
"""

import numpy as np
from scipy import stats
from tqdm.auto import tqdm


def calculate_moran_eigenvectors_adjacency(
    adjacency_matrix, n_vectors: int | str = "positive"
):
    """
    Calculate Moran's eigenvector maps from adjacency matrix
    """
    W = np.asarray(adjacency_matrix, dtype=float)
    n = W.shape[0]

    # Binary symmetric weights: the Moran eigenvectors need a symmetric matrix
    # (np.linalg.eigh reads only one triangle of the matrix it is given)
    if not np.array_equal(W, W.T):
        raise ValueError("the adjacency matrix must be symmetric")

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
        n_vectors = int(np.sum(eigenvalues > 1e-10))

    return eigenvectors[:, :n_vectors], eigenvalues[:n_vectors]


def normal_scores(values):
    """
    The normal scores of values: their ranks (the average rank for ties) as
    quantiles of the standard normal. The outcome is mostly zeros with a few
    huge values, and the spatial spectrum of such values is dominated by the
    huge ones; the spectrum of the normal scores describes how the outcome is
    spatially arranged.
    """
    ranks = stats.rankdata(values)
    return stats.norm.ppf((ranks - 0.5) / len(values))


def msr_county_permutation(
    eigenvectors, coefficients, county_values, block_lengths=None
):
    """
    One spatially constrained permutation of the counties. A surrogate
    spatial field is built from the spectral coefficients with random signs
    (so the surrogate has the same spatial autocorrelation as the field they
    come from), then rank-matched: the county with the r-th lowest surrogate
    value takes the block of the county with the r-th lowest observed value
    (`county_values`). Counties with the same observed value (the outcome is
    mostly zeros) are ranked in a random order each time.

    block_lengths gives the number of observations of each county; counties
    only swap with counties of the same number of observations (the ranking is
    done within each group). By default all counties are one group.

    Returns county_perm: county i takes the block of county county_perm[i].
    """
    surrogate = eigenvectors @ (
        coefficients * np.random.choice([-1.0, 1.0], size=len(coefficients))
    )
    ties = np.random.rand(len(county_values))
    lengths = (
        np.zeros(len(county_values))
        if block_lengths is None
        else np.asarray(block_lengths)
    )
    county_perm = np.empty(len(county_values), dtype=int)
    for length in np.unique(lengths):
        group = np.where(lengths == length)[0]
        by_observed = group[np.lexsort((ties[group], county_values[group]))]
        by_surrogate = group[np.argsort(surrogate[group], kind="stable")]
        county_perm[by_surrogate] = by_observed
    return county_perm


def _count_counties(county_indices, adjacency_matrix):
    """
    The number of counties in county_indices, which must number the counties
    0, 1, ..., n - 1 (the rows of the adjacency matrix) with no gap.
    """
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
    return n_counties


def _block_permuted_outcome(y, county_masks, county_perm):
    """
    The outcome after the counties' blocks of observations are swapped: the
    observations of county i (county_masks[i]) take the values of county
    county_perm[i]. A county with fewer observations than the county it takes
    from gets a random sample of them, and one with more gets a random sample
    with replacement (the draws use numpy's global random state).
    """
    n_counties = len(county_masks)
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
    return y_perm


def spearman_msr_blocked(
    x, y, adjacency_matrix, county_indices, n_permutations=99999, seed=108
):
    """
    Spatially constrained block permutation test of the Spearman correlation
    of x and y, for repeated measures (time series per county).

    The null swaps entire blocks of observations between counties, preserving
    within-county temporal structure, and only between counties with the same
    number of observations (so the outcome of a null draw is a permutation of
    the real outcome). Which counties swap follows a surrogate with the
    spatial spectrum of the outcome, so that neighbours tend to swap with
    similar counties (see `msr_county_permutation`).

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
        two-sided p-value
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

    n_counties = _count_counties(county_indices, adjacency_matrix)

    # Observed correlation
    rho_obs, _ = stats.spearmanr(x, y)

    # Get eigenvectors at county level
    # all the eigenvectors (both signs of spatial autocorrelation) carry the
    # spatial spectrum of the county values
    eigenvectors, eigenvalues = calculate_moran_eigenvectors_adjacency(
        adjacency_matrix, n_vectors=n_counties
    )

    # Pre-compute masks for each county (for efficiency)
    county_masks = [county_indices == i for i in range(n_counties)]

    # the county values the swap follows, and the spectrum the surrogates keep
    # (that of the normal scores of the county values)
    county_values = np.array([y[mask].mean() for mask in county_masks])
    scores = normal_scores(county_values)
    coefficients = eigenvectors.T @ (scores - scores.mean())
    county_obs_counts = np.array([np.sum(mask) for mask in county_masks])

    # Check for issues
    if any(count == 0 for count in county_obs_counts):
        raise ValueError(
            "Some counties in adjacency matrix have no observations in data"
        )

    # Permutations
    rho_null = np.zeros(n_permutations)

    for perm_i in tqdm(range(n_permutations)):
        # Generate county-level spatial permutation
        county_perm = msr_county_permutation(
            eigenvectors, coefficients, county_values, county_obs_counts
        )  # New county assignment for each original county

        y_perm = _block_permuted_outcome(y, county_masks, county_perm)

        # Calculate correlation for this permutation
        rho_null[perm_i], _ = stats.spearmanr(x, y_perm)  # type: ignore

    # P-value
    n_extreme = np.sum(np.abs(rho_null) >= np.abs(rho_obs))  # type: ignore
    p_value = (n_extreme + 1) / (n_permutations + 1)

    return rho_obs, p_value, rho_null


def spearman_msr_interval(rho_obs, rho_null):
    """
    The 95% interval of a correlation from its permutation null: rho_obs plus
    and minus the critical value of |rho| in the null. It excludes 0 exactly
    when the two-sided p-value of `spearman_msr_blocked` is at most 0.05 (the
    observed |rho| is above the k-th largest |null| value, k = (n + 1) / 20 - 1
    for n permutations), and its width is that of the null, which keeps the
    spatial structure of the data. It is not the inversion of the test.
    Returns (low, high, critical).
    """
    k = (len(rho_null) + 1) // 20 - 1
    if k < 0:
        raise ValueError(
            "at least 19 permutations are needed for the 5% interval"
        )
    critical = float(np.sort(np.abs(rho_null))[::-1][k])
    return rho_obs - critical, rho_obs + critical, critical


def _t_statistic(pinv, design, y, j):
    """The t statistic of coefficient j of the least-squares fit of y."""
    coef = pinv @ y
    resid = y - design @ coef
    sigma2 = resid @ resid / (len(y) - design.shape[1])
    return coef[j] / np.sqrt(sigma2 * np.sum(pinv[j] ** 2))


def _partial_null_outcome(eigenvectors, fitted, residual, county_masks):
    """
    One outcome under the null that the tested predictor is unrelated to it:
    the fitted values of the model without that predictor plus its residuals,
    whose county blocks are reassigned by a Moran spectral randomization of
    the counties' mean residuals.
    """
    county_values = np.array([residual[mask].mean() for mask in county_masks])
    coefficients = eigenvectors.T @ (county_values - county_values.mean())
    county_perm = msr_county_permutation(
        eigenvectors, coefficients, county_values
    )  # New county assignment for each original county
    return fitted + _block_permuted_outcome(
        residual, county_masks, county_perm
    )


def regression_msr_multiple_blocked(
    X,
    y,
    county_indices,
    adjacency_matrix,
    *,
    tested,
    n_permutations=99999,
    seed=108,
):
    """
    Spatially constrained permutation test of single coefficients of a
    least-squares regression of y on X.

    Each tested coefficient is tested given the other predictors (Freedman and
    Lane): the model without that predictor is fitted, its residuals are
    reassigned between the counties by a Moran spectral randomization
    (`msr_county_permutation`; whole blocks of a county's observations are
    swapped, see `_block_permuted_outcome`) and added back to its fitted
    values, and the full model is refitted to that outcome. The statistic is
    the coefficient's t statistic, so the null does not depend on the scale of
    the outcome or on how much the other predictors explain of it. The null
    keeps the effect of the other predictors and the spatial pattern of what
    they leave unexplained, and breaks only the relation of the tested
    predictor to the outcome.

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
    tested : sequence of int
        The coefficients to test, by their place in the fit (0 is the
        intercept, which cannot be tested; 1 is the first column of X). Each
        one has its own permutations, so list only the ones that are reported.
    n_permutations : int
        Number of permutations
    seed : int, optional
        Random seed

    Returns:
    --------
    results : dict
        'coef': observed coefficients (including intercept)
        't': {j: observed t statistic} for the tested coefficients
        'p_values': {j: two-sided p value} (Phipson-Smyth correction)
        't_null': {j: null t statistics (n_permutations)}
        'r_squared': R² of the full fit
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
    for j in tested:
        if not 1 <= j < n_coef:
            raise ValueError(
                f"coefficient {j} cannot be tested: the fit has {n_coef} "
                "coefficients and coefficient 0 is the intercept"
            )

    n_counties = _count_counties(county_indices, adjacency_matrix)

    # Observed coefficients
    pinv = np.linalg.pinv(X_design)
    coef_obs = pinv @ y

    # R²
    y_pred = X_design @ coef_obs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot

    # Get Moran eigenvectors at county level
    # all the eigenvectors (both signs of spatial autocorrelation) carry the
    # spatial spectrum of the county values
    eigenvectors, eigenvalues = calculate_moran_eigenvectors_adjacency(
        adjacency_matrix, n_vectors=n_counties
    )

    # Pre-compute masks for each county (for efficiency)
    county_masks = [county_indices == i for i in range(n_counties)]

    t_obs = {}
    t_null = {}
    p_values = {}
    for j in tested:
        t_obs[j] = _t_statistic(pinv, X_design, y, j)

        # the model without the tested predictor, and what it leaves
        reduced = np.delete(X_design, j, axis=1)
        fitted = reduced @ (np.linalg.pinv(reduced) @ y)
        residual = y - fitted

        t_null[j] = np.zeros(n_permutations)
        for perm_i in range(n_permutations):
            y_perm = _partial_null_outcome(
                eigenvectors, fitted, residual, county_masks
            )
            t_null[j][perm_i] = _t_statistic(pinv, X_design, y_perm, j)

        # P-value (two-tailed, Phipson-Smyth correction)
        n_extreme = np.sum(np.abs(t_null[j]) >= np.abs(t_obs[j]))
        p_values[j] = (n_extreme + 1) / (n_permutations + 1)

    return {
        "coef": coef_obs,
        "t": t_obs,
        "p_values": p_values,
        "t_null": t_null,
        "r_squared": r_squared,
    }
