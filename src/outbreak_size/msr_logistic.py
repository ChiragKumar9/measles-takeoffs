"""
Spatially constrained permutation test of one coefficient of a logistic
regression, for an outcome that is 0 or 1 for each county-year (for example,
whether a county is part of a multi-county outbreak).

`regression_msr_multiple_blocked` (outbreak_size.msr) reassigns the residuals
of a continuous outcome between counties. A 0/1 outcome has no residuals to add
back to fitted values, so this test reassigns the tested predictor instead
(Freedman and Lane's scheme applied to the predictor): the part of the
predictor that the other predictors do not explain is reassigned between the
counties by the same Moran spectral randomization (`msr_county_permutation`,
whole blocks of a county's observations, `_block_permuted_outcome`) and added
back to its fitted values, and the logistic model is refitted. The statistic is
the Wald z of the tested coefficient, so the null does not depend on the scale
of the predictor. The null keeps the effect of the other predictors and the
spatial pattern of what they leave unexplained in the tested predictor, and
breaks only its relation to the outcome. Its calibration is checked by simulating
outcomes in which the tested predictor has no effect (as
analysis/check_msr_regression_calibration.py).

The random draws use numpy's global state; the test seeds it.
"""

import numpy as np
import statsmodels.api as sm

from outbreak_size.msr import (
    _block_permuted_outcome,
    _count_counties,
    calculate_moran_eigenvectors_adjacency,
    msr_county_permutation,
)


def logistic_fit(design, y, ridge=1e-6, iterations=60):
    """
    Logistic regression of y (0/1) on design (its first column is the
    intercept) by Newton's method (iteratively reweighted least squares). A
    tiny ridge keeps a separated fit finite.
    Returns (coefficients, covariance matrix, converged).
    """
    beta = np.zeros(design.shape[1])
    for _ in range(iterations):
        eta = np.clip(design @ beta, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        w = np.maximum(p * (1 - p), 1e-9)
        hessian = design.T @ (design * w[:, None]) + ridge * np.eye(
            design.shape[1]
        )
        step = np.linalg.solve(hessian, design.T @ (y - p) - ridge * beta)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            return beta, np.linalg.inv(hessian), True
    return beta, np.linalg.inv(hessian), False  # type: ignore


def _standardized_design(X):
    """The intercept and the columns of X centered and divided by their SD (a constant column is only centered)."""
    center = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale == 0] = 1.0
    return (
        np.column_stack([np.ones(len(X)), (X - center) / scale]),
        center,
        scale,
    )


def logistic_msr_test(
    X,
    y,
    county_indices,
    adjacency_matrix,
    *,
    tested,
    n_permutations=9999,
    seed=108,
    clusters=None,
):
    """
    Test of coefficient `tested` of the logistic regression of y on X.

    Parameters:
    -----------
    X : array, shape (n_obs, p)
        Predictors without the intercept (a product term is a column of X). They are standardized inside the fit; the
        coefficients returned are per unit of each column of X.
    y : array, shape (n_obs,)
        0/1 outcome
    county_indices : array, shape (n_obs,)
        County of each observation, numbered 0, 1, ..., n_counties - 1 (the rows of the adjacency matrix)
    adjacency_matrix : array, shape (n_counties, n_counties)
        Symmetric county adjacency
    tested : int
        Column of X to test (0 is the first column)
    n_permutations : int
    seed : int, optional
    clusters : array, shape (n_obs,), optional
        Cluster of each observation (for example the outbreak): the p value of a Wald test with standard errors
        clustered by it is returned beside the permutation p value

    Returns:
    --------
    dict with 'coef' and 'se' (per unit of each column of X, in the order of X, intercept first),
    'cov' (covariance matrix of the coefficients of the columns of X, per unit of X, no intercept),
    'z' (Wald z of the tested coefficient), 'p_value' (two-sided, Phipson-Smyth), 'z_null',
    'p_clustered' (with clusters; nan if the clustered fit fails), 'mcfadden_r2' (1 - log likelihood / log likelihood
    of the model with only an intercept),
    'separated' (the fit has a coefficient above 15 in absolute value on the standardized scale, or did not converge)
    """
    if seed is not None:
        np.random.seed(seed)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    county_indices = np.asarray(county_indices, dtype=int)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if not (len(X) == len(y) == len(county_indices)):
        raise ValueError("X, y and county_indices must have the same length")
    if not set(np.unique(y)) <= {0.0, 1.0}:
        raise ValueError("y must be 0 or 1")
    if not 0 <= tested < X.shape[1]:
        raise ValueError(
            f"tested must be a column of X (0 to {X.shape[1] - 1})"
        )
    n_counties = _count_counties(county_indices, adjacency_matrix)

    design, center, scale = _standardized_design(X)
    j = tested + 1  # the intercept is column 0 of the design
    beta, cov, converged = logistic_fit(design, y)
    z_obs = beta[j] / np.sqrt(cov[j, j])

    eigenvectors, _ = calculate_moran_eigenvectors_adjacency(
        adjacency_matrix, n_vectors=n_counties
    )
    county_masks = [county_indices == i for i in range(n_counties)]
    reduced = np.delete(design, j, axis=1)
    fitted = reduced @ (np.linalg.pinv(reduced) @ design[:, j])
    residual = design[:, j] - fitted
    county_values = np.array([residual[mask].mean() for mask in county_masks])
    coefficients = eigenvectors.T @ (county_values - county_values.mean())

    z_null = np.empty(n_permutations)
    permuted = design.copy()
    for k in range(n_permutations):
        county_perm = msr_county_permutation(
            eigenvectors, coefficients, county_values
        )
        permuted[:, j] = fitted + _block_permuted_outcome(
            residual, county_masks, county_perm
        )
        b, c, _ = logistic_fit(permuted, y)
        z_null[k] = b[j] / np.sqrt(c[j, j])
    p_value = (1 + int((np.abs(z_null) >= abs(z_obs)).sum())) / (
        1 + n_permutations
    )

    # back to the units of X: the coefficient of column c is beta / scale; the intercept absorbs the centering
    coef = np.concatenate(
        [[beta[0] - np.sum(beta[1:] * center / scale)], beta[1:] / scale]
    )
    se = np.concatenate([[np.nan], np.sqrt(np.diag(cov))[1:] / scale])

    p_clustered = np.nan
    if clusters is not None:
        try:
            fit = sm.GLM(y, design, family=sm.families.Binomial()).fit(
                cov_type="cluster",
                cov_kwds={
                    "groups": np.unique(
                        np.asarray(clusters).astype(str), return_inverse=True
                    )[1]
                },
            )
            p_clustered = float(fit.pvalues[j])
        except Exception:  # a separated fit has no usable covariance
            pass

    probability = np.clip(
        1 / (1 + np.exp(-np.clip(design @ beta, -30, 30))), 1e-12, 1 - 1e-12
    )
    n_events, n = y.sum(), len(y)
    log_likelihood = np.sum(
        y * np.log(probability) + (1 - y) * np.log(1 - probability)
    )
    null_log_likelihood = n_events * np.log(n_events / n) + (
        n - n_events
    ) * np.log(1 - n_events / n)
    return dict(
        coef=coef,
        se=se,
        cov=cov[1:, 1:] / np.outer(scale, scale),
        z=float(z_obs),
        p_value=float(p_value),
        z_null=z_null,
        p_clustered=p_clustered,
        mcfadden_r2=float(1 - log_likelihood / null_log_likelihood),
        separated=bool(np.max(np.abs(beta)) > 15 or not converged),
    )
