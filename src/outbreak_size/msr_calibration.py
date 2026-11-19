"""
Pieces of the calibration check of the spatially constrained permutation tests
(analysis/check_msr_calibration.py: a covariate that is independent of the
real outcome but as spatially smooth as the real covariate is;
analysis/check_msr_regression_calibration.py: a 0/1 outcome on which the tested
predictor has no effect, with spatially smooth noise), so that the share of
simulations a test rejects can be compared with its level.
"""

import numpy as np


def morans_i(values, adjacency):
    """Moran's I of values over the counties of a binary adjacency matrix."""
    z = values - values.mean()
    return len(values) / adjacency.sum() * (z @ adjacency @ z) / (z @ z)


def smooth_field(adjacency, smoothness, rng):
    """
    A spatially autocorrelated field over the counties, with mean 0 and
    standard deviation 1: independent normal draws spread over the neighbours
    (a simultaneous autoregression with the row-standardized adjacency).
    smoothness is 0 for no spatial structure, and close to 1 for a very smooth
    field.
    """
    n = adjacency.shape[0]
    row_standardized = adjacency / np.maximum(
        adjacency.sum(axis=1, keepdims=True), 1
    )
    field = np.linalg.solve(
        np.eye(n) - smoothness * row_standardized, rng.standard_normal(n)
    )
    return (field - field.mean()) / field.std()


def smoothness_matching(adjacency, target_i, n_fields=20, seed=0):
    """
    The smoothness at which the fields of `smooth_field` have a mean Moran's I
    of target_i (found by bisection; Moran's I grows with the smoothness).
    Raises if target_i is out of the reach of the fields.
    """

    def mean_i(smoothness):
        rng = np.random.default_rng(seed)
        return np.mean(
            [
                morans_i(smooth_field(adjacency, smoothness, rng), adjacency)
                for _ in range(n_fields)
            ]
        )

    low, high = 0.0, 0.99
    if not mean_i(low) <= target_i <= mean_i(high):
        raise ValueError(
            f"Moran's I of {target_i:.3f} is out of the reach of the fields "
            f"({mean_i(low):.3f} to {mean_i(high):.3f})"
        )
    for _ in range(25):
        middle = (low + high) / 2
        if mean_i(middle) < target_i:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def independent_covariate(
    adjacency, county_indices, smoothness, between_share, rng
):
    """
    A covariate for each observation that has nothing to do with any outcome:
    a county-level smooth field (`smooth_field`) that all the observations of
    a county share, plus noise of each observation. between_share is the share
    of the covariate's variance that is between counties (1 means every
    observation of a county has the same value).
    """
    county_field = smooth_field(adjacency, smoothness, rng)[county_indices]
    noise = rng.standard_normal(len(county_indices))
    return (
        np.sqrt(between_share) * county_field
        + np.sqrt(1 - between_share) * noise
    )


def between_county_share(values, county_indices):
    """
    The share of the variance of values that is between counties: the variance
    of the county means over that plus the mean within-county variance of the
    counties with several observations (1 when none has).
    """
    counties = np.unique(county_indices)
    means = np.array([values[county_indices == c].mean() for c in counties])
    within = [
        values[county_indices == c].var(ddof=1)
        for c in counties
        if (county_indices == c).sum() > 1
    ]
    if not within:
        return 1.0
    return means.var() / (means.var() + np.mean(within))


def rejection_rate(p_values, level):
    """
    The share of p-values below level, and the standard error that share would
    have if the test were calibrated (so that the share is within about two
    standard errors of the level).
    """
    p_values = np.asarray(p_values)
    return np.mean(p_values < level), np.sqrt(
        level * (1 - level) / len(p_values)
    )


def calibration_verdict(rate, level, standard_error):
    """
    "calibrated" when the rejection rate is within two standard errors of the
    level, else "over-rejects" (too many false positives) or "conservative".
    """
    if abs(rate - level) <= 2 * standard_error:
        return "calibrated"
    return "over-rejects" if rate > level else "conservative"


def residual_smoothness(adjacency, residual, county_indices, seed=0):
    """
    The smoothness (`smooth_field`) whose fields have the Moran's I that the
    county means of residual have. Moran's I out of the reach of the fields
    is clamped: a residual with no positive spatial autocorrelation gets 0,
    and one smoother than any field gets 0.99.
    """
    n = adjacency.shape[0]
    county_means = np.bincount(
        county_indices, weights=residual, minlength=n
    ) / np.bincount(county_indices, minlength=n)
    target = morans_i(county_means, adjacency)
    try:
        return smoothness_matching(adjacency, target, seed=seed)
    except ValueError:
        return 0.0 if target <= 0 else 0.99


def logistic_null_outcome(
    adjacency, county_indices, logit, noise_scale, smoothness, rng
):
    """
    A 0/1 outcome for which the tested predictor has no effect: the logit of
    the model without that predictor plus spatially smooth noise (as smooth as
    `smoothness`, shared by the observations of a county, with standard
    deviation noise_scale on the logit scale), then a Bernoulli draw.
    """
    field = smooth_field(adjacency, smoothness, rng)[county_indices]
    probability = 1 / (1 + np.exp(-(logit + noise_scale * field)))
    return (rng.random(len(logit)) < probability).astype(float)


def match_noise_scale(
    adjacency,
    county_indices,
    logit,
    y,
    smoothness,
    scales=(0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0),
    n_draws=30,
    seed=0,
):
    """
    The noise scale (on the logit scale) of `logistic_null_outcome` whose
    outcomes have the between-county variance of the real outcome's residuals:
    the variance, over counties, of the county mean of (outcome - probability
    of the model without the tested predictor), compared at each of `scales`
    over n_draws simulated outcomes. Returns the scale that comes closest.
    Both are residuals from the probability of the model without the noise,
    so that the noise shows in them.
    """
    n = adjacency.shape[0]
    counts = np.bincount(county_indices, minlength=n)
    base = 1 / (1 + np.exp(-logit))

    def county_variance(outcome):
        return (
            np.bincount(county_indices, weights=outcome - base, minlength=n)
            / counts
        ).var()

    target = county_variance(y)
    rng = np.random.default_rng(seed)
    gaps = []
    for scale in scales:
        variances = []
        for _ in range(n_draws):
            field = smooth_field(adjacency, smoothness, rng)[county_indices]
            probability = 1 / (1 + np.exp(-(logit + scale * field)))
            outcome = (rng.random(len(logit)) < probability).astype(float)
            variances.append(county_variance(outcome))
        gaps.append(abs(np.mean(variances) - target))
    return scales[int(np.argmin(gaps))]
