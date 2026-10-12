import numpy as np

# the 23 quantile levels of the standard weighted interval score (WIS): the
# median plus 11 central prediction intervals, alpha = 2 * (lower level)
_LOWER_LEVELS = np.array(
    [0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
)


def crps(y, draws):
    """Continuous ranked probability score (CRPS) of each location's
    predictive distribution (the empirical distribution of the draws), in the
    units of y (cases). Lower is better.

    y: [n_locations]; draws: [n_draws, n_locations] predictive draws.
    Returns [n_locations].
    """
    y = np.asarray(y, dtype=float)
    x = np.sort(np.asarray(draws, dtype=float), axis=0)
    n = x.shape[0]
    mean_abs_error = np.abs(x - y).mean(axis=0)  # E|X - y|
    ranks = np.arange(1, n + 1)[:, None]
    # E|X - X'| over all ordered pairs of draws, from the sorted draws
    mean_abs_pair_difference = 2 * ((2 * ranks - n - 1) * x).sum(axis=0) / n**2
    return mean_abs_error - 0.5 * mean_abs_pair_difference


def weighted_interval_score(y, draws):
    """Weighted interval score (WIS, Bracher et al. 2021) of each location's
    predictive distribution, in the units of y (cases). Lower is better.

    y: [n_locations]; draws: [n_draws, n_locations] predictive draws.
    Returns [n_locations].
    """
    y = np.asarray(y, dtype=float)
    lower = np.quantile(draws, _LOWER_LEVELS, axis=0)  # [11, n_locations]
    upper = np.quantile(draws, 1 - _LOWER_LEVELS, axis=0)
    median = np.median(draws, axis=0)
    alpha = (2 * _LOWER_LEVELS)[:, None]
    # (alpha / 2) * interval score of the (1 - alpha) central interval
    weighted_is = (
        (alpha / 2) * (upper - lower)
        + np.maximum(lower - y, 0)
        + np.maximum(y - upper, 0)
    )
    k = len(_LOWER_LEVELS)
    return (0.5 * np.abs(y - median) + weighted_is.sum(axis=0)) / (k + 0.5)
