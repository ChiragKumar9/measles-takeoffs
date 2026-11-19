"""The risk percentile of units: the same recipe everywhere in the repo."""

import numpy as np
from scipy.stats import rankdata


def percentile_of_scores(scores, method: str = "average") -> np.ndarray:
    """
    The percentile (0 for the lowest score, 100 for the highest) of each unit
    by its score.

    Parameters
    ----------
    scores : array-like, shape (n_units,)
    method : how tied units are ranked (scipy.stats.rankdata); tied units
        share the average rank by default

    Returns
    -------
    np.ndarray, shape (n_units,), values in [0, 100]
    """
    # shift to 0-based so the lowest unit gets exactly 0
    ranks = rankdata(np.asarray(scores, dtype=float), method=method) - 1
    return ranks / ranks.max() * 100


def risk_percentile(samples_2d: np.ndarray, method: str = "average"):
    """
    The risk percentile of each unit: average its draws, then rank the units
    by that average (the percentile of the expected cases, as the scores of
    the table of the model scoring rank the places).

    Parameters
    ----------
    samples_2d : np.ndarray, shape (n_draws, n_units)
    method : how tied units are ranked (scipy.stats.rankdata)

    Returns
    -------
    np.ndarray, shape (n_units,), values in [0, 100]
    """
    return percentile_of_scores(np.mean(samples_2d, axis=0), method)
