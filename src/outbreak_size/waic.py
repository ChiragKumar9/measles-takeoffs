import numpy as np
from scipy import stats
from scipy.special import logsumexp


def waic_posterior(cases, mu_draws):
    """WAIC (= -2 * elpd, lower is better) of a Poisson model from posterior
    draws of its mean.

    cases: [n_locations]; mu_draws: [n_draws, n_locations].
    Returns (waic, p_waic), where p_waic is the effective number of parameters.
    """
    log_lik = stats.poisson.logpmf(
        np.asarray(cases).reshape(1, -1), np.asarray(mu_draws)
    )  # [n_draws, n_locations]
    lppd = (logsumexp(log_lik, axis=0) - np.log(log_lik.shape[0])).sum()
    p_waic = log_lik.var(axis=0).sum()
    return -2 * (lppd - p_waic), p_waic


def waic_point_estimate(cases, mu, k):
    """AIC-equivalent of WAIC (= -2 * (log-likelihood - k)) for a Poisson model
    with a fixed mean (no posterior), where k is the number of fitted
    parameters.

    cases, mu: [n_locations].
    """
    log_lik = stats.poisson.logpmf(np.asarray(cases), np.asarray(mu)).sum()
    return -2 * (log_lik - k)
