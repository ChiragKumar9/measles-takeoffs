import numpy as np
from scipy import stats
from scipy.special import logsumexp


def log_likelihood(cases, mu_draws, concentration_draws=None):
    """The log-probability of each location's cases under each posterior draw:
    a Poisson with the draw's mean, or, when `concentration_draws` is given, a
    negative binomial with that mean and the draw's concentration (variance =
    mean + mean**2 / concentration, the model of outbreak_size.poisson with
    likelihood="negative_binomial").

    cases: [n_locations]; mu_draws: [n_draws, n_locations];
    concentration_draws: [n_draws]. Returns [n_draws, n_locations].
    """
    cases = np.asarray(cases).reshape(1, -1)
    mu = np.asarray(mu_draws)
    if concentration_draws is None:
        return stats.poisson.logpmf(cases, mu)
    concentration = np.asarray(concentration_draws).reshape(-1, 1)
    # scipy's nbinom(n, p) has mean n * (1 - p) / p: n = concentration and
    # p = concentration / (concentration + mean) give this model's mean
    return stats.nbinom.logpmf(
        cases, concentration, concentration / (concentration + mu)
    )


def waic_posterior(cases, mu_draws, concentration_draws=None):
    """WAIC (= -2 * elpd, lower is better) of a Poisson model (or of a negative
    binomial one, if `concentration_draws` is given) from posterior draws of
    its mean.

    cases: [n_locations]; mu_draws: [n_draws, n_locations];
    concentration_draws: [n_draws] or None.
    Returns (waic, p_waic), where p_waic is the effective number of parameters.
    """
    log_lik = log_likelihood(
        cases, mu_draws, concentration_draws
    )  # [n_draws, n_locations]
    lppd = (logsumexp(log_lik, axis=0) - np.log(log_lik.shape[0])).sum()  # type: ignore
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
