import numpy as np
import scipy as sp


def _stuttering_chain_size(
    Re: float, N: int, k: float | None = None
) -> tuple[float, float]:
    """
    Expected total outbreak size and its variance for a subcritical branching
    process starting from a single index case.

    Uses:
        E[Z]   = 1 / (1 - Re)
        Var[Z] = sigma2 / (1 - Re)^3

    where sigma2 is the variance of the offspring distribution:
        Poisson(Re):              sigma2 = Re
        NegBin(mean=Re, clump=k): sigma2 = Re + Re^2 / k

    Parameters
    ----------
    Re : float
        Effective reproduction number. Must be strictly less than 1.
    N : int
        Total population size (used to cap E[Z]).
    k : float or None
        Dispersion parameter for negative binomial offspring distribution.
        If None, Poisson offspring are assumed.

    Returns
    -------
    tuple[float, float]
        (expected_size, variance) where expected_size is capped at N.
    """
    expected_size = 1.0 / (1.0 - Re)

    if k is None:
        # Poisson offspring: sigma2 = Re
        sigma2 = Re
    else:
        # Negative binomial offspring: sigma2 = Re + Re^2 / k
        sigma2 = Re + Re**2 / k

    variance = sigma2 / (1.0 - Re) ** 3

    return min(expected_size, N), variance


def final_outbreak_size(
    R0: float, s0: float, N: int, k: float | None = None
) -> tuple[float, float | None]:
    """
    Calculate the expected outbreak size in individuals and its variance.

    For subcritical spread (Re < 1), uses a stuttering chains (branching
    process) model and returns an analytical variance. The caller should use
    this expected size directly rather than scaling it via regression.

    For Re >= 1, uses the final size equation — variance is not computed in
    this regime and is returned as None. The caller should scale the
    deterministic fraction via a Poisson GLM against observed data.

    Parameters
    ----------
    R0 : float
        Basic reproduction number.
    s0 : float
        Initial susceptible fraction of the population (accounts for
        pre-existing immunity, so s0 <= 1). This is S0 / N, not 1.0.
    N : int
        Total population size.
    k : float or None
        Dispersion parameter for negative binomial offspring distribution,
        used in the stuttering chains model. If None, Poisson is assumed.

    Returns
    -------
    tuple[float, float | None]
        (expected_infections, variance). Variance is None for Re >= 1,
        since the final size equation gives a deterministic result.
    """
    Re = R0 * s0

    if Re < 1:
        expected_size, variance = _stuttering_chain_size(Re, N, k=k)
        return expected_size, variance

    # Re >= 1: use final size equation
    # Solve for z = s0 - s_f, the fraction of the population infected,
    # where s_f satisfies: s_f = s0 * exp(-R0 * z)
    # Rearranged: s0 * exp(-R0 * z) + z - s0 = 0
    def final_size_residual(z):
        return s0 * np.exp(-R0 * z) + z - s0

    z_final = sp.optimize.brentq(final_size_residual, 1e-10, s0)
    z_final = min(max(float(z_final), 0.0), 1.0)

    return z_final, None
