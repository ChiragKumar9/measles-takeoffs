import numpy as np
import scipy as sp


def final_outbreak_size(R0: float, s0: float) -> float:
    """
    Calculate the final outbreak size using the final size equation.

    Parameters
    ----------
    R0 : float
        Basic reproduction number.
    s0 : float
        Initial susceptible fraction of the population.

    Returns
    -------
    float
        Final outbreak size (fraction of individuals infected during the outbreak).
    """
    # If below herd immunity threshold, no outbreak possible
    if s0 <= 1 / R0:
        return 0.0

    # s_f = s0 * exp(-R0 * (s0 - s_f))
    # s0 - s_f = final outbreak size (fraction of population)
    # let z = s0 - s_f
    # then the equation becomes:
    # s0 - z = s0 * exp(-R0 * z)
    # 0 = s0 * exp(-R0 * z) + z - s0

    def final_size_residual(z):
        return s0 * np.exp(-R0 * z) + z - s0

    # Bracket is (0, s0] -- residual changes sign across this interval
    # when s0 > 1/R0
    z_final = sp.optimize.brentq(final_size_residual, 1e-10, s0)

    return min(max(float(z_final), 0), 1.0)
