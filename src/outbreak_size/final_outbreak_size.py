import numpy as np
import scipy as sp


def final_outbreak_size(R0: float, s0: float) -> float:
    """
    Calculate the final outbreak size using the final size equation.

    Parameters
    ----------
    R0 : floattexttext
        Basic reproduction number.
    initial_infected : int
        Initial number of infected individuals.
    population : int
        Total population size.

    Returns
    -------
    float
        Final outbreak size (fraction of individuals infected during the outbreak).
    """

    # s_f = s0 * exp(-R0 * (s0 - s_f))
    # s0 - s_f = final outbreak size (fraction of population)
    # let z = s0 - s_f
    # then the equation becomes:
    # s0 - z = s0 * exp(-R0 * z)
    # 0 = s0 * exp(-R0 * z) + z - s0

    def final_size_residual(z):
        return s0 * np.exp(-R0 * z) + z - s0

    # Initial guess for the final size (as a fraction of the population)
    z0 = [0.5]

    # Solve the final size equation
    z_final = sp.optimize.minimize(
        lambda z: final_size_residual(z) ** 2, x0=z0, method="L-BFGS-B"
    ).x[0]

    # Calculate the final outbreak size

    return min(max(float(z_final), 0), 1.0)
