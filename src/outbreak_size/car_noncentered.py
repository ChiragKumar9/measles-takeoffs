"""
An exact non-centred form of numpyro's CAR distribution, for the sampler.

numpyro's CAR(loc, correlation=c, conditional_precision=1, adj_matrix=W) is a
Normal(loc, inverse(D - c W)) with D = diag(row sums of W). Its prior depends
on c, so c and the field it describes are strongly tied together in the
posterior and NUTS moves c slowly (a funnel). With B = D^-1/2 W D^-1/2 =
V diag(lam) V^T (computed once from W), the field

    x = loc + D^-1/2 V diag((1 - c lam)^-1/2) z,    z ~ Normal(0, I),

has exactly the same distribution, because its covariance is
D^-1/2 V (I - c Lam)^-1 V^T D^-1/2 = (D^1/2 (I - c B) D^1/2)^-1 = (D - c W)^-1.
The sampler then works on z, whose prior does not depend on c. The model does
not change: same prior, same likelihood, same parameters.

The field keeps its name in the draws: the sampled site is
"<name>_decentered" (z) and "<name>" (x) is stored as a deterministic site.
"""

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer.reparam import Reparam


class CARNonCentered(Reparam):
    """
    Reparametrizes a CAR site of a numpyro model through white noise:
    numpyro.handlers.reparam(model, config={"site_name": CARNonCentered(W)}).
    `adjacency` is the symmetric adjacency matrix (dense or scipy sparse) of
    the units of that site, each with at least one neighbour; it must be the
    matrix the CAR distribution is given.
    """

    def __init__(self, adjacency):
        weights = np.asarray(
            adjacency.toarray()
            if hasattr(adjacency, "toarray")
            else adjacency,
            dtype=float,
        )
        degree = weights.sum(axis=1)
        if (degree <= 0).any():
            raise ValueError(
                "every unit needs a neighbour (the models leave the units "
                "without one out of the CAR term)"
            )
        scaled = weights / np.sqrt(np.outer(degree, degree))
        eigenvalues, eigenvectors = np.linalg.eigh(scaled)
        self.num_units = weights.shape[0]
        self.eigenvectors = jnp.asarray(eigenvectors)
        self.eigenvalues = jnp.asarray(eigenvalues)
        self.inverse_sqrt_degree = jnp.asarray(1.0 / np.sqrt(degree))

    def transform(self, white_noise, correlation):
        """The CAR field (with loc 0, conditional precision 1) of the
        white noise `white_noise` (one entry per unit)."""
        return self.inverse_sqrt_degree * (
            self.eigenvectors
            @ (white_noise * (1.0 - correlation * self.eigenvalues) ** -0.5)
        )

    def __call__(self, name, fn, obs):
        if obs is not None:
            raise ValueError("an observed CAR site cannot be reparametrized")
        white_noise = numpyro.sample(
            f"{name}_decentered",
            dist.Normal(jnp.zeros(self.num_units), 1.0).to_event(1),  # type: ignore
        )
        return None, fn.loc + self.transform(white_noise, fn.correlation)
