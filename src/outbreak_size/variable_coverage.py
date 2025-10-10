import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import polars as pl
from jax.scipy.optimize import minimize


class WithinCountyVariance:
    def __init__(
        self,
        coverage_fraction: jnp.ndarray,
        population: jnp.ndarray,
        variance_covariates: jnp.ndarray,
        groups: pl.Series,
        baseline=dist.Laplace(0, 1),
        priors=dist.Laplace(0, 1),
        group_fixed_effect=True,
        r0_prior=dist.LogNormal(jnp.log(15), jnp.log((18 / 12) ** 0.5)),
    ):
        self.coverage_fraction = coverage_fraction
        self.uncovered_fraction = 1.0 - coverage_fraction
        self.num_samples = coverage_fraction.shape[0]
        self.population = population
        self.variance_covariates = variance_covariates
        self.num_variance_covariates = variance_covariates.shape[1]
        # turn groups into one-hot encoded features
        self.groups = groups.cast(pl.Categorical).to_physical()
        self.num_groups = max(self.groups) + 1
        self.groups = self.groups.to_jax()
        if group_fixed_effect:
            self.num_variance_covariates += self.num_groups
            self.variance_covariates = jnp.hstack(
                [
                    self.variance_covariates,
                    jnp.eye(self.num_groups)[self.groups],
                ]
            )
        self.betas = priors
        self.baseline = baseline
        self.r0_prior = r0_prior

    def fit(self, cases=None):
        log_baseline_concentration = numpyro.sample(
            "log_baseline_concentration", self.baseline
        )
        with numpyro.plate("covariates", self.num_variance_covariates):
            regression_betas = numpyro.sample("regression_betas", self.betas)  # type: ignore
        covariate_effect = jnp.dot(self.variance_covariates, regression_betas)

        phi = log_baseline_concentration + covariate_effect
        phi = jnp.exp(phi)
        numpyro.deterministic("concentration", phi)

        # phi is a concentration parameter, it is alpha + beta in the convention beta
        # so alpha = coverage_fraction * phi, beta = (1 - coverage_fraction) * phi
        # Numpyro's BetaProportion does exactly this under the hood, but I do it
        # explicitly here so I can understand what's going on
        # and so it's easier to add a meaningful prior
        coverage_distribution = dist.Beta(
            1.0 + self.coverage_fraction * phi,
            1.0 + self.uncovered_fraction * phi,
        )

        with numpyro.plate("inferring_coverage", self.num_samples):
            inferred_coverage = numpyro.sample(
                "inferred_coverage", coverage_distribution
            )

        # Now we do the semi-mechanistic outbreak size inference
        # We need some estimate of R0
        r0 = numpyro.sample("r0", self.r0_prior)  # type: ignore

        # Now we infer the final outbreak size based on the inferred coverage
        final_size = final_outbreak_size_vmap(r0, 1.0 - inferred_coverage)  # type: ignore
        numpyro.deterministic("final_size", final_size)
        incidence = final_size - inferred_coverage
        # We assume that the outbreak occured in a cluster at that coverage
        proportion = jnp.exp(coverage_distribution.log_prob(inferred_coverage))
        attenuator = numpyro.sample("attenuator", dist.Beta(1, 1))
        incident_infections = (
            incidence * proportion * self.population * attenuator
        )
        incident_infections = incident_infections + jnp.finfo(jnp.float64).eps
        numpyro.deterministic("incident_infections", incident_infections)

        numpyro.sample("cases", dist.Poisson(incident_infections), obs=cases)


@jax.custom_jvp
def final_outbreak_size(r0, s0=1.0):
    # T = final outbreak size = 1 - S(\infty)
    # T = 1 - s0 * exp(-r0 * T)
    # 1 - s0 exp(-r0 * T) - T = 0

    def final_size_residual(t):
        return 1 - s0 * jnp.exp(-r0 * t[0]) - t[0]

    result = minimize(
        lambda t: final_size_residual(t) ** 2,
        x0=jnp.array([0.5]),
        method="BFGS",
    )

    t = result.x[0]
    t = jnp.clip(
        t,
        min=jnp.finfo(jnp.float64).tiny,
        max=1.0 - jnp.finfo(jnp.float64).eps,
    )
    return t


@final_outbreak_size.defjvp
def final_outbreak_size_jvp(primals, tangents):
    """
    Custom JVP rule using implicit differentiation
    We're solving: F(T, r0, s0) = 1 - s0 * exp(-r0 * T) - T = 0
    By implicit function theorem:
    dT/dr0 = -(∂F/∂r0) / (∂F/∂T)
    dT/ds0 = -(∂F/∂s0) / (∂F/∂T)
    """
    r0, s0 = primals
    r0_dot, s0_dot = tangents

    # Get the primal solution
    T = final_outbreak_size(r0, s0)

    # Compute partial derivatives of F(T, r0, s0) = 1 - s0 * exp(-r0 * T) - T
    exp_term = jnp.exp(-r0 * T)

    # ∂F/∂T = s0 * r0 * exp(-r0 * T) - 1
    dF_dT = s0 * r0 * exp_term - 1

    # ∂F/∂r0 = s0 * T * exp(-r0 * T)
    dF_dr0 = s0 * T * exp_term

    # ∂F/∂s0 = -exp(-r0 * T)
    dF_ds0 = -exp_term

    # Apply implicit function theorem
    # dT/dr0 = -(∂F/∂r0) / (∂F/∂T)
    dT_dr0 = -dF_dr0 / dF_dT

    # dT/ds0 = -(∂F/∂s0) / (∂F/∂T)
    dT_ds0 = -dF_ds0 / dF_dT

    # Apply chain rule
    T_dot = dT_dr0 * r0_dot + dT_ds0 * s0_dot

    return T, T_dot


# Vectorized version
final_outbreak_size_vmap = jax.vmap(final_outbreak_size, in_axes=(None, 0))
