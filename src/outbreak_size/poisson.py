import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import polars as pl
from jax.scipy.optimize import minimize
from jax.scipy.special import expit


class PartialPooledPoisson:
    def __init__(
        self,
        iv: jnp.ndarray,
        regression_covariates: jnp.ndarray,
        classification_covariates: jnp.ndarray,
        population: jnp.ndarray,
        coverage: jnp.ndarray,
        groups: pl.Series,
        baseline=None,
        priors=None,
        group_fixed_effect=True,
    ):
        self.iv = iv
        self.num_ivs = iv.shape[1]
        self.regression_covariates = regression_covariates
        self.num_regression_covariates = regression_covariates.shape[1]
        self.classification_covariates = classification_covariates
        self.num_classification_covariates = classification_covariates.shape[1]
        self.population = population
        self.coverage = coverage
        # turn groups into an index
        self.groups = groups.cast(pl.Categorical).to_physical()
        self.num_groups = max(self.groups) + 1
        self.groups = self.groups.to_jax()
        if group_fixed_effect:
            self.num_regression_covariates += self.num_groups
            self.regression_covariates = jnp.hstack(
                [
                    self.regression_covariates,
                    jnp.eye(self.num_groups)[self.groups],
                ]
            )
        if priors is None:
            self.betas = dist.Laplace(0, 1)
        else:
            self.betas = priors
        if baseline is None:
            self.baseline = dist.Laplace(0, 1)
        else:
            self.baseline = baseline

    def fit(self, cases=None):
        baseline_rate = numpyro.sample("baseline_rate", self.baseline)
        with numpyro.plate(
            "regression_covariates", self.num_regression_covariates
        ):
            regression_betas = numpyro.sample("regression_betas", self.betas)  # type: ignore
        covariate_effect = jnp.dot(
            self.regression_covariates, regression_betas
        )

        with numpyro.plate("iv", self.num_ivs):
            iv_global = numpyro.sample("iv_global", dist.Laplace(0, 1))
            iv_group_variabilities = numpyro.sample(
                "iv_group_variabilities", dist.HalfNormal(0.25)
            )
            with numpyro.plate("groups", self.num_groups):
                iv_group_offsets = numpyro.sample(
                    "iv_group_offsets", dist.Laplace(0, 1)
                )

        iv_effect = iv_group_variabilities * iv_group_offsets + iv_global

        mu = (
            baseline_rate
            + covariate_effect
            + jnp.sum(iv_effect[self.groups] * self.iv, axis=1)  # type: ignore
        )
        mu = jnp.exp(mu)
        mu = jnp.clip(mu, min=jnp.finfo(jnp.float64).tiny)

        baseline_logit = numpyro.sample("baseline_logit", self.baseline)
        with numpyro.plate(
            "classification_covariates", self.num_classification_covariates
        ):
            classification_betas = numpyro.sample(
                "classification_betas", self.betas
            )

        # gate = baseline_logit + jnp.dot(self.classification_covariates, classification_betas)
        # gate = expit(gate)
        gate = 0.0  # placeholder for now
        gate = jnp.clip(
            gate, jnp.finfo(jnp.float64).tiny, 1.0 - jnp.finfo(jnp.float64).eps
        )

        s0 = (100 - self.coverage) / 100.0
        mu = final_outbreak_size_vmap_both(mu, s0)  # type: ignore
        mu = (mu - s0) * self.population

        numpyro.sample(
            "cases",
            dist.ZeroInflatedPoisson(gate, mu),  # type: ignore
            obs=cases,
        )


@jax.custom_jvp
def final_outbreak_size(r0, s0=1.0):
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


# @jax.vmap
# def final_outbreak_size_vectorized(r0, s0=1.0):
#     """Vectorized version to solve for multiple R0 values at once"""
#     return final_outbreak_size(r0, s0)


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


# Vectorized versions
final_outbreak_size_vmap_both = jax.vmap(final_outbreak_size, in_axes=(0, 0))
