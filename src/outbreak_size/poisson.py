import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import polars as pl
from jax.scipy.special import expit


class PartialPooledPoisson:
    def __init__(
        self,
        iv: jnp.ndarray,
        regression_covariates: jnp.ndarray,
        classification_covariates: jnp.ndarray,
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

        numpyro.sample(
            "cases",
            dist.ZeroInflatedPoisson(gate, mu),  # type: ignore
            obs=cases,
        )
