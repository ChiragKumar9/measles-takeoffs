import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import polars as pl


class PartialPooledPoisson:
    def __init__(
        self,
        iv: jnp.ndarray,
        covariates: jnp.ndarray,
        groups: pl.Series,
        baseline=None,
        priors=None,
        group_fixed_effect=True,
    ):
        self.iv = iv
        self.num_ivs = iv.shape[1]
        self.covariates = covariates
        self.num_covariates = covariates.shape[1]
        # turn groups into an index
        self.groups = groups.cast(pl.Categorical).to_physical()
        self.num_groups = max(self.groups) + 1
        self.groups = self.groups.to_jax()
        if group_fixed_effect:
            self.num_covariates += self.num_groups
            self.covariates = jnp.hstack(
                [
                    self.covariates,
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
        baseline = numpyro.sample("baseline", self.baseline)
        with numpyro.plate("covariates", self.num_covariates):
            betas = numpyro.sample("betas", self.betas)  # type: ignore
        covariate_effect = jnp.dot(self.covariates, betas)

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
            baseline
            + covariate_effect
            + jnp.sum(iv_effect[self.groups] * self.iv, axis=1)  # type: ignore
        )
        mu = jnp.exp(mu)
        mu = jnp.clip(mu, min=jnp.finfo(jnp.float64).tiny)

        gate = 0.0  # placeholder for now
        gate = jnp.clip(
            gate, jnp.finfo(jnp.float64).tiny, 1.0 - jnp.finfo(jnp.float64).eps
        )

        numpyro.sample(
            "cases",
            dist.ZeroInflatedPoisson(gate, mu),  # type: ignore
            obs=cases,
        )
        return self
