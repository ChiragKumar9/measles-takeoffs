import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import polars as pl


class PartialPooledPoisson:
    def __init__(
        self,
        iv: jnp.ndarray,
        covariates: jnp.ndarray,
        counties: pl.Series,
        baseline=None,
        priors=None,
    ):
        self.iv = iv
        self.num_ivs = iv.shape[1]
        self.covariates = covariates
        self.num_covariates = covariates.shape[1]
        # turn counties into an index
        self.counties = counties.cast(pl.Categorical).to_physical()
        self.num_counties = max(self.counties) + 1
        self.counties = self.counties.to_jax()
        if priors is None:
            self.betas = dist.Normal(0, 1)
        else:
            self.betas = priors
        if baseline is None:
            self.baseline = dist.Normal(0, 1)
        else:
            self.baseline = baseline

    def fit(self, cases=None):
        baseline = numpyro.sample("baseline", self.baseline)
        with numpyro.plate("covariates", self.num_covariates):
            betas = numpyro.sample("betas", self.betas)  # type: ignore
        covariate_effect = jnp.dot(self.covariates, betas)

        with numpyro.plate("iv", self.num_ivs):
            iv_global = numpyro.sample("iv_global", dist.Normal(0, 1))
            iv_variabilities = numpyro.sample(
                "iv_variabilities", dist.HalfNormal(0.25)
            )
            with numpyro.plate("counties", self.num_counties):
                iv_county_offsets = numpyro.sample(
                    "iv_county_offsets", dist.Normal(0, 1)
                )

        iv_effect = iv_variabilities * iv_county_offsets + iv_global

        mu = (
            baseline
            + covariate_effect
            + jnp.sum(iv_effect[self.counties] * self.iv, axis=1)  # type: ignore
        )
        mu = jnp.exp(mu)
        mu = jnp.clip(mu, min=jnp.finfo(jnp.float64).tiny)

        numpyro.sample("cases", dist.Poisson(mu), obs=cases)
        return self
