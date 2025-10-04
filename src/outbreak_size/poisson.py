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

    def fit(self, cases):
        baseline = numpyro.sample("baseline", self.baseline)
        with numpyro.plate("covariates", self.num_covariates):
            betas = numpyro.sample("betas", self.betas)  # type: ignore
        covariate_effect = jnp.dot(self.covariates, betas)

        global_vacc_coverage = numpyro.sample(
            "global_vacc_coverage", dist.Normal(0, 1)
        )
        county_level_vacc_coverage_variability = numpyro.sample(
            "county_level_vacc_coverage_variability", dist.HalfNormal(0.25)
        )
        with numpyro.plate("counties", self.num_counties):
            county_vacc_coverage_deviation = numpyro.sample(
                "county_offset", dist.Normal(0, 1)
            )
        vaccine_coverage_effect = (
            global_vacc_coverage
            + county_level_vacc_coverage_variability
            * county_vacc_coverage_deviation
        )

        mu = (
            baseline
            + covariate_effect
            + vaccine_coverage_effect[self.counties] * self.iv  # type: ignore
        )
        mu = jnp.exp(mu)
        mu = jnp.clip(mu, min=jnp.finfo(jnp.float64).tiny)

        numpyro.sample("cases", dist.Poisson(mu), obs=cases)
        return self
