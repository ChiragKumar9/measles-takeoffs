import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl
import scipy as sp
from jax.scipy.special import expit, logit


class WithinCountyVarianceRegression:
    def __init__(
        self,
        coverage_fraction: jnp.ndarray,
        exemption_fraction: jnp.ndarray,
        fraction_nonmed_exempt: jnp.ndarray,
        population: jnp.ndarray,
        regression_controls: jnp.ndarray,
        groups: pl.Series,
        spatial_units: pl.Series,
        adjacency_matrix: np.ndarray,
        baseline=dist.Normal(0, 1),
        priors=dist.Normal(0, 1),
        spatial_correlation=dist.Beta(49, 1),
        spatial_precision=dist.HalfNormal(0.25),
    ):
        self.coverage_fraction = coverage_fraction
        self.uncovered_fraction = 1.0 - coverage_fraction
        self.exemption_fraction = exemption_fraction
        self.fraction_nonmed_exempt = fraction_nonmed_exempt
        self.num_samples = coverage_fraction.shape[0]
        self.log_population = jnp.log(population)
        self.regression_controls = regression_controls
        self.num_regression_controls = regression_controls.shape[1]
        # turn groups into one-hot encoded features
        self.groups = groups.cast(pl.Categorical).to_physical()
        self.num_groups = max(self.groups) + 1
        self.groups = self.groups.to_jax()
        self.variance_controls = jnp.eye(self.num_groups)[self.groups]
        self.betas = priors
        self.baseline = baseline
        self.spatial_correlation = spatial_correlation
        self.spatial_precision = spatial_precision
        self.spatial_units = (
            spatial_units.cast(pl.Categorical).to_physical().to_jax()
        )
        self.has_a_neighbor = adjacency_matrix.sum(axis=1) > 0
        # make the adjacency matrix only include counties with neighbors
        self.adjacency_matrix = adjacency_matrix[self.has_a_neighbor][
            :, self.has_a_neighbor
        ]
        self.adjacency_matrix = sp.sparse.csr_matrix(self.adjacency_matrix)

    def fit(self, cases=None):
        log_baseline_concentration = numpyro.sample(
            "log_baseline_concentration", self.baseline
        )

        log_variance_coverage_effect = numpyro.sample(
            "log_variance_coverage_effect", self.betas
        )  # type: ignore
        log_variance_exemption_effect = numpyro.sample(
            "log_variance_exemption_effect", self.betas
        )  # type: ignore
        log_variance_nonmed_exempt_effect = numpyro.sample(
            "log_variance_nonmed_exempt_effect", self.betas
        )  # type: ignore
        log_variance_population_effect = numpyro.sample(
            "log_variance_population_effect", self.betas
        )  # type: ignore

        with numpyro.plate("variance_controls", self.num_groups):
            variance_control_betas = numpyro.sample(
                "variance_control_betas", self.betas
            )  # type: ignore
        variance_control_effect = jnp.dot(
            self.variance_controls, variance_control_betas
        )

        phi = (
            log_baseline_concentration
            + log_variance_coverage_effect * self.coverage_fraction
            + log_variance_exemption_effect * self.exemption_fraction
            + log_variance_nonmed_exempt_effect * self.fraction_nonmed_exempt
            + log_variance_population_effect * self.log_population
            + variance_control_effect
        )
        phi = jnp.exp(phi)
        phi = phi + jnp.finfo(jnp.float64).eps
        numpyro.deterministic("concentration", phi)

        # logit-normal approximation to Beta distribution
        mean = self.coverage_fraction
        mean_inv_mean = mean * (1 - mean)
        logit_mean = logit(mean)
        # Delta method says that
        # Var(logit X) ~= Var(X) / mean_inv_mean^2
        # for us, this becomes mean_inv_mean / (phi + 1) all over mean_inv_mean^2
        logit_std = jnp.sqrt(1.0 / (mean_inv_mean * (phi + 1)))

        with numpyro.plate("inferring_coverage", self.num_samples):
            raw_coverage = numpyro.sample(
                "raw_coverage", dist.Normal(0.0, 1.0)
            )
            logit_coverage = logit_mean + logit_std * raw_coverage
            inferred_coverage = numpyro.deterministic(
                "inferred_coverage", expit(logit_coverage)
            )

        # Now we plug this into a regression to predict outbreak size
        with numpyro.plate(
            "regression_controls", self.num_regression_controls
        ):
            regression_betas = numpyro.sample("regression_betas", self.betas)  # type: ignore
        regression_controls_effect = jnp.dot(
            self.regression_controls, regression_betas
        )
        inferred_coverage_effect = numpyro.sample(
            "inferred_coverage_effect", self.betas
        )  # type: ignore
        baseline = numpyro.sample("baseline", self.baseline)
        mu = (
            self.log_population
            + baseline
            + regression_controls_effect
            + inferred_coverage_effect * inferred_coverage
        )
        mu = jnp.exp(mu)
        mu = mu + jnp.finfo(jnp.float64).eps

        # add spatial correlation
        correlation = numpyro.sample(
            "spatial_correlation", self.spatial_correlation
        )
        tau = numpyro.sample("spatial_precision", self.spatial_precision)
        car_distribution = dist.CAR(
            loc=0,
            correlation=correlation,  # type: ignore
            conditional_precision=1.0,  # type: ignore
            adj_matrix=self.adjacency_matrix,  # type: ignore
            is_sparse=True,
        )
        spatial_effect = numpyro.sample("spatial_effect", car_distribution)
        spatial_effect = tau * spatial_effect
        # spatial_effect is a vector of length of number of counties
        # need to expand spatial effects also to include counties without neighbors
        full_spatial_effect = jnp.zeros(self.has_a_neighbor.shape[0])
        full_spatial_effect = full_spatial_effect.at[self.has_a_neighbor].set(
            spatial_effect
        )
        full_spatial_effect = full_spatial_effect[self.spatial_units]  # type: ignore
        # and finally add to mu
        mu = mu

        numpyro.deterministic("expected_cases", mu)
        numpyro.sample("cases", dist.Poisson(mu), obs=cases)
