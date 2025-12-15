import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl
import scipy as sp


class PartialPooledPoisson:
    def __init__(
        self,
        iv: jnp.ndarray,
        pool: bool,
        regression_covariates: jnp.ndarray,
        exposure: jnp.ndarray,
        groups: pl.Series,
        spatial_units: pl.Series,
        adjacency_matrix: np.ndarray,
        baseline=dist.Normal(0, 1.0),
        priors=dist.Normal(0, 1.0),
        group_fixed_effect=True,
        spatial=True,
        spatial_correlation=dist.Beta(2, 2),
        spatial_sigma=dist.HalfNormal(10.0),
    ):
        self.iv = iv
        self.num_observations = iv.shape[0]
        self.num_ivs = iv.shape[1]
        self.pool = pool
        self.regression_covariates = regression_covariates
        self.num_regression_covariates = regression_covariates.shape[1]
        self.exposure = exposure
        # turn groups into an index
        self.groups = groups.replace(
            groups.unique(maintain_order=True), pl.arange(0, groups.n_unique())
        ).cast(pl.Int64)
        self.num_groups = max(self.groups) + 1
        self.groups = self.groups.to_jax()
        self.group_fixed_effect = group_fixed_effect
        if self.group_fixed_effect:
            # subtract 1 because we don't need a var for the last group
            self.num_regression_covariates += self.num_groups - 1
            # zero out the last row, setting the last group as the reference group
            group_fixed_effect = jnp.eye(self.num_groups)
            group_fixed_effect = group_fixed_effect.at[-1].set(0)
            self.regression_covariates = jnp.hstack(
                [
                    self.regression_covariates,
                    group_fixed_effect[self.groups],
                ]
            )
        self.betas = priors
        self.baseline = baseline
        self.spatial_correlation = spatial_correlation
        self.spatial_sigma = spatial_sigma
        self.spatial_units = spatial_units.replace(
            spatial_units.unique(maintain_order=True),
            pl.arange(0, spatial_units.n_unique()),
        ).cast(pl.Int64)
        self.spatial_units = self.spatial_units.to_jax()
        self.has_a_neighbor = adjacency_matrix.sum(axis=1) > 0
        # make the adjacency matrix only include counties with neighbors
        self.adjacency_matrix = adjacency_matrix[self.has_a_neighbor][
            :, self.has_a_neighbor
        ]
        self.is_adjacency_empty = self.adjacency_matrix.shape[0] == 0
        self.adjacency_matrix = sp.sparse.csr_matrix(self.adjacency_matrix)
        self.spatial = spatial

    def fit(self, cases=None):
        baseline_rate = numpyro.sample("baseline_rate", self.baseline)
        with numpyro.plate(
            "regression_covariates", self.num_regression_covariates
        ):
            regression_betas = numpyro.sample("regression_betas", self.betas)  # type: ignore

        if self.group_fixed_effect:
            # add a zero at the end because we set the last state as the reference state
            regression_betas = jnp.pad(
                regression_betas,
                # put one zero at the end of the array
                (0, 1),
            )

        covariate_effect = jnp.dot(
            self.regression_covariates, regression_betas
        )

        if self.pool:
            with numpyro.plate("iv", self.num_ivs):
                iv_global = numpyro.sample("iv_global", self.baseline)
                iv_group_variabilities = numpyro.sample(
                    "iv_group_variabilities", dist.HalfNormal(0.25)
                )

                with numpyro.plate("groups", self.num_groups):
                    iv_group_offsets = numpyro.sample(
                        "iv_group_offsets", dist.Normal(0, 1)
                    )

                iv_effect = (
                    iv_group_variabilities * iv_group_offsets + iv_global
                )

            iv_effect = jnp.sum(iv_effect[self.groups] * self.iv, axis=1)  # type: ignore
        else:
            # pull coefficients for each iv just like they are normal covariates
            with numpyro.plate("iv_no_pooling", self.num_ivs):
                iv_global = numpyro.sample("iv_global", self.betas)  # type: ignore
            iv_effect = jnp.dot(self.iv, iv_global)

        mu = (
            baseline_rate
            + jnp.log(self.exposure)
            + covariate_effect
            + iv_effect
        )

        spatial_random_sigma = numpyro.sample(
            "spatial_random_sigma", dist.HalfNormal(1.0)
        )
        with numpyro.plate(
            "spatial_random_effects_plate", self.has_a_neighbor.shape[0]
        ):
            spatial_random_effects = numpyro.sample(
                "spatial_random_effects", dist.Normal(0, 1)
            )
        full_spatial_effect = spatial_random_sigma * spatial_random_effects

        if not self.is_adjacency_empty:
            if self.spatial:
                # add spatial correlation
                correlation = numpyro.sample(
                    "spatial_correlation", self.spatial_correlation
                )
                spatial_sigma = numpyro.sample(
                    "spatial_sigma", self.spatial_sigma
                )
                car_distribution = dist.CAR(
                    loc=0,
                    correlation=(2 * correlation - 1),  # type: ignore
                    conditional_precision=1.0,  # type: ignore
                    adj_matrix=self.adjacency_matrix,  # type: ignore
                    is_sparse=True,
                )
                raw_spatial_effect = numpyro.sample(
                    "raw_spatial_effect", car_distribution
                )
                spatial_effect = spatial_sigma * raw_spatial_effect
                full_spatial_effect = full_spatial_effect.at[  # type: ignore
                    self.has_a_neighbor
                ].set(spatial_effect)

        # spatial_effect is a vector of length of number of counties
        # need to expand spatial effects also to include counties without neighbors
        full_spatial_effect = full_spatial_effect[self.spatial_units]  # type: ignore
        mu = mu + full_spatial_effect
        mu = jnp.exp(mu)
        mu = mu + jnp.finfo(jnp.float64).eps

        with numpyro.plate("observations", size=self.num_observations):
            numpyro.sample(
                "cases",
                dist.Poisson(mu),  # type: ignore
                obs=cases,
            )
