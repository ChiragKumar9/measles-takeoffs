import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl
import scipy as sp


class PartialPooledPoisson:
    def __init__(
        self,
        time_and_group_varying: jnp.ndarray | None,
        state_vary_rw: bool,
        group_varying_only: jnp.ndarray | None,
        time_varying_only: jnp.ndarray | None,
        controls: jnp.ndarray,
        population: jnp.ndarray,
        groups: pl.Series,
        years: pl.Series,
        spatial_units: pl.Series,
        adjacency_matrix: np.ndarray,
        baseline=dist.Normal(0, 10.0),
        priors=dist.Normal(0, 10.0),
        spatial_correlation=dist.Beta(2, 2),
        spatial_sigma=dist.HalfNormal(0.25),
        drift_scale=0.25,
    ):
        if time_and_group_varying is not None:
            self.time_and_group_varying = time_and_group_varying
            self.num_time_and_group_varying = time_and_group_varying.shape[1]
            self.state_vary_rw = state_vary_rw
        else:
            self.time_and_group_varying = None
            self.num_time_and_group_varying = 0
            self.state_vary_rw = None
        if group_varying_only is not None:
            self.group_varying_only = group_varying_only
            self.num_group_varying_only = group_varying_only.shape[1]
        else:
            self.group_varying_only = None
            self.num_group_varying_only = 0
        if time_varying_only is not None:
            self.time_varying_only = time_varying_only
            self.num_time_varying_only = time_varying_only.shape[1]
        else:
            self.time_varying_only = None
            self.num_time_varying_only = 0
        self.controls = controls
        self.num_controls = controls.shape[1]
        self.num_observations = controls.shape[0]
        self.population = population
        # turn groups into an index
        self.groups = groups.replace(
            groups.unique(maintain_order=True), pl.arange(0, groups.n_unique())
        ).cast(pl.Int64)
        self.num_groups = max(self.groups) + 1
        self.groups = self.groups.to_jax()
        # add fixed effects for groups
        # subtract 1 because we don't need a var for the last group
        self.num_controls += self.num_groups - 1
        fixed_effect = jnp.eye(self.num_groups)
        # zero out the last row, setting the last group as the reference group
        fixed_effect = fixed_effect.at[-1].set(0)
        self.controls = jnp.hstack(
            [
                self.controls,
                fixed_effect[self.groups],
            ]
        )
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
        self.adjacency_matrix = sp.sparse.csr_matrix(self.adjacency_matrix)
        self.years = years - years.min()
        self.years = -self.years + self.years.max()
        self.num_years = years.n_unique()
        self.years = self.years.to_jax()

        self.betas = priors
        self.baseline = baseline
        self.spatial_correlation = spatial_correlation
        self.spatial_sigma = spatial_sigma
        self.drift_scale = drift_scale

    def fit(self, cases=None):
        baseline_rate = numpyro.sample("baseline_rate", self.baseline)
        with numpyro.plate("regression_covariates", self.num_controls):
            regression_betas = numpyro.sample("regression_betas", self.betas)  # type: ignore

            # add a zero at the end because we set the last state as the reference state
            regression_betas = jnp.pad(regression_betas, (0, 1))

            covariate_effect = jnp.dot(self.controls, regression_betas)

        if self.num_group_varying_only > 0:
            with numpyro.plate(
                "group_varying_only", self.num_group_varying_only
            ):
                # sample betas
                group_varying_only_betas = numpyro.sample(
                    "group_varying_only_betas", self.betas
                )
                # each of these variables can vary by group (state)
                group_varying_only_group_variabilities = numpyro.sample(
                    "group_varying_only_group_variabilities",
                    dist.HalfNormal(self.drift_scale),
                )
                with numpyro.plate(
                    "group_varying_only_groups", self.num_groups
                ):
                    group_varying_only_group_offsets = numpyro.sample(
                        "group_varying_only_group_offsets", dist.Normal(0, 1)
                    )
                # Let's assemble the effect for each group
                group_varying_only_effect = (
                    group_varying_only_group_variabilities
                    * group_varying_only_group_offsets
                    + group_varying_only_betas
                )
                group_varying_only_effect = group_varying_only_effect[  # type: ignore
                    self.groups
                ]
                group_varying_only_effect = jnp.sum(
                    group_varying_only_effect * self.group_varying_only,  # type: ignore
                    axis=1,
                )
        else:
            group_varying_only_effect = 0.0

        if self.num_time_varying_only > 0:
            with numpyro.plate(
                "time_varying_only", self.num_time_varying_only
            ):
                # sample betas
                time_varying_only_betas = numpyro.sample(
                    "time_varying_only_betas", self.betas
                )
                # each of these variables has an effect that can vary over time
                # via a random walk
                with numpyro.plate("rw_time_varying_only", self.num_years - 1):
                    rw_offset_tvo = numpyro.sample(
                        "rw_offset_tvo", dist.Normal(0, 1)
                    )
                rw_drift_tvo = numpyro.sample(
                    "rw_drift_tvo", dist.HalfNormal(self.drift_scale)
                )
                rw_tvo = rw_drift_tvo * rw_offset_tvo
                rw_tvo = jnp.pad(
                    jnp.cumsum(rw_tvo, axis=0),
                    # pad the first year with zeros
                    ((1, 0), (0, 0)),
                )
                numpyro.deterministic(
                    "effects_time_varying_only",
                    rw_tvo + time_varying_only_betas,
                )
                time_varying_only_effect = (
                    rw_tvo[self.years] + time_varying_only_betas
                )  # type: ignore
                time_varying_only_effect = jnp.sum(
                    time_varying_only_effect * self.time_varying_only,  # type: ignore
                    axis=1,
                )
        else:
            time_varying_only_effect = 0.0

        if self.num_time_and_group_varying > 0:
            with numpyro.plate(
                "time_and_group_varying", self.num_time_and_group_varying
            ):
                # sample the coefficients for each special variable
                time_and_group_varying_betas = numpyro.sample(
                    "time_and_group_varying_betas", self.betas
                )

                # random walk for mean effect over time
                with numpyro.plate("rw", self.num_years - 1):
                    rw_offset = numpyro.sample("rw_offset", dist.Normal(0, 1))
                rw_drift = numpyro.sample(
                    "rw_drift", dist.HalfNormal(self.drift_scale)
                )
                rw = rw_drift * rw_offset
                rw = jnp.pad(
                    jnp.cumsum(rw, axis=0),
                    # pad the first year with zeros
                    ((1, 0), (0, 0)),
                )
                numpyro.deterministic(
                    "effects_over_time", rw + time_and_group_varying_betas
                )

                if self.state_vary_rw:
                    # random walk for the across-states variability over time
                    # we could merge this plate with the one above, but I'm choosing to
                    # keep them separate for clarity since this plate pertains to different
                    # variables
                    overall_sigma = numpyro.sample(
                        "overall_sigma", self.baseline
                    )
                    with numpyro.plate(
                        "rw_sigma_group_variability", self.num_years - 1
                    ):
                        rw_sigma_offset = numpyro.sample(
                            "rw_sigma_offset", dist.Normal(0, 1)
                        )
                    rw_sigma_drift = numpyro.sample(
                        "rw_sigma_drift", dist.HalfNormal(self.drift_scale)
                    )
                    group_variability = rw_sigma_drift * rw_sigma_offset
                    group_variability = jnp.pad(
                        jnp.cumsum(group_variability, axis=0),
                        # pad the first year with zeros
                        ((1, 0), (0, 0)),
                    )
                    group_variability = overall_sigma + group_variability

                    # transform to ensure positivity (this is a scale parameter)
                    group_variability = jnp.exp(group_variability)
                    # add a small number to avoid numerical issues
                    group_variability = (
                        group_variability + jnp.finfo(jnp.float64).eps
                    )
                    numpyro.deterministic(
                        "group_variability_over_time", group_variability
                    )

                    # now on to by group-variables for each year
                    with numpyro.plate("by_year", self.num_years):
                        with numpyro.plate("by_group", self.num_groups):
                            time_and_group_varying_group_offsets = (
                                numpyro.sample(
                                    "time_and_group_varying_group_offsets",
                                    dist.Normal(0, 1),
                                )
                            )

                    # assemble effect
                    time_and_group_varying_effect = (
                        time_and_group_varying_betas[None, :, None]  # type: ignore
                        + rw[:, :, None]
                        + group_variability[:, :, None]
                        * jnp.transpose(
                            time_and_group_varying_group_offsets, (1, 2, 0)
                        )
                    )

                    numpyro.deterministic(
                        "effects_over_time_by_group",
                        time_and_group_varying_effect,
                    )

                    time_and_group_varying_effect = (
                        time_and_group_varying_effect[
                            self.years, :, self.groups
                        ]
                    )  # type: ignore
                    numpyro.deterministic(
                        "effect_by_time_and_group",
                        time_and_group_varying_effect,
                    )

                else:
                    # No state-specific deviations - all states share the same random walk
                    # But states can still have constant offsets
                    time_and_group_varying_group_variabilities = (
                        numpyro.sample(
                            "time_and_group_varying_group_variabilities",
                            dist.HalfNormal(self.drift_scale),
                        )
                    )
                    with numpyro.plate(
                        "time_and_group_varying_groups", self.num_groups
                    ):
                        time_and_group_varying_group_offsets = numpyro.sample(
                            "time_and_group_varying_group_offsets",
                            dist.Normal(0, 1),
                        )

                    state_effect = (
                        time_and_group_varying_group_variabilities
                        * time_and_group_varying_group_offsets
                    )
                    state_effect = state_effect + time_and_group_varying_betas
                    state_effect = state_effect[self.groups]  # type: ignore

                    time_and_group_varying_effect = (
                        state_effect + rw[self.years]
                    )  # type: ignore
                    numpyro.deterministic(
                        "effects_over_time_and_group",
                        time_and_group_varying_effect,
                    )

                time_and_group_varying_effect = jnp.sum(
                    time_and_group_varying_effect
                    * self.time_and_group_varying,  # type: ignore
                    axis=1,
                )
        else:
            time_and_group_varying_effect = 0.0

        mu = (
            baseline_rate
            + jnp.log(self.population)  # offset by population
            + covariate_effect
            + group_varying_only_effect
            + time_varying_only_effect
            + time_and_group_varying_effect
        )

        # add spatial correlation
        correlation = numpyro.sample(
            "spatial_correlation", self.spatial_correlation
        )
        spatial_sigma = numpyro.sample("spatial_sigma", self.spatial_sigma)
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
        # spatial_effect is a vector of length of number of counties
        # need to expand spatial effects also to include counties without neighbors

        spatial_random_sigma = numpyro.sample(
            "spatial_random_sigma", self.spatial_sigma
        )
        with numpyro.plate(
            "spatial_random_effects_plate", self.has_a_neighbor.shape[0]
        ):
            spatial_random_effects = numpyro.sample(
                "spatial_random_effects", dist.Normal(0, 1)
            )
        full_spatial_effect = spatial_random_sigma * spatial_random_effects

        full_spatial_effect = full_spatial_effect.at[self.has_a_neighbor].set(  # type: ignore
            spatial_effect
        )
        full_spatial_effect = full_spatial_effect[self.spatial_units]  # type: ignore

        mu = mu + full_spatial_effect
        mu = jnp.exp(mu)
        mu = mu + jnp.finfo(jnp.float64).eps
        numpyro.deterministic("mu", mu)

        with numpyro.plate("observations", size=self.num_observations):
            numpyro.sample(
                "cases",
                dist.Poisson(mu),  # type: ignore
                obs=cases,
            )
