import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl
import scipy as sp
from numpyro.handlers import reparam

from outbreak_size.car_noncentered import CARNonCentered

# the distributions of the outbreak size around its mean: Poisson, or a
# negative binomial with one concentration parameter (variance = mean +
# mean**2 / concentration), for counts more variable than a Poisson allows
LIKELIHOODS = ("poisson", "negative_binomial")


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
        years: pl.Series = None,  # type: ignore
        baseline=dist.Normal(0, 10.0),
        beta_priors=dist.Normal(0, 1.0),
        global_priors=dist.Normal(0, 10.0),
        group_fixed_effect=True,
        spatial=True,
        spatial_correlation=dist.Beta(2, 2),
        spatial_sigma=dist.HalfNormal(0.25),
        car_sigma=dist.HalfNormal(0.25),
        pop_exponent=dist.Normal(0.5, 0.5),
        likelihood="poisson",
        concentration=dist.Exponential(0.1),
        non_centered_car=False,
        mmr_effectiveness=None,
        coverage_percent=None,
        coverage_mean_sd=None,
    ):
        """
        `mmr_effectiveness`, if given, is the prior of the share of the
        vaccinated who are immune. The first pooled variable (iv[:, 0]) is
        then the coverage on the natural scale, standardized, and the model
        rebuilds it from the sampled effectiveness e: (log(1 - e * coverage /
        100) - mean) / sd. It needs `coverage_percent` (the coverage of each
        row, in percent) and `coverage_mean_sd` (the mean and standard
        deviation the natural-scale column was standardized with, at a fixed
        effectiveness). The values of iv[:, 0] are then not used.
        """
        if likelihood not in LIKELIHOODS:
            raise ValueError(
                f"likelihood must be one of {LIKELIHOODS}, not {likelihood!r}"
            )
        if mmr_effectiveness is not None and (
            coverage_percent is None or coverage_mean_sd is None
        ):
            raise ValueError(
                "mmr_effectiveness needs coverage_percent and coverage_mean_sd"
            )
        self.mmr_effectiveness = mmr_effectiveness
        self.coverage_percent = coverage_percent
        self.coverage_mean_sd = coverage_mean_sd
        self.likelihood = likelihood
        self.concentration = concentration
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
        self.betas = beta_priors
        self.global_priors = global_priors
        self.baseline = baseline
        self.pop_exponent = pop_exponent
        self.spatial_correlation = spatial_correlation
        self.spatial_sigma = spatial_sigma
        self.spatial_units = spatial_units.replace(
            spatial_units.unique(maintain_order=True),
            pl.arange(0, spatial_units.n_unique()),
        ).cast(pl.Int64)
        self.spatial_units = self.spatial_units.to_jax()
        # the rows of the adjacency matrix are the spatial units in the order
        # they first appear; a unit that is spelled two ways (one county with
        # two names) would shift every later unit onto another one's row, and
        # JAX does not complain about out-of-range indices
        if spatial and spatial_units.n_unique() != adjacency_matrix.shape[0]:
            raise ValueError(
                f"{spatial_units.n_unique()} spatial units but the adjacency "
                f"matrix has {adjacency_matrix.shape[0]} rows"
            )
        self.has_a_neighbor = adjacency_matrix.sum(axis=1) > 0
        # make the adjacency matrix only include counties with neighbors
        self.adjacency_matrix = adjacency_matrix[self.has_a_neighbor][
            :, self.has_a_neighbor
        ]
        self.is_adjacency_empty = self.adjacency_matrix.shape[0] == 0
        self.adjacency_matrix = sp.sparse.csr_matrix(self.adjacency_matrix)
        self.spatial = spatial
        self.car_sigma = car_sigma
        if years is not None:
            self.years = (
                years.replace(
                    years.unique(maintain_order=True),
                    pl.arange(0, years.n_unique()),
                )
                .cast(pl.Int64)
                .to_jax()
            )
            self.num_years = int(self.years.max()) + 1
        else:
            self.years = None
            self.num_years = None
        # `non_centered_car`: the sampler works on white noise that is turned
        # into the CAR field (same prior, same model, better geometry for the
        # correlation; see car_noncentered.py). A model without a CAR term is
        # left as it is
        if non_centered_car and spatial and not self.is_adjacency_empty:
            car = CARNonCentered(self.adjacency_matrix)
            sites = (
                ["raw_spatial_effect"]
                if self.years is None
                else [
                    f"year_{year}/raw_spatial_effect"
                    for year in range(self.num_years)  # type: ignore
                ]
            )
            self.fit = reparam(self.fit, config={site: car for site in sites})  # type: ignore

    def fit(self, is_observed, cases=None):
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

        iv = self.iv
        if self.mmr_effectiveness is not None:
            effectiveness = numpyro.sample(
                "mmr_effectiveness", self.mmr_effectiveness
            )
            mean, sd = self.coverage_mean_sd  # type: ignore
            log_susceptible = jnp.log(
                1 - effectiveness * self.coverage_percent / 100  # type: ignore
            )
            iv = iv.at[:, 0].set((log_susceptible - mean) / sd)

        if self.pool:
            with numpyro.plate("iv", self.num_ivs):
                iv_global = numpyro.sample("iv_global", self.global_priors)  # type: ignore
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

            iv_effect = jnp.sum(iv_effect[self.groups] * iv, axis=1)  # type: ignore
        else:
            # pull coefficients for each iv just like they are normal covariates
            with numpyro.plate("iv_no_pooling", self.num_ivs):
                iv_global = numpyro.sample("iv_global", self.global_priors)  # type: ignore
            iv_effect = jnp.dot(iv, iv_global)

        # the expected size of a county scales as its population ** pop_exponent
        # (an exponent of 1 makes it proportional to the population); the
        # baseline is the log rate of a county of the reference population,
        # the median population of the observed rows
        # (a number instead of a prior fixes the exponent, with nothing sampled)
        pop_exponent = (
            self.pop_exponent
            if isinstance(self.pop_exponent, (int, float))
            else numpyro.sample("pop_exponent", self.pop_exponent)
        )
        log_reference = jnp.log(
            jnp.nanmedian(jnp.where(is_observed > 0, self.exposure, jnp.nan))
        )
        log_population = log_reference + pop_exponent * (
            jnp.log(self.exposure) - log_reference
        )

        mu = baseline_rate + log_population + covariate_effect + iv_effect

        if self.spatial:
            spatial_random_sigma = numpyro.sample(
                "spatial_random_sigma", self.spatial_sigma
            )
            if self.years is not None:
                # sample a non-neighbor random effect per (year, spatial_unit)
                with numpyro.plate("years_non_neighbor", self.num_years):  # type: ignore
                    with numpyro.plate(
                        "spatial_random_effects_plate",
                        self.has_a_neighbor.shape[0],
                    ):
                        spatial_random_effects = numpyro.sample(
                            "spatial_random_effects", dist.Normal(0, 1)
                        )
                # spatial_random_effects is (num_spatial_units, num_years); transpose to (num_years, num_spatial_units)
                full_spatial_effect = (
                    spatial_random_effects.T * spatial_random_sigma  # type: ignore
                )
            else:
                with numpyro.plate(
                    "spatial_random_effects_plate",
                    self.has_a_neighbor.shape[0],
                ):
                    spatial_random_effects = numpyro.sample(
                        "spatial_random_effects", dist.Normal(0, 1)
                    )
                full_spatial_effect = (
                    spatial_random_effects * spatial_random_sigma
                )

            if not self.is_adjacency_empty:
                # add spatial correlation
                if self.years is not None:
                    # Sample correlation and sigma once per year
                    with numpyro.plate("years_car_params", self.num_years):  # type: ignore
                        correlation = numpyro.sample(
                            "spatial_correlation", self.spatial_correlation
                        )
                        spatial_sigma = numpyro.sample(
                            "spatial_sigma", self.car_sigma
                        )

                    # Sample CAR effect per year with year-specific parameters
                    raw_spatial_effects = []
                    for year_idx in range(self.num_years):  # type: ignore
                        car_distribution = dist.CAR(
                            loc=0,
                            correlation=(2 * correlation[year_idx] - 1),  # type: ignore
                            conditional_precision=1.0,  # type: ignore
                            adj_matrix=self.adjacency_matrix,  # type: ignore
                            is_sparse=True,
                        )
                        with numpyro.handlers.scope(prefix=f"year_{year_idx}"):
                            raw_spatial_effect_year = numpyro.sample(
                                "raw_spatial_effect", car_distribution
                            )
                        raw_spatial_effects.append(raw_spatial_effect_year)

                    # Stack into (num_years, num_neighbors) array
                    raw_spatial_effect = jnp.stack(raw_spatial_effects, axis=0)

                    # Scale by year-specific sigma: broadcasting (num_years, 1) * (num_years, num_neighbors)
                    spatial_effect = (
                        spatial_sigma[:, None] * raw_spatial_effect  # type: ignore
                    )

                    full_spatial_effect = full_spatial_effect.at[  # type: ignore
                        :, self.has_a_neighbor
                    ].set(spatial_effect)
                else:
                    # Non-time-varying case (original code)
                    correlation = numpyro.sample(
                        "spatial_correlation", self.spatial_correlation
                    )
                    spatial_sigma = numpyro.sample(
                        "spatial_sigma", self.car_sigma
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

            if self.years is not None:
                # spatial_effect is a (num_years, num_spatial_units) matrix
                # index by (year, spatial_unit) to get a per-observation effect
                full_spatial_effect = full_spatial_effect[  # type: ignore
                    self.years, self.spatial_units
                ]  # type: ignore
            else:
                # spatial_effect is a vector of length of number of counties
                # need to expand spatial effects also to include counties without neighbors
                full_spatial_effect = full_spatial_effect[self.spatial_units]  # type: ignore
            mu = mu + full_spatial_effect
        mu = jnp.exp(mu)
        mu = mu + jnp.finfo(jnp.float64).eps
        # truncate mu at the population size to avoid overflow errors in the Poisson log prob
        mu = jnp.minimum(mu, self.exposure)

        numpyro.deterministic("mu", mu)

        if self.likelihood == "negative_binomial":
            concentration = numpyro.sample(
                "nb_concentration", self.concentration
            )
            outbreak_size = dist.NegativeBinomial2(mu, concentration)
        else:
            outbreak_size = dist.Poisson(mu)

        if cases is not None:
            # (named for the Poisson it was first written for)
            poisson_log_prob = outbreak_size.log_prob(cases)
            numpyro.factor("poisson_log_prob", poisson_log_prob * is_observed)
        elif self.likelihood == "negative_binomial":
            # a county cannot have more cases than people: the predicted
            # cases are the draws capped at the population (the raw draws
            # stay available as cases_uncapped)
            uncapped = jnp.asarray(
                numpyro.sample("cases_uncapped", outbreak_size)
            )
            numpyro.deterministic(
                "cases",
                jnp.minimum(uncapped, self.exposure.astype(uncapped.dtype)),
            )
        else:
            numpyro.sample("cases", outbreak_size)


# the variables that are partially pooled across states ("ivs"); every other
# variable is an ordinary regression covariate
IV_NAMES = {
    "estimated_mmr_cov",
    "nonmed_exempt_pct",
    "med_exempt_pct",
    "estimated_mmr_cov_5yr",
    "nonmed_exempt_pct_5yr",
    "med_exempt_pct_5yr",
    "school_sd",  # SD of the school-level MMR rates of a county (02, school SD fits)
}


def count_ivs(variables: list[str]) -> int:
    """Number of leading variables that are pooled ivs."""
    n = 0
    for variable in variables:
        if variable not in IV_NAMES:
            break
        n += 1
    return n
