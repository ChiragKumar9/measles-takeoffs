"""
What the rolling-validation, projection and policy scripts (06, 07 and 11)
give the outbreak model: the rows it is trained on stacked with the rows
it predicts, their covariates standardized with the training rows only, and
the model built from them (see outbreak_size.poisson).
"""

import math

import jax.numpy as jnp
import polars as pl
from numpyro import distributions as dist
from sklearn.preprocessing import StandardScaler

from outbreak_size import poisson
from outbreak_size.adjacency import require_same_unit_order

# the pooled ("iv") covariates, then the ordinary regression covariates of the
# "All covariates" model (the columns of outbreak_size.interventions.FEATURE_COLS)
IV_COLUMNS = ["estimated_mmr_cov", "nonmed_exempt_pct", "med_exempt_pct"]
REGRESSION_COLUMNS = [
    "fraction_under_5",
    "fraction_school_aged",
    "avg_house_size",
    "std_house_size",
    "fraction_public",
    "fraction_white",
    "fraction_graduate_degree",
    "log_population_density",
    "log_income",
]


def stack_rows(
    train_rows: pl.DataFrame,
    projection_rows: pl.DataFrame,
    cases_as_int: bool = False,
) -> pl.DataFrame:
    """
    The training rows with the rows to predict below them, in that order (the
    model's draws are matched to the rows by position). The projection rows
    have no outbreak cases: the missing cases_outbreak are filled with 0 as a
    placeholder that the model does not use (it is told which rows are
    observed). `cases_as_int` also makes the column integer.
    """
    rows = pl.concat([train_rows, projection_rows], how="diagonal_relaxed")
    cases = pl.col("cases_outbreak").fill_null(0)
    if cases_as_int:
        cases = cases.cast(pl.Int64)
    return rows.with_columns(cases_outbreak=cases)


def scale_covariates(
    rows: pl.DataFrame, columns: list[str], is_train: pl.Expr
) -> pl.DataFrame:
    """
    The covariates `columns` of every row of `rows`, in order, standardized
    with the mean and standard deviation of the training rows (the rows where
    `is_train` holds), so that nothing is learned from the rows to predict.
    """
    scaler = StandardScaler()
    scaler.fit(rows.filter(is_train).select(columns).to_numpy())
    return pl.DataFrame(
        scaler.transform(rows.select(columns).to_numpy()), schema=columns
    )


COVARIATE_SCALES = ("raw", "natural")
# the vaccination columns that change scale: the coverage (percent, also as its
# five-year average and as the average of the neighbours) and the exemptions
# (percent, also as their five-year averages)
COVERAGE_COLUMNS = (
    "estimated_mmr_cov",
    "estimated_mmr_cov_5yr",
    "neighbors_coverage",
)
EXEMPTION_COLUMNS = (
    "nonmed_exempt_pct",
    "med_exempt_pct",
    "nonmed_exempt_pct_5yr",
    "med_exempt_pct_5yr",
)


def apply_covariate_scale(
    rows: pl.DataFrame,
    covariate_scale: str,
    mmr_effectiveness: float | None = None,
) -> pl.DataFrame:
    """
    The rows with their vaccination covariates on the scale the model uses:
    "raw" leaves them as they are (percentages); "natural" puts the coverage
    on the log of the susceptible fraction, log(1 - mmr_effectiveness *
    coverage / 100), and each exemption percentage x on log(1 + x), so that a
    coefficient multiplies the expected outbreak size by the same factor for
    the same proportional change in the susceptible fraction, wherever the
    coverage starts (a drop from 99% to 95% coverage doubles the susceptibles,
    a drop from 50% to 46% changes them by 4%).

    Apply it to the training, projection and scenario rows alike, before
    scale_covariates. A coverage above 100% is an error: the susceptible
    fraction would go to zero or below, and a value is never clipped.
    """
    if covariate_scale not in COVARIATE_SCALES:
        raise ValueError(
            f"covariate_scale must be one of {COVARIATE_SCALES}, "
            f"not {covariate_scale!r}"
        )
    if covariate_scale == "raw":
        return rows
    if mmr_effectiveness is None:
        raise ValueError(
            "the natural covariate scale needs mmr_effectiveness "
            "(add it to params.yml, with the PMID of its source)"
        )
    coverage = [c for c in COVERAGE_COLUMNS if c in rows.columns]
    exemptions = [c for c in EXEMPTION_COLUMNS if c in rows.columns]
    for column in coverage:
        if rows[column].max() > 100:  # type: ignore
            raise ValueError(f"{column} has a coverage above 100%")
    return rows.with_columns(
        *[
            (1 - mmr_effectiveness * pl.col(c) / 100).log().alias(c)
            for c in coverage
        ],
        *[(1 + pl.col(c)).log().alias(c) for c in exemptions],
    )


def pooled_model(
    rows: pl.DataFrame,
    iv: pl.DataFrame,
    regression_covariates,
    pool: bool,
    spatial: bool,
    adjacency_matrix,
    prior_sd: float | None = None,
    baseline_mean: float = 0.0,
    likelihood: str = "poisson",
    non_centered_car: bool = False,
    spatial_sigma_scale: float | None = None,
) -> poisson.PartialPooledPoisson:
    """
    The outbreak model of the rows `rows` (with their pop, state and geoid):
    `iv` are the scaled covariates whose effect is pooled across the states
    (if `pool`) and `regression_covariates` the others, with a spatial
    (CAR) term over the counties if `spatial`. The Normal priors of the global
    coefficients, of the other regression coefficients (controls and state
    effects) and of the baseline have standard deviation `prior_sd`, or
    the model's own default if it is None (a model used only to predict from
    posterior draws does not use its priors). The baseline prior is centred on
    `baseline_mean`, the log attack rate of the data (see log_attack_rate).
    The two spatial scales (of the CAR field and of the independent county
    effects) have HalfNormal priors with scale `spatial_sigma_scale`, or the
    model's own default if it is None (as for `prior_sd`, a model used only to
    predict from posterior draws does not use it).
    `likelihood` is "poisson" or "negative_binomial" (see poisson.LIKELIHOODS).
    `non_centered_car` samples the CAR term through white noise (see
    car_noncentered.py): the same model, with better mixing of the spatial
    correlation.
    """
    priors: dict = {}
    if prior_sd is not None:
        priors = dict(
            global_priors=dist.Normal(0, prior_sd),
            beta_priors=dist.Normal(0, prior_sd),
            baseline=dist.Normal(baseline_mean, prior_sd),
        )
    if spatial_sigma_scale is not None:
        priors["car_sigma"] = dist.HalfNormal(spatial_sigma_scale)
        priors["spatial_sigma"] = dist.HalfNormal(spatial_sigma_scale)
    if spatial:
        # the callers build the adjacency matrix in the order the fips of the
        # rows first appear; the model numbers the units by their geoid
        require_same_unit_order(rows["fips"], rows["geoid"], "pooled model")
    return poisson.PartialPooledPoisson(
        iv=iv.to_jax(),
        pool=pool,
        regression_covariates=regression_covariates,
        exposure=rows["pop"].to_jax(),
        groups=rows["state"],
        group_fixed_effect=False,
        spatial=spatial,
        spatial_units=rows["geoid"],
        adjacency_matrix=adjacency_matrix,
        likelihood=likelihood,
        non_centered_car=non_centered_car,
        **priors,
    )


def log_attack_rate(rows: pl.DataFrame) -> float:
    """
    The log of the attack rate of the outbreak records `rows` (columns
    cases_outbreak and pop): their local cases over their population, every
    county-year being its own row (a county in two years counts twice, in the
    cases and in the population). It is the centre of the models' baseline
    prior.
    """
    cases = rows["cases_outbreak"].sum()
    if rows.height == 0 or cases <= 0:
        raise ValueError("no outbreak cases to take an attack rate from")
    return math.log(cases / rows["pop"].sum())


def no_regression_covariates(n_rows: int):
    """The placeholder for a model with no ordinary covariates: a column of
    zeros, which the model ignores."""
    return jnp.zeros((n_rows, 1))
