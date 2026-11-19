"""
The projection results of the 2026 counties and states: the importation
samples joined to the counties, the unconditioned case counts (the cases with
the importations) of every posterior draw, and their summaries and ranks (see
analysis/09_assemble_projection_results.py, which reads and writes the files).

The counts are jax arrays; call numpyro.enable_x64() first, as the script
does, to compute them in double precision.
"""

import jax.numpy as jnp
import numpy as np
import polars as pl

from outbreak_size.importation import IMPORT_CONDITIONING_VALIDATION_YEAR
from outbreak_size.ranking import risk_percentile

COUNTY_COLUMNS = [
    "fips",
    "state",
    "county",
    "year",
    "type",
    "unconditioned_count_rank",
    "unconditioned_incidence_rank",
    "mean_unconditioned_projection",
    "median_unconditioned_projection",
    "lower_unconditioned_projection",
    "upper_unconditioned_projection",
    "importation_probability_mean",
    "importation_probability_lower",
    "importation_probability_upper",
    "importation_probability_median",
    "estimated_mmr_cov_raw",
]
STATE_COLUMNS = [
    "state",
    "type",
    "unconditioned_risk_rank",
    "mean_unconditioned_projection",
    "median_unconditioned_projection",
    "lower_unconditioned_projection",
    "upper_unconditioned_projection",
    "weighted_mmr_coverage",
]


def join_importation(
    df: pl.DataFrame,
    import_df: pl.DataFrame,
    probability_samples: pl.DataFrame,
    count_samples: pl.DataFrame,
) -> pl.DataFrame:
    """
    Add the importation probability and the expected number of importations
    of each county (summaries and the posterior samples) to the 2026
    projection rows `df`.

    `import_df` has one row per county, validation year and row type (as
    written by 08_importation_model.py) and `probability_samples` and
    `count_samples` hold the samples of its rows in the same order. The 2026
    projection rows carry the year of their vaccination data (2025 and
    earlier) in `year`; their importation rows are those of validation year
    2026 (the imports are known through 2025: this is the posterior of every
    year after 2025) and `year` is left as it is. A row without importation samples is an
    error.
    """
    import_df = import_df.with_columns(
        importation_probability_samples=pl.Series(
            probability_samples["importation_probability_samples"].__array__()
        ),
        expected_importations_samples=pl.Series(
            count_samples["expected_importations_samples"].__array__()
        ),
    )
    df = df.with_columns(
        validation_year=pl.lit(
            IMPORT_CONDITIONING_VALIDATION_YEAR,
            dtype=import_df.schema["validation_year"],
        )
    )
    df = df.join(
        import_df.select(
            [
                "fips",
                "validation_year",
                "type",
                "importation_probability_mean",
                "importation_probability_lower",
                "importation_probability_upper",
                "importation_probability_median",
                "importation_probability_samples",
                "expected_importations_samples",
            ]
        ),
        on=["fips", "validation_year", "type"],
        how="left",
        maintain_order="left",
    ).drop("validation_year")

    if df["importation_probability_samples"].null_count() > 0:
        raise ValueError(
            "Missing importation probability samples after join — check "
            "fips/year/type alignment"
        )
    if df["expected_importations_samples"].null_count() > 0:
        raise ValueError(
            "Missing importation count samples after join — check "
            "fips/year/type alignment"
        )
    return df


def county_results(
    df: pl.DataFrame, projections, percentile_method: str = "average"
) -> pl.DataFrame:
    """
    The unconditioned case counts of the counties of `df` (which has the
    importation samples, see `join_importation`): for every posterior draw the
    importation probability times the projected outbreak cases (`projections`,
    one column per county, clipped at the county's population) plus the
    expected importations.

    Adds their rank among the counties by count and by incidence (the mean over
    the draws of each county's percentile, 0 to 100, the highest being the
    highest risk), the mean, median and 95% interval over the draws, and the
    draws themselves (unconditioned_count_replicates).
    """
    # clip projections at population size
    projections = jnp.minimum(projections, jnp.array(df["pop"]).reshape(1, -1))

    importation_probability_samples = np.vstack(
        df["importation_probability_samples"].to_numpy()  # type: ignore
    ).T
    importation_count_samples = np.vstack(
        df["expected_importations_samples"].to_numpy()  # type: ignore
    ).T

    unconditioned_count_samples = (
        importation_probability_samples * np.array(projections)
        + importation_count_samples
    )

    pop = df["pop"].to_numpy().reshape(1, -1)  # shape: (1, n_counties)

    # --- Count-based rank (cases only) ---
    # unconditioned_count_samples shape: (n_draws, n_counties)
    unconditioned_count_rank_percentiles = risk_percentile(
        np.array(unconditioned_count_samples), percentile_method
    )

    # --- Incidence-based rank (cases / population) ---
    unconditioned_incidence_samples = (
        unconditioned_count_samples / pop
    )  # shape: (n_draws, n_counties)
    unconditioned_incidence_rank_percentiles = risk_percentile(
        np.array(unconditioned_incidence_samples), percentile_method
    )

    return df.with_columns(
        unconditioned_count_rank=pl.Series(
            unconditioned_count_rank_percentiles
        ),
        unconditioned_incidence_rank=pl.Series(
            unconditioned_incidence_rank_percentiles
        ),
        mean_unconditioned_projection=pl.Series(
            np.array(jnp.mean(unconditioned_count_samples, axis=0))
        ),
        median_unconditioned_projection=pl.Series(
            np.array(jnp.median(unconditioned_count_samples, axis=0))
        ),
        lower_unconditioned_projection=pl.Series(
            np.array(jnp.percentile(unconditioned_count_samples, 2.5, axis=0))
        ),
        upper_unconditioned_projection=pl.Series(
            np.array(jnp.percentile(unconditioned_count_samples, 97.5, axis=0))
        ),
        unconditioned_count_replicates=pl.Series(
            unconditioned_count_samples.T.tolist()
        ),
    )


def state_results(
    df: pl.DataFrame, percentile_method: str = "average"
) -> pl.DataFrame:
    """
    The unconditioned case counts of the states, from the counties of `df`
    (with unconditioned_count_replicates, see `county_results`): each draw's
    count of a state is the sum over its counties. One row per state and row
    type, with the state's population and its coverage weighted by
    population, the mean, median and 95% interval of its count and its rank
    among the states (the percentile, 0 to 100, of its mean count).
    """
    state_df = df.group_by(["state", "type"], maintain_order=True).agg(
        pl.col("unconditioned_count_replicates").map_batches(
            lambda x: pl.Series([np.stack(x.to_list()).sum(axis=0).tolist()]),
            return_dtype=pl.List(pl.Float64),
        ),
        total_population=pl.col("pop").sum(),
        weighted_mmr_coverage=(
            pl.col("estimated_mmr_cov_raw") * pl.col("pop")
        ).sum()
        / pl.col("pop").sum(),
    )

    # state_count_replicates shape: (n_states, n_draws) — transpose for risk_percentile
    state_count_replicates = np.vstack(
        [
            np.array(row)
            for row in state_df["unconditioned_count_replicates"].to_list()
        ]
    )  # shape: (n_states, n_draws)

    # transpose to (n_draws, n_states) so draws are on axis=0
    uncond_state_count_rank_pcts = risk_percentile(
        state_count_replicates.T, percentile_method
    )

    return state_df.with_columns(
        unconditioned_risk_rank=pl.Series(uncond_state_count_rank_pcts),
        mean_unconditioned_projection=pl.Series(
            np.mean(state_count_replicates, axis=1)
        ),
        median_unconditioned_projection=pl.Series(
            np.median(state_count_replicates, axis=1)
        ),
        lower_unconditioned_projection=pl.Series(
            np.percentile(state_count_replicates, 2.5, axis=1)
        ),
        upper_unconditioned_projection=pl.Series(
            np.percentile(state_count_replicates, 97.5, axis=1)
        ),
    )
