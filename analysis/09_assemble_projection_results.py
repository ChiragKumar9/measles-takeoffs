import os
import pickle

import numpy as np
import polars as pl
from numpyro import enable_x64
from scipy.stats import rankdata

enable_x64()
import jax.numpy as jnp

output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)

percentile_method = "average"


def mean_percentile_rank(samples_2d: np.ndarray) -> np.ndarray:
    """
    For each posterior draw, rank units, shift to 0-based, convert to percentiles,
    then average percentiles across draws.

    Parameters
    ----------
    samples_2d : np.ndarray, shape (n_draws, n_units)

    Returns
    -------
    np.ndarray, shape (n_units,), values in [0, 100]
    """
    # rank each draw across units; shape stays (n_draws, n_units)
    ranked = np.apply_along_axis(
        lambda row: rankdata(row, method=percentile_method),
        axis=1,
        arr=samples_2d,
    )
    # shift to 0-based so the lowest unit gets exactly 0
    ranked = ranked - 1
    # divide each draw by that draw's max rank → percentiles in [0, 1]
    max_per_draw = ranked.max(axis=1, keepdims=True)  # shape (n_draws, 1)
    percentiles = ranked / max_per_draw  # shape (n_draws, n_units)
    # average across draws, scale to [0, 100]
    return percentiles.mean(axis=0) * 100


with open(os.path.join(output_dir, "risk_projections.pkl"), "rb") as f:
    projections = pickle.load(f)

df = pl.read_csv(
    os.path.join(output_dir, "projection_dataset_2026.csv"),
    schema_overrides={"fips": pl.String, "estimated_mmr_cov_raw": pl.Float64},
)
df = df.filter(pl.col("in_train") == 0)

# clip projections at population size
projections = jnp.minimum(projections, jnp.array(df["pop"]).reshape(1, -1))

# add importation probability
with open(
    os.path.join(output_dir, "empirical_importation_probability_samples.pkl"),
    "rb",
) as f:
    samples = pickle.load(f)
import_df = pl.read_csv(
    os.path.join(output_dir, "empirical_importation_probability.csv"),
    schema_overrides={"fips": pl.String},
)
import_df = import_df.with_columns(
    importation_probability_samples=pl.Series(
        samples["importation_probability_samples"].__array__()
    )
)
df = df.join(
    import_df.select(
        [
            "fips",
            "year",
            "type",
            "importation_probability_mean",
            "importation_probability_lower",
            "importation_probability_upper",
            "importation_probability_median",
            "importation_probability_samples",
        ]
    ),
    on=["fips", "year", "type"],
    how="left",
)

assert df["importation_probability_samples"].null_count() == 0, (
    "Missing importation probability samples after join — check fips/year/type alignment"
)

importation_probability_samples = np.vstack(
    df["importation_probability_samples"].to_numpy()  # type: ignore
).T
unconditioned_count_samples = (
    importation_probability_samples * np.array(projections)
    + importation_probability_samples
)

pop = df["pop"].to_numpy().reshape(1, -1)  # shape: (1, n_counties)

# --- Count-based rank (cases only) ---
# unconditioned_count_samples shape: (n_draws, n_counties)
unconditioned_count_rank_percentiles = mean_percentile_rank(
    np.array(unconditioned_count_samples)
)

# --- Incidence-based rank (cases / population) ---
unconditioned_incidence_samples = (
    unconditioned_count_samples / pop
)  # shape: (n_draws, n_counties)
unconditioned_incidence_rank_percentiles = mean_percentile_rank(
    np.array(unconditioned_incidence_samples)
)

df = df.with_columns(
    unconditioned_count_rank=pl.Series(unconditioned_count_rank_percentiles),
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

state_df = df.group_by(["state", "type"]).agg(
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

# state_count_replicates shape: (n_states, n_draws) — transpose for mean_percentile_rank
state_count_replicates = np.vstack(
    [
        np.array(row)
        for row in state_df["unconditioned_count_replicates"].to_list()
    ]
)  # shape: (n_states, n_draws)

# transpose to (n_draws, n_states) so draws are on axis=0
uncond_state_count_rank_pcts = mean_percentile_rank(state_count_replicates.T)

state_df = state_df.with_columns(
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

df = df.select(
    [
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
)

state_df = state_df.select(
    [
        "state",
        "type",
        "unconditioned_risk_rank",
        "mean_unconditioned_projection",
        "median_unconditioned_projection",
        "lower_unconditioned_projection",
        "upper_unconditioned_projection",
        "weighted_mmr_coverage",
    ]
)

df.write_csv(os.path.join(output_dir, "assembled_projection_results.csv"))
state_df.write_csv(
    os.path.join(output_dir, "state_assembled_projection_results.csv")
)
