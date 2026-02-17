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

# read the risk projections
with open(
    os.path.join(
        output_dir,
        "risk_projections.pkl",
    ),
    "rb",
) as f:
    projections = pickle.load(f)

# read the df used to make the risk projections
df = pl.read_csv(
    os.path.join(output_dir, "scaled_projection_dataset.csv"),
    # make sure we read fips codes as strings
    schema_overrides={"fips": pl.String, "estimated_mmr_cov_raw": pl.Float64},
)
df = df.filter(pl.col("in_train") == 0)

# make into an incidence measure
# clip at population size
projections = jnp.minimum(projections, jnp.array(df["pop"]).reshape(1, -1))
incidence = projections / df["pop"].to_numpy()

# concatenate to dataframe so we can print out counties by risk
df = df.with_columns(
    projections=pl.Series(incidence.T.__array__()),
)

# sort the dataframe by MMR coverage
# highest to lowest coverage, so lower coverages will have higher risk
# if we do ordinal sorting
df = df.sort("estimated_mmr_cov_raw", descending=True)
# extract the incidence data in this semi sorted form
incidence = jnp.array(df["projections"]).T
# now calculate the rank percentiles
percentile_projections = (
    rankdata(incidence, method=percentile_method, axis=1) - 1
)
percentile_projections = (
    percentile_projections
    / (percentile_projections.max(axis=1, keepdims=True) + 1)
    * 100
)
# put all the info into our dataframe
df = df.with_columns(
    median_conditioned_risk_rank=pl.Series(
        jnp.median(percentile_projections, axis=0).__array__()
    ),
    lower_conditioned_risk_rank=pl.Series(
        jnp.percentile(percentile_projections, 2.5, axis=0).__array__()
    ),
    upper_conditioned_risk_rank=pl.Series(
        jnp.percentile(percentile_projections, 97.5, axis=0).__array__()
    ),
    # median percentile projections
    median_conditioned_projection=pl.Series(
        jnp.median(incidence, axis=0).__array__()
    ),
    lower_conditioned_projection=pl.Series(
        jnp.percentile(incidence, 2.5, axis=0).__array__()
    ),
    upper_conditioned_projection=pl.Series(
        jnp.percentile(incidence, 97.5, axis=0).__array__()
    ),
)
# add in import probability
with open(
    os.path.join(output_dir, "empirical_importation_probability_samples.pkl"),
    "rb",
) as f:
    samples = pickle.load(f)
import_df = pl.read_csv(
    os.path.join(output_dir, "empirical_importation_probability.csv"),
    # make sure we read fips codes as strings
    schema_overrides={"fips": pl.String},
)
import_df = import_df.with_columns(
    importation_probability_samples=pl.Series(
        samples["importation_probability_samples"].__array__()
    )
)
# join with main dataframe
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
# multiply the importation probability samples with the incidence projections
importation_probability_samples = np.vstack(
    df["importation_probability_samples"].to_numpy()  # type: ignore
).T
unconditioned_risk_samples = importation_probability_samples * incidence
# calculate the rank percentiles
unconditioned_percentile_projections = (
    rankdata(unconditioned_risk_samples, method=percentile_method, axis=1) - 1
)
unconditioned_percentile_projections = (
    unconditioned_percentile_projections
    / (unconditioned_percentile_projections.max(axis=1, keepdims=True) + 1)
    * 100
)
df = df.with_columns(
    median_unconditioned_risk_rank=pl.Series(
        jnp.median(unconditioned_percentile_projections, axis=0).__array__()
    ),
    lower_unconditioned_risk_rank=pl.Series(
        jnp.percentile(
            unconditioned_percentile_projections, 2.5, axis=0
        ).__array__()
    ),
    upper_unconditioned_risk_rank=pl.Series(
        jnp.percentile(
            unconditioned_percentile_projections, 97.5, axis=0
        ).__array__()
    ),
    # median percentile projections
    median_unconditioned_projection=pl.Series(
        jnp.median(unconditioned_risk_samples, axis=0).__array__()
    ),
    lower_unconditioned_projection=pl.Series(
        jnp.percentile(unconditioned_risk_samples, 2.5, axis=0).__array__()
    ),
    upper_unconditioned_projection=pl.Series(
        jnp.percentile(unconditioned_risk_samples, 97.5, axis=0).__array__()
    ),
)

df = df.with_columns(
    conditioned_projections=pl.Series(projections.T.__array__()),
    unconditioned_projections=pl.Series(
        (projections * importation_probability_samples).T.__array__()
    ),
)

state_df = df.group_by(["state", "type"]).agg(
    pl.col("conditioned_projections").map_elements(
        lambda x: np.sum(np.stack([x]), axis=0).tolist(),
        return_dtype=pl.List(pl.Int64),
    ),
    pl.col("unconditioned_projections").map_elements(
        lambda x: np.sum(np.stack([x]), axis=0).tolist(),
        return_dtype=pl.List(pl.Float64),
    ),
    # count number of counties per state
    total_population=pl.col("pop").sum(),
    weighted_mmr_coverage=(
        pl.col("estimated_mmr_cov_raw") * pl.col("pop")
    ).sum()
    / pl.col("pop").sum(),
)

# now sum across all counties
state_df = state_df.with_columns(
    pl.col("conditioned_projections").map_elements(
        lambda x: np.array(x).sum(axis=0).tolist(),
        return_dtype=pl.List(pl.Int64),
    ),
    pl.col("unconditioned_projections").map_elements(
        lambda x: np.array(x).sum(axis=0).tolist(),
        return_dtype=pl.List(pl.Float64),
    ),
)

# make an incidence measure
state_df = state_df.with_columns(
    conditioned_incidence=pl.col("conditioned_projections")
    / pl.col("total_population"),
    unconditioned_incidence=pl.col("unconditioned_projections")
    / pl.col("total_population"),
)

# extract the incidence data to do numpy rank ordering
conditioned_incidence = np.vstack(state_df["conditioned_incidence"].to_numpy())  # type: ignore
conditioned_percentile_projections = rankdata(
    conditioned_incidence, method=percentile_method, axis=0
)
conditioned_percentile_projections = (
    conditioned_percentile_projections
    / (conditioned_percentile_projections.max(axis=0) + 1)
    * 100
)

unconditioned_incidence = np.vstack(
    state_df["unconditioned_incidence"].to_numpy()  # type: ignore
)
unconditioned_percentile_projections = rankdata(
    unconditioned_incidence, method=percentile_method, axis=0
)
unconditioned_percentile_projections = (
    unconditioned_percentile_projections
    / (unconditioned_percentile_projections.max(axis=0) + 1)
    * 100
)

# now calculate median and 95% CrI across samples
state_df = state_df.with_columns(
    median_conditioned_risk_rank=pl.Series(
        jnp.median(conditioned_percentile_projections, axis=1).__array__()
    ),
    lower_conditioned_risk_rank=pl.Series(
        jnp.percentile(
            conditioned_percentile_projections, 2.5, axis=1
        ).__array__()
    ),
    upper_conditioned_risk_rank=pl.Series(
        jnp.percentile(
            conditioned_percentile_projections, 97.5, axis=1
        ).__array__()
    ),
    # median incidences
    median_conditioned_projection=pl.Series(
        jnp.median(conditioned_incidence, axis=1).__array__()
    ),
    lower_conditioned_projection=pl.Series(
        jnp.percentile(conditioned_incidence, 2.5, axis=1).__array__()
    ),
    upper_conditioned_projection=pl.Series(
        jnp.percentile(conditioned_incidence, 97.5, axis=1).__array__()
    ),
    median_unconditioned_risk_rank=pl.Series(
        jnp.median(unconditioned_percentile_projections, axis=1).__array__()
    ),
    lower_unconditioned_risk_rank=pl.Series(
        jnp.percentile(
            unconditioned_percentile_projections, 2.5, axis=1
        ).__array__()
    ),
    upper_unconditioned_risk_rank=pl.Series(
        jnp.percentile(
            unconditioned_percentile_projections, 97.5, axis=1
        ).__array__()
    ),
    # median incidences
    median_unconditioned_projection=pl.Series(
        jnp.median(unconditioned_incidence, axis=1).__array__()
    ),
    lower_unconditioned_projection=pl.Series(
        jnp.percentile(unconditioned_incidence, 2.5, axis=1).__array__()
    ),
    upper_unconditioned_projection=pl.Series(
        jnp.percentile(unconditioned_incidence, 97.5, axis=1).__array__()
    ),
)

# take just the columns we need
df = df.select(
    [
        "fips",
        "state",
        "county",
        "year",
        "type",
        # conditioned on importation risk ranks
        "median_conditioned_risk_rank",
        "lower_conditioned_risk_rank",
        "upper_conditioned_risk_rank",
        "median_conditioned_projection",
        "lower_conditioned_projection",
        "upper_conditioned_projection",
        # unconditioned on importation risk ranks
        "median_unconditioned_risk_rank",
        "lower_unconditioned_risk_rank",
        "upper_unconditioned_risk_rank",
        "median_unconditioned_projection",
        "lower_unconditioned_projection",
        "upper_unconditioned_projection",
        # importation probabilities
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
        # conditioned on importation risk ranks
        "median_conditioned_risk_rank",
        "lower_conditioned_risk_rank",
        "upper_conditioned_risk_rank",
        "median_conditioned_projection",
        "lower_conditioned_projection",
        "upper_conditioned_projection",
        # unconditioned on importation risk ranks
        "median_unconditioned_risk_rank",
        "lower_unconditioned_risk_rank",
        "upper_unconditioned_risk_rank",
        "median_unconditioned_projection",
        "lower_unconditioned_projection",
        "upper_unconditioned_projection",
        "weighted_mmr_coverage",
    ]
)

# save to csv
df.write_csv(os.path.join(output_dir, "assembled_projection_results.csv"))

state_df.write_csv(
    os.path.join(output_dir, "state_assembled_projection_results.csv")
)
