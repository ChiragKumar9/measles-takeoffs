import os
import pickle

import polars as pl
import yaml
from numpyro import enable_x64

enable_x64()

from outbreak_size.results import (
    COUNTY_COLUMNS,
    STATE_COLUMNS,
    county_results,
    join_importation,
    state_results,
)
from outbreak_size.samples_io import load_samples

output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)

percentile_method = "average"

# read analysis params yaml
with open("params.yml", "r") as f:
    params = yaml.safe_load(f)

GLOBALS_PRIOR_SD = params["validation_globals_prior_sd"]

# The first outbreak year projected: the fit has seen the cases of 2026. Its
# projection rows are the baseline scenario's values for school year 2026 (the
# last known school year, 2025-26, trended one year along each county's slope,
# as every value before a scenario is: projection_dataset_baseline_2027, the
# draws risk_projections_baseline_2027_*). The map of the validation figure,
# the table of its projections and the targets of the NME policy analyses
# (10) read these results (assembled_projection_results.csv).
FIRST_PROJECTION_YEAR = 2027

# add importation probability (the posterior of the imports through 2025
# serves every year after 2025: results.join_importation)
with open(
    os.path.join(output_dir, "empirical_importation_probability_samples.pkl"),
    "rb",
) as f:
    samples = pickle.load(f)

with open(
    os.path.join(output_dir, "empirical_importation_count_samples.pkl"),
    "rb",
) as f:
    count_samples = pickle.load(f)
import_df = pl.read_csv(
    os.path.join(output_dir, "empirical_importation_probability.csv"),
    schema_overrides={"fips": pl.String},
)


def assemble(projection_file, samples_file):
    """The county and state results of one form of the projection rows."""
    projections = load_samples(os.path.join(output_dir, samples_file))
    df = pl.read_csv(
        os.path.join(output_dir, projection_file),
        schema_overrides={
            "fips": pl.String,
            "estimated_mmr_cov_raw": pl.Float64,
        },
    )
    df = df.filter(pl.col("in_train") == 0)
    # the draws are matched to the rows by position
    if projections.shape[1] != df.height:
        raise ValueError(
            f"{samples_file} has {projections.shape[1]} columns, but "
            f"{projection_file} has {df.height} projection rows"
        )
    df = join_importation(df, import_df, samples, count_samples)

    # the ranks and summaries of the counties (the projections are clipped at
    # the population size) and of the states
    df = county_results(df, projections, percentile_method)
    state_df = state_results(df, percentile_method)
    return df, state_df


scenario, scenario_state = assemble(
    f"projection_dataset_baseline_{FIRST_PROJECTION_YEAR}.csv",
    f"risk_projections_baseline_{FIRST_PROJECTION_YEAR}_globals_prior_sd={GLOBALS_PRIOR_SD}.npz",
)

scenario.select(COUNTY_COLUMNS).write_csv(
    os.path.join(output_dir, "assembled_projection_results.csv")
)
scenario_state.select(STATE_COLUMNS).write_csv(
    os.path.join(output_dir, "state_assembled_projection_results.csv")
)
