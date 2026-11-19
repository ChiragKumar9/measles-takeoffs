import gc
import os

os.environ["XLA_FLAGS"] = "--xla_gpu_enable_fast_min_max=false"
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"

import jax

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")


import polars as pl
import yaml
from jax.random import PRNGKey
from numpyro import enable_x64, set_host_device_count
from numpyro.infer import Predictive

enable_x64()
set_host_device_count(1)

from outbreak_size import adjacency
from outbreak_size.interventions import FEATURE_COLS, build_projection_rows
from outbreak_size.model_designs import (
    SET_COLUMNS,
    design_model,
    read_record,
    record_file_name,
)
from outbreak_size.model_inputs import (
    apply_covariate_scale,
    scale_covariates,
    stack_rows,
)
from outbreak_size.samples_io import load_samples, save_samples
from outbreak_size.validation import require_no_nulls

with open("params.yml", "r") as f:
    params = yaml.safe_load(f)

GLOBALS_PRIOR_SD = params["validation_globals_prior_sd"]
# the likelihood of the outcome and the scale of the vaccination covariates (see
# params.yml)
LIKELIHOOD = params["validation_likelihood"]
COVARIATE_SCALE = params["covariate_scale"]
MMR_EFFECTIVENESS = params.get("mmr_effectiveness")

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)

INTERVENTIONS = [
    "remove_bans",
    "ban_top_risk_states",
    "ban_top_nme_states",
    "ban_top_risk_counties",
    "improve_cases_previously_counties",
    "improve_bottom_decile_5pct",
    "improve_bottom_decile_10pct",
    "improve_top_risk_states_5pct",
    "improve_top_risk_states_10pct",
    "improve_top_risk_counties_5pct",
    "improve_top_risk_counties_10pct",
]

VACCINATION_SCENARIOS = [
    "no_change",
    "baseline",
    "targeted_worsening",
    "universal_worsening",
    "targeted_improvement",
    "universal_improvement",
]

FUTURE_YEARS = [2027, 2028, 2029, 2030]


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Step 1: the model script 07 fitted (its design is recorded next to its
    # draws; never the choice file of 06b, which may have changed since), and
    # the training data (the same as the main run; the scaler of
    # each combination is fitted on it)
    # -------------------------------------------------------------------------
    design, years = read_record(
        os.path.join(output_dir, record_file_name(GLOBALS_PRIOR_SD))
    )
    print(f"Model of script 07: {design.name}")
    print(f"  trained on {years[0]}-{years[-1]}")
    print(
        f"  vaccination terms: {SET_COLUMNS[design.covariates][0]}, "
        f"{'spatial' if design.spatial else 'no spatial term'}, "
        f"{'pooled by state' if design.pooled else 'not pooled'}"
    )
    if not design.has_exemptions:
        print(
            "WARNING: this model has no exemption terms, so the changes of the "
            "exemption rates in the scenarios have no effect on its projections "
            "(the changes of the coverage still do)"
        )
    # the cases through 2026: the years script 07 trained on include 2026
    train = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset_with_2026.csv"),
        schema_overrides={"fips": pl.String},
    ).filter(pl.col("year").is_in(years))
    assert sorted(train["year"].unique().to_list()) == years, (
        f"the training data have the years {sorted(train['year'].unique())}, "
        f"not the {years} of script 07's record"
    )

    print("Loaded training data.")

    # -------------------------------------------------------------------------
    # Step 2: Load posterior samples from main projection run (fit once, reuse)
    # -------------------------------------------------------------------------

    # posterior_samples must be saved from the main run. Load them:
    posterior_samples = load_samples(
        os.path.join(
            output_dir,
            f"posterior_samples_globals_prior_sd={GLOBALS_PRIOR_SD}.npz",
        )
    )
    print(f"Loaded posterior samples: {list(posterior_samples.keys())}")
    # the draws belong to the recorded design: the number of pooled vaccination
    # terms must agree
    n_iv = len(SET_COLUMNS[design.covariates][0])
    if posterior_samples["iv_global"].shape[-1] != n_iv:
        raise ValueError(
            f"the posterior samples have {posterior_samples['iv_global'].shape[-1]} pooled "
            f"terms but the recorded model ({design.name}) has {n_iv}: run script 07 again"
        )
    # the draws of the spatial field are white noise (non-centred) or the
    # field itself; predicting with the other form would redraw the field, so
    # the model is built in the form the draws were made in
    fitted_non_centered = "raw_spatial_effect_decentered" in posterior_samples

    # -------------------------------------------------------------------------
    # Step 3: Load intervention base datasets (the latest observed values,
    # held fixed)
    # -------------------------------------------------------------------------
    intervention_bases = {}
    for intervention in INTERVENTIONS:
        intervention_bases[intervention] = pl.read_csv(
            os.path.join(output_dir, f"intervention_base_{intervention}.csv"),
            schema_overrides={"fips": pl.String},
        )
        n_intervened = (
            intervention_bases[intervention]
            .filter(pl.col("intervention_applied"))
            .height
        )
        print(f"Loaded {intervention}: {n_intervened} intervened counties")

    # -------------------------------------------------------------------------
    # Step 4: Load adjacency data once
    # -------------------------------------------------------------------------
    adjacency_data = pl.read_csv(
        os.path.join(data_dir, "county_adjacencies.txt"),
        separator="|",
        schema_overrides={
            "County GEOID": pl.String,
            "Neighbor GEOID": pl.String,
        },
    )

    # -------------------------------------------------------------------------
    # Step 5: Run all combinations - OUTER LOOP: vaccination scenarios
    # -------------------------------------------------------------------------
    for vax_scenario in VACCINATION_SCENARIOS:
        print(f"\n{'=' * 80}")
        print(f"VACCINATION SCENARIO: {vax_scenario}")
        print(f"{'=' * 80}\n")

        for future_year in FUTURE_YEARS:
            print(f"\n{'-' * 80}")
            print(f"Year: {future_year}")
            print(f"{'-' * 80}\n")

            # Load the vaccination scenario projection for this year once
            vax_df = pl.read_csv(
                os.path.join(
                    output_dir,
                    f"projection_dataset_{vax_scenario}_{future_year}.csv",
                ),
                schema_overrides={"fips": pl.String},
            )

            for intervention in INTERVENTIONS:
                print(f"  Processing: {intervention}")

                intervention_base = intervention_bases[intervention]

                # Projection rows in the order of the scenario file (the
                # order 07 fitted the spatial effects in), with the
                # intervened counties edited in place
                proj_df = build_projection_rows(
                    vax_df, intervention_base, intervention
                ).with_columns(in_train=pl.lit(0))

                # Stack training data on top
                combined = stack_rows(
                    train.with_columns(in_train=pl.lit(1)), proj_df
                )

                # a null model input would silently give its county -1 cases
                require_no_nulls(
                    combined,
                    [*FEATURE_COLS, "pop", "state", "geoid"],
                    f"11 model inputs of {intervention} {vax_scenario} {future_year}",
                )

                # the vaccination covariates on the scale of the model, after the
                # intervention has edited the raw values
                combined = apply_covariate_scale(
                    combined, COVARIATE_SCALE, MMR_EFFECTIVENESS
                )

                # Scale features with the training rows
                scaled = scale_covariates(
                    combined, FEATURE_COLS, pl.col("in_train") == 1
                )

                # Assemble adjacency matrix for this county set
                adj_matrix = adjacency.assemble_adjacency_matrix(
                    adjacency_data,
                    combined["fips"].unique(maintain_order=True).to_list(),
                )

                # Build model with new covariates, reuse posterior
                model = design_model(
                    design,
                    combined,
                    scaled,
                    adj_matrix,
                    likelihood=LIKELIHOOD,
                    non_centered_car=fitted_non_centered,
                )

                # Predictive only — no MCMC refitting. The same key as the
                # baseline projections of 07, on rows in the same order, so
                # the Poisson draws of a county are mostly shared between the
                # baseline and the intervention. They are not exactly paired:
                # JAX's Poisson sampler for large means loops until its
                # slowest element accepts, so changing the means of the
                # intervened counties can redraw some other counties as well
                predictive = Predictive(model.fit, posterior_samples)
                pred_samples = predictive(
                    PRNGKey(params["seed"]),
                    is_observed=combined["in_train"].to_jax(),
                    cases=None,
                )

                predictions = pred_samples["cases"][
                    :, (combined["in_train"] == 0).to_jax()
                ]

                assert predictions.shape[1] == proj_df.height, (
                    f"Shape mismatch: {predictions.shape[1]} predictions but {proj_df.height} projection rows"
                )

                # Save predictions alongside county metadata so downstream
                # code always knows which counties were intervened
                out_path = os.path.join(
                    output_dir,
                    f"risk_projections_{intervention}_{vax_scenario}_{future_year}.npz",
                )
                save_samples(
                    out_path,
                    {
                        "predictions": predictions,
                        "fips": proj_df["fips"].to_list(),
                        "intervention_applied": proj_df[
                            "intervention_applied"
                        ].to_list(),
                        "state": proj_df["state"].to_list(),
                        "county": proj_df["county"].to_list(),
                    },
                )

                # JAX keeps every function it compiles, one per scenario here;
                # after about 110 of them the process runs out of memory
                # mappings ("LLVM compilation error: Cannot allocate
                # memory") although RAM is free, so free them after each file
                jax.clear_caches()
                gc.collect()

                print(
                    f"    Saved {out_path} — shape {predictions.shape}, "
                    f"{sum(proj_df['intervention_applied'].to_list())} intervened counties"
                )
