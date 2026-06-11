import os

os.environ["XLA_FLAGS"] = "--xla_gpu_enable_fast_min_max=false"
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"

import jax

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

import pickle

import numpyro.distributions as dist
import polars as pl
import yaml
from jax.random import PRNGKey
from numpyro import enable_x64, set_host_device_count
from numpyro.infer import Predictive
from sklearn.preprocessing import StandardScaler

enable_x64()
set_host_device_count(1)

from outbreak_size import adjacency, poisson

GLOBALS_PRIOR_SD = 1

with open("params.yml", "r") as f:
    params = yaml.safe_load(f)

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)

FEATURE_COLS = [
    "estimated_mmr_cov",
    "nonmed_exempt_pct",
    "med_exempt_pct",
    "fraction_under_5",
    "fraction_school_aged",
    "avg_house_size",
    "std_house_size",
    "fraction_public",
    "fraction_white",
    "fraction_graduate_degree",
    "pop_density",
    "log_income",
]

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

FUTURE_YEARS = [2026, 2027, 2028, 2029, 2030]

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Step 1: Fit scaler on training data (same training set as main run)
    # -------------------------------------------------------------------------
    train = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        schema_overrides={"fips": pl.String},
    ).filter(pl.col("year") >= 2019)

    scaler = StandardScaler()
    scaler.fit(train.select(FEATURE_COLS).to_numpy())
    print("Scaler fit on training data.")

    # -------------------------------------------------------------------------
    # Step 2: Load posterior samples from main projection run (fit once, reuse)
    # -------------------------------------------------------------------------

    # posterior_samples must be saved from the main run. Load them:
    with open(
        os.path.join(
            output_dir,
            f"posterior_samples_globals_prior_sd={GLOBALS_PRIOR_SD}.pkl",
        ),
        "rb",
    ) as f:
        posterior_samples = pickle.load(f)
    print(f"Loaded posterior samples: {list(posterior_samples.keys())}")

    # -------------------------------------------------------------------------
    # Step 3: Load intervention base datasets (2026 values, held fixed)
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
                intervened_fips = intervention_base.filter(
                    pl.col("intervention_applied")
                )["fips"].to_list()

                # Build combined dataset:
                # - remove_bans: apply delta offset to vaccination scenario
                #   projected values so counties evolve forward from their
                #   reverted pre-ban starting point
                # - all other interventions: use frozen 2026 values
                if intervention == "remove_bans":
                    intervened_rows = (
                        vax_df.filter(pl.col("fips").is_in(intervened_fips))
                        .join(
                            intervention_base.filter(
                                pl.col("intervention_applied")
                            ).select(
                                [
                                    "fips",
                                    "mmr_cov_delta",
                                    "nme_delta",
                                    "intervention_applied",
                                ]
                            ),
                            on="fips",
                            how="left",
                        )
                        .with_columns(
                            [
                                (
                                    pl.col("estimated_mmr_cov")
                                    + pl.col("mmr_cov_delta")
                                )
                                .clip(0.0, 100.0)
                                .alias("estimated_mmr_cov"),
                                (
                                    pl.col("nonmed_exempt_pct")
                                    + pl.col("nme_delta")
                                )
                                .clip(0.0, 100.0)
                                .alias("nonmed_exempt_pct"),
                            ]
                        )
                        .drop(["mmr_cov_delta", "nme_delta"])
                    )
                else:
                    intervened_rows = intervention_base.filter(
                        pl.col("intervention_applied")
                    ).with_columns(pl.lit(future_year).alias("year"))

                non_intervened_rows = vax_df.filter(
                    ~pl.col("fips").is_in(intervened_fips)
                ).with_columns(intervention_applied=pl.lit(False))

                proj_df = pl.concat(
                    [intervened_rows, non_intervened_rows],
                    how="diagonal_relaxed",
                ).with_columns(in_train=pl.lit(0))

                # Stack training data on top
                combined = pl.concat(
                    [train.with_columns(in_train=pl.lit(1)), proj_df],
                    how="diagonal_relaxed",
                ).with_columns(
                    cases_outbreak=pl.col("cases_outbreak").fill_null(0)
                )

                # Scale features
                scaled = pl.DataFrame(
                    scaler.transform(combined.select(FEATURE_COLS).to_numpy()),
                    schema=FEATURE_COLS,
                )

                # Assemble adjacency matrix for this county set
                adj_matrix = adjacency.assemble_adjacency_matrix(
                    adjacency_data,
                    combined["fips"].unique(maintain_order=True).to_list(),
                )

                # Build model with new covariates, reuse posterior
                model = poisson.PartialPooledPoisson(
                    iv=scaled.select(
                        [
                            "estimated_mmr_cov",
                            "nonmed_exempt_pct",
                            "med_exempt_pct",
                        ]
                    ).to_jax(),
                    pool=True,
                    regression_covariates=scaled.select(
                        [
                            "fraction_under_5",
                            "fraction_school_aged",
                            "avg_house_size",
                            "std_house_size",
                            "fraction_public",
                            "fraction_white",
                            "fraction_graduate_degree",
                            "pop_density",
                            "log_income",
                        ]
                    ).to_jax(),
                    exposure=combined["pop"].to_jax(),
                    groups=combined["state"],
                    group_fixed_effect=False,
                    spatial=True,
                    spatial_units=combined["geoid"],
                    adjacency_matrix=adj_matrix,
                    spatial_sigma=dist.HalfNormal(0.25),
                    car_sigma=dist.HalfNormal(0.25),
                )

                # Predictive only — no MCMC refitting
                predictive = Predictive(model.fit, posterior_samples)
                pred_samples = predictive(
                    PRNGKey(params["seed"] + 1),
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
                    f"risk_projections_{intervention}_{vax_scenario}_{future_year}.pkl",
                )
                with open(out_path, "wb") as f:
                    pickle.dump(
                        {
                            "predictions": predictions,
                            "fips": proj_df["fips"].to_list(),
                            "intervention_applied": proj_df[
                                "intervention_applied"
                            ].to_list(),
                            "state": proj_df["state"].to_list(),
                            "county": proj_df["county"].to_list(),
                        },
                        f,
                    )

                print(
                    f"    Saved {out_path} — shape {predictions.shape}, "
                    f"{sum(proj_df['intervention_applied'].to_list())} intervened counties"
                )
