from numpyro import enable_x64, set_host_device_count

enable_x64()
set_host_device_count(4)  # use all cores for MCMC

import os

import polars as pl
import yaml
from jax.random import PRNGKey
from numpyro.infer import Predictive, init_to_median

from outbreak_size import adjacency
from outbreak_size.convergence import (
    RHAT_MAX,
    convergence_summary,
    fit_with_fallback,
)
from outbreak_size.headline_models import (
    HEADLINE_FILE_NAME,
    read_headline_models,
)
from outbreak_size.model_designs import (
    design_model,
    record_file_name,
    training_years,
    write_record,
)
from outbreak_size.model_inputs import (
    IV_COLUMNS,
    REGRESSION_COLUMNS,
    apply_covariate_scale,
    log_attack_rate,
    scale_covariates,
    stack_rows,
)
from outbreak_size.samples_io import save_samples
from outbreak_size.validation import (
    require_no_nulls,
    require_same_fips,
)

# read analysis params yaml
with open("params.yml", "r") as f:
    params = yaml.safe_load(f)

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)

GLOBALS_PRIOR_SD = params["validation_globals_prior_sd"]
# the likelihood of the outcome and the scale of the vaccination covariates (see
# params.yml)
LIKELIHOOD = params["validation_likelihood"]
COVARIATE_SCALE = params["covariate_scale"]
# the projections are fit once, so the CAR field is sampled non-centred by default
# (the rolling validation, 06, starts centred and falls back; see params.yml)
NON_CENTERED_CAR = params["projection_non_centered_car"]
SPATIAL_SIGMA_SCALE = params["spatial_sigma_prior_scale"]
MMR_EFFECTIVENESS = params.get("mmr_effectiveness")
# the first outbreak year projected: the fit has seen the cases through 2026
FIRST_PROJECTION_YEAR = 2027
LAST_PROJECTION_YEAR = 2030
# the design (covariates, spatial term, pooling, training window) fitted here, for script 11
MODEL_RECORD_FILE = os.path.join(
    output_dir, record_file_name(GLOBALS_PRIOR_SD)
)

if __name__ == "__main__":
    # the model: the headline statistical model, as the user chose it
    # (06b_choose_headline_models.py), with the years it is trained on
    headline = read_headline_models(
        os.path.join(output_dir, HEADLINE_FILE_NAME), fig_dir="figures"
    )
    design = headline.statistical
    years = training_years(design.start_year, FIRST_PROJECTION_YEAR)
    print(f"Headline statistical model:\n  {design.name}")
    print(f"  trained on {years[0]}-{years[-1]}")
    print(f"  chosen on {headline.chosen_on}: {headline.rationale}")

    # read the projection rows of the first projected year (the baseline
    # scenario's: its predictions are the first year's projection, so no
    # separate base prediction is made)
    base_file = f"projection_dataset_baseline_{FIRST_PROJECTION_YEAR}.csv"
    df = pl.read_csv(
        os.path.join(output_dir, base_file),
        schema_overrides={"fips": pl.String},
    )
    # no observed rows (the cases of 2026 are in the training data), only one
    # projection row per county
    assert (df["type"] == "projection").all(), f"{base_file} has observed rows"
    base_fips = df["fips"].to_list()

    # the cases of 2011-2026 (the _with_2026 files of 00 and 01)
    train = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset_with_2026.csv"),
        schema_overrides={"fips": pl.String},
    )

    # the centre of the baseline prior: the attack rate of the outbreak
    # records up to and including 2020
    baseline_mean = log_attack_rate(train.filter(pl.col("year") <= 2020))

    train = train.filter(pl.col("year").is_in(years))
    # a year without cases in the data would train on fewer years than the
    # record of this fit says (and script 11 reads that record)
    assert sorted(train["year"].unique().to_list()) == years, (
        f"the training data have the years {sorted(train['year'].unique())}, "
        f"not {years}"
    )

    # get the columns we need
    train = train.select(
        [
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
            "log_population_density",
            "log_income",
            "fips",
            "state",
            "county",
            "year",
            "pop",
            "cases_outbreak",
            "geoid",
        ]
    ).with_columns(
        in_train=pl.lit(1),
    )

    # concatenate with the projection dataset (the cases of the projection rows
    # are placeholders)
    df = stack_rows(
        train, df.with_columns(in_train=pl.lit(0)), cases_as_int=True
    )

    # a null model input would silently give its county -1 cases
    require_no_nulls(
        df,
        [
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
            "log_population_density",
            "log_income",
            "pop",
            "state",
            "geoid",
        ],
        "07 baseline model inputs",
    )

    # scale the data with the training rows
    print("Fitting scaler on training data...")
    df = apply_covariate_scale(df, COVARIATE_SCALE, MMR_EFFECTIVENESS)
    scaled_df = scale_covariates(
        df, IV_COLUMNS + REGRESSION_COLUMNS, pl.col("in_train") == 1
    )

    # make an adjacency matrix for the spatial model

    # spatial component
    adjacency_data = pl.read_csv(
        os.path.join(data_dir, "county_adjacencies.txt"),
        separator="|",
        schema_overrides={
            "County GEOID": pl.String,
            "Neighbor GEOID": pl.String,
        },
    )

    adjacency_matrix = adjacency.assemble_adjacency_matrix(
        adjacency_data, df["fips"].unique(maintain_order=True).to_list()
    )

    build_model = lambda non_centered_car: design_model(
        design,
        df,
        scaled_df,
        adjacency_matrix,
        prior_sd=GLOBALS_PRIOR_SD,
        baseline_mean=baseline_mean,
        likelihood=LIKELIHOOD,
        non_centered_car=non_centered_car,
        spatial_sigma_scale=SPATIAL_SIGMA_SCALE,
    )

    # fit model (refit with the other form of the CAR field if this one has
    # an r_hat above the threshold, as in 02 and 03)
    model, mcmc = fit_with_fallback(
        build_model,
        dict(
            is_observed=df["in_train"].to_jax(),
            cases=df["cases_outbreak"].to_jax(),
        ),
        seed=params["seed"],
        non_centered_car=NON_CENTERED_CAR,
        label=f"{design.name} (globals prior sd {GLOBALS_PRIOR_SD})",
        spatial=design.spatial,
        init_strategy=init_to_median(num_samples=50),
        target_accept_prob=0.7,
        record_path=os.path.join(
            output_dir, "convergence_07_risk_projections.csv"
        ),
    )

    mcmc.print_summary()

    posterior_samples = mcmc.get_samples()

    # a fit that did not converge is saved under a name of its own, to inspect:
    # the draws, the record and the projections of the last good run stay as
    # they were, so that 09 and 11 never mix the files of two runs
    fit_quality = convergence_summary(mcmc)
    if fit_quality["max_rhat"] > RHAT_MAX:
        not_converged_file = os.path.join(
            output_dir,
            f"posterior_samples_globals_prior_sd={GLOBALS_PRIOR_SD}_not_converged.npz",
        )
        save_samples(not_converged_file, posterior_samples)
        raise RuntimeError(
            f"the fit of {design.name} did not converge: r_hat "
            f"{fit_quality['max_rhat']:.3f} ({fit_quality['worst_site']}), "
            f"above {RHAT_MAX}; its draws are in {not_converged_file} and "
            "nothing else was written"
        )

    save_samples(
        os.path.join(
            output_dir,
            f"posterior_samples_globals_prior_sd={GLOBALS_PRIOR_SD}.npz",
        ),
        posterior_samples,
    )
    # the design these draws belong to, for script 11
    write_record(
        MODEL_RECORD_FILE,
        design,
        years,
        dict(rationale=headline.rationale, chosen_on=headline.chosen_on),
    )

    # the form of the CAR field that was kept (the white noise of the
    # non-centred form, or the field itself); the models that predict from the
    # draws must be built the same way
    fitted_non_centered = "raw_spatial_effect_decentered" in posterior_samples

    feature_cols = [
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
        "log_population_density",
        "log_income",
    ]

    future_years = list(range(FIRST_PROJECTION_YEAR, LAST_PROJECTION_YEAR + 1))
    scenarios = [
        "baseline",
        "targeted_worsening",
        "universal_worsening",
        "targeted_improvement",
        "universal_improvement",
        "no_change",
    ]

    for scenario in scenarios:
        for future_year in future_years:
            print(f"\nRunning scenario={scenario}, year={future_year}")

            proj_df = pl.read_csv(
                os.path.join(
                    output_dir,
                    f"projection_dataset_{scenario}_{future_year}.csv",
                ),
                schema_overrides={"fips": pl.String},
            ).with_columns(in_train=pl.lit(0))

            # the spatial effects of the base fit are used by position, so
            # the rows must be the base's counties in the base's order
            require_same_fips(
                proj_df["fips"],
                base_fips,
                f"projection_dataset_{scenario}_{future_year}.csv",
            )

            combined = stack_rows(train, proj_df, cases_as_int=True)

            require_no_nulls(
                combined,
                [*feature_cols, "pop", "state", "geoid"],
                f"07 model inputs of {scenario} {future_year}",
            )
            combined = apply_covariate_scale(
                combined, COVARIATE_SCALE, MMR_EFFECTIVENESS
            )
            scaled = scale_covariates(
                combined, feature_cols, pl.col("in_train") == 1
            )

            # the same model as above on these rows (it is only used to
            # predict from the posterior draws, so its priors are not used)
            scenario_model = design_model(
                design,
                combined,
                scaled,
                adjacency_matrix,  # reuse from original run
                likelihood=LIKELIHOOD,
                non_centered_car=fitted_non_centered,
            )

            scenario_predictive = Predictive(
                scenario_model.fit, posterior_samples
            )
            scenario_samples = scenario_predictive(
                PRNGKey(params["seed"]),
                is_observed=combined["in_train"].to_jax(),
                cases=None,
            )

            future_predictions = scenario_samples["cases"][
                :, (combined["in_train"] == 0).to_jax()
            ]

            save_samples(
                os.path.join(
                    output_dir,
                    f"risk_projections_{scenario}_{future_year}_globals_prior_sd={GLOBALS_PRIOR_SD}.npz",
                ),
                future_predictions,
            )

            print(
                f"  Saved: risk_projections_{scenario}_{future_year}_globals_prior_sd={GLOBALS_PRIOR_SD}.npz  "
                f"({future_predictions.shape[1]} counties)"
            )
