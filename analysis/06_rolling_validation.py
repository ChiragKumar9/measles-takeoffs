import argparse
import os

import jax.numpy as jnp
import numpy as np
import polars as pl
import yaml
from jax import clear_caches
from jax.random import PRNGKey
from numpyro import enable_x64, set_host_device_count
from scipy import stats

enable_x64()
set_host_device_count(4)  # use all cores for MCMC
from numpyro.infer import Predictive, init_to_median

from outbreak_size import adjacency
from outbreak_size.convergence import fit_with_fallback
from outbreak_size.headline_models import (
    HEADLINE_FILE_NAME,
    read_headline_models,
)
from outbreak_size.mechanistic import (
    fit_mechanistic,
    limit_to_population,
    predict_mean,
    require_populations_above_threshold,
    theory_sizes,
)
from outbreak_size.model_designs import training_years
from outbreak_size.model_inputs import (
    COVARIATE_SCALES,
    apply_covariate_scale,
    log_attack_rate,
    no_regression_covariates,
    pooled_model,
    scale_covariates,
    stack_rows,
)
from outbreak_size.poisson import LIKELIHOODS
from outbreak_size.samples_io import save_samples
from outbreak_size.validation import validation_file_name, variant_file_suffix

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)

# read analysis params yaml
with open("params.yml", "r") as f:
    params = yaml.safe_load(f)

GLOBALS_PRIOR_SD = params["validation_globals_prior_sd"]
# the likelihood of the outcome and the scale of the vaccination covariates (see
# params.yml). To run another variant, give them on the command line, for
# example: uv run analysis/06_rolling_validation.py --likelihood poisson
# --covariate-scale raw. A variant other than the negative binomial with the
# natural scale writes files whose names say which variant it is
parser = argparse.ArgumentParser()
parser.add_argument(
    "--likelihood",
    choices=LIKELIHOODS,
    default=params["validation_likelihood"],
)
parser.add_argument(
    "--covariate-scale",
    choices=COVARIATE_SCALES,
    default=params["covariate_scale"],
)
parser.add_argument(
    "--headline-2026",
    action="store_true",
    help=(
        "Fit only the headline models (chosen with "
        "06b_choose_headline_models.py) for the validation year 2026: the "
        "statistical model, trained on its window of 2019-2025, predicts "
        "projection_dataset_2026. The mechanistic model of the window of the "
        "headline mechanistic model is fitted too (and that of the "
        "statistical model's window if it is another). Run it after the "
        "headline models are chosen."
    ),
)
args = parser.parse_args()
HEADLINE_2026 = args.headline_2026
LIKELIHOOD = args.likelihood
COVARIATE_SCALE = args.covariate_scale
VARIANT_SUFFIX = variant_file_suffix(LIKELIHOOD, COVARIATE_SCALE)
NON_CENTERED_CAR = params["validation_non_centered_car"]
SPATIAL_SIGMA_SCALE = params["spatial_sigma_prior_scale"]
MMR_EFFECTIVENESS = params.get("mmr_effectiveness")
MECHANISTIC_R0 = params["mechanistic_r0"]

if __name__ == "__main__":
    print(
        f"Rolling validation variant: likelihood={LIKELIHOOD}, "
        f"covariate_scale={COVARIATE_SCALE}"
    )
    # Load dataset
    df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
    )
    # the centre of the baseline prior: the attack rate of the outbreak
    # records up to and including 2020, so that no validation year is used
    baseline_mean = log_attack_rate(df.filter(pl.col("year") <= 2020))

    headline_design = None
    headline_years: list[int] = []
    # the window of the headline mechanistic model, if it is not the statistical
    # model's: only its mechanistic fit is wanted (not a statistical fit)
    mechanistic_only_years: list[int] = []
    if HEADLINE_2026:
        # the models to validate on 2026: the headline models, as the user
        # chose them (the same for every variant of the fits), with the years
        # they are trained on before 2026
        headline = read_headline_models(
            os.path.join(output_dir, HEADLINE_FILE_NAME), fig_dir="figures"
        )
        headline_design = headline.statistical
        headline_years = training_years(headline_design.start_year, 2026)
        print(
            f"Headline model for 2026: {headline_design.name}, trained on "
            f"{headline_years[0]}-{headline_years[-1]}"
        )
        # the training data hold the cases up to 2025 only: a window that
        # needs 2026 would train on the year it predicts
        assert headline_years[-1] == 2025
        assert (
            sorted(
                df.filter(pl.col("year").is_in(headline_years))["year"]
                .unique()
                .to_list()
            )
            == headline_years
        ), "the training data lack a year of the headline model's window"
        # the headline mechanistic model, on the window it says: its fit is
        # the one scored on 2026 (and the one the discordance table ranks by)
        mechanistic_years = training_years(
            headline.mechanistic_start_year, 2026
        )
        if mechanistic_years != headline_years:
            assert (
                sorted(
                    df.filter(pl.col("year").is_in(mechanistic_years))["year"]
                    .unique()
                    .to_list()
                )
                == mechanistic_years
            ), "the training data lack a year of the mechanistic window"
            mechanistic_only_years = mechanistic_years
            print(
                f"Headline mechanistic model: {headline.mechanistic}, "
                f"fitted on {mechanistic_years[0]}-{mechanistic_years[-1]}"
            )

    # spatial component
    adjacency_data = pl.read_csv(
        os.path.join(data_dir, "county_adjacencies.txt"),
        separator="|",
        schema_overrides={
            "County GEOID": pl.String,
            "Neighbor GEOID": pl.String,
        },
    )
    all_covariates = [
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
    models = {
        "MMR coverage": ["estimated_mmr_cov"],
        "MMR coverage + exemptions": [
            "estimated_mmr_cov",
            "nonmed_exempt_pct",
            "med_exempt_pct",
        ],
        "All covariates": all_covariates,
    }
    # test the mechanistic model (separate)
    # test the three variable variations, with and without state fixed effects,
    # with and without the car term, with and without pooling
    # the mechanistic model only depends on the training years, so it is
    # fitted once for each set of years, not once for every model below
    mechanistic_done_for_years = set()
    cars, pools, model_items = (
        [True, False],
        [True, False],
        list(models.items()),
    )
    if HEADLINE_2026:
        assert headline_design is not None
        cars = [headline_design.spatial]
        pools = [headline_design.pooled]
        model_items = [
            (headline_design.covariates, models[headline_design.covariates])
        ]
    for car in cars:
        for pool in pools:
            for model_name, vars in model_items:
                print(
                    f"Running rolling validation for model: {model_name}, "
                    f"CAR: {car}, Pooling: {pool}"
                )

                # train on progressing windows of time
                train_years = [
                    [
                        2011,
                        2012,
                        2013,
                        2014,
                        2015,
                        2016,
                        2017,
                        2018,
                        2019,
                        2020,
                    ],
                    [
                        2011,
                        2012,
                        2013,
                        2014,
                        2015,
                        2016,
                        2017,
                        2018,
                        2019,
                        2020,
                        2021,
                    ],
                    [
                        2011,
                        2012,
                        2013,
                        2014,
                        2015,
                        2016,
                        2017,
                        2018,
                        2019,
                        2020,
                        2021,
                        2022,
                    ],
                    [
                        2011,
                        2012,
                        2013,
                        2014,
                        2015,
                        2016,
                        2017,
                        2018,
                        2019,
                        2020,
                        2021,
                        2022,
                        2023,
                    ],
                    [
                        2011,
                        2012,
                        2013,
                        2014,
                        2015,
                        2016,
                        2017,
                        2018,
                        2019,
                        2020,
                        2021,
                        2022,
                        2023,
                        2024,
                    ],
                    [2019, 2020],
                    [2019, 2020, 2021],
                    [2019, 2020, 2021, 2022],
                    [2019, 2020, 2021, 2022, 2023],
                    [2019, 2020, 2021, 2022, 2023, 2024],
                    [2020, 2021],
                    [2021, 2022],
                    [2022, 2023],
                    [2023, 2024],
                ]

                if HEADLINE_2026:
                    train_years = [headline_years]
                    if mechanistic_only_years:
                        train_years.append(mechanistic_only_years)

                for years in train_years:
                    # the file of the model samples of this configuration
                    output_file = os.path.join(
                        output_dir,
                        validation_file_name(
                            years,
                            pool,
                            car,
                            f"{model_name}{VARIANT_SUFFIX}",
                            GLOBALS_PRIOR_SD,
                        ),
                    )

                    print(f"Running rolling validation for years: {years}")
                    # filter dataset to only include the training years
                    train_df = df.filter(pl.col("year").is_in(years))
                    # denote as training
                    train_df = train_df.with_columns(
                        type=pl.lit("train"), cases_train=pl.lit(1)
                    )

                    # grab the test data as the entire projection dataset
                    test_df = pl.read_csv(
                        os.path.join(
                            output_dir,
                            f"projection_dataset_{years[-1] + 1}.csv",
                        ),
                        schema_overrides={"fips": pl.String},
                    )

                    # label this data with type and cases_train
                    test_df = test_df.with_columns(
                        type=pl.lit("projection"),
                        cases_train=pl.lit(0),
                    )

                    # concatenate the train and test data (any null
                    # cases_outbreak is filled with 0s as a placeholder)
                    train_df = stack_rows(train_df, test_df)

                    # we also need to test the mechanistic model
                    # (once for each set of years, not for every model)
                    if str(years) not in mechanistic_done_for_years:
                        mechanistic_done_for_years.add(str(years))

                        # The model is in outbreak_size.mechanistic (see its
                        # documentation). It is fitted to the places with
                        # observed outbreak data in the training years. It has
                        # no state or year effects: this is the comparator
                        # whose ranking of places follows their coverage alone
                        # (with the effects, the order would also depend on
                        # the state and the year).
                        train_only = train_df.filter(
                            pl.col("cases_train") == 1
                        )
                        train_lower, train_size = theory_sizes(
                            train_only["estimated_mmr_cov"].to_numpy(),
                            train_only["pop"].to_numpy(),
                            MECHANISTIC_R0,
                            MMR_EFFECTIVENESS,
                        )
                        fit = fit_mechanistic(
                            train_lower,
                            train_size,
                            train_only["cases_outbreak"].to_numpy(),
                        )
                        search = fit["search"]
                        print(
                            f"Mechanistic model, years {years}: threshold "
                            f"size {np.exp(fit['log_threshold']):.2f} cases, "
                            f"slope {fit['slope']:.3f}, "
                            f"{fit['n_parameters']} parameters, "
                            f"log-likelihood {fit['log_likelihood']:.1f}; "
                            "search: fit at the lowest and highest bound "
                            f"{search['drop_at_lowest_bound']:.1f} and "
                            f"{search['drop_at_highest_bound']:.1f} below the "
                            f"best, {len(search['other_peaks'])} other "
                            "peak(s) of the likelihood near the best"
                        )

                        # the places to predict: every row of the projection
                        # dataset of the next year
                        test_only = train_df.filter(pl.col("cases_train") == 0)
                        population_test = test_only["pop"].to_numpy()
                        coverage_test = test_only[
                            "estimated_mmr_cov"
                        ].to_numpy()
                        test_lower, test_size = theory_sizes(
                            coverage_test,
                            population_test,
                            MECHANISTIC_R0,
                            MMR_EFFECTIVENESS,
                        )
                        # The mean is not limited by the population, which is
                        # what a ranking of the places uses. The numbers of
                        # cases drawn use the mean limited to the population.
                        mean_test = predict_mean(fit, test_lower, test_size)
                        require_populations_above_threshold(
                            fit, population_test
                        )
                        mu = limit_to_population(mean_test, population_test)

                        # draw posterior predictive samples
                        samples = stats.poisson.rvs(
                            mu=mu,
                            size=(8000, mu.shape[0]),
                            random_state=np.random.default_rng(params["seed"]),
                        )

                        # The draws, and for each place its mean and its
                        # coverage as arrays of one row, so that the places
                        # can be selected like the columns of the draws.
                        save_samples(
                            os.path.join(
                                output_dir,
                                f"predictive_final_outbreak_size_predictions_{years}.npz",
                            ),
                            {
                                "cases": samples,
                                "mean": mean_test.reshape(1, -1),
                                "mean_limited_to_population": mu.reshape(
                                    1, -1
                                ),
                                "coverage": coverage_test.reshape(1, -1),
                            },
                        )

                        # these only exist when the block above ran
                        del mean_test, mu, samples

                    # the window of the best mechanistic model only needs the
                    # mechanistic fit above
                    if HEADLINE_2026 and years == mechanistic_only_years:
                        continue

                    # the model samples are fitted only when they do not exist
                    # yet; the mechanistic model above is refit (its files
                    # overwritten) on every run
                    if os.path.exists(output_file):
                        print(
                            f"Skipping: Results already exist for years={years}, pool={pool}, CAR={car}, model={model_name}"
                        )
                        continue

                    # assemble the adjacency matrix
                    adjacency_matrix = adjacency.assemble_adjacency_matrix(
                        adjacency_data,
                        train_df["fips"].unique(maintain_order=True).to_list(),  # type: ignore
                    )

                    # the vaccination covariates on the scale of the model (the
                    # mechanistic model above uses the raw coverage)
                    train_df = apply_covariate_scale(
                        train_df, COVARIATE_SCALE, MMR_EFFECTIVENESS
                    )

                    # scale the covariates with the training rows
                    covariates = scale_covariates(
                        train_df, vars, pl.col("cases_train") == 1
                    )
                    # set up model
                    if model_name != "All covariates":
                        if model_name == "MMR coverage":
                            ivs = 1
                        else:
                            ivs = 3
                        # no regression covariates
                        # use zeros for the regression covariates since the model will ignore them
                        regression_covariates = no_regression_covariates(
                            covariates.shape[0]
                        )
                    else:
                        ivs = 3
                        regression_covariates = jnp.array(
                            covariates[:, 3:].to_numpy()
                        )
                    build_model = lambda non_centered_car: pooled_model(
                        train_df,
                        covariates[:, :ivs],
                        regression_covariates,
                        pool=pool,
                        spatial=car,
                        adjacency_matrix=adjacency_matrix,
                        prior_sd=GLOBALS_PRIOR_SD,
                        baseline_mean=baseline_mean,
                        likelihood=LIKELIHOOD,
                        non_centered_car=non_centered_car,
                        spatial_sigma_scale=SPATIAL_SIGMA_SCALE,
                    )

                    # fit model
                    # print everything about this model run

                    if GLOBALS_PRIOR_SD == 10:
                        if len(years) >= 3:
                            target_accept_prob = 0.9
                        else:
                            target_accept_prob = 0.9
                    else:
                        if len(years) >= 5 and years[0] == 2019:
                            target_accept_prob = 0.8
                        else:
                            target_accept_prob = 0.85

                    print(
                        f"Fitting model for years: {years}, CAR: {car}, Pooling: {pool}, Model: {model_name}"
                    )
                    # the CAR field is sampled in the form of params.yml
                    # (centred by default); a spatial fit with an r_hat above
                    # the threshold is refit with the other form, as in 02 and
                    # 03 (the model is the same, so the forms mix in the grid).
                    # The diagnostics of every fit are recorded
                    model, mcmc = fit_with_fallback(
                        build_model,
                        dict(
                            is_observed=train_df["cases_train"].to_jax(),
                            cases=train_df["cases_outbreak"].to_jax(),
                        ),
                        seed=params["seed"],
                        non_centered_car=NON_CENTERED_CAR,
                        label=f"{model_name}_{years}_pool={pool}_CAR={car}{VARIANT_SUFFIX}",
                        spatial=car,
                        init_strategy=init_to_median(num_samples=100),
                        target_accept_prob=target_accept_prob,
                        record_path=os.path.join(
                            output_dir, "convergence_06_rolling_validation.csv"
                        ),
                    )

                    mcmc.print_summary()

                    # get posterior samples
                    posterior_samples = mcmc.get_samples()

                    # pass the posterior samples to the model to generate predictions
                    predictive = Predictive(model.fit, posterior_samples)

                    predictive_samples = predictive(
                        PRNGKey(params["seed"]),
                        is_observed=train_df["cases_train"].to_jax(),
                        cases=None,
                    )

                    # extract just the cases and take only those for the test year
                    all_predictions = predictive_samples["cases"][
                        :, (train_df["cases_train"] == 0).to_jax()
                    ]

                    save_samples(output_file, all_predictions)  # type: ignore

                    del (
                        model,
                        mcmc,
                        posterior_samples,
                        predictive,
                        predictive_samples,
                        all_predictions,
                    )
                    clear_caches()
