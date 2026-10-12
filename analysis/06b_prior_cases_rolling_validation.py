import os
import pickle

import jax.numpy as jnp
import numpy as np
import polars as pl
import yaml
from jax import clear_caches
from jax.random import PRNGKey
from numpyro import distributions as dist
from numpyro import enable_x64, set_host_device_count
from sklearn.preprocessing import StandardScaler

enable_x64()
set_host_device_count(4)  # use all cores for MCMC
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median

from outbreak_size import adjacency, poisson

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)

GLOBALS_PRIOR_SD = 1

ALL_COVARIATES_VARS = [
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

# base model name -> (pooled iv covariates, number of pooled ("ivs") terms).
# All non-pooled covariates beyond the pooled ivs -- here just
# prior_cases_log1p -- go into `regression_covariates` (mirrors
# 06_rolling_validation.py:50-70, 250-264: "MMR coverage" and "MMR coverage +
# exemptions" normally pass a zeros placeholder for regression_covariates,
# since they have no non-pooled covariates; here we replace that placeholder
# with the prior-cases covariate).
BASE_MODELS = {
    "All covariates": (ALL_COVARIATES_VARS, 3),
    "MMR coverage": (["estimated_mmr_cov"], 1),
    "MMR coverage + exemptions": (
        ["estimated_mmr_cov", "nonmed_exempt_pct", "med_exempt_pct"],
        3,
    ),
}

# read analysis params yaml
with open("params.yml", "r") as f:
    params = yaml.safe_load(f)


def add_prior_cases(df, hist_df, win_start, year_col):
    """Add a `prior_cases_log1p` column: log1p(sum of `hist_df["cases"]` over
    win_start <= hist_year <= row[year_col] - 1, matched on state+county).
    Rows with no matching history (including counties absent from hist_df
    entirely) get 0, matching the naive-surveillance baseline's fill_null(0)
    (table_model_performance_validation.py:1603-1617)."""
    keys = df.select(["state", "county", year_col]).unique()
    matched = keys.join(
        hist_df.select(["state", "county", "year", "cases"]).rename(
            {"year": "hist_year", "cases": "hist_cases"}
        ),
        on=["state", "county"],
        how="inner",
    ).filter(
        (pl.col("hist_year") >= win_start)
        & (pl.col("hist_year") <= pl.col(year_col) - 1)
    )
    agg = matched.group_by(
        ["state", "county", year_col], maintain_order=True
    ).agg(pl.col("hist_cases").sum().alias("prior_cases"))
    keys = keys.join(
        agg, on=["state", "county", year_col], how="left"
    ).with_columns(pl.col("prior_cases").fill_null(0.0))
    df = df.join(keys, on=["state", "county", year_col], how="left")
    return df.with_columns(
        (pl.col("prior_cases") + 1).log().alias("prior_cases_log1p")
    )


if __name__ == "__main__":
    # Load dataset
    df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
    )

    # full case history, used to build the prior_cases covariate regardless
    # of a given run's training window
    hist_df = df.select(["state", "county", "year", "cases"])

    # spatial component
    adjacency_data = pl.read_csv(
        os.path.join(data_dir, "county_adjacencies.txt"),
        separator="|",
        schema_overrides={
            "County GEOID": pl.String,
            "Neighbor GEOID": pl.String,
        },
    )

    # (base_name, car, pool) architecture combos to run with the prior-cases
    # covariate. CAR=False/pool=False and CAR=True/pool=False were added
    # after the baseline (non-prior-cases) sweep showed both outperform the
    # CAR=True/pool=True headline architecture at t=10 and t=5 (CAR=False/
    # pool=True showed no such benefit, so it's excluded).
    ARCH_COMBOS = [
        ("All covariates", True, True),
        ("MMR coverage", True, True),
        ("MMR coverage + exemptions", True, True),
        ("All covariates", False, False),
        ("MMR coverage", False, False),
        ("MMR coverage + exemptions", False, False),
        ("All covariates", True, False),
        ("MMR coverage", True, False),
        ("MMR coverage + exemptions", True, False),
    ]

    # (start_year, prior-cases window start, window-suffix name)
    # for start_year=2011 the two window definitions coincide, so only one
    # combo is run there
    window_defs = [
        (2011, 2011, "since 2011"),
        (2019, 2011, "since 2011"),
        (2019, 2019, "in-window"),
    ]
    # for the CAR=True/pool=False architecture, skip "in-window": across
    # both previously-fit architectures, it was the worst of the 3 window
    # defs for every base model at both thresholds -- no reason to expect
    # otherwise here, and CAR=True fits are the slow ones worth trimming
    combos = [
        (
            start_year,
            win_start,
            base_name,
            car,
            pool,
            f"{base_name} + prior cases {suffix}",
        )
        for base_name, car, pool in ARCH_COMBOS
        for start_year, win_start, suffix in window_defs
        if not (car and not pool and suffix == "in-window")
    ]

    for start_year, win_start, base_name, car, pool, model_variant in combos:
        base_vars, ivs = BASE_MODELS[base_name]
        print(
            f"Running rolling validation for model: {model_variant}, "
            f"start_year: {start_year}, car: {car}, pool: {pool}"
        )
        for validation_year in [2021, 2022, 2023, 2024, 2025]:
            years = list(range(start_year, validation_year))

            output_file = os.path.join(
                output_dir,
                f"samples_partial_pooled_poisson_validation_testing_{years}_pool={pool}_CAR={car}_{model_variant}_globals_prior_sd={GLOBALS_PRIOR_SD}.pkl",
            )
            if os.path.exists(output_file):
                print(f"Skipping: Results already exist for {output_file}")
                continue

            print(f"Running rolling validation for years: {years}")
            # filter dataset to only include the training years
            train_df = df.filter(pl.col("year").is_in(years))
            train_df = train_df.with_columns(
                type=pl.lit("train"), cases_train=pl.lit(1)
            )

            # grab the test data as the entire projection dataset
            test_df = pl.read_csv(
                os.path.join(
                    output_dir, f"projection_dataset_{validation_year}.csv"
                ),
                schema_overrides={"fips": pl.String},
            )
            test_df = test_df.with_columns(
                type=pl.lit("projection"), cases_train=pl.lit(0)
            )

            train_df = pl.concat([train_df, test_df], how="diagonal_relaxed")
            train_df = train_df.with_columns(
                cases_outbreak=pl.col("cases_outbreak").fill_null(0)
            )

            # prior-cases covariate: train rows use their own `year`; test
            # (projection) rows all use the validation year, since that's the
            # year being predicted (the projection file's own `year` column
            # can be a stale covariate vintage, e.g. 2020 for a 2021
            # validation county)
            train_df = train_df.with_columns(
                year_for_prior=pl.when(pl.col("type") == "train")
                .then(pl.col("year"))
                .otherwise(pl.lit(validation_year))
            )
            train_df = add_prior_cases(
                train_df, hist_df, win_start, year_col="year_for_prior"
            )

            vars = base_vars + ["prior_cases_log1p"]

            # assemble the adjacency matrix
            adjacency_matrix = adjacency.assemble_adjacency_matrix(
                adjacency_data,
                train_df["fips"].unique(maintain_order=True).to_list(),  # type: ignore
            )

            # scale the data (fit on train rows only)
            covariates = train_df.filter(pl.col("cases_train") == 1).select(
                vars
            )
            scaler = StandardScaler()
            covariates_scaled = scaler.fit_transform(covariates.to_numpy())
            covariates_scaled = np.vstack(
                [
                    covariates_scaled,
                    scaler.transform(
                        train_df.filter(pl.col("cases_train") == 0)
                        .select(vars)
                        .to_numpy()
                    ),
                ]
            )
            covariates = pl.DataFrame(
                covariates_scaled, schema=[*covariates.columns]
            )

            # ivs (from BASE_MODELS) = the pooled vaccine/exemption terms;
            # any remaining covariates (the demographic set for "All
            # covariates", or just prior_cases_log1p for the other two
            # bases) are non-pooled regression covariates
            regression_covariates = jnp.array(covariates[:, ivs:].to_numpy())

            model = poisson.PartialPooledPoisson(
                iv=covariates[:, :ivs].to_jax(),
                pool=pool,
                regression_covariates=regression_covariates,
                exposure=train_df["pop"].to_jax(),
                groups=train_df["state"],  # type: ignore
                group_fixed_effect=False,
                spatial=car,
                spatial_units=train_df["geoid"],  # type: ignore
                adjacency_matrix=adjacency_matrix,
                spatial_sigma=dist.HalfNormal(0.25),
                car_sigma=dist.HalfNormal(0.25),
                global_priors=dist.Normal(0, GLOBALS_PRIOR_SD),
                baseline=dist.Normal(0, GLOBALS_PRIOR_SD),
            )

            if len(years) >= 5 and years[0] == 2019:
                target_accept_prob = 0.8
            else:
                target_accept_prob = 0.85

            print(f"Fitting model for years: {years}, model: {model_variant}")
            nuts_kernel = NUTS(
                model.fit,
                init_strategy=init_to_median(num_samples=100),
                target_accept_prob=target_accept_prob,
            )
            mcmc = MCMC(
                nuts_kernel,
                num_warmup=2000,
                num_samples=2000,
                num_chains=4,
            )
            mcmc.run(
                PRNGKey(params["seed"]),
                is_observed=train_df["cases_train"].to_jax(),
                cases=train_df["cases_outbreak"].to_jax(),
            )

            mcmc.print_summary()

            posterior_samples = mcmc.get_samples()
            predictive = Predictive(model.fit, posterior_samples)
            predictive_samples = predictive(
                PRNGKey(params["seed"]),
                is_observed=train_df["cases_train"].to_jax(),
                cases=None,
            )

            all_predictions = predictive_samples["cases"][
                :, (train_df["cases_train"] == 0).to_jax()
            ]

            with open(output_file, "wb") as file:
                pickle.dump(all_predictions, file)  # type: ignore

            del (
                model,
                mcmc,
                nuts_kernel,
                posterior_samples,
                predictive,
                predictive_samples,
                all_predictions,
            )
            clear_caches()
