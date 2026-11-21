import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro.distributions as dist
import polars as pl
import yaml
from jax.random import PRNGKey
from matplotlib import rc
from numpyro import enable_x64, set_host_device_count
from sklearn.preprocessing import StandardScaler

enable_x64()
set_host_device_count(4)  # use all cores for MCMC
from functools import partial

from numpyro.infer import Predictive, init_to_median

from outbreak_size import adjacency, random_walk
from outbreak_size.convergence import MAX_DIVERGENCES, NEFF_MIN
from outbreak_size.convergence import fit_with_fallback as _fit_with_fallback

# script 03 holds every fit to more than r_hat, as script 02 does (06 and 07
# only need r_hat): a centred fit with a divergent transition or an effective
# sample size below NEFF_MIN is refit with the non-centred CAR field
fit_with_fallback = partial(
    _fit_with_fallback, neff_min=NEFF_MIN, max_divergences=MAX_DIVERGENCES
)
from outbreak_size.model_inputs import apply_covariate_scale, log_attack_rate
from outbreak_size.samples_io import save_samples

plt.rcParams["figure.figsize"] = [9, 9]
font = {"family": "Nimbus Roman", "size": 28}
rc("font", **font)

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)

# read analysis params yaml
with open("params.yml", "r") as f:
    params = yaml.safe_load(f)

# RESUME_03=1 skips the fits that already have a row of this version of the code (one with a min_bfmi) in the
# convergence file, so that a run that was killed (for example, out of memory) goes on with the fits it did
# not reach instead of starting again; the files of the skipped fits are not touched. Unset, every fit is run.
RESUME_03 = os.environ.get("RESUME_03") == "1"
CONVERGENCE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "outputs",
    "convergence_03_random_walk_poisson.csv",
)

# SD of the Normal(0, sd) priors on the baseline and the coefficients of the
# inferential fits (the rolling validation and everything built on it use
# `validation_globals_prior_sd`)
INFERENTIAL_PRIOR_SD = params["inferential_globals_prior_sd"]
# the prior of both spatial scales (the CAR field and the independent county effects), the same in 02, 03, 06 and 07
SPATIAL_SIGMA_PRIOR = dist.HalfNormal(params["spatial_sigma_prior_scale"])
# the likelihood of the outcome and the scale of the vaccination covariates (see
# params.yml)
LIKELIHOOD = params["inferential_likelihood"]
COVARIATE_SCALE = params["covariate_scale"]
NON_CENTERED_CAR = params["inferential_non_centered_car"]
MMR_EFFECTIVENESS = params.get("mmr_effectiveness")

if __name__ == "__main__":
    # Load dataset
    df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
    )
    # the centre of the baseline prior of every model below: the attack rate
    # of the outbreak records of the whole study period
    baseline_mean = log_attack_rate(df)
    adjacency_matrix = pl.read_csv(
        os.path.join(data_dir, "county_adjacencies.txt"),
        separator="|",
        schema_overrides={
            "County GEOID": pl.String,
            "Neighbor GEOID": pl.String,
        },
    )
    years = [
        col
        for col in df.columns
        if col.startswith("year_")
        and not col.endswith("_2025")
        and col != "year_every_two_contiguous"
    ]

    # these happen to be the four counties in Minnesota for that year (Crow Wing, Hennepin, Le Sueur, Ramsey)
    exclude_16_17 = df.filter(
        (pl.col("state") == "Minnesota") & (pl.col("year") == 2017)
    ).select("year", "county", "state")
    assert exclude_16_17.height == 4
    exclude_16_17 = exclude_16_17.with_columns(
        year_county_state=pl.col("year").cast(pl.String)
        + pl.col("county")
        + pl.col("state")
    )
    exclude_16_17.write_csv(os.path.join(output_dir, "exclude_16_17.csv"))
    done_fits = set()
    if RESUME_03:
        recorded = pl.read_csv(
            CONVERGENCE_FILE,
            schema_overrides={"label": pl.String},
            infer_schema_length=None,
        )
        done_fits = set(
            recorded.filter(pl.col("min_bfmi").is_not_null())["label"]
        )
        print(f"RESUME_03: {len(done_fits)} fits are already done")
    for exclusions_16_17 in [True, False]:
        for oddities_2025 in [True, False]:
            additional_info = ""
            if oddities_2025:
                if exclusions_16_17:
                    df = df.with_columns(
                        year_county_state=pl.col("year").cast(pl.String)
                        + pl.col("county")
                        + pl.col("state")
                    )
                    df_model = df.filter(
                        ~pl.col("year_county_state").is_in(
                            exclude_16_17["year_county_state"].to_list()
                        )
                    )

                    assembled_adjacency_matrix = (
                        adjacency.assemble_adjacency_matrix(
                            adjacency_matrix,
                            df_model["fips"]
                            .unique(maintain_order=True)
                            .to_list(),
                        )
                    )
                    additional_info = "_exclude_16_17"

                else:
                    df_model = df
                    assembled_adjacency_matrix = (
                        adjacency.assemble_adjacency_matrix(
                            adjacency_matrix,
                            df_model["fips"]
                            .unique(maintain_order=True)
                            .to_list(),
                        )
                    )
            else:
                # Filter out 2025 oddities
                df_model = df.filter(pl.col("oddities_2025") == False)

                if exclusions_16_17:
                    # Also exclude 16-17 data
                    df_model = df_model.with_columns(
                        year_county_state=pl.col("year").cast(pl.String)
                        + pl.col("county")
                        + pl.col("state")
                    )
                    df_model = df_model.filter(
                        ~pl.col("year_county_state").is_in(
                            exclude_16_17["year_county_state"].to_list()
                        )
                    )

                    assembled_adjacency_matrix = (
                        adjacency.assemble_adjacency_matrix(
                            adjacency_matrix,
                            df_model["fips"]
                            .unique(maintain_order=True)
                            .to_list(),
                        )
                    )
                    additional_info = "_exclude_16_17_no_2025_oddities"
                else:
                    assembled_adjacency_matrix = (
                        adjacency.assemble_adjacency_matrix(
                            adjacency_matrix,
                            df_model["fips"]
                            .unique(maintain_order=True)
                            .to_list(),
                        )
                    )
                    additional_info = "_no_2025_oddities"

            df_model = apply_covariate_scale(
                df_model, COVARIATE_SCALE, MMR_EFFECTIVENESS
            )

            covariates = df_model.select(
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
                ]
            )
            # standardize covariates
            scaler = StandardScaler()
            covariates = pl.DataFrame(
                scaler.fit_transform(covariates.to_numpy()),
                schema=covariates.columns,
            )
            vaccine_measures = [
                "estimated_mmr_cov",
                "nonmed_exempt_pct",
                "med_exempt_pct",
            ]
            ivs = covariates.select(vaccine_measures)
            covariates = covariates.drop(vaccine_measures)
            # set up model
            # we want to run this in a few different modes
            # we want to allow controls to be constant over time but vary by state
            # and we want to allow the controls to be constant by state but vary by time
            for mode in [
                "time_and_group_varying",
                "time_varying_only",
                # "group_varying_only",
                # "constant",
            ]:
                fit_label = (
                    f"{mode}, oddities_2025={oddities_2025}{additional_info}"
                )
                if fit_label in done_fits:
                    print(f"RESUME_03: skipping {fit_label} (already fitted)")
                    continue
                controls = jnp.hstack([df_model.select(years).to_jax()])
                if mode == "time_and_group_varying":
                    time_and_group_varying = pl.concat(
                        [ivs, covariates], how="horizontal"
                    ).to_jax()
                    group_varying_only = None
                    time_varying_only = None
                elif mode == "time_varying_only":
                    time_and_group_varying = ivs.to_jax()
                    group_varying_only = None
                    time_varying_only = covariates.to_jax()
                elif mode == "group_varying_only":
                    time_and_group_varying = ivs.to_jax()
                    group_varying_only = covariates.to_jax()
                    time_varying_only = None
                elif mode == "constant":
                    time_and_group_varying = ivs.to_jax()
                    group_varying_only = None
                    time_varying_only = None
                    controls = jnp.hstack(
                        [
                            covariates.to_jax(),
                            controls,
                        ]
                    )
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                adjacency.require_same_unit_order(
                    df_model["fips"], df_model["geoid"], "03"
                )
                build_model = (
                    lambda non_centered_car: random_walk.PartialPooledPoisson(
                        time_and_group_varying=time_and_group_varying,
                        state_vary_rw=False,
                        group_varying_only=group_varying_only,
                        time_varying_only=time_varying_only,
                        controls=controls,
                        population=df_model["pop"].to_jax(),
                        groups=df_model["state"],
                        years=df_model["year"],
                        spatial_units=df_model["geoid"],
                        adjacency_matrix=assembled_adjacency_matrix,
                        baseline=dist.Normal(
                            baseline_mean, INFERENTIAL_PRIOR_SD
                        ),
                        priors=dist.Normal(0, INFERENTIAL_PRIOR_SD),
                        likelihood=LIKELIHOOD,
                        non_centered_car=non_centered_car,
                        spatial_sigma=SPATIAL_SIGMA_PRIOR,
                    )
                )

                # fit model (refit with the non-centred CAR field if the
                # centred fit has an r_hat above the threshold)
                model, mcmc = fit_with_fallback(
                    build_model,
                    dict(cases=df_model["cases_outbreak"].to_jax()),
                    seed=params["seed"],
                    non_centered_car=NON_CENTERED_CAR,
                    label=fit_label,
                    init_strategy=init_to_median(),
                    target_accept_prob=0.98,
                    record_path=os.path.join(
                        output_dir, "convergence_03_random_walk_poisson.csv"
                    ),
                )

                mcmc.print_summary()

                # save posterior samples
                posterior_samples = mcmc.get_samples()

                save_samples(
                    os.path.join(
                        output_dir,
                        f"samples_random_walk_poisson_CAR_{mode}_2025_oddities={oddities_2025}{additional_info}.npz",
                    ),
                    posterior_samples,
                )

                predictive = Predictive(model.fit, posterior_samples)
                predictive_samples = predictive(
                    PRNGKey(params["seed"]), cases=None
                )

                # quick plot
                # plot
                median = jnp.median(predictive_samples["cases"], axis=0)
                lower = jnp.quantile(
                    predictive_samples["cases"], 0.025, axis=0
                )
                upper = jnp.quantile(
                    predictive_samples["cases"], 0.975, axis=0
                )
                fig, ax = plt.subplots(1, 1)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.get_xaxis().tick_bottom()
                ax.get_yaxis().tick_left()
                ax.tick_params(axis="x", direction="out")
                ax.tick_params(axis="y", direction="out")
                # offset the spines
                for spine in ax.spines.values():
                    spine.set_position(("outward", 5))
                # put the grid behind
                ax.set_axisbelow(True)
                ax.errorbar(
                    df_model["cases_outbreak"],
                    median,
                    yerr=[median - lower, upper - median],
                    fmt="o",
                    ecolor="blue",
                    c="blue",
                    capsize=0,
                    alpha=0.4,
                )
                ax.set_xlabel("Observed outbreak size")
                ax.set_ylabel("Estimated outbreak size")
                ax.set_xscale("symlog")
                ax.set_yscale("symlog")
                ax.plot(
                    [
                        0,
                        1.05
                        * max(
                            df_model["cases_outbreak"].max(),  # type: ignore
                            median.max(),  # type: ignore
                        ),
                    ],
                    [
                        0,
                        1.05
                        * max(
                            df_model["cases_outbreak"].max(),  # type: ignore
                            median.max(),  # type: ignore
                        ),
                    ],
                    ls="--",
                    c="black",
                )
                ax.set_xlim(
                    -0.3,
                    1.1
                    * max(
                        df_model["cases_outbreak"].max(),  # type: ignore
                        median.max(),  # type: ignore
                    ),
                )
                ax.set_ylim(
                    -0.3,
                    1.1
                    * max(
                        df_model["cases_outbreak"].max(),  # type: ignore
                        median.max(),  # type: ignore
                    ),
                )

                # plt.show()

                # print out some metrics
                rmse = jnp.sqrt(
                    jnp.mean(
                        (median - df_model["cases_outbreak"].to_jax()) ** 2
                    )
                )
                mae = jnp.mean(
                    jnp.abs(median - df_model["cases_outbreak"].to_jax())
                )
                print(f"RMSE: {rmse}")
                print(f"MAE: {mae}")
                # R2
                print(
                    jnp.corrcoef(df_model["cases_outbreak"].to_jax(), median)[
                        0, 1
                    ]
                    ** 2
                )
                # R2 in log scale
                log_r2 = (
                    jnp.corrcoef(
                        jnp.log1p(df_model["cases_outbreak"].to_jax()),
                        jnp.log1p(median),
                    )[0, 1]
                    ** 2
                )
                print(f"Log R2: {log_r2}")

                # save the predictive samples
                save_samples(
                    os.path.join(
                        output_dir,
                        f"predictive_random_walk_poisson_CAR_{mode}_2025_oddities={oddities_2025}{additional_info}.npz",
                    ),
                    predictive_samples,
                )
