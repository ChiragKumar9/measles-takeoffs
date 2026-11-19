import os
from itertools import product

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import polars as pl
import yaml
from jax import clear_caches
from jax.random import PRNGKey
from matplotlib import rc
from numpyro import enable_x64, set_host_device_count
from sklearn.preprocessing import StandardScaler

enable_x64()
set_host_device_count(4)  # use all cores for MCMC
from functools import partial

from numpyro.infer import MCMC, NUTS, Predictive, init_to_median

from outbreak_size import adjacency, poisson
from outbreak_size.convergence import MAX_DIVERGENCES, NEFF_MIN
from outbreak_size.convergence import fit_with_fallback as _fit_with_fallback

# script 02 holds every fit to more than r_hat (06 and 07 only need r_hat): a
# centred fit with a divergent transition or an effective sample size below
# NEFF_MIN is refit with the non-centred CAR field, like one with a high r_hat
fit_with_fallback = partial(
    _fit_with_fallback, neff_min=NEFF_MIN, max_divergences=MAX_DIVERGENCES
)
from outbreak_size.model_inputs import apply_covariate_scale, log_attack_rate
from outbreak_size.outbreak_clusters import (
    CLUSTER_NAMES,
    CLUSTER_TAGS,
    cluster_seeds_2025,
    drop_places_of_year,
    is_leave_out_fit,
    outbreak_cases_all_counties,
    places_in_cluster,
    region_members,
)
from outbreak_size.samples_io import save_samples

NUM_SAMPLES = 15
TARGET_ACCEPT_PROB = 0.85
# fits of the baseline loop whose divergences and r_hat did not pass at 0.85
# even with the non-centred CAR field (scratchpad/probe_out, 2026-11-11):
# fit label -> its target acceptance. At 0.95 the non-centred fit of the
# 2021-2025 rows with known imports has r_hat 1.009, n_eff 457 and no
# divergences (r_hat 1.016 and 3 divergences at 0.85; r_hat 1.018 and n_eff 318
# at 0.99). The centred fit of the 2021-2025 rows with all-vaccine coverage
# only had 11 divergences at 0.85 and its non-centred refit 74; at 0.95 the
# centred fit has r_hat 1.002, n_eff 1732 and no divergences, so it is not refit
TARGET_ACCEPT_PROB_BY_FIT = {
    "All covariates_[2021, 2022, 2023, 2024, 2025]_imports_known_five_year=False": 0.95,
    "All covariates_[2021, 2022, 2023, 2024, 2025]_all_vaccine_coverage_only_five_year=False": 0.95,
}

plt.rcParams["figure.figsize"] = [9, 9]
font = {"family": "Nimbus Roman", "size": 28}
rc("font", **font)

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)
CONVERGENCE_FILE = os.path.join(
    output_dir, "convergence_02_fit_hierarchical_poisson.csv"
)

# read analysis params yaml
with open("params.yml", "r") as f:
    params = yaml.safe_load(f)

# SD of the Normal(0, sd) priors on the global coefficients and the baseline
# of the inferential fits (the rolling validation and everything built on it
# use `validation_globals_prior_sd`)
INFERENTIAL_PRIOR_SD = params["inferential_globals_prior_sd"]
# the prior of both spatial scales (the CAR field and the independent county effects), the same in 02, 03, 06 and 07
SPATIAL_SIGMA_PRIOR = dist.HalfNormal(params["spatial_sigma_prior_scale"])
# the likelihood of the outcome and the scale of the vaccination covariates (see
# params.yml)
LIKELIHOOD = params["inferential_likelihood"]
COVARIATE_SCALE = params["covariate_scale"]
NON_CENTERED_CAR = params["inferential_non_centered_car"]
MMR_EFFECTIVENESS = params.get("mmr_effectiveness")
# the baseline fits are also run on the other scale, as a comparison
COVARIATE_SCALES = [
    COVARIATE_SCALE,
    "raw" if COVARIATE_SCALE == "natural" else "natural",
]

models = {
    "MMR coverage": ["estimated_mmr_cov"],
    "MMR coverage + exemptions": [
        "estimated_mmr_cov",
        "nonmed_exempt_pct",
        "med_exempt_pct",
    ],
    "All covariates": [
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
    ],
}

# the school SD fits: the school data start in 2013-14 (outbreak year 2014)
# (2014-2020 and 2021-2025: the two periods; 2014-2024 and 2025: the multi-county outbreaks are mainly those of 2025)
SCHOOL_SD_YEAR_BUCKETS = [
    [2014, 2015, 2016, 2017, 2018, 2019, 2020],
    [2021, 2022, 2023, 2024, 2025],
    [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    [2025],
]
# the school SD fits that had divergences at the default target_accept_prob get a higher one
# ((tag, first year, last year): target_accept_prob; the not pooled 2014-2020 fit had 4 divergences at 0.85;
# the pooled 2025 fit had 21 divergences at 0.85 and a non-centred refit with r_hat 1.037, n_eff 98 and 163
# divergences; at 0.95 the refit has r_hat 1.0097, n_eff 545 and no divergences, scratchpad/probe_out/sd25_c1_centred_ta095.log;
# the not pooled 2025 fit had a refit with 4 divergences at 0.85; at 0.95 it has r_hat 1.0036, n_eff 539 and no divergences,
# scratchpad/probe_out/sd25np_c1_centred_ta095.log)
SCHOOL_SD_TARGET_ACCEPT = {
    ("school_sd_not_pooled", 2014, 2020): 0.95,
    ("school_sd", 2025, 2025): 0.95,
    ("school_sd_not_pooled", 2025, 2025): 0.95,
}
# set it to "<tag>:<first year>-<last year>" (e.g. "school_sd:2025-2025") to fit only that school SD fit, and nothing else of the script,
# so that a fit whose setting was changed can be fitted again without the other fits (the files of that fit are replaced)
ONLY_SCHOOL_SD = os.environ.get("ONLY_SCHOOL_SD")


def fit_school_sd(df, adjacency_data, baseline_mean, pooled_by_state=True):
    """
    All covariates plus the SD of the school-level MMR rates of the county
    (`sd_unweighted` of data/school_vaccination_counties.csv: every school
    counts once), fit to the county-years that have it. With pooled_by_state the SD is a pooled iv, so
    its coefficient varies by state like those of the vaccination variables
    (files tagged school_sd); otherwise it is an ordinary regression covariate
    with one coefficient for all states (tagged school_sd_not_pooled). The
    settings are those of the baseline fits of the main loop (state fixed
    effect, pooling, CAR, no neighbors, no exemption cutoff, the covariate
    scale of params.yml).
    """
    tag = "school_sd" if pooled_by_state else "school_sd_not_pooled"
    school = pl.read_csv(
        os.path.join(data_dir, "school_vaccination_counties.csv"),
        schema_overrides={"county_fips": pl.String},
    ).select("county_fips", "year_end", school_sd="sd_unweighted")
    base_vars = models["All covariates"]
    assert base_vars[:3] == [
        "estimated_mmr_cov",
        "nonmed_exempt_pct",
        "med_exempt_pct",
    ]
    # as an iv it goes right after the vaccination ivs (anywhere later it
    # would be a fixed covariate); otherwise it goes last
    if pooled_by_state:
        vars = [*base_vars[:3], "school_sd", *base_vars[3:]]
    else:
        vars = [*base_vars, "school_sd"]
    for years in SCHOOL_SD_YEAR_BUCKETS:
        if (
            ONLY_SCHOOL_SD
            and ONLY_SCHOOL_SD != f"{tag}:{years[0]}-{years[-1]}"
        ):
            continue
        print(f"Fitting {years}, All covariates + school SD ({tag})")
        df_year = (
            df.filter(pl.col("year").is_in(years))
            .join(
                school,
                left_on=["fips", "year"],
                right_on=["county_fips", "year_end"],
                how="left",
                maintain_order="left",
            )
            .filter(pl.col("school_sd").is_not_null())
        )
        assert df_year.select(vars).null_count().sum_horizontal().item() == 0
        print(
            f"  {df_year.height} county-years, {df_year['fips'].n_unique()} counties, {df_year['state'].n_unique()} states"
        )
        adjacency_matrix = adjacency.assemble_adjacency_matrix(
            adjacency_data,
            df_year["fips"].unique(maintain_order=True).to_list(),
        )
        df_year = apply_covariate_scale(
            df_year, COVARIATE_SCALE, MMR_EFFECTIVENESS
        )
        covariates = df_year.select(vars)
        covariates_scaled = StandardScaler().fit_transform(
            covariates.to_numpy()
        )
        year_fixed_effects = [
            f"year_{y}" for y in years[1:]
        ]  # none for a single year
        if year_fixed_effects:
            covariates_scaled = np.hstack(
                [
                    covariates_scaled,
                    df_year.select(year_fixed_effects).to_numpy(),
                ]
            )
        covariates = pl.DataFrame(
            covariates_scaled,
            schema=[*covariates.columns, *year_fixed_effects],
        )
        ivs = poisson.count_ivs(vars)
        if pooled_by_state:
            assert ivs == 4 and covariates.columns[:4] == [
                *vars[:3],
                "school_sd",
            ]
        else:
            assert ivs == 3 and "school_sd" in covariates.columns[ivs:]
        adjacency.require_same_unit_order(
            df_year["fips"], df_year["geoid"], "02 school SD fits"
        )
        build_model = lambda non_centered_car: poisson.PartialPooledPoisson(
            iv=covariates[:, :ivs].to_jax(),
            pool=True,
            regression_covariates=covariates[:, ivs:].to_jax(),
            exposure=df_year["pop"].to_jax(),
            groups=df_year["state"],
            group_fixed_effect=True,
            spatial=True,
            spatial_units=df_year["geoid"],
            adjacency_matrix=adjacency_matrix,
            years=None,  # type: ignore
            likelihood=LIKELIHOOD,
            non_centered_car=non_centered_car,
            car_sigma=SPATIAL_SIGMA_PRIOR,
            spatial_sigma=SPATIAL_SIGMA_PRIOR,
            global_priors=dist.Normal(0, INFERENTIAL_PRIOR_SD),
            beta_priors=dist.Normal(0, INFERENTIAL_PRIOR_SD),
            baseline=dist.Normal(baseline_mean, INFERENTIAL_PRIOR_SD),
        )
        model, mcmc = fit_with_fallback(
            build_model,
            dict(
                is_observed=jnp.ones(df_year.height),
                cases=df_year["cases_outbreak"].to_jax(),
            ),
            seed=params["seed"],
            non_centered_car=NON_CENTERED_CAR,
            label=f"All covariates_{years}_{tag}",
            init_strategy=init_to_median(num_samples=NUM_SAMPLES),
            target_accept_prob=SCHOOL_SD_TARGET_ACCEPT.get(
                (tag, years[0], years[-1]), TARGET_ACCEPT_PROB
            ),
            record_path=CONVERGENCE_FILE,
        )
        mcmc.print_summary()
        posterior_samples = mcmc.get_samples()
        save_samples(
            os.path.join(
                output_dir,
                f"samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_{years}_{tag}.npz",
            ),
            posterior_samples,
        )
        predictive_samples = Predictive(model.fit, posterior_samples)(
            PRNGKey(params["seed"]),
            is_observed=jnp.ones(df_year.height),
            cases=None,
        )
        save_samples(
            os.path.join(
                output_dir,
                f"predictive_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_{years}_{tag}.npz",
            ),
            predictive_samples,
        )
        del model, mcmc, posterior_samples, predictive_samples
        clear_caches()


def fit_all_covariates_poisson_raw(df, adjacency_data, baseline_mean, years):
    """
    All covariates with the Poisson likelihood and the vaccination covariates on
    the raw scale (percentages, not transformed): the model of the baseline fits
    of the main loop (state fixed effect, pooling, CAR, no neighbors, no
    exemption cutoff) with these two settings in place of those of params.yml,
    and with the model as it was before the population exponent got a prior and
    the density was logged: the exponent is fixed at 1 (expected cases
    proportional to the population, nothing estimated) and the covariate is
    pop_density (people per km^2), not its log10. The files are tagged
    likelihood=poisson_covariate_scale=raw_pop_exponent=1_pop_density.
    """
    tag = "likelihood=poisson_covariate_scale=raw_pop_exponent=1_pop_density"
    vars = models["All covariates"].copy()
    vars[vars.index("log_population_density")] = "pop_density"
    print(f"Fitting {years}, All covariates ({tag})")
    df_year = df.filter(pl.col("year").is_in(years))
    adjacency_matrix = adjacency.assemble_adjacency_matrix(
        adjacency_data,
        df_year["fips"].unique(maintain_order=True).to_list(),
    )
    df_year = apply_covariate_scale(df_year, "raw", MMR_EFFECTIVENESS)
    covariates = df_year.select(vars)
    covariates_scaled = StandardScaler().fit_transform(covariates.to_numpy())
    year_fixed_effects = [f"year_{y}" for y in years[1:]]
    covariates_scaled = np.hstack(
        [covariates_scaled, df_year.select(year_fixed_effects).to_numpy()]
    )
    covariates = pl.DataFrame(
        covariates_scaled, schema=[*covariates.columns, *year_fixed_effects]
    )
    ivs = poisson.count_ivs(vars)
    adjacency.require_same_unit_order(
        df_year["fips"], df_year["geoid"], "02 Poisson raw fit"
    )
    build_model = lambda non_centered_car: poisson.PartialPooledPoisson(
        iv=covariates[:, :ivs].to_jax(),
        pool=True,
        regression_covariates=covariates[:, ivs:].to_jax(),
        exposure=df_year["pop"].to_jax(),
        groups=df_year["state"],
        group_fixed_effect=True,
        spatial=True,
        spatial_units=df_year["geoid"],
        adjacency_matrix=adjacency_matrix,
        years=None,  # type: ignore
        likelihood="poisson",
        pop_exponent=1,  # type: ignore
        non_centered_car=non_centered_car,
        car_sigma=SPATIAL_SIGMA_PRIOR,
        spatial_sigma=SPATIAL_SIGMA_PRIOR,
        global_priors=dist.Normal(0, INFERENTIAL_PRIOR_SD),
        beta_priors=dist.Normal(0, INFERENTIAL_PRIOR_SD),
        baseline=dist.Normal(baseline_mean, INFERENTIAL_PRIOR_SD),
    )
    model, mcmc = fit_with_fallback(
        build_model,
        dict(
            is_observed=jnp.ones(df_year.height),
            cases=df_year["cases_outbreak"].to_jax(),
        ),
        seed=params["seed"],
        non_centered_car=NON_CENTERED_CAR,
        label=f"All covariates_{years}_{tag}",
        init_strategy=init_to_median(num_samples=NUM_SAMPLES),
        target_accept_prob=TARGET_ACCEPT_PROB,
        record_path=CONVERGENCE_FILE,
    )
    mcmc.print_summary()
    posterior_samples = mcmc.get_samples()
    save_samples(
        os.path.join(
            output_dir,
            f"samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_{years}_{tag}.npz",
        ),
        posterior_samples,
    )
    predictive_samples = Predictive(model.fit, posterior_samples)(
        PRNGKey(params["seed"]),
        is_observed=jnp.ones(df_year.height),
        cases=None,
    )
    save_samples(
        os.path.join(
            output_dir,
            f"predictive_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_{years}_{tag}.npz",
        ),
        predictive_samples,
    )
    del model, mcmc, posterior_samples, predictive_samples
    clear_caches()


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

    # spatial component
    adjacency_data = pl.read_csv(
        os.path.join(data_dir, "county_adjacencies.txt"),
        separator="|",
        schema_overrides={
            "County GEOID": pl.String,
            "Neighbor GEOID": pl.String,
        },
    )

    if ONLY_SCHOOL_SD:
        fit_school_sd(
            df,
            adjacency_data,
            baseline_mean,
            pooled_by_state=ONLY_SCHOOL_SD.split(":")[0] == "school_sd",
        )
        raise SystemExit

    # The leave-one-cluster-out fits: the places of each cluster of 2025
    # outbreaks that panel d of figure_predicting_county_level_outbreaks.py
    # leaves out are dropped from 2025, one cluster at a time (the clusters are
    # found, as there, from the cases of all the counties, also those without
    # vaccine data; outbreak_size.outbreak_clusters). None is the fit on all
    # the rows.
    LEAVE_OUT: dict[str | None, list[str]] = {None: []}
    cluster_regions = region_members()
    cluster_places = outbreak_cases_all_counties(
        pl.read_csv(
            os.path.join(data_dir, "raw_merged_outbreaks_vaccinations.csv"),
            schema_overrides={"fips": pl.String},
        ),
        cluster_regions,
    )
    cluster_seeds = cluster_seeds_2025(adjacency_data, cluster_regions)
    for cluster_name in CLUSTER_NAMES:
        LEAVE_OUT[cluster_name] = places_in_cluster(
            cluster_places,
            adjacency_data,
            cluster_regions,
            2025,
            cluster_seeds[cluster_name],
        )
        in_data = df.filter(
            (pl.col("year") == 2025)
            & pl.col("fips").is_in(LEAVE_OUT[cluster_name])
        )
        print(
            f"leave-one-cluster-out, {cluster_name}: "
            f"{len(LEAVE_OUT[cluster_name])} places of the cluster, "
            f"{in_data.height} of them in the data of the fits "
            f"({in_data['cases_outbreak'].sum()} outbreak cases of 2025)"
        )

    # The baseline fits are also run on subsets of the county-years (None is
    # all of them): those with a recorded number of imports, those whose
    # coverage is the MMR coverage, and those whose coverage is the
    # all-vaccine coverage (used where the MMR coverage is missing or 0). A
    # region whose counties use both has "mixed" coverage and is in neither of
    # the last two (see coverage_source in process_vaccinations_df)
    SAMPLE_SUBSETS: dict[str | None, pl.Expr | None] = {None: None}
    if "imports_known" in df.columns:
        SAMPLE_SUBSETS["imports_known"] = pl.col("imports_known")
    else:
        print(
            "skipped: analysis_dataset.csv has no imports_known column (rerun 00 and 01), so the fits on rows with known imports were not run."
        )
    if "coverage_source" in df.columns:
        SAMPLE_SUBSETS["mmr_coverage_only"] = (
            pl.col("coverage_source") == "mmr"
        )
        SAMPLE_SUBSETS["all_vaccine_coverage_only"] = (
            pl.col("coverage_source") == "all_vaccine"
        )
    else:
        print(
            "skipped: analysis_dataset.csv has no coverage_source column (rerun 00 and 01), so the fits on one kind of coverage were not run."
        )

    year_buckets = [
        [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
        [2021, 2022, 2023, 2024, 2025],
        # [
        #     2011,
        #     2012,
        #     2013,
        #     2014,
        #     2015,
        #     2016,
        #     2017,
        #     2018,
        #     2019,
        #     2020,
        #     2021,
        #     2022,
        #     2023,
        #     2024,
        #     2025,
        # ],
    ]
    outbreaks_in_clusters = (
        None  # only used when exclude_clustered is True, which is never run
    )
    if os.path.exists(os.path.join(data_dir, "outbreaks_in_clusters.csv")):
        outbreaks_in_clusters = pl.read_csv(
            os.path.join(data_dir, "outbreaks_in_clusters.csv"),
            schema_overrides={"fips": pl.String},
        )
        outbreaks_in_clusters = outbreaks_in_clusters.filter(
            pl.col("clustered")
        )
        outbreaks_in_clusters = outbreaks_in_clusters.with_columns(
            fips_year=pl.col("fips") + pl.col("year").cast(pl.String)
        )

    for exclude_clustered in [False]:
        for me_cutoff in [None, 5.0, 10.0]:
            for use_neighbors in [False, True]:
                for state_fixed_effect in [True]:
                    for pool in [True]:
                        for five_year in [True, False]:
                            for car in [True]:
                                for model_name, base_vars in models.items():
                                    for (
                                        years,
                                        covariate_scale,
                                        sample_subset,
                                        leave_out,
                                    ) in product(
                                        year_buckets,
                                        COVARIATE_SCALES,
                                        SAMPLE_SUBSETS,
                                        LEAVE_OUT,
                                    ):
                                        vars = base_vars.copy()
                                        additional_info = ""

                                        # The other covariate scale and the
                                        # subsets of the county-years are fit
                                        # only for the plain baseline fits; their
                                        # files are tagged
                                        if (
                                            covariate_scale != COVARIATE_SCALE
                                            or sample_subset is not None
                                        ):
                                            if (
                                                me_cutoff is not None
                                                or use_neighbors
                                                or five_year
                                                or (
                                                    covariate_scale
                                                    != COVARIATE_SCALE
                                                    and sample_subset
                                                    is not None
                                                )
                                                # the subsets are of All covariates only
                                                or (
                                                    sample_subset is not None
                                                    and model_name
                                                    != "All covariates"
                                                )
                                            ):
                                                continue
                                        if covariate_scale != COVARIATE_SCALE:
                                            additional_info += f"_covariate_scale={covariate_scale}"
                                        if sample_subset is not None:
                                            additional_info += (
                                                f"_{sample_subset}"
                                            )

                                        # The leave-one-cluster-out fits are only of the plain baseline
                                        # of the years with 2025: All covariates
                                        if (
                                            leave_out is not None
                                            and not is_leave_out_fit(
                                                model_name=model_name,
                                                years=years,
                                                default_covariate_scale=covariate_scale
                                                == COVARIATE_SCALE,
                                                sample_subset=sample_subset,
                                                me_cutoff=me_cutoff,
                                                use_neighbors=use_neighbors,
                                                five_year=five_year,
                                                exclude_clustered=exclude_clustered,
                                            )
                                        ):
                                            continue

                                        # Skip me_cutoff != None unless specific conditions are met
                                        if me_cutoff is not None:
                                            if not (
                                                model_name == "All covariates"
                                                and state_fixed_effect
                                                and pool
                                                and car
                                                and not five_year
                                                and not use_neighbors
                                            ):
                                                continue

                                        # Skip five_year=True unless specific conditions are met
                                        if five_year:
                                            if not (
                                                state_fixed_effect
                                                and pool
                                                and car
                                                and not use_neighbors
                                            ):
                                                continue

                                        if use_neighbors:
                                            if (
                                                not (
                                                    state_fixed_effect
                                                    and pool
                                                    and car
                                                )
                                                or five_year
                                                # the neighbors fits are of All covariates only
                                                or model_name
                                                != "All covariates"
                                            ):
                                                continue  # doesn't make sense to use neighbors when doing five-year

                                        print(
                                            f"Fitting {years}, model: {model_name}, five_year={five_year}, CAR={car}, pool={pool}, covariate scale={covariate_scale}"
                                        )
                                        df_year = df.filter(
                                            pl.col("year").is_in(years)
                                        )
                                        if leave_out is not None:
                                            n_rows = df_year.height
                                            df_year = drop_places_of_year(
                                                df_year,
                                                LEAVE_OUT[leave_out],
                                                2025,
                                            )
                                            additional_info += f"_leave_out_{CLUSTER_TAGS[leave_out]}"
                                            print(
                                                f"  leaving out {leave_out}: "
                                                f"{n_rows - df_year.height} rows of 2025 dropped"
                                            )
                                        if me_cutoff is not None:
                                            df_year = df_year.filter(
                                                pl.col("med_exempt_pct")
                                                <= me_cutoff
                                            )
                                            additional_info += (
                                                f"_me_cutoff={me_cutoff}"
                                            )

                                        if sample_subset is not None:
                                            df_year = df_year.filter(
                                                SAMPLE_SUBSETS[sample_subset]  # type: ignore
                                            )

                                        if exclude_clustered:
                                            # it only makes sense to exclude clustered outbreaks
                                            # if we are not using neighbors and we have all the other
                                            # stuff that we need turned on
                                            if (
                                                (not use_neighbors)
                                                and state_fixed_effect
                                                and pool
                                                and (not five_year)
                                                and car
                                                and (me_cutoff is None)
                                            ):
                                                assert (
                                                    outbreaks_in_clusters
                                                    is not None
                                                ), (
                                                    "outbreaks_in_clusters.csv not provided"
                                                )
                                                additional_info += (
                                                    "_exclude_clustered"
                                                )
                                                df_year = df_year.with_columns(
                                                    fips_year=pl.col("fips")
                                                    + pl.col("year").cast(
                                                        pl.String
                                                    )
                                                )
                                                df_year = df_year.filter(
                                                    ~pl.col("fips_year").is_in(
                                                        outbreaks_in_clusters[
                                                            "fips_year"
                                                        ].to_list()
                                                    )
                                                )
                                                df_year = df_year.drop(
                                                    "fips_year"
                                                )
                                            else:
                                                continue

                                        if five_year:
                                            # adjust the vaccine measures to be five-year averages
                                            if len(vars) == 1:
                                                vars = [
                                                    "estimated_mmr_cov_5yr"
                                                ]
                                            elif len(vars) == 3:
                                                vars = [
                                                    "estimated_mmr_cov_5yr",
                                                    "nonmed_exempt_pct_5yr",
                                                    "med_exempt_pct_5yr",
                                                ]
                                            else:
                                                vars = [
                                                    "estimated_mmr_cov_5yr",
                                                    "nonmed_exempt_pct_5yr",
                                                    "med_exempt_pct_5yr",
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
                                        elif use_neighbors:
                                            # doesn't make sense to use neighbors when doing five-year
                                            # the neighbors' coverage is an
                                            # ordinary regression covariate,
                                            # added after the model's variables
                                            vars.append("neighbors_coverage")
                                            additional_info += "_use_neighbors"
                                            df_year = df_year.filter(
                                                pl.col(
                                                    "neighbors_coverage"
                                                ).is_not_null()
                                            )

                                        adjacency_matrix = adjacency.assemble_adjacency_matrix(
                                            adjacency_data,
                                            df_year["fips"]
                                            .unique(maintain_order=True)
                                            .to_list(),
                                        )

                                        df_year = apply_covariate_scale(
                                            df_year,
                                            covariate_scale,
                                            MMR_EFFECTIVENESS,
                                        )

                                        covariates = df_year.select(vars)
                                        # scale covariates
                                        scaler = StandardScaler()
                                        covariates_scaled = (
                                            scaler.fit_transform(
                                                covariates.to_numpy()
                                            )
                                        )
                                        # append year variables
                                        # take everything but the first year as the reference year
                                        year_fixed_effects = [
                                            f"year_{y}" for y in years[1:]
                                        ]
                                        covariates_scaled = np.hstack(
                                            [
                                                covariates_scaled,
                                                df_year.select(
                                                    year_fixed_effects
                                                ).to_numpy(),
                                            ]
                                        )
                                        covariates = pl.DataFrame(
                                            covariates_scaled,
                                            schema=[
                                                *covariates.columns,
                                                *year_fixed_effects,
                                            ],
                                        )

                                        # set up model
                                        ivs = poisson.count_ivs(vars)
                                        adjacency.require_same_unit_order(
                                            df_year["fips"],
                                            df_year["geoid"],
                                            "02 main grid",
                                        )
                                        build_model = (
                                            lambda non_centered_car: poisson.PartialPooledPoisson(
                                                iv=covariates[
                                                    :, :ivs
                                                ].to_jax(),
                                                pool=pool,
                                                regression_covariates=covariates[
                                                    :, ivs:
                                                ].to_jax(),
                                                exposure=df_year[
                                                    "pop"
                                                ].to_jax(),
                                                groups=df_year["state"],
                                                group_fixed_effect=state_fixed_effect,
                                                spatial=car,
                                                spatial_units=df_year["geoid"],
                                                adjacency_matrix=adjacency_matrix,
                                                years=None,  # type: ignore
                                                likelihood=LIKELIHOOD,
                                                non_centered_car=non_centered_car,
                                                car_sigma=SPATIAL_SIGMA_PRIOR,
                                                spatial_sigma=SPATIAL_SIGMA_PRIOR,
                                                global_priors=dist.Normal(
                                                    0, INFERENTIAL_PRIOR_SD
                                                ),
                                                beta_priors=dist.Normal(
                                                    0, INFERENTIAL_PRIOR_SD
                                                ),
                                                baseline=dist.Normal(
                                                    baseline_mean,
                                                    INFERENTIAL_PRIOR_SD,
                                                ),
                                            )
                                        )

                                        # fit model
                                        # refit with the non-centred CAR field if the centred fit has an
                                        # r_hat above the threshold
                                        fit_label = f"{model_name}_{years}{additional_info}_five_year={five_year}"
                                        model, mcmc = fit_with_fallback(
                                            build_model,
                                            dict(
                                                is_observed=jnp.ones(
                                                    df_year.height
                                                ),
                                                cases=df_year[
                                                    "cases_outbreak"
                                                ].to_jax(),
                                            ),
                                            seed=params["seed"],
                                            non_centered_car=NON_CENTERED_CAR,
                                            label=fit_label,
                                            init_strategy=init_to_median(
                                                num_samples=NUM_SAMPLES
                                            ),
                                            target_accept_prob=TARGET_ACCEPT_PROB_BY_FIT.get(
                                                fit_label, TARGET_ACCEPT_PROB
                                            ),
                                            record_path=CONVERGENCE_FILE,
                                        )

                                        mcmc.print_summary()

                                        # save posterior samples
                                        posterior_samples = mcmc.get_samples()

                                        save_samples(
                                            os.path.join(
                                                output_dir,
                                                f"samples_partial_pooled_poisson_state_fixed_effect={state_fixed_effect}_pool={pool}_five_year={five_year}_CAR={car}_{model_name}_{years}{additional_info}.npz",
                                            ),
                                            posterior_samples,
                                        )

                                        predictive = Predictive(
                                            model.fit, posterior_samples
                                        )
                                        predictive_samples = predictive(
                                            PRNGKey(params["seed"]),
                                            is_observed=jnp.ones(
                                                df_year.height
                                            ),
                                            cases=None,
                                        )

                                        # quick plot
                                        # plot
                                        median = jnp.median(
                                            predictive_samples["cases"], axis=0
                                        )
                                        lower = jnp.quantile(
                                            predictive_samples["cases"],
                                            0.025,
                                            axis=0,
                                        )
                                        upper = jnp.quantile(
                                            predictive_samples["cases"],
                                            0.975,
                                            axis=0,
                                        )
                                        fig, ax = plt.subplots(1, 1)
                                        ax.spines["top"].set_visible(False)
                                        ax.spines["right"].set_visible(False)
                                        ax.get_xaxis().tick_bottom()
                                        ax.get_yaxis().tick_left()
                                        ax.tick_params(
                                            axis="x", direction="out"
                                        )
                                        ax.tick_params(
                                            axis="y", direction="out"
                                        )
                                        # offset the spines
                                        for spine in ax.spines.values():
                                            spine.set_position(("outward", 5))
                                        # put the grid behind
                                        ax.set_axisbelow(True)
                                        ax.errorbar(
                                            df_year["cases_outbreak"],
                                            median,
                                            yerr=[
                                                median - lower,
                                                upper - median,
                                            ],
                                            fmt="o",
                                            ecolor="blue",
                                            c="blue",
                                            capsize=0,
                                            alpha=0.4,
                                        )
                                        ax.set_xlabel("Observed outbreak size")
                                        ax.set_ylabel(
                                            "Estimated outbreak size"
                                        )
                                        ax.set_xscale("symlog")
                                        ax.set_yscale("symlog")
                                        ax.plot(
                                            [
                                                0,
                                                1.05
                                                * max(
                                                    df_year[
                                                        "cases_outbreak"
                                                    ].max(),  # type: ignore
                                                    median.max(),  # type: ignore
                                                ),
                                            ],
                                            [
                                                0,
                                                1.05
                                                * max(
                                                    df_year[
                                                        "cases_outbreak"
                                                    ].max(),  # type: ignore
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
                                                df_year[
                                                    "cases_outbreak"
                                                ].max(),  # type: ignore
                                                median.max(),  # type: ignore
                                            ),
                                        )
                                        ax.set_ylim(
                                            -0.3,
                                            1.1
                                            * max(
                                                df_year[
                                                    "cases_outbreak"
                                                ].max(),  # type: ignore
                                                median.max(),  # type: ignore
                                            ),
                                        )

                                        # plt.show()

                                        # print out some metrics
                                        rmse = jnp.sqrt(
                                            jnp.mean(
                                                (
                                                    median
                                                    - df_year[
                                                        "cases_outbreak"
                                                    ].to_jax()
                                                )
                                                ** 2
                                            )
                                        )
                                        mae = jnp.mean(
                                            jnp.abs(
                                                median
                                                - df_year[
                                                    "cases_outbreak"
                                                ].to_jax()
                                            )
                                        )
                                        print(f"RMSE: {rmse}")
                                        print(f"MAE: {mae}")
                                        # R2
                                        print(
                                            jnp.corrcoef(
                                                df_year[
                                                    "cases_outbreak"
                                                ].to_jax(),
                                                median,
                                            )[0, 1]
                                            ** 2
                                        )
                                        # R2 in log scale
                                        log_r2 = (
                                            jnp.corrcoef(
                                                jnp.log1p(
                                                    df_year[
                                                        "cases_outbreak"
                                                    ].to_jax()
                                                ),
                                                jnp.log1p(median),
                                            )[0, 1]
                                            ** 2
                                        )
                                        print(f"Log R2: {log_r2}")
                                        plt.close()

                                        # save the predictive samples
                                        save_samples(
                                            os.path.join(
                                                output_dir,
                                                f"predictive_partial_pooled_poisson_state_fixed_effect={state_fixed_effect}_pool={pool}_five_year={five_year}_CAR={car}_{model_name}_{years}{additional_info}.npz",
                                            ),
                                            predictive_samples,
                                        )

                                        del (
                                            model,
                                            mcmc,
                                            posterior_samples,
                                            predictive,
                                            predictive_samples,
                                        )
                                        clear_caches()

    # All covariates with the effectiveness of the MMR vaccine (the share of
    # the vaccinated who are immune) estimated instead of fixed at
    # `mmr_effectiveness`, in each year bucket. The settings are those of the
    # baseline fits of the main loop (state fixed effect, pooling, CAR, no
    # neighbors, no exemption cutoff). The bounds of the prior are an
    # assumption, not literature values: 0.999 keeps the susceptible fraction
    # of a county at 100% coverage above zero.
    if COVARIATE_SCALE == "natural":
        for years in year_buckets:
            print(f"Fitting {years}, All covariates, effectiveness estimated")
            vars = models["All covariates"]
            df_year = df.filter(pl.col("year").is_in(years))
            coverage_percent = df_year["estimated_mmr_cov"].to_jax()
            adjacency_matrix = adjacency.assemble_adjacency_matrix(
                adjacency_data,
                df_year["fips"].unique(maintain_order=True).to_list(),
            )
            df_year = apply_covariate_scale(
                df_year, "natural", MMR_EFFECTIVENESS
            )
            covariates = df_year.select(vars)
            scaler = StandardScaler()
            covariates_scaled = scaler.fit_transform(covariates.to_numpy())
            # the model rebuilds the coverage column from the effectiveness;
            # at the fixed one it must give this column
            coverage_mean_sd = (scaler.mean_[0], scaler.scale_[0])  # type: ignore
            log_susceptible = np.log(
                1 - MMR_EFFECTIVENESS * np.asarray(coverage_percent) / 100
            )
            np.testing.assert_allclose(
                (log_susceptible - coverage_mean_sd[0]) / coverage_mean_sd[1],
                covariates_scaled[:, 0],
            )
            year_fixed_effects = [f"year_{y}" for y in years[1:]]
            covariates_scaled = np.hstack(
                [
                    covariates_scaled,
                    df_year.select(year_fixed_effects).to_numpy(),
                ]
            )
            covariates = pl.DataFrame(
                covariates_scaled,
                schema=[*covariates.columns, *year_fixed_effects],
            )
            ivs = poisson.count_ivs(vars)
            adjacency.require_same_unit_order(
                df_year["fips"],
                df_year["geoid"],
                "02 effectiveness fits",
            )
            build_model = (
                lambda non_centered_car: poisson.PartialPooledPoisson(
                    iv=covariates[:, :ivs].to_jax(),
                    pool=True,
                    regression_covariates=covariates[:, ivs:].to_jax(),
                    exposure=df_year["pop"].to_jax(),
                    groups=df_year["state"],
                    group_fixed_effect=True,
                    spatial=True,
                    spatial_units=df_year["geoid"],
                    adjacency_matrix=adjacency_matrix,
                    years=None,  # type: ignore
                    likelihood=LIKELIHOOD,
                    non_centered_car=non_centered_car,
                    car_sigma=SPATIAL_SIGMA_PRIOR,
                    spatial_sigma=SPATIAL_SIGMA_PRIOR,
                    global_priors=dist.Normal(0, INFERENTIAL_PRIOR_SD),
                    beta_priors=dist.Normal(0, INFERENTIAL_PRIOR_SD),
                    baseline=dist.Normal(baseline_mean, INFERENTIAL_PRIOR_SD),
                    mmr_effectiveness=dist.Uniform(0.8, 0.999),
                    coverage_percent=coverage_percent,
                    coverage_mean_sd=coverage_mean_sd,
                )
            )
            model, mcmc = fit_with_fallback(
                build_model,
                dict(
                    is_observed=jnp.ones(df_year.height),
                    cases=df_year["cases_outbreak"].to_jax(),
                ),
                seed=params["seed"],
                non_centered_car=NON_CENTERED_CAR,
                label=f"All covariates_{years}_fit_population_immunity",
                init_strategy=init_to_median(num_samples=NUM_SAMPLES),
                target_accept_prob=TARGET_ACCEPT_PROB,
                record_path=CONVERGENCE_FILE,
            )
            mcmc.print_summary()
            posterior_samples = mcmc.get_samples()
            save_samples(
                os.path.join(
                    output_dir,
                    f"samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_{years}_fit_population_immunity.npz",
                ),
                posterior_samples,
            )
            predictive_samples = Predictive(model.fit, posterior_samples)(
                PRNGKey(params["seed"]),
                is_observed=jnp.ones(df_year.height),
                cases=None,
            )
            save_samples(
                os.path.join(
                    output_dir,
                    f"predictive_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_{years}_fit_population_immunity.npz",
                ),
                predictive_samples,
            )
            del model, mcmc, posterior_samples, predictive_samples
            clear_caches()
    else:
        print(
            "skipped: the fits that estimate the MMR effectiveness need covariate_scale: natural in params.yml."
        )

    # All covariates plus the school SD, in the county-years that have school data
    fit_school_sd(df, adjacency_data, baseline_mean)
    fit_school_sd(df, adjacency_data, baseline_mean, pooled_by_state=False)

    # All covariates with the Poisson likelihood, the raw covariate scale, the
    # population exponent fixed at 1 and pop_density, in both periods (the
    # spatial effect of 2021-2025 is used by the supplemental figure of the
    # school SD panel; the coverage effect of both by the supplemental figure
    # of the coverage effect on the linear scale)
    for years in year_buckets:
        fit_all_covariates_poisson_raw(
            df, adjacency_data, baseline_mean, years
        )

    # R01 predictions panel: All covariates, 2021-2025, no pooling, with a
    # state fixed effect, no CAR term. The main loop above only fits
    # car=True, so
    # this configuration is fit here as a one-off instead of widening that
    # loop (which would refit every car=True combination too). Must run
    # before `df = df.sort("year")` below, since it isn't stable and this
    # filter needs to match the unsorted row order that
    # figure_predicting_county_level_outbreaks.py's own df.filter(year in
    # 2021-2025) uses.
    df_year = df.filter(pl.col("year").is_in([2021, 2022, 2023, 2024, 2025]))
    vars = [
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
    df_year = apply_covariate_scale(
        df_year, COVARIATE_SCALE, MMR_EFFECTIVENESS
    )
    covariates = df_year.select(vars)
    scaler = StandardScaler()
    covariates_scaled = scaler.fit_transform(covariates.to_numpy())
    r01_years = [2021, 2022, 2023, 2024, 2025]
    year_fixed_effects = [f"year_{y}" for y in r01_years[1:]]
    covariates_scaled = np.hstack(
        [
            covariates_scaled,
            df_year.select(year_fixed_effects).to_numpy(),
        ]
    )
    covariates = pl.DataFrame(
        covariates_scaled,
        schema=[*covariates.columns, *year_fixed_effects],
    )
    adjacency.require_same_unit_order(
        df_year["fips"], df_year["geoid"], "02 all years"
    )
    model = poisson.PartialPooledPoisson(
        iv=covariates[:, :3].to_jax(),
        pool=False,
        regression_covariates=covariates[:, 3:].to_jax(),
        exposure=df_year["pop"].to_jax(),
        groups=df_year["state"],
        group_fixed_effect=True,
        spatial=False,
        spatial_units=df_year["geoid"],
        adjacency_matrix=adjacency.assemble_adjacency_matrix(
            adjacency_data,
            df_year["fips"].unique(maintain_order=True).to_list(),
        ),
        years=None,  # type: ignore
        likelihood=LIKELIHOOD,
        non_centered_car=NON_CENTERED_CAR,
        global_priors=dist.Normal(0, INFERENTIAL_PRIOR_SD),
        beta_priors=dist.Normal(0, INFERENTIAL_PRIOR_SD),
        baseline=dist.Normal(baseline_mean, INFERENTIAL_PRIOR_SD),
    )
    nuts_kernel = NUTS(
        model.fit,
        init_strategy=init_to_median(num_samples=NUM_SAMPLES),
        target_accept_prob=TARGET_ACCEPT_PROB,
    )
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=2000,
        num_samples=2000,
        num_chains=4,
    )
    mcmc.run(
        PRNGKey(params["seed"]),
        is_observed=jnp.ones(df_year.height),
        cases=df_year["cases_outbreak"].to_jax(),
    )
    mcmc.print_summary()
    posterior_samples = mcmc.get_samples()
    save_samples(
        os.path.join(
            output_dir,
            f"samples_partial_pooled_poisson_state_fixed_effect=True_pool=False_five_year=False_CAR=False_All covariates_{r01_years}.npz",
        ),
        posterior_samples,
    )
    predictive = Predictive(model.fit, posterior_samples)
    predictive_samples = predictive(
        PRNGKey(params["seed"]),
        is_observed=jnp.ones(df_year.height),
        cases=None,
    )
    save_samples(
        os.path.join(
            output_dir,
            f"predictive_partial_pooled_poisson_state_fixed_effect=True_pool=False_five_year=False_CAR=False_All covariates_{r01_years}.npz",
        ),
        predictive_samples,
    )
    del (
        model,
        mcmc,
        nuts_kernel,
        posterior_samples,
        predictive,
        predictive_samples,
    )
    clear_caches()

    # run a variant of the model in the modern era where we exclude places
    # that previously had outbreaks
    df = df.sort("year", maintain_order=True)

    df = df.with_columns(
        previous_outbreak_size=pl.cum_sum("cases_outbreak").over(
            ["county", "state"]
        )
    ).with_columns(
        # want to remove the current year's outbreak size
        previous_outbreak_size=pl.col("previous_outbreak_size")
        - pl.col("cases_outbreak")
    )

    # drop places where we had previously seen the county
    df_year = df.filter(pl.col("previous_outbreak_size") == 0)
    df_year = df_year.filter(
        pl.col("year").is_in([2021, 2022, 2023, 2024, 2025])
    )

    vars = [
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

    df_year = apply_covariate_scale(
        df_year, COVARIATE_SCALE, MMR_EFFECTIVENESS
    )

    covariates = df_year.select(vars)
    # scale covariates
    scaler = StandardScaler()
    covariates_scaled = scaler.fit_transform(covariates.to_numpy())
    # append year variables
    # take everything but the first year as the reference year
    years = df_year["year"].unique().sort().to_list()
    year_fixed_effects = [f"year_{y}" for y in years[1:]]
    covariates_scaled = np.hstack(
        [
            covariates_scaled,
            df_year.select(year_fixed_effects).to_numpy(),
        ]
    )
    covariates = pl.DataFrame(
        covariates_scaled,
        schema=[
            *covariates.columns,
            *year_fixed_effects,
        ],
    )
    adjacency.require_same_unit_order(
        df_year["fips"], df_year["geoid"], "02 prior outbreaks"
    )
    build_model = lambda non_centered_car: poisson.PartialPooledPoisson(
        iv=covariates[:, :3].to_jax(),
        pool=True,
        regression_covariates=covariates[:, 3:].to_jax(),
        exposure=df_year["pop"].to_jax(),
        groups=df_year["state"],
        group_fixed_effect=True,
        spatial=True,
        spatial_units=df_year["geoid"],
        adjacency_matrix=adjacency.assemble_adjacency_matrix(
            adjacency_data,
            df_year["fips"].unique(maintain_order=True).to_list(),
        ),
        years=None,  # type: ignore
        likelihood=LIKELIHOOD,
        non_centered_car=non_centered_car,
        car_sigma=SPATIAL_SIGMA_PRIOR,
        spatial_sigma=SPATIAL_SIGMA_PRIOR,
        global_priors=dist.Normal(0, INFERENTIAL_PRIOR_SD),
        beta_priors=dist.Normal(0, INFERENTIAL_PRIOR_SD),
        baseline=dist.Normal(baseline_mean, INFERENTIAL_PRIOR_SD),
    )

    # fit model
    # refit with the non-centred CAR field if the centred fit has an
    # r_hat above the threshold
    model, mcmc = fit_with_fallback(
        build_model,
        dict(
            is_observed=jnp.ones(df_year.height),
            cases=df_year["cases_outbreak"].to_jax(),
        ),
        seed=params["seed"],
        non_centered_car=NON_CENTERED_CAR,
        label="All covariates_[2021, 2022, 2023, 2024, 2025]_no_previous_outbreak",
        init_strategy=init_to_median(num_samples=NUM_SAMPLES),
        # this fit did not converge at 0.85 (non-centred refit: r_hat 1.015, n_eff 397 and
        # 4 divergences) nor at 0.95 (2 divergences, n_eff 390) or 0.99 (r_hat 1.014) with
        # 2000 warmup draws; at 0.99 with 4000 the centred fit has r_hat 1.0067, n_eff 485
        # and no divergences (scratchpad/probe_out/np_w2_centred_ta099_warm4000.log); the
        # 2000 draws per chain that are kept are the same
        target_accept_prob=0.99,
        num_warmup=4000,
        record_path=CONVERGENCE_FILE,
    )
    mcmc.print_summary()
    posterior_samples = mcmc.get_samples()
    save_samples(
        os.path.join(
            output_dir,
            "samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_[2021, 2022, 2023, 2024, 2025]_no_previous_outbreak.npz",
        ),
        posterior_samples,
    )

    # run a variant where we use the previous outbreak size as a predictor
    df_year = df.filter(pl.col("year").is_in([2021, 2022, 2023, 2024, 2025]))
    vars = [
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
        "previous_outbreak_size",
    ]

    df_year = apply_covariate_scale(
        df_year, COVARIATE_SCALE, MMR_EFFECTIVENESS
    )

    covariates = df_year.select(vars)
    # scale covariates
    scaler = StandardScaler()
    covariates_scaled = scaler.fit_transform(covariates.to_numpy())
    # append year variables
    # take everything but the first year as the reference year
    years = [2021, 2022, 2023, 2024, 2025]
    year_fixed_effects = [f"year_{y}" for y in years[1:]]
    covariates_scaled = np.hstack(
        [
            covariates_scaled,
            df_year.select(year_fixed_effects).to_numpy(),
        ]
    )
    covariates = pl.DataFrame(
        covariates_scaled,
        schema=[
            *covariates.columns,
            *year_fixed_effects,
        ],
    )
    adjacency.require_same_unit_order(
        df_year["fips"], df_year["geoid"], "02 sensitivity"
    )
    build_model = lambda non_centered_car: poisson.PartialPooledPoisson(
        iv=covariates[:, :3].to_jax(),
        pool=True,
        regression_covariates=covariates[:, 3:].to_jax(),
        exposure=df_year["pop"].to_jax(),
        groups=df_year["state"],
        group_fixed_effect=True,
        spatial=True,
        spatial_units=df_year["geoid"],
        adjacency_matrix=adjacency.assemble_adjacency_matrix(
            adjacency_data,
            df_year["fips"].unique(maintain_order=True).to_list(),
        ),
        years=None,  # type: ignore
        likelihood=LIKELIHOOD,
        non_centered_car=non_centered_car,
        car_sigma=SPATIAL_SIGMA_PRIOR,
        spatial_sigma=SPATIAL_SIGMA_PRIOR,
        global_priors=dist.Normal(0, INFERENTIAL_PRIOR_SD),
        beta_priors=dist.Normal(0, INFERENTIAL_PRIOR_SD),
        baseline=dist.Normal(baseline_mean, INFERENTIAL_PRIOR_SD),
    )

    # fit model
    # refit with the non-centred CAR field if the centred fit has an
    # r_hat above the threshold
    model, mcmc = fit_with_fallback(
        build_model,
        dict(
            is_observed=jnp.ones(df_year.height),
            cases=df_year["cases_outbreak"].to_jax(),
        ),
        seed=params["seed"],
        non_centered_car=NON_CENTERED_CAR,
        label="All covariates_[2021, 2022, 2023, 2024, 2025]_previous_outbreak_size",
        init_strategy=init_to_median(num_samples=50),
        target_accept_prob=0.8,
        record_path=CONVERGENCE_FILE,
    )
    mcmc.print_summary()
    posterior_samples = mcmc.get_samples()
    save_samples(
        os.path.join(
            output_dir,
            "samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_[2021, 2022, 2023, 2024, 2025]_previous_outbreak_size.npz",
        ),
        posterior_samples,
    )

    # sensitivity analysis: year-specific CAR effect indexed by two-year buckets
    # buckets: 2021/2022, 2023/2024, 2025
    # all covariates, pooling=True, state fixed effects=True, 2021-2025 sample
    two_year_bucket_map = {
        2021: 0,
        2022: 0,
        2023: 1,
        2024: 1,
        2025: 2,
    }
    df_year = df.filter(pl.col("year").is_in(list(two_year_bucket_map.keys())))
    vars = [
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
    df_year = apply_covariate_scale(
        df_year, COVARIATE_SCALE, MMR_EFFECTIVENESS
    )
    covariates = df_year.select(vars)
    # scale covariates
    scaler = StandardScaler()
    covariates_scaled = scaler.fit_transform(covariates.to_numpy())
    # append year variables
    # take everything but the first year as the reference year
    all_years = sorted(two_year_bucket_map.keys())
    year_fixed_effects = [f"year_{y}" for y in all_years[1:]]
    covariates_scaled = np.hstack(
        [
            covariates_scaled,
            df_year.select(year_fixed_effects).to_numpy(),
        ]
    )
    covariates = pl.DataFrame(
        covariates_scaled,
        schema=[
            *covariates.columns,
            *year_fixed_effects,
        ],
    )
    # map each observation's calendar year to its two-year bucket index
    car_year_index = df_year["year"].replace(
        pl.Series(list(two_year_bucket_map.keys())),
        pl.Series(list(two_year_bucket_map.values())),
    )
    adjacency.require_same_unit_order(
        df_year["fips"], df_year["geoid"], "02 two-year buckets"
    )
    build_model = lambda non_centered_car: poisson.PartialPooledPoisson(
        iv=covariates[:, :3].to_jax(),
        pool=True,
        regression_covariates=covariates[:, 3:].to_jax(),
        exposure=df_year["pop"].to_jax(),
        groups=df_year["state"],
        group_fixed_effect=True,
        spatial=True,
        spatial_units=df_year["geoid"],
        adjacency_matrix=adjacency.assemble_adjacency_matrix(
            adjacency_data,
            df_year["fips"].unique(maintain_order=True).to_list(),
        ),
        years=car_year_index,
        likelihood=LIKELIHOOD,
        non_centered_car=non_centered_car,
        car_sigma=SPATIAL_SIGMA_PRIOR,
        spatial_sigma=SPATIAL_SIGMA_PRIOR,
        global_priors=dist.Normal(0, INFERENTIAL_PRIOR_SD),
        beta_priors=dist.Normal(0, INFERENTIAL_PRIOR_SD),
        baseline=dist.Normal(baseline_mean, INFERENTIAL_PRIOR_SD),
    )

    # fit model
    # refit with the non-centred CAR field if the centred fit has an
    # r_hat above the threshold
    model, mcmc = fit_with_fallback(
        build_model,
        dict(
            is_observed=jnp.ones(df_year.height),
            cases=df_year["cases_outbreak"].to_jax(),
        ),
        seed=params["seed"],
        non_centered_car=NON_CENTERED_CAR,
        label="All covariates_2021_2025_CAR_by_two_year_bucket",
        init_strategy=init_to_median(num_samples=NUM_SAMPLES),
        target_accept_prob=TARGET_ACCEPT_PROB,
        record_path=CONVERGENCE_FILE,
    )
    mcmc.print_summary()
    posterior_samples = mcmc.get_samples()
    save_samples(
        os.path.join(
            output_dir,
            "samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_2021_2025_CAR_by_two_year_bucket.npz",
        ),
        posterior_samples,
    )

    # %% IMPUTED DATASET ANALYSIS
    print("\n" + "=" * 80)
    print("RUNNING MODELS ON IMPUTED DATASET")
    print("=" * 80)

    if not os.path.exists(
        os.path.join(data_dir, "analysis_dataset_imputed.csv")
    ):
        print(
            "skipped: analysis_dataset_imputed.csv not provided, so the imputed-data models were not run."
        )
        raise SystemExit(0)
    # Load imputed dataset
    df_imputed = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset_imputed.csv"),
        schema_overrides={"fips": pl.String},
    )

    # Define year buckets and model specifications
    imputed_year_buckets = [
        [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
        [2021, 2022, 2023, 2024, 2025],
    ]

    # All covariates
    vars_imputed = [
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

    for years in imputed_year_buckets:
        print(f"\nFitting imputed dataset: years {years}")
        print(
            "Model: All covariates, five_year=False, CAR=True, pool=True, state_fixed_effect=True"
        )

        # Filter to years
        df_year_imputed = df_imputed.filter(pl.col("year").is_in(years))

        # One adjacency row per spatial unit (geoid), in the order the model
        # numbers the units; a unit's fips finds its neighbours. The Iowa 2025
        # regions share the state's fips (19), which is no county, so they have
        # no neighbours. This is why this fit does not call
        # adjacency.require_same_unit_order (as the fits above do): several
        # geoids sharing one fips is expected here, and the matrix is already
        # built one row per geoid, so its rows follow the units by construction.
        counties_imputed = df_year_imputed.unique(
            subset="geoid", maintain_order=True
        )["fips"].to_list()

        # Reassemble adjacency matrix for this subset
        adjacency_matrix_subset = adjacency.assemble_adjacency_matrix(
            adjacency_data,
            counties_imputed,
        )

        # Prepare covariates
        df_year_imputed = apply_covariate_scale(
            df_year_imputed, COVARIATE_SCALE, MMR_EFFECTIVENESS
        )
        covariates_imputed = df_year_imputed.select(vars_imputed)

        # Scale covariates
        scaler = StandardScaler()
        covariates_scaled = scaler.fit_transform(covariates_imputed.to_numpy())

        # Append year fixed effects (all but first year)
        year_fixed_effects = [f"year_{y}" for y in years[1:]]
        covariates_scaled = np.hstack(
            [
                covariates_scaled,
                df_year_imputed.select(year_fixed_effects).to_numpy(),
            ]
        )

        covariates_imputed = pl.DataFrame(
            covariates_scaled,
            schema=[
                *covariates_imputed.columns,
                *year_fixed_effects,
            ],
        )

        # Set up model
        build_model = lambda non_centered_car: poisson.PartialPooledPoisson(
            iv=covariates_imputed[:, :3].to_jax(),
            pool=True,
            regression_covariates=covariates_imputed[:, 3:].to_jax(),
            exposure=df_year_imputed["pop"].to_jax(),
            groups=df_year_imputed["state"],
            group_fixed_effect=True,
            spatial=True,
            spatial_units=df_year_imputed["geoid"],
            adjacency_matrix=adjacency_matrix_subset,
            years=None,  # type: ignore
            likelihood=LIKELIHOOD,
            non_centered_car=non_centered_car,
            car_sigma=SPATIAL_SIGMA_PRIOR,
            spatial_sigma=SPATIAL_SIGMA_PRIOR,
            global_priors=dist.Normal(0, INFERENTIAL_PRIOR_SD),
            beta_priors=dist.Normal(0, INFERENTIAL_PRIOR_SD),
            baseline=dist.Normal(baseline_mean, INFERENTIAL_PRIOR_SD),
        )

        # Fit model
        # refit with the non-centred CAR field if the centred fit has an
        # r_hat above the threshold
        model, mcmc = fit_with_fallback(
            build_model,
            dict(
                is_observed=jnp.ones(df_year_imputed.height),
                cases=df_year_imputed["cases_outbreak"].to_jax(),
            ),
            seed=params["seed"],
            non_centered_car=NON_CENTERED_CAR,
            label=f"All covariates_{years}_IMPUTED",
            init_strategy=init_to_median(num_samples=NUM_SAMPLES),
            target_accept_prob=0.7,
            record_path=CONVERGENCE_FILE,
        )

        mcmc.print_summary()

        # Save posterior samples
        posterior_samples = mcmc.get_samples()
        save_samples(
            os.path.join(
                output_dir,
                f"samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_{years}_IMPUTED.npz",
            ),
            posterior_samples,
        )

        # Generate predictive samples
        predictive = Predictive(model.fit, posterior_samples)
        predictive_samples = predictive(
            PRNGKey(params["seed"]),
            is_observed=jnp.ones(df_year_imputed.height),
            cases=None,
        )

        # Quick plot
        median = jnp.median(predictive_samples["cases"], axis=0)
        lower = jnp.quantile(predictive_samples["cases"], 0.025, axis=0)
        upper = jnp.quantile(predictive_samples["cases"], 0.975, axis=0)

        fig, ax = plt.subplots(1, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="out")
        for spine in ax.spines.values():
            spine.set_position(("outward", 5))
        ax.set_axisbelow(True)

        ax.errorbar(
            df_year_imputed["cases_outbreak"],
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
                * max(df_year_imputed["cases_outbreak"].max(), median.max()),  # type: ignore
            ],
            [
                0,
                1.05
                * max(df_year_imputed["cases_outbreak"].max(), median.max()),  # type: ignore
            ],
            ls="--",
            c="black",
        )
        ax.set_xlim(
            -0.3,
            1.1 * max(df_year_imputed["cases_outbreak"].max(), median.max()),  # type: ignore
        )
        ax.set_ylim(
            -0.3,
            1.1 * max(df_year_imputed["cases_outbreak"].max(), median.max()),  # type: ignore
        )

        plt.savefig(
            os.path.join(output_dir, f"predictive_check_imputed_{years}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Print metrics
        rmse = jnp.sqrt(
            jnp.mean(
                (median - df_year_imputed["cases_outbreak"].to_jax()) ** 2
            )
        )
        mae = jnp.mean(
            jnp.abs(median - df_year_imputed["cases_outbreak"].to_jax())
        )
        r2 = (
            jnp.corrcoef(
                df_year_imputed["cases_outbreak"].to_jax(),
                median,
            )[0, 1]
            ** 2
        )
        log_r2 = (
            jnp.corrcoef(
                jnp.log1p(df_year_imputed["cases_outbreak"].to_jax()),
                jnp.log1p(median),
            )[0, 1]
            ** 2
        )

        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        print(f"R²: {r2}")
        print(f"Log R²: {log_r2}")

        # Check how many imputed observations
        if "imputation_source" in df_year_imputed.columns:
            n_imputed = df_year_imputed.filter(
                pl.col("imputation_source").is_not_null()
            ).height
            print(
                f"Number of imputed observations: {n_imputed} / {df_year_imputed.height}"
            )

        # Clean up
        del (
            model,
            mcmc,
            posterior_samples,
            predictive,
            predictive_samples,
        )
        clear_caches()

    print("\n" + "=" * 80)
    print("IMPUTED DATASET ANALYSIS COMPLETE")
    print("=" * 80)
