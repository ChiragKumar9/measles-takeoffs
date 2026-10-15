import os
import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml
from jax import clear_caches
from jax.random import PRNGKey
from matplotlib import rc
from numpyro import enable_x64, set_host_device_count
from sklearn.preprocessing import StandardScaler

enable_x64()
set_host_device_count(4)  # use all cores for MCMC
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median

from outbreak_size import adjacency, poisson

NUM_SAMPLES = 15
TARGET_ACCEPT_PROB = 0.85

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

if __name__ == "__main__":
    # Load dataset
    df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
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
            "pop_density",
            "log_income",
        ],
    }

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
                                    for years in year_buckets:
                                        vars = base_vars.copy()
                                        additional_info = ""

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
                                                or model_name
                                                == "All covariates"
                                            ):
                                                continue  # doesn't make sense to use neighbors when doing five-year

                                        print(
                                            f"Fitting {years}, model: {model_name}, five_year={five_year}, CAR={car}, pool={pool}"
                                        )
                                        df_year = df.filter(
                                            pl.col("year").is_in(years)
                                        )
                                        if me_cutoff is not None:
                                            df_year = df_year.filter(
                                                pl.col("med_exempt_pct")
                                                <= me_cutoff
                                            )
                                            additional_info += (
                                                f"_me_cutoff={me_cutoff}"
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
                                                    "pop_density",
                                                    "log_income",
                                                ]
                                        elif use_neighbors:
                                            # doesn't make sense to use neighbors when doing five-year
                                            # we only run this model variant when using
                                            # just vaccine coverage measures
                                            if len(vars) == 1:
                                                vars.append(
                                                    "neighbors_coverage"
                                                )
                                            elif len(vars) == 3:
                                                vars.extend(
                                                    [
                                                        "neighbors_coverage",
                                                    ]
                                                )
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
                                        if len(vars) == 1:
                                            ivs = 1
                                        else:
                                            ivs = 3
                                        model = poisson.PartialPooledPoisson(
                                            iv=covariates[:, :ivs].to_jax(),
                                            pool=pool,
                                            regression_covariates=covariates[
                                                :, ivs:
                                            ].to_jax(),
                                            exposure=df_year["pop"].to_jax(),
                                            groups=df_year["state"],
                                            group_fixed_effect=state_fixed_effect,
                                            spatial=car,
                                            spatial_units=df_year["geoid"],
                                            adjacency_matrix=adjacency_matrix,
                                            years=None,  # type: ignore
                                        )

                                        # fit model
                                        nuts_kernel = NUTS(
                                            model.fit,
                                            init_strategy=init_to_median(
                                                num_samples=NUM_SAMPLES
                                            ),
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
                                            is_observed=jnp.ones(
                                                df_year.height
                                            ),
                                            cases=df_year[
                                                "cases_outbreak"
                                            ].to_jax(),
                                        )

                                        mcmc.print_summary()

                                        # save posterior samples
                                        posterior_samples = mcmc.get_samples()

                                        with open(
                                            os.path.join(
                                                output_dir,
                                                f"samples_partial_pooled_poisson_state_fixed_effect={state_fixed_effect}_pool={pool}_five_year={five_year}_CAR={car}_{model_name}_{years}{additional_info}.pkl",
                                            ),
                                            "wb",
                                        ) as file:
                                            pickle.dump(
                                                posterior_samples, file
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
                                                    median.max(),
                                                ),
                                            ],
                                            [
                                                0,
                                                1.05
                                                * max(
                                                    df_year[
                                                        "cases_outbreak"
                                                    ].max(),  # type: ignore
                                                    median.max(),
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
                                                median.max(),
                                            ),
                                        )
                                        ax.set_ylim(
                                            -0.3,
                                            1.1
                                            * max(
                                                df_year[
                                                    "cases_outbreak"
                                                ].max(),  # type: ignore
                                                median.max(),
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
                                        with open(
                                            os.path.join(
                                                output_dir,
                                                f"predictive_partial_pooled_poisson_state_fixed_effect={state_fixed_effect}_pool={pool}_five_year={five_year}_CAR={car}_{model_name}_{years}{additional_info}.pkl",
                                            ),
                                            "wb",
                                        ) as file:
                                            pickle.dump(
                                                predictive_samples, file
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
        "pop_density",
        "log_income",
    ]
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
    with open(
        os.path.join(
            output_dir,
            f"samples_partial_pooled_poisson_state_fixed_effect=True_pool=False_five_year=False_CAR=False_All covariates_{r01_years}.pkl",
        ),
        "wb",
    ) as file:
        pickle.dump(posterior_samples, file)
    predictive = Predictive(model.fit, posterior_samples)
    predictive_samples = predictive(
        PRNGKey(params["seed"]),
        is_observed=jnp.ones(df_year.height),
        cases=None,
    )
    with open(
        os.path.join(
            output_dir,
            f"predictive_partial_pooled_poisson_state_fixed_effect=True_pool=False_five_year=False_CAR=False_All covariates_{r01_years}.pkl",
        ),
        "wb",
    ) as file:
        pickle.dump(predictive_samples, file)
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
    df = df.sort("year")

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
        "pop_density",
        "log_income",
    ]

    covariates = df_year.select(vars)
    # scale covariates
    scaler = StandardScaler()
    covariates_scaled = scaler.fit_transform(covariates.to_numpy())
    # append year variables
    # take everything but the first year as the reference year
    years = df_year["year"].unique().to_list()
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
    model = poisson.PartialPooledPoisson(
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
    )

    # fit model
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
    with open(
        os.path.join(
            output_dir,
            f"samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=True_CAR=True_All covariates_[2021, 2022, 2023, 2024, 2025]_no_previous_outbreak.pkl",
        ),
        "wb",
    ) as file:
        pickle.dump(posterior_samples, file)

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
        "pop_density",
        "log_income",
        "previous_outbreak_size",
    ]

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
    model = poisson.PartialPooledPoisson(
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
    )

    # fit model
    nuts_kernel = NUTS(
        model.fit,
        init_strategy=init_to_median(num_samples=50),
        target_accept_prob=0.8,
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
    with open(
        os.path.join(
            output_dir,
            f"samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=True_CAR=True_All covariates_[2021, 2022, 2023, 2024, 2025]_previous_outbreak_size.pkl",
        ),
        "wb",
    ) as file:
        pickle.dump(posterior_samples, file)

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
        "pop_density",
        "log_income",
    ]
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
    model = poisson.PartialPooledPoisson(
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
    )

    # fit model
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
    with open(
        os.path.join(
            output_dir,
            "samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=True_CAR=True_All covariates_2021_2025_CAR_by_two_year_bucket.pkl",
        ),
        "wb",
    ) as file:
        pickle.dump(posterior_samples, file)

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
        "pop_density",
        "log_income",
    ]

    for years in imputed_year_buckets:
        print(f"\nFitting imputed dataset: years {years}")
        print(
            f"Model: All covariates, five_year=False, CAR=True, pool=True, state_fixed_effect=True"
        )

        # Filter to years
        df_year_imputed = df_imputed.filter(pl.col("year").is_in(years))

        # Get counties in this subset
        counties_imputed = (
            df_year_imputed["fips"].unique(maintain_order=True).to_list()
        )

        # Reassemble adjacency matrix for this subset
        adjacency_matrix_subset = adjacency.assemble_adjacency_matrix(
            adjacency_data,
            counties_imputed,
        )

        # Prepare covariates
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
        model = poisson.PartialPooledPoisson(
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
        )

        # Fit model
        nuts_kernel = NUTS(
            model.fit,
            init_strategy=init_to_median(num_samples=NUM_SAMPLES),
            target_accept_prob=0.7,
        )
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=2000,
            num_samples=2000,
            num_chains=4,
        )
        mcmc.run(
            PRNGKey(params["seed"]),
            is_observed=jnp.ones(df_year_imputed.height),
            cases=df_year_imputed["cases_outbreak"].to_jax(),
        )

        mcmc.print_summary()

        # Save posterior samples
        posterior_samples = mcmc.get_samples()
        with open(
            os.path.join(
                output_dir,
                f"samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_{years}_IMPUTED.pkl",
            ),
            "wb",
        ) as file:
            pickle.dump(posterior_samples, file)

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
            nuts_kernel,
            posterior_samples,
            predictive,
            predictive_samples,
        )
        clear_caches()

    print("\n" + "=" * 80)
    print("IMPUTED DATASET ANALYSIS COMPLETE")
    print("=" * 80)
