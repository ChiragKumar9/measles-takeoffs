import os
import pickle

import jax.numpy as jnp
import polars as pl
import yaml
from numpyro import enable_x64, set_host_device_count
from sklearn.preprocessing import StandardScaler

enable_x64()
set_host_device_count(4)  # use all cores for MCMC

import numpy as np
import numpyro.distributions as dist
from jax.random import PRNGKey
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median

from outbreak_size import adjacency, poisson

# read analysis params yaml
with open("params.yml", "r") as f:
    params = yaml.safe_load(f)

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)
fig_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")

if __name__ == "__main__":
    nme_coverage_regression = pl.read_csv(
        os.path.join(
            fig_dir,
            "table_supplemental_mmr_coverage_change_nme_policy_interventions.csv",
        )
    )

    nme_coverage_regression = nme_coverage_regression.filter(
        (pl.col("State") == "All states") & (pl.col("Year") == "1 year after")
    )
    slope = nme_coverage_regression["Slope (95% CI)"][0]
    # get whatever is before the " ["
    slope_median = str(slope).split(" [")
    slope_lower = float(slope_median[1].split(", ")[0])
    slope_upper = float(slope_median[1].split(", ")[1].replace("]", ""))
    slope_median = float(slope_median[0])
    intercept = nme_coverage_regression["Intercept"][0]

    for slope, name in zip(
        [slope_median, slope_lower, slope_upper], ["median", "lower", "upper"]
    ):
        df = pl.read_csv(
            os.path.join(output_dir, "projection_dataset.csv"),
            schema_overrides={"fips": pl.String},
        )

        # we want to make different "versions" of df based on our scenarios
        # we want to run one scenario where we set NMEs to 0 for every county
        # and evaluate what happens
        # this is equivalent to setting NMEs to 0 in the top 5 highest risk states
        # and top 5 highest NME states and top 250 counties
        mmr = df["estimated_mmr_cov"]
        nme = df["nonmed_exempt_pct"]
        df_no_nmes = df.with_columns(
            nonmed_exempt_pct=pl.lit(0.0),
            estimated_mmr_cov=pl.col("estimated_mmr_cov")
            + slope * pl.col("nonmed_exempt_pct")
            + intercept,
            estimated_mmr_cov_raw=mmr,
            nonmed_exempt_pct_raw=nme,
        ).with_columns(
            change_in_mmr_cov=pl.col("estimated_mmr_cov")
            - pl.col("estimated_mmr_cov_raw")
        )
        # make sure that no values of mmr coverage are greater than 100
        df_no_nmes = df_no_nmes.with_columns(
            estimated_mmr_cov=pl.when(pl.col("estimated_mmr_cov") > 100)
            .then(100)
            .otherwise(pl.col("estimated_mmr_cov"))
        )
        # add label
        df_no_nmes = df_no_nmes.with_columns(remove_nmes=pl.lit(True))

        # we also want to do an analysis where we add NMEs back to the states that
        # have banned them
        states_with_policies = {
            "California": 2016,
            "Maine": 2021,
            "New York": 2019,
            "Connecticut": 2022,
        }

        vac_df = pl.read_csv(
            os.path.join(data_dir, "processed_vaccinations_data.csv"),
            schema_overrides={"nonmed_exempt_pct": pl.Float64},
        )

        for state, year in states_with_policies.items():
            vac_state_year = vac_df.filter(
                (pl.col("state") == state) & (pl.col("year") == year - 1)
            )
            if vac_state_year.height == 0:
                continue

            df = df.join(
                vac_state_year[["state", "county", "nonmed_exempt_pct"]],
                on=["state", "county"],
                how="left",
                suffix=f"_pre_policy_{state}",
            )

            # replace the nme values with these
            # recall that coverage change = slope * nme_pct + intercept
            df = df.with_columns(
                nonmed_exempt_pct=pl.when(
                    pl.col(
                        f"nonmed_exempt_pct_pre_policy_{state}"
                    ).is_not_null()
                    & (pl.col("state") == state)
                )
                .then(pl.col(f"nonmed_exempt_pct_pre_policy_{state}"))
                .otherwise(pl.col("nonmed_exempt_pct")),
            ).with_columns(
                # and reduce the coverage accordingly
                estimated_mmr_cov=pl.when(
                    pl.col(
                        f"nonmed_exempt_pct_pre_policy_{state}"
                    ).is_not_null()
                    & (pl.col("state") == state)
                )
                .then(
                    pl.col("estimated_mmr_cov")
                    - slope * pl.col(f"nonmed_exempt_pct_pre_policy_{state}")
                    - intercept
                )
                .otherwise(pl.col("estimated_mmr_cov")),
            )

        df = df.filter(
            pl.col("state").is_in(list(states_with_policies.keys()))
        )
        # add label
        df = df.with_columns(add_nmes_back=pl.lit(True))

        df = pl.concat([df, df_no_nmes], how="diagonal_relaxed")
        df = df.with_columns(
            add_nmes_back=pl.when(pl.col("add_nmes_back").is_null())
            .then(pl.lit(False))
            .otherwise(pl.col("add_nmes_back"))
        )

        train_df = pl.read_csv(
            os.path.join(data_dir, "analysis_dataset.csv"),
            schema_overrides={"fips": pl.String},
        ).filter(
            pl.col("year").is_in([2019, 2020, 2021, 2022, 2023, 2024, 2025])
        )

        train_df_to_scale = train_df.select(
            [
                "estimated_mmr_cov",
                "nonmed_exempt_pct",
                pl.col("estimated_mmr_cov").alias("estimated_mmr_cov_raw"),
                pl.col("nonmed_exempt_pct").alias("nonmed_exempt_pct_raw"),
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
        )

        scaler = StandardScaler()
        print("Fitting scaler on training data...")
        train_df_scaled = scaler.fit_transform(train_df_to_scale.to_numpy())
        train_df = train_df.with_columns(
            pl.DataFrame(
                train_df_scaled,
                [
                    "estimated_mmr_cov",
                    "nonmed_exempt_pct",
                    "estimated_mmr_cov_raw",
                    "nonmed_exempt_pct_raw",
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
            )
        )

        print("Scaling projection data...")
        scaled_array = scaler.transform(
            df.select(
                [
                    "estimated_mmr_cov",
                    "nonmed_exempt_pct",
                    "estimated_mmr_cov_raw",
                    "nonmed_exempt_pct_raw",
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
            ).to_numpy()
        )

        df = df.with_columns(
            pl.DataFrame(
                scaled_array,
                [
                    "estimated_mmr_cov",
                    "nonmed_exempt_pct",
                    "estimated_mmr_cov_raw",
                    "nonmed_exempt_pct_raw",
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
            )
        )

        # concatenate the two arrays
        train_df = train_df.with_columns(in_train=pl.lit(1))
        df = df.with_columns(
            in_train=pl.lit(0),
            # we just need some value here
            cases_outbreak=pl.lit(0),
        )

        df = df.drop_nans(
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
                "pop_density",
                "log_income",
            ]
        )

        # we want to run a few different model setups
        # we want to run with just the places where we add back NMEs
        # the highest risk states and their borders
        # the highest NME states
        # the highest risk counties

        # label a column based on whether it is a top risk state
        state_df = pl.read_csv(
            os.path.join(output_dir, "state_assembled_projection_results.csv")
        ).filter(pl.col("type") == "projection")
        top_5_risk_states_unconditioned = (
            state_df.sort(by="median_unconditioned_risk_rank", descending=True)
            .head(5)["state"]
            .to_list()
        )

        top_5_risk_states_conditioned = (
            state_df.sort(by="median_conditioned_risk_rank", descending=True)
            .head(5)["state"]
            .to_list()
        )

        df = df.with_columns(
            top_risk_state_conditioned=pl.when(
                pl.col("state").is_in(top_5_risk_states_conditioned)
            )
            .then(True)
            .otherwise(False),
            top_risk_state_unconditioned=pl.when(
                pl.col("state").is_in(top_5_risk_states_unconditioned)
            )
            .then(True)
            .otherwise(False),
        )

        # label a column based on whether the county borders a high risk state
        adjacency_matrix = pl.read_csv(
            os.path.join(data_dir, "county_adjacencies.txt"),
            separator="|",
            schema_overrides={
                "County GEOID": pl.String,
                "Neighbor GEOID": pl.String,
            },
        )

        add_nmes_back_fips = df.filter(pl.col("add_nmes_back") == True)[
            "fips"
        ].to_list()
        neighbors = []
        for fip in add_nmes_back_fips:
            temp = adjacency_matrix.filter(pl.col("County GEOID") == fip)[
                "Neighbor GEOID"
            ].to_list()
            neighbors.extend(temp)
        bordering_fips = list(set(neighbors) - set(add_nmes_back_fips))

        df = df.with_columns(
            background_add_nmes_back_state=pl.col("add_nmes_back") == False
        )
        df = df.with_columns(
            bordering_add_nmes_back_state=(
                pl.col("fips").is_in(bordering_fips)
            )
            & (pl.col("background_add_nmes_back_state") == True)
        )

        high_risk_fips = df.filter(
            pl.col("top_risk_state_unconditioned") == True
        )["fips"].to_list()
        neighbors = []
        for fip in high_risk_fips:
            temp = adjacency_matrix.filter(pl.col("County GEOID") == fip)[
                "Neighbor GEOID"
            ].to_list()
            neighbors.extend(temp)
        bordering_fips = list(set(neighbors) - set(high_risk_fips))

        df = df.with_columns(
            background_top_risk_states=pl.col("add_nmes_back") == False
        )
        df = df.with_columns(
            bordering_high_risk_state=(pl.col("fips").is_in(bordering_fips))
            & (pl.col("background_top_risk_states") == True)
        )

        # label a column based on whether it is a top nme state
        top_5_nme_states = (
            df.filter(pl.col("add_nmes_back") == False)
            .group_by("state")
            .agg(
                nme_pct=(
                    pl.col("nonmed_exempt_pct_raw")
                    * pl.col("pop_5_9")
                    / pl.sum("pop_5_9")
                ).sum()
            )
            .sort("nme_pct", descending=True)
            .head(5)["state"]
            .to_list()
        )
        df = df.with_columns(
            top_nme_state=pl.when(pl.col("state").is_in(top_5_nme_states))
            .then(True)
            .otherwise(False),
        )

        high_nme_fips = df.filter(pl.col("top_nme_state") == True)[
            "fips"
        ].to_list()
        neighbors = []
        for fip in high_nme_fips:
            temp = adjacency_matrix.filter(pl.col("County GEOID") == fip)[
                "Neighbor GEOID"
            ].to_list()
            neighbors.extend(temp)
        bordering_fips = list(set(neighbors) - set(high_nme_fips))

        df = df.with_columns(
            background_top_nme_states=pl.col("add_nmes_back") == False
        )
        df = df.with_columns(
            bordering_high_nme_state=(pl.col("fips").is_in(bordering_fips))
            & (pl.col("background_top_nme_states") == True)
        )

        # add a column for the county's risk level
        risk_df = pl.read_csv(
            os.path.join(output_dir, "assembled_projection_results.csv"),
            schema_overrides={"fips": pl.String},
        )

        risk_df = risk_df[
            [
                "fips",
                "state",
                "county",
                "year",
                "median_conditioned_risk_rank",
                "median_unconditioned_risk_rank",
            ]
        ]

        df = df.join(
            risk_df,
            on=["fips", "state", "county", "year"],
            how="left",
        )

        df = df.with_columns(
            top_ten_pct_counties=pl.when(
                pl.col("median_unconditioned_risk_rank")
                >= pl.max("median_unconditioned_risk_rank") * 0.9
            )
            .then(True)
            .otherwise(False)
        )

        high_risk_county_fips = df.filter(
            pl.col("top_ten_pct_counties") == True
        )["fips"].to_list()
        neighbors = []
        for fip in high_risk_county_fips:
            temp = adjacency_matrix.filter(pl.col("County GEOID") == fip)[
                "Neighbor GEOID"
            ].to_list()
            neighbors.extend(temp)
        bordering_fips = list(set(neighbors) - set(high_risk_county_fips))

        df = df.with_columns(
            background_top_risk_counties=pl.col("add_nmes_back") == False
        )
        df = df.with_columns(
            bordering_high_risk_counties=(pl.col("fips").is_in(bordering_fips))
            & (pl.col("background_top_risk_counties") == True)
        )

        df.write_csv(os.path.join(output_dir, "nme_policy_analysis.csv"))

        add_nmes_back_df = df.filter(pl.col("add_nmes_back") == True)
        not_add_nmes_back_df = df.filter(
            (pl.col("add_nmes_back") == False) & (pl.col("in_train") == 0)
        )
        not_add_nmes_back_df = not_add_nmes_back_df.with_columns(
            # reset the coverages
            estimated_mmr_cov=pl.col("estimated_mmr_cov_raw"),
            nonmed_exempt_pct=pl.col("nonmed_exempt_pct_raw"),
        )
        add_nmes_back_df = pl.concat(
            [add_nmes_back_df, not_add_nmes_back_df], how="diagonal_relaxed"
        )

        top_risk_df = df.filter(pl.col("top_risk_state_unconditioned") == True)
        not_top_risk_df = df.filter(
            (pl.col("top_risk_state_unconditioned") == False)
            & (pl.col("in_train") == 0)
            & (pl.col("add_nmes_back") != True)
        )
        not_top_risk_df = not_top_risk_df.with_columns(
            # reset the coverages
            estimated_mmr_cov=pl.col("estimated_mmr_cov_raw"),
            nonmed_exempt_pct=pl.col("nonmed_exempt_pct_raw"),
        )
        top_risk_df = pl.concat(
            [top_risk_df, not_top_risk_df], how="diagonal_relaxed"
        )

        top_nme_df = df.filter(pl.col("top_nme_state") == True)
        not_top_nme_df = df.filter(
            (pl.col("top_nme_state") == False)
            & (pl.col("in_train") == 0)
            & (pl.col("add_nmes_back") != True)
        )
        not_top_nme_df = not_top_nme_df.with_columns(
            # reset the coverages
            estimated_mmr_cov=pl.col("estimated_mmr_cov_raw"),
            nonmed_exempt_pct=pl.col("nonmed_exempt_pct_raw"),
        )
        top_nme_df = pl.concat(
            [top_nme_df, not_top_nme_df], how="diagonal_relaxed"
        )

        top_county_df = df.filter(pl.col("top_ten_pct_counties") == True)
        not_top_county_df = df.filter(
            (pl.col("top_ten_pct_counties") == False)
            & (pl.col("in_train") == 0)
            & (pl.col("add_nmes_back") != True)
        )
        not_top_county_df = not_top_county_df.with_columns(
            # reset the coverages
            estimated_mmr_cov=pl.col("estimated_mmr_cov_raw"),
            nonmed_exempt_pct=pl.col("nonmed_exempt_pct_raw"),
        )
        top_county_df = pl.concat(
            [top_county_df, not_top_county_df], how="diagonal_relaxed"
        )

        dfs = [
            add_nmes_back_df,
            top_risk_df,
            top_nme_df,
            top_county_df,
        ]

        df = df.with_columns(
            geoid=pl.when(pl.col("fips").is_null())
            .then(pl.col("county"))
            .otherwise(pl.col("county") + pl.col("fips"))
        )

        pl.concat(dfs, how="diagonal_relaxed").write_csv(
            os.path.join(
                output_dir, "nme_policy_analysis_broken_up_by_scenario.csv"
            )
        )

        results = []
        for sim, df in zip(
            [
                "add nmes back",
                "top risk states",
                "top nme states",
                "top risk counties",
            ],
            dfs,
        ):
            print(f"Running NME policy analysis scenario: {sim}...")
            print(df.shape)
            df_temp = pl.concat([train_df, df], how="diagonal_relaxed")

            adjacency_matrix = pl.read_csv(
                os.path.join(data_dir, "county_adjacencies.txt"),
                separator="|",
                schema_overrides={
                    "County GEOID": pl.String,
                    "Neighbor GEOID": pl.String,
                },
            )
            adj_matrix = adjacency.assemble_adjacency_matrix(
                adjacency_matrix,
                df_temp["fips"].unique(maintain_order=True).to_list(),
            )

            model = poisson.PartialPooledPoisson(
                iv=df_temp.select(
                    [
                        "estimated_mmr_cov",
                        "nonmed_exempt_pct",
                        "med_exempt_pct",
                    ]
                ).to_jax(),
                pool=True,
                regression_covariates=df_temp.select(
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
                exposure=df_temp["pop"].to_jax(),
                groups=df_temp["state"],
                group_fixed_effect=False,
                spatial=True,
                spatial_units=df_temp["geoid"],
                adjacency_matrix=adj_matrix,
                spatial_sigma=dist.HalfNormal(1.0),
            )

            # fit model
            nuts_kernel = NUTS(
                model.fit,
                init_strategy=init_to_median(),
                target_accept_prob=0.99,
            )
            mcmc = MCMC(
                nuts_kernel,
                num_warmup=4000,
                num_samples=2000,
                num_chains=4,
            )
            mcmc.run(
                PRNGKey(params["seed"]),
                is_observed=df_temp["in_train"].to_jax(),
                cases=df_temp["cases_outbreak"].to_jax(),
            )

            mcmc.print_summary()

            posterior_samples = mcmc.get_samples()

            # pass the posterior samples to the model to generate predictions
            predictive = Predictive(model.fit, posterior_samples)

            predictive_samples = predictive(
                PRNGKey(params["seed"]),
                is_observed=df_temp["in_train"].to_jax(),
                cases=None,
            )

            # extract just the cases and take only those for the test year
            all_predictions = predictive_samples["cases"][
                :, (df_temp["in_train"] == 0).to_jax()
            ]

            print(all_predictions.shape)
            results.append(all_predictions)

        # pickle it
        with open(
            os.path.join(
                output_dir,
                f"nme_policy_analysis_{name}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(jnp.concat(results, axis=1), f)
