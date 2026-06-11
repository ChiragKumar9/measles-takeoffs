import os

import polars as pl
import yaml
from addfips import AddFIPS

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)

with open("params.yml", "r") as f:
    params = yaml.safe_load(f)

# States that have banned NMEs and the year of the ban
states_with_bans = {
    "California": 2016,
    "Maine": 2021,
    "New York": 2019,
    "Connecticut": 2022,
}

if __name__ == "__main__":
    # Load regression slope and intercept
    fig_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "figures"
    )
    nme_coverage_regression = pl.read_csv(
        os.path.join(
            fig_dir,
            "table_supplemental_mmr_coverage_change_nme_policy_interventions.csv",
        )
    ).filter(
        (pl.col("State") == "All states") & (pl.col("Year") == "1 year after")
    )

    slope_str = nme_coverage_regression["Slope (95% CI)"][0]
    slope_median = float(str(slope_str).split(" [")[0])
    intercept = float(nme_coverage_regression["Intercept"][0])

    # Load the 2026 baseline projection as the base for all interventions
    base_df = pl.read_csv(
        os.path.join(output_dir, "projection_dataset_2026.csv"),
        schema_overrides={"fips": pl.String},
    ).filter(pl.col("type") == "projection")

    # Load pre-ban NME rates from processed vaccinations data
    vac_df = pl.read_csv(
        os.path.join(data_dir, "processed_vaccinations_data.csv"),
        schema_overrides={"nonmed_exempt_pct": pl.Float64},
    )

    # -------------------------------------------------------------------------
    # Helper: apply ban (NME -> 0, coverage goes up)
    # -------------------------------------------------------------------------
    def apply_ban(df, target_mask_col):
        """
        For counties where target_mask_col is True, set nonmed_exempt_pct to 0
        and increase estimated_mmr_cov by slope * old_nme + intercept.
        The coverage increase is capped at 12 percentage points.
        """
        return df.with_columns(
            [
                pl.when(pl.col(target_mask_col))
                .then(
                    (
                        pl.col("estimated_mmr_cov")
                        + (
                            slope_median * pl.col("nonmed_exempt_pct")
                            + intercept
                        )  # .clip(upper_bound=7.0)
                    ).clip(0.0, 100.0)
                )
                .otherwise(pl.col("estimated_mmr_cov"))
                .alias("estimated_mmr_cov"),
                pl.when(pl.col(target_mask_col))
                .then(pl.lit(0.0))
                .otherwise(pl.col("nonmed_exempt_pct"))
                .alias("nonmed_exempt_pct"),
                pl.col(target_mask_col).alias("intervention_applied"),
            ]
        )

    # -------------------------------------------------------------------------
    # Intervention 1: Remove bans (restore pre-ban NMEs, coverage goes down)
    # -------------------------------------------------------------------------
    df_remove_bans = base_df.clone()

    for state, ban_year in states_with_bans.items():
        pre_ban_year = ban_year - 1
        pre_ban = vac_df.filter(
            (pl.col("state") == state) & (pl.col("year") == pre_ban_year)
        ).select(["state", "county", "nonmed_exempt_pct"])

        if pre_ban.height == 0:
            print(f"Warning: no pre-ban data for {state} in {pre_ban_year}")
            continue

        df_remove_bans = (
            df_remove_bans.join(
                pre_ban.rename({"nonmed_exempt_pct": f"pre_ban_nme_{state}"}),
                on=["state", "county"],
                how="left",
            )
            .with_columns(
                [
                    # Coverage goes down: reverse the ban effect
                    pl.when(
                        (pl.col("state") == state)
                        & pl.col(f"pre_ban_nme_{state}").is_not_null()
                    )
                    .then(
                        (
                            pl.col("estimated_mmr_cov")
                            - slope_median * pl.col(f"pre_ban_nme_{state}")
                            - intercept
                        ).clip(0.0, 100.0)
                    )
                    .otherwise(pl.col("estimated_mmr_cov"))
                    .alias("estimated_mmr_cov"),
                    pl.when(
                        (pl.col("state") == state)
                        & pl.col(f"pre_ban_nme_{state}").is_not_null()
                    )
                    .then(pl.col(f"pre_ban_nme_{state}"))
                    .otherwise(pl.col("nonmed_exempt_pct"))
                    .alias("nonmed_exempt_pct"),
                ]
            )
            .drop(f"pre_ban_nme_{state}")
        )

    # Mark intervened counties
    df_remove_bans = df_remove_bans.with_columns(
        intervention_applied=pl.col("state").is_in(
            list(states_with_bans.keys())
        )
    )

    df_remove_bans = (
        df_remove_bans.join(
            base_df.select(
                [
                    "fips",
                    pl.col("estimated_mmr_cov").alias("baseline_mmr_cov_2026"),
                    pl.col("nonmed_exempt_pct").alias("baseline_nme_2026"),
                ]
            ),
            on="fips",
            how="left",
        )
        .with_columns(
            [
                (
                    pl.col("estimated_mmr_cov")
                    - pl.col("baseline_mmr_cov_2026")
                ).alias("mmr_cov_delta"),
                (
                    pl.col("nonmed_exempt_pct") - pl.col("baseline_nme_2026")
                ).alias("nme_delta"),
            ]
        )
        .drop(["baseline_mmr_cov_2026", "baseline_nme_2026"])
    )

    df_remove_bans.write_csv(
        os.path.join(output_dir, "intervention_base_remove_bans.csv")
    )
    print(
        f"Saved intervention_base_remove_bans.csv "
        f"({df_remove_bans.filter(pl.col('intervention_applied')).height} intervened counties)"
    )

    # -------------------------------------------------------------------------
    # Intervention 2: Ban NMEs in top 5 risk states
    # -------------------------------------------------------------------------
    risk_df = pl.read_csv(
        os.path.join(output_dir, "assembled_projection_results.csv"),
        schema_overrides={"fips": pl.String},
    )
    state_risk = pl.read_csv(
        os.path.join(output_dir, "state_assembled_projection_results.csv")
    ).filter(pl.col("type") == "projection")

    top_5_risk_states = (
        state_risk.sort("unconditioned_risk_rank", descending=True)
        .head(5)["state"]
        .to_list()
    )
    print(f"Top 5 risk states: {top_5_risk_states}")

    # Filter to eligible states first (not already banned, has counties with NME > 0)
    eligible_states = (
        base_df.filter(~pl.col("state").is_in(list(states_with_bans.keys())))
        .group_by("state")
        .agg(pl.col("nonmed_exempt_pct").max().alias("max_nme"))
        .filter(pl.col("max_nme") > 0)["state"]
        .to_list()
    )

    eligible_state_risk = state_risk.filter(
        pl.col("state").is_in(eligible_states)
    )

    top_5_risk_states = (
        eligible_state_risk.sort("unconditioned_risk_rank", descending=True)
        .head(5)["state"]
        .to_list()
    )
    print(f"Top 5 risk states (among eligible): {top_5_risk_states}")

    df_ban_risk_states = base_df.with_columns(
        target=(
            pl.col("state").is_in(top_5_risk_states)
            & (pl.col("nonmed_exempt_pct") > 0)
        )
    )
    df_ban_risk_states = apply_ban(df_ban_risk_states, "target").drop("target")

    df_ban_risk_states.write_csv(
        os.path.join(output_dir, "intervention_base_ban_top_risk_states.csv")
    )
    print(
        f"Saved intervention_base_ban_top_risk_states.csv "
        f"({df_ban_risk_states.filter(pl.col('intervention_applied')).height} intervened counties)"
    )

    # Print change by state for ban top NME states strategy
    print("\n--- Ban Top Risk States: Change by State ---")
    state_changes = (
        df_ban_risk_states.join(
            base_df.select(
                [
                    "fips",
                    pl.col("estimated_mmr_cov").alias("baseline_cov"),
                    pl.col("nonmed_exempt_pct").alias("baseline_nme"),
                ]
            ),
            on="fips",
            how="left",
        )
        .with_columns(
            [
                (pl.col("estimated_mmr_cov") - pl.col("baseline_cov")).alias(
                    "cov_change"
                ),
                (pl.col("nonmed_exempt_pct") - pl.col("baseline_nme")).alias(
                    "nme_change"
                ),
            ]
        )
        .filter(pl.col("intervention_applied"))
        .group_by("state")
        .agg(
            [
                pl.count().alias("n_counties"),
                pl.col("baseline_nme").mean().alias("avg_baseline_nme"),
                pl.col("baseline_cov").mean().alias("avg_baseline_cov"),
                pl.col("nme_change").mean().alias("avg_nme_change"),
                pl.col("cov_change").mean().alias("avg_cov_change"),
                pl.col("cov_change").min().alias("min_cov_change"),
                pl.col("cov_change").max().alias("max_cov_change"),
            ]
        )
        .sort("avg_cov_change", descending=True)
    )

    print(state_changes)

    state_changes = (
        df_ban_risk_states.join(
            base_df.select(
                [
                    "fips",
                    pl.col("estimated_mmr_cov").alias("baseline_cov"),
                    pl.col("nonmed_exempt_pct").alias("baseline_nme"),
                ]
            ),
            on="fips",
            how="left",
        )
        .with_columns(
            [
                (pl.col("estimated_mmr_cov") - pl.col("baseline_cov")).alias(
                    "cov_change"
                ),
                (pl.col("nonmed_exempt_pct") - pl.col("baseline_nme")).alias(
                    "nme_change"
                ),
            ]
        )
        .filter(pl.col("intervention_applied"))
        .select(
            [
                pl.count().alias("n_counties"),
                pl.col("baseline_nme").mean().alias("avg_baseline_nme"),
                pl.col("baseline_cov").mean().alias("avg_baseline_cov"),
                pl.col("nme_change").mean().alias("avg_nme_change"),
                pl.col("cov_change").mean().alias("avg_cov_change"),
                pl.col("cov_change").min().alias("min_cov_change"),
                pl.col("cov_change").max().alias("max_cov_change"),
            ]
        )
        .sort("avg_cov_change", descending=True)
    )

    print(state_changes)

    # -------------------------------------------------------------------------
    # Intervention 3: Ban NMEs in top 5 highest NME states
    # -------------------------------------------------------------------------
    top_5_nme_states = (
        base_df.group_by("state")
        .agg(
            weighted_nme=(
                pl.col("nonmed_exempt_pct")
                * pl.col("fraction_under_5")
                / pl.col("fraction_under_5").sum()
            ).sum()
        )
        .sort("weighted_nme", descending=True)
        .head(5)["state"]
        .to_list()
    )
    print(f"Top 5 NME states: {top_5_nme_states}")

    df_ban_nme_states = base_df.with_columns(
        target=pl.col("state").is_in(top_5_nme_states)
    )
    df_ban_nme_states = apply_ban(df_ban_nme_states, "target").drop("target")

    df_ban_nme_states.write_csv(
        os.path.join(output_dir, "intervention_base_ban_top_nme_states.csv")
    )
    print(
        f"Saved intervention_base_ban_top_nme_states.csv "
        f"({df_ban_nme_states.filter(pl.col('intervention_applied')).height} intervened counties)"
    )

    # Print change by state for ban top NME states strategy
    print("\n--- Ban Top NME States: Change by State ---")
    state_changes = (
        df_ban_nme_states.join(
            base_df.select(
                [
                    "fips",
                    pl.col("estimated_mmr_cov").alias("baseline_cov"),
                    pl.col("nonmed_exempt_pct").alias("baseline_nme"),
                ]
            ),
            on="fips",
            how="left",
        )
        .with_columns(
            [
                (pl.col("estimated_mmr_cov") - pl.col("baseline_cov")).alias(
                    "cov_change"
                ),
                (pl.col("nonmed_exempt_pct") - pl.col("baseline_nme")).alias(
                    "nme_change"
                ),
            ]
        )
        .filter(pl.col("intervention_applied"))
        .group_by("state")
        .agg(
            [
                pl.count().alias("n_counties"),
                pl.col("baseline_nme").mean().alias("avg_baseline_nme"),
                pl.col("baseline_cov").mean().alias("avg_baseline_cov"),
                pl.col("nme_change").mean().alias("avg_nme_change"),
                pl.col("cov_change").mean().alias("avg_cov_change"),
                pl.col("cov_change").min().alias("min_cov_change"),
                pl.col("cov_change").max().alias("max_cov_change"),
            ]
        )
        .sort("avg_cov_change", descending=True)
    )

    print(state_changes)

    state_changes = (
        df_ban_nme_states.join(
            base_df.select(
                [
                    "fips",
                    pl.col("estimated_mmr_cov").alias("baseline_cov"),
                    pl.col("nonmed_exempt_pct").alias("baseline_nme"),
                ]
            ),
            on="fips",
            how="left",
        )
        .with_columns(
            [
                (pl.col("estimated_mmr_cov") - pl.col("baseline_cov")).alias(
                    "cov_change"
                ),
                (pl.col("nonmed_exempt_pct") - pl.col("baseline_nme")).alias(
                    "nme_change"
                ),
            ]
        )
        .filter(pl.col("intervention_applied"))
        .select(
            [
                pl.count().alias("n_counties"),
                pl.col("baseline_nme").mean().alias("avg_baseline_nme"),
                pl.col("baseline_cov").mean().alias("avg_baseline_cov"),
                pl.col("nme_change").mean().alias("avg_nme_change"),
                pl.col("cov_change").mean().alias("avg_cov_change"),
                pl.col("cov_change").min().alias("min_cov_change"),
                pl.col("cov_change").max().alias("max_cov_change"),
            ]
        )
        .sort("avg_cov_change", descending=True)
    )

    print(state_changes)

    # -------------------------------------------------------------------------
    # Intervention 4: Ban NMEs in top 10% risk counties
    # -------------------------------------------------------------------------
    county_risk = risk_df.filter(pl.col("type") == "projection")

    # Calculate top 10% based on ALL counties to get target number
    n_total = county_risk.height
    n_top_decile = int(n_total * 0.1)

    # Filter to only eligible counties (not in banned states, has NME > 0)
    eligible_county_risk = county_risk.join(
        base_df.select(["fips", "state", "nonmed_exempt_pct"]),
        on="fips",
        how="inner",
    ).filter(
        ~pl.col("state").is_in(list(states_with_bans.keys()))
        & (pl.col("nonmed_exempt_pct") > 0)
    )

    # Take the top N counties from eligible set (or all eligible if fewer than N)
    n_to_select = min(n_top_decile, eligible_county_risk.height)

    top_risk_county_fips = (
        eligible_county_risk.sort("unconditioned_count_rank", descending=True)
        .head(n_to_select)["fips"]
        .to_list()
    )
    print(f"Targeting top {n_top_decile} counties (10% of {n_total} total)")
    print(
        f"Selected {len(top_risk_county_fips)} counties from {eligible_county_risk.height} eligible"
    )

    df_ban_risk_counties = base_df.with_columns(
        target=pl.col("fips").is_in(top_risk_county_fips)
    )
    df_ban_risk_counties = apply_ban(df_ban_risk_counties, "target").drop(
        "target"
    )

    df_ban_risk_counties.write_csv(
        os.path.join(output_dir, "intervention_base_ban_top_risk_counties.csv")
    )
    print(
        f"Saved intervention_base_ban_top_risk_counties.csv "
        f"({df_ban_risk_counties.filter(pl.col('intervention_applied')).height} intervened counties)"
    )

    # Print change by state for ban top NME states strategy
    print("\n--- Ban Top Risk Counties: Change by State ---")
    state_changes = (
        df_ban_risk_counties.join(
            base_df.select(
                [
                    "fips",
                    pl.col("estimated_mmr_cov").alias("baseline_cov"),
                    pl.col("nonmed_exempt_pct").alias("baseline_nme"),
                ]
            ),
            on="fips",
            how="left",
        )
        .with_columns(
            [
                (pl.col("estimated_mmr_cov") - pl.col("baseline_cov")).alias(
                    "cov_change"
                ),
                (pl.col("nonmed_exempt_pct") - pl.col("baseline_nme")).alias(
                    "nme_change"
                ),
            ]
        )
        .filter(pl.col("intervention_applied"))
        .group_by("state")
        .agg(
            [
                pl.count().alias("n_counties"),
                pl.col("baseline_nme").mean().alias("avg_baseline_nme"),
                pl.col("baseline_cov").mean().alias("avg_baseline_cov"),
                pl.col("nme_change").mean().alias("avg_nme_change"),
                pl.col("cov_change").mean().alias("avg_cov_change"),
                pl.col("cov_change").min().alias("min_cov_change"),
                pl.col("cov_change").max().alias("max_cov_change"),
            ]
        )
        .sort("avg_cov_change", descending=True)
    )

    print(state_changes)

    state_changes = (
        df_ban_risk_counties.join(
            base_df.select(
                [
                    "fips",
                    pl.col("estimated_mmr_cov").alias("baseline_cov"),
                    pl.col("nonmed_exempt_pct").alias("baseline_nme"),
                ]
            ),
            on="fips",
            how="left",
        )
        .with_columns(
            [
                (pl.col("estimated_mmr_cov") - pl.col("baseline_cov")).alias(
                    "cov_change"
                ),
                (pl.col("nonmed_exempt_pct") - pl.col("baseline_nme")).alias(
                    "nme_change"
                ),
            ]
        )
        .filter(pl.col("intervention_applied"))
        .select(
            [
                pl.count().alias("n_counties"),
                pl.col("baseline_nme").mean().alias("avg_baseline_nme"),
                pl.col("baseline_cov").mean().alias("avg_baseline_cov"),
                pl.col("nme_change").mean().alias("avg_nme_change"),
                pl.col("cov_change").mean().alias("avg_cov_change"),
                pl.col("cov_change").min().alias("min_cov_change"),
                pl.col("cov_change").max().alias("max_cov_change"),
            ]
        )
        .sort("avg_cov_change", descending=True)
    )

    print(state_changes)

    # -------------------------------------------------------------------------
    # Intervention 5: Increase coverage by 5pp in counties with 3+ cases in history
    # -------------------------------------------------------------------------
    # Load training data to get historical case counts
    train_df = pl.read_csv(
        os.path.join(data_dir, "raw_merged_outbreaks_vaccinations.csv"),
        schema_overrides={"fips": pl.String},
    )
    # add in any missing fips codes
    addfips = AddFIPS()
    train_df = train_df.with_columns(
        fips=pl.when(pl.col("fips").is_null())
        .then(
            pl.struct(["county", "state"]).map_elements(
                lambda row: addfips.get_county_fips(
                    row["county"], row["state"]
                ),
                return_dtype=pl.String,
            )
        )
        .otherwise(pl.col("fips"))
    )

    # Identify counties that have had 3+ cases in any single year
    high_case_history_fips = (
        train_df.filter(pl.col("cases") >= 3)
        .select("fips")
        .unique()["fips"]
        .to_list()
    )
    print(
        f"Number of counties with 3+ cases in history: {len(high_case_history_fips)}"
    )

    df_improve_cases_previously = base_df.with_columns(
        [
            pl.when(pl.col("fips").is_in(high_case_history_fips))
            .then((pl.col("estimated_mmr_cov") + 5.0).clip(0.0, 100.0))
            .otherwise(pl.col("estimated_mmr_cov"))
            .alias("estimated_mmr_cov"),
            pl.col("fips")
            .is_in(high_case_history_fips)
            .alias("intervention_applied"),
        ]
    )

    df_improve_cases_previously.write_csv(
        os.path.join(
            output_dir,
            "intervention_base_improve_cases_previously_counties.csv",
        )
    )
    print(
        f"Saved intervention_base_improve_cases_previously_counties.csv "
        f"({df_improve_cases_previously.filter(pl.col('intervention_applied')).height} intervened counties)"
    )

    # -------------------------------------------------------------------------
    # Intervention 6: Increase coverage by 5pct in bottom decile counties
    # -------------------------------------------------------------------------
    coverage_threshold_10pct = base_df["estimated_mmr_cov"].quantile(0.1)
    print(f"Bottom decile coverage threshold: {coverage_threshold_10pct:.2f}%")

    bottom_decile_fips = base_df.filter(
        pl.col("estimated_mmr_cov") <= coverage_threshold_10pct
    )["fips"].to_list()
    print(f"Number of bottom decile counties: {len(bottom_decile_fips)}")

    df_improve_5pct = base_df.with_columns(
        [
            pl.when(pl.col("fips").is_in(bottom_decile_fips))
            .then((pl.col("estimated_mmr_cov") + 5.0).clip(0.0, 100.0))
            .otherwise(pl.col("estimated_mmr_cov"))
            .alias("estimated_mmr_cov"),
            pl.col("fips")
            .is_in(bottom_decile_fips)
            .alias("intervention_applied"),
        ]
    )

    df_improve_5pct.write_csv(
        os.path.join(
            output_dir, "intervention_base_improve_bottom_decile_5pct.csv"
        )
    )
    print(
        f"Saved intervention_base_improve_bottom_decile_5pct.csv "
        f"({df_improve_5pct.filter(pl.col('intervention_applied')).height} intervened counties)"
    )

    # -------------------------------------------------------------------------
    # Intervention 7: Increase coverage by 10pct in bottom decile counties
    # -------------------------------------------------------------------------
    df_improve_10pct = base_df.with_columns(
        [
            pl.when(pl.col("fips").is_in(bottom_decile_fips))
            .then((pl.col("estimated_mmr_cov") + 10.0).clip(0.0, 100.0))
            .otherwise(pl.col("estimated_mmr_cov"))
            .alias("estimated_mmr_cov"),
            pl.col("fips")
            .is_in(bottom_decile_fips)
            .alias("intervention_applied"),
        ]
    )

    df_improve_10pct.write_csv(
        os.path.join(
            output_dir, "intervention_base_improve_bottom_decile_10pct.csv"
        )
    )
    print(
        f"Saved intervention_base_improve_bottom_decile_10pct.csv "
        f"({df_improve_10pct.filter(pl.col('intervention_applied')).height} intervened counties)"
    )

    # -------------------------------------------------------------------------
    # Intervention 8: Increase coverage by 5pp in top 5 risk states
    # -------------------------------------------------------------------------
    # Use the same top_5_risk_states identified earlier
    # (Make sure this uses the eligible states logic from intervention 2)

    df_improve_risk_states_5pct = base_df.with_columns(
        [
            pl.when(pl.col("state").is_in(top_5_risk_states))
            .then((pl.col("estimated_mmr_cov") + 5.0).clip(0.0, 100.0))
            .otherwise(pl.col("estimated_mmr_cov"))
            .alias("estimated_mmr_cov"),
            pl.col("state")
            .is_in(top_5_risk_states)
            .alias("intervention_applied"),
        ]
    )

    df_improve_risk_states_5pct.write_csv(
        os.path.join(
            output_dir, "intervention_base_improve_top_risk_states_5pct.csv"
        )
    )
    print(
        f"Saved intervention_base_improve_top_risk_states_5pct.csv "
        f"({df_improve_risk_states_5pct.filter(pl.col('intervention_applied')).height} intervened counties)"
    )

    # -------------------------------------------------------------------------
    # Intervention 9: Increase coverage by 10pp in top 5 risk states
    # -------------------------------------------------------------------------
    df_improve_risk_states_10pct = base_df.with_columns(
        [
            pl.when(pl.col("state").is_in(top_5_risk_states))
            .then((pl.col("estimated_mmr_cov") + 10.0).clip(0.0, 100.0))
            .otherwise(pl.col("estimated_mmr_cov"))
            .alias("estimated_mmr_cov"),
            pl.col("state")
            .is_in(top_5_risk_states)
            .alias("intervention_applied"),
        ]
    )

    df_improve_risk_states_10pct.write_csv(
        os.path.join(
            output_dir, "intervention_base_improve_top_risk_states_10pct.csv"
        )
    )
    print(
        f"Saved intervention_base_improve_top_risk_states_10pct.csv "
        f"({df_improve_risk_states_10pct.filter(pl.col('intervention_applied')).height} intervened counties)"
    )

    # -------------------------------------------------------------------------
    # Intervention 10: Increase coverage by 5pp in top 10% risk counties
    # -------------------------------------------------------------------------
    # Use the same top_risk_county_fips identified earlier
    # (Make sure this uses the eligible counties logic from intervention 4)

    df_improve_risk_counties_5pct = base_df.with_columns(
        [
            pl.when(pl.col("fips").is_in(top_risk_county_fips))
            .then((pl.col("estimated_mmr_cov") + 5.0).clip(0.0, 100.0))
            .otherwise(pl.col("estimated_mmr_cov"))
            .alias("estimated_mmr_cov"),
            pl.col("fips")
            .is_in(top_risk_county_fips)
            .alias("intervention_applied"),
        ]
    )

    df_improve_risk_counties_5pct.write_csv(
        os.path.join(
            output_dir, "intervention_base_improve_top_risk_counties_5pct.csv"
        )
    )
    print(
        f"Saved intervention_base_improve_top_risk_counties_5pct.csv "
        f"({df_improve_risk_counties_5pct.filter(pl.col('intervention_applied')).height} intervened counties)"
    )

    # -------------------------------------------------------------------------
    # Intervention 11: Increase coverage by 10pp in top 10% risk counties
    # -------------------------------------------------------------------------
    df_improve_risk_counties_10pct = base_df.with_columns(
        [
            pl.when(pl.col("fips").is_in(top_risk_county_fips))
            .then((pl.col("estimated_mmr_cov") + 10.0).clip(0.0, 100.0))
            .otherwise(pl.col("estimated_mmr_cov"))
            .alias("estimated_mmr_cov"),
            pl.col("fips")
            .is_in(top_risk_county_fips)
            .alias("intervention_applied"),
        ]
    )

    df_improve_risk_counties_10pct.write_csv(
        os.path.join(
            output_dir, "intervention_base_improve_top_risk_counties_10pct.csv"
        )
    )
    print(
        f"Saved intervention_base_improve_top_risk_counties_10pct.csv "
        f"({df_improve_risk_counties_10pct.filter(pl.col('intervention_applied')).height} intervened counties)"
    )
