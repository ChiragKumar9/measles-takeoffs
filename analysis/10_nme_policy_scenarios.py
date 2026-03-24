import os

import polars as pl
import yaml

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
        """
        return df.with_columns(
            [
                pl.when(pl.col(target_mask_col))
                .then(
                    (
                        pl.col("estimated_mmr_cov")
                        + slope_median * pl.col("nonmed_exempt_pct")
                        + intercept
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

    df_ban_risk_states = base_df.with_columns(
        target=pl.col("state").is_in(top_5_risk_states)
    )
    df_ban_risk_states = apply_ban(df_ban_risk_states, "target").drop("target")

    df_ban_risk_states.write_csv(
        os.path.join(output_dir, "intervention_base_ban_top_risk_states.csv")
    )
    print(
        f"Saved intervention_base_ban_top_risk_states.csv "
        f"({df_ban_risk_states.filter(pl.col('intervention_applied')).height} intervened counties)"
    )

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

    # -------------------------------------------------------------------------
    # Intervention 4: Ban NMEs in top 10% risk counties
    # -------------------------------------------------------------------------
    county_risk = risk_df.filter(pl.col("type") == "projection")
    risk_threshold = county_risk["unconditioned_count_rank"].max() * 0.9  # type: ignore

    top_risk_county_fips = county_risk.filter(
        pl.col("unconditioned_count_rank") >= risk_threshold
    )["fips"].to_list()
    print(f"Number of top 10% risk counties: {len(top_risk_county_fips)}")

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
