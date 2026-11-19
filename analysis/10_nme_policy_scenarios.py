import os

import polars as pl
import yaml

from outbreak_size.interventions import (
    STATES_WITH_BANS,
    apply_ban,
    bottom_decile_fips,
    change_summary,
    counties_with_case_history,
    coverage_change_fit,
    raise_coverage,
    remove_bans,
    state_coverage_changes,
    states_with_exemptions,
    top_exemption_states,
    top_ranked,
    top_risk_counties,
)

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)

with open("params.yml", "r") as f:
    params = yaml.safe_load(f)


def save_intervention(df, file_name):
    """Write an intervention's rows and say how many counties it changes."""
    df.write_csv(os.path.join(output_dir, file_name))
    print(
        f"Saved {file_name} "
        f"({df.filter(pl.col('intervention_applied')).height} intervened counties)"
    )


if __name__ == "__main__":
    # Load the baseline-scenario 2027 rows as the base for all interventions
    # (one projection row per county; the 2026 file has train rows too)
    base_df = pl.read_csv(
        os.path.join(output_dir, "projection_dataset_baseline_2027.csv"),
        schema_overrides={"fips": pl.String},
    )

    # Load pre-ban NME rates from processed vaccinations data (the state
    # values the imputation filled in are left out of the ban estimate)
    vac_df = pl.read_csv(
        os.path.join(data_dir, "processed_vaccinations_data.csv"),
        schema_overrides={
            "fips": pl.String,
            "estimated_mmr_cov": pl.Float64,
            "nonmed_exempt_pct": pl.Float64,
        },
    ).filter(~pl.col("_is_imputed"))

    # How much a ban raises a county's coverage: the median line of the
    # coverage change one year after each state's ban on the county's NME rate
    # before the ban (the line of the policy figure's regression table, which
    # is fitted by the same function)
    ban_changes = pl.concat(
        [
            state_coverage_changes(vac_df, state, ban_year)
            for state, ban_year in STATES_WITH_BANS.items()
        ]
    )
    fit = coverage_change_fit(ban_changes, "estimated_mmr_cov_change_1yr")
    slope_median, intercept = float(fit[0]), float(fit[1])
    banned_states = list(STATES_WITH_BANS.keys())

    # -------------------------------------------------------------------------
    # Intervention 1: Remove bans (restore pre-ban NMEs, coverage goes down)
    # -------------------------------------------------------------------------
    df_remove_bans = remove_bans(
        base_df, vac_df, STATES_WITH_BANS, slope_median, intercept
    )
    save_intervention(df_remove_bans, "intervention_base_remove_bans.csv")

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

    top_5_risk_states = top_ranked(
        state_risk, "unconditioned_risk_rank", "state", 5
    )
    print(f"Top 5 risk states: {top_5_risk_states}")

    # Filter to eligible states first (not already banned, has counties with NME > 0)
    eligible_states = states_with_exemptions(base_df, banned_states)
    eligible_state_risk = state_risk.filter(
        pl.col("state").is_in(eligible_states)
    )

    top_5_risk_states = top_ranked(
        eligible_state_risk, "unconditioned_risk_rank", "state", 5
    )
    print(f"Top 5 risk states (among eligible): {top_5_risk_states}")

    df_ban_risk_states = base_df.with_columns(
        target=(
            pl.col("state").is_in(top_5_risk_states)
            & (pl.col("nonmed_exempt_pct") > 0)
        )
    )
    df_ban_risk_states = apply_ban(
        df_ban_risk_states, "target", slope_median, intercept
    ).drop("target")
    save_intervention(
        df_ban_risk_states, "intervention_base_ban_top_risk_states.csv"
    )

    # Print change by state for ban top risk states strategy
    print("\n--- Ban Top Risk States: Change by State ---")
    print(change_summary(df_ban_risk_states, base_df, by_state=True))
    print(change_summary(df_ban_risk_states, base_df, by_state=False))

    # -------------------------------------------------------------------------
    # Intervention 3: Ban NMEs in top 5 highest NME states
    # -------------------------------------------------------------------------
    top_5_nme_states = top_exemption_states(base_df, 5)
    print(f"Top 5 NME states: {top_5_nme_states}")

    # only counties that currently have nonmedical exemptions are intervened
    # (as in the other ban interventions)
    df_ban_nme_states = base_df.with_columns(
        target=pl.col("state").is_in(top_5_nme_states)
        & (pl.col("nonmed_exempt_pct") > 0)
    )
    df_ban_nme_states = apply_ban(
        df_ban_nme_states, "target", slope_median, intercept
    ).drop("target")
    save_intervention(
        df_ban_nme_states, "intervention_base_ban_top_nme_states.csv"
    )

    # Print change by state for ban top NME states strategy
    print("\n--- Ban Top NME States: Change by State ---")
    print(change_summary(df_ban_nme_states, base_df, by_state=True))
    print(change_summary(df_ban_nme_states, base_df, by_state=False))

    # -------------------------------------------------------------------------
    # Intervention 4: Ban NMEs in top 10% risk counties
    # -------------------------------------------------------------------------
    county_risk = risk_df.filter(pl.col("type") == "projection")

    # the top 10% is of ALL counties; the counties are taken from the
    # eligible ones (not in banned states, has NME > 0)
    top_risk_county_fips, n_top_decile, n_total, n_eligible = (
        top_risk_counties(county_risk, base_df, banned_states, share=0.1)
    )
    print(f"Targeting top {n_top_decile} counties (10% of {n_total} total)")
    print(
        f"Selected {len(top_risk_county_fips)} counties from {n_eligible} eligible"
    )

    df_ban_risk_counties = base_df.with_columns(
        target=pl.col("fips").is_in(top_risk_county_fips)
    )
    df_ban_risk_counties = apply_ban(
        df_ban_risk_counties, "target", slope_median, intercept
    ).drop("target")
    save_intervention(
        df_ban_risk_counties, "intervention_base_ban_top_risk_counties.csv"
    )

    # Print change by state for ban top risk counties strategy
    print("\n--- Ban Top Risk Counties: Change by State ---")
    print(change_summary(df_ban_risk_counties, base_df, by_state=True))
    print(change_summary(df_ban_risk_counties, base_df, by_state=False))

    # -------------------------------------------------------------------------
    # Intervention 5: Increase coverage by 5pp in counties with 3+ cases in history
    # -------------------------------------------------------------------------
    # Load training data to get historical case counts (through 2026)
    train_df = pl.read_csv(
        os.path.join(
            data_dir, "raw_merged_outbreaks_vaccinations_with_2026.csv"
        ),
        schema_overrides={"fips": pl.String},
    )

    # Identify counties that have had 3+ cases in any single year
    high_case_history_fips = counties_with_case_history(train_df, min_cases=3)
    print(
        f"Number of counties with 3+ cases in history: {len(high_case_history_fips)}"
    )

    df_improve_cases_previously = raise_coverage(
        base_df, pl.col("fips").is_in(high_case_history_fips), 5.0
    )
    save_intervention(
        df_improve_cases_previously,
        "intervention_base_improve_cases_previously_counties.csv",
    )

    # -------------------------------------------------------------------------
    # Intervention 6: Increase coverage by 5pct in bottom decile counties
    # -------------------------------------------------------------------------
    bottom_decile, coverage_threshold_10pct = bottom_decile_fips(base_df, 0.1)
    print(f"Bottom decile coverage threshold: {coverage_threshold_10pct:.2f}%")
    print(f"Number of bottom decile counties: {len(bottom_decile)}")

    df_improve_5pct = raise_coverage(
        base_df, pl.col("fips").is_in(bottom_decile), 5.0
    )
    save_intervention(
        df_improve_5pct, "intervention_base_improve_bottom_decile_5pct.csv"
    )

    # -------------------------------------------------------------------------
    # Intervention 7: Increase coverage by 10pct in bottom decile counties
    # -------------------------------------------------------------------------
    df_improve_10pct = raise_coverage(
        base_df, pl.col("fips").is_in(bottom_decile), 10.0
    )
    save_intervention(
        df_improve_10pct, "intervention_base_improve_bottom_decile_10pct.csv"
    )

    # -------------------------------------------------------------------------
    # Intervention 8: Increase coverage by 5pp in top 5 risk states
    # -------------------------------------------------------------------------
    # Use the same top_5_risk_states identified earlier (the eligible states
    # of intervention 2)
    df_improve_risk_states_5pct = raise_coverage(
        base_df, pl.col("state").is_in(top_5_risk_states), 5.0
    )
    save_intervention(
        df_improve_risk_states_5pct,
        "intervention_base_improve_top_risk_states_5pct.csv",
    )

    # -------------------------------------------------------------------------
    # Intervention 9: Increase coverage by 10pp in top 5 risk states
    # -------------------------------------------------------------------------
    df_improve_risk_states_10pct = raise_coverage(
        base_df, pl.col("state").is_in(top_5_risk_states), 10.0
    )
    save_intervention(
        df_improve_risk_states_10pct,
        "intervention_base_improve_top_risk_states_10pct.csv",
    )

    # -------------------------------------------------------------------------
    # Intervention 10: Increase coverage by 5pp in top 10% risk counties
    # -------------------------------------------------------------------------
    # Use the same top_risk_county_fips identified earlier (the eligible
    # counties of intervention 4)
    df_improve_risk_counties_5pct = raise_coverage(
        base_df, pl.col("fips").is_in(top_risk_county_fips), 5.0
    )
    save_intervention(
        df_improve_risk_counties_5pct,
        "intervention_base_improve_top_risk_counties_5pct.csv",
    )

    # -------------------------------------------------------------------------
    # Intervention 11: Increase coverage by 10pp in top 10% risk counties
    # -------------------------------------------------------------------------
    df_improve_risk_counties_10pct = raise_coverage(
        base_df, pl.col("fips").is_in(top_risk_county_fips), 10.0
    )
    save_intervention(
        df_improve_risk_counties_10pct,
        "intervention_base_improve_top_risk_counties_10pct.csv",
    )
