"""
The vaccination rows of a policy scenario once an intervention is applied
(see analysis/10_nme_policy_scenarios.py for the interventions and
analysis/11_run_nme_policy_analyses.py for the projections that use them),
and the coverage change after a state bans nonmedical exemptions that the
ban interventions are based on.
"""

import polars as pl
from addfips import AddFIPS
from scipy import stats

# the model's covariates: the vaccination variables an intervention changes,
# then the demographic ones it does not
FEATURE_COLS = [
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

# the states with a ban on nonmedical exemptions, and the year of the ban: the
# interventions remove these bans (or extend them to more states), and the
# vaccination scenarios never bring exemptions back in these states
STATES_WITH_BANS = {
    "California": 2016,
    "Maine": 2021,
    "New York": 2019,
    "Connecticut": 2022,
}


def build_projection_rows(
    vax_df: pl.DataFrame, intervention_base: pl.DataFrame, intervention: str
) -> pl.DataFrame:
    """
    The projection rows of a vaccination scenario with an intervention
    applied.

    The rows stay in the order of vax_df: the county spatial effects of the
    baseline projection are reused by position, so reordering the rows (moving
    the intervened counties to the top, say) would give every county the
    spatial effect of another one. Only the intervened counties' values
    change:
      - remove_bans: the scenario's projected values plus the delta that
        reverts the county to its pre-ban starting point, so the county
        evolves forward from there
      - all other interventions: the values of the intervention base (the
        county's values after the intervention, from the latest observed
        school year, 2025-26), kept unchanged in every projection year
    """
    applied = intervention_base.filter(pl.col("intervention_applied"))
    is_intervened = pl.col("fips").is_in(applied["fips"].to_list())

    if intervention == "remove_bans":
        deltas = applied.select("fips", "mmr_cov_delta", "nme_delta")
        rows = (
            vax_df.join(deltas, on="fips", how="left", maintain_order="left")
            .with_columns(
                estimated_mmr_cov=pl.when(is_intervened)
                .then(
                    (
                        pl.col("estimated_mmr_cov") + pl.col("mmr_cov_delta")
                    ).clip(0.0, 100.0)
                )
                .otherwise(pl.col("estimated_mmr_cov")),
                nonmed_exempt_pct=pl.when(is_intervened)
                .then(
                    (pl.col("nonmed_exempt_pct") + pl.col("nme_delta")).clip(
                        0.0, 100.0
                    )
                )
                .otherwise(pl.col("nonmed_exempt_pct")),
            )
            .drop("mmr_cov_delta", "nme_delta")
        )
    else:
        frozen = applied.select("fips", *FEATURE_COLS).rename(
            {col: f"{col}_frozen" for col in FEATURE_COLS}
        )
        rows = (
            vax_df.join(frozen, on="fips", how="left", maintain_order="left")
            .with_columns(
                [
                    pl.when(is_intervened)
                    .then(pl.col(f"{col}_frozen"))
                    .otherwise(pl.col(col))
                    .alias(col)
                    for col in FEATURE_COLS
                ]
            )
            .drop([f"{col}_frozen" for col in FEATURE_COLS])
        )
    return rows.with_columns(intervention_applied=is_intervened)


def apply_ban(df, target_mask_col, slope, intercept):
    """
    For counties where target_mask_col is True, set nonmed_exempt_pct to 0
    and increase estimated_mmr_cov by slope * old_nme + intercept (the
    regression of the coverage change on the old exemption).
    The coverage increase is not capped (the coverage is only kept
    within 0-100).
    """
    return df.with_columns(
        [
            pl.when(pl.col(target_mask_col))
            .then(
                (
                    pl.col("estimated_mmr_cov")
                    + (
                        slope * pl.col("nonmed_exempt_pct") + intercept
                    )  # deliberately not capped
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


def state_coverage_changes(
    df: pl.DataFrame, state: str, ban_year: int
) -> pl.DataFrame:
    """
    How the MMR coverage of each county of a state changed after the state
    banned nonmedical exemptions (NMEs).

    df has one row per county-year, with the columns state, year, county, fips,
    nonmed_exempt_pct and estimated_mmr_cov (both percentages). The result has
    one row per county of the year before the ban (in that year's order), with
    its NME rate then (nonmed_exempt_pct, the rate a ban removes) and the
    coverage change, in percentage points, from that year to:
      - the ban year (estimated_mmr_cov_change_policy_year),
      - the year after (estimated_mmr_cov_change_1yr),
      - two years after (estimated_mmr_cov_change_2yr).
    A change is null for a county with no row in the later year.
    """
    for col, dtype in (
        ("fips", pl.String),
        ("nonmed_exempt_pct", pl.Float64),
        ("estimated_mmr_cov", pl.Float64),
    ):
        # fips must stay a string (an integer drops the leading zero), and
        # both percentages must be floats
        if df.schema[col] != dtype:
            raise ValueError(f"{col} must be {dtype}, not {df.schema[col]}")

    def one_year(year: int, columns: list[str]) -> pl.DataFrame:
        return df.filter(
            (pl.col("state") == state) & (pl.col("year") == year)
        ).select(columns)

    changes = one_year(
        ban_year - 1,
        ["state", "year", "county", "fips", "nonmed_exempt_pct"]
        + ["estimated_mmr_cov"],
    )
    for label, year, suffix in (
        ("year-1", ban_year - 1, None),
        ("policy year", ban_year, "_policy_year"),
        ("+1", ban_year + 1, "_1yr"),
        ("+2", ban_year + 2, "_2yr"),
    ):
        table = (
            changes
            if suffix is None
            else one_year(
                year,
                ["fips", "year", "nonmed_exempt_pct", "estimated_mmr_cov"],
            )
        )
        # a county is matched across years by its fips, so a missing or
        # repeated fips would match it to the wrong county (or to two rows)
        if (
            table["fips"].null_count() > 0
            or table["fips"].n_unique() != table.height
        ):
            raise ValueError(
                f"{state}: the {label} rows must have one non-null fips per county"
            )
        if suffix is not None:
            changes = changes.join(
                table,
                on="fips",
                how="left",
                suffix=suffix,
                maintain_order="left",
            ).with_columns(
                (
                    pl.col(f"estimated_mmr_cov{suffix}")
                    - pl.col("estimated_mmr_cov")
                ).alias(f"estimated_mmr_cov_change{suffix}")
            )
    return changes


def coverage_change_fit(
    changes: pl.DataFrame, change_col: str
) -> tuple[float, float, float, float]:
    """
    The Theil-Sen line of the coverage change on the county's NME rate before
    the ban: (slope, intercept, slope lower bound, slope upper bound), with
    the slope's 95% confidence interval. A ban is assumed to raise the
    coverage of a county by slope * (its NME rate) + intercept percentage
    points (apply_ban does this). changes stacks state_coverage_changes of
    several states, and change_col is one of its change columns; counties
    with a null change are left out of the fit.
    """
    fit_rows = changes.select("nonmed_exempt_pct", change_col).drop_nulls()
    return stats.theilslopes(
        fit_rows[change_col].to_numpy(),
        fit_rows["nonmed_exempt_pct"].to_numpy(),
    )


def remove_bans(
    base_df: pl.DataFrame,
    vac_df: pl.DataFrame,
    states_with_bans: dict[str, int],
    slope: float,
    intercept: float,
) -> pl.DataFrame:
    """
    The intervention that removes the bans: each county of a state with a ban
    goes back to the exemption rate it had the year before the ban, and its
    coverage goes down by what the ban is assumed to have added
    (slope * that exemption rate + intercept, as in `apply_ban`).

    `base_df` has the projection rows of the baseline year (one per county)
    and `vac_df` the vaccination data by county and year. Counties of a state
    whose pre-ban year is missing in `vac_df` (the state is then reported) or
    that have no pre-ban rate are left as they are. The result adds
    intervention_applied (the county is in a state with a ban) and the changes
    from the baseline, mmr_cov_delta and nme_delta.
    """
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
                maintain_order="left",
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
                            - slope * pl.col(f"pre_ban_nme_{state}")
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

    return (
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
            maintain_order="left",
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


def top_ranked(
    frame: pl.DataFrame, rank_col: str, column: str, n: int
) -> list:
    """
    The values of `column` of the n rows with the highest `rank_col`. The
    ranks run from the lowest risk to the highest, so the top ones are the
    highest-risk places. Rows with the same rank are taken in the
    order of `frame`.
    """
    return (
        frame.sort(rank_col, descending=True, maintain_order=True)
        .head(n)[column]
        .to_list()
    )


def states_with_exemptions(
    base_df: pl.DataFrame, banned_states: list[str]
) -> list[str]:
    """The states without a ban that have a county with a nonmedical
    exemption rate above 0 (the states where a ban would change something)."""
    return (
        base_df.filter(~pl.col("state").is_in(banned_states))
        .group_by("state")
        .agg(pl.col("nonmed_exempt_pct").max().alias("max_nme"))
        .filter(pl.col("max_nme") > 0)["state"]
        .to_list()
    )


def top_exemption_states(base_df: pl.DataFrame, n: int = 5) -> list[str]:
    """The n states with the highest nonmedical exemption rate, where a
    state's rate is the average over its counties weighted by the fraction of
    their population under 5 (the rate among the state's young children)."""
    return (
        base_df.group_by("state")
        .agg(
            weighted_nme=(
                pl.col("nonmed_exempt_pct")
                * pl.col("fraction_under_5")
                / pl.col("fraction_under_5").sum()
            ).sum()
        )
        .sort(["weighted_nme", "state"], descending=[True, False])
        .head(n)["state"]
        .to_list()
    )


def top_risk_counties(
    county_risk: pl.DataFrame,
    base_df: pl.DataFrame,
    banned_states: list[str],
    share: float = 0.1,
):
    """
    The highest-risk counties that a ban could change. The number to target is
    `share` of ALL counties of `county_risk` (rounded down); they are taken
    from the counties that are not in a state with a ban and have a
    nonmedical exemption rate above 0 (all of those if there are fewer),
    highest `unconditioned_count_rank` first.

    Returns their fips, the number targeted, the number of counties and the
    number of eligible counties.
    """
    n_total = county_risk.height
    n_top = int(n_total * share)

    # Filter to only eligible counties (not in banned states, has NME > 0)
    eligible = county_risk.join(
        base_df.select(["fips", "state", "nonmed_exempt_pct"]),
        on="fips",
        how="inner",
        maintain_order="left",
    ).filter(
        ~pl.col("state").is_in(banned_states)
        & (pl.col("nonmed_exempt_pct") > 0)
    )

    # Take the top N counties from eligible set (or all eligible if fewer than N)
    n_to_select = min(n_top, eligible.height)
    fips = top_ranked(
        eligible, "unconditioned_count_rank", "fips", n_to_select
    )
    return fips, n_top, n_total, eligible.height


def counties_with_case_history(
    cases_df: pl.DataFrame, min_cases: int = 3
) -> list[str]:
    """
    The fips of the counties that had at least `min_cases` cases in a single
    year. `cases_df` has one row per county-year (columns county, state, fips
    and cases); a row with no fips gets the county's from its name.
    """
    # add in any missing fips codes
    addfips = AddFIPS()
    cases_df = cases_df.with_columns(
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
    return (
        cases_df.filter(pl.col("cases") >= min_cases)
        .select("fips")
        .unique()["fips"]
        .to_list()
    )


def bottom_decile_fips(base_df: pl.DataFrame, quantile: float = 0.1):
    """The fips of the counties whose coverage is at or below the `quantile`
    of the coverage of all counties, and that coverage."""
    threshold = base_df["estimated_mmr_cov"].quantile(quantile)
    fips = base_df.filter(pl.col("estimated_mmr_cov") <= threshold)[
        "fips"
    ].to_list()
    return fips, threshold


def raise_coverage(
    base_df: pl.DataFrame, target: pl.Expr, points: float
) -> pl.DataFrame:
    """
    The intervention that raises the coverage of the targeted counties by
    `points` percentage points (kept within 0-100). `target` is a boolean
    expression of the rows of `base_df`; the result marks the targeted rows
    in intervention_applied.
    """
    return base_df.with_columns(
        [
            pl.when(target)
            .then((pl.col("estimated_mmr_cov") + points).clip(0.0, 100.0))
            .otherwise(pl.col("estimated_mmr_cov"))
            .alias("estimated_mmr_cov"),
            target.alias("intervention_applied"),
        ]
    )


def change_summary(
    df_after: pl.DataFrame, base_df: pl.DataFrame, by_state: bool
) -> pl.DataFrame:
    """
    How an intervention changed the counties it was applied to, from the
    baseline `base_df` to `df_after` (which has intervention_applied): the
    number of counties, their average baseline exemption rate and coverage,
    the average change of each and the smallest and largest coverage change.
    One row per state if `by_state` (the state with the largest average
    coverage change first), else one row for all.
    """
    changes = (
        df_after.join(
            base_df.select(
                [
                    "fips",
                    pl.col("estimated_mmr_cov").alias("baseline_cov"),
                    pl.col("nonmed_exempt_pct").alias("baseline_nme"),
                ]
            ),
            on="fips",
            how="left",
            maintain_order="left",
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
    )
    columns = [
        pl.len().alias("n_counties"),
        pl.col("baseline_nme").mean().alias("avg_baseline_nme"),
        pl.col("baseline_cov").mean().alias("avg_baseline_cov"),
        pl.col("nme_change").mean().alias("avg_nme_change"),
        pl.col("cov_change").mean().alias("avg_cov_change"),
        pl.col("cov_change").min().alias("min_cov_change"),
        pl.col("cov_change").max().alias("max_cov_change"),
    ]
    if by_state:
        summary = changes.group_by("state").agg(columns)
    else:
        summary = changes.select(columns)
    return summary.sort("avg_cov_change", descending=True)
