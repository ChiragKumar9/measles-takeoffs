"""
Vaccination scenarios: how coverage and the nonmedical and medical exemption
rates of each county change from the baseline year to 2030 under each scenario
(see analysis/figure_measles_burden_overview.py, which writes the projected
vaccination data that the risk projections use).
"""

import numpy as np
import polars as pl
from scipy.stats import linregress

from outbreak_size.baseline import (
    as_of_values,
    county_source_rows,
    trend_to_year,
)
from outbreak_size.data_year import county_data_year_of_outbreak
from outbreak_size.interventions import STATES_WITH_BANS

# the name each scenario has in the tables and figures of the projections
# (the policy and validation figure scripts and their supplemental tables)
VACCINATION_SCENARIO_LABELS = {
    "no_change": "No changes in vaccination",
    "baseline": "Vaccination trends continue",
    "targeted_worsening": "Vaccinations decline 2x faster",
    "universal_worsening": "Vaccinations decline 1.5x faster",
    "targeted_improvement": "Universal 95% coverage by 2030",
    "universal_improvement": "1pp improvement year-over-year",
}


def project_coverage_scenarios(
    vaccination_data,
    vaccination_data_roc,
    reported_rows,
    state_series,
    universe,
    latest_year=2026,
    end_year=2030,
    states_with_bans=tuple(STATES_WITH_BANS),
):
    """
    Project vaccine coverage, nonmedical exemption, and medical exemption rates
    under 5 scenarios.

    Parameters:
    -----------
    vaccination_data : pl.DataFrame
        The vaccination rows before the state imputation, with year, fips,
        estimated_mmr_cov, nonmed_exempt_pct, med_exempt_pct, etc.
    vaccination_data_roc : pl.DataFrame
        DataFrame with fips and slope (from linear regression on coverage)
    reported_rows : pl.DataFrame
        Rows as reported (before interpolation and state fill), with
        county_reported
    state_series : pl.DataFrame
        State values (vaccination_wrangling.state_level_series)
    universe : pl.DataFrame
        fips, county and state of every county to project
    latest_year : int
        The outbreak year before the first projected one (default 2026): the
        baseline is the data of the school years before latest_year + 1, and
        the projections are for outbreak years latest_year + 1 to end_year
    end_year : int
        Final outbreak year to project to (default 2030)
    states_with_bans : tuple of str
        The states with a ban on nonmedical exemptions (interventions.
        STATES_WITH_BANS). In every scenario the nonmedical exemption rate of
        a county in these states never rises above its baseline value: a
        positive slope of the rate is set to 0 (a falling rate stays as it is)

    An outbreak year Y uses the vaccination data of school year Y - 1 (a
    school year is named by the year it starts in), so the value of outbreak
    year Y is the baseline plus the slope for the school years from the
    baseline's to Y - 1.

    Returns:
    --------
    pl.DataFrame with projections in long format
    """

    # 1. Baseline of every county, each field as of its own latest data up to
    # latest_year (see as_of_values), with the slope of the series it is from
    as_of = as_of_values(
        reported_rows,
        vaccination_data,
        state_series,
        universe,
        latest_year + 1,
        states_with_bans=states_with_bans,
    )
    # every value is first trended to the last known school year, the one the
    # outbreak year latest_year uses (a value observed earlier moves along its
    # slope, never backwards), so that the baseline year of every county is
    # the same and "no change" holds the trended value, not a stale one
    as_of = trend_to_year(as_of, latest_year - 1)
    baseline = as_of.select(
        "fips",
        "county",
        "state",
        pl.col("estimated_mmr_cov").alias("baseline_coverage"),
        pl.col("reference_year").alias("baseline_year"),
        pl.col("estimated_mmr_cov_slope").alias("slope_baseline"),
        pl.col("nonmed_exempt_pct").alias("baseline_nonmed_exempt"),
        pl.col("nonmed_exempt_pct_slope").alias("slope_nonmed_baseline"),
        pl.col("med_exempt_pct").alias("baseline_med_exempt"),
        pl.col("med_exempt_pct_slope").alias("slope_med_baseline"),
    )
    for field, source in (
        ("estimated_mmr_cov", "coverage"),
        ("nonmed_exempt_pct", "nonmed"),
        ("med_exempt_pct", "med"),
    ):
        by_source = (
            as_of.group_by(f"{field}_source").len().sort(f"{field}_source")
        )
        print(
            f"Baseline {source} from: "
            f"{dict(zip(by_source[f'{field}_source'], by_source['len']))}"
        )

    # 2. Slopes of the counties' own reported values, for the national averages
    county_rows = county_source_rows(reported_rows)

    # 3. Calculate slopes for nonmed and med exemptions via linear regression
    def calc_slopes(df, outcome_col, alias):
        return (
            df.filter(pl.col("year") >= 2020)
            .filter(pl.col(outcome_col).is_not_null())
            .group_by("fips", maintain_order=True)
            .map_groups(
                lambda g: pl.DataFrame(
                    {
                        "fips": [g["fips"][0]],
                        alias: [
                            linregress(
                                g["year"].cast(pl.Float64).to_numpy(),
                                g[outcome_col].cast(pl.Float64).to_numpy(),
                            ).slope  # type: ignore
                            if g.height >= 2
                            else float("nan")
                        ],
                    }
                )
            )
            .filter(~pl.col(alias).is_nan())
        )

    nonmed_roc = calc_slopes(county_rows, "nonmed_exempt_pct", "slope_nonmed")
    med_roc = calc_slopes(county_rows, "med_exempt_pct", "slope_med")

    # 4. Calculate national average slopes
    valid_slopes = vaccination_data_roc.filter(
        ~pl.col("slope").is_nan() & ~pl.col("slope").is_null()
    )
    national_avg_slope = valid_slopes["slope"].mean()

    valid_nonmed = nonmed_roc.filter(
        ~pl.col("slope_nonmed").is_nan() & ~pl.col("slope_nonmed").is_null()
    )
    national_avg_slope_nonmed = valid_nonmed["slope_nonmed"].mean()

    valid_med = med_roc.filter(
        ~pl.col("slope_med").is_nan() & ~pl.col("slope_med").is_null()
    )
    national_avg_slope_med = valid_med["slope_med"].mean()

    print(f"National average slope (coverage):     {national_avg_slope:.4f}")
    print(
        f"National average slope (nonmed exempt): {national_avg_slope_nonmed:.4f}"
    )
    print(
        f"National average slope (med exempt):    {national_avg_slope_med:.4f}"
    )
    print(
        f"Number of counties with coverage slope data: {valid_slopes.height}"
    )
    print(
        f"Number of counties with nonmed slope data:   {valid_nonmed.height}"
    )
    print(f"Number of counties with med slope data:      {valid_med.height}")

    # 5. The slopes are already on the baseline
    projection_base = baseline

    # 6. Describe the bottom 30% of the declining counties by coverage slope
    declining = projection_base.filter(pl.col("slope_baseline") < 0)
    n_bottom_30 = int(np.ceil(declining.height * 0.3))

    print(f"\nNumber of declining counties: {declining.height}")
    print(f"Number in bottom 30%: {n_bottom_30}")
    print(
        f"Worst slope in bottom 30%: {declining.sort('slope_baseline').head(1)['slope_baseline'][0]:.4f}"
    )
    print(
        f"Best slope in bottom 30%: {declining.sort('slope_baseline').head(n_bottom_30).tail(1)['slope_baseline'][0]:.4f}"
    )

    # 7. Define scenario modifications
    def apply_scenario_a(df):
        """(a) Baseline - current trends"""
        return df.with_columns(
            [
                pl.col("slope_baseline").alias("slope_scenario"),
                pl.col("slope_nonmed_baseline").alias("slope_nonmed_scenario"),
                pl.col("slope_med_baseline").alias("slope_med_scenario"),
            ]
        )

    def apply_scenario_b(df):
        """(b) Targeted worsening"""
        return df.with_columns(
            [
                pl.when(pl.col("slope_baseline") < 0)
                .then(pl.col("slope_baseline") * 2.0)
                .otherwise(national_avg_slope)
                .alias("slope_scenario"),
                pl.when(pl.col("slope_baseline") < 0)
                .then(
                    pl.col("slope_nonmed_baseline")
                    + pl.col("slope_baseline").abs()
                )
                .otherwise(national_avg_slope_nonmed)
                .alias("slope_nonmed_scenario"),
                pl.col("slope_med_baseline").alias("slope_med_scenario"),
            ]
        )

    def apply_scenario_c(df):
        """(c) Universal worsening"""
        return df.with_columns(
            [
                pl.when(pl.col("slope_baseline") < 0)
                .then(pl.col("slope_baseline") * 1.5)
                .otherwise(pl.col("slope_baseline"))
                .alias("slope_scenario"),
                pl.when(pl.col("slope_baseline") < 0)
                .then(
                    pl.col("slope_nonmed_baseline")
                    + (pl.col("slope_baseline") * 0.5).abs()
                )
                .otherwise(pl.col("slope_nonmed_baseline"))
                .alias("slope_nonmed_scenario"),
                pl.col("slope_med_baseline").alias("slope_med_scenario"),
            ]
        )

    def apply_scenario_d(df):
        """(d) Targeted improvement: a county below 95% coverage rises in a
        straight line to 95% in the school year before end_year (the one
        outbreak year end_year uses), and its nonmedical exemption falls
        by as many points a year; a county at or above 95% stays where it is
        (coverage and nonmedical exemption). Medical exemption follows its
        own slope."""
        required_coverage_slope = (95.0 - pl.col("baseline_coverage")) / (
            county_data_year_of_outbreak(end_year) - pl.col("baseline_year")
        )
        below_95 = pl.col("baseline_coverage") < 95.0
        return df.with_columns(
            [
                pl.when(below_95)
                .then(required_coverage_slope)
                .otherwise(pl.lit(0.0))
                .alias("slope_scenario"),
                pl.when(below_95)
                .then(-required_coverage_slope)
                .otherwise(pl.lit(0.0))
                .alias("slope_nonmed_scenario"),
                pl.col("slope_med_baseline").alias("slope_med_scenario"),
            ]
        )

    def apply_scenario_e(df):
        """(e) Universal improvement"""
        return df.with_columns(
            [
                pl.lit(1.0).alias("slope_scenario"),
                pl.lit(-1.0).alias("slope_nonmed_scenario"),
                pl.col("slope_med_baseline").alias("slope_med_scenario"),
            ]
        )

    def apply_scenario_f(df):
        """(f) No change - coverage frozen at baseline"""
        return df.with_columns(
            [
                pl.lit(0.0).alias("slope_scenario"),
                pl.lit(0.0).alias("slope_nonmed_scenario"),
                pl.lit(0.0).alias("slope_med_scenario"),
            ]
        )

    scenarios = [
        ("baseline", apply_scenario_a),
        ("targeted_worsening", apply_scenario_b),
        ("universal_worsening", apply_scenario_c),
        ("targeted_improvement", apply_scenario_d),
        ("universal_improvement", apply_scenario_e),
        ("no_change", apply_scenario_f),  # Add this line
    ]

    # 8. Create projections for each scenario
    all_scenarios = []

    for scenario_name, modifier_func in scenarios:
        print(f"\nProcessing scenario: {scenario_name}")

        scenario_df = modifier_func(projection_base.clone())
        # a state with a ban does not get its exemptions back, in any
        # scenario: the rate of its counties never rises above the baseline
        # value (a falling rate stays as it is)
        scenario_df = scenario_df.with_columns(
            pl.when(
                pl.col("state").is_in(list(states_with_bans))
                & (pl.col("slope_nonmed_scenario") > 0)
            )
            .then(pl.lit(0.0))
            .otherwise(pl.col("slope_nonmed_scenario"))
            .alias("slope_nonmed_scenario")
        )

        projection_years = list(range(latest_year + 1, end_year + 1))

        for proj_year in projection_years:
            # school years from the baseline's to the one that outbreak year
            # proj_year uses (proj_year - 1)
            years_ahead = county_data_year_of_outbreak(
                proj_year
            ) - scenario_df["baseline_year"].cast(pl.Int32)

            scenario_df = scenario_df.with_columns(
                [
                    (
                        pl.col("baseline_coverage").cast(pl.Float64)
                        + pl.col("slope_scenario").cast(pl.Float64)
                        * years_ahead
                    )
                    .clip(0.0, 100.0)
                    .alias(f"cov_{proj_year}"),
                    (
                        pl.col("baseline_nonmed_exempt").cast(pl.Float64)
                        + pl.col("slope_nonmed_scenario").cast(pl.Float64)
                        * years_ahead
                    )
                    .clip(0.0, 100.0)
                    .alias(f"nonmed_{proj_year}"),
                    (
                        pl.col("baseline_med_exempt").cast(pl.Float64)
                        + pl.col("slope_med_scenario").cast(pl.Float64)
                        * years_ahead
                    )
                    .clip(0.0, 100.0)
                    .alias(f"med_{proj_year}"),
                ]
            )

        # Reshape coverage to long
        cov_long = (
            scenario_df.select(
                [
                    "fips",
                    "county",
                    "state",
                    "baseline_year",
                    "baseline_coverage",
                    "slope_baseline",
                    "slope_scenario",
                    *[f"cov_{y}" for y in projection_years],
                ]
            )
            .unpivot(
                index=[
                    "fips",
                    "county",
                    "state",
                    "baseline_year",
                    "baseline_coverage",
                    "slope_baseline",
                    "slope_scenario",
                ],
                on=[f"cov_{y}" for y in projection_years],
                variable_name="year_col",
                value_name="projected_coverage",
            )
            .with_columns(
                pl.col("year_col")
                .str.replace("cov_", "")
                .cast(pl.Int32)
                .alias("year")
            )
            .drop("year_col")
        )

        # Reshape nonmed exemptions to long
        nonmed_long = (
            scenario_df.select(
                [
                    "fips",
                    "baseline_nonmed_exempt",
                    "slope_nonmed_baseline",
                    "slope_nonmed_scenario",
                    *[f"nonmed_{y}" for y in projection_years],
                ]
            )
            .unpivot(
                index=[
                    "fips",
                    "baseline_nonmed_exempt",
                    "slope_nonmed_baseline",
                    "slope_nonmed_scenario",
                ],
                on=[f"nonmed_{y}" for y in projection_years],
                variable_name="year_col",
                value_name="projected_nonmed_exempt",
            )
            .with_columns(
                pl.col("year_col")
                .str.replace("nonmed_", "")
                .cast(pl.Int32)
                .alias("year")
            )
            .drop("year_col")
        )

        # Reshape med exemptions to long
        med_long = (
            scenario_df.select(
                [
                    "fips",
                    "baseline_med_exempt",
                    "slope_med_baseline",
                    "slope_med_scenario",
                    *[f"med_{y}" for y in projection_years],
                ]
            )
            .unpivot(
                index=[
                    "fips",
                    "baseline_med_exempt",
                    "slope_med_baseline",
                    "slope_med_scenario",
                ],
                on=[f"med_{y}" for y in projection_years],
                variable_name="year_col",
                value_name="projected_med_exempt",
            )
            .with_columns(
                pl.col("year_col")
                .str.replace("med_", "")
                .cast(pl.Int32)
                .alias("year")
            )
            .drop("year_col")
        )

        # Join all three into one long frame
        scenario_long = (
            cov_long.join(
                nonmed_long,
                on=["fips", "year"],
                how="left",
                maintain_order="left",
            )
            .join(
                med_long,
                on=["fips", "year"],
                how="left",
                maintain_order="left",
            )
            .with_columns(pl.lit(scenario_name).alias("scenario"))
        )

        all_scenarios.append(scenario_long)

        # Print summary statistics
        for col, label in [
            ("projected_coverage", "coverage"),
            ("projected_nonmed_exempt", "nonmed exempt"),
            ("projected_med_exempt", "med exempt"),
        ]:
            avg = scenario_long.filter(pl.col("year") == end_year)[col].mean()
            mn = scenario_long.filter(pl.col("year") == end_year)[col].min()
            print(f"  Avg {label} in {end_year}: {avg:.2f}%   Min: {mn:.2f}%")

    combined = pl.concat(all_scenarios)
    return combined
