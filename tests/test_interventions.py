"""The vaccination rows after an intervention (outbreak_size.interventions)."""

import polars as pl
import pytest

from outbreak_size.interventions import (
    FEATURE_COLS,
    apply_ban,
    bottom_decile_fips,
    build_projection_rows,
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

FIPS = ["01001", "01003", "01005", "01007"]


def scenario_rows():
    """A scenario year: four counties, every feature column the same
    number (the county's index + 1) so that edited values stand out."""
    return pl.DataFrame(
        {"fips": FIPS, "state": ["A", "A", "B", "B"]}
        | {col: [float(i + 1) for i in range(4)] for col in FEATURE_COLS}
    )


def intervention_base(applied):
    """The intervention's starting rows: its 2026 values (every feature column
    set to 50 + the county's index) and the change an intervention that is
    undone later makes (a coverage change of +4 and an exemption change of
    -2)."""
    return pl.DataFrame(
        {
            "fips": FIPS,
            "intervention_applied": [f in applied for f in FIPS],
            "mmr_cov_delta": [4.0] * 4,
            "nme_delta": [-2.0] * 4,
        }
        | {col: [50.0 + i for i in range(4)] for col in FEATURE_COLS}
    )


def test_remove_bans_adds_the_change_to_the_intervened_counties_only():
    rows = build_projection_rows(
        scenario_rows(),
        intervention_base({"01003", "01005"}),
        "remove_bans",
    )
    assert rows["estimated_mmr_cov"].to_list() == [1.0, 2.0 + 4, 3.0 + 4, 4.0]
    assert rows["nonmed_exempt_pct"].to_list() == [1.0, 2.0 - 2, 3.0 - 2, 4.0]
    assert rows["intervention_applied"].to_list() == [False, True, True, False]


def test_remove_bans_keeps_the_other_columns():
    rows = build_projection_rows(
        scenario_rows(), intervention_base({"01003"}), "remove_bans"
    )
    assert rows["med_exempt_pct"].to_list() == [1.0, 2.0, 3.0, 4.0]
    assert rows["log_income"].to_list() == [1.0, 2.0, 3.0, 4.0]


def test_remove_bans_stays_within_0_and_100():
    base = intervention_base({"01001", "01003"}).with_columns(
        mmr_cov_delta=pl.Series([200.0, 0.0, 0.0, 0.0]),
        nme_delta=pl.Series([0.0, -50.0, 0.0, 0.0]),
    )
    rows = build_projection_rows(scenario_rows(), base, "remove_bans")
    assert rows["estimated_mmr_cov"][0] == 100.0
    assert rows["nonmed_exempt_pct"][1] == 0.0


@pytest.mark.parametrize("intervention", ["ban_top_risk_states", "improve_x"])
def test_other_interventions_set_the_intervened_counties_to_their_base_values(
    intervention,
):
    rows = build_projection_rows(
        scenario_rows(), intervention_base({"01005", "01007"}), intervention
    )
    for col in FEATURE_COLS:
        assert rows[col].to_list() == [1.0, 2.0, 52.0, 53.0], col


def test_rows_stay_in_the_order_of_the_scenario_file():
    # the baseline's spatial effects are reused by position
    scenario = scenario_rows().reverse()
    rows = build_projection_rows(
        scenario, intervention_base({"01005"}), "ban_top_risk_states"
    )
    assert rows["fips"].to_list() == FIPS[::-1]
    assert rows["estimated_mmr_cov"].to_list() == [4.0, 52.0, 2.0, 1.0]


def test_no_intervened_county_changes_nothing():
    for name in ("remove_bans", "ban_top_risk_states"):
        rows = build_projection_rows(
            scenario_rows(), intervention_base(set()), name
        )
        assert rows.drop("intervention_applied").equals(scenario_rows())
        assert not rows["intervention_applied"].any()


# --- a nonmedical exemption ban ---------------------------------------------


def ban_rows():
    return pl.DataFrame(
        {
            "fips": ["a", "b", "c"],
            "estimated_mmr_cov": [90.0, 95.0, 99.0],
            "nonmed_exempt_pct": [4.0, 2.0, 5.0],
            "target": [True, False, True],
        }
    )


def test_a_ban_removes_the_exemption_and_raises_coverage():
    out = apply_ban(ban_rows(), "target", slope=0.5, intercept=1.0)
    # coverage + slope * old exemption + intercept, only where targeted
    assert out["estimated_mmr_cov"].to_list() == [
        90 + 0.5 * 4 + 1,
        95.0,
        100.0,
    ]
    assert out["nonmed_exempt_pct"].to_list() == [0.0, 2.0, 0.0]
    assert out["intervention_applied"].to_list() == [True, False, True]


def test_a_ban_keeps_coverage_within_0_and_100():
    out = apply_ban(ban_rows(), "target", slope=10.0, intercept=0.0)
    assert out["estimated_mmr_cov"][2] == 100.0  # 99 + 50 clipped
    out = apply_ban(ban_rows(), "target", slope=-10.0, intercept=0.0)
    assert out["estimated_mmr_cov"][0] == 50.0
    assert out["estimated_mmr_cov"][2] == 49.0


def test_a_ban_with_no_target_changes_nothing():
    rows = ban_rows().with_columns(target=pl.lit(False))
    out = apply_ban(rows, "target", slope=0.5, intercept=1.0)
    assert out.drop("intervention_applied").equals(rows)


# --- coverage change after a ban ---------------------------------------------

BAN_YEAR = 2016
NME_RATES = [1.0, 2.0, 3.0, 4.0, 6.0]


def coverage_change_rows(
    state="A", slope=0.5, intercept=1.0, ban_year=BAN_YEAR
):
    """Five counties observed in the year before a ban and in the two years
    after it. Coverage is 90 before the ban. A county's change after the ban
    is exactly slope * (its NME rate) + intercept in the ban year, twice that
    one year after, and three times that two years after."""
    rows = []
    for i, nme in enumerate(NME_RATES):
        fips = f"0100{i + 1}"
        change = slope * nme + intercept
        for k, year in enumerate(
            [ban_year - 1, ban_year, ban_year + 1, ban_year + 2]
        ):
            rows.append(
                {
                    "state": state,
                    "year": year,
                    "county": f"County {i + 1}",
                    "fips": fips,
                    "nonmed_exempt_pct": nme,
                    "estimated_mmr_cov": 90.0 + k * change,
                }
            )
    return pl.DataFrame(rows)


def test_state_coverage_changes_hand_computed():
    """County 1 (NME 1): change 0.5 * 1 + 1 = 1.5 in the ban year, 3.0 a year
    later and 4.5 two years later; county 5 (NME 6): 4.0, 8.0 and 12.0."""
    out = state_coverage_changes(coverage_change_rows(), "A", BAN_YEAR)
    assert out.height == 5
    assert out["nonmed_exempt_pct"].to_list() == NME_RATES
    for col, county_1, county_5 in (
        ("estimated_mmr_cov_change_policy_year", 1.5, 4.0),
        ("estimated_mmr_cov_change_1yr", 3.0, 8.0),
        ("estimated_mmr_cov_change_2yr", 4.5, 12.0),
    ):
        assert out[col][0] == pytest.approx(county_1)
        assert out[col][4] == pytest.approx(county_5)


def test_state_coverage_changes_uses_only_that_state():
    """Another state's rows (even with the same fips) are not matched in."""
    other = coverage_change_rows(state="B", slope=9.0, intercept=9.0)
    both = pl.concat([coverage_change_rows(), other])
    assert state_coverage_changes(both, "A", BAN_YEAR).equals(
        state_coverage_changes(coverage_change_rows(), "A", BAN_YEAR)
    )


def test_state_coverage_changes_missing_later_county_is_null():
    """A county absent in a later year has a null change there, not a
    change measured against the wrong county."""
    rows = coverage_change_rows().filter(
        ~((pl.col("fips") == "01002") & (pl.col("year") == BAN_YEAR + 1))
    )
    out = state_coverage_changes(rows, "A", BAN_YEAR)
    assert out.height == 5
    assert out["estimated_mmr_cov_change_1yr"].null_count() == 1
    assert out["estimated_mmr_cov_change_1yr"][1] is None
    assert out["estimated_mmr_cov_change_policy_year"].null_count() == 0


def test_state_coverage_changes_keeps_year_before_ban_order():
    rows = coverage_change_rows()
    flipped = pl.concat(
        [rows.filter(pl.col("year") != BAN_YEAR - 1)]
        + [rows.filter(pl.col("year") == BAN_YEAR - 1).reverse()]
    )
    out = state_coverage_changes(flipped, "A", BAN_YEAR)
    assert out["fips"].to_list() == [f"0100{i}" for i in (5, 4, 3, 2, 1)]


@pytest.mark.parametrize("year", [BAN_YEAR - 1, BAN_YEAR, BAN_YEAR + 2])
def test_state_coverage_changes_rejects_repeated_fips(year):
    rows = coverage_change_rows()
    repeated = pl.concat([rows, rows.filter(pl.col("year") == year).head(1)])
    with pytest.raises(ValueError, match="one non-null fips"):
        state_coverage_changes(repeated, "A", BAN_YEAR)


def test_state_coverage_changes_rejects_null_fips():
    rows = coverage_change_rows().with_columns(
        fips=pl.when(
            (pl.col("year") == BAN_YEAR) & (pl.col("fips") == "01001")
        )
        .then(None)
        .otherwise(pl.col("fips"))
    )
    with pytest.raises(ValueError, match="one non-null fips"):
        state_coverage_changes(rows, "A", BAN_YEAR)


@pytest.mark.parametrize(
    "col, dtype",
    [
        ("fips", pl.Int64),
        ("nonmed_exempt_pct", pl.Float32),
        ("estimated_mmr_cov", pl.Int64),
    ],
)
def test_state_coverage_changes_rejects_wrong_dtype(col, dtype):
    """An integer fips would lose its leading zero, and the percentages must
    be floats, so a reader that gets these wrong is rejected."""
    rows = coverage_change_rows()
    if col == "fips":
        rows = rows.with_columns(pl.col("fips").str.to_integer())
    else:
        rows = rows.with_columns(pl.col(col).cast(dtype))
    with pytest.raises(ValueError, match=col):
        state_coverage_changes(rows, "A", BAN_YEAR)


def test_coverage_change_fit_recovers_exact_line():
    """When every county follows change = 0.5 * NME + 1 exactly, the
    Theil-Sen line is that line, and the 95% interval contains the slope."""
    changes = state_coverage_changes(coverage_change_rows(), "A", BAN_YEAR)
    slope, intercept, lower, upper = coverage_change_fit(
        changes, "estimated_mmr_cov_change_policy_year"
    )
    assert slope == pytest.approx(0.5)
    assert intercept == pytest.approx(1.0)
    assert lower <= slope <= upper
    # a year later every change is twice as large
    slope, intercept, *_ = coverage_change_fit(
        changes, "estimated_mmr_cov_change_1yr"
    )
    assert (slope, intercept) == pytest.approx((1.0, 2.0))


def test_coverage_change_fit_leaves_out_null_changes():
    """A county with no later row does not enter the fit (an outlier whose
    change is null would otherwise move the line)."""
    rows = coverage_change_rows()
    odd = rows.filter(pl.col("fips") == "01005").with_columns(
        estimated_mmr_cov=pl.when(pl.col("year") == BAN_YEAR + 1)
        .then(None)
        .otherwise(pl.col("estimated_mmr_cov"))
    )
    changes = state_coverage_changes(
        pl.concat([rows.filter(pl.col("fips") != "01005"), odd]),
        "A",
        BAN_YEAR,
    )
    slope, intercept, *_ = coverage_change_fit(
        changes, "estimated_mmr_cov_change_1yr"
    )
    assert (slope, intercept) == pytest.approx((1.0, 2.0))


def test_coverage_change_fit_ignores_the_order_of_the_states():
    """The pooled line is the same whichever state is listed first, so
    the scripts that list the four ban states in different orders agree."""
    states = {"A": 2016, "B": 2019, "C": 2021}
    rows = {
        "A": coverage_change_rows("A", 0.5, 1.0, 2016),
        "B": coverage_change_rows("B", 0.8, 0.2, 2019),
        "C": coverage_change_rows("C", 0.3, 2.0, 2021),
    }
    # the counties of B and C get other NME rates, and C drops one county
    # in the later years, so that the pooled data are not a perfect line
    rows["B"] = rows["B"].with_columns(
        nonmed_exempt_pct=pl.col("nonmed_exempt_pct") + 0.5
    )
    rows["C"] = rows["C"].filter(
        ~((pl.col("fips") == "01003") & (pl.col("year") > 2021))
    )
    fits = []
    for order in (["A", "B", "C"], ["C", "A", "B"], ["B", "C", "A"]):
        pooled = pl.concat(
            [state_coverage_changes(rows[s], s, states[s]) for s in order]
        )
        fits.append(
            tuple(coverage_change_fit(pooled, "estimated_mmr_cov_change_1yr"))
        )
    assert fits[0] == fits[1] == fits[2]


# --- the interventions of the policy scenarios --------------------------------


def policy_base(rows):
    """rows: (fips, state, county, coverage, nonmedical exemption rate)"""
    return pl.DataFrame(
        rows,
        schema={
            "fips": pl.String,
            "state": pl.String,
            "county": pl.String,
            "estimated_mmr_cov": pl.Float64,
            "nonmed_exempt_pct": pl.Float64,
        },
        orient="row",
    )


def policy_vaccinations(rows):
    """rows: (state, county, year, nonmedical exemption rate)"""
    return pl.DataFrame(
        rows,
        schema={
            "state": pl.String,
            "county": pl.String,
            "year": pl.Int64,
            "nonmed_exempt_pct": pl.Float64,
        },
        orient="row",
    )


def removed(base, vaccinations, bans={"CA": 2016, "ME": 2021}):
    return remove_bans(base, vaccinations, bans, slope=0.5, intercept=1.0)


def test_remove_bans_intervention_restores_the_pre_ban_exemption(capsys):
    """California banned in 2016 with 4% and 2% exemption in counties a and b
    the year before: the ban is assumed to have added 0.5 * 4 + 1 = 3 and
    0.5 * 2 + 1 = 2 points of coverage, which are taken off again."""
    base = policy_base(
        [
            ("06001", "CA", "a", 90.0, 0.0),
            ("06003", "CA", "b", 90.0, 0.0),
            ("48001", "TX", "c", 80.0, 3.0),
        ]
    )
    vaccinations = policy_vaccinations(
        [
            ("CA", "a", 2015, 4.0),
            ("CA", "b", 2015, 2.0),
            ("CA", "a", 2016, 0.0),
        ]
    )
    out = removed(base, vaccinations)
    assert out["estimated_mmr_cov"].to_list() == [87.0, 88.0, 80.0]
    assert out["nonmed_exempt_pct"].to_list() == [4.0, 2.0, 3.0]
    assert out["mmr_cov_delta"].to_list() == [-3.0, -2.0, 0.0]
    assert out["nme_delta"].to_list() == [4.0, 2.0, 0.0]
    applied = dict(zip(out["county"], out["intervention_applied"]))
    assert applied["a"] and applied["b"] and not applied["c"]
    # Maine banned in 2021 and the data have no 2020 rows for it
    assert "no pre-ban data for ME in 2020" in capsys.readouterr().out


def test_remove_bans_intervention_covers_every_state_with_a_ban():
    """California (banned 2016) and New York (2019) each go back to their own
    pre-ban year: 0.5 * 4 + 1 = 3 points off in California and 0.5 * 6 + 1 =
    4 in New York."""
    base = policy_base(
        [
            ("06001", "CA", "a", 90.0, 0.0),
            ("36001", "NY", "n", 90.0, 0.0),
            ("48001", "TX", "c", 80.0, 3.0),
        ]
    )
    vaccinations = policy_vaccinations(
        [("CA", "a", 2015, 4.0), ("NY", "n", 2018, 6.0)]
    )
    out = removed(base, vaccinations, {"CA": 2016, "NY": 2019})
    assert out["estimated_mmr_cov"].to_list() == [87.0, 86.0, 80.0]
    assert out["nonmed_exempt_pct"].to_list() == [4.0, 6.0, 3.0]
    assert out["intervention_applied"].to_list() == [True, True, False]


def test_remove_bans_intervention_leaves_counties_without_a_pre_ban_rate():
    base = policy_base(
        [("06001", "CA", "a", 90.0, 0.0), ("06099", "CA", "z", 91.0, 0.0)]
    )
    vaccinations = policy_vaccinations([("CA", "a", 2015, 4.0)])
    out = removed(base, vaccinations, {"CA": 2016})
    assert out["estimated_mmr_cov"].to_list() == [87.0, 91.0]
    assert out["nonmed_exempt_pct"].to_list() == [4.0, 0.0]


def test_remove_bans_intervention_stays_within_0_and_100():
    base = policy_base([("06001", "CA", "a", 2.0, 0.0)])
    vaccinations = policy_vaccinations([("CA", "a", 2015, 50.0)])
    out = removed(base, vaccinations, {"CA": 2016})
    assert out["estimated_mmr_cov"].to_list() == [0.0]


def test_remove_bans_intervention_keeps_the_order_of_the_counties():
    base = policy_base(
        [
            ("06003", "CA", "b", 90.0, 0.0),
            ("48001", "TX", "c", 80.0, 3.0),
            ("06001", "CA", "a", 90.0, 0.0),
        ]
    )
    vaccinations = policy_vaccinations(
        [("CA", "a", 2015, 4.0), ("CA", "b", 2015, 2.0)]
    )
    out = removed(base, vaccinations, {"CA": 2016})
    assert out["county"].to_list() == ["b", "c", "a"]


def test_top_ranked_takes_the_highest_ranks():
    frame = pl.DataFrame(
        {"state": ["a", "b", "c", "d"], "rank": [3.0, 1.0, 4.0, 2.0]}
    )
    assert top_ranked(frame, "rank", "state", 2) == ["c", "a"]
    assert top_ranked(frame, "rank", "state", 10) == ["c", "a", "d", "b"]


def test_states_with_exemptions_leaves_out_banned_and_exemption_free_states():
    base = policy_base(
        [
            ("1", "CA", "a", 90.0, 5.0),  # banned
            ("2", "TX", "b", 90.0, 0.0),  # no exemptions anywhere
            ("3", "TX", "c", 90.0, 0.0),
            ("4", "OH", "d", 90.0, 0.0),
            ("5", "OH", "e", 90.0, 2.0),  # one county with exemptions
        ]
    )
    assert states_with_exemptions(base, ["CA"]) == ["OH"]


def test_top_exemption_states_weights_counties_by_children_under_5():
    """A state's rate is its counties' rates weighted by their fraction of
    population under 5. A has (10 * 0.1 + 0 * 0.1) / 0.2 = 5; B has
    (8 * 0.1 + 0 * 0.3) / 0.4 = 2; C has 3. An unweighted average would put B
    (4) above C (3), so the order tells the two apart."""
    base = pl.DataFrame(
        {
            "state": ["A", "A", "B", "B", "C"],
            "nonmed_exempt_pct": [10.0, 0.0, 8.0, 0.0, 3.0],
            "fraction_under_5": [0.1, 0.1, 0.1, 0.3, 0.5],
        }
    )
    assert top_exemption_states(base, 3) == ["A", "C", "B"]
    assert top_exemption_states(base, 1) == ["A"]


def risk_problem():
    """Twenty counties, ranks 1 to 20 (20 the highest risk). The two highest
    are in a banned state, the third has no exemptions."""
    fips = [f"{i:02d}" for i in range(1, 21)]
    risk = pl.DataFrame(
        {
            "fips": fips,
            "unconditioned_count_rank": [float(i) for i in range(1, 21)],
        }
    )
    base = pl.DataFrame(
        {
            "fips": fips,
            "state": ["CA" if i >= 19 else "TX" for i in range(1, 21)],
            "nonmed_exempt_pct": [
                0.0 if i == 18 else 1.0 for i in range(1, 21)
            ],
        }
    )
    return risk, base


def test_top_risk_counties_skip_banned_and_exemption_free_counties():
    risk, base = risk_problem()
    fips, n_top, n_total, n_eligible = top_risk_counties(risk, base, ["CA"])
    assert (n_top, n_total) == (2, 20)  # 10% of all 20 counties
    assert n_eligible == 17  # not the 2 banned counties or the one with 0%
    assert fips == ["17", "16"]


def test_top_risk_counties_takes_all_eligible_when_there_are_fewer():
    risk, base = risk_problem()
    base = base.with_columns(
        nonmed_exempt_pct=pl.when(pl.col("fips") == "01")
        .then(1.0)
        .otherwise(0.0)
    )
    fips, n_top, _, n_eligible = top_risk_counties(risk, base, ["CA"])
    assert (n_top, n_eligible) == (2, 1)
    assert fips == ["01"]


def test_counties_with_case_history_have_3_cases_in_one_year():
    """The threshold counts a single year, and is inclusive; a county with no
    fips gets it from its name (Autauga, Alabama is 01001)."""
    cases = pl.DataFrame(
        {
            "county": ["x", "x", "y", "Autauga", "z"],
            "state": ["A", "A", "A", "Alabama", "A"],
            "fips": ["01001x", "01001x", "01003y", None, "01005z"],
            "cases": [3, 3, 2, 4, 1],
        }
    )
    assert sorted(counties_with_case_history(cases)) == ["01001", "01001x"]
    assert sorted(counties_with_case_history(cases, min_cases=2)) == [
        "01001",
        "01001x",
        "01003y",
    ]


def test_bottom_decile_is_the_counties_at_or_below_the_10th_percentile():
    base = policy_base(
        [(f"{i}", "A", f"c{i}", 80.0 + i, 1.0) for i in range(10)]
    )
    fips, threshold = bottom_decile_fips(base)
    assert threshold == 81.0
    assert fips == ["0", "1"]


def test_raise_coverage_adds_points_within_0_and_100_to_the_targets():
    base = policy_base(
        [
            ("1", "A", "a", 90.0, 1.0),
            ("2", "A", "b", 98.0, 1.0),
            ("3", "A", "c", 50.0, 1.0),
        ]
    )
    out = raise_coverage(base, pl.col("fips").is_in(["1", "2"]), 5.0)
    assert out["estimated_mmr_cov"].to_list() == [95.0, 100.0, 50.0]
    assert out["intervention_applied"].to_list() == [True, True, False]
    assert out["nonmed_exempt_pct"].to_list() == [1.0, 1.0, 1.0]


def summary_problem():
    """States X (counties a and b) and Y (c) are intervened, county d (state
    Z) is left alone."""
    base = policy_base(
        [
            ("1", "X", "a", 80.0, 5.0),
            ("2", "X", "b", 90.0, 3.0),
            ("3", "Y", "c", 70.0, 0.0),
            ("4", "Z", "d", 60.0, 2.0),
        ]
    )
    after = base.with_columns(
        estimated_mmr_cov=pl.Series([85.0, 93.0, 71.0, 60.0]),
        nonmed_exempt_pct=pl.Series([0.0, 0.0, 0.0, 2.0]),
        intervention_applied=pl.Series([True, True, True, False]),
    )
    return base, after


def test_change_summary_by_state_describes_the_intervened_counties():
    base, after = summary_problem()
    out = change_summary(after, base, by_state=True)
    # the state with the largest average coverage change first; Z was not
    # intervened
    assert out["state"].to_list() == ["X", "Y"]
    x = out.row(0, named=True)
    assert x["n_counties"] == 2
    assert x["avg_baseline_nme"] == 4.0
    assert x["avg_baseline_cov"] == 85.0
    assert x["avg_nme_change"] == -4.0
    assert x["avg_cov_change"] == 4.0  # (5 + 3) / 2
    assert (x["min_cov_change"], x["max_cov_change"]) == (3.0, 5.0)
    assert out.row(1, named=True)["avg_cov_change"] == 1.0


def test_change_summary_overall_is_one_row():
    base, after = summary_problem()
    out = change_summary(after, base, by_state=False)
    assert out.height == 1 and "state" not in out.columns
    row = out.row(0, named=True)
    assert row["n_counties"] == 3  # not county d
    assert row["avg_cov_change"] == 3.0  # (5 + 3 + 1) / 3
    assert (row["min_cov_change"], row["max_cov_change"]) == (1.0, 5.0)
