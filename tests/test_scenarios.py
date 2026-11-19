"""Coverage and exemption scenarios to 2030 (outbreak_size.scenarios)."""

import polars as pl
import pytest

from outbreak_size.scenarios import (
    VACCINATION_SCENARIO_LABELS,
    project_coverage_scenarios,
)

# Five counties of one state with their own reported values, 2020-2024
# (E only 2022-2024). Per year: coverage, nonmedical and medical exemption.
SERIES = {
    # fips: (coverage by year, nonmedical by year, medical by year)
    "A": ([90, 89, 88, 87, 86], [2.0, 2.5, 3.0, 3.5, 4.0], [1.0] * 5),
    "B": ([80, 81, 82, 83, 84], [5.0] * 5, [1.0] * 5),
    "C": ([96.0] * 5, [1.0] * 5, [2.0] * 5),
    "D": ([88, 86, 84, 82, 80], [3.0] * 5, [0.0] * 5),
    "E": ([97, 98, 99], [1.0] * 3, [1.0] * 3),
}
# the coverage slopes of those counties; E's years are 2022-2024
ROC = {"A": -1.0, "B": 1.0, "C": 0.0, "D": -2.0, "E": 1.0}
# averages over the five counties: coverage (-1 + 1 + 0 - 2 + 1) / 5, and
# nonmedical (0.5 + 0 + 0 + 0 + 0) / 5
NATIONAL_COVERAGE_SLOPE = -0.2
NATIONAL_NONMED_SLOPE = 0.1


@pytest.fixture(scope="module")
def projections():
    rows = []
    for fips, (cov, nonmed, med) in SERIES.items():
        years = range(2024 - len(cov) + 1, 2025)
        for y, c, n, m in zip(years, cov, nonmed, med):
            rows.append(
                dict(
                    state="S",
                    county=fips.lower(),
                    fips=fips,
                    year=y,
                    estimated_mmr_cov=float(c),
                    nonmed_exempt_pct=float(n),
                    med_exempt_pct=float(m),
                    county_reported=True,
                )
            )
    reported = pl.DataFrame(rows)
    state_series = pl.DataFrame(
        {
            "state": ["S"],
            "year": [2024],
            "mmr_cov_state": [90.0],
            "nonmed_exempt_pct_state": [2.0],
            "med_exempt_pct_state": [1.0],
            "series_source": ["cdc"],
        }
    )
    universe = pl.DataFrame(
        {
            "fips": list(SERIES),
            "county": [f.lower() for f in SERIES],
            "state": ["S"] * 5,
        }
    )
    roc = pl.DataFrame({"fips": list(ROC), "slope": list(ROC.values())})
    return project_coverage_scenarios(
        reported.drop("county_reported"),
        roc,
        reported,
        state_series,
        universe,
        latest_year=2025,
        end_year=2030,
    )


def value(projections, scenario, fips, year, column="projected_coverage"):
    got = projections.filter(
        (pl.col("scenario") == scenario)
        & (pl.col("fips") == fips)
        & (pl.col("year") == year)
    )
    assert got.height == 1
    return got[column][0]


def test_every_scenario_has_a_label_in_the_tables_and_figures(projections):
    assert set(VACCINATION_SCENARIO_LABELS) == set(projections["scenario"])
    # each name is one the reader can tell from the others
    assert len(set(VACCINATION_SCENARIO_LABELS.values())) == len(
        VACCINATION_SCENARIO_LABELS
    )


def test_every_scenario_projects_every_county_for_every_year(projections):
    assert set(projections["scenario"]) == {
        "baseline",
        "targeted_worsening",
        "universal_worsening",
        "targeted_improvement",
        "universal_improvement",
        "no_change",
    }
    # scenarios x counties x 2026-2030 (the data end in school year 2024, the
    # one the outbreak year latest_year = 2025 uses)
    assert projections.height == 6 * 5 * 5
    assert sorted(projections["year"].unique()) == list(range(2026, 2031))
    assert (
        projections.select("scenario", "fips", "year").is_duplicated().sum()
        == 0
    )


def test_baseline_continues_each_countys_own_slope(projections):
    # outbreak year 2027 uses school year 2026-27: two steps from the 2024 value
    assert value(projections, "baseline", "A", 2027) == pytest.approx(84.0)
    assert value(projections, "baseline", "B", 2027) == pytest.approx(86.0)
    # 2030 uses school year 2029: five steps
    assert value(projections, "baseline", "A", 2030) == pytest.approx(81.0)
    nonmed = "projected_nonmed_exempt"
    assert value(projections, "baseline", "A", 2027, nonmed) == pytest.approx(
        5.0
    )


def test_no_change_freezes_everything_at_the_baseline(projections):
    for fips, (cov, nonmed, med) in SERIES.items():
        for year in (2027, 2030):
            frozen = value(projections, "no_change", fips, year)
            assert frozen == pytest.approx(cov[-1])
            assert value(
                projections, "no_change", fips, year, "projected_nonmed_exempt"
            ) == pytest.approx(nonmed[-1])


def test_targeted_worsening_doubles_declines_and_gives_the_rest_the_average(
    projections,
):
    # declining counties: slope x 2
    assert value(
        projections, "targeted_worsening", "A", 2027
    ) == pytest.approx(82.0)
    assert value(
        projections, "targeted_worsening", "D", 2027
    ) == pytest.approx(72.0)
    # the others decline at the national average slope
    expected_b = 84.0 + NATIONAL_COVERAGE_SLOPE * 2
    assert value(
        projections, "targeted_worsening", "B", 2027
    ) == pytest.approx(expected_b)
    # exemptions: declining counties gain |coverage slope| on their own slope,
    # the others follow the national average
    nonmed = "projected_nonmed_exempt"
    assert value(
        projections, "targeted_worsening", "A", 2027, nonmed
    ) == pytest.approx(4.0 + (0.5 + 1.0) * 2)
    assert value(
        projections, "targeted_worsening", "B", 2027, nonmed
    ) == pytest.approx(5.0 + NATIONAL_NONMED_SLOPE * 2)


def test_universal_worsening_makes_declines_one_and_a_half_times_steeper(
    projections,
):
    assert value(
        projections, "universal_worsening", "A", 2027
    ) == pytest.approx(83.0)
    assert value(
        projections, "universal_worsening", "D", 2027
    ) == pytest.approx(74.0)
    # counties that are not declining keep their own slope
    assert value(
        projections, "universal_worsening", "B", 2027
    ) == pytest.approx(86.0)
    nonmed = "projected_nonmed_exempt"
    assert value(
        projections, "universal_worsening", "A", 2027, nonmed
    ) == pytest.approx(4.0 + (0.5 + 0.5) * 2)


def test_universal_improvement_adds_a_point_a_year_and_removes_exemptions(
    projections,
):
    assert value(
        projections, "universal_improvement", "A", 2027
    ) == pytest.approx(88.0)
    nonmed = "projected_nonmed_exempt"
    assert value(
        projections, "universal_improvement", "A", 2027, nonmed
    ) == pytest.approx(2.0)
    assert (
        value(projections, "universal_improvement", "C", 2027, nonmed) == 0.0
    )  # clipped


def test_targeted_improvement_reaches_95_by_2030_for_counties_below_it(
    projections,
):
    nonmed = "projected_nonmed_exempt"
    for fips in ("A", "B", "D"):
        assert value(
            projections, "targeted_improvement", fips, 2030
        ) == pytest.approx(95.0)
    # A: 86 in school year 2024; outbreak 2030 uses school year 2029, five
    # years on, so 1.8 points a year; its nonmedical exemption falls by the
    # same points a year
    years = (2030 - 1) - 2024
    slope = (95.0 - 86.0) / years
    assert slope == pytest.approx(1.8)
    assert value(
        projections, "targeted_improvement", "A", 2027
    ) == pytest.approx(86.0 + slope * 2)
    assert value(
        projections, "targeted_improvement", "A", 2027, nonmed
    ) == pytest.approx(4.0 - slope * 2)


def test_targeted_improvement_leaves_counties_at_or_above_95_where_they_are(
    projections,
):
    # C (96, flat) and E (99, rising by a point a year in its own series) do not
    # move: coverage, nonmedical exemption and, here, the medical one
    nonmed = "projected_nonmed_exempt"
    for fips, coverage, exemption in (("C", 96.0, 1.0), ("E", 99.0, 1.0)):
        for year in (2027, 2030):
            assert value(
                projections, "targeted_improvement", fips, year
            ) == pytest.approx(coverage)
            assert value(
                projections, "targeted_improvement", fips, year, nonmed
            ) == pytest.approx(exemption)


def test_projections_stay_within_0_and_100(projections):
    for column in (
        "projected_coverage",
        "projected_nonmed_exempt",
        "projected_med_exempt",
    ):
        assert projections[column].min() >= 0.0
        assert projections[column].max() <= 100.0
    assert value(projections, "baseline", "E", 2027) == 100.0  # 99 + 2 clipped


def run(last_school_year, latest_year=2026):
    """The projections for series that end in school year `last_school_year`."""
    rows = []
    for fips, (cov, nonmed, med) in SERIES.items():
        years = range(last_school_year - len(cov) + 1, last_school_year + 1)
        for y, c, n, m in zip(years, cov, nonmed, med):
            rows.append(
                dict(
                    state="S",
                    county=fips.lower(),
                    fips=fips,
                    year=y,
                    estimated_mmr_cov=float(c),
                    nonmed_exempt_pct=float(n),
                    med_exempt_pct=float(m),
                    county_reported=True,
                )
            )
    reported = pl.DataFrame(rows)
    state_series = pl.DataFrame(
        {
            "state": ["S"],
            "year": [last_school_year],
            "mmr_cov_state": [90.0],
            "nonmed_exempt_pct_state": [2.0],
            "med_exempt_pct_state": [1.0],
            "series_source": ["cdc"],
        }
    )
    universe = pl.DataFrame(
        {
            "fips": list(SERIES),
            "county": [f.lower() for f in SERIES],
            "state": ["S"] * 5,
        }
    )
    roc = pl.DataFrame({"fips": list(ROC), "slope": list(ROC.values())})
    return project_coverage_scenarios(
        reported.drop("county_reported"),
        roc,
        reported,
        state_series,
        universe,
        latest_year=latest_year,
        end_year=2030,
    )


def test_outbreak_year_y_takes_the_value_of_school_year_y_minus_1():
    # data through school year 2025 (2025-26): outbreak 2026 would be that
    # year itself, so the first projected outbreak year 2027 is one step on
    projections = run(2025)
    assert sorted(projections["year"].unique()) == [2027, 2028, 2029, 2030]
    # A: 86 in 2025, slope -1: 85 at school year 2026 (outbreak 2027) ... 82
    for year, expected in ((2027, 85.0), (2028, 84.0), (2030, 82.0)):
        assert value(projections, "baseline", "A", year) == pytest.approx(
            expected
        )
    # one step on from the same series through 2024 (two steps there)
    assert value(run(2024), "baseline", "A", 2027) == pytest.approx(84.0)


def test_the_scenarios_of_a_county_with_data_through_2025():
    projections = run(2025)
    # universal improvement: a point a year over the 4 school years
    # 2026..2029 that outbreak years 2027..2030 use, from 86 in 2025
    assert value(projections, "universal_improvement", "A", 2030) == (
        pytest.approx(90.0)
    )
    # targeted improvement reaches 95 for outbreak 2030 (school year 2029)
    for fips in ("A", "B", "D"):
        assert value(projections, "targeted_improvement", fips, 2030) == (
            pytest.approx(95.0)
        )
    # no_change stays at the baseline
    assert value(projections, "no_change", "A", 2027) == pytest.approx(86.0)
    assert value(projections, "no_change", "A", 2030) == pytest.approx(86.0)


# ---- states with a ban on nonmedical exemptions: no scenario brings them back
# Three counties in New York (a state with a ban) and the same three in a state
# without one, 2020-2024. Per year: coverage, nonmedical and medical exemption.
BAN_SERIES = {
    # 1: no exemptions (the ban) and falling coverage
    "1": ([95, 94, 93, 92, 91], [0.0] * 5, [1.0] * 5),
    # 2: an exemption rate that was rising (+0.5 a year) and falling coverage
    "2": ([95, 94, 93, 92, 91], [1.0, 1.5, 2.0, 2.5, 3.0], [1.0] * 5),
    # 3: a flat exemption rate of 2 and rising coverage
    "3": ([80, 81, 82, 83, 84], [2.0] * 5, [1.0] * 5),
}
BAN_ROC = {"1": -1.0, "2": -1.0, "3": 1.0}
ALL_SCENARIOS = [
    "baseline",
    "targeted_worsening",
    "universal_worsening",
    "targeted_improvement",
    "universal_improvement",
    "no_change",
]


def run_ban_scenarios(**arguments):
    """The scenarios of the counties of BAN_SERIES in New York (fips "NY1" ...)
    and, with the same series, in Ohio ("OH1" ...)."""
    states = {"NY": "New York", "OH": "Ohio"}
    rows, fips_list = [], []
    for prefix, state in states.items():
        for number, (cov, nonmed, med) in BAN_SERIES.items():
            fips = f"{prefix}{number}"
            fips_list.append(fips)
            for y, c, n, m in zip(range(2020, 2025), cov, nonmed, med):
                rows.append(
                    dict(
                        state=state,
                        county=fips.lower(),
                        fips=fips,
                        year=y,
                        estimated_mmr_cov=float(c),
                        nonmed_exempt_pct=float(n),
                        med_exempt_pct=float(m),
                        county_reported=True,
                    )
                )
    reported = pl.DataFrame(rows)
    state_series = pl.DataFrame(
        {
            "state": list(states.values()),
            "year": [2024, 2024],
            "mmr_cov_state": [90.0, 90.0],
            "nonmed_exempt_pct_state": [2.0, 2.0],
            "med_exempt_pct_state": [1.0, 1.0],
            "series_source": ["cdc", "cdc"],
        }
    )
    universe = pl.DataFrame(
        {
            "fips": fips_list,
            "county": [f.lower() for f in fips_list],
            "state": [states[f[:2]] for f in fips_list],
        }
    )
    roc = pl.DataFrame(
        {
            "fips": fips_list,
            "slope": [BAN_ROC[f[2:]] for f in fips_list],
        }
    )
    return project_coverage_scenarios(
        reported.drop("county_reported"),
        roc,
        reported,
        state_series,
        universe,
        latest_year=2025,
        end_year=2030,
        **arguments,
    )


@pytest.fixture(scope="module")
def ban_projections():
    return run_ban_scenarios()


NME = "projected_nonmed_exempt"


def test_no_scenario_raises_the_exemptions_of_a_county_in_a_state_with_a_ban(
    ban_projections,
):
    for scenario in ALL_SCENARIOS:
        for number, (_, nonmed, _) in BAN_SERIES.items():
            for year in range(2027, 2031):
                got = value(
                    ban_projections, scenario, f"NY{number}", year, NME
                )
                assert got <= nonmed[-1] + 1e-9, (scenario, number, year, got)


def test_the_same_counties_without_a_ban_keep_their_exemptions_rising(
    ban_projections,
):
    # hand-computed: baseline NME 0, a coverage slope of -1, 2027 is two steps
    # from the 2024 value. Universal worsening adds half the slope's size a
    # year (0.5 x 2), targeted worsening the slope's size (1.0 x 2); in New
    # York both stay at 0
    for scenario, expected in (
        ("universal_worsening", 1.0),
        ("targeted_worsening", 2.0),
    ):
        assert value(ban_projections, scenario, "OH1", 2027, NME) == (
            pytest.approx(expected)
        )
        assert value(ban_projections, scenario, "NY1", 2027, NME) == 0.0
    # a rising exemption rate under the baseline: +0.5 a year, 3.0 in 2024
    assert value(ban_projections, "baseline", "OH2", 2027, NME) == (
        pytest.approx(4.0)
    )
    assert value(ban_projections, "baseline", "NY2", 2027, NME) == (
        pytest.approx(3.0)
    )
    assert value(ban_projections, "baseline", "NY2", 2030, NME) == (
        pytest.approx(3.0)
    )


def test_a_ban_state_still_lowers_exemptions_in_the_improvement_scenarios(
    ban_projections,
):
    # a flat rate of 2 falls a point a year under universal improvement
    # (2 steps by 2027, clipped at 0 later), in the state with a ban as in
    # the state without one
    for prefix in ("NY", "OH"):
        assert value(
            ban_projections, "universal_improvement", f"{prefix}3", 2027, NME
        ) == pytest.approx(0.0)
        assert value(
            ban_projections, "universal_improvement", f"{prefix}3", 2030, NME
        ) == pytest.approx(0.0)
    for scenario in ("targeted_improvement", "universal_improvement"):
        for year in (2027, 2030):
            assert value(
                ban_projections, scenario, "NY3", year, NME
            ) == pytest.approx(
                value(ban_projections, scenario, "OH3", year, NME)
            )


def test_the_ban_rule_leaves_coverage_and_medical_exemptions_alone(
    ban_projections,
):
    for scenario in ALL_SCENARIOS:
        for number in BAN_SERIES:
            for year in (2027, 2030):
                for column in ("projected_coverage", "projected_med_exempt"):
                    assert value(
                        ban_projections, scenario, f"NY{number}", year, column
                    ) == pytest.approx(
                        value(
                            ban_projections,
                            scenario,
                            f"OH{number}",
                            year,
                            column,
                        )
                    )


def test_without_states_with_bans_the_scenarios_are_as_they_were():
    off = run_ban_scenarios(states_with_bans=())
    assert value(off, "universal_worsening", "NY1", 2027, NME) == (
        pytest.approx(1.0)
    )
    assert value(off, "baseline", "NY2", 2027, NME) == pytest.approx(4.0)


def test_the_policy_scenarios_script_has_no_list_of_ban_states_of_its_own():
    import ast
    import os

    script = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "analysis",
        "10_nme_policy_scenarios.py",
    )
    with open(script) as f:
        tree = ast.parse(f.read())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "outbreak_size.interventions"
        for alias in node.names
    }
    assert "STATES_WITH_BANS" in imported
    assigned = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    # the one list is interventions.STATES_WITH_BANS
    assert not {n for n in assigned if n.lower() == "states_with_bans"}


def test_the_default_states_are_the_states_with_bans():
    import inspect

    from outbreak_size.interventions import STATES_WITH_BANS
    from outbreak_size.scenarios import project_coverage_scenarios

    default = (
        inspect.signature(project_coverage_scenarios)
        .parameters["states_with_bans"]
        .default
    )
    assert set(default) == set(STATES_WITH_BANS)
    assert "West Virginia" not in default  # it allows nonmedical exemptions


def test_a_listed_state_is_held_whatever_its_name():
    # Ohio listed: its counties get the held rule that New York's get by default
    held = run_ban_scenarios(states_with_bans=("Ohio",))
    assert value(held, "universal_worsening", "OH1", 2027, NME) == (
        pytest.approx(0.0)
    )
    assert value(held, "baseline", "OH2", 2030, NME) == pytest.approx(3.0)
    # and New York, not listed, is as it was
    assert value(held, "baseline", "NY2", 2027, NME) == pytest.approx(4.0)
