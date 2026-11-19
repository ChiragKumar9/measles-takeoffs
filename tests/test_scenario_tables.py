"""Tables of the future projections (outbreak_size.scenario_tables)."""

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from outbreak_size import scenario_tables as st

# --- pieces --------------------------------------------------------------------


def test_cases_are_prediction_times_probability_plus_importations():
    out = st.cases_with_importations(
        jnp.array([[10.0, 20.0]]),
        jnp.array([[0.5, 0.25]]),
        jnp.array([[1.0, 3.0]]),
    )
    assert out.tolist() == [[6.0, 8.0]]  # 10 * 0.5 + 1 and 20 * 0.25 + 3


def test_percent_change_is_zero_where_the_baseline_has_no_cases():
    out = st.percent_change(
        jnp.array([[15.0, 5.0, 3.0]]), jnp.array([[10.0, 10.0, 0.0]])
    )
    assert out.tolist() == [[50.0, -50.0, 0.0]]  # type: ignore


def test_draw_summary_has_mean_median_and_the_95_percent_interval():
    mean, median, lower, upper = st.draw_summary(
        jnp.array([1.0, 2.0, 3.0, 9.0])
    )
    assert float(mean) == pytest.approx(3.75)
    assert float(median) == pytest.approx(2.5)
    assert float(lower) == pytest.approx(1.075, rel=1e-5)
    assert float(upper) == pytest.approx(8.55, rel=1e-5)


def test_draw_summary_along_an_axis_summarizes_each_unit():
    draws = jnp.array([[1.0, 10.0], [3.0, 20.0]])
    mean, median, _, _ = st.draw_summary(draws, axis=0)
    assert mean.tolist() == [2.0, 15.0] and median.tolist() == [2.0, 15.0]


def test_risk_ranks_by_cases_and_by_incidence():
    """Unit 0 has the most cases but the largest population, so it is the
    highest by count and the lowest by incidence."""
    cases = jnp.array([[10.0, 5.0, 2.0]] * 3)
    count_rank, incidence_rank = st.risk_ranks(
        cases, jnp.array([1000.0, 10.0, 1.0])
    )
    assert count_rank.tolist() == pytest.approx([100.0, 50.0, 0.0])
    assert incidence_rank.tolist() == pytest.approx([0.0, 50.0, 100.0])


@pytest.mark.parametrize(
    "scenario",
    [
        "targeted_worsening",
        "universal_worsening",
        "targeted_improvement",
        "universal_improvement",
        "no_change",
        "baseline",
    ],
)
def test_a_scenarios_draws_are_not_bounded_by_the_baseline(scenario):
    # a worsening scenario below the baseline (or an improving one above it)
    # in a county stays as the model drew it
    samples = {
        ("baseline", 2027): jnp.array([[5.0, 4.0, 5.0]]),
        (scenario, 2027): jnp.array([[3.0, 7.0, 9.0]]),
    }
    raw = st._scenario_samples(scenario, 2027, samples, 3)
    assert raw.tolist() == [[3.0, 7.0, 9.0]]


# --- the vaccination scenario tables -------------------------------------------

LABELS = {
    "baseline": "Baseline",
    "targeted_worsening": "Worse",
    "no_change": "None",
}
FIPS = ["01", "02", "03"]


def base_df():
    return pl.DataFrame(
        {
            "fips": FIPS,
            "state": ["A", "A", "B"],
            "county": ["x", "y", "z"],
            "pop": [100, 400, 1000],
        }
    )


def samples_by_scenario(years=(2026, 2027)):
    """Constant draws (3) of the outbreak cases of the three counties."""
    out = {}
    for scenario, cases in (
        ("baseline", [10.0, 20.0, 30.0]),
        ("targeted_worsening", [5.0, 40.0, 30.0]),
        ("no_change", [10.0, 20.0, 30.0]),
    ):
        for year in years:
            out[(scenario, year)] = jnp.array([cases] * 3)
    return out


def scenario_table(**overrides):
    args = dict(
        vaccination_scenarios=["baseline", "targeted_worsening"],
        future_years=[2026, 2027],
        future_base_df=base_df(),
        all_future_samples=samples_by_scenario(),
        importation_samples=jnp.full((3, 3), 0.5),
        importation_count_samples=jnp.full((3, 3), 1.0),
        scenario_labels=LABELS,
    )
    args.update(overrides)
    return st.vaccination_scenario_effects(**args)


def test_scenario_table_has_a_row_per_county_year_and_scenario():
    out = scenario_table()
    assert out.height == 3 * 2 * 2
    assert out.select("vaccination_scenario").unique().height == 2
    assert out.columns[:5] == [
        "state",
        "county",
        "fips",
        "year",
        "vaccination_scenario",
    ]
    assert set(out["vaccination_scenario"]) == {"Baseline", "Worse"}
    # sorted by scenario, year, state and county
    assert out.head(3)["county"].to_list() == ["x", "y", "z"]


def test_scenario_table_ranks_counties_by_cases_and_by_incidence():
    """Baseline cases are 0.5 * [10, 20, 30] + 1 = [6, 11, 16] (the incidence
    [0.06, 0.0275, 0.016]): the counties rise by cases and fall by incidence."""
    out = scenario_table(
        vaccination_scenarios=["baseline"], future_years=[2026]
    )
    assert out["count_risk_percentile"].to_list() == pytest.approx(
        [0.0, 50.0, 100.0]
    )
    assert out["incidence_risk_percentile"].to_list() == pytest.approx(
        [100.0, 50.0, 0.0]
    )


def test_scenario_table_importation_columns_summarize_the_draws():
    probabilities = jnp.array(
        [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1], [0.2, 0.2, 0.2]]
    )
    out = scenario_table(
        vaccination_scenarios=["baseline"],
        future_years=[2026],
        importation_samples=probabilities,
    )
    assert out["importation_probability_mean"].to_list() == pytest.approx(
        [0.2, 0.2, 0.2]
    )
    assert out["importation_probability_median"].to_list() == pytest.approx(
        [0.2] * 3
    )
    assert out["importation_probability_p025"].to_list()[0] == pytest.approx(
        0.105, rel=1e-5
    )
    assert out["importation_probability_p975"].to_list()[0] == pytest.approx(
        0.295, rel=1e-5
    )


def test_a_worsening_scenario_is_bounded_by_the_first_years_baseline():
    """Cases of county x under the worsening scenario (5) are below the
    baseline's of 2026 (10), so they are raised to 10 before the table is
    made: its cases (and rank) are those after the bound, in 2027 as well,
    whatever the baseline of 2027 is (here 100 everywhere)."""
    samples = samples_by_scenario()
    for year in (2027,):
        samples[("baseline", year)] = jnp.full((3, 3), 100.0)
    out = scenario_table(
        vaccination_scenarios=["targeted_worsening"],
        future_years=[2026, 2027],
        all_future_samples=samples,
    )
    # county x: 5 is raised to 10, y stays 40 (above 20), z stays 30
    cases = 0.5 * np.array([10.0, 40.0, 30.0]) + 1  # [6, 21, 16]
    expected_rank = (np.argsort(np.argsort(cases)) / 2 * 100).tolist()
    for year in (2026, 2027):
        ranks = out.filter(pl.col("year") == year)["count_risk_percentile"]
        assert ranks.to_list() == pytest.approx(expected_rank)


def test_scenario_table_refuses_draws_with_another_number_of_counties():
    samples = samples_by_scenario()
    samples[("baseline", 2026)] = jnp.zeros((3, 2))
    with pytest.raises(ValueError, match="expected 3 counties, got 2"):
        scenario_table(all_future_samples=samples)


def state_scenario_table(**overrides):
    args = dict(
        vaccination_scenarios=["baseline"],
        future_years=[2026],
        future_base_df=base_df(),
        all_future_samples=samples_by_scenario(),
        importation_samples=jnp.full((3, 3), 0.5),
        importation_count_samples=jnp.full((3, 3), 1.0),
        scenario_labels=LABELS,
    )
    args.update(overrides)
    return st.state_vaccination_scenario_effects(**args)


def test_state_scenario_table_sums_counties_and_ranks_the_states():
    """State A is counties x and y (cases 6 + 11 = 17, population 500),
    state B is county z (16, population 1000): A is higher by cases (100 against
    0) and by incidence (0.034 against 0.016)."""
    out = state_scenario_table()
    assert out["state"].to_list() == ["A", "B"]
    assert out["count_risk_percentile"].to_list() == pytest.approx(
        [100.0, 0.0]
    )
    assert out["incidence_risk_percentile"].to_list() == pytest.approx(
        [100.0, 0.0]
    )


def test_state_importation_probability_is_the_chance_of_at_least_one_import():
    """State A has counties with probability 0.5 and 0.5 in every draw:
    1 - 0.5 * 0.5 = 0.75, and they are expected to have 0.5 + 0.5 = 1
    counties with an import."""
    out = state_scenario_table()
    a = out.filter(pl.col("state") == "A").row(0, named=True)
    b = out.filter(pl.col("state") == "B").row(0, named=True)
    assert a["importation_probability_mean"] == pytest.approx(0.75)
    assert a["expected_counties_with_import_mean"] == pytest.approx(1.0)
    assert b["importation_probability_mean"] == pytest.approx(0.5)
    assert b["expected_counties_with_import_median"] == pytest.approx(0.5)


# --- the intervention tables ----------------------------------------------------

INTERVENTION_LABELS = {"ban": "Ban", "improve": "Improve"}


def baseline_draws(scenario, year):
    return jnp.array([[10.0, 20.0, 30.0, 40.0]] * 3)


def intervention_pkg(intervention, scenario, year):
    """The intervention projects counties 01, 02 and 04 only (not 03); under
    'ban' they have 5, 10 and 40 cases, under 'improve' as many as the
    baseline."""
    cases = {"ban": [5.0, 10.0, 40.0], "improve": [10.0, 20.0, 40.0]}[
        intervention
    ]
    return {"fips": ["01", "02", "04"], "predictions": jnp.array([cases] * 3)}


BASELINE_FIPS = ["01", "02", "03", "04"]
LOCATIONS = pl.DataFrame(
    {
        "fips": BASELINE_FIPS,
        "state": ["A", "A", "B", "B"],
        "county": ["x", "y", "z", "w"],
        "pop": [100, 400, 1000, 500],
    }
)


def intervention_args(load_intervention=intervention_pkg, years=(2026, 2027)):
    return dict(
        interventions=["ban", "improve"],
        vaccination_scenarios=["baseline"],
        future_years=list(years),
        load_baseline=baseline_draws,
        load_intervention=load_intervention,
        importation_samples=jnp.full((3, 4), 1.0),
        importation_count_samples=jnp.zeros((3, 4)),
        baseline_fips=BASELINE_FIPS,
        scenario_labels={"baseline": "Baseline"},
        intervention_labels=INTERVENTION_LABELS,
    )


def test_county_effects_cover_the_counties_both_project():
    out = st.county_intervention_effects(
        locations=LOCATIONS, **intervention_args()
    )
    assert sorted(set(out["fips"])) == ["01", "02", "04"]  # not county 03
    assert out.height == 3 * 2 * 2  # counties, years and interventions
    assert out["intervention"].unique().sort().to_list() == ["Ban", "Improve"]


def test_county_percent_change_is_against_the_baseline_and_cumulates():
    """Under 'ban' county 01 has 5 cases against the baseline's 10 in each
    year: -50% each year, and -50% cumulatively (10 against 20). County 04 has
    as many cases as the baseline (0%)."""
    out = st.county_intervention_effects(
        locations=LOCATIONS, **intervention_args()
    )
    ban = out.filter(pl.col("intervention") == "Ban")
    x = ban.filter(pl.col("fips") == "01").sort("year")
    assert x["pct_change_mean"].to_list() == pytest.approx([-50.0, -50.0])
    assert x["cumulative_pct_change_mean"].to_list() == pytest.approx(
        [-50.0, -50.0]
    )
    w = ban.filter(pl.col("fips") == "04")
    assert w["pct_change_mean"].to_list() == pytest.approx([0.0, 0.0])
    improve = out.filter(pl.col("intervention") == "Improve")
    assert improve["pct_change_mean"].to_list() == pytest.approx([0.0] * 6)


def test_cumulative_change_weights_the_years_by_their_cases():
    """Baseline cases of county 01 are 10 in 2026 and 30 in 2027; the
    intervention's are 5 and 30: single-year changes -50% and 0%, but over
    both years 35 against 40 is -12.5%."""

    def load_baseline(scenario, year):
        value = 10.0 if year == 2026 else 30.0
        return jnp.array([[value, 20.0, 30.0, 40.0]] * 3)

    def load_intervention(intervention, scenario, year):
        value = 5.0 if year == 2026 else 30.0
        return {
            "fips": ["01", "02", "04"],
            "predictions": jnp.array([[value, 20.0, 40.0]] * 3),
        }

    args = intervention_args(load_intervention)
    args["load_baseline"] = load_baseline
    out = st.county_intervention_effects(locations=LOCATIONS, **args)
    x = out.filter(
        (pl.col("fips") == "01") & (pl.col("intervention") == "Ban")
    )
    x = x.sort("year")
    assert x["pct_change_mean"].to_list() == pytest.approx([-50.0, 0.0])
    assert x["cumulative_pct_change_mean"].to_list() == pytest.approx(
        [-50.0, -12.5]
    )


def test_an_intervention_that_changes_its_counties_between_years_is_refused():
    def load_intervention(intervention, scenario, year):
        pkg = intervention_pkg(intervention, scenario, year)
        if year == 2027:
            pkg["fips"] = ["02", "01", "04"]  # another order
        return pkg

    with pytest.raises(ValueError):
        st.county_intervention_effects(
            locations=LOCATIONS, **intervention_args(load_intervention)
        )


def test_state_effects_sum_the_counties_of_each_state():
    """Under 'ban' state A (counties 01 and 02) has 5 + 10 = 15 cases against
    30 (-50%); state B has only county 04 here, with 40 against 40 (the
    unprojected county 03 is not in either)."""
    out = st.state_intervention_effects(
        locations=LOCATIONS, **intervention_args()
    )
    ban = out.filter(
        (pl.col("intervention") == "Ban") & (pl.col("year") == 2026)
    )
    changes = dict(zip(ban["state"], ban["pct_change_mean"]))
    assert changes["A"] == pytest.approx(-50.0)
    assert changes["B"] == pytest.approx(0.0)
    assert out.height == 2 * 2 * 2  # states, years and interventions


def test_nationwide_summary_has_baseline_and_interventions():
    out = st.nationwide_summary(
        **{k: v for k, v in intervention_args().items()}
    )
    assert out["intervention"].unique().sort().to_list() == [
        "Ban",
        "Improve",
        "baseline",
    ]
    baseline = out.filter(pl.col("intervention") == "baseline").sort("year")
    # all four counties: 10 + 20 + 30 + 40 (each draw, importation probability 1)
    assert baseline["single_year_cases_mean"].to_list() == pytest.approx(
        [100.0, 100.0]
    )
    assert baseline["cumulative_cases_mean"].to_list() == pytest.approx(
        [100.0, 200.0]
    )
    ban = out.filter(pl.col("intervention") == "Ban").sort("year")
    # the counties both project: 5 + 10 + 40 = 55 a year
    assert ban["single_year_cases_mean"].to_list() == pytest.approx(
        [55.0, 55.0]
    )
    assert ban["cumulative_cases_median"].to_list() == pytest.approx(
        [55.0, 110.0]
    )
    assert ban["single_year_cases_p025"].to_list() == pytest.approx(
        [55.0, 55.0]
    )
