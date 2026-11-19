"""The importation formulas (outbreak_size.importation)."""

import numpy as np
import polars as pl
import pytest

from outbreak_size.importation import (
    STATES_WITHOUT_2025_IMPORTS,
    calculate_yearly_import_probability,
    distribute_state_level_risk_to_county,
    draw_imports,
    future_importation_draws,
    pre_year_for_validation_year,
    probability_at_least_one_import,
    scale_importation_probability,
    total_case_draws,
)


def test_total_cases_are_imports_plus_outbreak_only_when_there_is_an_import():
    outbreak = np.array([[5.0, 7.0], [3.0, 4.0]])
    imports = np.array([[0, 2], [1, 0]])
    # draw (0, 0): no import, so no cases; draw (0, 1): 2 imports + 7;
    # draw (1, 0): 1 import + 3; draw (1, 1): no import, so no cases
    expected = np.array([[0.0, 9.0], [4.0, 0.0]])
    np.testing.assert_array_equal(
        total_case_draws(outbreak, imports), expected
    )


def test_draws_of_imports_have_the_expected_mean_and_chance_of_none():
    rate = 0.5
    imports = draw_imports(
        np.full((200_000, 1), rate), np.random.default_rng(1)
    )
    assert imports.dtype == np.int32
    assert imports.mean() == pytest.approx(rate, rel=0.02)
    # the probability of no import is exp(-rate)
    assert (imports == 0).mean() == pytest.approx(np.exp(-rate), abs=0.005)


def test_total_cases_have_mean_rate_plus_chance_of_an_import_times_outbreak():
    # by hand: expected imports 0.5 and an outbreak of 10 cases give
    # 0.5 + (1 - exp(-0.5)) * 10 = 4.4347
    rate, outbreak_size = 0.5, 10.0
    imports = draw_imports(
        np.full((400_000, 1), rate), np.random.default_rng(2)
    )
    total = total_case_draws(np.full((400_000, 1), outbreak_size), imports)
    by_hand = rate + (1 - np.exp(-rate)) * outbreak_size
    assert total.mean() == pytest.approx(by_hand, rel=0.01)
    # no cases in the draws without an import
    assert (total == 0).mean() == pytest.approx(np.exp(-rate), abs=0.005)


@pytest.mark.parametrize("rate", [0.001, 0.05, 0.7, 3.0])
@pytest.mark.parametrize("scaler", [0.5, 1.0, 2.79, 10.0])
def test_scaling_the_probability_scales_the_rate(rate, scaler):
    probability = 1 - np.exp(-rate)
    scaled = scale_importation_probability(probability, scaler)
    assert scaled == pytest.approx(1 - np.exp(-scaler * rate))


def test_scaling_by_one_changes_nothing_and_by_zero_gives_zero():
    probability = np.array([0.0, 0.2, 0.9, 1.0])
    assert scale_importation_probability(probability, 1.0) == pytest.approx(
        probability
    )
    assert (scale_importation_probability(probability, 0.0) == 0).all()


def test_scaled_probability_stays_in_0_1_and_grows_with_the_scaler():
    probability = np.linspace(0, 1, 11)
    low = scale_importation_probability(probability, 1.5)
    high = scale_importation_probability(probability, 3.0)
    assert ((0 <= low) & (low <= 1)).all()
    assert (high >= low).all()
    # the old way, probability * scaler capped at 1, overstates it
    assert (np.minimum(probability * 3.0, 1.0) >= high - 1e-12).all()


@pytest.mark.parametrize(
    "state_alpha, beta, share",
    [
        (3.0, 12.0, 0.02),
        (21.0, 12.0, 0.0088),
        (1.0, 17.0, 0.4),
        (6.0, 12.0, 1.0),
    ],
)
def test_county_probability_matches_a_simulation_of_the_construction(
    state_alpha, beta, share
):
    # 08: the county's rate is share * (the state's Gamma(alpha, beta) rate)
    rng = np.random.default_rng(0)
    state_rate = rng.gamma(state_alpha, 1 / beta, size=2_000_000)
    simulated = (1 - np.exp(-share * state_rate)).mean()
    formula = probability_at_least_one_import(state_alpha * share, beta, share)
    assert formula == pytest.approx(simulated, abs=2e-4)


def test_county_probability_is_higher_for_a_bigger_share():
    shares = np.array([0.001, 0.01, 0.1, 0.5])
    probability = probability_at_least_one_import(4.0 * shares, 12.0, shares)
    assert (np.diff(probability) > 0).all()


# --- which posterior the rows of a validation year may use -------------------


@pytest.mark.parametrize(
    "validation_year, expected",
    [
        (2021, 2021),  # what was known before the year being predicted
        (2025, 2025),
        (2020, None),  # not validated: every year of data
        (2026, None),  # predicts 2026: every year of data
    ],
)
def test_year_of_the_posterior_the_rows_of_a_year_may_use(
    validation_year, expected
):
    assert pre_year_for_validation_year(validation_year) == expected


def test_a_validation_year_never_uses_its_own_or_later_imports():
    # imports "before i" stop before year i, so i may be the year predicted
    for validation_year in range(2021, 2026):
        i = pre_year_for_validation_year(validation_year)
        assert i is not None and i <= validation_year


# --- the states' yearly import-rate posteriors ------------------------------


def state_table():
    return pl.DataFrame(
        {
            "state": ["California", "South Carolina", "Wyoming"],
            "imported": [30, 1, 0],
            **{
                f"imported_pre_{i}": [3 * (i - 2020), 0, 0]
                for i in (2021, 2022, 2023, 2024, 2025)
            },
        }
    )


def test_posterior_alpha_adds_the_imports_to_the_prior():
    out = calculate_yearly_import_probability(state_table(), 16)
    assert out["posterior_alpha"].to_list() == [31, 2, 1]
    assert out["posterior_alpha_pre_2023"].to_list() == [10, 1, 1]


def test_posterior_beta_counts_the_years_before_each_year():
    # 16 years of data (2010-2025) plus the prior; before i: i - 2010 years
    out = calculate_yearly_import_probability(state_table(), 16)
    for i in (2021, 2022, 2023, 2024, 2025):
        assert out[f"posterior_beta_pre_{i}"].to_list() == [i - 2010 + 1] * 3


def test_south_carolina_has_one_year_fewer_in_the_all_years_posterior():
    # its 2025 imports are not in the data: a missing year, not a zero year
    assert STATES_WITHOUT_2025_IMPORTS == ["South Carolina"]
    out = calculate_yearly_import_probability(state_table(), 16)
    betas = dict(zip(out["state"], out["posterior_beta"]))
    assert betas == {"California": 17, "South Carolina": 16, "Wyoming": 17}


def test_a_state_with_no_imports_is_not_observed():
    out = calculate_yearly_import_probability(state_table(), 16)
    assert out["observed"].to_list() == [True, True, False]


# --- from a state's rate to its counties ------------------------------------


def county_rows(
    pops=(3e6, 1e6, 5e5), train_pop=3.1e6, pops_2025=None, c2_data_year=2024
):
    """The 2026 projection rows of three counties (two in California, one in
    Wyoming), the rows of validation year 2025 (the two California counties
    with vaccination data from 2024, or from `c2_data_year` for c2) and a
    train row of county c1 in 2025."""

    def row(fips, state, year, validation_year, kind, pop):
        return dict(
            fips=fips,
            state=state,
            county=fips,
            year=year,
            validation_year=validation_year,
            type=kind,
            pop=pop,
        )

    pops_2025 = pops if pops_2025 is None else pops_2025
    rows = []
    for fips, state, pop in zip(
        ["c1", "c2", "w1"], ["California", "California", "Wyoming"], pops
    ):
        rows.append(row(fips, state, 2024, 2026, "projection", pop))
    rows.append(
        row("c1", "California", 2024, 2025, "projection", pops_2025[0])
    )
    rows.append(
        row("c2", "California", c2_data_year, 2025, "projection", pops_2025[1])
    )
    rows.append(row("c1", "California", 2025, 2025, "train", train_pop))
    return pl.DataFrame(rows)


KNOWN = pl.DataFrame(
    {
        "year": [2025],
        "state": ["California"],
        "county": ["c1"],
        "imported": [1],
    }
)


@pytest.fixture(scope="module")
def distributed():
    states = calculate_yearly_import_probability(state_table(), 16)
    out, _ = distribute_state_level_risk_to_county(
        states, county_rows(), KNOWN, seed=1
    )
    return out


def rate_draws(out, fips, validation_year, kind):
    got = out.filter(
        (pl.col("fips") == fips)
        & (pl.col("validation_year") == validation_year)
        & (pl.col("type") == kind)
    )
    assert got.height == 1
    return np.array(got["expected_importations_samples"][0])


def test_a_projection_row_of_2026_uses_every_year_of_data(distributed):
    # California: alpha 31, beta 17 -> mean rate 31 / 17, times c1's share 3/4
    draws = rate_draws(distributed, "c1", 2026, "projection")
    assert draws.mean() == pytest.approx(31 / 17 * 0.75, abs=0.02)


def test_a_validation_row_uses_only_the_years_before_its_prediction_year(
    distributed,
):
    # data from 2024 predicts 2025: imports before 2025 (alpha 16, beta 16)
    draws = rate_draws(distributed, "c1", 2025, "projection")
    assert draws.mean() == pytest.approx(16 / 16 * 0.75, abs=0.02)


def test_a_train_row_has_the_same_state_draws_as_its_projection_twin(
    distributed,
):
    train = rate_draws(distributed, "c1", 2025, "train")
    twin = rate_draws(distributed, "c1", 2025, "projection")
    # the same draw of the state's rate; only the population share differs,
    # and it is the county's population over the state's population
    assert train / twin == pytest.approx(np.full(8000, 3.1 / 3.0))


def test_counties_of_a_state_share_each_draw_of_its_rate(distributed):
    c1 = rate_draws(distributed, "c1", 2026, "projection")
    c2 = rate_draws(distributed, "c2", 2026, "projection")
    assert c1 / c2 == pytest.approx(np.full(8000, 3.0))


def test_states_have_independent_draws(distributed):
    california = rate_draws(distributed, "c1", 2026, "projection")
    wyoming = rate_draws(distributed, "w1", 2026, "projection")
    assert abs(np.corrcoef(california, wyoming)[0, 1]) < 0.05


def test_the_probability_of_an_import_is_one_minus_exp_of_minus_the_rate(
    distributed,
):
    row = distributed.filter(
        (pl.col("fips") == "c2") & (pl.col("validation_year") == 2026)
    )
    rate = np.array(row["expected_importations_samples"][0])
    probability = np.array(row["importation_probability_samples"][0])
    assert probability == pytest.approx(1 - np.exp(-rate))
    assert row["importation_probability_mean"][0] == pytest.approx(
        probability.mean()
    )


def test_a_seed_makes_the_draws_reproducible():
    states = calculate_yearly_import_probability(state_table(), 16)

    def first_draws(seed):
        out, _ = distribute_state_level_risk_to_county(
            states, county_rows(), KNOWN, seed
        )
        return rate_draws(out, "c1", 2026, "projection")

    assert (first_draws(5) == first_draws(5)).all()
    assert not (first_draws(5) == first_draws(6)).all()


def test_a_county_without_a_population_is_refused():
    states = calculate_yearly_import_probability(state_table(), 16)
    rows = county_rows().with_columns(
        pop=pl.when(pl.col("fips") == "w1").then(None).otherwise(pl.col("pop"))
    )
    with pytest.raises(ValueError, match="no population share"):
        distribute_state_level_risk_to_county(states, rows, KNOWN, seed=1)


def test_a_county_with_older_vaccination_data_has_the_same_posterior():
    # c2's data is from 2021 instead of 2024, but both predict 2025: the same
    # draws of the state's rate, so the rates differ only by their shares
    states = calculate_yearly_import_probability(state_table(), 16)
    out, _ = distribute_state_level_risk_to_county(
        states, county_rows(c2_data_year=2021), KNOWN, seed=1
    )
    c1 = rate_draws(out, "c1", 2025, "projection")
    c2 = rate_draws(out, "c2", 2025, "projection")
    assert c1 / c2 == pytest.approx(np.full(8000, 3.0))


def test_the_share_is_over_the_state_population_of_the_same_validation_year():
    # in 2025 the two California counties have 2e6 each: c1's share is 1/2
    # in 2025 and 3/4 in 2026
    states = calculate_yearly_import_probability(state_table(), 16)
    out, _ = distribute_state_level_risk_to_county(
        states, county_rows(pops_2025=(2e6, 2e6), train_pop=2e6), KNOWN, seed=1
    )
    in_2025 = rate_draws(out, "c1", 2025, "projection")
    in_2026 = rate_draws(out, "c1", 2026, "projection")
    assert in_2025.mean() == pytest.approx(16 / 16 * 0.5, abs=0.02)
    assert in_2026.mean() == pytest.approx(31 / 17 * 0.75, abs=0.02)
    train = rate_draws(out, "c1", 2025, "train")
    assert train / in_2025 == pytest.approx(np.full(8000, 1.0))


def test_a_row_of_a_year_without_projection_rows_uses_its_own_group():
    # a Wyoming train row of 2025 (no Wyoming projection rows that year) takes
    # the sum of the Wyoming rows of its own year and type
    states = calculate_yearly_import_probability(state_table(), 16)
    rows = pl.concat(
        [
            county_rows(),
            pl.DataFrame(
                dict(
                    fips=["w1", "w2", "w3"],
                    state=["Wyoming"] * 3,
                    county=["w1", "w2", "w3"],
                    year=[2025, 2025, 2024],
                    validation_year=[2025, 2025, 2024],
                    type=["train"] * 3,
                    pop=[1e5, 3e5, 1e5],
                )
            ),
        ]
    )
    out, _ = distribute_state_level_risk_to_county(states, rows, KNOWN, seed=1)
    w1 = out.filter(
        (pl.col("fips") == "w1")
        & (pl.col("validation_year") == 2025)
        & (pl.col("type") == "train")
    )
    # Wyoming has alpha 1 and beta 17 before 2025 (0 imports, prior alpha 1)
    # (not the 2024 train row w3: another validation year)
    assert w1["posterior_alpha"].item() == pytest.approx(1 * 0.25)
    log_share = np.log10(1e5) / (np.log10(1e5) + np.log10(3e5))
    assert w1["posterior_alpha_log"].item() == pytest.approx(1 * log_share)


# --- a region unit and the members that have a row of their own --------------

REGIONS = {"U": ["m1", "m2", "m3"]}


def unit_data(
    unit_pop, own_row_members=(), members=("m1", "m2", "m3"), others=("o1",)
):
    """The rows of validation year 2025 of one state (California): a
    projection row of every member of the region unit "U" and of every other
    county (100 people each), a train row of each member with a row of its own
    (100 people) and the train row of the unit with population `unit_pop`."""

    def row(fips, kind, pop):
        return dict(
            fips=fips,
            state="California",
            county=fips,
            year=2025 if kind == "train" else 2024,
            validation_year=2025,
            type=kind,
            pop=pop,
        )

    rows = [row(f, "projection", 100.0) for f in (*members, *others)]
    rows += [row(f, "train", 100.0) for f in own_row_members]
    rows.append(row("U", "train", unit_pop))
    return pl.DataFrame(rows)


def shares_of(rows):
    """The population share of each (fips, type) row: its posterior alpha over
    the state's alpha of all the years (California: 31)."""
    states = calculate_yearly_import_probability(state_table(), 16)
    out, _ = distribute_state_level_risk_to_county(
        states, rows, KNOWN, seed=1, regions=REGIONS
    )
    return {
        (r["fips"], r["type"]): r["posterior_alpha"] / 31
        for r in out.select("fips", "type", "posterior_alpha").iter_rows(
            named=True
        )
    }


def test_a_region_unit_stands_for_the_members_without_a_row_of_their_own():
    # m1 has a row of its own, so the unit stands for m2 and m3 (200 of the
    # state's 400 people), not for the 300 of its own population: the unit,
    # m1 and o1 are the places left, and they add up to the state
    shares = shares_of(unit_data(unit_pop=300.0, own_row_members=["m1"]))
    assert shares[("U", "train")] == pytest.approx(200 / 400)
    assert shares[("m1", "train")] == pytest.approx(100 / 400)
    assert shares[("U", "train")] + shares[("m1", "train")] + shares[
        ("o1", "projection")
    ] == pytest.approx(1)


def test_a_region_unit_takes_the_population_of_its_members_not_its_own():
    # the unit's recorded 150 is not the 300 of its three members (say, a
    # member list from an older run): the share is that of the members
    shares = shares_of(unit_data(unit_pop=150.0))
    assert shares[("U", "train")] == pytest.approx(300 / 400)
    assert shares[("U", "train")] + shares[("o1", "projection")] == (
        pytest.approx(1)
    )


def test_shares_that_do_not_add_up_to_the_state_are_refused():
    # none of the unit's members is among the rows, so its 300 people are
    # extra: 300 + 400 people over the state's 400
    rows = unit_data(
        unit_pop=300.0, members=(), others=("o1", "o2", "o3", "o4")
    )
    with pytest.raises(ValueError, match="do not add up to 1"):
        shares_of(rows)


def test_the_expected_cases_are_the_average_of_the_simulated_cases():
    from outbreak_size.importation import (
        draw_imports,
        expected_case_draws,
        total_case_draws,
    )

    rate = np.array([[0.05, 0.4, 1.5, 6.0]])
    size = np.array([[40.0, 5.0, 12.0, 3.0]])
    probability = 1 - np.exp(-rate)
    expected = expected_case_draws(size, probability, rate)[0]
    # by hand: size * (1 - exp(-rate)) + rate
    assert expected.tolist() == pytest.approx(
        [
            40 * (1 - np.exp(-0.05)) + 0.05,
            5 * (1 - np.exp(-0.4)) + 0.4,
            12 * (1 - np.exp(-1.5)) + 1.5,
            3 * (1 - np.exp(-6.0)) + 6.0,
        ]
    )
    # and the mean of many simulated counties of the same draw
    n = 400_000
    imports = draw_imports(np.tile(rate, (n, 1)), np.random.default_rng(0))
    simulated = total_case_draws(np.tile(size, (n, 1)), imports).mean(axis=0)
    assert simulated == pytest.approx(expected, rel=0.03)


def test_the_expected_cases_do_not_depend_on_a_seed_and_keep_the_shape():
    from outbreak_size.importation import expected_case_draws

    size = np.arange(6.0).reshape(2, 3)
    rate = np.full((2, 3), 0.5)
    probability = 1 - np.exp(-rate)
    out = expected_case_draws(size, probability, rate)
    assert out.shape == (2, 3)
    assert (out == expected_case_draws(size, probability, rate)).all()


# --- the importation draws of the projection counties ---------------------------


def import_rows():
    """Importation rows of three counties: the projection rows of validation
    year 2026 (the ones used) and two rows that must be left out (a train row
    and a row of another validation year), with 2 draws each."""

    def row(fips, year, kind, probability, count):
        return dict(
            fips=fips,
            validation_year=year,
            type=kind,
            importation_probability_samples=[probability, probability / 2],
            expected_importations_samples=[count, count * 2],
        )

    return pl.DataFrame(
        [
            row("01", 2026, "projection", 0.1, 1.0),
            row("02", 2026, "projection", 0.2, 2.0),
            row("03", 2026, "projection", 0.3, 3.0),
            row("01", 2026, "train", 0.9, 9.0),
            row("02", 2025, "projection", 0.8, 8.0),
        ]
    )


def test_future_draws_are_those_of_the_2026_projection_rows_by_county():
    base = pl.DataFrame({"fips": ["03", "01", "02"], "pop": [3, 1, 2]})
    probability, count = future_importation_draws(import_rows(), base)
    # (n_draws, n_counties), the counties in the order of future_base
    assert probability.shape == (2, 3)
    assert probability.tolist() == [[0.3, 0.1, 0.2], [0.15, 0.05, 0.1]]
    assert count.tolist() == [[3.0, 1.0, 2.0], [6.0, 2.0, 4.0]]


def test_future_draws_refuse_a_county_without_a_row():
    base = pl.DataFrame({"fips": ["01", "99"]})
    with pytest.raises(ValueError, match="no importation_probability_samples"):
        future_importation_draws(import_rows(), base)


def test_future_draws_refuse_two_rows_of_a_county():
    rows = pl.concat([import_rows(), import_rows().head(1)])
    with pytest.raises(ValueError, match="two importation rows"):
        future_importation_draws(rows, pl.DataFrame({"fips": ["01"]}))


def test_future_draws_refuse_a_county_twice_in_the_projections():
    base = pl.DataFrame({"fips": ["01", "01"]})
    with pytest.raises(ValueError, match="twice in the projections"):
        future_importation_draws(import_rows(), base)
