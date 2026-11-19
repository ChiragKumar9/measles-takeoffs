"""The yearly scores summarized over a window, with a bootstrap interval
(outbreak_size.summary_scores)."""

import numpy as np
import polars as pl
import pytest

from outbreak_size.county_scores import CONDITIONS, outcome_scores
from outbreak_size.summary_scores import (
    YearPlaces,
    check_means,
    column_level,
    expected_from_yearly,
    is_outcome_column,
    level_is_scored,
    stratified_multiplicity,
    summarize,
    summary_table,
)

CASES = CONDITIONS[0]


def places(truth, offset=0.0):
    truth = np.array(truth)
    return YearPlaces(
        truth=truth,
        score=truth + np.linspace(0, 1, len(truth)) + offset,
        state=np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]),
        windows={level: truth >= level for level in (1, 5, 10, 20)},
        forecast=None,
        distribution=None,
    )


# 2021 has places with 10+ and 20+ cases; 2022 has a place with 7 cases and none with 10+
YEAR_2021 = places([0, 0, 0, 1, 3, 7, 12, 0, 25, 0])
YEAR_2022 = places([0, 0, 0, 1, 3, 7, 0, 0, 0, 0], offset=0.3)
YEARS = {2021: YEAR_2021, 2022: YEAR_2022}


def test_the_level_of_a_column_and_its_kind():
    assert column_level("AUROC, 10+ (%)") == 10
    assert column_level("Sensitivity, 5+, top 20% (%)") == 5
    assert column_level("RMSE, 3+ (cases)") == 3
    assert column_level("WIS (cases)") is None
    assert is_outcome_column("PPV, 1+, top 10%, any window (%)")
    assert not is_outcome_column("RMSE, 0+ (cases)")
    assert not is_outcome_column("Spearman rho, 3+")


def test_a_year_is_scored_at_a_level_when_places_reach_it():
    truth = [0, 3, 6, 0]
    assert level_is_scored(truth, 5, outcome=True)
    assert not level_is_scored(truth, 10, outcome=True)  # nothing to find
    assert level_is_scored(truth, 0, outcome=False)  # every place counts at 0+
    assert not level_is_scored(
        [4, 5, 6], 3, outcome=True
    )  # every place has it: no AUROC
    assert level_is_scored([4, 5, 6], 3, outcome=False)


def test_the_places_of_a_state_are_drawn_together_within_their_level():
    truth = np.array([0, 0, 12, 0, 0, 0])
    state = np.array([0, 0, 1, 2, 2, 3])
    rng = np.random.default_rng(0)
    for _ in range(50):
        m = stratified_multiplicity(truth, state, rng)
        # the only place of 10+ cases is the only member of its group: always once
        assert m[2] == 1
        # the two places of state 0 are drawn together, and so are those of state 2
        assert m[0] == m[1] and m[3] == m[4]
    # states are resampled with replacement: not every draw is the same
    draws = {
        tuple(stratified_multiplicity(truth, state, np.random.default_rng(s)))
        for s in range(30)
    }
    assert len(draws) > 1


def test_a_year_without_a_level_is_left_out_for_that_level_only():
    out = summarize(
        "m",
        CASES,
        YEARS,
        {"both": [2021, 2022]},
        replicates=8,
        seed=1,
        condition_index=0,
    )["both"]
    a21 = outcome_scores(YEAR_2021.score, YEAR_2021.truth, 5)[0]
    a22 = outcome_scores(YEAR_2022.score, YEAR_2022.truth, 5)[0]
    # 5+ is reached in both years: the mean of the two
    assert out["AUROC, 5+ (%)"][0] == pytest.approx((a21 + a22) / 2)
    # 10+ and 20+ only in 2021: that year alone
    assert out["AUROC, 10+ (%)"][0] == pytest.approx(
        outcome_scores(YEAR_2021.score, YEAR_2021.truth, 10)[0]
    )
    assert out["AUROC, 20+ (%)"][0] == pytest.approx(
        outcome_scores(YEAR_2021.score, YEAR_2021.truth, 20)[0]
    )
    # the other metrics of the level follow the same years: the PPV at 10+ is that of 2021
    assert not np.isnan(out["PPV, 10+, top 10% (%)"][0])
    # a window with no year that has the level has no score
    only_2022 = summarize(
        "m",
        CASES,
        YEARS,
        {"2022": [2022]},
        replicates=4,
        seed=1,
        condition_index=0,
    )["2022"]
    assert np.isnan(only_2022["AUROC, 10+ (%)"][0])
    assert np.isnan(only_2022["PPV, 10+, top 10% (%)"][0]) and not np.isnan(
        only_2022["PPV, 5+, top 10% (%)"][0]
    )


def test_the_interval_is_reproducible_and_brackets_nothing_impossible():
    a = summarize(
        "m",
        CASES,
        YEARS,
        {"both": [2021, 2022]},
        replicates=30,
        seed=7,
        condition_index=0,
    )["both"]
    b = summarize(
        "m",
        CASES,
        YEARS,
        {"both": [2021, 2022]},
        replicates=30,
        seed=7,
        condition_index=0,
    )["both"]
    assert a == b
    c = summarize(
        "m",
        CASES,
        YEARS,
        {"both": [2021, 2022]},
        replicates=30,
        seed=8,
        condition_index=0,
    )["both"]
    assert a != c
    for column, (value, lower, upper) in a.items():
        if not np.isnan(lower):
            assert lower <= upper, column
    # percentages stay between 0 and 100
    assert 0 <= a["AUROC, 5+ (%)"][1] <= a["AUROC, 5+ (%)"][2] <= 100


def test_the_table_has_a_mean_a_lower_and_an_upper_column_for_every_score():
    s = summarize(
        "m",
        CASES,
        YEARS,
        {"2021-2022": [2021, 2022], "2022-2022": [2022]},
        replicates=5,
        seed=1,
        condition_index=0,
    )
    table = summary_table({("m", CASES): s})
    assert table.height == 2
    assert table["Evaluation year"].to_list() == ["2021-2022", "2022-2022"]
    assert "AUROC, 5+ (%)" in table.columns
    assert (
        "AUROC, 5+ (%), lower" in table.columns
        and "AUROC, 5+ (%), upper" in table.columns
    )
    # the mean, lower and upper columns of a score sit together
    columns = table.columns
    i = columns.index("AUROC, 5+ (%)")
    assert columns[i + 1 : i + 3] == [
        "AUROC, 5+ (%), lower",
        "AUROC, 5+ (%), upper",
    ]
    assert not any(c.startswith("Places") for c in columns)


def test_the_places_of_a_model_come_from_the_place_file_and_the_value_table():
    import polars as pl

    from outbreak_size.county_scores import DISTRIBUTION_COLUMNS
    from outbreak_size.place_scores import place_value_table, score_column
    from outbreak_size.summary_scores import (
        value_groups,
        window_column,
        year_places,
    )

    cases = [0, 2, 0, 12]
    frame = pl.DataFrame(
        {
            "year": [2021] * 4,
            "state": ["A", "A", "B", "C"],
            "cases": cases,
            "cases_outbreak": [
                str(c) for c in [0, 1, 0, 12]
            ],  # text, as the first table run wrote it
            **{
                window_column(level): [c >= level for c in cases]
                for level in (1, 5, 10, 20)
            },
            score_column(CONDITIONS[0], "M"): [0.1, 0.5, 0.2, 0.9],
            score_column(CONDITIONS[1], "M"): [0.3, 0.4, 0.2, 0.8],
        }
    )
    draws = {c: np.array([1.0, 2.0, 3.0, 4.0]) for c in DISTRIBUTION_COLUMNS}
    values = place_value_table(
        {
            ("M", 2021, CONDITIONS[0]): {
                "forecast": np.array([1.0, 2.0, 3.0, 4.0]),
                "distribution": draws,
            },
            ("M", 2021, CONDITIONS[1]): {
                "forecast": np.array([7.0, 8.0]),
                "distribution": {
                    c: np.array([5.0, 6.0]) for c in DISTRIBUTION_COLUMNS
                },
            },
        },
        {2021: np.array([c >= 1 for c in cases])},
        CONDITIONS,
    )
    groups = value_groups(values)
    every = year_places(frame, groups, "M", CONDITIONS[0], 2021)
    assert every.truth.tolist() == cases and every.windows[10].tolist() == [
        False,
        False,
        False,
        True,
    ]
    assert every.forecast.tolist() == [1.0, 2.0, 3.0, 4.0]
    # the outbreak cases are those of the places with a case: places 1 and 3
    outbreak = year_places(frame, groups, "M", CONDITIONS[1], 2021)
    assert outbreak.truth.tolist() == [
        1,
        12,
    ]  # the outbreak cases, read as numbers
    assert outbreak.score.tolist() == [0.4, 0.8] and outbreak.windows is None
    assert outbreak.forecast.tolist() == [7.0, 8.0]
    assert outbreak.state.tolist() == [0, 1]


def test_the_means_of_the_summary_are_checked_against_the_yearly_table():
    yearly = pl.DataFrame(
        {
            "Model": ["M", "M"],
            "Condition": [CASES, CASES],
            "Evaluation year": ["2021", "2022"],
            "Places scored": [10, 10],
            "Places with 10+ cases": [2, 0],
            "AUROC, 10+ (%)": [80.0, float("nan")],
            "WIS": [1.0, 3.0],
        }
    )
    assert (
        expected_from_yearly(
            yearly, "M", CASES, [2021, 2022], "AUROC, 10+ (%)"
        )
        == 80.0
    )
    assert expected_from_yearly(yearly, "M", CASES, [2021, 2022], "WIS") == 2.0
    table = pl.DataFrame(
        {
            "Model": ["M"],
            "Evaluation year": ["2021-2025"],
            "Condition": [CASES],
            "WIS": [2.0],
            "WIS, lower": [0.0],
        }
    )
    assert check_means(table, yearly, {"2021-2025": [2021, 2022]}) == (1, 0.0)
    with pytest.raises(ValueError):
        check_means(
            table.with_columns(pl.lit(2.5).alias("WIS")),
            yearly,
            {"2021-2025": [2021, 2022]},
        )


def test_a_single_year_summary_has_the_yearly_scores_and_an_interval():
    from polars.testing import assert_frame_equal

    from outbreak_size.county_scores import ScoreBook
    from outbreak_size.place_scores import (
        place_score_table,
        place_value_table,
        score_column,
    )
    from outbreak_size.summary_scores import (
        single_year_summary,
        window_column,
    )

    rng = np.random.default_rng(0)
    n = 120
    truth = np.zeros(n, dtype=int)
    # places at every level, in different states (a state is 4 places)
    truth[[3, 9, 14, 27, 40, 55, 70, 88, 101, 110]] = [
        1, 2, 6, 12, 25, 3, 7, 8, 15, 30,
    ]  # fmt: skip
    introduced = truth >= 1
    windows = {level: truth >= level for level in (1, 5, 10, 20)}
    book = ScoreBook(workers=1)
    place_scores = {}
    for model, has_draws in (("A", True), ("B", False)):
        score = truth + rng.normal(size=n) * (1.5 if has_draws else 6)
        draws = (
            rng.poisson(np.maximum(truth, 0.5) * 1.3, size=(200, n))
            if has_draws
            else None
        )
        place_scores[(model, 2026)] = (score, score)
        book.add(
            model,
            2026,
            truth,
            truth[introduced],
            score,
            score[introduced],
            draws,
            None if draws is None else draws[:, introduced],
            windows_all=windows,
        )
    yearly = book.table().filter(pl.col("Evaluation year") == "2026")
    frame = pl.DataFrame(
        {
            "year": np.full(n, 2026),
            "fips": [f"{i:05d}" for i in range(n)],
            "state": [f"S{i // 4}" for i in range(n)],
            "county": [f"C{i}" for i in range(n)],
            "pop": np.full(n, 1000),
            "cases": truth,
            "cases_outbreak": truth,
            **{window_column(k): v for k, v in windows.items()},
        }
    )
    place_table = place_score_table(
        {2026: frame},
        place_scores,
        CONDITIONS,
        [window_column(k) for k in windows],
    )
    value_table = place_value_table(
        book.place_values, {2026: introduced}, CONDITIONS
    )

    table = single_year_summary(2026, place_table, value_table, 40, 108)
    assert table.height == 4  # two models, two conditions
    assert set(table["Evaluation year"]) == {"2026"}
    assert score_column(CONDITIONS[0], "A") in place_table.columns
    # the means are the scores of the year itself, for every column
    checked, worst = check_means(table, yearly, {"2026": [2026]})
    assert checked > 100 and worst < 1e-9
    # an interval for every score that the year can be scored on, in order
    both = [c for c in table.columns if c.endswith(", lower")]
    assert both
    for column in both:
        lower, upper = (
            table[column],
            table[column[: -len(", lower")] + ", upper"],
        )
        ok = lower.is_not_nan() & lower.is_not_null()
        assert (lower.filter(ok) <= upper.filter(ok) + 1e-9).all(), column
    auroc = table.filter(
        (pl.col("Model") == "A") & (pl.col("Condition") == CONDITIONS[0])
    )
    assert (
        auroc["AUROC, 10+ (%), lower"][0] < auroc["AUROC, 10+ (%), upper"][0]
    )
    # a model with draws has the scores of its distribution, one without has none
    assert (
        table.filter(pl.col("Model") == "A")["WIS (cases)"].is_not_null().all()
    )
    assert table.filter(pl.col("Model") == "B")["WIS (cases)"].is_null().all()
    # the draws are those S1 uses for the year: the seed, the condition's
    # position (0 cases, 1 outbreak cases) and the year
    from outbreak_size.summary_scores import value_groups, year_places

    for index, condition in enumerate(CONDITIONS):
        expected = summarize(
            "A",
            condition,
            {
                2026: year_places(
                    place_table,
                    value_groups(value_table),
                    "A",
                    condition,
                    2026,
                )
            },
            {"2026": [2026]},
            40,
            108,
            index,
        )["2026"]
        row = table.filter(
            (pl.col("Model") == "A") & (pl.col("Condition") == condition)
        )
        for column, values in expected.items():
            got = (
                row[column][0],
                row[f"{column}, lower"][0],
                row[f"{column}, upper"][0],
            )
            assert got == pytest.approx(values, nan_ok=True), column
    # the same seed gives the same interval, another seed another one
    assert_frame_equal(
        table, single_year_summary(2026, place_table, value_table, 40, 108)
    )
    other = single_year_summary(2026, place_table, value_table, 40, 7)
    assert not other["AUROC, 10+ (%), lower"].equals(
        table["AUROC, 10+ (%), lower"]
    )
