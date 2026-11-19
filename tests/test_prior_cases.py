"""The prior-cases benchmark (outbreak_size.prior_cases) and the scoring of a
ranking with many ties."""

import numpy as np
import polars as pl
import pytest

from outbreak_size.county_scores import outcome_scores, top_share_metrics
from outbreak_size.prior_cases import place_key, prior_cases_score

# place "a": 3 cases in 2011, 2 in 2020, 5 in 2022; place "b": 1 case in 2010
# (before the first year) and 4 in 2024; place "c": none
HISTORY = pl.DataFrame(
    {
        "place": ["a", "a", "a", "b", "b"],
        "year": [2011, 2020, 2022, 2010, 2024],
        "cases": [3, 2, 5, 1, 4],
    }
)
PLACES = pl.Series(["a", "b", "c", "never seen"])


def test_the_score_sums_the_cases_from_2011_up_to_the_year_before():
    # 2022 excludes its own year: a has 3 + 2; b's 2010 case is before 2011
    assert prior_cases_score(HISTORY, PLACES, 2022).tolist() == [5, 0, 0, 0]
    # 2023 adds a's 2022 cases
    assert prior_cases_score(HISTORY, PLACES, 2023).tolist() == [10, 0, 0, 0]
    # 2025 adds b's 2024 cases; a place with no row scores 0
    assert prior_cases_score(HISTORY, PLACES, 2025).tolist() == [10, 4, 0, 0]
    # a window that starts later drops the first year
    assert prior_cases_score(
        HISTORY, PLACES, 2023, first_year=2012
    ).tolist() == [7, 0, 0, 0]


def test_the_scores_follow_the_order_of_the_places():
    out = prior_cases_score(HISTORY, pl.Series(["c", "b", "a"]), 2025)
    assert out.tolist() == [0, 4, 10]


def test_a_place_and_year_twice_in_the_history_is_an_error():
    twice = pl.concat([HISTORY, HISTORY.head(1)])
    with pytest.raises(ValueError, match="twice"):
        prior_cases_score(twice, PLACES, 2025)


def test_a_place_is_its_county_fips_or_else_its_state_and_name():
    rows = pl.DataFrame(
        {
            "fips": ["06037", "10", None],
            "state": ["California", "Delaware", "Virginia"],
            "county": ["Los Angeles", "State", "Fairfax"],
        }
    )
    keys = rows.select(
        place_key(pl.col("fips"), pl.col("state"), pl.col("county"))
    )
    assert keys.to_series().to_list() == [
        "06037",
        "Delaware|State",
        "Virginia|Fairfax",
    ]


def test_tied_places_are_scored_the_same_in_any_order():
    # most places tie at 0, as in the benchmark: shuffling the rows (score and
    # outcome together) must not change any metric
    rng = np.random.default_rng(0)
    score = np.array(
        [0] * 90 + [2, 2, 5, 7, 7, 9, 12, 12, 12, 30], dtype=float
    )
    truth = rng.integers(0, 15, size=100)
    outcome = truth >= 5
    base_ranking = outcome_scores(score, truth, 5)
    base_top = top_share_metrics(score, outcome, None, 0.10)
    for _ in range(5):
        order = rng.permutation(100)
        assert outcome_scores(score[order], truth[order], 5) == pytest.approx(
            base_ranking
        )
        shuffled = top_share_metrics(score[order], outcome[order], None, 0.10)
        assert shuffled == pytest.approx(base_top)
