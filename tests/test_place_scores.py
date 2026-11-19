"""The table of every place's scores (outbreak_size.place_scores)."""

import numpy as np
import polars as pl
import pytest

from outbreak_size.place_scores import (
    place_score_table,
    place_value_table,
    score_column,
)

CONDITIONS = ("all", "outbreak")


def frame(n, offset=0):
    return pl.DataFrame(
        {
            "fips": [f"0{i + offset}" for i in range(n)],
            "state": ["A"] * n,
            "county": [f"c{i}" for i in range(n)],
            "pop": [100] * n,
            "cases": list(range(n)),
            "cases_outbreak": list(range(n)),
            "log_income": [1.0] * n,
        }
    )


def test_one_row_per_place_and_year_with_a_column_per_model_and_condition():
    places = {2021: frame(2), 2022: frame(3)}
    scores = {
        ("M1", 2021): ([1.0, 2.0], [3.0, 4.0]),
        ("M1", 2022): ([5.0, 6.0, 7.0], [8.0, 9.0, 10.0]),
        ("M2", 2021): ([0.5, 0.6], [0.7, 0.8]),
        ("M2", 2022): ([1.5, 1.6, 1.7], [1.8, 1.9, 2.0]),
    }
    out = place_score_table(places, scores, CONDITIONS, ["log_income"])
    assert out.height == 5
    assert out["year"].to_list() == [2021, 2021, 2022, 2022, 2022]
    assert out[score_column("all", "M1")].to_list() == [1, 2, 5, 6, 7]
    assert out[score_column("outbreak", "M2")].to_list() == [
        0.7,
        0.8,
        1.8,
        1.9,
        2.0,
    ]
    assert out.columns[:7] == [
        "year",
        "fips",
        "state",
        "county",
        "pop",
        "cases",
        "cases_outbreak",
    ]
    assert "log_income" in out.columns


def test_a_missing_year_or_a_wrong_number_of_scores_is_an_error():
    places = {2021: frame(2), 2022: frame(3)}
    with pytest.raises(ValueError, match="no scores in 2022"):
        place_score_table(
            places, {("M", 2021): ([1.0, 2.0], [1.0, 2.0])}, CONDITIONS, []
        )
    wrong = {
        ("M", 2021): ([1.0, 2.0], [1.0, 2.0]),
        ("M", 2022): ([1.0], [1.0, 2.0, 3.0]),
    }
    with pytest.raises(ValueError, match="1 scores in 2022"):
        place_score_table(places, wrong, CONDITIONS, [])


def test_cases_saved_as_text_become_integers():
    places = {
        2021: frame(2).with_columns(pl.col("cases_outbreak").cast(pl.String))
    }
    out = place_score_table(
        places, {("M", 2021): ([1.0, 2.0], [1.0, 2.0])}, CONDITIONS, []
    )
    assert (
        out["cases_outbreak"].dtype == pl.Int64
        and out["cases"].dtype == pl.Int64
    )
    assert out["cases_outbreak"].to_list() == [0, 1]


def test_the_value_table_has_the_places_of_each_condition_and_no_row_for_a_ranking():
    values = {
        ("M", 2021, "all"): {
            "forecast": np.array([1.0, 2.0, 3.0]),
            "distribution": {"WIS (cases)": np.array([0.1, 0.2, 0.3])},
        },
        ("M", 2021, "outbreak"): {
            "forecast": np.array([5.0, 6.0]),
            "distribution": {"WIS (cases)": np.array([0.5, 0.6])},
        },
        ("R", 2021, "all"): {"forecast": None, "distribution": None},
    }
    out = place_value_table(
        values, {2021: np.array([True, False, True])}, CONDITIONS
    )
    assert out.height == 5
    assert out.filter(pl.col("Condition") == "outbreak")["row"].to_list() == [
        0,
        2,
    ]  # the places with a case
    assert out.filter(pl.col("Condition") == "all")["row"].to_list() == [
        0,
        1,
        2,
    ]
    assert "R" not in out["Model"].to_list()
    with pytest.raises(ValueError, match="forecasts for"):
        place_value_table(
            values, {2021: np.array([True, True, True])}, CONDITIONS
        )
