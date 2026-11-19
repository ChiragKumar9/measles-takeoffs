"""Flags, groups and covariate percentiles of the discordance figure
(outbreak_size.flag_discordance)."""

import numpy as np
import polars as pl
import pytest

from outbreak_size.flag_discordance import (
    COVARIATES,
    GROUPS,
    MECHANISTIC_FLAG,
    MECHANISTIC_SCORE,
    STATISTICAL_FLAG,
    STATISTICAL_SCORE,
    add_flags,
    add_groups,
    add_percentiles,
    covariate_gaps,
    different_covariates,
    discordance_places,
    discordance_rates,
    group_counts,
    group_medians,
    group_percentiles,
    top_share_flags,
)


def test_the_highest_share_of_the_scores_is_flagged():
    score = [5, 1, 4, 2, 3, 9, 8, 7, 6, 0]
    # a share of 0.2 of 10 places: the 2 highest (9 and 8)
    assert top_share_flags(score, 0.2).tolist() == [
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
    ]


def test_a_tie_above_the_cutoff_is_fine_and_a_tie_across_it_is_an_error():
    # 10 places, 2 flagged: the two 9s are the two highest and tie with each other
    score = [9, 9, 3, 2, 1, 1, 0, 0, 0, 0]
    assert top_share_flags(score, 0.2).tolist()[:3] == [True, True, False]
    # the second and third highest tie: one place would have to be flagged
    with pytest.raises(ValueError, match="tie"):
        top_share_flags([9, 8, 8, 2, 1, 1, 0, 0, 0, 0], 0.2)
    with pytest.raises(ValueError, match="no score"):
        top_share_flags([1.0, np.nan] + [0.0] * 8, 0.2)


def places():
    """Two years of 5 places, a flag share of 0.2 (one place per year per model)."""
    return pl.DataFrame(
        {
            "year": [2024] * 5 + [2025] * 5,
            "cases": [50, 12, 0, 3, 11, 30, 0, 10, 2, 1],
            STATISTICAL_SCORE: [9.0, 8, 1, 2, 3, 1, 2, 3, 9, 4],
            MECHANISTIC_SCORE: [1.0, 2, 3, 4, 9, 9, 8, 1, 2, 3],
            "fraction_white": [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.5,
                0.5,
                0.6,
                0.7,
                None,
            ],
        }
    )


def test_each_year_flags_its_own_highest_share():
    flagged = add_flags(places(), 0.2)
    # statistical: place 1 of 2024 (score 9) and place 4 of 2025 (score 9)
    assert flagged.filter(pl.col("year") == 2024)[
        STATISTICAL_FLAG
    ].to_list() == [True, False, False, False, False]
    assert flagged.filter(pl.col("year") == 2025)[
        STATISTICAL_FLAG
    ].to_list() == [False, False, False, True, False]
    # mechanistic: place 5 of 2024 and place 1 of 2025
    assert flagged.filter(pl.col("year") == 2024)[
        MECHANISTIC_FLAG
    ].to_list() == [False, False, False, False, True]
    assert flagged.filter(pl.col("year") == 2025)[
        MECHANISTIC_FLAG
    ].to_list() == [True, False, False, False, False]


def test_places_with_enough_cases_are_grouped_by_who_flagged_them():
    grouped = add_groups(add_flags(places(), 0.2), 10)
    # 10+ cases: 2024 places 1, 2, 5 and 2025 places 1, 3
    assert grouped.height == 5
    assert grouped["cases"].to_list() == [50, 12, 11, 30, 10]
    assert grouped["group"].to_list() == [
        "Statistical only",  # 2024 place 1: statistical only
        "Neither",  # 2024 place 2
        "Mechanistic only",  # 2024 place 5: mechanistic only
        "Mechanistic only",  # 2025 place 1: mechanistic only
        "Neither",  # 2025 place 3
    ]


def test_counts_per_year_and_pooled_with_the_share_the_models_disagree_on():
    grouped = add_groups(add_flags(places(), 0.2), 10)
    counts = group_counts(grouped)
    assert set(counts["Group"]) == set(GROUPS)
    # every year has a row for every group, zeros included
    assert counts.height == len(GROUPS) * 3  # 2024, 2025, All years

    def get(year, group):
        return counts.filter(
            (pl.col("Year") == year) & (pl.col("Group") == group)
        )

    assert get("2024", "Statistical only")["Places"].item() == 1
    assert get("2024", "Both")["Places"].item() == 0
    # 2024: 3 places, 2 flagged by one model only; 2025: 2 places, 1; all years 3 of 5
    assert get("2024", "Neither")[
        "Share flagged by one model only"
    ].item() == pytest.approx(2 / 3)
    assert get("2025", "Both")[
        "Share flagged by one model only"
    ].item() == pytest.approx(1 / 2)
    assert get("All years", "Both")[
        "Share flagged by one model only"
    ].item() == pytest.approx(3 / 5)
    assert get("All years", "Mechanistic only")["Places"].item() == 2


def test_percentiles_are_ranks_among_the_places_of_the_same_year_with_ties_averaged():
    out = add_percentiles(places(), ["fraction_white"])[
        "fraction_white percentile"
    ].to_list()
    # 2024: values 0.1 .. 0.5 -> ranks 1..5 of 5 -> 20, 40, 60, 80, 100
    assert out[:5] == pytest.approx([20, 40, 60, 80, 100])
    # 2025: 0.5, 0.5, 0.6, 0.7 and a missing value: ranks 1.5, 1.5, 3, 4 of the 4 with values
    assert out[5:9] == pytest.approx([37.5, 37.5, 75, 100])
    assert out[9] is None


def test_group_percentiles_give_the_quartiles_of_the_places_with_a_value():
    grouped = pl.DataFrame(
        {
            "group": ["Both"] * 5 + ["Neither"],
            "x percentile": [10.0, 20, 30, 40, 50, None],
        }
    )
    out = group_percentiles(grouped, ["x"], ["Both", "Neither"])
    both = out.filter(pl.col("Group") == "Both").row(0, named=True)
    assert (both["Places"], both["Q1"], both["Median"], both["Q3"]) == (
        5,
        20,
        30,
        40,
    )
    none = out.filter(pl.col("Group") == "Neither").row(0, named=True)
    assert none["Places"] == 0 and np.isnan(none["Median"])


def test_a_covariate_is_different_when_the_medians_of_the_two_groups_are_far_apart():
    summary = pl.DataFrame(
        {
            "Covariate": ["a", "a", "b", "b", "c", "c"],
            "Group": ["Statistical only", "Mechanistic only"] * 3,
            "Places": [5, 5, 5, 5, 5, 0],
            "Q1": [0.0] * 6,
            "Median": [70.0, 55.0, 50.0, 60.0, 30.0, float("nan")],
            "Q3": [0.0] * 6,
        }
    )
    # a: gap 15; b: gap exactly 10 (enough); c: a group without places (no median)
    assert different_covariates(summary, 10) == ["a", "b"]
    assert different_covariates(summary, 12) == ["a"]


def test_the_outcome_column_of_the_groups_can_be_the_outbreak_cases():
    flagged = add_flags(places(), 0.2).with_columns(
        cases_outbreak=pl.Series([40, 0, 0, 0, 12, 0, 0, 11, 0, 0])
    )
    # outbreak cases of 10+: 2024 places 1 and 5, 2025 place 3 (cases 0 and 12 and 10 elsewhere)
    grouped = add_groups(flagged, 10, "cases_outbreak")
    assert grouped["cases_outbreak"].to_list() == [40, 12, 11]
    assert grouped["group"].to_list() == [
        "Statistical only",
        "Mechanistic only",
        "Neither",
    ]


def test_discordance_rates_have_the_models_own_flags_as_denominators():
    # 2024: Both 1, Statistical only 3, Mechanistic only 0
    # 2025: Both 0, Statistical only 0, Mechanistic only 2
    grouped = pl.DataFrame(
        {
            "year": [2024] * 4 + [2025] * 2,
            "group": ["Both"]
            + ["Statistical only"] * 3
            + ["Mechanistic only"] * 2,
        }
    )
    rates = discordance_rates(group_counts(grouped))

    def get(year):
        return rates.filter(pl.col("Year") == year).row(0, named=True)

    # 4 flagged by the statistical model, 3 of them not by the mechanistic
    assert get("2024")["Flagged by statistical"] == 4
    assert get("2024")["Of them not flagged by mechanistic"] == pytest.approx(
        3 / 4
    )
    # 1 flagged by the mechanistic model, shared with the statistical model
    assert get("2024")["Of them not flagged by statistical"] == 0
    assert get("2024")["Of them flagged by one only"] == pytest.approx(3 / 4)
    # no place flagged by the statistical model in 2025: no share
    assert np.isnan(get("2025")["Of them not flagged by mechanistic"])
    assert get("2025")["Of them not flagged by statistical"] == 1
    # all years: 6 places, 5 flagged by one model only
    assert get("All years")["Flagged by either"] == 6
    assert get("All years")["Of them flagged by one only"] == pytest.approx(
        5 / 6
    )


def test_covariate_gaps_use_strictly_more_than_the_minimum_and_both_units():
    grouped = pl.DataFrame(
        {
            "group": ["Statistical only"] * 3 + ["Mechanistic only"] * 3,
            "x": [0.5, 0.7, 0.9, 0.1, 0.2, 0.3],
            "x percentile": [70.0, 80, 90, 55, 60, 65],
            "y": [1.0, 2, 3, 1, 2, 3],
            # a gap of exactly 10 points: not more than 10
            "y percentile": [70.0, 70, 70, 60, 60, 60],
        }
    )
    groups = ["Statistical only", "Mechanistic only"]
    gaps = covariate_gaps(
        group_percentiles(grouped, ["x", "y"], groups),
        group_medians(grouped, ["x", "y"], groups),
        10,
    )
    x, y = (
        gaps.filter(pl.col("Covariate") == c).row(0, named=True) for c in "xy"
    )
    assert x["Percentile gap"] == pytest.approx(20.0)  # medians 80 and 60
    assert x["Gap in own units"] == pytest.approx(0.5)  # medians 0.7 and 0.2
    assert x["Gap over 10 points"] is True
    assert y["Percentile gap"] == pytest.approx(10.0)
    assert y["Gap over 10 points"] is False


def discordance_input():
    """2022: 11 places, ten with cases (a flag share of 0.2 flags two per model) and
    one without (its covariates still count in the percentiles); 2021: two places
    with many cases that the first year leaves out."""
    n = 11
    stat = [9.0, 8, 1, 2, 3, 4, 5, 6, 7, 0.5, 0.1]
    mech = [1.0, 2, 3, 4, 9, 8, 5, 6, 7, 0.5, 0.1]
    cases = [50, 12, 8, 3, 11, 30, 6, 10, 2, 1, 0]
    rows = pl.DataFrame(
        {
            "year": [2022] * n,
            "fips": [f"{i:05d}" for i in range(n)],
            "state": ["S"] * n,
            "county": [f"C{i}" for i in range(n)],
            "cases": cases,
            "cases_outbreak": cases,
            STATISTICAL_SCORE: stat,
            MECHANISTIC_SCORE: mech,
            **{c: [float(i) for i in range(n)] for c in COVARIATES},
        }
    )
    older = rows.head(2).with_columns(
        year=pl.lit(2021, dtype=pl.Int64),
        cases=pl.lit(99, dtype=pl.Int64),
        cases_outbreak=pl.lit(99, dtype=pl.Int64),
    )
    return pl.concat([older, rows])


def test_only_the_flagged_places_with_enough_cases_are_kept():
    kept = discordance_places(
        discordance_input(),
        2022,
        share=0.2,
        level=10,
        outcome="cases_outbreak",
    )
    # the places with 10+ outbreak cases: 0, 1 (flagged by the statistical
    # model), 4, 5 (the mechanistic model) and 7 (neither: left out); most cases first
    assert kept["fips"].to_list() == ["00000", "00005", "00001", "00004"]
    assert kept["group"].to_list() == [
        "Statistical only",
        "Mechanistic only",
        "Statistical only",
        "Mechanistic only",
    ]
    assert set(kept["year"]) == {2022}


def test_the_file_holds_percentiles_and_covariates_and_no_model_score():
    kept = discordance_places(
        discordance_input(),
        2022,
        share=0.2,
        level=10,
        outcome="cases_outbreak",
    )
    assert STATISTICAL_SCORE not in kept.columns
    assert MECHANISTIC_SCORE not in kept.columns
    assert kept.columns[:8] == [
        "year",
        "fips",
        "state",
        "county",
        "cases_outbreak",
        "group",
        f"{STATISTICAL_SCORE} percentile",
        f"{MECHANISTIC_SCORE} percentile",
    ]
    first = kept.row(0, named=True)  # place 0
    # the score percentile is among the 10 places with a case of the year (9.0 is
    # the highest), the covariate percentile among all 11 places (value 0.0 is
    # the lowest)
    assert first[f"{STATISTICAL_SCORE} percentile"] == pytest.approx(100.0)
    assert first[f"{MECHANISTIC_SCORE} percentile"] == pytest.approx(20.0)
    assert first["estimated_mmr_cov"] == 0.0
    assert first["estimated_mmr_cov percentile"] == pytest.approx(100 / 11)
