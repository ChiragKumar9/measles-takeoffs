"""CCD kindergarten weights of the school SD (outbreak_size.ccd_enrollment)."""

import numpy as np
import polars as pl
import pytest

from outbreak_size.ccd_enrollment import (
    county_sds,
    match_to_ccd,
    normalize_name,
    weighted_sd,
)


def test_weighted_sd_is_the_population_sd_of_the_weighted_rates():
    # mean (90 + 3 * 100) / 4 = 97.5; variance (7.5^2 + 3 * 2.5^2) / 4 = 18.75
    assert weighted_sd([90, 100], [1, 3]) == pytest.approx(np.sqrt(18.75))
    assert weighted_sd([90, 100], [5, 5]) == pytest.approx(5.0)  # unweighted


def test_weighted_sd_leaves_out_rows_of_weight_0_and_handles_one_row():
    assert weighted_sd([90, 100, 10], [1, 3, 0]) == pytest.approx(
        weighted_sd([90, 100], [1, 3])
    )
    assert weighted_sd([90, 10], [4, 0]) == 0.0  # one school: exactly 0
    assert weighted_sd([90, 100], [0, 0]) is None


def test_normalize_name_ignores_case_punctuation_and_filler_words():
    assert normalize_name("Alpha Elementary School") == "alpha elementary"
    assert normalize_name("ALPHA  ELEMENTARY, SCHOOL") == "alpha elementary"
    assert normalize_name(None) is None


SCHOOLS = pl.DataFrame(
    {
        "ncessch": ["s1", "s2", "s3", "s4"],
        "leaid": ["d1", "d1", "d1", "d2"],
        "school_name": [
            "Alpha Elementary School",
            "Beta Middle School",
            "Gamma Elementary",
            "Delta Elementary",
        ],
        "state": ["TX"] * 4,
        "county_fips": ["48001"] * 4,
        "lat": [40.0, 40.1, 40.2, 40.3],
        "lon": [-100.0, -100.0, -100.0, -100.0],
        "kg_offered": ["Yes", "No", "Yes", "Yes"],
        "kg_count": [50, None, None, 20],  # Gamma offers it but has no count
    },
    schema_overrides={"kg_count": pl.Int64},
)
DISTRICTS = pl.DataFrame(
    {
        "leaid": ["d1", "d2"],
        "lea_name": ["Alpha ISD", "Delta ISD"],
        "kg_count": [70, 20],
    }
)


def rows(*records):
    return pl.DataFrame(
        records,
        schema=[
            "school_name",
            "unit",
            "lon",
            "lat",
            "mmr_rate",
            "enrollment",
        ],
        orient="row",
    ).with_columns(
        state=pl.lit("TX"),
        county_fips=pl.lit("48001"),
        mmr_rate=pl.col("mmr_rate").cast(pl.Float64),
        enrollment=pl.col("enrollment").cast(pl.Float64),
    )


def test_a_row_matches_by_name_or_by_place_and_takes_the_kindergarten_weight():
    matched = match_to_ccd(
        rows(
            ("ALPHA ELEMENTARY", "school", None, None, 90, None),  # by name
            ("Some name", "school", -100.0003, 40.0, 80, None),  # 25 m away
            ("Beta Middle", "school", -100.0, 40.1, 70, None),  # no K
            ("Gamma Elementary", "school", -100.0, 40.2, 60, None),  # no count
            ("Nowhere", "school", -99.0, 41.0, 50, None),  # no match
        ),
        SCHOOLS,
        DISTRICTS,
    )
    assert matched["match"].to_list() == [
        "name",
        "xy",
        "name+xy",
        "name+xy",
        "none",
    ]
    assert matched["ncessch"].to_list() == ["s1", "s1", "s2", "s3", None]
    assert matched["ccd_weight"].to_list() == [50.0, 50.0, 0.0, None, None]


def test_a_district_row_matches_the_district_of_its_name_in_the_county():
    matched = match_to_ccd(
        rows(
            ("Alpha ISD", "district", None, None, 90, None),
            ("Other ISD", "district", None, None, 80, None),
        ),
        SCHOOLS,
        DISTRICTS,
    )
    assert matched["match"].to_list() == ["district_name", "none"]
    assert matched["ccd_weight"].to_list() == [70.0, None]


def county_table(records, **kwargs):
    matched = match_to_ccd(rows(*records), SCHOOLS, DISTRICTS)
    return county_sds(matched, **kwargs).row(0, named=True)


def test_ccd_only_weights_a_county_whose_rows_are_all_matched():
    row = county_table(
        [
            ("Alpha Elementary", "school", None, None, 90, None),
            ("Delta Elementary", "school", None, None, 100, None),
            ("Beta Middle", "school", None, None, 10, None),  # no K: leaves
        ]
    )
    # weights 50 and 20 (Delta is in the county): mean (4500 + 2000) / 70
    mean = (90 * 50 + 100 * 20) / 70
    expected = np.sqrt((50 * (90 - mean) ** 2 + 20 * (100 - mean) ** 2) / 70)
    assert row["sd_ccd"] == pytest.approx(expected)


def test_ccd_only_needs_every_row_matched_unless_a_share_is_allowed():
    records = [
        ("Alpha Elementary", "school", None, None, 90, 30),
        ("Delta Elementary", "school", None, None, 100, 30),
        ("Unknown School", "school", None, None, 60, 30),  # not in CCD
    ]
    assert county_table(records)["sd_ccd"] is None
    allowed = county_table(records, max_unmatched_share=0.4)
    assert allowed["sd_ccd"] == pytest.approx(weighted_sd([90, 100], [50, 20]))


def test_a_matched_school_with_a_missing_count_makes_the_county_incomplete():
    row = county_table(
        [
            ("Alpha Elementary", "school", None, None, 90, None),
            ("Gamma Elementary", "school", None, None, 60, None),
        ],
        max_unmatched_share=1.0,
    )
    assert row["sd_ccd"] is None


def test_the_hierarchy_takes_our_enrollment_for_a_row_ccd_does_not_have():
    row = county_table(
        [
            ("Alpha Elementary", "school", None, None, 90, None),  # CCD: 50
            ("Unknown School", "school", None, None, 60, 25),  # ours: 25
        ]
    )
    assert row["sd_ccd"] is None
    assert row["sd_hierarchy"] == pytest.approx(
        weighted_sd([90, 60], [50, 25])
    )
    # a row with neither weight leaves the county unweighted
    row = county_table(
        [
            ("Alpha Elementary", "school", None, None, 90, None),
            ("Unknown School", "school", None, None, 60, None),
        ]
    )
    assert row["sd_hierarchy"] is None
