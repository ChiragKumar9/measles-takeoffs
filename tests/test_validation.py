"""Guards that stop a run whose inputs do not line up (outbreak_size.validation)."""

import itertools

import numpy as np
import polars as pl
import pytest

from outbreak_size.model_inputs import COVARIATE_SCALES
from outbreak_size.poisson import LIKELIHOODS
from outbreak_size.validation import (
    keep_rows,
    projection_twin_mask,
    region_member_groups,
    region_member_mask,
    require_no_nulls,
    require_same_fips,
    require_unique_places,
    require_width,
    validation_file_name,
    variant_file_suffix,
)


def test_the_variant_of_the_existing_fit_files_has_no_suffix():
    # these fits (negative binomial, natural scale) were made before there
    # were variants, and their file names have to stay valid
    assert variant_file_suffix("negative_binomial", "natural") == ""


@pytest.mark.parametrize(
    "likelihood, covariate_scale, expected",
    [
        ("poisson", "natural", "_likelihood=poisson"),
        ("negative_binomial", "raw", "_covariate_scale=raw"),
        ("poisson", "raw", "_likelihood=poisson_covariate_scale=raw"),
    ],
)
def test_other_variants_say_what_differs(
    likelihood, covariate_scale, expected
):
    assert variant_file_suffix(likelihood, covariate_scale) == expected


def test_no_two_variants_write_to_the_same_file():
    suffixes = [
        variant_file_suffix(likelihood, scale)
        for likelihood, scale in itertools.product(
            LIKELIHOODS, COVARIATE_SCALES
        )
    ]
    assert len(suffixes) == len(set(suffixes)) == 4


def test_an_unknown_setting_is_refused():
    with pytest.raises(ValueError, match="likelihood"):
        variant_file_suffix("gaussian", "natural")
    with pytest.raises(ValueError, match="covariate_scale"):
        variant_file_suffix("poisson", "log")


def test_keep_rows_keeps_the_marked_columns():
    draws = np.arange(12).reshape(3, 4)
    keep = np.array([True, False, True, True])
    kept = keep_rows(draws, keep, "draws")
    assert kept.shape == (3, 3)
    assert (kept == draws[:, [0, 2, 3]]).all()


def test_keep_rows_raises_when_the_widths_differ():
    draws = np.zeros((3, 4))
    with pytest.raises(ValueError, match="different runs"):
        keep_rows(draws, np.array([True, False, True]), "draws")


def test_require_width_returns_the_draws_or_raises():
    draws = np.zeros((2, 5))
    assert require_width(draws, 5, "draws") is draws
    with pytest.raises(ValueError, match="5 columns of draws for 6 rows"):
        require_width(draws, 6, "draws")


def test_require_unique_places_allows_a_county_in_different_years():
    df = pl.DataFrame(
        {"fips": ["01001", "01001"], "validation_year": [2021, 2022]}
    )
    require_unique_places(df, "rows")


def test_require_unique_places_does_not_check_places_keyed_by_name():
    df = pl.DataFrame(
        {
            "fips": ["North, Alabama", "North, Alabama"],
            "validation_year": [2021, 2021],
        }
    )
    require_unique_places(df, "rows")


def test_require_unique_places_raises_on_a_county_twice_in_a_year():
    df = pl.DataFrame(
        {"fips": ["01001", "01001"], "validation_year": [2021, 2021]}
    )
    with pytest.raises(ValueError, match="repeat a county"):
        require_unique_places(df, "rows")


def test_require_no_nulls_passes_clean_data():
    df = pl.DataFrame({"state": ["A"], "x": [1.0], "y": [2]})
    require_no_nulls(df, ["x", "y"], "inputs")


def test_require_no_nulls_raises_on_null_and_names_the_places():
    df = pl.DataFrame(
        {
            "state": ["A", "B"],
            "county": ["a1", "b1"],
            "x": [1.0, None],
        }
    )
    with pytest.raises(ValueError, match="b1"):
        require_no_nulls(df, ["x"], "inputs")


def test_require_no_nulls_raises_on_nan():
    df = pl.DataFrame({"state": ["A"], "x": [float("nan")]})
    with pytest.raises(ValueError, match="missing values"):
        require_no_nulls(df, ["x"], "inputs")


# --- predictions and counties line up --------------------------------------


def test_require_same_fips_accepts_the_same_list():
    require_same_fips(["01001", "01003"], ["01001", "01003"], "year")
    require_same_fips(("01001", "01003"), ["01001", "01003"], "year")


@pytest.mark.parametrize(
    "fips",
    [
        ["01003", "01001"],
        ["01001"],
        ["01001", "01003", "01005"],
        ["01001", "x"],
    ],
)
def test_require_same_fips_raises_on_another_order_or_content(fips):
    with pytest.raises(ValueError, match="2027"):
        require_same_fips(fips, ["01001", "01003"], "2027")


def test_projection_twins_are_the_projection_rows_of_a_train_county_year():
    rows = pl.DataFrame(
        {
            "fips": ["01001", "01001", "01003", "01001", "North", "North"],
            "validation_year": [2021, 2021, 2021, 2022, 2021, 2021],
            "type": [
                "train",
                "projection",
                "projection",
                "projection",
                "train",
                "projection",
            ],
        }
    )
    mask = projection_twin_mask(rows)
    # the train rows stay, a county with only a projection row stays (01003,
    # and 01001 in 2022, when it had no outbreak), and the name-keyed place
    # is handled like a county
    assert mask.tolist() == [False, True, False, False, False, True]


def test_dropping_the_twins_leaves_one_row_per_county_year():
    rows = pl.DataFrame(
        {
            "fips": ["a", "a", "b", "b", "c"],
            "validation_year": [2021] * 5,
            "type": ["train", "projection"] * 2 + ["projection"],
        }
    )
    kept = rows.filter(pl.Series(~projection_twin_mask(rows)))
    assert kept.select("fips", "validation_year").is_duplicated().sum() == 0
    assert sorted(kept["fips"]) == ["a", "b", "c"]


# --- region units: a region is one row, its member counties are dropped -----

REGIONS = {"R": ["49001", "49017", "49021", "49099"], "S": ["49017"]}


def region_rows():
    return pl.DataFrame(
        {
            "fips": ["49001", "49017", "49021", "R", "49001", "01001"]
            + ["49099"],
            "validation_year": [2025, 2025, 2025, 2025, 2024, 2025, 2024],
            "type": ["projection", "projection", "train", "train"]
            + ["projection", "projection", "projection"],
        }
    )


def test_the_members_of_a_region_are_its_projection_rows_of_that_year():
    # 49021 has its own observed row (train): it stays its own row; 49099 has
    # a row only in 2024; 49001 in 2024 is another year
    groups = region_member_groups(region_rows(), {"R": REGIONS["R"]})
    assert groups == [(3, [0, 1])]


def test_a_county_in_two_regions_raises():
    with pytest.raises(ValueError, match="member of two regions"):
        region_member_groups(
            region_rows().with_columns(
                fips=pl.Series(
                    ["49001", "49017", "49021", "R", "49001", "S", "49099"]
                )
            ),
            REGIONS,
        )


def test_a_region_without_member_rows_makes_no_group(capsys):
    rows = region_rows().filter(pl.col("fips") != "49001")
    rows = rows.filter(pl.col("fips") != "49017")
    assert region_member_groups(rows, {"R": REGIONS["R"]}) == []
    assert "has no member rows" in capsys.readouterr().out


def test_the_mask_marks_only_the_member_rows_to_drop():
    mask = region_member_mask(region_rows(), {"R": REGIONS["R"]})
    # the two projection members of R in 2025; not the region unit, the
    # member with its own observed row, the same county in 2024, a county
    # that is in no region or a county that has a row only in another year
    assert mask.tolist() == [True, True, False, False, False, False, False]


def test_no_region_in_the_rows_marks_nothing():
    rows = region_rows().filter(pl.col("fips") != "R")
    assert not region_member_mask(rows, {"R": REGIONS["R"]}).any()


def test_a_statewide_place_is_a_unit_of_the_counties_of_its_state():
    # Delaware reports at the state only: its 2026 train row has the state
    # FIPS "10" and its three counties have projection rows. The counties of
    # another state, and the counties of Delaware in 2025 (no statewide row
    # that year), are not its members
    rows = pl.DataFrame(
        {
            "fips": ["10", "10001", "10003", "10005", "49001", "10001"],
            "validation_year": [2026] * 5 + [2025],
            "type": ["train"] + ["projection"] * 5,
        }
    )
    assert region_member_groups(rows, {}) == [(0, [1, 2, 3])]
    assert region_member_mask(rows, {}).tolist() == [
        False,
        True,
        True,
        True,
        False,
        False,
    ]


def test_a_county_of_a_statewide_place_with_its_own_observed_row_stays():
    rows = pl.DataFrame(
        {
            "fips": ["10", "10001", "10003"],
            "validation_year": [2026] * 3,
            "type": ["train", "train", "projection"],
        }
    )
    assert region_member_mask(rows, {}).tolist() == [False, False, True]


def test_dropping_the_members_leaves_one_prediction_per_place():
    rows = region_rows()
    kept = rows.filter(
        pl.Series(~region_member_mask(rows, {"R": REGIONS["R"]}))
    )
    # the region unit and the member with its own row remain
    assert kept.filter(pl.col("validation_year") == 2025)[
        "fips"
    ].to_list() == [
        "49021",
        "R",
        "01001",
    ]


def test_a_region_unit_that_is_a_member_of_another_region_raises():
    rows = pl.DataFrame(
        {
            "fips": ["R", "S", "49017"],
            "validation_year": [2025] * 3,
            "type": ["train", "projection", "projection"],
        }
    )
    with pytest.raises(ValueError, match="is a region unit and a member"):
        region_member_groups(rows, {"R": ["S"], "S": ["49017"]})


def test_a_short_validation_file_name_is_the_full_one():
    name = validation_file_name([2019, 2020], True, False, "All covariates", 1)
    assert (
        name
        == "samples_partial_pooled_poisson_validation_testing_[2019, 2020]_pool=True_CAR=False_All covariates_globals_prior_sd=1.npz"
    )


def test_a_name_too_long_for_the_file_system_writes_the_years_as_a_range():
    from outbreak_size.validation import MAX_FILE_NAME_BYTES

    years = list(range(2011, 2025))
    model = "All covariates + prior cases since 2011" + variant_file_suffix(
        "poisson", "raw"
    )
    name = validation_file_name(years, False, False, model, 1)
    assert "[2011-2024]" in name and "2012" not in name
    assert (
        len((name + ".part").encode()) <= 255
        and len(name.encode()) <= MAX_FILE_NAME_BYTES
    )
    # the same fit with the default variant fits as it is: its name keeps every year
    default = validation_file_name(
        years, False, False, "All covariates + prior cases since 2011", 1
    )
    assert "2012" in default and len(default.encode()) <= MAX_FILE_NAME_BYTES


def test_years_that_are_not_consecutive_cannot_be_shortened():
    with pytest.raises(ValueError):
        validation_file_name(
            [2011, 2013] + list(range(2014, 2030)), False, False, "x" * 200, 1
        )


# --- the draws of one validation year, in the order of another county list ----


def test_the_draws_are_aligned_to_the_counties_with_zeros_for_the_missing():
    from outbreak_size.validation import align_samples_to_counties

    draws = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # rows of the fold: A, B and a region (a name, not a county)
    aligned, has_row = align_samples_to_counties(
        draws, ["A", "B", "Region"], ["B", "C", "A"]
    )
    assert aligned.tolist() == [[2.0, 0.0, 1.0], [5.0, 0.0, 4.0]]
    assert has_row.tolist() == [True, False, True]


def test_a_fips_that_is_the_fips_of_two_rows_is_refused():
    from outbreak_size.validation import align_samples_to_counties

    with pytest.raises(ValueError, match="two rows"):
        align_samples_to_counties(np.zeros((1, 2)), ["A", "A"], ["A"])
