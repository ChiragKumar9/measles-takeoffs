"""What the model is given: stacked rows, scaled covariates, the model
(outbreak_size.model_inputs)."""

import math
from typing import Any

import numpy as np
import polars as pl
import pytest

from outbreak_size.model_inputs import (
    IV_COLUMNS,
    REGRESSION_COLUMNS,
    apply_covariate_scale,
    log_attack_rate,
    no_regression_covariates,
    pooled_model,
    scale_covariates,
    stack_rows,
)

# --- stacking the rows --------------------------------------------------------


def train_rows():
    return pl.DataFrame(
        {"fips": ["a", "b"], "cases_outbreak": [3, 0], "is_train": [1, 1]}
    )


def projection_rows():
    # no observed cases for the rows to predict (and an extra column)
    return pl.DataFrame(
        {"fips": ["c", "d"], "is_train": [0, 0], "extra": [1, 2]}
    )


def test_rows_to_predict_go_below_the_training_rows_with_zero_cases():
    out = stack_rows(train_rows(), projection_rows())
    assert out["fips"].to_list() == ["a", "b", "c", "d"]
    assert out["is_train"].to_list() == [1, 1, 0, 0]
    assert out["cases_outbreak"].to_list() == [3, 0, 0, 0]
    # a column only the projection rows have is null for the others
    assert out["extra"].to_list() == [None, None, 1, 2]


def test_the_cases_can_be_made_integer():
    train = train_rows().with_columns(
        cases_outbreak=pl.col("cases_outbreak").cast(pl.Float64)
    )
    out = stack_rows(train, projection_rows(), cases_as_int=True)
    assert out.schema["cases_outbreak"] == pl.Int64
    assert out["cases_outbreak"].to_list() == [3, 0, 0, 0]
    assert (
        stack_rows(train, projection_rows()).schema["cases_outbreak"]
        == pl.Float64
    )


# --- scaling -------------------------------------------------------------------


def covariate_rows():
    """Three training rows (1, 2, 3) and two rows to predict (5, 11), and a
    second covariate that is constant (7) in the training rows."""
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 5.0, 11.0],
            "z": [7.0, 7.0, 7.0, 8.0, 9.0],
            "is_train": [1, 1, 1, 0, 0],
        }
    )


def test_training_rows_are_standardized_and_the_others_use_their_statistics():
    """The training values 1, 2, 3 have mean 2 and standard deviation
    sqrt(2 / 3); 5 is (5 - 2) / sqrt(2 / 3) = 3.674 of those above the mean."""
    out = scale_covariates(covariate_rows(), ["x"], pl.col("is_train") == 1)
    sd = (2 / 3) ** 0.5
    assert out["x"].to_list() == pytest.approx(
        [(v - 2) / sd for v in (1, 2, 3, 5, 11)]
    )
    train = out["x"].to_numpy()[:3]
    assert train.mean() == pytest.approx(0.0, abs=1e-12)
    assert train.std() == pytest.approx(1.0)


def test_scaling_keeps_the_row_order_and_the_requested_columns():
    rows = covariate_rows().reverse()  # the training rows are now last
    out = scale_covariates(rows, ["z", "x"], pl.col("is_train") == 1)
    assert out.columns == ["z", "x"]
    assert out.height == 5
    sd = (2 / 3) ** 0.5
    assert out["x"].to_list() == pytest.approx(
        [(v - 2) / sd for v in (11, 5, 3, 2, 1)]
    )


def test_a_covariate_constant_in_the_training_rows_is_not_divided_by_zero():
    """z is 7 in every training row, so its standard deviation is 0 and it is
    only centered (sklearn leaves a zero spread as 1)."""
    out = scale_covariates(covariate_rows(), ["z"], pl.col("is_train") == 1)
    assert out["z"].to_list() == pytest.approx([0.0, 0.0, 0.0, 1.0, 2.0])


# --- the model -----------------------------------------------------------------

N_COUNTIES = 4


def model_rows():
    return pl.DataFrame(
        {
            "pop": [100, 200, 300, 400],
            "state": ["A", "A", "B", "B"],
            "geoid": [f"g{i}" for i in range(N_COUNTIES)],
            "fips": [f"0100{i}" for i in range(N_COUNTIES)],
        }
    )


def chain_adjacency():
    adjacency = np.zeros((N_COUNTIES, N_COUNTIES))
    for i in range(N_COUNTIES - 1):
        adjacency[i, i + 1] = adjacency[i + 1, i] = 1
    return adjacency


def scaled_covariates():
    rng = np.random.default_rng(0)
    columns = IV_COLUMNS + REGRESSION_COLUMNS
    return pl.DataFrame(
        rng.normal(size=(N_COUNTIES, len(columns))), schema=columns
    )


def test_the_model_gets_the_rows_exposure_groups_and_covariates():
    scaled = scaled_covariates()
    model: Any = pooled_model(
        model_rows(),
        scaled.select(IV_COLUMNS[:1]),
        no_regression_covariates(N_COUNTIES),
        pool=False,
        spatial=True,
        adjacency_matrix=chain_adjacency(),
    )
    assert model.iv.shape == (N_COUNTIES, 1)
    assert model.num_ivs == 1
    assert model.exposure.tolist() == [100, 200, 300, 400]
    assert model.num_groups == 2  # the states, in the order they appear
    assert model.groups.tolist() == [0, 0, 1, 1]
    assert model.num_regression_covariates == 1  # the placeholder column
    assert (model.pool, model.spatial, model.group_fixed_effect) == (
        False,
        True,
        False,
    )


def test_a_given_spatial_sigma_scale_applies_to_both_spatial_scales():
    scaled = scaled_covariates()
    args = (
        model_rows(),
        scaled.select(IV_COLUMNS),
        no_regression_covariates(N_COUNTIES),
    )
    kwargs: dict = dict(
        pool=True, spatial=True, adjacency_matrix=chain_adjacency()
    )
    model: Any = pooled_model(*args, spatial_sigma_scale=1.0, **kwargs)
    assert float(model.car_sigma.scale) == 1.0  # the CAR field
    assert float(model.spatial_sigma.scale) == 1.0  # the independent effects
    default: Any = pooled_model(*args, **kwargs)
    assert float(default.car_sigma.scale) == 0.25  # the model's own default
    assert float(default.spatial_sigma.scale) == 0.25


def test_a_given_prior_sd_applies_to_the_coefficients_and_baseline():
    scaled = scaled_covariates()
    args = (
        model_rows(),
        scaled.select(IV_COLUMNS),
        no_regression_covariates(N_COUNTIES),
    )
    kwargs: dict = dict(
        pool=True, spatial=False, adjacency_matrix=chain_adjacency()
    )
    model: Any = pooled_model(*args, prior_sd=3.0, **kwargs)
    assert float(model.global_priors.scale) == 3.0
    assert float(model.betas.scale) == 3.0  # the other regression coefficients
    assert float(model.baseline.scale) == 3.0
    default: Any = pooled_model(*args, **kwargs)
    assert (
        float(default.global_priors.scale) == 10.0
    )  # the model's own default
    assert float(default.betas.scale) == 1.0
    assert float(default.baseline.scale) == 10.0


def test_the_placeholder_for_no_regression_covariates_is_a_zero_column():
    placeholder = no_regression_covariates(5)
    assert placeholder.shape == (5, 1)
    assert not np.asarray(placeholder).any()


def test_a_spatial_model_raises_when_the_fips_and_geoid_order_differ():
    rows = model_rows().with_columns(
        fips=pl.Series(["01000", "01000", "01001", "01003"])
    )
    with pytest.raises(ValueError, match="pooled model: the spatial units"):
        pooled_model(
            rows,
            scaled_covariates().select(IV_COLUMNS[:1]),
            no_regression_covariates(N_COUNTIES),
            pool=False,
            spatial=True,
            adjacency_matrix=chain_adjacency(),
        )


def test_a_model_without_a_spatial_term_does_not_check_the_order():
    rows = model_rows().with_columns(
        fips=pl.Series(["01000", "01000", "01001", "01003"])
    )
    pooled_model(
        rows,
        scaled_covariates().select(IV_COLUMNS[:1]),
        no_regression_covariates(N_COUNTIES),
        pool=False,
        spatial=False,
        adjacency_matrix=chain_adjacency(),
    )


def test_the_baseline_prior_is_centred_on_the_given_mean():
    scaled = scaled_covariates()
    args = (
        model_rows(),
        scaled.select(IV_COLUMNS),
        no_regression_covariates(N_COUNTIES),
    )
    kwargs: dict = dict(
        pool=True, spatial=False, adjacency_matrix=chain_adjacency()
    )
    model: Any = pooled_model(
        *args, prior_sd=3.0, baseline_mean=-12.0, **kwargs
    )
    assert float(model.baseline.loc) == -12.0
    assert float(model.baseline.scale) == 3.0
    assert float(model.global_priors.loc) == 0.0  # only the baseline moves
    default: Any = pooled_model(*args, prior_sd=3.0, **kwargs)
    assert float(default.baseline.loc) == 0.0


# --- the attack rate that centres the baseline prior ---------------------------


def test_the_attack_rate_counts_every_county_year_as_its_own_row():
    # county a in two years (10 and 0 local cases), county b in one year
    rows = pl.DataFrame(
        {
            "cases_outbreak": [10, 0, 30],
            "pop": [100_000, 100_000, 300_000],
        }
    )
    # 40 cases over 100k + 100k + 300k people: a's population counts twice
    assert log_attack_rate(rows) == pytest.approx(np.log(40 / 500_000))


def test_the_attack_rate_needs_cases():
    with pytest.raises(ValueError, match="no outbreak cases"):
        log_attack_rate(
            pl.DataFrame({"cases_outbreak": [0, 0], "pop": [10, 20]})
        )
    with pytest.raises(ValueError, match="no outbreak cases"):
        log_attack_rate(
            pl.DataFrame(
                {"cases_outbreak": [], "pop": []},
                schema={"cases_outbreak": pl.Int64, "pop": pl.Int64},
            )
        )


# --- the covariate scale ------------------------------------------------------


def vaccination_rows():
    return pl.DataFrame(
        {
            "estimated_mmr_cov": [95.0, 100.0, 0.0],
            "nonmed_exempt_pct": [3.0, 0.0, 1.0],
            "med_exempt_pct": [0.0, 1.0, 0.0],
            "estimated_mmr_cov_5yr": [90.0, 90.0, 90.0],
            "fraction_white": [0.5, 0.6, 0.7],
        }
    )


def test_the_raw_scale_leaves_the_rows_as_they_are():
    rows = vaccination_rows()
    assert apply_covariate_scale(rows, "raw").equals(rows)


def test_the_natural_scale_puts_coverage_on_the_log_susceptible_fraction():
    # with 97% effectiveness: 95% coverage leaves 1 - 0.97 * 0.95 = 0.0785
    # susceptible, 100% leaves 0.03, and 0% coverage leaves everyone (log 1 = 0)
    out = apply_covariate_scale(vaccination_rows(), "natural", 0.97)
    assert out["estimated_mmr_cov"].to_list() == pytest.approx(
        [math.log(0.0785), math.log(0.03), 0.0]
    )
    # the five-year average is changed the same way: 1 - 0.97 * 0.9 = 0.127
    assert out["estimated_mmr_cov_5yr"].to_list() == pytest.approx(
        [math.log(0.127)] * 3
    )


def test_the_natural_scale_puts_exemptions_on_log_one_plus_the_percentage():
    out = apply_covariate_scale(vaccination_rows(), "natural", 0.97)
    assert out["nonmed_exempt_pct"].to_list() == pytest.approx(
        [math.log(4.0), 0.0, math.log(2.0)]
    )
    assert out["med_exempt_pct"].to_list() == pytest.approx(
        [0.0, math.log(2.0), 0.0]
    )


def test_the_natural_scale_leaves_other_columns_and_the_row_order_alone():
    rows = vaccination_rows()
    out = apply_covariate_scale(rows, "natural", 0.97)
    assert out["fraction_white"].to_list() == rows["fraction_white"].to_list()
    assert out.columns == rows.columns


def test_a_proportional_change_in_the_susceptibles_is_the_same_step_anywhere():
    # a doubling of the susceptible fraction is the same step on the natural
    # scale wherever it happens: here from 0.03 (100% coverage) to 0.06
    rows = pl.DataFrame(
        {"estimated_mmr_cov": [100.0, 100.0 - 0.03 / 0.97 * 100]}
    )
    out = apply_covariate_scale(rows, "natural", 0.97)["estimated_mmr_cov"]
    # susceptible fractions 0.03 and 0.06
    assert out[1] - out[0] == pytest.approx(math.log(2.0))


def test_a_coverage_above_100_is_refused_not_clipped():
    rows = pl.DataFrame({"estimated_mmr_cov": [99.0, 100.5]})
    with pytest.raises(ValueError, match="above 100%"):
        apply_covariate_scale(rows, "natural", 0.97)


def test_the_natural_scale_needs_the_effectiveness():
    with pytest.raises(ValueError, match="needs mmr_effectiveness"):
        apply_covariate_scale(vaccination_rows(), "natural")


def test_an_unknown_covariate_scale_is_refused():
    with pytest.raises(ValueError, match="covariate_scale must be one of"):
        apply_covariate_scale(vaccination_rows(), "log")
