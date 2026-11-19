"""The designs of the statistical models, their names, the model that scripts 07 and 11 build
from a design, and the choice of the design from S2 (outbreak_size.model_designs)."""

import json
from typing import Any

import numpy as np
import polars as pl
import pytest

from outbreak_size.county_scores import CONDITIONS
from outbreak_size.model_designs import (
    DESIGNS_BY_NAME,
    ModelDesign,
    design_model,
    read_record,
    record_file_name,
    select_best_statistical_model,
    start_year_of_model_name,
    statistical_model_name,
    trained_since,
    training_years,
    write_record,
)
from outbreak_size.model_inputs import (
    IV_COLUMNS,
    REGRESSION_COLUMNS,
    pooled_model,
)
from outbreak_size.model_selection import (
    LARGEST_GAP_COLUMN,
    MEAN_AUROC_COLUMN,
    MEAN_GAP_COLUMN,
    SUMMARY_LABEL,
)

N = 4


def rows():
    return pl.DataFrame(
        {
            "pop": [100, 200, 300, 400],
            "state": ["A", "A", "B", "B"],
            "geoid": [f"g{i}" for i in range(N)],
            "fips": [f"0100{i}" for i in range(N)],
        }
    )


def chain():
    a = np.zeros((N, N))
    for i in range(N - 1):
        a[i, i + 1] = a[i + 1, i] = 1
    return a


def scaled():
    columns = IV_COLUMNS + REGRESSION_COLUMNS
    return pl.DataFrame(
        np.random.default_rng(0).normal(size=(N, len(columns))), schema=columns
    )


# --- the names and windows -----------------------------------------------------


def test_there_are_36_designs_with_the_names_of_the_scoring_tables():
    assert len(DESIGNS_BY_NAME) == 36
    # the names are those of S1 and S2
    assert (
        "Statistical: All covariates + spatial + state-specific vaccine "
        "coefficients, trained since 2019" in DESIGNS_BY_NAME
    )
    assert (
        "Statistical: MMR coverage, trained on last two years"
        in DESIGNS_BY_NAME
    )
    assert (
        statistical_model_name("MMR coverage + exemptions", True, False, 2011)
        == "Statistical: MMR coverage + exemptions + spatial, trained since 2011"
    )
    design = DESIGNS_BY_NAME[
        "Statistical: MMR coverage + exemptions + spatial, trained since 2011"
    ]
    assert design == ModelDesign(
        "MMR coverage + exemptions", True, False, 2011
    )
    assert design.has_exemptions
    assert not DESIGNS_BY_NAME[
        "Statistical: MMR coverage, trained since 2011"
    ].has_exemptions


def test_the_training_windows():
    assert training_years(2011, 2026) == list(range(2011, 2026))
    assert training_years(2019, 2026) == list(range(2019, 2026))
    assert training_years(-2, 2026) == [
        2024,
        2025,
    ]  # as in the rolling validation
    assert training_years(-2, 2021) == [2019, 2020]


# --- the model built from a design ---------------------------------------------


def test_the_all_covariates_design_pools_three_terms_and_regresses_nine():
    # the model 07 fitted before its design was chosen from S2: the pooled model
    # on the IV and REGRESSION columns, pooled by state, with a spatial term
    design = DESIGNS_BY_NAME[
        "Statistical: All covariates + spatial + state-specific vaccine "
        "coefficients, trained since 2019"
    ]
    kwargs: dict = dict(prior_sd=1.0, spatial_sigma_scale=0.5)
    a: Any = design_model(design, rows(), scaled(), chain(), **kwargs)
    b: Any = pooled_model(
        rows(),
        scaled().select(IV_COLUMNS),
        scaled().select(REGRESSION_COLUMNS).to_jax(),
        pool=True,
        spatial=True,
        adjacency_matrix=chain(),
        **kwargs,
    )
    assert a.iv.shape == (N, 3) and a.num_regression_covariates == 9
    assert np.asarray(a.iv) == pytest.approx(np.asarray(b.iv))
    assert np.asarray(a.regression_covariates) == pytest.approx(
        np.asarray(b.regression_covariates)
    )
    assert np.asarray(a.regression_covariates) == pytest.approx(
        scaled().select(REGRESSION_COLUMNS).to_numpy()
    )
    assert (a.pool, a.spatial, a.group_fixed_effect) == (True, True, False)
    for prior in (
        "global_priors",
        "betas",
        "baseline",
        "car_sigma",
        "spatial_sigma",
    ):
        assert float(getattr(a, prior).scale) == float(getattr(b, prior).scale)
    assert float(a.global_priors.scale) == 1.0
    assert float(a.car_sigma.scale) == float(a.spatial_sigma.scale) == 0.5


@pytest.mark.parametrize(
    "name, n_iv, n_regression, spatial, pooled",
    [
        ("Statistical: MMR coverage, trained since 2011", 1, 1, False, False),
        (
            "Statistical: MMR coverage + exemptions + spatial, trained since 2019",
            3,
            1,
            True,
            False,
        ),
        (
            "Statistical: MMR coverage + state-specific vaccine coefficients, "
            "trained on last two years",
            1,
            1,
            False,
            True,
        ),
        (
            "Statistical: All covariates, trained since 2011",
            3,
            9,
            False,
            False,
        ),
    ],
)
def test_the_model_has_the_covariates_and_terms_of_its_design(
    name, n_iv, n_regression, spatial, pooled
):
    model: Any = design_model(DESIGNS_BY_NAME[name], rows(), scaled(), chain())
    assert model.iv.shape == (N, n_iv)
    # fewer covariates than All covariates: a placeholder column (a model with
    # no ordinary regression covariates)
    assert model.regression_covariates.shape == (N, n_regression)
    assert model.spatial == spatial
    assert model.pool == pooled
    assert np.asarray(model.iv) == pytest.approx(
        scaled().select(IV_COLUMNS[:n_iv]).to_numpy()
    )


# --- the choice of the design from S2 ------------------------------------------

CASES = CONDITIONS[0]
A = "Statistical: All covariates + spatial + state-specific vaccine coefficients, trained since 2019"
B = "Statistical: MMR coverage + exemptions, trained since 2011"
C = "Statistical: MMR coverage, trained on last two years"
M = "Mechanistic: MMR coverage, trained since 2019"


def s2(gaps):
    """An S2 with the given {model: mean gap}, in the order of the dict, and a
    mechanistic model and a comparator with a gap smaller than any."""
    rows_ = [
        dict(
            Pool="Statistical",
            Model=m,
            **{
                MEAN_AUROC_COLUMN: 90.0 - g,
                MEAN_GAP_COLUMN: g,
                LARGEST_GAP_COLUMN: 2 * g,
            },
        )
        for m, g in gaps.items()
    ]
    rows_.append(
        dict(
            Pool="Mechanistic",
            Model=M,
            **{
                MEAN_AUROC_COLUMN: 91.0,
                MEAN_GAP_COLUMN: 0.0,
                LARGEST_GAP_COLUMN: 0.0,
            },
        )
    )
    rows_.append(
        dict(
            Pool="",
            Model="Population size rank order",
            **{MEAN_AUROC_COLUMN: 99.0},
        )
    )
    return pl.from_dicts(rows_, infer_schema_length=None)


def s1(scores):
    """An S1 with {model: (AUPRC 5+, AUROC 10+, AUPRC 10+)} of the cases including
    imports, 2022-2025 (and a different condition that must be ignored)."""
    out = []
    for m, (a, b, c) in scores.items():
        out.append(
            {
                "Model": m,
                "Evaluation year": SUMMARY_LABEL,
                "Condition": CASES,
                "AUPRC, 5+ (%)": a,
                "AUROC, 10+ (%)": b,
                "AUPRC, 10+ (%)": c,
            }
        )
        out.append(
            {
                "Model": m,
                "Evaluation year": SUMMARY_LABEL,
                "Condition": CONDITIONS[1],
                "AUPRC, 5+ (%)": 99.0,
                "AUROC, 10+ (%)": 99.0,
                "AUPRC, 10+ (%)": 99.0,
            }
        )
    return pl.from_dicts(out)


SCORES = {A: (5.0, 90.0, 5.0), B: (7.0, 90.0, 5.0), C: (5.0, 90.0, 5.0)}


def test_the_smallest_mean_gap_of_the_statistical_pool_is_selected():
    # not the first row, not the mechanistic model (gap 0), not the comparator
    design, candidates = select_best_statistical_model(
        s2({A: 1.3, B: 1.2, C: 1.4}), s1(SCORES)
    )
    assert design.name == B
    assert candidates["Model"].to_list() == [B, A, C]
    assert candidates[MEAN_GAP_COLUMN].to_list() == [
        1.2,
        1.3,
        1.4,
    ]


def test_a_tie_on_the_gap_goes_to_the_later_scores_in_order():
    # A and B tie on the gap: B has the higher AUPRC 5+ (the first tie-break)
    design, _ = select_best_statistical_model(
        s2({A: 1.2, B: 1.2, C: 1.5}), s1(SCORES)
    )
    assert design.name == B
    # the same AUPRC 5+: the AUROC 10+ decides, then the AUPRC 10+
    tied = {A: (5.0, 90.0, 5.0), B: (5.0, 91.0, 1.0), C: (5.0, 90.0, 5.0)}
    assert (
        select_best_statistical_model(s2({A: 1.2, B: 1.2, C: 1.5}), s1(tied))[
            0
        ].name
        == B
    )
    tied = {A: (5.0, 90.0, 6.0), B: (5.0, 90.0, 1.0), C: (5.0, 90.0, 5.0)}
    assert (
        select_best_statistical_model(s2({A: 1.2, B: 1.2, C: 1.5}), s1(tied))[
            0
        ].name
        == A
    )


def test_the_record_gives_back_the_design_and_the_years(tmp_path):
    design = DESIGNS_BY_NAME[A]
    choice = dict(rationale="the reason", chosen_on="2026-11-10")
    path = str(tmp_path / record_file_name(1))
    assert (
        record_file_name(1) == "risk_projection_model_globals_prior_sd=1.json"
    )
    write_record(path, design, [2019, 2020], choice)
    assert read_record(path) == (design, [2019, 2020])
    # the reason and the date of the choice are kept with the design
    with open(path) as f:
        assert json.load(f)["choice"] == choice


@pytest.mark.parametrize("start_year", [2011, 2019, -2])
def test_the_name_of_a_model_says_the_window_it_was_trained_on(start_year):
    for name in (
        "Mechanistic: MMR coverage" + trained_since(start_year),
        "Mechanistic with multi-chain outbreaks: MMR coverage"
        + trained_since(start_year),
    ):
        assert start_year_of_model_name(name) == start_year
    # the window the discordance figure's mechanistic fit of 2026 uses
    assert training_years(-2, 2026) == [2024, 2025]
    assert training_years(2019, 2026) == list(range(2019, 2026))


def test_a_name_without_a_training_window_is_an_error():
    with pytest.raises(ValueError, match="no training window"):
        start_year_of_model_name("Statistical: All covariates")
