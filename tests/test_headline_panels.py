"""The numbers behind the validation and 2026 panels of the main figure
(outbreak_size.headline_panels)."""

import numpy as np
import polars as pl
import pytest

from outbreak_size.county_scores import CONDITIONS
from outbreak_size.headline_panels import (
    PERFORMANCE_METRICS,
    case_bucket,
    conditioned_national_totals,
    performance_values,
)

MODELS = {"Statistical": "stat model", "Mechanistic": "mech model"}


def score_table(window, offset, with_intervals, extra_columns=None):
    """A table in the form of S1: a row per model and condition, a value
    for every metric that says which row, metric and window it is."""
    rows = []
    for m, name in enumerate(["stat model", "mech model", "other model"]):
        for c, condition in enumerate(CONDITIONS):
            row = {
                "Model": name,
                "Condition": condition,
                "Evaluation year": window,
            }
            for k, template in enumerate(PERFORMANCE_METRICS.values()):
                for level in (5, 10):
                    column = template.format(level=level)
                    # 10+ columns carry a code of the model, condition, metric
                    value = offset + 100 * m + 10 * c + k + level / 100
                    row[column] = value
                    if with_intervals:
                        row[f"{column}, lower"] = value - 1
                        row[f"{column}, upper"] = value + 1
            rows.append({**row, **(extra_columns or {})})
    return pl.DataFrame(rows)


def test_each_value_is_read_from_its_own_row_and_column():
    s1 = pl.concat(
        [
            score_table("2022-2025", 0, True),
            score_table("2021-2025", 5000, True),  # another window: not used
        ]
    )
    later = pl.concat(
        [
            score_table(2026, 1000, False, {"places": "all places"}),
            score_table(2026, 9000, False, {"places": "without Hudspeth"}),
        ]
    )
    values = performance_values(s1, later, MODELS, 10, "2022-2025", 2026)
    # 6 metrics, 2 models, 2 conditions, 2 periods
    assert values.height == 6 * 2 * 2 * 2
    assert set(values["period"]) == {"2022-2025", "2026"}
    assert set(values["condition"]) == {"All cases", "Outbreak cases"}
    for row in values.iter_rows(named=True):
        m = 0 if row["model"] == "Statistical" else 1
        c = 0 if row["condition"] == "All cases" else 1
        k = list(PERFORMANCE_METRICS).index(row["metric"])
        offset = 0 if row["period"] == "2022-2025" else 1000
        expected = offset + 100 * m + 10 * c + k + 0.10  # the 10+ column
        assert row["value"] == pytest.approx(expected)
        if row["period"] == "2022-2025":
            assert row["lower"] == pytest.approx(expected - 1)
            assert row["upper"] == pytest.approx(expected + 1)
        else:  # the table of the later year has no intervals
            assert np.isnan(row["lower"]) and np.isnan(row["upper"])


def test_the_later_table_may_have_intervals_and_a_text_year():
    s1 = score_table("2022-2025", 0, True)
    later = score_table("2026", 1000, True, {"places": "all places"})
    values = performance_values(s1, later, MODELS, 5, "2022-2025", 2026)
    later_rows = values.filter(pl.col("period") == "2026")
    assert later_rows["lower"].is_not_nan().all()
    assert (later_rows["upper"] - later_rows["lower"] == 2).all()


def test_a_missing_duplicate_or_empty_value_stops_the_figure():
    s1 = score_table("2022-2025", 0, True)
    later = score_table(2026, 1000, False, {"places": "all places"})
    with pytest.raises(ValueError, match="0 rows"):
        performance_values(
            s1, later, {"Statistical": "no such model"}, 10, "2022-2025", 2026
        )
    with pytest.raises(ValueError, match="2 rows"):
        performance_values(
            pl.concat([s1, s1]), later, MODELS, 10, "2022-2025", 2026
        )
    # no row of that year in the later table
    with pytest.raises(ValueError, match="0 rows"):
        performance_values(s1, later, MODELS, 10, "2022-2025", 2027)
    broken = s1.with_columns(
        pl.when(pl.col("Model") == "stat model")
        .then(None)
        .otherwise(pl.col("AUROC, 10+ (%)"))
        .alias("AUROC, 10+ (%)")
    )
    with pytest.raises(ValueError, match="no value"):
        performance_values(broken, later, MODELS, 10, "2022-2025", 2026)


def test_no_imports_means_no_cases_and_the_totals_follow_the_scaler():
    rng = np.random.default_rng(0)
    size = np.full((50, 4), 10.0)
    assert (
        conditioned_national_totals(size, np.full((50, 4), 2.0), 0.0, rng) == 0
    ).all()
    assert (
        conditioned_national_totals(size, np.zeros((50, 4)), 5.0, rng) == 0
    ).all()
    # hand-computed: a place with Poisson(1) imports N and an outbreak of 10 has
    # E[(N + 10) * (N >= 1)] = 1 + 10 * (1 - exp(-1)) = 7.321 cases; 4 places
    # whose expected imports of 2 are conditioned by 0.5 have 29.28 in all
    n = 200_000
    totals = conditioned_national_totals(
        np.full((n, 4), 10.0), np.full((n, 4), 2.0), 0.5, rng
    )
    assert totals.shape == (n,)
    assert totals.mean() == pytest.approx(
        4 * (1 + 10 * (1 - np.exp(-1))), rel=5e-3
    )
    # the same generator state gives the same draws
    a = conditioned_national_totals(
        size, np.full((50, 4), 2.0), 1.0, np.random.default_rng(3)
    )
    b = conditioned_national_totals(
        size, np.full((50, 4), 2.0), 1.0, np.random.default_rng(3)
    )
    assert (a == b).all()


def test_the_buckets_of_the_observed_cases():
    assert case_bucket([0, 1, 9, 10, 200]).tolist() == [0, 1, 1, 2, 2]
