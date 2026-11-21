import numpy as np
import polars as pl
import pytest

from outbreak_size.county_scores import CONDITIONS
from outbreak_size.model_selection import (
    LARGEST_GAP_COLUMN,
    MEAN_AUROC_COLUMN,
    MEAN_GAP_COLUMN,
    WEIGHTED_GAP_COLUMN,
    best_of_pool,
    consistency_table,
    rank_models,
)

YEARS = (2021, 2022, 2023, 2024, 2025)
# AUROC for 5+ cases of the cases including imports, by model and year
AUROC5 = {
    "Statistical: A": [0.9, 0.6, 0.5, 0.7, 0.75],
    "Statistical: B": [0.8, 0.9, 0.6, 0.6, 0.65],
    "Statistical: C": [0.7, 0.7, 0.9, 0.8, 0.85],
    "Mechanistic: M1": [0.5, 0.3, 0.2, 0.4, 0.45],
    "Mechanistic: M2": [0.6, 0.2, 0.3, 0.5, 0.35],
    "Mechanistic: M3": [0.4, 0.1, 0.1, 0.2, 0.25],
    "Vaccine coverage rank order": [0.1, 0.1, 0.1, 0.1, 0.1],
    "Population size rank order": [0.3, 0.3, 0.3, 0.3, 0.3],
}
# the places with 5+ cases (the events) of each year
EVENTS = {2021: 2, 2022: 4, 2023: 4, 2024: 8, 2025: 62}
POOLS = {
    "Statistical": ["Statistical: A", "Statistical: B", "Statistical: C"],
    "Mechanistic": ["Mechanistic: M1", "Mechanistic: M2", "Mechanistic: M3"],
}
PASS_FORWARD = {"Statistical": 1, "Mechanistic": 2}
COMPARATORS = {
    "Vaccine coverage ranking": "Vaccine coverage rank order",
    "Population size ranking": "Population size rank order",
}


def toy_table(auroc5=AUROC5, auprc5=None, events=EVENTS):
    """A table of scores with the columns the selection reads. The other
    scores are fixed functions of the AUROC for 5+ cases (the AUPRC for 5+
    cases a tenth of it, the AUROC for 10+ a point lower, the AUPRC for 10+ a
    twentieth), and the outbreak cases score half a point higher."""
    rows = []
    for model, values in auroc5.items():
        for year, au5 in zip(YEARS, values):
            ap5 = au5 / 10 if auprc5 is None else auprc5[model][year - 2021]
            for condition, shift in zip(CONDITIONS, (0.0, 0.5)):
                rows.append(
                    {
                        "Model": model,
                        "Evaluation year": str(year),
                        "Condition": condition,
                        "Places with 5+ cases": events[year],
                        "AUROC, 5+ (%)": au5 + shift,
                        "AUPRC, 5+ (%)": ap5 + shift,
                        "AUROC, 10+ (%)": au5 - 1 + shift,
                        "AUPRC, 10+ (%)": au5 / 20 + shift,
                    }
                )
    return pl.from_dicts(rows)


def test_ranks_share_a_rank_when_every_key_ties_and_break_ties_by_the_next():
    keys = {
        "a": (0.9, 0.1),
        "b": (0.9, 0.2),  # the same first key as a, a better second key
        "c": (0.8, 0.5),
        "d": (0.8, 0.5),  # the same as c on every key
        "e": (-np.inf, 0.9),  # a missing first score ranks last
    }
    assert rank_models(keys) == {"b": 1, "a": 2, "c": 3, "d": 3, "e": 5}


def test_consistency_of_every_model_with_the_best_of_its_pool():
    out = consistency_table(toy_table(), POOLS, PASS_FORWARD, COMPARATORS)
    stat = out.filter(pl.col("Pool") == "Statistical")
    # best AUROC of the pool by year: 0.9 (A), 0.9 (B), 0.9 (C), 0.8 (C), 0.85 (C)
    # gaps: A 0, 0.3, 0.4, 0.1, 0.1; B 0.1, 0, 0.3, 0.2, 0.2; C 0.2, 0.2, 0, 0, 0
    # the summary is over 2022-2025: the mean gaps are A 0.225, B 0.175 and
    # C 0.05; the event-weighted gaps (weights 4, 4, 8, 62) are A 9.8 / 78,
    # B 15.2 / 78 and C 0.8 / 78, and the models are sorted by those
    assert stat["Model"].to_list() == [
        "Statistical: C",
        "Statistical: A",
        "Statistical: B",
    ]
    a = stat.filter(pl.col("Model") == "Statistical: A")
    assert a[MEAN_GAP_COLUMN].item() == pytest.approx(0.225)
    assert a[LARGEST_GAP_COLUMN].item() == pytest.approx(0.4)
    assert a[MEAN_AUROC_COLUMN].item() == pytest.approx(0.6375)
    # the event-weighted gap, weights 4, 4, 8, 62 (2022-2025, sum 78):
    # A (0.3, 0.4, 0.1, 0.1): 9.8 / 78; B (0, 0.3, 0.2, 0.2): 15.2 / 78;
    # C (0.2, 0, 0, 0): 0.8 / 78
    assert a[WEIGHTED_GAP_COLUMN].item() == pytest.approx(9.8 / 78)
    b = stat.filter(pl.col("Model") == "Statistical: B")
    assert b[WEIGHTED_GAP_COLUMN].item() == pytest.approx(15.2 / 78)
    c0 = stat.filter(pl.col("Model") == "Statistical: C")
    assert c0[WEIGHTED_GAP_COLUMN].item() == pytest.approx(0.8 / 78)
    # 2021 is in the per-year columns but not in the summary
    assert a["Gap from the best of the pool, 2021"].item() == pytest.approx(0)
    assert [a[f"Selection rank, {y}"].item() for y in YEARS] == [1, 3, 3, 2, 2]
    # one place passed forward, over the 5 years: A is first only in 2021
    assert a["Years passed forward (of 5)"].item() == 1
    # the flags per year: A (ranks 1, 3, 3, 2, 2) was passed forward from 2021 only
    assert a["Passed forward from 2021"].item() is True
    assert a["Passed forward from 2022"].item() is False
    assert a["Passed forward from 2025 (not scored later)"].item() is False
    # the mechanistic pool passes two: M3 is always last, so M1 is in the top two every year
    m1_flags = out.filter(pl.col("Model") == "Mechanistic: M1")
    assert m1_flags["Passed forward from 2023"].item() is True
    # comparators have no flags
    assert (
        out.filter(pl.col("Model") == "Population size rank order")[
            "Passed forward from 2021"
        ].item()
        is None
    )
    c = stat.filter(pl.col("Model") == "Statistical: C")
    assert c["Years passed forward (of 5)"].item() == 3
    # the mechanistic pool passes two forward
    m1 = out.filter(pl.col("Model") == "Mechanistic: M1")
    assert m1["Years passed forward (of 5)"].item() == 5
    # the comparators have their AUROC only
    pop = out.filter(pl.col("Model") == "Population size rank order")
    assert pop[MEAN_AUROC_COLUMN].item() == pytest.approx(0.3)
    assert pop[MEAN_GAP_COLUMN].item() is None
    assert pop[WEIGHTED_GAP_COLUMN].item() is None


# --- the best model of a pool ---------------------------------------------------


def test_the_best_of_a_pool_depends_on_the_years_the_gap_is_taken_over():
    # X is the best in 2021 only; Y is the best in every later year. 2021 has
    # the most events here (100, against 1 in each later year)
    auroc = {
        "Statistical: X": [0.9, 0.5, 0.5, 0.5, 0.5],
        "Statistical: Y": [0.45, 0.6, 0.6, 0.6, 0.6],
    }
    events = {2021: 100, 2022: 1, 2023: 1, 2024: 1, 2025: 1}
    pools = {"Statistical": list(auroc)}
    s3 = consistency_table(
        toy_table(auroc, events=events), pools, {"Statistical": 1}, {}
    )
    # event-weighted gap over 2021-2025: X 0.4 / 104, Y 45 / 104; over
    # 2022-2025 (equal weights): X 0.1, Y 0
    assert best_of_pool(s3, "Statistical", YEARS) == "Statistical: X"
    assert best_of_pool(s3, "Statistical", YEARS[1:]) == "Statistical: Y"


def test_the_best_of_a_pool_weights_every_year_by_its_events():
    # the same models with the default events (2, 4, 4, 8, 62): 2025 carries
    # most of the weight, and Y is the best there, so Y is the best over all
    # years though its mean gap over 2021-2025 (0.09) is not the smaller
    auroc = {
        "Statistical: X": [0.9, 0.5, 0.5, 0.5, 0.5],
        "Statistical: Y": [0.45, 0.6, 0.6, 0.6, 0.6],
    }
    s3 = consistency_table(
        toy_table(auroc), {"Statistical": list(auroc)}, {"Statistical": 1}, {}
    )
    assert best_of_pool(s3, "Statistical", YEARS) == "Statistical: Y"
    # the events of each year are in the table, the same for every model
    assert s3["Places with 5+ cases, 2025"].to_list() == [62, 62]


def test_two_models_with_the_same_mean_gap_are_an_error():
    auroc = {"Statistical: X": [0.9] * 5, "Statistical: Y": [0.9] * 5}
    s3 = consistency_table(
        toy_table(auroc), {"Statistical": list(auroc)}, {"Statistical": 1}, {}
    )
    with pytest.raises(ValueError, match="tie"):
        best_of_pool(s3, "Statistical", YEARS)


def test_the_models_passed_forward_include_every_tie_at_the_last_place():
    # c and d tie on every score; with 3 to pass forward, d comes along (4 passed forward)
    auroc = {
        "Statistical: A": [0.9] * 5,
        "Statistical: B": [0.8] * 5,
        "Statistical: C": [0.7] * 5,
        "Statistical: D": [0.7] * 5,
        "Statistical: E": [0.6] * 5,
    }
    pools = {"Statistical": list(auroc)}
    table = toy_table(auroc)
    for n, expected in ((3, 4), (2, 2), (4, 4), (5, 5)):
        out = consistency_table(table, pools, {"Statistical": n}, {})
        assert int(out["Passed forward from 2021"].sum()) == expected, n


def test_the_event_weighted_gap_leaves_out_a_year_without_events_or_a_score():
    # no place with 5+ cases in 2022: that year has no weight in the weighted
    # gap, though it stays in the mean gap
    events = {2021: 2, 2022: 0, 2023: 4, 2024: 8, 2025: 62}
    out = consistency_table(
        toy_table(events=events), POOLS, PASS_FORWARD, COMPARATORS
    )
    a = out.filter(pl.col("Model") == "Statistical: A")
    # A's gaps 2023-2025: 0.4, 0.1, 0.1 with weights 4, 8, 62 (sum 74)
    assert a[WEIGHTED_GAP_COLUMN].item() == pytest.approx(
        (0.4 * 4 + 0.1 * 8 + 0.1 * 62) / 74
    )
    assert a[MEAN_GAP_COLUMN].item() == pytest.approx(0.225)
    # a model with no AUROC in 2025 (NaN) is weighted over the other years
    auroc = {m: list(v) for m, v in AUROC5.items()}
    auroc["Statistical: A"][4] = float("nan")
    out = consistency_table(toy_table(auroc), POOLS, PASS_FORWARD, COMPARATORS)
    a = out.filter(pl.col("Model") == "Statistical: A")
    assert a[WEIGHTED_GAP_COLUMN].item() == pytest.approx(
        (0.3 * 4 + 0.4 * 4 + 0.1 * 8) / 16
    )
