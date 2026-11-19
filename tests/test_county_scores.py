import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from outbreak_size import scoring
from outbreak_size.county_scores import (
    CONDITIONS,
    ScoreBook,
    count_columns,
    distribution_place_values,
    distribution_scores,
    fold_error,
    forecast_metrics,
    outcome_scores,
    place_values,
    point_spearman,
    score_condition,
    score_places,
    top_share_column,
    top_share_metrics,
)

# three places with 0, 3 and 10 observed cases, and two draws of each: the
# first draw is exact, the second is off by +1, +1 and -2. The forecast of a
# place is the mean of its draws: 0.5, 3.5 and 9.0 cases
TRUTH = np.array([0, 3, 10])
DRAWS = np.array([[0, 3, 10], [1, 4, 8]])


def test_metrics_of_the_forecast_for_all_places():
    m = forecast_metrics(DRAWS.mean(axis=0), TRUTH)
    # errors of the forecast (0.5, 3.5, 9.0) are +0.5, +0.5 and -1.0 cases:
    # RMSE = sqrt((0.25 + 0.25 + 1) / 3), in cases
    assert m["RMSE, 0+ (cases)"] == pytest.approx(np.sqrt(0.5))
    # the median of the absolute errors 0.5, 0.5 and 1.0, in cases
    assert m["MAE, 0+ (cases)"] == pytest.approx(0.5)
    # squared Pearson correlation in percent, by hand: the sums of products and
    # squares of the deviations are 44.1667, 37.1667 and 52.6667, so
    # r^2 = 44.1667^2 / (37.1667 * 52.6667) = 0.99655
    assert m["R², 0+ (%)"] == pytest.approx(99.6552, abs=1e-3)
    assert "Median fold error, 0+" not in m
    assert "Spearman rho, 0+" not in " ".join(m)


def test_metrics_of_the_forecast_for_places_with_cases():
    m = forecast_metrics(DRAWS.mean(axis=0), TRUTH)
    # places with 3+ cases are the second and third: errors +0.5 and -1.0
    assert m["RMSE, 3+ (cases)"] == pytest.approx(np.sqrt(0.625))
    assert m["MAE, 3+ (cases)"] == pytest.approx(0.75)
    # fold errors on cases + 1: 4.5 / 4 = 1.125 and 11 / 10 = 1.1; the median
    # of two values is their mean
    assert m["Median fold error, 3+"] == pytest.approx((1.125 + 1.1) / 2)
    # two places that keep their order: a squared correlation of exactly 100%
    assert m["R², 3+ (%)"] == pytest.approx(100)
    # one place with 10+ cases: errors and fold error, but no correlation
    assert m["RMSE, 10+ (cases)"] == pytest.approx(1)
    assert m["MAE, 10+ (cases)"] == pytest.approx(1)
    assert m["Median fold error, 10+"] == pytest.approx(1.1)
    assert np.isnan(m["R², 10+ (%)"])


def test_level_without_places_is_nan():
    m = forecast_metrics(np.array([0.2, 0.5, 1.0]), np.array([0, 1, 2]))
    assert np.isnan(m["RMSE, 10+ (cases)"])
    assert np.isnan(m["Median fold error, 10+"])


def test_fold_error_is_one_when_predicted_and_observed_are_zero():
    assert fold_error(0.0, 0.0) == 1.0
    # predicted 0, observed 3: (3 + 1) / (0 + 1)
    assert fold_error(0.0, 3.0) == 4.0
    # symmetric: observed 0, predicted 3
    assert fold_error(3.0, 0.0) == 4.0


def test_forecast_without_variation_has_no_correlation():
    m = forecast_metrics(np.array([0.0, 0.0, 0.0]), TRUTH)
    assert np.isnan(m["R², 0+ (%)"])
    # the errors do not need any variation
    assert m["RMSE, 0+ (cases)"] == pytest.approx(np.sqrt((9 + 100) / 3))


def test_interval_coverage_includes_both_ends():
    # 41 draws of 0..40: the 2.5th and 97.5th percentiles are exactly 1 and 39
    draws = np.tile(np.arange(41.0)[:, None], (1, 4))
    truth = np.array([0, 1, 39, 40])
    coverage = distribution_scores(draws, truth)["95% interval coverage (%)"]
    assert coverage == 50.0


def test_outcome_scores_and_the_single_class_case():
    truth = np.array([0, 0, 5, 10])
    score = np.array([0.1, 0.2, 0.9, 0.8])
    assert outcome_scores(score, truth, 5) == (100.0, 100.0)
    auroc, _ = outcome_scores(score, truth, 10)
    assert auroc == 100 * roc_auc_score([0, 0, 0, 1], score)
    # no place with 20+ cases, and every place with 0+ cases: not defined
    assert np.isnan(outcome_scores(score, truth, 20)).all()
    assert np.isnan(outcome_scores(score, truth, 0)).all()


def test_point_spearman():
    truth = np.array([1, 2, 3])
    assert point_spearman(np.array([3, 2, 1]), truth, 1) == pytest.approx(-1)
    assert np.isnan(point_spearman(np.array([1, 1, 1]), truth, 1))
    assert np.isnan(point_spearman(np.array([1, 2, 3]), truth, 3))


def test_scores_of_the_whole_distribution_use_the_scoring_functions():
    rng = np.random.default_rng(0)
    draws = rng.poisson(2.0, size=(300, 12)).astype(float)
    truth = rng.poisson(2.0, size=12)
    s = distribution_scores(draws, truth)
    assert s["CRPS (cases)"] == pytest.approx(
        np.mean(scoring.crps(truth, draws))
    )
    assert s["WIS (cases)"] == pytest.approx(
        np.mean(scoring.weighted_interval_score(truth, draws))
    )
    assert s["CRPS of the cases above 10 (cases)"] == pytest.approx(
        np.mean(scoring.crps_above_threshold(truth, draws, 10))
    )


def two_year_book():
    """A model with draws and a model with only a ranking, over two years."""
    rng = np.random.default_rng(1)
    book = ScoreBook(workers=2)
    years = {}
    for year in (2021, 2022):
        truth = np.array([0, 0, 0, 1, 3, 7, 12, 0, 25, 0])
        draws = rng.poisson(truth + 1.0, size=(400, truth.size))
        years[year] = (truth, draws)
        windows = {level: truth >= level for level in (1, 5, 10, 20)}
        book.add(
            "with draws",
            year,
            truth,
            truth,
            draws.mean(axis=0),
            draws.mean(axis=0),
            draws,
            draws,
            windows_all=windows,
        )
        ranking = -np.arange(10.0)
        book.add(
            "ranking only",
            year,
            truth,
            truth,
            ranking,
            ranking,
            windows_all=windows,
        )
    return book, years


def test_table_has_a_row_per_year_condition_and_the_average():
    book, _ = two_year_book()
    table = book.table()
    assert table.height == 2 * 2 * 3  # models x conditions x (2 years + mean)
    mean_rows = table.filter(table["Evaluation year"] == "2021-2025")
    assert mean_rows["Condition"].to_list() == [
        CONDITIONS[0],
        CONDITIONS[1],
        CONDITIONS[0],
        CONDITIONS[1],
    ]
    first = table.filter(
        (table["Model"] == "with draws")
        & (table["Condition"] == CONDITIONS[0])
        & (table["Evaluation year"] == "2021")
    )
    assert first["Places scored"].item() == 10
    # the observed counts 3, 7, 12 and 25
    assert first["Places with 3+ cases"].item() == 4
    # a flag of the top 10% of 10 places marks 1 place, of the top 20% 2
    assert first["Places flagged, top 10%"].item() == 1
    assert first["Places flagged, top 20%"].item() == 2
    # the average of the years has no counts
    average = table.filter(
        (table["Model"] == "with draws")
        & (table["Condition"] == CONDITIONS[0])
        & (table["Evaluation year"] == "2021-2025")
    )
    assert average["Places flagged, top 10%"].item() is None


def test_metrics_of_a_model_are_those_of_the_mean_of_its_draws():
    book, years = two_year_book()
    table = book.table()
    truth, draws = years[2021]
    row = table.filter(
        (table["Model"] == "with draws")
        & (table["Condition"] == CONDITIONS[0])
        & (table["Evaluation year"] == "2021")
    )
    expected = forecast_metrics(draws.mean(axis=0), truth)
    for column, value in expected.items():
        if np.isnan(value):
            assert row[column].item() is None or np.isnan(row[column].item())
        else:
            assert row[column].item() == pytest.approx(value)
    # the Spearman rho is that of the risk score given with the draws
    assert row["Spearman rho, 0+"].item() == pytest.approx(
        point_spearman(draws.mean(axis=0), truth, 0)
    )
    # no metric has an interval
    assert not [c for c in table.columns if c.endswith(("_lower", "_upper"))]


def test_average_row_is_the_mean_of_the_years():
    book, years = two_year_book()
    table = book.table()
    column = "RMSE, 0+ (cases)"
    rows = table.filter(
        (table["Model"] == "with draws")
        & (table["Condition"] == CONDITIONS[0])
    )
    yearly = rows.filter(rows["Evaluation year"] != "2021-2025")[column]
    average = rows.filter(rows["Evaluation year"] == "2021-2025")
    assert average[column].item() == pytest.approx(yearly.mean())


def test_ranking_only_model_has_a_spearman_and_no_forecast_metrics():
    book, years = two_year_book()
    table = book.table()
    rows = table.filter(
        (table["Model"] == "ranking only")
        & (table["Condition"] == CONDITIONS[0])
        & (table["Evaluation year"] == "2021")
    )
    truth = years[2021][0]
    assert rows["Spearman rho, 0+"].item() == pytest.approx(
        point_spearman(-np.arange(10.0), truth, 0)
    )
    assert rows["RMSE, 0+ (cases)"].item() is None
    assert rows["CRPS (cases)"].item() is None
    assert rows["AUROC, 5+ (%)"].item() is not None


def test_flagging_the_top_share_by_hand():
    score = np.arange(10.0, 0.0, -1)  # place 0 has the highest score
    outcome = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1], dtype=bool)
    window = outcome.copy()
    window[1] = True  # place 1 has the outcome in a later year
    m = top_share_metrics(score, outcome, window, 0.2)
    # places 0 and 1 are flagged; place 0 has the outcome, place 1 does not
    assert m[("Sensitivity", False)] == pytest.approx(100 / 3)
    assert m[("PPV", False)] == pytest.approx(50)
    assert m[("Specificity", False)] == pytest.approx(100 * 6 / 7)
    assert m[("NPV", False)] == pytest.approx(100 * 6 / 8)
    # with the outcome in the year or later, both flagged places have it
    assert m[("PPV", True)] == pytest.approx(100)
    assert m[("NPV", True)] == pytest.approx(100 * 6 / 8)


def test_flags_of_tied_places_are_split_like_a_random_choice():
    # no information in the score: the share flagged is the share found, and
    # the PPV is the share of places with the outcome
    outcome = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1], dtype=bool)
    m = top_share_metrics(np.zeros(10), outcome, outcome, 0.2)
    assert m[("Sensitivity", False)] == pytest.approx(20)
    assert m[("PPV", False)] == pytest.approx(30)
    assert m[("Specificity", False)] == pytest.approx(80)
    assert top_share_column("PPV", 5, 0.1, True) == (
        "PPV, 5+, top 10%, any window (%)"
    )


def test_fixed_flag_count_ties_the_four_metrics_together():
    rng = np.random.default_rng(3)
    score = rng.normal(size=200)
    outcome = rng.random(200) < 0.1
    m = top_share_metrics(score, outcome, outcome, 0.1)
    # sensitivity = PPV * flagged / positives, whatever the score
    ratio = 20 / outcome.sum()
    assert m[("Sensitivity", False)] == pytest.approx(
        m[("PPV", False)] * ratio
    )


def test_scores_of_the_distribution_do_not_depend_on_the_block():
    rng = np.random.default_rng(0)
    draws = rng.poisson(2.0, size=(300, 11)).astype(float)
    truth = rng.poisson(2.0, size=11)
    a = distribution_scores(draws, truth, block=3)
    b = distribution_scores(draws, truth, block=1000)
    assert a == pytest.approx(b)


def test_any_window_columns_are_only_for_the_cases_including_imports():
    book, _ = two_year_book()
    table = book.table()
    column = "PPV, 5+, top 10%, any window (%)"
    rows = table.filter(
        (table["Model"] == "with draws") & (table["Evaluation year"] == "2021")
    )
    cases = rows.filter(rows["Condition"] == CONDITIONS[0])
    outbreak = rows.filter(rows["Condition"] == CONDITIONS[1])
    assert cases[column].item() is not None
    assert outbreak[column].item() is None
    # the PPV of the year is there for both
    assert outbreak["PPV, 5+, top 10% (%)"].item() is not None


def test_flags_without_a_window_have_the_four_metrics_only():
    score = np.arange(10.0, 0.0, -1)
    outcome = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1], dtype=bool)
    m = top_share_metrics(score, outcome, None, 0.2)
    assert set(m) == {
        ("Sensitivity", False),
        ("Specificity", False),
        ("PPV", False),
        ("NPV", False),
    }
    # the same values as with a window
    with_window = top_share_metrics(score, outcome, outcome, 0.2)
    for key, value in m.items():
        assert with_window[key] == pytest.approx(value)


def test_count_columns_are_in_the_table_and_the_same_for_every_model():
    book, _ = two_year_book()
    table = book.table()
    counts = count_columns()
    assert counts[0] == "Places scored" and len(counts) == 8
    assert all(column in table.columns for column in counts)
    yearly = table.filter(
        (table["Evaluation year"] == "2021")
        & (table["Condition"] == CONDITIONS[0])
    )
    # the two models (with draws and a ranking) score the same 10 places
    assert yearly["Places scored"].to_list() == [10, 10]
    assert yearly["Places with 5+ cases"].to_list() == [3, 3]
    # there are no columns that name a variant or an interval
    assert "Likelihood" not in table.columns


def test_the_means_of_the_place_values_are_the_distribution_scores():
    rng = np.random.default_rng(3)
    truth = np.array([0, 2, 5, 11, 0, 1])
    draws = rng.poisson(truth + 1.0, size=(200, truth.size))
    values = distribution_place_values(draws, truth)
    means = distribution_scores(draws, truth)
    assert set(values) == set(means)
    for name, per_place in values.items():
        assert per_place.shape == (truth.size,)
        assert means[name] == pytest.approx(per_place.mean())


def test_a_row_made_from_the_values_of_the_places_is_the_row_of_the_draws():
    rng = np.random.default_rng(4)
    truth = np.array([0, 0, 0, 1, 3, 7, 12, 0, 25, 0])
    draws = rng.poisson(truth + 1.0, size=(300, truth.size))
    windows = {level: truth >= level for level in (1, 5, 10, 20)}
    forecast, distribution = place_values(truth, draws)
    from_draws = score_condition(
        "m", 2021, CONDITIONS[0], truth, draws.mean(axis=0), windows, draws
    )
    from_places = score_places(
        "m",
        2021,
        CONDITIONS[0],
        truth,
        draws.mean(axis=0),
        windows,
        forecast,
        distribution,
    )
    assert from_places == from_draws
    # a ranking without draws has no forecast and no distribution
    assert place_values(truth, None) == (None, None)


def test_the_score_book_keeps_the_forecast_and_distribution_of_each_place():
    book, years = two_year_book()
    book.table()
    truth, draws = years[2022]
    kept = book.place_values[("with draws", 2022, CONDITIONS[0])]
    assert kept["forecast"] == pytest.approx(draws.mean(axis=0))
    assert kept["distribution"]["WIS (cases)"].shape == (truth.size,)
    ranking = book.place_values[("ranking only", 2022, CONDITIONS[1])]
    assert ranking["forecast"] is None and ranking["distribution"] is None


def test_a_given_forecast_replaces_the_mean_of_the_draws_and_only_that():
    rng = np.random.default_rng(8)
    truth = np.array([0, 0, 3, 7, 0, 12])
    draws = rng.poisson(truth + 1.0, size=(200, truth.size))
    given = np.linspace(0.5, 9.0, truth.size)
    forecast, distribution = place_values(truth, draws, forecast=given)
    mean_forecast, mean_distribution = place_values(truth, draws)
    assert forecast.tolist() == pytest.approx(given.tolist())
    assert not np.allclose(forecast, mean_forecast)
    # the scores of the whole distribution still come from the draws
    for name in mean_distribution:
        assert distribution[name].tolist() == pytest.approx(
            mean_distribution[name].tolist()
        )
    with pytest.raises(AssertionError, match="forecasts for"):
        place_values(truth, draws, forecast=given[:-1])
    # a ranking without draws has no forecast, given or not
    assert place_values(truth, None, forecast=given) == (None, None)


def test_the_point_forecast_metrics_use_the_forecast_given():
    rng = np.random.default_rng(9)
    truth = np.array([0, 0, 3, 7, 0, 12, 1, 0])
    draws = rng.poisson(truth + 1.0, size=(300, truth.size))
    windows = {level: truth >= level for level in (1, 5, 10, 20)}
    score = draws.mean(axis=0)
    exact = truth + 0.0  # a perfect forecast: no error at all
    given = score_condition(
        "m", 2021, CONDITIONS[0], truth, score, windows, draws, forecast=exact
    )
    from_draws = score_condition(
        "m", 2021, CONDITIONS[0], truth, score, windows, draws
    )
    assert given["RMSE, 0+ (cases)"] == pytest.approx(0.0)
    assert from_draws["RMSE, 0+ (cases)"] > 0.5
    # the ranking metrics use the score, and the distribution the draws
    for column in ("AUROC, 1+ (%)", "WIS (cases)", "CRPS (cases)"):
        assert given[column] == pytest.approx(from_draws[column])


def test_the_score_book_keeps_the_forecast_given_for_the_cases_only():
    rng = np.random.default_rng(10)
    truth = np.array([0, 0, 0, 1, 3, 7, 12, 0, 25, 0])
    draws = rng.poisson(truth + 1.0, size=(100, truth.size))
    windows = {level: truth >= level for level in (1, 5, 10, 20)}
    given = np.arange(10.0)
    book = ScoreBook(workers=2)
    book.add(
        "m",
        2021,
        truth,
        truth,
        draws.mean(axis=0),
        draws.mean(axis=0),
        draws,
        draws,
        windows_all=windows,
        forecast_all=given,
    )
    book.table()
    cases = book.place_values[("m", 2021, CONDITIONS[0])]["forecast"]
    outbreak = book.place_values[("m", 2021, CONDITIONS[1])]["forecast"]
    assert cases.tolist() == pytest.approx(given.tolist())
    assert outbreak.tolist() == pytest.approx(draws.mean(axis=0).tolist())
