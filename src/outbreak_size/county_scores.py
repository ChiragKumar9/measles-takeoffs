"""
Scores of the predictions of the places (counties, and regions made of several
counties) of one validation year, for the table of county-level prediction
scoring.

Two kinds of prediction are scored. A model with predictive draws (an array of
simulated case counts for every place) gives one forecast for each place, the
mean of its draws (the expected number of cases), and is also scored on its
whole predictive distribution. A model with only a risk score for each place
(a ranking: vaccine coverage, population) is scored from the ranking alone.
Every prediction is scored under two conditions, each with its own places and
observed counts, which the caller chooses: all the cases of every place
including the imported ones, and the outbreak cases (those that are not
imported) of the places that had at least one case of any kind that year.

Every metric that scores one forecast per place uses the mean of the draws
(the draws are not scored one by one: a single draw is a simulated outcome, so
its RMSE or Spearman rho would include the noise of the simulation). None of
these metrics has an interval.
Metrics of the forecast (the mean of the draws): RMSE, MAE, median fold error
    and R^2, each for the places with at least 0, 3 and 10 cases (the fold
    error from 3 on).
Metrics of the risk score of each place (for a model with draws, the mean of
    its draws unless another score is given): Spearman rho with the observed
    cases for the places with at least 0, 3 and 10 cases, and the AUROC and the
    AUPRC for the outcomes "at least 1, 5, 10 and 20 cases".
Metrics of the whole predictive distribution (all the draws of a place): WIS,
    CRPS, the CRPS of log(1 + cases), the CRPS of the cases above 10, and the
    share of places whose observed count is inside the 95% interval of their
    draws.
Metrics of the places flagged as the highest risk: the places with the highest
10% (or 20%) of the scores are flagged, the same share for every model, a
share fixed before the outcomes are seen. The table gives the number of places
flagged, to compare with the number of places that have each outcome (the
highest sensitivity is flagged / places with the outcome, the highest PPV is
the reverse). For each outcome it gives the sensitivity, specificity, PPV and
NPV of the flags against the cases of the year. For the cases including
imports it also gives the PPV and NPV against the places that had the outcome
in that year or in any later validation year (the "any window" columns); the
outbreak cases have none, because their places are those that had a case in
the year, so a place with its first case in a later year is not among them.
When places tie at the cutoff the flags are split among them in proportion, so
every model flags exactly the same number of places. With the number of
flagged places fixed, the four metrics all follow from one number, the flagged
places that had the outcome, and rank models in the same order within a year
and outcome.

The columns of the metrics, and what the units are:
    RMSE, MAE (the median absolute error, not the mean) and the CRPS of the
    cases are in cases; R^2 is the squared Pearson correlation in percent (not
    one minus the share of variance left over); the fold error is the larger
    of (predicted + 1) / (observed + 1) and its inverse, so a place with 0
    predicted and 0 observed has a fold error of 1 (a perfect forecast).

The row of the years 2021-2025 of a model is the plain average of its yearly
values (a year without a value is skipped).
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

from outbreak_size import scoring

CONDITIONS = ("All cases (including imports)", "Outbreak cases (no imports)")
# the metrics of a forecast (and the Spearman rho) are computed for the places
# with at least this many cases
POINT_LEVELS = (0, 3, 10)
# the fold error is a ratio, so it needs a place with cases
FOLD_ERROR_LEVELS = (3, 10)
# the outcomes "at least this many cases" of the AUROC and the AUPRC
OUTCOME_LEVELS = (1, 5, 10, 20)
# the places with at least this many cases that are counted in the table
COUNT_LEVELS = (1, 3, 5, 10, 20)
# the shares of the places flagged as the highest risk
TOP_SHARES = (0.10, 0.20)
# the interval of the draws that the coverage uses
LOWER_PERCENTILE = 2.5
UPPER_PERCENTILE = 97.5
# the scores of the whole distribution are computed this many places at a time
# (each place is scored on its own, so the result does not depend on it)
PLACE_BLOCK = 400
# scoring threads, and how many scorings may be waiting at once (each holds
# the draws of a model until it is done)
WORKERS = 4

POINT_METRICS = ("RMSE", "MAE", "Median fold error", "Spearman rho", "R²")
DISTRIBUTION_COLUMNS = (
    "WIS (cases)",
    "CRPS (cases)",
    "CRPS of log(1 + cases)",
    "CRPS of the cases above 10 (cases)",
    "95% interval coverage (%)",
)


def point_column(metric, level):
    """The column of a metric of the forecast, for the places with at least
    `level` cases."""
    units = {"RMSE": " (cases)", "MAE": " (cases)", "R²": " (%)"}
    return f"{metric}, {level}+{units.get(metric, '')}"


def squared_correlation(forecast, truth):
    """Squared Pearson correlation of a forecast with the observed cases (NaN
    when either does not vary)."""
    if np.std(forecast) == 0 or np.std(truth) == 0:
        return np.nan
    return float(np.corrcoef(forecast, truth)[0, 1] ** 2)


def fold_error(predicted, observed):
    """The larger of (predicted + 1) / (observed + 1) and its inverse: always
    defined, and exactly 1 for a place predicted and observed at 0."""
    ratio = (predicted + 1.0) / (observed + 1.0)
    return np.maximum(ratio, 1.0 / ratio)


def forecast_metrics(forecast, truth):
    """
    RMSE, MAE (the median absolute error), median fold error and R^2 of a
    forecast, for the places with at least each of POINT_LEVELS cases.
    forecast: [n_places] expected cases (the mean of the draws of each place);
    truth: [n_places] observed cases. Returns {column: value}; a level with
    no place gives NaN, and so does the R^2 of a level whose forecast or
    observed counts do not vary. Units: RMSE and MAE in cases, the fold error
    on cases + 1 (see fold_error), R^2 in percent.
    """
    forecast = np.asarray(forecast, dtype=float)
    truth = np.asarray(truth, dtype=float)
    out = {}
    for level in POINT_LEVELS:
        in_level = truth >= level
        f, y = forecast[in_level], truth[in_level]
        error = f - y
        values = {
            "RMSE": np.sqrt(np.mean(error**2)) if y.size else np.nan,
            "MAE": np.median(np.abs(error)) if y.size else np.nan,
            "R²": 100 * squared_correlation(f, y) if y.size >= 2 else np.nan,
        }
        if level in FOLD_ERROR_LEVELS:
            values["Median fold error"] = (
                np.median(fold_error(f, y)) if y.size else np.nan
            )
        for metric, value in values.items():
            out[point_column(metric, level)] = float(value)
    return out


def _covered(draws, truth):
    """For each place, whether its observed count is inside the 2.5th to
    97.5th percentile interval of its draws, both ends included."""
    draws = np.asarray(draws, dtype=float)
    lower = np.percentile(draws, LOWER_PERCENTILE, axis=0)
    upper = np.percentile(draws, UPPER_PERCENTILE, axis=0)
    return (lower <= truth) & (truth <= upper)


def distribution_place_values(draws, truth, block=PLACE_BLOCK):
    """The scores of the whole predictive distribution of each place:
    {column of DISTRIBUTION_COLUMNS: [n_places] value of every place} (the WIS,
    three versions of the CRPS, and 100 for a place inside the 95% interval of
    its draws and 0 for one outside). The places are scored `block` at a time,
    to keep the memory small and the sorting fast; every place is scored on its
    own, so the block does not change the result.
    """
    truth = np.asarray(truth, dtype=float)
    parts = {name: [] for name in DISTRIBUTION_COLUMNS}
    for start in range(0, len(truth), block):
        places = slice(start, start + block)
        d = np.asarray(draws[:, places], dtype=float)
        y = truth[places]
        parts["WIS (cases)"].append(scoring.weighted_interval_score(y, d))
        parts["CRPS (cases)"].append(scoring.crps(y, d))
        parts["CRPS of log(1 + cases)"].append(
            scoring.crps(np.log1p(y), np.log1p(d))
        )
        parts["CRPS of the cases above 10 (cases)"].append(
            scoring.crps_above_threshold(y, d, 10)
        )
        parts["95% interval coverage (%)"].append(100.0 * _covered(d, y))
    return {name: np.concatenate(values) for name, values in parts.items()}


def distribution_scores(draws, truth, block=PLACE_BLOCK):
    """The scores of the whole predictive distribution of the places: the mean
    over the places of the WIS and of three versions of the CRPS, and the
    percent of places inside the 95% interval of their draws (see
    distribution_place_values)."""
    return {
        name: float(np.mean(values))
        for name, values in distribution_place_values(
            draws, truth, block
        ).items()
    }


def outcome_scores(score, truth, level):
    """(AUROC, AUPRC) in percent of the score as a ranking of the places with
    at least `level` cases; NaN for both when every place, or none, has them
    (an AUROC needs both kinds)."""
    outcome = np.asarray(truth) >= level
    if outcome.all() or not outcome.any():
        return np.nan, np.nan
    score = np.asarray(score, dtype=float)
    return (
        float(100 * roc_auc_score(outcome, score)),
        float(100 * average_precision_score(outcome, score)),
    )


def point_spearman(score, truth, level):
    """Spearman rho of a score with the observed cases of the places with at
    least `level` cases; NaN when there are fewer than two such places or the
    score is the same for all of them."""
    in_level = np.asarray(truth) >= level
    if in_level.sum() < 2:
        return np.nan
    score = np.asarray(score, dtype=float)[in_level]
    if np.all(score == score[0]):
        return np.nan
    return float(stats.spearmanr(np.asarray(truth)[in_level], score)[0])  # type: ignore


def flagged_column(share):
    """The column of the number of places flagged for a `share`."""
    return f"Places flagged, top {round(100 * share)}%"


def count_columns():
    """The columns that count places: the places scored, those with each
    number of cases and those flagged. They are the same for every model of a
    year and condition."""
    return [
        "Places scored",
        *[f"Places with {k}+ cases" for k in COUNT_LEVELS],
        *[flagged_column(share) for share in TOP_SHARES],
    ]


def top_share_column(metric, level, share, any_window=False):
    """The column of a metric of the flagged places, for the outcome "at least
    `level` cases" and the highest `share` of the scores flagged."""
    window = ", any window" if any_window else ""
    return f"{metric}, {level}+, top {round(100 * share)}%{window} (%)"


def top_share_metrics(score, outcome, window_outcome, share):
    """
    Sensitivity, specificity, PPV and NPV (percent) of flagging the `share` of
    the places with the highest scores, with the outcome of the year, and the
    PPV and NPV with the outcome in the year or any later one.

    score: [n_places] risk score, higher is higher risk; outcome: [n_places]
    whether the place had the outcome in the year; window_outcome: whether it
    had it in the year or any later validation year (None: no "any window"
    metrics). The number flagged is ceil(share * n_places). Places tied at the
    cutoff are flagged in proportion (the average over every way of choosing
    among them), so the number flagged is the same for every model. Returns
    {(metric, any_window): value}, to name with top_share_column, for
    Sensitivity, Specificity, PPV and NPV (and PPV and NPV with any_window). A
    metric with an empty denominator is NaN.
    """
    score = np.asarray(score, dtype=float)
    n = len(score)
    flagged = int(np.ceil(share * n))
    cutoff = np.sort(score)[::-1][flagged - 1]
    above = score > cutoff
    tied = score == cutoff
    # the share of the tied places that are flagged
    tied_share = (flagged - above.sum()) / tied.sum()

    def counts(y):
        """(flagged with the outcome, with the outcome, without it)"""
        y = np.asarray(y, dtype=bool)
        hits = y[above].sum() + tied_share * y[tied].sum()
        return hits, y.sum(), n - y.sum()

    def percent(numerator, denominator):
        return 100 * numerator / denominator if denominator > 0 else np.nan

    hits, positives, negatives = counts(outcome)
    # places with the outcome that are not flagged
    missed = positives - hits
    # of the places not flagged, those without the outcome are the true
    # negatives; of the places flagged, those without it are the false alarms
    out = {
        ("Sensitivity", False): percent(hits, positives),
        ("Specificity", False): percent(
            negatives - (flagged - hits), negatives
        ),
        ("PPV", False): percent(hits, flagged),
        ("NPV", False): percent(n - flagged - missed, n - flagged),
    }
    if window_outcome is not None:
        window_hits, window_positives, _ = counts(window_outcome)
        window_missed = window_positives - window_hits
        out[("PPV", True)] = percent(window_hits, flagged)
        out[("NPV", True)] = percent(n - flagged - window_missed, n - flagged)
    return out


def place_values(truth, draws, model="", year="", forecast=None):
    """What the draws give for each place: (the forecast, the mean of the draws
    of each place or the `forecast` given (the exact expected cases of each
    place, when the mean of the draws would only estimate it with the noise of
    the simulated imports); the scores of the whole distribution of each place,
    see distribution_place_values), or (None, None) for a ranking without
    draws."""
    if draws is None:
        return None, None
    assert draws.shape[1] == len(truth), (
        f"{model}, {year}: {draws.shape[1]} places in the draws but "
        f"{len(truth)} observed"
    )
    if forecast is None:
        forecast = np.mean(draws, axis=0)
    else:
        forecast = np.asarray(forecast, dtype=float)
        assert forecast.shape == (len(truth),), (
            f"{model}, {year}: {forecast.shape} forecasts for "
            f"{len(truth)} places"
        )
    return forecast, distribution_place_values(draws, truth)


def score_condition(
    model, year, condition, truth, score, windows, draws, forecast=None
):
    """
    Scores one model in one year under one condition and returns the row of
    the table. truth: observed cases of the places of the condition; score: the
    risk score of each place (AUROC, AUPRC, flags and Spearman rho); draws:
    [n_draws, n_places] predictive draws of the same places, or None for a
    ranking without draws (which has only the ranking metrics). The metrics of
    one forecast per place use the mean of the draws (or `forecast`, see
    place_values); the metrics of the whole distribution use all of them.
    `windows` maps each outcome level to whether
    the place had the outcome that year or in a later one (for the "any
    window" PPV and NPV); None for a condition without those columns.
    """
    forecast, distribution = place_values(truth, draws, model, year, forecast)
    return score_places(
        model, year, condition, truth, score, windows, forecast, distribution
    )


def score_places(
    model, year, condition, truth, score, windows, forecast, distribution
):
    """
    The row of the table of one model in one year under one condition, from the
    values of each place: truth, score and windows as in score_condition;
    forecast: [n_places] expected cases (the mean of the draws), or None for a
    ranking without draws; distribution: {column: [n_places] value of every
    place} (distribution_place_values), or None. A resample of the places
    (repeating places) is scored with the same function as the places
    themselves.
    """
    row = {
        "Model": model,
        "Evaluation year": str(year),
        "Condition": condition,
        "Places scored": len(truth),
    }
    for level in COUNT_LEVELS:
        row[f"Places with {level}+ cases"] = int((truth >= level).sum())
    # how many places each flag marks: next to the places with each outcome it
    # gives the highest sensitivity (flagged / places with the outcome) and the
    # highest PPV (places with the outcome / flagged) that a flag can reach
    for share in TOP_SHARES:
        row[flagged_column(share)] = int(np.ceil(share * len(truth)))
    for level in OUTCOME_LEVELS:
        auroc, auprc = outcome_scores(score, truth, level)
        row[f"AUROC, {level}+ (%)"] = auroc
        row[f"AUPRC, {level}+ (%)"] = auprc
        for share in TOP_SHARES:
            flagged = top_share_metrics(
                score,
                truth >= level,
                None if windows is None else windows[level],
                share,
            )
            for (metric, any_window), value in flagged.items():
                column = top_share_column(metric, level, share, any_window)
                row[column] = float(value)
    # the Spearman rho of the risk score, the same for every row of the table
    for level in POINT_LEVELS:
        row[point_column("Spearman rho", level)] = point_spearman(
            score, truth, level
        )
    if forecast is None:
        return row
    row.update(forecast_metrics(forecast, truth))
    row.update({name: float(np.mean(v)) for name, v in distribution.items()})
    return row


def score_with_values(
    model, year, condition, truth, score, windows, draws, forecast=None
):
    """score_condition, also returning the values of each place that the draws
    gave: (row, {"forecast": ..., "distribution": ...}). The ScoreBook keeps
    them, so that the places can be resampled later without the draws."""
    forecast, distribution = place_values(truth, draws, model, year, forecast)
    row = score_places(
        model, year, condition, truth, score, windows, forecast, distribution
    )
    return row, {"forecast": forecast, "distribution": distribution}


class ScoreBook:
    """
    Collects the scores of the models, one set per model, year and condition,
    and builds the table: one row per model, condition and year, then a row of
    the average over the years. The scoring runs in a few threads (the heavy
    work is in numpy, which does not hold the interpreter lock); a scoring
    holds the draws of its model until it is done, so no more than `workers`
    wait at a time.
    """

    def __init__(self, workers=WORKERS):
        self.rows = []
        # the values of each place that the draws gave, by (model, year, condition)
        self.place_values = {}
        self._executor = ThreadPoolExecutor(max_workers=workers)
        self._waiting = deque()
        self._max_waiting = workers

    def add(
        self,
        model,
        year,
        truth_all,
        truth_outbreak,
        score_all,
        score_outbreak,
        draws_all=None,
        draws_outbreak=None,
        *,
        windows_all,
        forecast_all=None,
    ):
        """
        Scores one model in one year under both conditions. Each condition
        has its own places: truth_*, score_* and draws_* of a condition are
        all about the same places (the caller restricts the places of the
        outbreak condition to those that had a case). truth_*: the observed
        cases ([n_places]); score_*: the risk score of every place (for the
        AUROC, the AUPRC, the flags and the Spearman rho); draws_*:
        [n_draws, n_places] predictive draws (their mean is the forecast), or
        None for a model with only a ranking; windows_all: {outcome level:
        whether the place had that many cases in the year or in a later
        validation year}, for the "any window" PPV and NPV of the cases
        including imports (the outbreak condition has none). forecast_all:
        the forecast of the cases including imports of every place when it is
        not the mean of draws_all (the exact expected cases).
        """
        conditions = zip(
            CONDITIONS,
            (truth_all, truth_outbreak),
            (score_all, score_outbreak),
            (windows_all, None),
            (draws_all, draws_outbreak),
            (forecast_all, None),
        )
        for condition, truth, score, windows, draws, forecast in conditions:
            if windows is not None:
                for level in OUTCOME_LEVELS:
                    assert level in windows, f"no window outcome for {level}+"
            while len(self._waiting) >= self._max_waiting:
                self._collect()
            # copies of the small arrays: the caller may reuse its variables
            future = self._executor.submit(
                score_with_values,
                model,
                year,
                condition,
                np.array(truth),
                np.array(score, dtype=float),
                None
                if windows is None
                else {k: np.array(v, dtype=bool) for k, v in windows.items()},
                draws,
                None if forecast is None else np.array(forecast, dtype=float),
            )
            self._waiting.append(future)

    def _collect(self):
        """Takes the oldest scoring that was submitted, waiting for it."""
        row, values = self._waiting.popleft().result()
        self.rows.append(row)
        self.place_values[
            (row["Model"], int(row["Evaluation year"]), row["Condition"])
        ] = values

    def table(self):
        """The scores as a polars table: for each model and condition (in the
        order added) its yearly rows and a row of 2021-2025, the average of the
        years."""
        while self._waiting:
            self._collect()
        self._executor.shutdown()
        grouped = {}
        for row in self.rows:
            key = (row["Model"], row["Condition"])
            grouped.setdefault(key, []).append(row)
        labels = ("Model", "Evaluation year", "Condition")
        out = []
        for (model, condition), rows in grouped.items():
            out.extend(rows)
            pooled = {
                "Model": model,
                "Evaluation year": "2021-2025",
                "Condition": condition,
            }
            for column in rows[0]:
                if column in labels or column.startswith("Places"):
                    continue
                values = np.array([r[column] for r in rows], dtype=float)
                values = values[~np.isnan(values)]
                mean = values.mean() if values.size else np.nan
                pooled[column] = float(mean)
            out.append(pooled)
        table = pl.from_dicts(out, infer_schema_length=None)
        return table.select(self.column_order(table.columns))

    @staticmethod
    def column_order(columns):
        """The columns in the order of the table: the labels and counts, the
        metrics of the ranking and of the flagged places, the metrics of the
        forecast and the Spearman rho, then the scores of the whole
        distribution.
        """
        first = ["Model", "Evaluation year", "Condition"]
        counts = count_columns()
        ranking = []
        for level in OUTCOME_LEVELS:
            ranking += [f"AUROC, {level}+ (%)", f"AUPRC, {level}+ (%)"]
            for share in TOP_SHARES:
                for metric, any_window in (
                    ("Sensitivity", False),
                    ("Specificity", False),
                    ("PPV", False),
                    ("NPV", False),
                    ("PPV", True),
                    ("NPV", True),
                ):
                    ranking.append(
                        top_share_column(metric, level, share, any_window)
                    )
        point = [
            point_column(metric, level)
            for level in POINT_LEVELS
            for metric in POINT_METRICS
        ]
        ordered = first + counts + ranking + point + list(DISTRIBUTION_COLUMNS)
        return [c for c in ordered if c in columns]
