"""
The scores of the table of county-level prediction scoring (S1) over a window of
years, with an interval: for each model and condition, the mean over the years of
the window of each yearly score (county_scores.score_places), and the 95% interval
of that mean from a bootstrap of the places within each year.

Which years count. A score of "at least `level` cases" (the AUROC, AUPRC,
sensitivity, specificity, PPV and NPV, and the forecast metrics and Spearman rho of
the places with at least `level` cases) needs a place with that many cases: the
year of a condition that has none is left out of the mean, for every metric of
that level, and the other levels keep it (a year with no place of 10+ cases still
counts for 5+). An outcome also needs a place without it (an AUROC needs both
kinds). The scores of the whole distribution (WIS, CRPS, coverage) use every year.
The same rule serves the point estimates and every replicate of the bootstrap.

The bootstrap. In each year the places are resampled by cluster. The cluster of a
place is its state, with the states joined where an outbreak crosses their border
(outbreak_clusters.bootstrap_clusters), so the places of one state, and of one
outbreak, stay together. Each score is resampled for its own level of cases: the
places with that many cases or more are separated from the others, and within each
of the two groups the clusters are drawn with replacement (a draw repeats all the
places of the cluster). Splitting only on the outcome of the score keeps at least
one place with the outcome and one without in every replicate, so the score is
defined, and it lets which places carry the outbreaks vary. The scores without a
level (and those at 0+, which every place has) are resampled without a split.
Every kind of resample has draws of its own, made from the seed, the condition, the
year and the level, so the models of a condition and year are resampled alike and
the scores of a level are the same whichever levels are computed. The interval is
the 2.5th and 97.5th percentile of the means of the replicates. It counts the
sampling of the places; it does not count that other years could have looked
different.
"""

import re
import warnings
from dataclasses import dataclass

import numpy as np
import polars as pl

from outbreak_size.county_scores import (
    CONDITIONS,
    DISTRIBUTION_COLUMNS,
    OUTCOME_LEVELS,
    ScoreBook,
    score_places,
)
from outbreak_size.place_scores import score_column

# the kind of resample of the scores that have no level (and of those at 0+): no
# split of the places
PLAIN = -1
OUTCOME_METRICS = (
    "AUROC",
    "AUPRC",
    "Sensitivity",
    "Specificity",
    "PPV",
    "NPV",
)
LEVEL_PATTERN = re.compile(r", (\d+)\+")
LOWER, UPPER = 2.5, 97.5


def resample_kind(level):
    """The kind of resample a score is computed on: its own level (the places with
    that many cases or more against the others), or PLAIN for a score without a
    level or at 0+, which every place has."""
    return PLAIN if level is None or level == 0 else level


def column_level(column):
    """The level of cases ("at least k cases") a column is about: k, or None for
    a score that has no level (WIS, CRPS, coverage)."""
    match = LEVEL_PATTERN.search(column)
    return int(match.group(1)) if match else None


def is_outcome_column(column):
    """Whether the column scores the outcome "at least `level` cases" of every
    place (which needs places with and without it), as opposed to the forecast
    and Spearman metrics of the places with at least `level` cases."""
    return column.startswith(OUTCOME_METRICS)


def level_is_scored(truth, level, outcome):
    """Whether a year with these observed counts has a score at `level`: a place
    with the level, and for an outcome also a place without it."""
    reached = int((np.asarray(truth) >= level).sum())
    return 0 < reached < len(truth) if outcome else reached >= 1


@dataclass
class YearPlaces:
    """The places of one model in one year under one condition, with the values
    that score_places needs. `cluster` is the cluster of each place (an integer): its
    state, with the states joined where an outbreak crosses their border."""

    truth: np.ndarray
    score: np.ndarray
    cluster: np.ndarray
    windows: dict | None
    forecast: np.ndarray | None
    distribution: dict | None

    def take(self, multiplicity):
        """The places repeated `multiplicity` times each (0 leaves one out)."""
        rep = lambda a: np.repeat(a, multiplicity)  # noqa: E731
        return YearPlaces(
            rep(self.truth),
            rep(self.score),
            rep(self.cluster),
            None
            if self.windows is None
            else {k: rep(v) for k, v in self.windows.items()},
            None if self.forecast is None else rep(self.forecast),
            None
            if self.distribution is None
            else {k: rep(v) for k, v in self.distribution.items()},
        )

    def row(self, model, year, condition):
        return score_places(
            model,
            year,
            condition,
            self.truth,
            self.score,
            self.windows,
            self.forecast,
            self.distribution,
        )


def stratified_multiplicity(truth, cluster, rng, level=PLAIN):
    """How many times each place is drawn: the places with `level` cases or more
    are separated from the others (no split for PLAIN), in each of the groups the
    clusters are drawn with replacement, and every place of a cluster is repeated
    as many times as its cluster was drawn."""
    truth, cluster = np.asarray(truth), np.asarray(cluster)
    stratum = (
        np.zeros(len(truth), dtype=int)
        if level == PLAIN
        else (truth >= level).astype(int)
    )
    multiplicity = np.zeros(len(truth), dtype=int)
    for s in np.unique(stratum):
        members = np.flatnonzero(stratum == s)
        clusters, inverse = np.unique(cluster[members], return_inverse=True)
        drawn = np.bincount(
            rng.integers(0, len(clusters), len(clusters)),
            minlength=len(clusters),
        )
        multiplicity[members] = drawn[inverse]
    return multiplicity


def score_columns(row):
    """The scores of a row of score_places (not the labels or the counts of
    places)."""
    return [
        c
        for c in row
        if c not in ("Model", "Evaluation year", "Condition")
        and not c.startswith("Places")
    ]


def summarize(
    model,
    condition,
    years,
    windows,
    replicates,
    seed,
    condition_index,
    levels=None,
):
    """
    The mean over each window of the yearly scores of one model and condition, with
    the interval of the bootstrap.

    years: {year: YearPlaces}; windows: {label: [years]}; the draws of each kind
    of resample are made from seed, the condition, the year and the level only, so
    every model of the condition gets the same ones. `levels`, if given, are the
    kinds of resample to compute (levels of cases, and PLAIN): the scores of the
    other kinds get no interval (NaN) and the rest are those of a full run.
    Returns {window label: {column: (mean, lower, upper)}}.
    """
    year_list = sorted(years)
    unit_rows = {y: years[y].row(model, y, condition) for y in year_list}
    columns = score_columns(unit_rows[year_list[0]])
    unit = np.array(
        [[unit_rows[y][c] for c in columns] for y in year_list], dtype=float
    )
    # which years count for each column: [years, columns]
    counts = np.array(
        [
            [
                True
                if column_level(c) is None
                else level_is_scored(
                    years[y].truth, column_level(c), is_outcome_column(c)
                )
                for c in columns
            ]
            for y in year_list
        ]
    )
    # the kind of resample of each column, and the kinds to compute
    kind_of = [resample_kind(column_level(c)) for c in columns]
    kinds = sorted(k for k in set(kind_of) if levels is None or k in levels)
    reps = np.full((replicates, len(year_list), len(columns)), np.nan)
    for j, y in enumerate(year_list):
        for kind in kinds:
            of_kind = [i for i, k in enumerate(kind_of) if k == kind]
            # PLAIN is -1: the entries of the seed are not negative
            rng = np.random.default_rng([seed, condition_index, y, kind + 1])
            for b in range(replicates):
                resample = years[y].take(
                    stratified_multiplicity(
                        years[y].truth, years[y].cluster, rng, kind
                    )
                )
                row = resample.row(model, y, condition)
                reps[b, j, of_kind] = [row[columns[i]] for i in of_kind]

    def mean_over(values, in_window):
        """The mean over the years of the window that count, ignoring a missing
        score (axis -2 is the years)."""
        masked = np.where(in_window, values, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", RuntimeWarning
            )  # a column with no year to count
            return np.nanmean(masked, axis=-2)

    out = {}
    for label, window in windows.items():
        in_window = counts & np.isin(year_list, window)[:, None]
        point = mean_over(unit, in_window)
        boot = mean_over(reps, in_window)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lower = np.nanpercentile(boot, LOWER, axis=0)
            upper = np.nanpercentile(boot, UPPER, axis=0)
        out[label] = {
            c: (float(point[i]), float(lower[i]), float(upper[i]))
            for i, c in enumerate(columns)
        }
    return out


def summary_table(summaries):
    """The table of the summaries: one row per model, condition and window, and for
    every score three columns (the mean, the lower and the upper end of the
    interval). summaries: {(model, condition): the output of summarize}."""
    rows = []
    for (model, condition), by_window in summaries.items():
        for label, scores in by_window.items():
            row = {
                "Model": model,
                "Evaluation year": label,
                "Condition": condition,
            }
            for column in ScoreBook.column_order(
                ["Model", "Evaluation year", "Condition", *scores]
            )[3:]:
                value, lower, upper = scores[column]
                row[column] = value
                row[f"{column}, lower"] = lower
                row[f"{column}, upper"] = upper
            rows.append(row)
    return pl.from_dicts(rows, infer_schema_length=None)


def window_column(level):
    """The column of the place file with whether each place had `level` cases in
    its year or in a later validation year (for the "any window" metrics)."""
    return f"window_{level}"


def value_groups(values):
    """The long table of forecasts and distribution scores (place_scores.
    place_value_table) as {(model, condition, year): its rows}."""
    return {
        (model, condition, year): group.sort("row")
        for (model, condition, year), group in values.partition_by(
            ["Model", "Condition", "year"], as_dict=True
        ).items()
    }


def year_places(places, groups, model, condition, year):
    """
    The YearPlaces of a model in a year under a condition, from the place file
    (place_scores.place_score_table) and the groups of value_groups: the places
    the condition scores (all of them, or those with a case for the outbreak
    cases), their cases, the score of the model, their cluster (the cluster column of the place file),
    and the forecast and distribution scores if the model has draws in the
    condition.
    """
    rows = places.filter(pl.col("year") == year)
    outbreak = condition == CONDITIONS[1]
    keep = (
        (rows["cases"].cast(pl.Int64) >= 1).to_numpy()
        if outbreak
        else np.ones(rows.height, dtype=bool)
    )
    truth = (
        rows["cases_outbreak" if outbreak else "cases"]
        .cast(pl.Int64)
        .to_numpy()[keep]
    )
    group = groups.get((model, condition, year))
    forecast = distribution = None
    if group is not None:
        if group["row"].to_list() != np.flatnonzero(keep).tolist():
            raise ValueError(
                f"{model}, {year}, {condition}: the values are not those of the places scored"
            )
        forecast = group["Forecast"].to_numpy()
        distribution = {c: group[c].to_numpy() for c in DISTRIBUTION_COLUMNS}
    windows = None
    if not outbreak:
        windows = {
            level: rows[window_column(level)].to_numpy()
            for level in OUTCOME_LEVELS
        }
    return YearPlaces(
        truth=truth,
        score=rows[score_column(condition, model)].to_numpy()[keep],
        cluster=np.unique(
            rows["cluster"].to_numpy()[keep], return_inverse=True
        )[1],
        windows=windows,
        forecast=forecast,
        distribution=distribution,
    )


def summarize_task(task):
    """One model and condition (the unit of work of a process of the pool). task:
    (model, condition, condition_index, years, windows, replicates, seed)."""
    model, condition, condition_index, years, windows, replicates, seed = task
    return (model, condition), summarize(
        model, condition, years, windows, replicates, seed, condition_index
    )


def expected_from_yearly(yearly, model, condition, window, column):
    """The mean of the yearly values of `column` (rows of the yearly scores of the
    table script) over the years of the window that have a
    place of the level of the column: the number the summary reproduces when the
    places are not resampled."""
    rows = yearly.filter(
        (pl.col("Model") == model)
        & (pl.col("Condition") == condition)
        & pl.col("Evaluation year").is_in([str(y) for y in window])
    )
    level = column_level(column)
    if level is not None:
        reached = (
            rows["Places scored"]
            if level == 0
            else rows[f"Places with {level}+ cases"]
        )
        scored = rows["Places scored"]
        counts = (
            (reached > 0) & (reached < scored)
            if is_outcome_column(column)
            else reached >= 1
        )
        rows = rows.filter(counts)
    values = rows[column].drop_nans().drop_nulls()
    return float(values.mean()) if values.len() else np.nan


def check_means(table, yearly, windows, tolerance=2e-3):
    """Raise if a mean of the summary table differs from the mean of the yearly
    scores (expected_from_yearly) by more than `tolerance`. Returns the number of
    values checked and the largest difference."""
    worst, checked = 0.0, 0
    for row in table.iter_rows(named=True):
        window = windows[row["Evaluation year"]]
        for column in table.columns[3:]:
            if column.endswith((", lower", ", upper")) or row[column] is None:
                continue
            expected = expected_from_yearly(
                yearly, row["Model"], row["Condition"], window, column
            )
            value = row[column]
            if np.isnan(expected) and (value is None or np.isnan(value)):
                continue
            difference = abs(value - expected)
            if not difference < tolerance:
                raise ValueError(
                    f"{row['Model']}, {row['Condition']}, {row['Evaluation year']}, {column}: "
                    f"{value} here and {expected} from the yearly scores"
                )
            worst, checked = max(worst, difference), checked + 1
    return checked, worst


def single_year_summary(year, place_table, value_table, replicates, seed):
    """
    The table of the scores of every model and condition in one year with the
    interval of the bootstrap of its places: the rows of S1 for a window of
    that year alone (the interval counts the sampling of the places, as for
    S1; one year has no other year to vary).

    place_table and value_table are those of place_scores (the table script
    writes them for 2021-2025; the headline table of 2026 builds them for
    its year). Every model with a column in place_table is summarized, under
    both conditions, with the draws of the bootstrap that S1 would use for
    that year (the same seed and condition). Returns summary_table's table,
    with the year as the label of "Evaluation year".
    """
    groups = value_groups(value_table)
    prefix = score_column(CONDITIONS[0], "")
    models = [
        c[len(prefix) :] for c in place_table.columns if c.startswith(prefix)
    ]
    summaries = {}
    for model in models:
        for index, condition in enumerate(CONDITIONS):
            summaries[(model, condition)] = summarize(
                model,
                condition,
                {
                    year: year_places(
                        place_table, groups, model, condition, year
                    )
                },
                {str(year): [year]},
                replicates,
                seed,
                index,
            )
    return summary_table(summaries)
