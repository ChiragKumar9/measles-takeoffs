"""
The selection of models, and their consistency across the years, from the table of
yearly scores of county-level prediction (outbreak_size.county_scores).

The models are in pools, and each pool is selected on its own. In every year
the models of a pool are ranked on their scores of the cases including imports:
first the AUROC for 5+ cases, and where models tie the AUPRC for 5+ cases, then
the AUROC for 10+ cases, then the AUPRC for 10+ cases (SELECTION_METRICS; a
score that is missing ranks last). Models that tie on all four keep the same
rank, and no other information separates them. The models passed forward from a
year are those whose rank is at most the number to pass forward, so every
model tied with the last one passed forward is passed forward too (the set can
be larger than the number). They are then scored in the next year, which was
not used to select them.

consistency_table describes how close every model is to the best model of its pool in
each year, and whether it was passed forward from each year. best_of_pool gives
the best model of a pool: the smallest mean gap from the best of the pool, over
the years given.
"""

import numpy as np
import polars as pl

from outbreak_size.county_scores import CONDITIONS

CASES, OUTBREAK = CONDITIONS
# the scores that rank the models, in the order they break ties
SELECTION_METRICS = (
    "AUROC, 5+ (%)",
    "AUPRC, 5+ (%)",
    "AUROC, 10+ (%)",
    "AUPRC, 10+ (%)",
)
YEARS = (2021, 2022, 2023, 2024, 2025)
# the names of the benchmarks (rankings and the import model) in the scoring tables
COVERAGE_RANK_MODEL_NAME = "Vaccine coverage rank order"
POPULATION_RANK_MODEL_NAME = "Population size rank order"
IMPORT_MODEL_NAME = "Import model alone"
PRIOR_CASES_MODEL_NAME = "Prior cases summed since 2011"
# the years over which the best model of each pool is chosen (its mean gap)
BEST_MODEL_YEARS = YEARS[1:]
# the summary columns of consistency_table are over these years (2021 has
# places with 5+ cases too few to rank the models on, so the per-year columns
# keep it but the mean and the largest gap do not)
SUMMARY_LABEL = f"{BEST_MODEL_YEARS[0]}-{BEST_MODEL_YEARS[-1]}"
MEAN_AUROC_COLUMN = f"Mean AUROC, 5+ (%), {SUMMARY_LABEL}"
MEAN_GAP_COLUMN = f"Mean gap from the best of the pool, {SUMMARY_LABEL}"
LARGEST_GAP_COLUMN = f"Largest gap from the best of the pool, {SUMMARY_LABEL}"


def _value(table, model, year, condition, metric):
    """The score of a model in a year (NaN when it has none)."""
    rows = table.filter(
        (pl.col("Model") == model)
        & (pl.col("Evaluation year") == str(year))
        & (pl.col("Condition") == condition)
    )
    assert rows.height == 1, (model, year, condition, rows.height)
    value = rows[metric].item()
    return np.nan if value is None else float(value)


def _keys(table, models, year):
    """{model: its SELECTION_METRICS of the cases including imports}, with a
    missing score as minus infinity so that it ranks last."""
    keys = {}
    for model in models:
        values = [
            _value(table, model, year, CASES, m) for m in SELECTION_METRICS
        ]
        keys[model] = tuple(-np.inf if np.isnan(v) else v for v in values)
    return keys


def rank_models(keys):
    """{model: rank} from {model: tuple of scores, higher is better}: one plus
    the number of models with a better tuple (compared key by key, the first
    key first), so models with the same tuple share a rank."""
    return {
        model: 1 + sum(other > key for other in keys.values())
        for model, key in keys.items()
    }


def consistency_table(table, pools, pass_forward, comparators):
    """
    How close every model is to the best model of its pool in each year, on
    the AUROC for 5+ cases of the cases including imports.

    One row for each model: its AUROC in each year, its rank in each year (the
    rank of one_year_ahead_table), whether it was passed forward from each year
    (its rank is within the number to pass forward from its pool), its gap from the best AUROC of its pool in
    each year (in points), the mean and the largest gap over 2022-2025 (the
    years the headline model is chosen on), and the
    number of years it was passed forward (its rank within the number passed
    forward from its pool, of the 5 years; the models passed forward from 2025 are
    not scored in a later year); the models of a pool are sorted by their
    mean gap. The comparators have their AUROC only.
    """
    metric = SELECTION_METRICS[0]
    out = []
    for pool, models in pools.items():
        values = {
            y: {m: _value(table, m, y, CASES, metric) for m in models}
            for y in YEARS
        }
        ranks = {y: rank_models(_keys(table, models, y)) for y in YEARS}
        best = {y: np.nanmax(list(values[y].values())) for y in YEARS}
        rows = []
        for model in models:
            aurocs = [values[y][model] for y in YEARS]
            gaps = [best[y] - values[y][model] for y in YEARS]
            row = {"Pool": pool, "Model": model}
            row.update(
                {f"AUROC, 5+ (%), {y}": values[y][model] for y in YEARS}
            )
            row.update(
                {f"Selection rank, {y}": ranks[y][model] for y in YEARS}
            )
            row.update(
                {
                    f"Passed forward from {y}"
                    + (" (not scored later)" if y == YEARS[-1] else ""): bool(
                        ranks[y][model] <= pass_forward[pool]
                    )
                    for y in YEARS
                }
            )
            row.update(
                {
                    f"Gap from the best of the pool, {y}": best[y]
                    - values[y][model]
                    for y in YEARS
                }
            )
            summary = [YEARS.index(y) for y in BEST_MODEL_YEARS]
            row[MEAN_AUROC_COLUMN] = float(
                np.nanmean([aurocs[i] for i in summary])
            )
            row[MEAN_GAP_COLUMN] = float(
                np.nanmean([gaps[i] for i in summary])
            )
            row[LARGEST_GAP_COLUMN] = float(
                np.nanmax([gaps[i] for i in summary])
            )
            row["Years passed forward (of 5)"] = int(
                sum(ranks[y][model] <= pass_forward[pool] for y in YEARS)
            )
            rows.append(row)
        rows.sort(key=lambda r: r[MEAN_GAP_COLUMN])
        out.extend(rows)
    for model in comparators.values():
        aurocs = [_value(table, model, y, CASES, metric) for y in YEARS]
        row = {"Pool": "", "Model": model}
        row.update({f"AUROC, 5+ (%), {y}": a for y, a in zip(YEARS, aurocs)})
        row[MEAN_AUROC_COLUMN] = float(
            np.nanmean([aurocs[YEARS.index(y)] for y in BEST_MODEL_YEARS])
        )
        out.append(row)
    return pl.from_dicts(out, infer_schema_length=None)


def best_of_pool(consistency, pool, years):
    """The model of `pool` with the smallest mean gap from the best of the pool
    over `years` (the yearly gaps of consistency_table, in the AUROC for 5+
    cases). Two models with the same mean gap are an error: nothing here
    would decide between them."""
    gaps = (
        consistency.filter(pl.col("Pool") == pool)
        .select(
            "Model",
            pl.mean_horizontal(
                [f"Gap from the best of the pool, {y}" for y in years]
            ).alias("gap"),
        )
        .sort("gap", maintain_order=True)
    )
    if gaps.height > 1 and gaps["gap"][0] == gaps["gap"][1]:
        raise ValueError(
            f"{gaps['Model'][0]} and {gaps['Model'][1]} tie for the best of "
            f"the {pool} pool over {years[0]}-{years[-1]}"
        )
    return gaps["Model"][0]
