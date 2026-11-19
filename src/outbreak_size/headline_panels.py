"""
The numbers behind the validation and 2026 panels of
analysis/figure_validation_and_risk_projections.py: the performance of the
headline statistical model and of the mechanistic model (read from the tables of
the scoring), the national cases of the year that the models predict when the
imports are conditioned on an earlier year, and the bucket of every place by its
observed cases.
"""

import numpy as np
import polars as pl

from outbreak_size.county_scores import CONDITIONS
from outbreak_size.importation import draw_imports, total_case_draws

# the metrics of the performance panel, in the order of its x axis, and the
# column of the scoring tables of each (the flags are of the top 20% of the
# places; `{level}` is the number of cases that makes a place an outcome)
PERFORMANCE_METRICS = {
    "AUC": "AUROC, {level}+ (%)",
    "Sensitivity": "Sensitivity, {level}+, top 20% (%)",
    "Specificity": "Specificity, {level}+, top 20% (%)",
    "AUPRC": "AUPRC, {level}+ (%)",
    "PPV": "PPV, {level}+, top 20% (%)",
    "NPV": "NPV, {level}+, top 20% (%)",
}
# the label of each condition of the scoring tables
CONDITION_LABELS = dict(zip(CONDITIONS, ("All cases", "Outbreak cases")))
# the observed cases that put a place in each bucket of the risk panel: none,
# 1 to 9, 10 or more
BUCKET_LABELS = ("0", "1–9", "≥10")


def performance_values(
    earlier,
    later,
    models,
    level,
    earlier_window,
    later_year,
    places="all places",
):
    """
    The performance of each model in each condition and period: one row per
    metric, model, condition and period, with the value and the lower and upper
    end of its interval (NaN when the table has none).

    earlier: the table S1 (scores averaged over windows of years, with their
    intervals); later: the table of the scoring of one later year (the rows of
    `places`, if the table has that column, and of that year); models: {label:
    the name of the model in the tables}; level: the cases that make a place
    an outcome (10 for 10+ cases). Every value must be in exactly one row of
    its table.
    """
    sources = (
        (
            earlier_window,
            earlier.filter(
                pl.col("Evaluation year").cast(pl.String) == earlier_window
            ),
        ),
        (
            str(later_year),
            later.filter(
                (pl.col("Evaluation year").cast(pl.String) == str(later_year))
                & (
                    pl.col("places") == places
                    if "places" in later.columns
                    else pl.lit(True)
                )
            ),
        ),
    )
    rows = []
    for period, table in sources:
        for label, name in models.items():
            for condition, condition_label in CONDITION_LABELS.items():
                row = table.filter(
                    (pl.col("Model") == name)
                    & (pl.col("Condition") == condition)
                )
                if row.height != 1:
                    raise ValueError(
                        f"{row.height} rows for {name}, {condition}, {period} "
                        "in the table of the scores: expected one"
                    )
                for metric, template in PERFORMANCE_METRICS.items():
                    column = template.format(level=level)
                    value = row[column][0]
                    if value is None or np.isnan(value):
                        raise ValueError(
                            f"{column} of {name}, {condition}, {period} has no value"
                        )
                    rows.append(
                        {
                            "metric": metric,
                            "model": label,
                            "condition": condition_label,
                            "period": period,
                            "value": float(value),
                            **{
                                end: float(row[f"{column}, {end}"][0])
                                if f"{column}, {end}" in row.columns
                                and row[f"{column}, {end}"][0] is not None
                                else np.nan
                                for end in ("lower", "upper")
                            },
                        }
                    )
    return pl.DataFrame(rows)


def conditioned_national_totals(size, expected_imports, scaler, rng):
    """
    Draws of the cases of a year in all the places together, with the imports
    conditioned on a level of imports: each place has Poisson imports with its
    expected imports times `scaler`, and in a draw with at least one import its
    cases are the imports and the outbreak that follows (importation.
    total_case_draws).

    size: [n_draws, n_places] outbreak size draws (cases if there is an
    outbreak); expected_imports: [n_draws, n_places] expected imports (cases);
    scaler: the ratio of the imports to condition on to the expected imports (no
    unit); rng: a numpy Generator. Returns [n_draws] cases.
    """
    imports = draw_imports(np.asarray(expected_imports) * scaler, rng)
    return total_case_draws(size, imports).sum(axis=1)


def case_bucket(cases):
    """The bucket (0, 1 or 2: see BUCKET_LABELS) of each place from its observed
    cases."""
    cases = np.asarray(cases)
    return np.where(cases >= 10, 2, np.where(cases >= 1, 1, 0))
