"""
The scores of every model for every place and year, as the table of county-level
prediction scoring (analysis/table_model_performance_validation.py) used them.
One row per place and year, in the order the table scores them, with the cases of
the place, its covariates, and one column per model and condition (the scores of
the outbreak cases are those of all the places of the year; the table scores only
the places with at least one case of any kind). The table script keeps it in
memory; the one file made from it is that of the discordance figure
(flag_discordance.discordance_places).
"""

import numpy as np
import polars as pl

# the columns that identify a place and say what happened in it
PLACE_COLUMNS = [
    "year",
    "fips",
    "state",
    "county",
    "pop",
    "cases",
    "cases_outbreak",
]


def score_column(condition, model):
    """The name of the column with the scores of `model` in `condition` (a
    county_scores.CONDITIONS label)."""
    return f"{condition} | {model}"


def place_score_table(places_by_year, scores, conditions, other_columns):
    """
    The table of every place and year.

    places_by_year: {year: the places of the year (a frame with PLACE_COLUMNS,
    but `year`, and `other_columns`: the covariates and the outcomes of the "any
    window" metrics), in the order they are scored};
    scores: {(model, year): (the scores of the cases of every place, the scores
    of the outbreak cases of every place)}; conditions: the two condition labels.
    Every model must have a score for every place of every year.
    """
    models = list(dict.fromkeys(model for model, _ in scores))
    years = list(places_by_year)
    parts = []
    for year in years:
        places = places_by_year[year]
        columns = {
            "year": np.full(places.height, year),
            **{c: places[c] for c in PLACE_COLUMNS[1:-2]},
            # the table script builds the outbreak cases with a string replace, so
            # they may be text
            **{c: places[c].cast(pl.Int64) for c in PLACE_COLUMNS[-2:]},
            **{c: places[c] for c in other_columns},
        }
        for model in models:
            if (model, year) not in scores:
                raise ValueError(f"{model} has no scores in {year}")
            for condition, score in zip(conditions, scores[(model, year)]):
                score = np.asarray(score, dtype=float)
                if len(score) != places.height:
                    raise ValueError(
                        f"{model} has {len(score)} scores in {year}, "
                        f"for {places.height} places"
                    )
                columns[score_column(condition, model)] = score
        parts.append(pl.DataFrame(columns))
    return pl.concat(parts)


# the forecast and the scores of the whole distribution of every place (one row per
# model, condition, year and place), which the intervals of the table of scores need
VALUE_COLUMNS = (
    "Forecast",
    "WIS (cases)",
    "CRPS (cases)",
    "CRPS of log(1 + cases)",
    "CRPS of the cases above 10 (cases)",
    "95% interval coverage (%)",
)


def place_value_table(place_values, introduced_by_year, conditions):
    """
    The forecast and the scores of the distribution of each place, in long form.

    place_values: {(model, year, condition): {"forecast": [n] or None,
    "distribution": {column: [n]} or None}} (county_scores.ScoreBook); a model
    without draws in a condition has no rows. introduced_by_year: {year: [n places
    of the year] whether the place had a case}, because the places of the outbreak
    condition are those. Row is the position of the place among all the places of
    its year (the order of place_score_table).
    """
    parts = []
    for (model, year, condition), values in place_values.items():
        if values["forecast"] is None:
            continue
        n = len(values["forecast"])
        outbreak = condition == conditions[1]
        rows = (
            np.flatnonzero(introduced_by_year[year])
            if outbreak
            else np.arange(n)
        )
        if len(rows) != n:
            raise ValueError(
                f"{model}, {year}, {condition}: {n} forecasts for {len(rows)} places"
            )
        parts.append(
            pl.DataFrame(
                {
                    "Model": [model] * n,
                    "Condition": [condition] * n,
                    "year": np.full(n, year),
                    "row": rows,
                    "Forecast": np.asarray(values["forecast"], dtype=float),
                    **{
                        c: np.asarray(v, dtype=float)
                        for c, v in values["distribution"].items()
                    },
                }
            )
        )
    return pl.concat(parts)
