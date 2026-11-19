"""
The prior-cases benchmark of the scoring table: it ranks the places of a
validation year by the cases they had in earlier years, with no model and no
imports. A place's score in year t is the sum of its reported measles cases
(all of them, imported and local) over the years first_year to t - 1, so the
year predicted is never in its own score. A place with no row in the history
had no reported case: its score is 0 (the history holds only the county-years
with at least one case). Most places score 0 and tie, and the scoring handles
ties without depending on the order of the places (see county_scores).
"""

import numpy as np
import polars as pl

FIRST_YEAR = 2011


def place_key(fips: pl.Expr, state: pl.Expr, county: pl.Expr) -> pl.Expr:
    """The key of a place across years: its county fips, or its state and name
    for a place with no 5-digit county fips (the keying of the scoring table
    script, without the year)."""
    return (
        pl.when(fips.str.len_chars() == 5)
        .then(fips)
        .otherwise(pl.concat_str([state, county], separator="|"))
    )


def prior_cases_score(
    history: pl.DataFrame,
    places: pl.Series,
    year: int,
    first_year: int = FIRST_YEAR,
) -> np.ndarray:
    """
    The prior-cases score of each place of `places` (their place_key values, in
    the order returned) for the validation year `year`.

    history: one row per place and year with columns "place", "year" and
    "cases"; a place and year that appear twice would be counted twice, so
    that is an error.
    """
    if history.height != history.unique(["place", "year"]).height:
        raise ValueError("the case history has a place and year twice")
    prior = (
        history.filter(
            (pl.col("year") >= first_year) & (pl.col("year") < year)
        )
        .group_by("place")
        .agg(pl.col("cases").sum())
    )
    scores = pl.DataFrame({"place": places}).join(
        prior, on="place", how="left", maintain_order="left"
    )
    return scores["cases"].fill_null(0).to_numpy().astype(float)
