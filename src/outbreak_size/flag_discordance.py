"""
Where two models flag different places: the statistical and the mechanistic model
of the table of county-level prediction scoring (analysis/table_model_performance_validation.py),
for the cases including imports, and the places that had many cases.

The input is one row per place and year (every place the table scores, see
place_scores), with the cases it had, the sociodemographic covariates of the place,
and the score of each model (columns STATISTICAL_SCORE and MECHANISTIC_SCORE, higher
is higher risk). A
model flags the highest `share` of the scores of a year, the rule of the table's
sensitivity and specificity (county_scores.top_share_metrics). The places with at
least `level` cases are put in four groups: flagged by both models, by the
statistical model only, by the mechanistic model only, or by neither.
"""

import numpy as np
import polars as pl

STATISTICAL_SCORE = "Statistical score"
MECHANISTIC_SCORE = "Mechanistic score"
STATISTICAL_FLAG = "Statistical flagged"
MECHANISTIC_FLAG = "Mechanistic flagged"
GROUPS = ["Both", "Statistical only", "Mechanistic only", "Neither"]
# the rule of the discordance figure: the highest SHARE of the scores of the places
# with a case flagged (as in the table's sensitivity), the places with at least LEVEL
# outbreak cases (the column OUTCOME) compared
SHARE = 0.10
LEVEL = 5
OUTCOME = "cases_outbreak"
# the places the figure draws (discordance_places), written by
# analysis/table_flag_discordance_places.py
PLACES_FILE_NAME = "flag_discordance_places.csv"
# the same for 2026 (the same two models, the headline models)
PLACES_FILE_NAME_2026 = "flag_discordance_places_2026.csv"
# the vaccination and sociodemographic covariates compared, with their labels
COVARIATES = {
    "estimated_mmr_cov": "MMR coverage",
    "nonmed_exempt_pct": "Nonmedical exemptions",
    "med_exempt_pct": "Medical exemptions",
    "fraction_under_5": "Fraction under 5",
    "fraction_school_aged": "Fraction school-aged",
    "avg_house_size": "Average household size",
    "fraction_public": "Fraction in public school",
    "fraction_white": "Fraction white",
    "fraction_graduate_degree": "Fraction with graduate degree",
    "log_population_density": "Population density (log)",
    "log_income": "Income (log)",
}


def top_share_flags(score, share):
    """
    Which places have the highest `share` of the scores: the ceil(share * n)
    highest. Places that tie at the cutoff cannot be split into a yes and a no,
    so a tie that crosses the cutoff is an error (the table's metrics split such
    places in proportion; a flag has no proportion).
    """
    score = np.asarray(score, dtype=float)
    if np.isnan(score).any():
        raise ValueError("a place has no score")
    flagged = int(np.ceil(share * len(score)))
    cutoff = np.sort(score)[::-1][flagged - 1]
    if (score >= cutoff).sum() != flagged:
        raise ValueError(
            f"{(score == cutoff).sum()} places tie at the cutoff of the top "
            f"{share:.0%}: the flags are not defined"
        )
    return score >= cutoff


def add_flags(places, share):
    """`places` with the flags of both models, year by year (the share of the
    places of each year)."""
    parts = []
    for year in places["year"].unique(maintain_order=True):
        rows = places.filter(pl.col("year") == year)
        parts.append(
            rows.with_columns(
                pl.Series(
                    STATISTICAL_FLAG,
                    top_share_flags(rows[STATISTICAL_SCORE], share),
                ),
                pl.Series(
                    MECHANISTIC_FLAG,
                    top_share_flags(rows[MECHANISTIC_SCORE], share),
                ),
            )
        )
    return pl.concat(parts)


def add_groups(places, level, outcome="cases"):
    """The places (from add_flags) with at least `level` cases in the column
    `outcome` ("cases", or "cases_outbreak" for the outbreak cases), each with its
    group: GROUPS."""
    return places.filter(pl.col(outcome) >= level).with_columns(
        group=pl.when(pl.col(STATISTICAL_FLAG) & pl.col(MECHANISTIC_FLAG))
        .then(pl.lit("Both"))
        .when(pl.col(STATISTICAL_FLAG))
        .then(pl.lit("Statistical only"))
        .when(pl.col(MECHANISTIC_FLAG))
        .then(pl.lit("Mechanistic only"))
        .otherwise(pl.lit("Neither"))
    )


def discordance_places(
    places, first_year, share=SHARE, level=LEVEL, outcome=OUTCOME
):
    """
    The places the discordance figure draws, and nothing else: of the places with a
    case in `first_year` or later, those with at least `level` cases in `outcome`
    that at least one model flagged (the highest `share` of the scores of the
    places with a case of their year), most cases first (the order of `places`
    breaks ties). `places` is every place the table scores, with the columns
    STATISTICAL_SCORE and MECHANISTIC_SCORE and the COVARIATES. Each place has its
    group, the percentile of each model's score among the places with a case of its
    year, and each covariate in its own units and as a percentile among ALL the
    places of its year (so the file holds no score of any other place, and no
    model's performance by year).
    """
    places = add_percentiles(places, list(COVARIATES))
    flagged = add_flags(
        places.filter((pl.col("cases") >= 1) & (pl.col("year") >= first_year)),
        share,
    )
    flagged = add_percentiles(flagged, [STATISTICAL_SCORE, MECHANISTIC_SCORE])
    grouped = (
        add_groups(flagged, level, outcome)
        .filter(pl.col("group") != "Neither")
        .sort(outcome, descending=True, maintain_order=True)
    )
    return grouped.select(
        "year",
        "fips",
        "state",
        "county",
        outcome,
        "group",
        f"{STATISTICAL_SCORE} percentile",
        f"{MECHANISTIC_SCORE} percentile",
        *COVARIATES,
        *[f"{c} percentile" for c in COVARIATES],
    )


def group_counts(grouped):
    """The number of places in each group in each year, and in all the years
    ("All years"), as rows of Year, Group and Places (a group with no place has
    0), and the share of the places that only one of the models flags."""
    years = [str(y) for y in sorted(grouped["year"].unique())]
    rows = []
    for label, part in [
        (y, grouped.filter(pl.col("year") == int(y))) for y in years
    ] + [("All years", grouped)]:
        counts = {g: int((part["group"] == g).sum()) for g in GROUPS}
        total = sum(counts.values())
        disagree = counts["Statistical only"] + counts["Mechanistic only"]
        for g in GROUPS:
            rows.append(
                {
                    "Year": label,
                    "Group": g,
                    "Places": counts[g],
                    "Share flagged by one model only": disagree / total,
                }
            )
    return pl.DataFrame(rows)


def add_percentiles(places, columns):
    """The percentile rank (0 to 100, ties at their average rank) of each
    covariate among all the places of the same year, in new columns named
    "<covariate> percentile". A place with no value has no percentile."""
    return places.with_columns(
        [
            (
                100
                * pl.col(c).rank("average").over("year")
                / pl.col(c).count().over("year")
            ).alias(f"{c} percentile")
            for c in columns
        ]
    )


def group_percentiles(grouped, columns, groups):
    """The number of places, the 25th, 50th and 75th percentile of the
    percentile ranks of each covariate for each of `groups`."""
    rows = []
    for column in columns:
        for group in groups:
            values = grouped.filter(pl.col("group") == group)[
                f"{column} percentile"
            ].drop_nulls()
            q1, median, q3 = (
                np.percentile(values.to_numpy(), [25, 50, 75])
                if len(values)
                else (np.nan,) * 3
            )
            rows.append(
                {
                    "Covariate": column,
                    "Group": group,
                    "Places": len(values),
                    "Q1": float(q1),
                    "Median": float(median),
                    "Q3": float(q3),
                }
            )
    return pl.DataFrame(rows)


def different_covariates(summary, min_gap):
    """The covariates (of group_percentiles' `summary`) on which the places only the
    statistical model flagged and the places only the mechanistic model flagged
    differ: the medians of their percentile ranks are at least `min_gap` points
    apart. In the order of the summary; a covariate with no place in either group
    is left out."""
    medians = summary.pivot(
        on="Group", index="Covariate", values="Median", maintain_order=True
    )
    gap = (medians["Statistical only"] - medians["Mechanistic only"]).abs()
    # a missing median (a group with no place) is NaN, which polars ranks above any number
    return medians.filter((gap >= min_gap) & gap.is_not_nan())[
        "Covariate"
    ].to_list()


def discordance_rates(counts):
    """
    From group_counts' `counts`, for each year and all years: the places (of the
    places with enough cases that at least one model flagged) the statistical
    model flagged and the share of them the mechanistic model did not flag, the
    same for the mechanistic model, and the places either flagged and the
    share of them flagged by one model only. A share with no place is NaN.
    """
    rows = []
    for year in counts["Year"].unique(maintain_order=True):
        n = {
            g: int(
                counts.filter(
                    (pl.col("Year") == year) & (pl.col("Group") == g)
                )["Places"].item()
            )
            for g in GROUPS
        }
        stat = n["Both"] + n["Statistical only"]
        mech = n["Both"] + n["Mechanistic only"]
        either = n["Both"] + n["Statistical only"] + n["Mechanistic only"]
        nan = float("nan")
        rows.append(
            {
                "Year": year,
                "Flagged by statistical": stat,
                "Of them not flagged by mechanistic": (
                    n["Statistical only"] / stat if stat else nan
                ),
                "Flagged by mechanistic": mech,
                "Of them not flagged by statistical": (
                    n["Mechanistic only"] / mech if mech else nan
                ),
                "Flagged by either": either,
                "Of them flagged by one only": (
                    (n["Statistical only"] + n["Mechanistic only"]) / either
                    if either
                    else nan
                ),
            }
        )
    return pl.DataFrame(rows)


def group_medians(grouped, columns, groups):
    """The median of each covariate (in its own units, not as a percentile) of
    the places of each of `groups`, over the places with a value: rows of
    Covariate, Group and Median (NaN if no place has a value)."""
    rows = []
    for column in columns:
        for group in groups:
            values = grouped.filter(pl.col("group") == group)[
                column
            ].drop_nulls()
            rows.append(
                {
                    "Covariate": column,
                    "Group": group,
                    "Median": float(values.median())  # type: ignore
                    if len(values)
                    else float("nan"),
                }
            )
    return pl.DataFrame(rows)


def covariate_gaps(summary, raw_medians, min_gap):
    """
    One row per covariate: the medians of the places only the statistical
    model flagged and only the mechanistic model flagged, as percentiles
    (`summary`, from group_percentiles) and in the covariate's own units
    (`raw_medians`, from group_medians), the gap between the two (statistical
    minus mechanistic), and whether the percentile gap is more than `min_gap`
    points in either direction. The covariates in the order of `summary`.
    """
    stat, mech = "Statistical only", "Mechanistic only"

    def medians(table, suffix):
        wide = table.pivot(
            on="Group", index="Covariate", values="Median", maintain_order=True
        )
        return wide.select(
            "Covariate",
            pl.col(stat).alias(f"Statistical only, {suffix}"),
            pl.col(mech).alias(f"Mechanistic only, {suffix}"),
        )

    table = medians(summary, "percentile").join(
        medians(raw_medians, "own units"),
        on="Covariate",
        how="left",
        maintain_order="left",
    )
    gap = pl.col("Statistical only, percentile") - pl.col(
        "Mechanistic only, percentile"
    )
    return table.with_columns(
        gap.alias("Percentile gap"),
        (
            pl.col("Statistical only, own units")
            - pl.col("Mechanistic only, own units")
        ).alias("Gap in own units"),
        (gap.abs() > min_gap).alias(f"Gap over {min_gap} points"),
    )
