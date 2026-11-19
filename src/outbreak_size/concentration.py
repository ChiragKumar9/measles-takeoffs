"""
The case-concentration curves of analysis/figure_validation_and_risk_projections.py:
the share of the cases in the places at or below each risk percentile. A ranking
that sends the cases to the highest percentiles keeps the curve low until the
right edge; the diagonal is a ranking that tells nothing. The percentile is the
one of every risk percentile of the repo (outbreak_size.ranking: average the
draws, then rank; 0 for the lowest place, 100 for the highest).
"""

import numpy as np

from outbreak_size.importation import expected_case_draws
from outbreak_size.ranking import percentile_of_scores, risk_percentile

# the percentiles the curves are evaluated at
GRID = np.linspace(0, 100, 1001)
# the rankings of the validation panels, in the order of their legend
STATISTICAL = "Statistical model"
MECHANISTIC = "Mechanistic model"
COVERAGE = "Vaccine coverage alone"
POPULATION = "Population size alone"
IMPORTS = "Imports alone"
OBSERVED = "Observed cases (the best ranking)"
RANKINGS = [STATISTICAL, MECHANISTIC, COVERAGE, POPULATION, IMPORTS, OBSERVED]


def cumulative_share(percentiles, cases, grid=GRID):
    """The % of the cases that are in the places with a percentile at or below
    each value of `grid` (places with the same percentile are all in or all
    out, so the order of tied places does not matter)."""
    order = np.argsort(np.asarray(percentiles, dtype=float), kind="stable")
    sorted_percentiles = np.asarray(percentiles, dtype=float)[order]
    cumulative = np.concatenate(
        [[0.0], np.cumsum(np.asarray(cases, dtype=float)[order])]
    )
    total = cumulative[-1]
    if total <= 0:
        return np.zeros(len(grid))
    at_or_below = np.searchsorted(sorted_percentiles, grid, side="right")
    return 100 * cumulative[at_or_below] / total


def mean_curve(curves):
    """The curves of several years averaged with the same weight for each year
    (a year with many cases does not outweigh the others, as the scores of the
    table of the model scoring are averaged over the years)."""
    return np.mean(np.vstack(curves), axis=0)


def rankings(
    size_statistical,
    size_mechanistic,
    import_probability,
    expected_imports,
    coverage,
    population,
    cases,
):
    """
    The risk percentile of every place under each ranking of RANKINGS:
    the statistical and the mechanistic model rank the places by their expected
    cases (outbreak size times the probability of an import, plus the expected
    imports, averaged over the draws), the vaccine coverage (the lower, the
    higher the risk), the population and the expected imports each rank them by
    that number alone, and the observed cases rank them by what happened (the
    best ranking there is: with no ties, no other ranking with no ties has a
    lower curve at any percentile).

    The draws are [n_draws, n_places]; coverage (%), population and cases are
    [n_places]. Returns {ranking: [n_places] percentiles}.
    """
    return {
        STATISTICAL: risk_percentile(
            expected_case_draws(
                size_statistical, import_probability, expected_imports
            )
        ),
        MECHANISTIC: risk_percentile(
            expected_case_draws(
                size_mechanistic, import_probability, expected_imports
            )
        ),
        COVERAGE: percentile_of_scores(-np.asarray(coverage, dtype=float)),
        POPULATION: percentile_of_scores(np.asarray(population, dtype=float)),
        IMPORTS: percentile_of_scores(
            np.asarray(expected_imports, dtype=float).mean(axis=0)
        ),
        # places with the same cases are in any order (it changes no curve), so
        # the best ranking has no ties and is a bound for the others
        OBSERVED: percentile_of_scores(
            np.asarray(cases, dtype=float), method="ordinal"
        ),
    }


def validation_curves(percentiles_by_ranking, cases, grid=GRID):
    """{ranking: the cumulative share of the observed cases} of one year."""
    return {
        name: cumulative_share(percentiles, cases, grid)
        for name, percentiles in percentiles_by_ranking.items()
    }


def scenario_curve(case_draws, grid=GRID):
    """The curve of a scenario: the places ranked by the percentile of their
    expected cases, and the cases of each place its mean over the draws.
    case_draws: [n_draws, n_places] cases (with imports)."""
    case_draws = np.asarray(case_draws)
    return cumulative_share(
        risk_percentile(case_draws), case_draws.mean(axis=0), grid
    )
