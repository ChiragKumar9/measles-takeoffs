"""
The doses of the vaccine that an intervention needs: the children it adds to
those vaccinated, each vaccinated with the full series.

A county's kindergarten cohort is a fifth of its population aged 5-9 (the
ACS 5-year estimate, B01001: boys and girls of 5-9; the five single years of age
of the group are about the same size). The children of the cohort are assumed
not vaccinated unless the coverage says so, and a child that an intervention
adds to the vaccinated needs the whole series. Per county and year:

    doses = DOSES_PER_CHILD [doses per child]
            * cohort [children]
            * coverage increase [percentage points] / 100 [points per fraction]

A coverage that the intervention leaves lower than the scenario's baseline
adds no children: the increase is floored at 0 (it is not a negative number of
doses).
"""

import numpy as np

from outbreak_size.data_year import county_data_year_of_outbreak
from outbreak_size.utils import _cache_key

# the user's assumption: a child that was not vaccinated needs the full series
# of two doses of the MMR vaccine
DOSES_PER_CHILD = 2
# the ages of the ACS group of 5-9 (the cohort of kindergarten is one of them)
YEARS_OF_AGE_IN_GROUP = 5


def census_answer(county, data_year, cache):
    """The Census answer saved in `cache` (a CensusCache) for the county and
    the year of its data, as script 05 asked for it (None if there is none)."""
    return cache.get(_cache_key(county, data_year))


def cohort_from_answer(answer):
    """The kindergarten cohort (children) of a county from its Census answer:
    a fifth of the boys and girls aged 5-9 (CensusQuery.query_5_9_pop)."""
    group = answer["B01001_004E"] + answer["B01001_028E"]
    return group / YEARS_OF_AGE_IN_GROUP


def kindergarten_cohorts(fips, population, outbreak_year, cache):
    """
    The kindergarten cohort (children) of each county in `fips`: a fifth of the
    population aged 5-9 of the ACS estimates saved in `cache` (a CensusCache),
    in the vintage that script 05 used for the population of the projection
    rows of `outbreak_year` (the year of the county data of the outbreak year,
    capped at the latest ACS year, and Connecticut's at 2021: _cache_key). The
    total population of that vintage must be the county's `population`, which
    shows it is the vintage of the population of the table (a county with no
    answer or another total is an error).
    """
    data_year = county_data_year_of_outbreak(outbreak_year)
    cohorts = []
    for county, total in zip(fips, population):
        answer = census_answer(county, data_year, cache)
        if answer is None:
            raise KeyError(
                f"no Census answer saved for county {county}, {data_year}"
            )
        if abs(answer["B01001_001E"] - total) > 0.5:
            raise ValueError(
                f"county {county}: the saved population "
                f"{answer['B01001_001E']} is not {total}"
            )
        cohorts.append(cohort_from_answer(answer))
    return cohorts


def additional_doses(cohort, coverage_change_points):
    """
    The doses needed in a year: DOSES_PER_CHILD * cohort (children) *
    coverage change (percentage points, floored at 0) / 100. Arrays of one
    value per county.
    """
    return (
        DOSES_PER_CHILD
        * np.asarray(cohort, dtype=float)
        * np.maximum(np.asarray(coverage_change_points, dtype=float), 0.0)
        / 100.0
    )
