"""
The year of the county data that the outbreaks of a year are matched with.

Everything known about a county that is used to explain its outbreaks of a
year (its vaccination coverage and exemptions, its neighbours' coverage, the
census: population, income, education, schooling, ...) is that of the year
BEFORE: the school year that starts in the year before the outbreak year (the
"year" of the vaccination data), and the ACS 5-year estimates of that year.
Ask for the data of an outbreak year through these functions, so that the
rule is in one place.

Time-invariant data (land size, household size) have no year to ask for.
"""


def county_data_year_of_outbreak(outbreak_year):
    """The year of the county data (vaccination school year, census year) used
    for the outbreaks of `outbreak_year` (a year or a polars expression)."""
    return outbreak_year - 1


def outbreak_year_of_county_data(data_year):
    """The outbreak year whose county data are those of `data_year` (the
    inverse of county_data_year_of_outbreak)."""
    return data_year + 1
