"""Population variables of counties (outbreak_size.population)."""

import polars as pl
import pytest

from outbreak_size.population import add_census_ratios


def counts():
    return pl.DataFrame(
        {
            "fips": ["01001"],
            "pop": [1000.0],
            "land_size_sq_km": [10.0],
            "pop_5_9": [60.0],
            "public_schooled": [30.0],
            "private_schooled": [15.0],
            "pop_25_up": [500.0],
            "number_graduate_degree": [50.0],
            "school_aged_population": [250.0],
            "pop_under_5": [60.0],
            "median_household_income": [100000.0],
        }
    )


def test_ratios_are_taken_against_the_right_denominators():
    out = add_census_ratios(counts()).row(0, named=True)
    assert out["pop_density"] == pytest.approx(100.0)  # people per sq km
    # schooling is of the children aged 5-9
    assert out["fraction_public"] == pytest.approx(30 / 60)
    assert out["fraction_private"] == pytest.approx(15 / 60)
    # the degree is of the people aged 25 and over
    assert out["fraction_graduate_degree"] == pytest.approx(50 / 500)
    # the age groups are shares of the total population
    assert out["fraction_school_aged"] == pytest.approx(0.25)
    assert out["fraction_under_5"] == pytest.approx(0.06)
    assert out["fraction_5_9"] == pytest.approx(0.06)


def test_logs_are_base_ten():
    out = add_census_ratios(counts()).row(0, named=True)
    assert out["log_income"] == pytest.approx(5.0)
    assert out["log_population"] == pytest.approx(3.0)
    assert out["log_population_density"] == pytest.approx(2.0)


def test_the_input_columns_are_kept():
    out = add_census_ratios(counts())
    assert set(counts().columns) <= set(out.columns)
    assert out.height == 1
