import polars as pl

from outbreak_size.data_year import (
    county_data_year_of_outbreak,
    outbreak_year_of_county_data,
)


def test_the_county_data_of_an_outbreak_year_are_those_of_the_year_before():
    assert county_data_year_of_outbreak(2023) == 2022
    assert outbreak_year_of_county_data(2022) == 2023


def test_the_two_are_inverses():
    for year in range(2011, 2028):
        assert (
            outbreak_year_of_county_data(county_data_year_of_outbreak(year))
            == year
        )


def test_it_works_on_a_polars_column():
    df = pl.DataFrame({"year": [2021, 2025]})
    out = df.select(data_year=county_data_year_of_outbreak(pl.col("year")))
    assert out["data_year"].to_list() == [2020, 2024]
