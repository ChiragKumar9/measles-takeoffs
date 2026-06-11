import os
import zipfile

# The bit of logic below this is to fix an error caused by the Census changing its API
# requirement on 12 May 2026. The Census package did not properly pass the API key to all
# requests, so we have to patch the elements that handle that logic.
from functools import lru_cache
from io import BytesIO

import polars as pl
import requests
from census import Census
from census.core import Client, supported_years
from tqdm.auto import tqdm


def _patched_field_type(self, field, year):
    url = self.definition_url % (year, self.dataset, field)
    resp = self.session.get(url, params={"key": self._key})

    type_map = {
        "fips-for": str,
        "fips-in": str,
        "int": float,
        "long": float,
        "float": float,
        "string": str,
    }
    if resp.status_code == 200:
        try:
            predicate_type = resp.json().get("predicateType", "string")
            return type_map.get(predicate_type, str)
        except Exception:
            return str
    return str


def _patched_tables(self, year=None):
    if year is None:
        year = self.default_year
    tables_url = self.groups_url % (year, self.dataset)
    resp = self.session.get(tables_url, params={"key": self._key})
    return resp.json()["groups"]


@supported_years()
def _patched_fields(self, year=None, flat=False):
    if year is None:
        year = self.default_year

    data = {}
    fields_url = self.definitions_url % (year, self.dataset)
    resp = self.session.get(fields_url, params={"key": self._key})
    obj = resp.json()

    if flat:
        for key, elem in obj["variables"].items():
            if key in ["for", "in"]:
                continue
            data[key] = "{}: {}".format(
                elem.get("concept", ""), elem.get("label", "")
            )
    else:
        data = obj["variables"]
        data.pop("for", None)
        data.pop("in", None)
    return data


Client._field_type = lru_cache(maxsize=1024)(_patched_field_type)
Client.tables = _patched_tables
Client.fields = _patched_fields


def browse_df(df, page_size=10, auto_exit=False):
    """Browse dataframe page by page"""
    total_rows = df.height
    total_pages = (total_rows + page_size - 1) // page_size

    current_page = 0

    while True:
        start_idx = current_page * page_size
        end_idx = min(start_idx + page_size, total_rows)

        print(f"\n--- Page {current_page + 1} of {total_pages} ---")
        print(f"Showing rows {start_idx + 1} to {end_idx} of {total_rows}")
        print(df.slice(start_idx, page_size))

        if current_page == total_pages - 1:
            print("\n[Last page reached]")
            if auto_exit:
                break

        command = (
            input(
                "\nEnter 'n' for next, 'p' for previous, 'q' to quit, or row number: "
            )
            .strip()
            .lower()
        )

        if command == "q":
            break
        elif command == "n" and current_page < total_pages - 1:
            current_page += 1
        elif command == "p" and current_page > 0:
            current_page -= 1
        elif command.isdigit():
            row_num = int(command) - 1
            if 0 <= row_num < total_rows:
                page = row_num // page_size
                current_page = page
        else:
            print("Invalid command. Please try again.")


def census_variables(year: int = 2023, survey: str = "acs5") -> pl.DataFrame:
    url = f"https://api.census.gov/data/{year}/acs/{survey}/variables.json"
    response = requests.get(url)
    data = response.json()

    # Convert to searchable DataFrame
    variables_df = pl.DataFrame(
        [
            {
                "code": code,
                "label": info.get("label", ""),
                "concept": info.get("concept", ""),
            }
            for code, info in data["variables"].items()
            if code not in ["for", "in", "ucgid"]
        ]
    )

    return variables_df


def fips_land_sizes() -> pl.DataFrame:
    # Download and process in memory
    url = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2020_Gazetteer/2020_Gaz_counties_national.zip"
    response = requests.get(url)

    # Create a ZipFile object from the downloaded content
    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        # We know the one file in here
        with zip_ref.open("2020_Gaz_counties_national.txt") as f:
            df = pl.read_csv(
                f, separator="\t", schema_overrides={"GEOID": pl.String}
            )

    return df


class CensusQuery:
    def __init__(self, key: str):
        self.census = Census(key)
        self.land_size = None
        # sourced from here
        # https://www.minneapolisfed.org/about-us/monetary-policy/inflation-calculator/consumer-price-index-1913-
        # https://www.bls.gov/cpi/tables/supplemental-files/historical-cpi-u-202308.pdf
        self.cpi_inflation = {
            2010: 0.016,
            2011: 0.032,
            2012: 0.021,
            2013: 0.015,
            2014: 0.016,
            2015: 0.001,
            2016: 0.013,
            2017: 0.021,
            2018: 0.024,
            2019: 0.018,
            2020: 0.012,
            2021: 0.047,
            2022: 0.08,
            2023: 0.041,
            2024: 0.029,
        }
        self.within_county_mobility = None

    def query_5_9_pop(self, year: list[int], fips: list[str]) -> pl.DataFrame:
        """Get population of children aged 5-9 and their school type for given
        FIPS codes and year."""

        assert len(fips) == len(year), (
            "FIPS and year lists must be the same length."
        )
        pop_data = list()
        # query census for each fips/year combo
        for fip, y in tqdm(zip(fips, year), total=len(fips)):
            pop_data.append(query(self.census, fip, y))
        pop_df = [pl.DataFrame(pop_datum) for pop_datum in pop_data]
        # make a median household income column for inflation adjusting below
        pop_df = [
            df.with_columns(median_household_income=pl.col("B19019_001E"))
            for df in pop_df
        ]
        # inflation adjust median household income to 2024 dollars
        for idx, y in enumerate(year):
            for yy in range(y + 1, 2024 + 1):
                pop_df[idx] = pop_df[idx].with_columns(
                    median_household_income=pl.col("median_household_income")
                    * (1 + self.cpi_inflation[yy])
                )
        pop_df = pl.concat(pop_df, how="diagonal_relaxed")
        pop_df = pop_df.with_columns(
            fips=pl.Series(fips),
            pop=pl.col("B01001_001E"),
            pop_under_5=pl.col("B01001_003E") + pl.col("B01001_027E"),
            pop_5_9=pl.col("B01001_004E") + pl.col("B01001_028E"),
            school_aged_population=(
                pl.col("B01001_003E")
                + pl.col("B01001_027E")  # Under 5 population
                + pl.col("B01001_004E")
                + pl.col("B01001_028E")  # 5-9 population
                + pl.col("B01001_005E")
                + pl.col("B01001_029E")  # 10-14 population
                + pl.col("B01001_006E")
                + pl.col("B01001_030E")  # 15-17 population
                + (pl.col("B01001_007E") + pl.col("B01001_031E"))
                / 2  # 18-19 population
            ),
            public_schooled=pl.col("B14003_005E") + pl.col("B14003_033E"),
            private_schooled=pl.col("B14003_014E") + pl.col("B14003_042E"),
            number_graduate_degree=(
                pl.col("B15003_023E")
                + pl.col("B15003_024E")
                + pl.col("B15003_025E")
            ),
            # this is the denominator for the education question
            pop_25_up=pl.col("B15003_001E"),
            pop_15_up=(
                pl.col("B15003_001E")
                + pl.col("B01001_006E")
                + pl.col("B01001_030E")  # 15-17 population
                + pl.col("B01001_007E")
                + pl.col("B01001_031E")  # 18-19 population
                + pl.col("B01001_008E")
                + pl.col("B01001_032E")  # 20 population
                + pl.col("B01001_009E")
                + pl.col("B01001_033E")  # 21 population
                + pl.col("B01001_010E")
                + pl.col("B01001_034E")  # 22-24 population
            ),
            fraction_white=pl.col("B02001_002E") / pl.col("B02001_001E"),
        ).select(
            [
                "fips",
                "pop",
                "pop_under_5",
                "pop_5_9",
                "school_aged_population",
                # school-age population
                "public_schooled",
                "private_schooled",
                "median_household_income",
                # num with graduate degree
                "number_graduate_degree",
                # 25 and over population
                "pop_25_up",
                "pop_15_up",
                "fraction_white",
            ]
        )
        return pop_df

    def query_land_size(self, fips: list[str]) -> pl.DataFrame:
        """Get land size in square km for given FIPS codes."""
        if self.land_size is None:
            self.land_size = fips_land_sizes()

        land_df = self.land_size
        land_df = land_df.with_columns(
            fips=pl.col("GEOID"),
            # land area is in square meters, convert to square km
            land_size_sq_km=(
                pl.col("ALAND").cast(pl.Float64, strict=False) / (1000 * 1000)
            ),
        ).select(["fips", "land_size_sq_km"])
        land_df_list = []
        for fip in fips:
            if len(fip) == 5:
                land_df_list.append(land_df.filter(pl.col("fips") == fip))
            else:
                temp = land_df.filter(pl.col("fips").str.head(2) == fip)
                temp = temp.with_columns(fips=pl.lit(fip))
                land_df_list.append(temp)
        land_df = pl.concat(land_df_list, how="diagonal_relaxed")
        # group by fips code and sum
        land_df = land_df.group_by("fips").agg(pl.sum("land_size_sq_km"))
        return land_df

    def query_within_county_mobility(
        self, year: list[int], fips: list[str]
    ) -> pl.DataFrame:
        """Get within-county mobility for given FIPS codes and year."""
        assert len(fips) == len(year), (
            "FIPS and year lists must be the same length."
        )
        if self.within_county_mobility is None:
            mobility = pl.read_csv(
                os.path.join("data", "bts_mobility_statistics.csv"),
                schema_overrides={"County FIPS": pl.String},
            )
            self.within_county_mobility = mobility
        mobility = self.within_county_mobility
        year_fips = pl.DataFrame({"year": year, "fips": fips})
        mobility = mobility.with_columns(
            year=pl.col("Date"), fips=pl.col("County FIPS")
        )
        mobility = mobility.join(year_fips, on=["year", "fips"], how="inner")

        mobility = mobility.with_columns(
            within_county_estimate=pl.col("Number of Trips <1")
            + pl.col("Number of Trips 1-3")
            + pl.col("Number of Trips 3-5")
            + pl.col("Number of Trips 5-10")
        )
        mobility = mobility.select(["year", "fips", "within_county_estimate"])
        return mobility


def query(census, fip, y):
    if y > 2024:
        # census data is not available beyond 2024 yet
        y = 2024
    elif y <= 2009:
        # the vars we want aren't available pre 2009
        y = 2010
    if fip[:2] == "09":
        # for some reason Connecticut data is not available after 2021
        y = min(y, 2021)
    if len(fip) < 5:
        fcn = census.acs5.state
        args = {"state_fips": fip[:2], "year": y}
    else:
        fcn = census.acs5.state_county
        args = {"state_fips": fip[:2], "county_fips": fip[2:], "year": y}
    # education data were not gathered pre 2012
    if y >= 2012:
        try:
            return fcn(
                fields=[
                    "B01001_001E",
                    "B01001_003E",
                    "B01001_004E",
                    "B01001_005E",
                    "B01001_006E",
                    "B01001_007E",
                    "B01001_008E",
                    "B01001_009E",
                    "B01001_010E",
                    "B01001_027E",
                    "B01001_028E",
                    "B01001_029E",
                    "B01001_030E",
                    "B01001_031E",
                    "B01001_032E",
                    "B01001_033E",
                    "B01001_034E",
                    "B14003_005E",
                    "B14003_014E",
                    "B14003_033E",
                    "B14003_042E",
                    "B19019_001E",
                    "B15003_023E",
                    "B15003_024E",
                    "B15003_025E",
                    "B15003_001E",
                    "B02001_001E",
                    "B02001_002E",
                ],
                **args,
            )[0]  # census queries return a list of one dict
        except Exception:
            print(
                f"Error querying census for FIPS {fip} and year {y}. Returning empty dict."
            )
            assert False
    else:
        # gather variables normally except education for 2012
        income_schooling = fcn(
            fields=[
                "B01001_001E",
                "B01001_003E",
                "B01001_004E",
                "B01001_005E",
                "B01001_006E",
                "B01001_007E",
                "B01001_008E",
                "B01001_009E",
                "B01001_010E",
                "B01001_027E",
                "B01001_028E",
                "B01001_029E",
                "B01001_030E",
                "B01001_031E",
                "B01001_032E",
                "B01001_033E",
                "B01001_034E",
                "B14003_005E",
                "B14003_014E",
                "B14003_033E",
                "B14003_042E",
                "B19019_001E",
                "B02001_001E",
                "B02001_002E",
            ],
            **args,
        )
        # education was only gathered from 2012 onwards
        args["year"] = 2012
        education = fcn(
            fields=[
                "B15003_023E",
                "B15003_024E",
                "B15003_025E",
                "B15003_001E",
            ],
            **args,
        )
        # merge the two queries
        # need the 0th element because census returns a list of dicts
        return {**income_schooling[0], **education[0]}
