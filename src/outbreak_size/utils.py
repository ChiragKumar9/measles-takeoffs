import hashlib
import inspect
import json
import os
import sqlite3
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

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


@lru_cache(maxsize=1)
def fips_land_sizes() -> pl.DataFrame:
    # Downloaded once per run, not once per CensusQuery (00 makes one for the
    # counties and one for each region, 05 one for each year); a failed
    # download is not kept. Callers only derive new frames from the result.
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


# where the answers of the Census are kept between runs (outputs/ is where the
# results of the code go; delete the file to ask the Census again)
CENSUS_CACHE_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
    "outputs",
    "census_cache.sqlite",
)

# how many requests to the Census are in flight at once (the scripts that ask
# for the populations of many counties pass this to CensusQuery)
CENSUS_WORKERS = 8


class CensusCache:
    """
    The answers of the Census to `query`, kept in a SQLite file so that a
    later request for the same county and year (in this run or the next one)
    is not sent again, and a run that stops (a lost connection) can be started
    again and goes on from the answers it already has. An answer is saved
    when it arrives; a failed query saves nothing.
    """

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS answers "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.db.commit()

    def get(self, key: str):
        row = self.db.execute(
            "SELECT value FROM answers WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else json.loads(row[0])

    def put(self, key: str, value) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO answers (key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
        self.db.commit()


class CensusQuery:
    def __init__(
        self, key: str, cache_path: str | None = None, workers: int = 1
    ):
        assert workers >= 1, "workers must be at least 1"
        self.census = Census(key)
        # the answers of the Census, kept between runs (None: ask every time)
        self.cache = None if cache_path is None else CensusCache(cache_path)
        # how many requests are in flight at once
        self.workers = workers
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

    def query_5_9_pop(self, year: list[int], fips: list[str]) -> pl.DataFrame:
        """Get population of children aged 5-9 and their school type for given
        FIPS codes and year."""

        assert len(fips) == len(year), (
            "FIPS and year lists must be the same length."
        )
        if self.cache is None and self.workers == 1:
            # query census for each fips/year combo, one after the other
            pop_data = list()
            for fip, y in tqdm(zip(fips, year), total=len(fips)):
                pop_data.append(query(self.census, fip, y))
        else:
            pop_data = fetch_answers(
                self.census, self.cache, fips, year, workers=self.workers
            )
        pop_df = [pl.DataFrame(pop_datum) for pop_datum in pop_data]
        # make a median household income column for inflation adjusting below
        pop_df = [
            df.with_columns(median_household_income=pl.col("B19019_001E"))
            for df in pop_df
        ]
        # inflation adjust median household income to the dollars of the
        # latest ACS year, from the year the estimates are really of
        # (_census_year: a county whose estimates stop earlier, such as
        # Connecticut's, is inflated from that year)
        for idx, (fip, y) in enumerate(zip(fips, year)):
            for yy in range(_census_year(fip, y) + 1, LATEST_ACS5_YEAR + 1):
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
            # master's, professional and doctorate degrees of men and women
            number_graduate_degree=(
                pl.col("B15002_016E")
                + pl.col("B15002_017E")
                + pl.col("B15002_018E")
                + pl.col("B15002_033E")
                + pl.col("B15002_034E")
                + pl.col("B15002_035E")
            ),
            # this is the denominator for the education question
            pop_25_up=pl.col("B15002_001E"),
            pop_15_up=(
                pl.col("B15002_001E")
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
        # group by fips code and sum; the rows are in the order of `fips` (a
        # group_by does not keep the order of the rows unless it is asked to)
        land_df = land_df.group_by("fips", maintain_order=True).agg(
            pl.sum("land_size_sq_km")
        )
        return land_df


# the latest year of the ACS 5-year estimates that is released (the 2021-2025
# estimates are due in December 2026; the census package must support the year
# too, and the cpi_inflation of the CensusQuery must reach it)
LATEST_ACS5_YEAR = 2024


def _census_year(fip, y):
    """The year of the ACS 5-year estimates that is asked for the place `fip`
    when the year `y` is wanted: the latest year there are data for."""
    if y > LATEST_ACS5_YEAR:
        y = LATEST_ACS5_YEAR
    elif y <= 2009:
        # the vars we want aren't available pre 2009
        y = 2010
    if fip[:2] == "09":
        # for some reason Connecticut data is not available after 2021
        y = min(y, 2021)
    return y


@lru_cache(maxsize=1)
def _query_version():
    """Changes when `query`, `_census_year` or ACS_FIELDS change (the fields asked, the
    year moved), so that answers saved by the code of before are not used."""
    source = (
        inspect.getsource(query)
        + inspect.getsource(_census_year)
        + repr(ACS_FIELDS)
    )
    return hashlib.sha1(source.encode()).hexdigest()[:12]


def _cache_key(fip, y):
    """The key of the answer for the place `fip` and the year `y` in the
    cache: the year that is really asked (_census_year), so the years after
    2024 share the answer of 2024."""
    return f"{_query_version()}|{fip}|{_census_year(fip, y)}"


def cached_query(cache: CensusCache, census, fip, y):
    """`query(census, fip, y)`, asking the Census only if `cache` has no
    answer yet."""
    key = _cache_key(fip, y)
    answer = cache.get(key)
    if answer is None:
        answer = query(census, fip, y)
        cache.put(key, answer)
    return answer


def fetch_answers(
    census,
    cache: CensusCache | None,
    fips,
    year,
    workers: int = CENSUS_WORKERS,
    attempts: int = 3,
    wait: float = 2.0,
):
    """
    The answer of `query` for each (fip, year) asked, in the order asked, with
    up to `workers` requests in flight at once.

    Each distinct place and year (the year that is really asked) is requested
    once, and not at all if `cache` has it. The cache is only used here, in the
    thread that called, and each answer is saved when it arrives, so a run
    that stops (a failed query, a lost connection) keeps the answers it got
    and a new run goes on from them. A failed query is tried `attempts` times
    (waiting `wait`, 2 * `wait`, ... seconds in between); if it still fails
    the error is raised, as `query` raises it, and no more requests are sent.
    """
    keys = [(fip, _census_year(fip, y)) for fip, y in zip(fips, year)]
    answers = {}
    missing = []
    for key in dict.fromkeys(keys):
        answer = None if cache is None else cache.get(_cache_key(*key))
        if answer is None:
            missing.append(key)
        else:
            answers[key] = answer

    def ask(key):
        for attempt in range(attempts):
            try:
                return query(census, *key)
            except AssertionError:
                if attempt == attempts - 1:
                    raise
                time.sleep(wait * (attempt + 1))

    with tqdm(total=len(missing)) as bar:

        def keep(key, answer):
            answers[key] = answer
            if cache is not None:
                cache.put(_cache_key(*key), answer)
            bar.update()

        # The census client looks up the type of each field the first time
        # a year is asked: ask one place of each year alone first, so that
        # the threads do not all do the lookups at once
        first = {}
        for key in missing:
            first.setdefault(key[1], key)
        for key in first.values():
            keep(key, ask(key))
        rest = [key for key in missing if key not in answers]

        pool = ThreadPoolExecutor(max_workers=workers)
        futures = {pool.submit(ask, key): key for key in rest}
        try:
            for future in as_completed(futures):
                keep(futures[future], future.result())
        except BaseException:
            for future in futures:
                future.cancel()
            raise
        finally:
            # let the requests in flight finish, and keep their answers
            pool.shutdown(wait=True)
            for future, key in futures.items():
                if (
                    future.done()
                    and not future.cancelled()
                    and future.exception() is None
                    and key not in answers
                ):
                    keep(key, future.result())
    return [answers[key] for key in keys]


# the fields asked of the ACS 5-year estimates (tables B01001: sex by age;
# B14003: school enrollment by type; B19019: median household income; B15002:
# sex by educational attainment, 25 and over, which exists in every year from
# 2010 - its later counterpart B15003 gives the same counts from 2012;
# B02001: race)
ACS_FIELDS = [
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
    "B15002_001E",
    "B15002_016E",
    "B15002_017E",
    "B15002_018E",
    "B15002_033E",
    "B15002_034E",
    "B15002_035E",
    "B02001_001E",
    "B02001_002E",
]


def query(census, fip, y):
    y = _census_year(fip, y)
    if len(fip) < 5:
        fcn = census.acs5.state
        args = {"state_fips": fip[:2], "year": y}
    else:
        fcn = census.acs5.state_county
        args = {"state_fips": fip[:2], "county_fips": fip[2:], "year": y}
    try:
        return fcn(fields=ACS_FIELDS, **args)[
            0
        ]  # census queries return a list of one dict
    except Exception:
        print(
            f"Error querying census for FIPS {fip} and year {y}. Returning empty dict."
        )
        assert False
