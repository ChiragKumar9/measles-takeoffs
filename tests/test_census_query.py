"""Census queries of the county populations (outbreak_size.utils), run against
a stand-in for the census client so that no request leaves the machine."""

import polars as pl
import pytest

from outbreak_size import utils

EDUCATION_FIELDS = [
    "B15002_001E",
    "B15002_016E",
    "B15002_017E",
    "B15002_018E",
    "B15002_033E",
    "B15002_034E",
    "B15002_035E",
]


class FakeAcs5:
    """Records each call. Every field of the first (population and income)
    query is answered 1.0 and every education field 2.0, so a merged answer
    shows which query each value came from."""

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def _answer(self, fields):
        if self.fail:
            raise RuntimeError("no such data")
        return [{f: 2.0 if f in EDUCATION_FIELDS else 1.0 for f in fields}]

    def state(self, fields, state_fips, year):
        self.calls.append(("state", fields, state_fips, None, year))
        return self._answer(fields)

    def state_county(self, fields, state_fips, county_fips, year):
        self.calls.append(("county", fields, state_fips, county_fips, year))
        return self._answer(fields)


class FakeCensus:
    def __init__(self, **kwargs):
        self.acs5 = FakeAcs5(**kwargs)


def years_asked(fip, year):
    census = FakeCensus()
    utils.query(census, fip, year)
    return [call[4] for call in census.acs5.calls]


# --- which year is asked for -------------------------------------------------


@pytest.mark.parametrize(
    "year, asked",
    [
        (2030, 2024),  # nothing beyond 2024 is available yet
        (2025, 2024),
        (2024, 2024),
        (2015, 2015),
        (2012, 2012),
        (2009, 2010),  # the variables are not available before 2010
        (2000, 2010),
    ],
)
def test_the_year_is_moved_into_the_years_the_census_has(year, asked):
    assert years_asked("01001", year)[0] == asked


@pytest.mark.parametrize("fip", ["09001", "09"])
def test_connecticut_stops_at_2021(fip):
    assert years_asked(fip, 2023) == [2021]
    assert years_asked(fip, 2019) == [2019]


# --- state or county ---------------------------------------------------------


def test_a_county_is_asked_with_its_state_and_county_codes():
    census = FakeCensus()
    utils.query(census, "06037", 2020)
    ((kind, _, state, county, year),) = census.acs5.calls
    assert (kind, state, county, year) == ("county", "06", "037", 2020)


def test_a_state_code_asks_for_the_whole_state():
    census = FakeCensus()
    utils.query(census, "06", 2020)
    ((kind, _, state, county, year),) = census.acs5.calls
    assert (kind, state, county, year) == ("state", "06", None, 2020)


# --- the fields asked: the same one query in every year ----------------------


@pytest.mark.parametrize("year", [2010, 2011, 2012, 2020])
def test_one_query_gets_the_population_and_the_education_fields_in_every_year(
    year,
):
    """The education table B15002 exists from 2010, so no year needs the
    education of another (later) year."""
    census = FakeCensus()
    out = utils.query(census, "06037", year)
    (call,) = census.acs5.calls
    assert call[4] == max(year, 2010)
    assert set(EDUCATION_FIELDS) <= set(call[1])
    assert "B01001_001E" in call[1] and "B19019_001E" in call[1]
    assert out["B15002_001E"] == 2.0 and out["B01001_001E"] == 1.0


def test_no_field_comes_from_the_education_table_that_starts_in_2012():
    assert not [f for f in utils.ACS_FIELDS if f.startswith("B15003")]


def test_a_failed_query_stops_the_run():
    with pytest.raises(AssertionError):
        utils.query(FakeCensus(fail=True), "06037", 2020)


# --- population variables built from the census fields -----------------------


class TableCensus:
    """Answers each field with the number in VALUES (0 for the others)."""

    VALUES = {
        "B01001_001E": 1000.0,  # total population
        "B01001_003E": 50.0,  # males under 5
        "B01001_027E": 40.0,  # females under 5
        "B01001_004E": 60.0,  # 5-9
        "B01001_028E": 50.0,
        "B01001_005E": 70.0,  # 10-14
        "B01001_029E": 60.0,
        "B01001_006E": 30.0,  # 15-17
        "B01001_030E": 20.0,
        "B01001_007E": 10.0,  # 18-19
        "B01001_031E": 6.0,
        "B01001_008E": 4.0,  # 20
        "B01001_032E": 2.0,
        "B01001_009E": 3.0,  # 21
        "B01001_033E": 1.0,
        "B01001_010E": 8.0,  # 22-24
        "B01001_034E": 5.0,
        "B14003_005E": 20.0,  # enrolled in public school
        "B14003_033E": 10.0,
        "B14003_014E": 5.0,  # enrolled in private school
        "B14003_042E": 3.0,
        "B19019_001E": 50000.0,  # median household income
        "B15002_016E": 4.0,  # graduate degrees of men
        "B15002_017E": 2.0,
        "B15002_018E": 1.0,
        "B15002_033E": 3.0,  # and of women
        "B15002_034E": 3.0,
        "B15002_035E": 2.0,
        "B15002_001E": 500.0,  # population 25 and over
        "B02001_001E": 1000.0,
        "B02001_002E": 700.0,  # white
    }

    def __init__(self):
        self.acs5 = self

    def state_county(self, fields, state_fips, county_fips, year):
        return [{f: self.VALUES.get(f, 0.0) for f in fields}]


def population(year):
    census_query = utils.CensusQuery("a-key")
    census_query.census = TableCensus()  # type: ignore
    return census_query.query_5_9_pop([year], ["06037"]).row(0, named=True)


def test_population_variables_are_sums_of_the_age_and_sex_counts():
    row = population(2024)
    assert row["fips"] == "06037"
    assert row["pop"] == 1000
    assert row["pop_under_5"] == 50 + 40
    assert row["pop_5_9"] == 60 + 50
    # under 5, 5-9, 10-14, 15-17 and half of the 18-19 year olds
    assert row["school_aged_population"] == 90 + 110 + 130 + 50 + (10 + 6) / 2
    assert row["public_schooled"] == 20 + 10
    assert row["private_schooled"] == 5 + 3
    assert row["number_graduate_degree"] == (4 + 2 + 1) + (3 + 3 + 2)
    assert row["pop_25_up"] == 500
    # the 25 and over plus every age group of 15 to 24
    assert row["pop_15_up"] == 500 + 50 + 16 + 6 + 4 + 13
    assert row["fraction_white"] == pytest.approx(0.7)


def test_income_is_brought_to_2024_dollars_with_each_years_inflation():
    cpi = utils.CensusQuery("a-key").cpi_inflation
    assert population(2024)["median_household_income"] == 50000
    assert population(2022)["median_household_income"] == pytest.approx(
        50000 * (1 + cpi[2023]) * (1 + cpi[2024])
    )
    assert population(2010)["median_household_income"] == pytest.approx(
        50000 * _compound(cpi, 2011, 2024)
    )


def _compound(cpi, first, last):
    factor = 1.0
    for year in range(first, last + 1):
        factor *= 1 + cpi[year]
    return factor


def test_connecticut_income_is_inflated_from_the_year_its_estimates_are_of():
    """Connecticut's estimates stop at 2021, so the income asked for 2023 is
    that of 2021 and is inflated over 2022-2024, as the income asked for 2021
    (not over 2024 alone, as for an estimate of 2023)."""
    cpi = utils.CensusQuery("a-key").cpi_inflation
    census_query = utils.CensusQuery("a-key")
    census_query.census = TableCensus()  # type: ignore
    ct = census_query.query_5_9_pop([2023, 2021], ["09001", "09001"])
    expected = 50000 * _compound(cpi, 2022, 2024)
    assert ct["median_household_income"].to_list() == pytest.approx(
        [expected, expected]
    )
    other = census_query.query_5_9_pop([2023], ["06037"])
    assert other["median_household_income"][0] == pytest.approx(
        50000 * _compound(cpi, 2024, 2024)
    )


def test_the_year_and_fips_lists_must_have_the_same_length():
    with pytest.raises(AssertionError, match="same length"):
        utils.CensusQuery("a-key").query_5_9_pop([2020], ["06037", "06001"])


# --- the answers of the Census kept between runs -----------------------------


@pytest.fixture
def cache(tmp_path):
    return utils.CensusCache(str(tmp_path / "census.sqlite"))


def test_a_second_request_for_a_county_year_is_not_sent(cache):
    census = FakeCensus()
    first = utils.cached_query(cache, census, "06037", 2020)
    again = utils.cached_query(cache, census, "06037", 2020)
    assert len(census.acs5.calls) == 1
    assert first == again


def test_the_years_after_2024_share_the_answer_of_2024(cache):
    """query() asks 2024 for every later year, so 2025-2027 need no request."""
    census = FakeCensus()
    for year in (2024, 2025, 2026, 2027):
        utils.cached_query(cache, census, "06037", year)
    assert len(census.acs5.calls) == 1
    assert census.acs5.calls[0][4] == 2024


def test_connecticut_years_after_2021_share_the_answer_of_2021(cache):
    census = FakeCensus()
    for year in (2021, 2023, 2026):
        utils.cached_query(cache, census, "09001", year)
    assert len(census.acs5.calls) == 1
    assert census.acs5.calls[0][4] == 2021


def test_different_counties_and_years_are_each_asked(cache):
    census = FakeCensus()
    for fip, year in (("06037", 2020), ("06037", 2021), ("06001", 2020)):
        utils.cached_query(cache, census, fip, year)
    assert len(census.acs5.calls) == 3


def test_the_answer_is_the_one_query_gives(cache):
    """The same values and types (floats stay floats), first and from the file."""
    direct = utils.query(FakeCensus(), "06037", 2010)
    census = FakeCensus()
    first = utils.cached_query(cache, census, "06037", 2010)
    from_file = utils.cached_query(cache, census, "06037", 2010)
    assert first == direct == from_file
    assert {type(v) for v in from_file.values()} == {float}


def test_a_failed_query_saves_nothing_and_is_asked_again(cache):
    with pytest.raises(AssertionError):
        utils.cached_query(cache, FakeCensus(fail=True), "06037", 2020)
    census = FakeCensus()
    utils.cached_query(cache, census, "06037", 2020)
    assert len(census.acs5.calls) == 1


def test_the_answers_are_kept_in_the_file_for_the_next_run(tmp_path):
    path = str(tmp_path / "census.sqlite")
    first = utils.CensusQuery("a-key", cache_path=path)
    first.census = FakeCensus()  # type: ignore
    out = first.query_5_9_pop([2024, 2026, 2027], ["06037"] * 3)
    assert len(first.census.acs5.calls) == 1  # type: ignore

    # a new run: nothing is asked
    second = utils.CensusQuery("a-key", cache_path=path)
    second.census = FakeCensus()  # type: ignore
    again = second.query_5_9_pop([2024, 2026, 2027], ["06037"] * 3)
    assert second.census.acs5.calls == []  # type: ignore
    assert out.equals(again)


def test_a_change_of_the_query_code_does_not_use_the_old_answers(
    cache, monkeypatch
):
    census = FakeCensus()
    utils.cached_query(cache, census, "06037", 2020)
    monkeypatch.setattr(utils, "_query_version", lambda: "another-version")
    utils.cached_query(cache, census, "06037", 2020)
    assert len(census.acs5.calls) == 2


def test_without_a_cache_every_request_is_sent():
    query = utils.CensusQuery("a-key")
    assert query.cache is None
    query.census = FakeCensus()  # type: ignore
    query.query_5_9_pop([2024, 2025], ["06037", "06037"])
    assert len(query.census.acs5.calls) == 2  # type: ignore


# --- the land sizes are downloaded once per run -------------------------------


def _gazetteer_zip():
    import io
    import zipfile

    text = "USPS\tGEOID\tALAND\nCA\t06037\t10000000\nCA\t06001\t20000000\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("2020_Gaz_counties_national.txt", text)
    return buffer.getvalue()


def test_the_gazetteer_is_downloaded_once_for_many_census_queries(monkeypatch):
    calls = []

    class Response:
        content = _gazetteer_zip()

    monkeypatch.setattr(
        utils.requests, "get", lambda url: calls.append(url) or Response()
    )
    utils.fips_land_sizes.cache_clear()
    try:
        # a CensusQuery per call, as 00 and 05 make them
        # (sorted: the rows of query_land_size have no fixed order, and the
        # pipeline only joins them on fips)
        sizes = [
            utils.CensusQuery("a-key")
            .query_land_size(["06037", "06001"])
            .sort("fips")
            for _ in range(3)
        ]
        assert len(calls) == 1
        assert all(s.equals(sizes[0]) for s in sizes)
        assert sizes[0]["fips"].to_list() == ["06001", "06037"]
        assert sizes[0]["land_size_sq_km"].to_list() == [20.0, 10.0]
    finally:
        utils.fips_land_sizes.cache_clear()


def test_a_failed_download_is_asked_again(monkeypatch):
    attempts = []

    class Bad:
        content = b"not a zip"

    class Good:
        content = _gazetteer_zip()

    def get(url):
        attempts.append(url)
        return Bad() if len(attempts) == 1 else Good()

    monkeypatch.setattr(utils.requests, "get", get)
    utils.fips_land_sizes.cache_clear()
    try:
        with pytest.raises(Exception):
            utils.fips_land_sizes()
        assert utils.fips_land_sizes().height == 2
        assert len(attempts) == 2
    finally:
        utils.fips_land_sizes.cache_clear()


# --- the requests in parallel --------------------------------------------------


class EchoAcs5:
    """Answers every field with a number that says which place and year was
    asked, so that answers of different requests differ. `fail`: places that
    always fail; `flaky`: {place: how many times it fails before it answers}.
    `in_flight` counts the requests being answered at once."""

    def __init__(self, fail=(), flaky=None, hold=0.0):
        import threading

        self.calls = []
        self.fail = set(fail)
        self.flaky = dict(flaky or {})
        self.hold = hold
        self.lock = threading.Lock()
        self.in_flight = 0
        self.most_in_flight = 0

    def _answer(self, place, fields, year):
        import time

        with self.lock:
            self.calls.append((place, year))
            self.in_flight += 1
            self.most_in_flight = max(self.most_in_flight, self.in_flight)
            failing = place in self.fail or self.flaky.get(place, 0) > 0
            if self.flaky.get(place, 0) > 0:
                self.flaky[place] -= 1
        time.sleep(self.hold)
        with self.lock:
            self.in_flight -= 1
        if failing:
            raise RuntimeError("no such data")
        return [{f: float(int(place)) + year / 10_000 for f in fields}]

    def state(self, fields, state_fips, year):
        return self._answer(state_fips, fields, year)

    def state_county(self, fields, state_fips, county_fips, year):
        return self._answer(state_fips + county_fips, fields, year)


class EchoCensus:
    def __init__(self, **kwargs):
        self.acs5 = EchoAcs5(**kwargs)


ASKED_FIPS = ["06037", "06001", "06037", "09001", "48229", "06001", "48229"]
ASKED_YEARS = [2020, 2024, 2020, 2023, 2026, 2027, 2027]


def test_parallel_answers_are_those_of_one_at_a_time_in_the_order_asked():
    one_at_a_time = [
        utils.query(EchoCensus(), f, y)
        for f, y in zip(ASKED_FIPS, ASKED_YEARS)
    ]
    parallel = utils.fetch_answers(
        EchoCensus(), None, ASKED_FIPS, ASKED_YEARS, workers=4, wait=0
    )
    assert parallel == one_at_a_time
    # the answers do say which place and year they are of
    assert parallel[0] != parallel[1]


def test_each_distinct_place_and_year_is_asked_once():
    census = EchoCensus()
    utils.fetch_answers(
        census, None, ASKED_FIPS, ASKED_YEARS, workers=4, wait=0
    )
    # 06037 in 2020, 06001 in 2024, Connecticut in 2021, 48229 in 2024 (for
    # 2026 and 2027 too)
    assert sorted(census.acs5.calls) == sorted(
        [("06037", 2020), ("06001", 2024), ("09001", 2021), ("48229", 2024)]
    )


def test_one_place_of_each_year_is_asked_alone_before_the_others():
    census = EchoCensus()
    fips = ["06001", "06003", "06005", "06007", "06009", "06011"]
    years = [2020, 2020, 2020, 2022, 2022, 2022]
    utils.fetch_answers(census, None, fips, years, workers=4, wait=0)
    assert census.acs5.calls[:2] == [("06001", 2020), ("06007", 2022)]


def test_no_more_requests_are_in_flight_than_workers():
    census = EchoCensus(hold=0.02)
    fips = [f"06{n:03d}" for n in range(1, 40, 2)]
    for workers in (1, 3):
        census = EchoCensus(hold=0.02)
        utils.fetch_answers(
            census, None, fips, [2020] * len(fips), workers=workers, wait=0
        )
        assert census.acs5.most_in_flight <= workers
    assert census.acs5.most_in_flight > 1  # and they do go in parallel


def test_a_request_that_fails_once_is_tried_again():
    census = EchoCensus(flaky={"06003": 1})
    out = utils.fetch_answers(
        census,
        None,
        ["06001", "06003", "06005"],
        [2020] * 3,
        workers=2,
        wait=0,
    )
    assert [c for c in census.acs5.calls if c[0] == "06003"] == [
        ("06003", 2020)
    ] * 2
    assert out[1]["B01001_001E"] == 6003.0 + 2020 / 10_000


def test_a_request_that_keeps_failing_raises_after_its_attempts():
    census = EchoCensus(fail={"06003"})
    with pytest.raises(AssertionError):
        utils.fetch_answers(
            census,
            None,
            ["06001", "06003", "06005"],
            [2020] * 3,
            workers=2,
            attempts=3,
            wait=0,
        )
    assert [c for c in census.acs5.calls if c[0] == "06003"] == [
        ("06003", 2020)
    ] * 3


def test_a_failure_stops_the_run_but_keeps_the_answers_that_arrived(cache):
    fips = ["06001", "06005", "06007", "06009", "06003"]  # the last one fails
    with pytest.raises(AssertionError):
        utils.fetch_answers(
            EchoCensus(fail={"06003"}),
            cache,
            fips,
            [2020] * 5,
            workers=2,
            attempts=1,
            wait=0,
        )
    # a new run asks only for the place that failed
    census = EchoCensus()
    out = utils.fetch_answers(
        census, cache, fips, [2020] * 5, workers=2, wait=0
    )
    assert census.acs5.calls == [("06003", 2020)]
    assert [o["B01001_001E"] for o in out] == [
        float(int(f)) + 0.202 for f in fips
    ]


def test_the_table_of_populations_is_the_same_in_parallel(tmp_path):
    fips = ["06037", "06001", "06037", "09001", "48229"]
    years = [2020, 2024, 2020, 2023, 2026]
    serial = utils.CensusQuery("a-key")
    serial.census = EchoCensus()  # type: ignore
    parallel = utils.CensusQuery(
        "a-key", cache_path=str(tmp_path / "c.sqlite"), workers=4
    )
    parallel.census = EchoCensus()  # type: ignore
    assert serial.query_5_9_pop(years, fips).equals(
        parallel.query_5_9_pop(years, fips)
    )
    # and a run after it asks nothing
    again = utils.CensusQuery(
        "a-key", cache_path=str(tmp_path / "c.sqlite"), workers=4
    )
    again.census = EchoCensus()  # type: ignore
    again.query_5_9_pop(years, fips)
    assert again.census.acs5.calls == []  # type: ignore


# --- the land sizes keep the order of the places asked -----------------------


def test_the_land_sizes_come_in_the_order_of_the_places_asked(monkeypatch):
    import random

    geoids = [
        f"{state:02d}{county:03d}"
        for state in (1, 2, 4)
        for county in range(1, 120, 2)
    ]
    gazetteer = pl.DataFrame(
        {
            "GEOID": geoids,
            "ALAND": [1_000_000 * (i + 1) for i in range(len(geoids))],
        }
    )
    monkeypatch.setattr(utils, "fips_land_sizes", lambda: gazetteer)
    asked = random.Random(1).sample(geoids, 150) + ["02", "04"]
    random.Random(2).shuffle(asked)
    expected = {
        g: gazetteer.filter(pl.col("GEOID") == g)["ALAND"][0] / 1e6
        for g in asked
        if len(g) == 5
    }
    expected["02"] = (
        sum(
            a
            for g, a in zip(gazetteer["GEOID"], gazetteer["ALAND"])
            if g.startswith("02")
        )
        / 1e6
    )
    expected["04"] = (
        sum(
            a
            for g, a in zip(gazetteer["GEOID"], gazetteer["ALAND"])
            if g.startswith("04")
        )
        / 1e6
    )
    for _ in range(25):  # an order that came by chance would not hold twice
        out = utils.CensusQuery("a-key").query_land_size(asked)
        assert out["fips"].to_list() == asked
        assert out["land_size_sq_km"].to_list() == [expected[g] for g in asked]
