import numpy as np
import pytest

from outbreak_size.doses import additional_doses, kindergarten_cohorts
from outbreak_size.utils import CensusCache, _cache_key


def test_doses_hand_computed_with_the_negative_change_floored():
    """Cohort 100 children: +5 points is 5 more children, each needing 2
    doses: 10 doses; -3 points (the intervention below the baseline) adds
    none (0 doses, not -6)."""
    doses = additional_doses([100.0, 100.0, 250.0], [5.0, -3.0, 4.0])
    assert doses.tolist() == pytest.approx([10.0, 0.0, 20.0])


def _cache(tmp_path, answers):
    cache = CensusCache(str(tmp_path / "cache.sqlite"))
    for (fips, year), answer in answers.items():
        cache.put(_cache_key(fips, year), answer)
    return cache


def answer(total, boys, girls):
    return {"B01001_001E": total, "B01001_004E": boys, "B01001_028E": girls}


def test_the_cohort_is_a_fifth_of_the_boys_and_girls_aged_5_to_9(tmp_path):
    # outbreak year 2027 uses the county data of 2026, which the cache holds
    # as the latest ACS year (2024); Connecticut (09) is held at 2021
    cache = _cache(
        tmp_path,
        {
            ("01001", 2026): answer(10000, 300, 200),
            ("09001", 2026): answer(5000, 100, 150),
        },
    )
    cohorts = kindergarten_cohorts(
        ["01001", "09001"], [10000.0, 5000.0], 2027, cache
    )
    assert cohorts == pytest.approx([100.0, 50.0])  # 500 / 5, 250 / 5


def test_a_county_without_a_saved_answer_is_an_error(tmp_path):
    cache = _cache(tmp_path, {})
    with pytest.raises(KeyError):
        kindergarten_cohorts(["01001"], [10000.0], 2027, cache)


def test_a_saved_population_that_is_not_the_tables_is_an_error(tmp_path):
    cache = _cache(tmp_path, {("01001", 2026): answer(9000, 300, 200)})
    with pytest.raises(ValueError):
        kindergarten_cohorts(["01001"], [10000.0], 2027, cache)
