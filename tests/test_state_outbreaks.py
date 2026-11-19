"""Summaries of a state's outbreaks (outbreak_size.state_outbreaks)."""

import pytest

from outbreak_size.state_outbreaks import weighted_mean_population


def test_the_mean_is_weighted_by_the_weights():
    records = [(100.0, 1.0), (200.0, 3.0)]
    assert weighted_mean_population(records) == pytest.approx(175.0)


def test_equal_weights_give_the_plain_mean():
    assert weighted_mean_population(
        [(10.0, 2.0), (30.0, 2.0)]
    ) == pytest.approx(20.0)


def test_weights_of_zero_give_the_plain_mean():
    assert weighted_mean_population(
        [(10.0, 0.0), (50.0, 0.0)]
    ) == pytest.approx(30.0)


def test_no_outbreak_gives_none_not_zero():
    assert weighted_mean_population([]) is None


def test_a_single_outbreak_is_its_own_mean():
    assert weighted_mean_population([(42.0, 7.0)]) == pytest.approx(42.0)
