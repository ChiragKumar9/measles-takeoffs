"""Trend of the time-varying coverage effect (outbreak_size.trend)."""

import numpy as np
import pytest
from scipy import stats

from outbreak_size.trend import trend_draws

YEARS = np.arange(2011, 2026)  # 15 years
ROW_YEARS = np.repeat(YEARS, 2)  # two counties per year
SLOPE = "1a slope of %-effect (pp/yr)"
LOG_SLOPE = "1b % change per year in rate ratio (log scale)"
A_SLOPE = "2a A basis: sA*mubar/100 (cases/pp/yr)"
B_SLOPE = "2b B basis: slope of AME (cases/pp/yr)"
SPEARMAN = "6 Spearman rho, effect vs year (per draw)"


def beta_from_effect(effect_pct):
    """The log rate ratio whose % change per pp is `effect_pct`."""
    return np.log(1 + np.asarray(effect_pct) / 100)


def flat_mu(value, n_draws=1):
    return np.full((n_draws, len(ROW_YEARS)), float(value))


def test_a_linear_fall_in_the_effect_by_hand():
    """The effect falls 1 percentage point a year, from 0% in 2011 to -14% in
    2025, in counties that all expect 50 cases. The slope is -1 pp a year; in
    cases it is -1 * 50 / 100 = -0.5 per pp a year; the fitted line ends at
    -14% and loses 14 * 50 / 100 = 7 cases per pp over the years; and the
    effect falls every year (rho -1)."""
    effect = -(YEARS - 2011.0)
    out = trend_draws(
        beta_from_effect(effect)[None, :], flat_mu(50), ROW_YEARS, YEARS
    )
    assert out[SLOPE][0] == pytest.approx(-1.0)
    assert out[A_SLOPE][0] == pytest.approx(-0.5)
    assert out["3 %-scale line: effect 2011 (% per pp)"][0] == pytest.approx(0)
    assert out["3 %-scale line: effect 2025 (% per pp)"][0] == pytest.approx(
        -14.0
    )
    assert out["3 %-scale line: A-basis cases diff"][0] == pytest.approx(-7.0)
    # the difference of the two ends of the line, and of the raw yearly values
    # (which are on the line here), is -14 pp
    assert out["3 %-scale line: effect diff (pp)"][0] == pytest.approx(-14.0)
    assert out["5 raw yearly: effect diff (pp)"][0] == pytest.approx(-14.0)
    assert out["5 raw yearly: A-basis cases diff"][0] == pytest.approx(-7.0)
    assert out[SPEARMAN][0] == pytest.approx(-1.0)
    assert out["mubar (cases per county-year)"][0] == pytest.approx(50.0)


def test_spearman_rho_agrees_with_scipy_on_a_curve_that_is_not_monotone():
    effect = np.random.default_rng(0).uniform(-20, 20, (1, len(YEARS)))
    out = trend_draws(beta_from_effect(effect), flat_mu(50), ROW_YEARS, YEARS)
    assert out[SPEARMAN][0] == pytest.approx(
        float(stats.spearmanr(YEARS, effect[0])[0])  # type: ignore
    )


@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
def test_a_constant_effect_has_no_trend_even_when_outbreaks_grow():
    """The effect stays -10% while the expected outbreak size grows from 10 to
    100 cases: the trend of the effect is 0, but the cases it averts per pp
    grow, by -0.1 * (100 - 10) / 14 per pp a year (the slope of
    -0.1 * mu_t)."""
    beta = np.full((1, len(YEARS)), np.log(0.9))
    mu_by_year = np.linspace(10, 100, len(YEARS))
    mu = np.repeat(mu_by_year, 2)[None, :]
    out = trend_draws(beta, mu, ROW_YEARS, YEARS)
    assert out[SLOPE][0] == pytest.approx(0.0, abs=1e-9)
    assert out[B_SLOPE][0] == pytest.approx(-0.1 * 90 / 14)


def test_the_log_scale_slope_compounds_per_year():
    """beta falls by 0.02 a year: the rate ratio per pp changes by
    exp(-0.02) - 1 a year, and the fitted line ends at beta = -0.28."""
    beta = (-0.02 * (YEARS - 2011.0))[None, :]
    out = trend_draws(beta, flat_mu(50), ROW_YEARS, YEARS)
    assert out[LOG_SLOPE][0] == pytest.approx(100 * (np.exp(-0.02) - 1))
    assert out["4 log-scale line: A-basis cases 2025"][0] == pytest.approx(
        50 * (np.exp(-0.28) - 1)
    )


def test_every_draw_gets_its_own_trend():
    """Three draws whose effects fall by 1, 2 and 0.5 pp a year."""
    effects = np.stack([-(YEARS - 2011.0) * s for s in (1.0, 2.0, 0.5)])
    out = trend_draws(
        beta_from_effect(effects), flat_mu(50, 3), ROW_YEARS, YEARS
    )
    assert out[SLOPE] == pytest.approx([-1.0, -2.0, -0.5])
    assert out[A_SLOPE] == pytest.approx([-0.5, -1.0, -0.25])
    assert all(v.shape == (3,) for v in out.values())


def outlier_in_2025():
    """An effect falling 1pp a year until 2024, then jumping to +50% in 2025,
    for counties that expect 50 cases in every year but 2025 (1000)."""
    effect = -(YEARS - 2011.0)
    effect[-1] = 50.0
    mu = np.repeat(np.where(YEARS == 2025, 1000.0, 50.0), 2)[None, :]
    return beta_from_effect(effect)[None, :], mu


def test_dropped_years_leave_the_trend_and_the_average_size():
    beta, mu = outlier_in_2025()
    kept = trend_draws(beta, mu, ROW_YEARS, YEARS, drop=(2025,))
    assert kept[SLOPE][0] == pytest.approx(-1.0)  # the 2011-2024 line
    assert kept["mubar (cases per county-year)"][0] == pytest.approx(50.0)
    # the labels name the last year kept
    assert kept["3 %-scale line: effect 2024 (% per pp)"][0] == pytest.approx(
        -13.0
    )
    assert "3 %-scale line: effect 2025 (% per pp)" not in kept
    # with the year in, the jump bends the line and raises the mean size
    full = trend_draws(beta, mu, ROW_YEARS, YEARS)
    assert full[SLOPE][0] != pytest.approx(-1.0, abs=0.1)
    assert full["mubar (cases per county-year)"][0] > 50


def test_dropping_no_year_is_the_default():
    beta, mu = outlier_in_2025()
    a = trend_draws(beta, mu, ROW_YEARS, YEARS)
    b = trend_draws(beta, mu, ROW_YEARS, YEARS, drop=())
    assert all(np.array_equal(a[k], b[k]) for k in a)


@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
def test_the_average_size_is_over_the_kept_county_years():
    """Counties expect 10, 20, 30 ... cases in turn: the mean is over all of
    them, and a dropped year's counties are left out."""
    mu = np.arange(1, len(ROW_YEARS) + 1, dtype=float)[None, :] * 10
    beta = np.zeros((1, len(YEARS)))
    out = trend_draws(beta, mu, ROW_YEARS, YEARS)
    assert out["mubar (cases per county-year)"][0] == pytest.approx(mu.mean())
    out = trend_draws(beta, mu, ROW_YEARS, YEARS, drop=(2011, 2012))
    assert out["mubar (cases per county-year)"][0] == pytest.approx(
        mu[:, 4:].mean()
    )


def test_a_flat_effect_has_no_defined_spearman_rho():
    """When a draw's yearly effects are all equal there is no rank order to
    correlate with the year, so rho is undefined (NaN), not 0."""
    beta = np.full((1, len(YEARS)), np.log(0.9))
    with pytest.warns(RuntimeWarning, match="invalid value"):
        out = trend_draws(beta, flat_mu(50), ROW_YEARS, YEARS)
    assert np.isnan(out[SPEARMAN][0])
