"""
The effect of the vaccination covariates (coverage and exemptions) on the
outbreak size, from the coefficient of the standardized covariate, on the
scale the model was fit on (see outbreak_size.model_inputs.apply_covariate_scale).

- "raw": the covariate is in percent, standardized. A 1 percentage point
  increase changes the outbreak size by 100 * (exp(coefficient / SD) - 1)
  percent, wherever it starts.
- "natural": the coverage covariate is log(susceptible fraction) and an
  exemption covariate is log(1 + exemption percentage), both standardized. The
  model is then a power law, outbreak size ~ (susceptible fraction) ** e, and
  e = coefficient / SD (the elasticity) is the same wherever coverage starts,
  but the effect of 1 percentage point is not: the susceptible fraction is
  1 - effectiveness * coverage / 100, so a percentage point matters more where
  coverage is high. The % change per percentage point is therefore averaged
  over a set of REFERENCE county-years, each at its own coverage (exemptions:
  each at its own 1 + exemption percentage). Use the same reference rows for
  every number that is compared (periods, years, states), so that the
  comparison is of the elasticities and not of the coverage of the rows.

The SD is that of the covariate column the model was standardized with (the
population SD, as sklearn's StandardScaler), on the model's scale.
"""

import numpy as np
import polars as pl

from outbreak_size.model_inputs import apply_covariate_scale


def model_scale_column(
    rows: pl.DataFrame,
    column: str,
    covariate_scale: str,
    mmr_effectiveness: float | None = None,
) -> np.ndarray:
    """The column of the rows on the model's scale (log susceptible fraction,
    log(1 + exemption percentage), or the percent itself), without missing
    values."""
    scaled = apply_covariate_scale(
        rows.select(column), covariate_scale, mmr_effectiveness
    )
    return scaled[column].drop_nulls().to_numpy()


def covariate_sd(
    rows: pl.DataFrame,
    column: str,
    covariate_scale: str,
    mmr_effectiveness: float | None = None,
) -> float:
    """
    The SD of the column of the rows the model was fit on, on the model's
    scale. numpy's default (ddof=0) matches StandardScaler; polars' `.std()`
    (ddof=1) does not.
    """
    return float(
        model_scale_column(
            rows, column, covariate_scale, mmr_effectiveness
        ).std()
    )


def log_change_per_step(coefficient, sd, covariate_scale):
    """
    The log of the factor the outbreak size is multiplied by per step of the
    covariate: a 1% increase in the susceptible fraction (or in 1 + the
    exemption percentage) on the natural scale, log(1.01) * coefficient / sd,
    or 1 percentage point on the raw scale, coefficient / sd.
    """
    effect = np.asarray(coefficient) / sd
    if covariate_scale == "natural":
        return np.log(1.01) * effect
    if covariate_scale == "raw":
        return effect
    raise ValueError(f"unknown covariate scale {covariate_scale!r}")


def percent_change(coefficient, sd, covariate_scale):
    """
    The % change in the outbreak size per step of the covariate (see
    log_change_per_step): 100 * (1.01 ** (coefficient / sd) - 1) per 1%
    increase on the natural scale, and 100 * (exp(coefficient / sd) - 1) per
    percentage point on the raw scale.
    """
    return 100 * np.expm1(
        log_change_per_step(coefficient, sd, covariate_scale)
    )


def value_kind(column):
    """ "exemption" for an exemption column, else "coverage"."""
    return "exemption" if "exempt" in column else "coverage"


def factor_per_pp(
    effect, values, covariate_scale, mmr_effectiveness=None, kind="coverage"
):
    """
    The factor the outbreak size is multiplied by when the covariate of each
    county-year rises by 1 percentage point. `effect` is the coefficient / SD:
    one per draw (the result is then [draws, county-years]) or one per draw and
    county-year ([draws, county-years]). `values` are the covariate (coverage
    or exemption percentage) of each county-year; `kind` is "coverage" or
    "exemption".
    """
    effect = np.asarray(effect, dtype=float)
    if effect.ndim == 1:
        effect = effect[:, None]
    values = np.asarray(values, dtype=float)[None, :]
    if covariate_scale == "raw":
        return np.exp(effect) * np.ones_like(values)
    if covariate_scale != "natural":
        raise ValueError(f"unknown covariate scale {covariate_scale!r}")
    if mmr_effectiveness is None:
        raise ValueError("the natural scale needs mmr_effectiveness")
    if kind == "coverage":
        before = 1 - mmr_effectiveness * values / 100
        after = 1 - mmr_effectiveness * (values + 1) / 100
    else:
        before, after = 1 + values, 2 + values
    return (after / before) ** effect


def marginal_percent_change_per_pp(
    effect, values, covariate_scale, mmr_effectiveness=None, kind="coverage"
):
    """The % change in the outbreak size per percentage point of the
    covariate, averaged over the county-years with these `values`, per draw."""
    factor = factor_per_pp(
        effect, values, covariate_scale, mmr_effectiveness, kind
    )
    return 100 * (factor - 1).mean(axis=1)


def percent_change_per_pp(
    coefficient,
    fit_rows,
    column,
    reference_rows,
    covariate_scale,
    mmr_effectiveness=None,
):
    """
    The % change in the outbreak size per percentage point of `column`, per
    draw, from the coefficients (per SD of the column of the `fit_rows` the
    model was fit on), averaged over the `reference_rows` (see the top of this
    module). `coefficient` is [draws], or [draws, groups] for one coefficient
    per group (the result then has the same shape).
    """
    sd = covariate_sd(fit_rows, column, covariate_scale, mmr_effectiveness)
    values = reference_rows[column].drop_nulls().to_numpy()
    kind = value_kind(column)
    coefficient = np.asarray(coefficient, dtype=float) / sd

    def one(effect):
        return marginal_percent_change_per_pp(
            effect, values, covariate_scale, mmr_effectiveness, kind
        )

    if coefficient.ndim == 2:
        return np.stack(
            [one(coefficient[:, g]) for g in range(coefficient.shape[1])],
            axis=1,
        )
    return one(coefficient)


def log_effect_per_pp(
    coefficient,
    fit_rows,
    column,
    reference_rows,
    covariate_scale,
    mmr_effectiveness=None,
):
    """
    The change in log outbreak size per percentage point of the coverage
    `column` for a coefficient (per SD of the column of the `fit_rows`), at the
    mean coverage of the `reference_rows`: the slope of log(susceptible
    fraction) at that coverage times the elasticity (natural scale), or
    coefficient / SD (raw). For the standard deviations of coefficients (their
    size, not their direction), so that they are in a unit of coverage.
    """
    effect = np.asarray(coefficient, dtype=float) / covariate_sd(
        fit_rows, column, covariate_scale, mmr_effectiveness
    )
    if covariate_scale == "raw":
        return effect
    assert covariate_scale == "natural" and value_kind(column) == "coverage"
    mean = reference_rows[column].drop_nulls().mean()
    slope = mmr_effectiveness / 100 / (1 - mmr_effectiveness * mean / 100)  # type: ignore
    return effect * slope


def cases_change_per_pp(
    effect,
    cases,
    values,
    covariate_scale,
    mmr_effectiveness=None,
    kind="coverage",
):
    """
    The change in the outbreak size, in cases, per percentage point of the
    covariate, averaged over the county-years (each with its observed cases and
    covariate as the starting point), per draw: the average of
    cases * (factor - 1), for the average outbreak.
    """
    factor = factor_per_pp(
        effect, values, covariate_scale, mmr_effectiveness, kind
    )
    return (np.asarray(cases, dtype=float)[None, :] * (factor - 1)).mean(
        axis=1
    )


def posterior_contrast(before, after):
    """
    Compare the draws of one quantity in two fits (independent posteriors):
    the probability that a draw of `after` is above a draw of `before` (all
    pairs; ties count half), the two-sided empirical p-value (twice the smaller
    of that probability and its complement), and the median and 95% interval of
    the difference of paired draws (the draws are independent, so pairing by
    position is a random pairing).
    """
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    ordered = np.sort(before)
    below = np.searchsorted(ordered, after, side="left")
    up_to = np.searchsorted(ordered, after, side="right")
    p_after_above = float(
        (below + 0.5 * (up_to - below)).sum() / (len(before) * len(after))
    )
    n = min(len(before), len(after))
    difference = after[:n] - before[:n]
    low, median, high = np.percentile(difference, [2.5, 50, 97.5])
    return {
        "P(after > before)": p_after_above,
        "two-sided p": 2 * min(p_after_above, 1 - p_after_above),
        "difference median": float(median),
        "difference 2.5%": float(low),
        "difference 97.5%": float(high),
    }


def effect_label(covariate="coverage"):
    """The axis label of the % change in outbreak size per percentage point of
    the covariate ("coverage" or "exemption"), on two lines."""
    what = "MMR vaccine coverage" if covariate == "coverage" else "exemptions"
    return f"% change in outbreak size per 1pp\nincrease in {what}"


def variability_label():
    """The axis label of a standard deviation of a coverage coefficient, in
    the unit of log_effect_per_pp."""
    return (
        "(log outbreak size per 1pp of\nMMR coverage, at the average coverage)"
    )
