"""
Trend statistics of the time-varying effect of vaccine coverage on outbreak
size, one value per posterior draw (see
analysis/figure_coverage_predictability_changes_over_time.py, which prints
them).
"""

import numpy as np
from scipy import stats


def trend_draws(beta, mu, row_years, years, drop=()):
    """
    Per-draw trend statistics for the time-varying coverage effect.

    beta      [S, T]  log of the factor the outbreak size is multiplied by per +1pp
              of coverage, chronological (on the natural covariate scale: log(1 +
              the % change per pp / 100), averaged over a fixed set of county-years;
              see outbreak_size.coverage_effect)
    mu        [S, N]  model-expected cases for each county-year
    row_years [N]     calendar year of each county-year (columns of mu)
    years     [T]     calendar years matching beta's columns
    drop      years to leave out (sensitivity analysis)

    Returns a dict of [S] arrays (f, l = first and last kept year):
      1. slopes: of E_t = 100*(exp(beta_t)-1) [pp of %-effect / yr], and the
         log-scale version 100*(exp(slope of beta_t)-1) [% / yr in the rate
         ratio per +1pp]
      2. in cases: A = mubar*sA/100 (fixed average outbreak size) and
         sB = slope of AME_t = mean_i(mu_it)*(exp(beta_t)-1) (year-specific
         size), both [cases / pp coverage / yr]
      3. endpoints f and l, from the line fit on the %-scale, the line fit on
         the log scale, and the raw yearly estimates; effects in % per pp,
         cases per county-year per +1pp; "diff" = l minus f, per draw
      4. Spearman rho between the yearly effect E_t and the year (how
         monotone the trend is)
      mubar  mean of mu over all kept county-years [cases per county-year]
    """
    years = np.asarray(years)
    row_years = np.asarray(row_years)
    keep = ~np.isin(years, list(drop))
    row_keep = ~np.isin(row_years, list(drop))
    yk = years[keep]
    xc = yk - yk.mean()  # centered year, [yr]

    def slope_and_ends(y):  # y [S, Tk] -> slope [S], fitted first/last [S]
        ybar = y.mean(axis=1)
        s = (y - ybar[:, None]) @ xc / (xc @ xc)
        return s, ybar + s * xc[0], ybar + s * xc[-1]

    growth = np.exp(beta[:, keep]) - 1  # [S, Tk], fractional change per pp
    E = 100 * growth  # [S, Tk], % change per pp
    # AME_t: expected cases per county-year in year t associated with +1pp
    ame = np.stack(
        [
            (mu[:, row_years == yr] * g[:, None]).mean(axis=1)
            for yr, g in zip(yk, growth.T)
        ],
        axis=1,
    )  # [S, Tk], cases per pp
    mubar = mu[:, row_keep].mean(axis=1)  # [S], cases per county-year

    sA, E_first, E_last = slope_and_ends(E)
    sB, ame_first, ame_last = slope_and_ends(ame)
    # same trend fitted on the log rate-ratio scale: exp(slope)-1 is the
    # compounded % change per year in the rate ratio per +1pp
    b_log, lb_first, lb_last = slope_and_ends(beta[:, keep])
    Elog_first = 100 * (np.exp(lb_first) - 1)  # % per pp
    Elog_last = 100 * (np.exp(lb_last) - 1)
    f, l = yk[0], yk[-1]
    out = {
        "1a slope of %-effect (pp/yr)": sA,
        "1b % change per year in rate ratio (log scale)": 100
        * (np.exp(b_log) - 1),
        "2a A basis: sA*mubar/100 (cases/pp/yr)": mubar * sA / 100,
        "2b B basis: slope of AME (cases/pp/yr)": sB,
        "2c B minus A (cases/pp/yr)": sB - mubar * sA / 100,
    }
    # endpoints: (label, effect % per pp at f, at l, AME cases per pp at f, at l)
    for label, e0, e1, a0, a1 in [
        ("3 %-scale line", E_first, E_last, ame_first, ame_last),
        ("4 log-scale line", Elog_first, Elog_last, None, None),
        ("5 raw yearly", E[:, 0], E[:, -1], ame[:, 0], ame[:, -1]),
    ]:
        out[f"{label}: effect {f} (% per pp)"] = e0
        out[f"{label}: effect {l} (% per pp)"] = e1
        out[f"{label}: effect diff (pp)"] = e1 - e0
        out[f"{label}: A-basis cases {f}"] = mubar * e0 / 100
        out[f"{label}: A-basis cases {l}"] = mubar * e1 / 100
        out[f"{label}: A-basis cases diff"] = mubar * (e1 - e0) / 100
        if a0 is not None:
            out[f"{label}: B-basis cases {f}"] = a0
            out[f"{label}: B-basis cases {l}"] = a1
            out[f"{label}: B-basis cases diff"] = a1 - a0
    # Spearman rho between each draw's yearly effect and the calendar year:
    # how monotone the trend is (the posterior carries the uncertainty; a
    # p-value from the median curve is not used because the random-walk
    # prior makes neighboring years dependent)
    ranks = stats.rankdata(E, axis=1)
    ranks = ranks - ranks.mean(axis=1, keepdims=True)
    year_ranks = stats.rankdata(yk) - stats.rankdata(yk).mean()
    out["6 Spearman rho, effect vs year (per draw)"] = (
        ranks @ year_ranks
    ) / np.sqrt((ranks**2).sum(axis=1) * (year_ranks**2).sum())
    out["mubar (cases per county-year)"] = mubar
    return out
