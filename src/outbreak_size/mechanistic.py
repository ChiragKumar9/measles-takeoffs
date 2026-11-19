"""
Mechanistic model of the number of outbreak cases in a place (a county-year)
from its vaccination coverage.

How many people each case infects (Re) follows from the share of the
population that is susceptible:

    susceptible = 1 - effectiveness * coverage / 100       (a fraction)
    Re          = r0 * susceptible                          (no units)

where effectiveness is the share of the vaccinated who are immune
(all-or-nothing immunity) and r0 is the number of people a case infects in a
fully susceptible population. The model treats two groups of places
differently, because outbreak theory behaves differently on either side of
Re = 1:

  * Higher-coverage places (Re < 1). Each case infects fewer than one other
    person on average, so an outbreak fizzles out. The average number of cases
    (the size, in cases) is 1 / (1 - Re). This comes from modelling each case
    as infecting a random number of others (with an average of Re), a model
    known as a branching process. The size is 1 case when nobody is
    susceptible and grows without limit as Re approaches 1. Call it `size`.

  * Lower-coverage places (Re >= 1). An outbreak can infect a large share of
    the population. The theory gives the fraction of the population infected
    in the end, z (no units), which solves
        susceptible * exp(-r0 * z) + z - susceptible = 0.
    A fitted line turns z into a number of cases. Call z `size` too; the
    group of a place tells which of the two it is.

The predicted number of cases (the mean, in cases per place and year) is

    lower-coverage:   log(mean) = log_threshold + slope * z + effect
    higher-coverage:  mean = exp(effect) * min(size, exp(log_threshold))

where
  * exp(log_threshold) is the number of cases predicted for a place right at
    Re = 1 (z = 0), the "threshold size", in cases. It is the starting point of
    the fitted line, and it is also the ceiling for the higher-coverage group.
    Without that ceiling, the theory's size for a place just under Re = 1 is
    huge, while the fitted line starts at a modest number, so a place with
    higher coverage could be predicted more cases than one with lower coverage.
    Using one number for both keeps the two groups joined at Re = 1.
  * slope (per unit of z, never below 0): more people infected by the theory
    never means fewer cases.
  * effect is an optional adjustment on the log scale, the sum of a year effect
    and a state effect (0 when no years or states are given). It multiplies
    the mean of both groups. The year effects add up to 0 over the years, and
    the state effects add up to 0 over the states (see "Why the effects add up
    to zero" below).

Because the slope is not negative and z is not negative, every lower-coverage
mean is at least exp(log_threshold) * exp(effect), and every higher-coverage
mean is at most that. Without effects, the mean therefore never rises with
coverage, so ranking places by their mean ranks them by coverage (see
`ranking_order`).

Why the effects add up to zero. The effect multiplies the size that outbreak
theory gives a higher-coverage place, and that size has a fixed scale: it is
not fitted. There must therefore be a year and a state whose effect is 0, where
the theory's size is used as it is, and every other effect says by how much the
theory's size is scaled up or down there relative to it. The usual way to fix
the scale of effects is to name one state and one year as the reference, with
effect 0. That does not work here: choosing another reference changes the
model, not only its labels, because it changes which places have the theory's
size used unscaled. The same data then give a different fit, and so a
different log-likelihood, depending on which state and year were named.
Requiring instead that the year effects add up to 0 across the years, and the
state effects across the states, makes no state or year special: the theory's
size is used unscaled for the average state and year, each state and each year
counting once however many places it has. The fit then does not change if the
states or the years are renamed or put in another order, and
exp(log_threshold) is the threshold size of that average state and year. (To
fit the effects, the last year and the last state are not given a parameter of
their own: each is minus the sum of the others. Which one is left out changes
how the model is written, not the model.)

Fitting. The counts of cases of the places are treated as Poisson with these
means (a standard model for counts), and the parameters are the ones that
make the observed counts most probable: they maximize the likelihood, the
probability of the observed counts under the model (the log-likelihood is its
logarithm). Both groups of places enter the likelihood: the threshold size is
shared, so the higher-coverage places inform it too. For a given threshold
size the model is an ordinary Poisson regression (in the slope and the
effects), whose likelihood has a single peak, so the slope rule is applied
exactly: fit with the slope free, and if it comes out negative, refit without
it (slope 0) with all other parameters refitted. The likelihood does not have a
single peak as a function of the threshold size itself, because the ceiling
moves with it. The threshold size is therefore found by a search: a fine grid
between two bounds, then a refinement around every peak close to the best one
(see `fit_mechanistic`).

States with no outbreak cases in any of their places cannot be given an effect:
the likelihood keeps rising as their effect falls, without a maximum. They are
left out of the fit and predicted a mean of 0 for all their places, which is
what an effect of minus infinity gives, and which matches their data. They
count as fitted parameters when counting parameters (one each), so that the
count does not depend on whether the data happened to be zero.

Counting parameters (for an AIC-style comparison, -2 * (log-likelihood - k)):
k is the number of free columns of the fit (the slope only when it is above 0,
because a slope held at 0 is not fitted; the year and state effects) plus one
for the threshold size plus one for each state left out.

The mean of the model is not limited by the population. For drawing numbers of
cases, or for evaluating a distribution, limit it with `limit_to_population`
(a place cannot have more cases than people). For ranking, use the unlimited
mean, so that a small place is not moved down the ranking only because of its
population.
"""

from typing import Any

import numpy as np
from scipy import optimize, special

from outbreak_size.final_outbreak_size import final_outbreak_size

# Lowest threshold size searched, in cases: a thousandth of a case per place
# and year. A threshold size that small means the model predicts essentially
# no outbreaks at all, and the likelihood of any data with outbreaks in it has
# collapsed long before that. The highest threshold size searched is the
# largest number of cases of any place in the data: a larger one would predict,
# for every lower-coverage place, more cases than were ever seen in one place.
# With year and state effects the threshold size is that of the reference year
# and state, so the highest bound is looser there. The checks in
# `fit_mechanistic` (the best value is not at a bound, and the fit is far worse
# at both bounds) are what make the bounds safe: they stop the fit if the
# bounds were too narrow to be sure of finding the best value.
LOWEST_THRESHOLD_SIZE = 0.001

# Distance between two values of the log of the threshold size in the first,
# coarse search. A step of 0.05 changes the threshold size by about 5%, small
# next to the width of the peaks of the likelihood.
GRID_STEP = 0.05

# A peak of the likelihood is refined only if it is within this many units of
# log-likelihood of the best value on the grid. A peak further below could
# only overtake the best one if the grid had misjudged the height of a peak by
# more than this, which a step of 0.05 does not do.
PEAK_WINDOW = 20.0

# At each bound of the search the log-likelihood must be at least this far
# below the best one (a likelihood ratio of about 1 in 500 million). If not,
# the bounds are too narrow to trust that the best value was found.
BOUND_DROP = 20.0


def theory_sizes(coverage, population, r0, effectiveness):
    """
    The group of each place and the size outbreak theory gives it.

    Parameters:
    -----------
    coverage : array, shape (n_places,)
        MMR coverage, in percent (0 to 100)
    population : array, shape (n_places,)
        Number of people of the place
    r0 : float
        Number of people each case infects in a fully susceptible population
    effectiveness : float
        Share of the vaccinated who are immune (all-or-nothing immunity), 0 to
        1

    Returns:
    --------
    lower_coverage : bool array, shape (n_places,)
        True where Re >= 1
    size : float array, shape (n_places,)
        For a lower-coverage place, the fraction of the population infected in
        the end (no units). For a higher-coverage place, the average number of
        cases of an outbreak, in cases, not more than the population.
    """
    susceptible = 1 - np.asarray(coverage, dtype=float) * effectiveness / 100
    lower_coverage = r0 * susceptible >= 1
    size = np.array(
        [
            final_outbreak_size(r0, s, n)[0]
            for s, n in zip(susceptible, population)
        ]
    )
    return lower_coverage, size


def poisson_log_likelihood(cases, mean):
    """
    Log-likelihood of the observed counts of cases for Poisson distributions
    with the given means, including the term that does not depend on the
    means, as statsmodels does. A place with mean 0 must have 0 cases (its
    term is then 0); otherwise the result is minus infinity.
    """
    cases = np.asarray(cases, dtype=float)
    mean = np.asarray(mean, dtype=float)
    return float(
        np.sum(special.xlogy(cases, mean) - mean - special.gammaln(cases + 1))
    )


def limit_to_population(mean, population):
    """A place cannot have more cases than people."""
    return np.minimum(mean, population)


def require_populations_above_threshold(fit, population):
    """
    Stop if a place has fewer people than the threshold size of the fit.

    The mean of a higher-coverage place is its theory size, which is at most
    its population, limited by the threshold size. If a place has fewer people
    than the threshold size, its mean is below that of places with the same or
    higher coverage, and the ranking by mean is no longer the ranking by
    coverage (a fit without effects ranks exactly as coverage does while every
    place has at least as many people as the threshold size).
    """
    threshold = float(np.exp(fit["log_threshold"]))
    smallest = float(np.min(population))
    # exp(log(100)) is 100.00000000000004: equal sizes must pass
    if smallest < threshold * (1 - 1e-9):
        raise ValueError(
            f"a place has {smallest:,.0f} people, fewer than the threshold "
            f"size of {threshold:,.2f} cases: the ranking by the mean of the "
            "mechanistic model is no longer the ranking by coverage"
        )


def ranking_order(mean, coverage):
    """
    Position of each place in a ranking from the highest to the lowest risk:
    0 for the place with the highest mean number of cases.

    Places are ordered by mean, from the highest, and places whose means are
    exactly equal are ordered by coverage, from the lowest. Places whose mean
    and coverage are both exactly equal share a position. Equal means are
    compared exactly, not with a tolerance: they come from the same formula
    applied to the same number, so they are the same number to the last digit.
    In a fit without effects the mean never rises with coverage, so this is
    exactly a ranking by coverage, and places with equal coverage stay tied.
    """
    key = np.column_stack(
        [-np.asarray(mean, dtype=float), np.asarray(coverage, dtype=float)]
    )
    return np.unique(key, axis=0, return_inverse=True)[1].ravel()


def _poisson_regression(design, cases, offset, start=None):
    """
    Poisson regression of cases on the columns of design, with a known term
    (the offset) added to the log of the mean:
        log(mean) = design @ coefficients + offset.

    The fit is improved step by step, each step using the slope and the
    curvature of the log-likelihood (Newton's method). The log-likelihood of
    this model has a single peak in the coefficients, so the steps reach it;
    any step that does not improve the fit is halved until it does.

    Returns (coefficients, log-likelihood without the term that does not
    depend on the coefficients, converged). With no columns there is nothing
    to fit, and the log-likelihood of the offset alone is returned.
    """

    def objective(coefficients):
        # The exponent is limited to keep the means finite when a step lands
        # far off; the limit is never reached at the maximum.
        eta = np.clip(design @ coefficients + offset, -30, 30)
        return float(np.sum(cases * eta - np.exp(eta)))

    coefficients = (
        np.zeros(design.shape[1]) if start is None else np.array(start, float)
    )
    current = objective(coefficients)
    if design.shape[1] == 0:
        return coefficients, current, True
    for _ in range(200):
        eta = np.clip(design @ coefficients + offset, -30, 30)
        mean = np.exp(eta)
        gradient = design.T @ (cases - mean)
        hessian = (design.T * mean) @ design
        step = np.linalg.solve(hessian, gradient)
        fraction = 1.0
        while fraction > 1e-8:
            new = objective(coefficients + fraction * step)
            if new >= current - 1e-12:
                break
            fraction /= 2
        coefficients = coefficients + fraction * step
        converged = abs(new - current) < 1e-10 * (1 + abs(current))  # type: ignore
        current = new  # type: ignore
        if converged:
            return coefficients, current, True
    return coefficients, current, False


def _fit_at_threshold(lower, size, cases, effects, log_threshold):
    """
    Fit the slope and the effects for a given threshold size (`log_threshold`
    is the log of it), with the slope not allowed below 0.

    The threshold size enters the model as a known term added to the log of
    the mean (the offset of `_poisson_regression`): it is the intercept of the
    lower-coverage places, and the ceiling of the higher-coverage places (log
    of the smaller of their theory size and the threshold size). With that term
    known, the model is an ordinary Poisson regression with a single peak, so
    when the slope comes out negative, the best allowed slope is exactly 0.
    """
    higher_size = np.where(lower, 1.0, size)
    offset = np.where(
        lower,
        log_threshold,
        np.log(np.minimum(higher_size, np.exp(log_threshold))),
    )
    slope_column = np.where(lower, size, 0.0).reshape(-1, 1)
    with_slope = np.hstack([slope_column, effects])
    coefficients, objective, converged = _poisson_regression(
        with_slope, cases, offset
    )
    if coefficients[0] >= 0:
        return dict(
            slope=float(coefficients[0]),
            effect_coefficients=coefficients[1:],
            objective=objective,
            converged=converged,
            design=with_slope,
        )
    # The slope came out negative. The best allowed slope is 0, and the other
    # parameters are refitted given it, not kept from the fit above. Without a
    # slope column, the lower-coverage mean is exactly exp(log_threshold) times
    # the effect, the same number as the ceiling of the higher-coverage group
    # (so the means that should tie do tie, to the last digit).
    coefficients, objective, converged = _poisson_regression(
        effects, cases, offset
    )
    return dict(
        slope=0.0,
        effect_coefficients=coefficients,
        objective=objective,
        converged=converged,
        design=effects,
    )


def _effects_design(years, states, cases):
    """
    Year and state effects as columns of a regression, such that the year
    effects add up to 0 over the years and the state effects add up to 0 over
    the states (see "Why the effects add up to zero" in the module docstring).

    The effect of a level (a year or a state) is the coefficient of its column,
    except for the last level, which has no column of its own: its effect is
    minus the sum of the others, which is what makes the effects add up to 0.
    The column of a level is 1 for the places of that level, -1 for the places
    of the last level, and 0 otherwise. A place of the last level therefore
    gets minus the sum of the coefficients, as it should. Which level is last
    (here the last in sorted order) only changes how the model is written, not
    the model: the fitted means are the same whichever level is left out.

    A state with no cases in any of its places is left out (see the module
    docstring), before the effects of the others are set up.

    Returns the columns for the kept places, which places are kept, the states
    left out, the name of each column ("year" or "state", and the level it
    belongs to), and the levels of the years and of the states in the order of
    the columns, with the last level (without a column) at the end.
    """
    years = np.asarray(years)
    states = np.asarray(states)
    cases = np.asarray(cases)
    dropped = sorted(s for s in set(states) if cases[states == s].sum() == 0)
    kept = ~np.isin(states, dropped)
    names, columns, levels = [], [], {}
    for kind, values in (("year", years[kept]), ("state", states[kept])):
        levels[kind] = sorted(set(values))
        last = levels[kind][-1]
        for level in levels[kind][:-1]:
            names.append((kind, level))
            is_level = (values == level).astype(float)
            is_last = (values == last).astype(float)
            columns.append(is_level - is_last)
    if columns:
        matrix = np.column_stack(columns)
    else:
        matrix = np.zeros((int(kept.sum()), 0))
    return matrix, kept, dropped, names, levels


def predict_mean(fit, lower, size, years=None, states=None):
    """
    Mean number of cases of each place under a fit of `fit_mechanistic`, in
    cases per place and year, not limited by the population.

    Parameters:
    -----------
    fit : dict
        Result of `fit_mechanistic`
    lower, size : arrays, shape (n_places,)
        Group and size of the places (see `theory_sizes`)
    years, states : arrays, shape (n_places,), optional
        Year and state of each place; required if the fit has effects. A state
        left out of the fit gets a mean of 0; a year or state the fit has never
        seen is an error.
    """
    lower = np.asarray(lower, dtype=bool)
    size = np.asarray(size, dtype=float)
    log_effect = np.zeros(len(size))
    left_out = np.zeros(len(size), dtype=bool)
    if fit["year_effects"] or fit["state_effects"]:
        if years is None or states is None:
            raise ValueError("this fit has effects: give the years and states")
        for i, (year, state) in enumerate(zip(years, states)):
            if state in fit["dropped_states"]:
                left_out[i] = True
            elif (
                year not in fit["year_effects"]
                or state not in fit["state_effects"]
            ):
                raise ValueError(
                    f"year {year} or state {state} is not in the fit"
                )
            else:
                log_effect[i] = (
                    fit["year_effects"][year] + fit["state_effects"][state]
                )
    log_threshold = fit["log_threshold"]
    z = np.where(lower, size, 0.0)
    # Both groups are written as exp(effect) times a number, and with a slope
    # of 0 the number is the same, exp(log_threshold), in both: the places
    # whose means should tie (those at the ceiling and the lower-coverage ones)
    # then get exactly the same mean, even with effects.
    lower_mean = np.exp(log_effect) * np.exp(log_threshold + fit["slope"] * z)
    higher_size = np.where(lower, 1.0, size)
    higher_mean = np.exp(log_effect) * np.minimum(
        higher_size, np.exp(log_threshold)
    )
    return np.where(left_out, 0.0, np.where(lower, lower_mean, higher_mean))


def fit_mechanistic(lower, size, cases, years=None, states=None):
    """
    Fit the mechanistic model (see the module docstring) to the observed
    numbers of cases of places.

    Parameters:
    -----------
    lower, size : arrays, shape (n_places,)
        Group and size of each place (see `theory_sizes`)
    cases : array, shape (n_places,)
        Observed outbreak cases of each place
    years, states : arrays, shape (n_places,), optional
        Year and state of each place. Give both to fit year and state effects,
        neither to fit none.

    Returns:
    --------
    dict with
      'log_threshold': log of the threshold size (the threshold size is in
          cases)
      'slope': per unit of z; 0 if the best allowed slope is 0
      'year_effects', 'state_effects': dicts from year or state to the effect
          on the log scale, adding up to 0 over the years and over the
          states; empty without effects
      'dropped_states': states left out because none of their places has cases
      'mean': mean cases of each place, not limited by the population, with 0
          for the places of dropped states
      'log_likelihood': Poisson log-likelihood of `cases` under 'mean'
      'n_parameters': k, as defined in the module docstring
      'search': diagnostics of the search for the threshold size: the best
          value, the other peaks within PEAK_WINDOW of it with how far below
          the best they are, and how far below the best the fit is at each
          bound

    Raises an error if a fit does not converge, or if the search for the
    threshold size cannot be trusted: its best value is within one grid step of
    a bound, or the fit at a bound is less than BOUND_DROP below the best.
    """
    lower = np.asarray(lower, dtype=bool)
    size = np.asarray(size, dtype=float)
    cases = np.asarray(cases, dtype=float)
    if (years is None) != (states is None):
        raise ValueError("give both years and states, or neither")
    if years is None:
        effects = np.zeros((len(cases), 0))
        kept = np.ones(len(cases), dtype=bool)
        dropped, names, levels = [], [], {}
    else:
        effects, kept, dropped, names, levels = _effects_design(
            years, states, cases
        )
    k_lower, k_size, k_cases = lower[kept], size[kept], cases[kept]

    # The search for the threshold size (see LOWEST_THRESHOLD_SIZE for the
    # bounds): first on a fine grid.
    low = np.log(LOWEST_THRESHOLD_SIZE)
    high = np.log(k_cases.max()) if k_cases.max() > 0 else low
    if high <= low + 10 * GRID_STEP:
        raise ValueError("the data have too few cases to fit the model")
    grid = np.arange(low, high + 1e-9, GRID_STEP)

    def profile(log_threshold):
        """Best log-likelihood for a threshold size (the slope, the effects
        are fitted); without the term that does not depend on them."""
        result = _fit_at_threshold(
            k_lower, k_size, k_cases, effects, log_threshold
        )
        if not result["converged"]:
            raise RuntimeError(
                "the fit did not converge at a threshold size of "
                f"{np.exp(log_threshold):.4g} cases"
            )
        return result["objective"]

    values = np.array([profile(g) for g in grid])
    best_index = int(np.argmax(values))
    if best_index < 1 or best_index > len(grid) - 2:
        raise RuntimeError(
            "the best threshold size is at a bound of the search "
            f"({np.exp(low):.4g} to {np.exp(high):.4g} cases)"
        )
    # Then every peak of the likelihood close to the best value is refined,
    # because the likelihood can have several peaks.
    peaks = {best_index} | {
        i
        for i in range(1, len(grid) - 1)
        if values[i] >= values[i - 1]
        and values[i] > values[i + 1]
        and values[best_index] - values[i] < PEAK_WINDOW
    }
    refined = []
    for i in sorted(peaks):
        result = optimize.minimize_scalar(
            lambda g: -profile(g),
            bounds=(grid[i - 1], grid[i + 1]),
            method="bounded",
            options={"xatol": 1e-6},
        )
        refined.append((float(result.x), -float(result.fun)))
    refined.sort(key=lambda peak: -peak[1])
    best_log_threshold, best_value = refined[0]
    search = dict(
        best_log_threshold=best_log_threshold,
        other_peaks=[(g, v - best_value) for g, v in refined[1:]],
        drop_at_lowest_bound=float(values[0] - best_value),
        drop_at_highest_bound=float(values[-1] - best_value),
    )
    if (  # type: ignore
        max(search["drop_at_lowest_bound"], search["drop_at_highest_bound"])
        > -BOUND_DROP
    ):
        raise RuntimeError(
            "the fit is not much worse at a bound of the search than at the "
            f"best threshold size ({search['drop_at_lowest_bound']:.1f} and "
            f"{search['drop_at_highest_bound']:.1f} log-likelihood units): "
            "the bounds are too narrow"
        )

    final = _fit_at_threshold(
        k_lower, k_size, k_cases, effects, best_log_threshold
    )
    n_columns = final["design"].shape[1]  # type: ignore
    if n_columns and np.linalg.matrix_rank(final["design"]) != n_columns:
        raise RuntimeError("the columns of the regression are not independent")

    # The effect of each level: the coefficient of its column, and for the last
    # level of the years and of the states (which has no column), minus the sum
    # of the others, so that the effects of the years add up to 0, and so do
    # those of the states.
    effects_by_kind = {"year": {}, "state": {}}
    for (kind, level), coefficient in zip(names, final["effect_coefficients"]):  # type: ignore
        effects_by_kind[kind][level] = float(coefficient)
    for kind, kind_levels in levels.items():
        others = effects_by_kind[kind].values()
        effects_by_kind[kind][kind_levels[-1]] = -float(sum(others))
    year_effects = effects_by_kind["year"]
    state_effects = effects_by_kind["state"]

    fit: dict[str, Any] = dict(
        log_threshold=best_log_threshold,
        slope=final["slope"],
        year_effects=year_effects,
        state_effects=state_effects,
        dropped_states=dropped,
        search=search,
        # the columns of the fit (the slope only if above 0, and the effects),
        # one for the threshold size, one for each state left out
        n_parameters=n_columns + 1 + len(dropped),
    )
    fit["mean"] = predict_mean(fit, lower, size, years, states)
    fit["log_likelihood"] = poisson_log_likelihood(cases, fit["mean"])
    return fit
