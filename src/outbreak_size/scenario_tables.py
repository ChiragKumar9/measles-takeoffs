"""
The supplemental tables of the future projections: for each vaccination
scenario (and intervention) and year, the expected cases with importations,
their percent change against the baseline, and the risk percentiles of the
counties and states (see analysis/figure_validation_and_risk_projections.py
and analysis/figure_nonmedical_exemption_policy_interventions.py, which load
the posterior draws, call these functions and write the tables).

The draws are jax arrays of shape (n_draws, n_counties); call
numpyro.enable_x64() first, as the scripts do, to compute in double precision.
The tables are built one year at a time and the draws freed after each use,
because they are large.
"""

import jax.numpy as jnp
import numpy as np
import polars as pl

from outbreak_size.ranking import risk_percentile
from outbreak_size.validation import require_same_fips


def cases_with_importations(predictions, probability, counts):
    """
    The expected cases of each county in each draw: the outbreak cases the
    model predicts if there is an importation, times the probability of one,
    plus the importations themselves. All the arrays are (n_draws,
    n_counties).
    """
    return predictions * probability + counts


def percent_change(intervention, baseline):
    """The percent change from `baseline` to `intervention` cases (0 where
    the baseline has no cases)."""
    return jnp.where(
        baseline > 0, ((intervention - baseline) / baseline) * 100, 0.0
    )


def draw_summary(samples, axis=None):
    """The mean, median and 2.5% and 97.5% percentiles of the draws (along
    `axis`, the whole array if None)."""
    return (
        jnp.mean(samples, axis=axis),
        jnp.median(samples, axis=axis),
        jnp.percentile(samples, 2.5, axis=axis),
        jnp.percentile(samples, 97.5, axis=axis),
    )


def risk_ranks(cases, population):
    """The percentile (0 to 100, the highest being the highest risk) of
    each unit by its mean cases over the draws and by its mean incidence (its
    cases per person). `cases` is (n_draws, n_units) and `population` (n_units,)."""
    count_rank = risk_percentile(np.array(cases))
    incidence_rank = risk_percentile(
        np.array(cases / population.reshape(1, -1))
    )
    return count_rank, incidence_rank


def _scenario_samples(vac_scenario, year, all_future_samples, n_counties):
    # no scenario bounds here. They used to keep a worsening scenario at or
    # above the baseline's draw of the first projected year (jnp.maximum, county
    # by county) and an improving one at or below it (jnp.minimum); see
    # SCENARIO BOUNDS at the top of analysis/figure_validation_and_risk_
    # projections.py for the rules, why they were removed, and how to restore
    raw = all_future_samples[(vac_scenario, year)]
    if raw.shape[1] != n_counties:
        raise ValueError(
            f"Shape mismatch for {vac_scenario} {year}: "
            f"expected {n_counties} counties, got {raw.shape[1]}"
        )
    return raw


def _float_column(values):
    """The values (a jax or numpy array) as the float64 column of a table."""
    return np.asarray(values, dtype=np.float64)


def _constant_column(value, n_rows):
    """A column of n_rows integers (the year of a block of rows)."""
    return pl.Series([value] * n_rows, dtype=pl.Int64)


# --- the vaccination scenarios (figure_validation_and_risk_projections) -------


def vaccination_scenario_effects(
    vaccination_scenarios,
    future_years,
    future_base_df,
    all_future_samples,
    importation_samples,
    importation_count_samples,
    scenario_labels,
):
    """
    The risk percentiles of each county in each vaccination scenario and
    year: one row per county, year and scenario, with the county's
    importation probability (mean, median and 95% interval over the draws) and
    its risk percentile by cases and by incidence.

    `future_base_df` has the counties (fips, state, county and pop) in the
    order of the columns of the draws, `all_future_samples` the outbreak
    draws by (scenario, year) and `scenario_labels` the name each scenario
    has in the table.
    """
    fips_to_location = {}
    for row in future_base_df.select(
        ["fips", "state", "county", "pop"]
    ).iter_rows(named=True):
        fips_to_location[row["fips"]] = {
            "state": row["state"],
            "county": row["county"],
            "pop": row["pop"],
        }

    common_fips = future_base_df["fips"].to_list()
    common_pop = jnp.array(future_base_df["pop"].to_list())

    # Calculate importation probability statistics per county
    # Shape: (n_counties,)
    (
        importation_prob_mean,
        importation_prob_median,
        importation_prob_p025,
        importation_prob_p975,
    ) = draw_summary(importation_samples, axis=0)

    # the county columns, the same in every block
    locations = [fips_to_location[fips] for fips in common_fips]
    county_columns = {
        "state": [location["state"] for location in locations],
        "county": [location["county"] for location in locations],
        "fips": common_fips,
    }
    probability_columns = {
        "importation_probability_mean": _float_column(importation_prob_mean),
        "importation_probability_median": _float_column(
            importation_prob_median
        ),
        "importation_probability_p025": _float_column(importation_prob_p025),
        "importation_probability_p975": _float_column(importation_prob_p975),
    }

    # one block of rows (a frame, not a list of dicts) for each scenario and
    # year
    blocks = []

    # Loop through all scenarios
    for vac_scenario in vaccination_scenarios:
        print(f"Processing vaccination scenario: {vac_scenario}")

        for year in future_years:
            raw = _scenario_samples(
                vac_scenario, year, all_future_samples, len(common_fips)
            )

            # Calculate cases for this year
            # Shape: (n_samples, n_counties)
            cases_county = cases_with_importations(
                raw, importation_samples, importation_count_samples
            )

            # --- SINGLE-YEAR risk percentiles ---
            count_rank, incidence_rank = risk_ranks(cases_county, common_pop)

            blocks.append(
                pl.DataFrame(
                    {
                        **county_columns,
                        "year": _constant_column(year, len(common_fips)),
                        "vaccination_scenario": [scenario_labels[vac_scenario]]
                        * len(common_fips),
                        **probability_columns,
                        "count_risk_percentile": _float_column(count_rank),
                        "incidence_risk_percentile": _float_column(
                            incidence_rank
                        ),
                    }
                )
            )

            # Free memory
            del cases_county, count_rank, incidence_rank

    # Sort for easier reading
    return pl.concat(blocks).sort(
        ["vaccination_scenario", "year", "state", "county"],
        maintain_order=True,
    )


def _state_columns(common_fips, fips_to_state):
    """The states of the counties (sorted) and, for each, the columns of the
    counties in it."""
    fips_states = [fips_to_state[f] for f in common_fips]
    unique_states = sorted(list(set(fips_states)))
    state_to_col_indices = {}
    for state in unique_states:
        state_to_col_indices[state] = jnp.array(
            [i for i, s in enumerate(fips_states) if s == state]
        )
    return unique_states, state_to_col_indices


def _float_summary(samples):
    mean, median, p025, p975 = draw_summary(samples)
    return {
        "mean": float(mean),
        "median": float(median),
        "p025": float(p025),
        "p975": float(p975),
    }


def state_vaccination_scenario_effects(
    vaccination_scenarios,
    future_years,
    future_base_df,
    all_future_samples,
    importation_samples,
    importation_count_samples,
    scenario_labels,
):
    """
    The risk percentiles of each state in each vaccination scenario and
    year: one row per state, year and scenario, with the state's probability
    of at least one import and expected number of counties with an import
    (mean, median and 95% interval over the draws) and its risk percentile
    by cases and by incidence. The inputs are those of
    `vaccination_scenario_effects`.
    """
    fips_to_state = dict(zip(future_base_df["fips"], future_base_df["state"]))

    # Calculate state populations
    state_pop_df = future_base_df.group_by("state").agg(
        pl.col("pop").sum().alias("state_pop")
    )
    state_to_pop = dict(zip(state_pop_df["state"], state_pop_df["state_pop"]))

    common_fips = future_base_df["fips"].to_list()
    unique_states, state_to_col_indices = _state_columns(
        common_fips, fips_to_state
    )

    # State-level importation quantities from the county probabilities
    # Shape: (n_samples, n_counties) -> aggregate to (n_samples, n_states)
    # - the probability that the state has at least one import: one minus the
    #   probability that no county has one, for each posterior draw
    # - the expected number of counties with an import: the sum of the county
    #   probabilities (not a probability, it can exceed 1)
    state_importation_stats = {}
    state_expected_counties_stats = {}
    for state in unique_states:
        col_indices = state_to_col_indices[state]
        county_probabilities = importation_samples[:, col_indices]
        state_importation_stats[state] = _float_summary(
            1 - jnp.prod(1 - county_probabilities, axis=1)
        )
        state_expected_counties_stats[state] = _float_summary(
            jnp.sum(county_probabilities, axis=1)
        )

    all_rows = []

    # Loop through all scenarios
    for vac_scenario in vaccination_scenarios:
        print(f"Processing state-level vaccination scenario: {vac_scenario}")

        for year in future_years:
            raw = _scenario_samples(
                vac_scenario, year, all_future_samples, len(common_fips)
            )

            # Calculate cases for this year at county level
            # Shape: (n_samples, n_counties)
            cases_county = cases_with_importations(
                raw, importation_samples, importation_count_samples
            )

            # Aggregate to state level for this year
            cases_state_year = {}
            for state in unique_states:
                col_indices = state_to_col_indices[state]
                # Sum across counties in this state for each sample
                # Shape: (n_samples,)
                cases_state_year[state] = jnp.sum(
                    cases_county[:, col_indices], axis=1
                )

            del cases_county

            # --- SINGLE-YEAR risk percentiles at state level ---
            # Stack all states' samples for this year
            # Shape: (n_samples, n_states)
            cases_state_year_array = jnp.stack(
                [cases_state_year[state] for state in unique_states], axis=1
            )
            state_pops = jnp.array(
                [state_to_pop[state] for state in unique_states]
            )
            count_rank_states, incidence_rank_states = risk_ranks(
                cases_state_year_array, state_pops
            )

            # Create rows for each state
            for idx, state in enumerate(unique_states):
                all_rows.append(
                    {
                        "state": state,
                        "year": year,
                        "vaccination_scenario": scenario_labels[vac_scenario],
                        "importation_probability_mean": state_importation_stats[
                            state
                        ]["mean"],
                        "importation_probability_median": state_importation_stats[
                            state
                        ]["median"],
                        "importation_probability_p025": state_importation_stats[
                            state
                        ]["p025"],
                        "importation_probability_p975": state_importation_stats[
                            state
                        ]["p975"],
                        "expected_counties_with_import_mean": state_expected_counties_stats[
                            state
                        ]["mean"],
                        "expected_counties_with_import_median": state_expected_counties_stats[
                            state
                        ]["median"],
                        "expected_counties_with_import_p025": state_expected_counties_stats[
                            state
                        ]["p025"],
                        "expected_counties_with_import_p975": state_expected_counties_stats[
                            state
                        ]["p975"],
                        "count_risk_percentile": float(count_rank_states[idx]),
                        "incidence_risk_percentile": float(
                            incidence_rank_states[idx]
                        ),
                    }
                )

            # Free memory
            del cases_state_year, cases_state_year_array
            del count_rank_states, incidence_rank_states

    # Sort for easier reading
    return pl.DataFrame(all_rows).sort(
        ["vaccination_scenario", "year", "state"]
    )


# --- the interventions (figure_nonmedical_exemption_policy_interventions) -----


def _common_counties(first_fips_list, baseline_fips, fips_to_idx):
    """The counties that the baseline and the intervention both project (in
    the baseline's order) and their columns in each."""
    proj_fips_to_idx = {f: j for j, f in enumerate(first_fips_list)}
    common_fips = [f for f in baseline_fips if f in proj_fips_to_idx]
    baseline_col_idx = jnp.array([fips_to_idx[f] for f in common_fips])
    proj_col_idx = jnp.array([proj_fips_to_idx[f] for f in common_fips])
    return common_fips, baseline_col_idx, proj_col_idx


def county_intervention_effects(
    interventions,
    vaccination_scenarios,
    future_years,
    load_baseline,
    load_intervention,
    locations,
    importation_samples,
    importation_count_samples,
    baseline_fips,
    scenario_labels,
    intervention_labels,
):
    """
    The percent change in outbreak size of each county under each
    intervention, in each vaccination scenario and year: one row per county,
    year, scenario and intervention, with the change in that year and the
    cumulative change since the first year (mean, median and 95% interval over
    the draws), and the county's risk percentiles under the intervention.

    The draws are loaded one year at a time: `load_baseline(scenario, year)`
    returns the baseline draws (one column per county of `baseline_fips`) and
    `load_intervention(intervention, scenario, year)` the intervention's
    {"fips": ..., "predictions": ...}, whose counties must be the same in
    every year. `locations` has the state, county and population of each
    county (fips, state, county and pop).
    """
    fips_to_idx = {f: i for i, f in enumerate(baseline_fips)}
    imp = importation_samples
    imp_count = importation_count_samples

    fips_to_location = {}
    for row in locations.iter_rows(named=True):
        fips_to_location[row["fips"]] = {
            "state": row["state"],
            "county": row["county"],
            "pop": row["pop"],
        }

    # one block of rows (a frame, not a list of dicts) for each scenario,
    # intervention and year
    blocks = []

    # Loop through all combinations
    for vac_scenario in vaccination_scenarios:
        print(f"Processing vaccination scenario: {vac_scenario}")

        for intervention in interventions:
            print(f"  Processing intervention: {intervention}")

            # Track cumulative cases across years for each county
            cumulative_baseline_by_county = None
            cumulative_intervention_by_county = None

            # Get FIPS mapping from first year
            first_year = future_years[0]
            first_pkg = load_intervention(
                intervention, vac_scenario, first_year
            )
            first_fips_list = first_pkg["fips"]
            common_fips, baseline_col_idx, proj_col_idx = _common_counties(
                first_fips_list, baseline_fips, fips_to_idx
            )

            # Get population for common counties
            common_pop = jnp.array(
                [fips_to_location[f]["pop"] for f in common_fips]
            )
            # the county columns of the rows of every year
            county_locations = [
                fips_to_location.get(
                    fips, {"state": "Unknown", "county": "Unknown", "pop": 0}
                )
                for fips in common_fips
            ]
            county_columns = {
                "state": [location["state"] for location in county_locations],
                "county": [
                    location["county"] for location in county_locations
                ],
                "fips": common_fips,
            }

            del first_pkg  # Free memory

            for year in future_years:
                baseline_samples = load_baseline(vac_scenario, year)
                intervention_pkg = load_intervention(
                    intervention, vac_scenario, year
                )
                require_same_fips(
                    intervention_pkg["fips"],
                    first_fips_list,
                    f"{intervention} {vac_scenario} {year}",
                )
                pred = intervention_pkg["predictions"]

                # Calculate cases for this year
                # Baseline: (n_samples, n_common_counties)
                b_county = cases_with_importations(
                    baseline_samples[:, baseline_col_idx],
                    imp[:, baseline_col_idx],
                    imp_count[:, baseline_col_idx],
                )

                # Intervention: (n_samples, n_common_counties)
                i_county = cases_with_importations(
                    pred[:, proj_col_idx],
                    imp[:, baseline_col_idx],
                    imp_count[:, baseline_col_idx],
                )
                # NOTE: no intervention bound here (INTERVENTION BOUND in
                # SCENARIO BOUNDS at the top of analysis/figure_validation_
                # and_risk_projections.py): i_county used to be capped at
                # b_county with jnp.minimum

                # Free memory
                del baseline_samples, intervention_pkg, pred

                # --- SINGLE-YEAR risk percentiles ---
                intervention_count_rank, intervention_incidence_rank = (
                    risk_ranks(i_county, common_pop)
                )

                # --- SINGLE-YEAR percent change (intervention vs baseline for this year only) ---
                # Shape: (n_samples, n_common_counties)
                pct_change_samples = percent_change(i_county, b_county)
                (
                    pct_change_mean,
                    pct_change_median,
                    pct_change_p025,
                    pct_change_p975,
                ) = draw_summary(pct_change_samples, axis=0)

                del pct_change_samples

                # --- CUMULATIVE percent change ---
                # Accumulate across years
                if cumulative_baseline_by_county is None:
                    cumulative_baseline_by_county = b_county
                    cumulative_intervention_by_county = i_county
                else:
                    cumulative_baseline_by_county = (
                        cumulative_baseline_by_county + b_county
                    )
                    cumulative_intervention_by_county = (
                        cumulative_intervention_by_county + i_county
                    )

                # Calculate cumulative percent change for EACH SAMPLE
                # Shape: (n_samples, n_common_counties)
                cumulative_pct_change_samples = percent_change(
                    cumulative_intervention_by_county,
                    cumulative_baseline_by_county,
                )
                (
                    cumulative_pct_change_mean,
                    cumulative_pct_change_median,
                    cumulative_pct_change_p025,
                    cumulative_pct_change_p975,
                ) = draw_summary(cumulative_pct_change_samples, axis=0)

                # The rows of the counties of this year, as a frame
                blocks.append(
                    pl.DataFrame(
                        {
                            **county_columns,
                            "year": _constant_column(year, len(common_fips)),
                            "vaccination_scenario": [
                                scenario_labels[vac_scenario]
                            ]
                            * len(common_fips),
                            "intervention": [intervention_labels[intervention]]
                            * len(common_fips),
                            "intervention_count_risk_percentile": _float_column(
                                intervention_count_rank
                            ),
                            "intervention_incidence_risk_percentile": _float_column(
                                intervention_incidence_rank
                            ),
                            "pct_change_mean": _float_column(pct_change_mean),
                            "pct_change_median": _float_column(
                                pct_change_median
                            ),
                            "pct_change_p025": _float_column(pct_change_p025),
                            "pct_change_p975": _float_column(pct_change_p975),
                            "cumulative_pct_change_mean": _float_column(
                                cumulative_pct_change_mean
                            ),
                            "cumulative_pct_change_median": _float_column(
                                cumulative_pct_change_median
                            ),
                            "cumulative_pct_change_p025": _float_column(
                                cumulative_pct_change_p025
                            ),
                            "cumulative_pct_change_p975": _float_column(
                                cumulative_pct_change_p975
                            ),
                        }
                    )
                )

                # Free memory after processing this year
                del b_county, i_county, cumulative_pct_change_samples
                del intervention_count_rank, intervention_incidence_rank

            # Free cumulative arrays before moving to next intervention
            del (
                cumulative_baseline_by_county,
                cumulative_intervention_by_county,
            )

    # Sort for easier reading
    return pl.concat(blocks).sort(
        ["vaccination_scenario", "intervention", "year", "state", "county"],
        maintain_order=True,
    )


def state_intervention_effects(
    interventions,
    vaccination_scenarios,
    future_years,
    load_baseline,
    load_intervention,
    locations,
    importation_samples,
    importation_count_samples,
    baseline_fips,
    scenario_labels,
    intervention_labels,
):
    """
    The percent change in outbreak size of each state under each
    intervention, in each vaccination scenario and year: one row per state,
    year, scenario and intervention, as `county_intervention_effects` but
    for the sum of the state's counties (`locations` has fips, state and pop of
    each county). The inputs are those of `county_intervention_effects`.
    """
    fips_to_idx = {f: i for i, f in enumerate(baseline_fips)}
    imp = importation_samples
    imp_count = importation_count_samples

    fips_to_state = dict(zip(locations["fips"], locations["state"]))

    # Calculate state populations
    state_pop = locations.group_by("state").agg(
        pl.col("pop").sum().alias("state_pop")
    )
    state_to_pop = dict(zip(state_pop["state"], state_pop["state_pop"]))

    all_rows = []

    # Loop through all combinations
    for vac_scenario in vaccination_scenarios:
        print(f"Processing vaccination scenario: {vac_scenario}")

        for intervention in interventions:
            print(f"  Processing intervention: {intervention}")

            # Track cumulative cases by STATE across years
            cumulative_baseline_by_state = {}
            cumulative_intervention_by_state = {}

            # Get FIPS mapping from first year
            first_year = future_years[0]
            first_pkg = load_intervention(
                intervention, vac_scenario, first_year
            )
            first_fips_list = first_pkg["fips"]
            common_fips, baseline_col_idx, proj_col_idx = _common_counties(
                first_fips_list, baseline_fips, fips_to_idx
            )

            # Create mapping from common_fips to states
            fips_states = [
                fips_to_state.get(f, "Unknown") for f in common_fips
            ]
            unique_states = list(set(fips_states))

            # Pre-compute which column indices belong to which state
            state_to_col_indices = {}
            for state in unique_states:
                state_to_col_indices[state] = jnp.array(
                    [i for i, s in enumerate(fips_states) if s == state]
                )

            del first_pkg

            for year in future_years:
                baseline_samples = load_baseline(vac_scenario, year)
                intervention_pkg = load_intervention(
                    intervention, vac_scenario, year
                )
                require_same_fips(
                    intervention_pkg["fips"],
                    first_fips_list,
                    f"{intervention} {vac_scenario} {year}",
                )
                pred = intervention_pkg["predictions"]

                # Calculate cases for this year (county level)
                # Shape: (n_samples, n_common_counties)
                b_county = cases_with_importations(
                    baseline_samples[:, baseline_col_idx],
                    imp[:, baseline_col_idx],
                    imp_count[:, baseline_col_idx],
                )
                i_county = cases_with_importations(
                    pred[:, proj_col_idx],
                    imp[:, baseline_col_idx],
                    imp_count[:, baseline_col_idx],
                )
                # NOTE: no intervention bound here (INTERVENTION BOUND in
                # SCENARIO BOUNDS at the top of analysis/figure_validation_
                # and_risk_projections.py): i_county used to be capped at
                # b_county with jnp.minimum

                # Free county-level data immediately
                del baseline_samples, intervention_pkg, pred

                # Aggregate to state level for this year
                b_state_year = {}
                i_state_year = {}

                for state in unique_states:
                    col_indices = state_to_col_indices[state]
                    # Sum across counties in this state for each sample
                    # Shape: (n_samples,)
                    b_state_year[state] = jnp.sum(
                        b_county[:, col_indices], axis=1
                    )
                    i_state_year[state] = jnp.sum(
                        i_county[:, col_indices], axis=1
                    )

                # Free county arrays
                del b_county, i_county

                # --- SINGLE-YEAR risk percentiles at state level ---
                # Stack all states' samples for this year
                # Shape: (n_samples, n_states)
                i_state_year_array = jnp.stack(
                    [i_state_year[state] for state in unique_states], axis=1
                )
                state_pops = jnp.array(
                    [state_to_pop[state] for state in unique_states]
                )
                (
                    intervention_count_rank_states,
                    intervention_incidence_rank_states,
                ) = risk_ranks(i_state_year_array, state_pops)

                # Accumulate across years at state level
                if not cumulative_baseline_by_state:
                    cumulative_baseline_by_state = b_state_year
                    cumulative_intervention_by_state = i_state_year
                else:
                    for state in unique_states:
                        cumulative_baseline_by_state[state] = (
                            cumulative_baseline_by_state[state]
                            + b_state_year[state]
                        )
                        cumulative_intervention_by_state[state] = (
                            cumulative_intervention_by_state[state]
                            + i_state_year[state]
                        )

                # Calculate percent changes at STATE level for each sample
                for idx, state in enumerate(unique_states):
                    # --- SINGLE-YEAR percent change (intervention vs baseline for this year only) ---
                    # Shape: (n_samples,)
                    pct_change = _float_summary(
                        percent_change(
                            i_state_year[state], b_state_year[state]
                        )
                    )

                    # --- CUMULATIVE percent change ---
                    # Shape: (n_samples,)
                    cumulative_pct_change = _float_summary(
                        percent_change(
                            cumulative_intervention_by_state[state],
                            cumulative_baseline_by_state[state],
                        )
                    )

                    all_rows.append(
                        {
                            "state": state,
                            "year": year,
                            "vaccination_scenario": scenario_labels[
                                vac_scenario
                            ],
                            "intervention": intervention_labels[intervention],
                            "intervention_count_risk_percentile": float(
                                intervention_count_rank_states[idx]
                            ),
                            "intervention_incidence_risk_percentile": float(
                                intervention_incidence_rank_states[idx]
                            ),
                            "pct_change_mean": pct_change["mean"],
                            "pct_change_median": pct_change["median"],
                            "pct_change_p025": pct_change["p025"],
                            "pct_change_p975": pct_change["p975"],
                            "cumulative_pct_change_mean": cumulative_pct_change[
                                "mean"
                            ],
                            "cumulative_pct_change_median": cumulative_pct_change[
                                "median"
                            ],
                            "cumulative_pct_change_p025": cumulative_pct_change[
                                "p025"
                            ],
                            "cumulative_pct_change_p975": cumulative_pct_change[
                                "p975"
                            ],
                        }
                    )

                # Free year-level state data
                del b_state_year, i_state_year
                del i_state_year_array
                del (
                    intervention_count_rank_states,
                    intervention_incidence_rank_states,
                )

            # Free cumulative state arrays before moving to next intervention
            del cumulative_baseline_by_state, cumulative_intervention_by_state

    # Sort for easier reading
    return pl.DataFrame(all_rows).sort(
        ["vaccination_scenario", "intervention", "year", "state"]
    )


def _nationwide_row(scenario_label, intervention_label, year, single, total):
    return {
        "vaccination_scenario": scenario_label,
        "intervention": intervention_label,
        "year": year,
        "single_year_cases_mean": single["mean"],
        "single_year_cases_median": single["median"],
        "single_year_cases_p025": single["p025"],
        "single_year_cases_p975": single["p975"],
        "cumulative_cases_mean": total["mean"],
        "cumulative_cases_median": total["median"],
        "cumulative_cases_p025": total["p025"],
        "cumulative_cases_p975": total["p975"],
    }


def nationwide_summary(
    interventions,
    vaccination_scenarios,
    future_years,
    load_baseline,
    load_intervention,
    importation_samples,
    importation_count_samples,
    baseline_fips,
    scenario_labels,
    intervention_labels,
):
    """
    The nationwide cases (the sum of all counties) with every vaccination
    scenario and intervention: one row per scenario, intervention ("baseline"
    for none) and year, with the mean, median and 95% interval over the draws
    of the cases in that year and cumulative since the first year. The inputs
    are those of `county_intervention_effects`.
    """
    fips_to_idx = {f: i for i, f in enumerate(baseline_fips)}
    imp = importation_samples
    imp_count = importation_count_samples

    all_rows = []

    # First, process baseline scenarios (no intervention)
    for vac_scenario in vaccination_scenarios:
        print(f"Processing baseline for vaccination scenario: {vac_scenario}")

        cumulative_baseline_nationwide = None

        for year in future_years:
            baseline_samples = load_baseline(vac_scenario, year)

            # Calculate cases for this year across all counties
            # Shape: (n_samples, n_counties)
            b_nationwide = cases_with_importations(
                baseline_samples, imp, imp_count
            )

            # Sum across all counties for each sample
            # Shape: (n_samples,)
            b_nationwide_total = jnp.sum(b_nationwide, axis=1)

            del baseline_samples, b_nationwide

            # Calculate single-year summary statistics
            single_year = _float_summary(b_nationwide_total)

            # Accumulate across years
            if cumulative_baseline_nationwide is None:
                cumulative_baseline_nationwide = b_nationwide_total
            else:
                cumulative_baseline_nationwide = (
                    cumulative_baseline_nationwide + b_nationwide_total
                )

            # Append the results for this year
            all_rows.append(
                _nationwide_row(
                    scenario_labels[vac_scenario],
                    "baseline",
                    year,
                    single_year,
                    _float_summary(cumulative_baseline_nationwide),
                )
            )

            del b_nationwide_total

        del cumulative_baseline_nationwide

    # Now process interventions
    for vac_scenario in vaccination_scenarios:
        print(
            f"Processing interventions for vaccination scenario: {vac_scenario}"
        )

        for intervention in interventions:
            print(f"  Processing intervention: {intervention}")

            cumulative_intervention_nationwide = None

            # Get FIPS mapping from first year
            first_year = future_years[0]
            first_pkg = load_intervention(
                intervention, vac_scenario, first_year
            )
            first_fips_list = first_pkg["fips"]
            common_fips, baseline_col_idx, proj_col_idx = _common_counties(
                first_fips_list, baseline_fips, fips_to_idx
            )
            del first_pkg

            for year in future_years:
                intervention_pkg = load_intervention(
                    intervention, vac_scenario, year
                )
                require_same_fips(
                    intervention_pkg["fips"],
                    first_fips_list,
                    f"{intervention} {vac_scenario} {year}",
                )
                pred = intervention_pkg["predictions"]

                # Calculate intervention cases
                i_county = cases_with_importations(
                    pred[:, proj_col_idx],
                    imp[:, baseline_col_idx],
                    imp_count[:, baseline_col_idx],
                )
                # NOTE: no intervention bound here (INTERVENTION BOUND in
                # SCENARIO BOUNDS at the top of analysis/figure_validation_
                # and_risk_projections.py): i_county used to be capped at
                # b_county with jnp.minimum (the baseline's cases of the
                # county, which this function does not compute any more)

                del intervention_pkg, pred

                # Sum across all counties for each sample
                # Shape: (n_samples,)
                i_nationwide_total = jnp.sum(i_county, axis=1)

                del i_county

                # Calculate single-year summary statistics
                single_year = _float_summary(i_nationwide_total)

                # Accumulate across years
                if cumulative_intervention_nationwide is None:
                    cumulative_intervention_nationwide = i_nationwide_total
                else:
                    cumulative_intervention_nationwide = (
                        cumulative_intervention_nationwide + i_nationwide_total
                    )

                # Append the results for this year
                all_rows.append(
                    _nationwide_row(
                        scenario_labels[vac_scenario],
                        intervention_labels[intervention],
                        year,
                        single_year,
                        _float_summary(cumulative_intervention_nationwide),
                    )
                )

                del i_nationwide_total

            del cumulative_intervention_nationwide

    # Sort for easier reading
    return pl.DataFrame(all_rows).sort(
        ["vaccination_scenario", "intervention", "year"]
    )
