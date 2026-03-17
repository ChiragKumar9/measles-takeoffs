import numpy as np
import polars as pl
from scipy import stats


def extract_vaccine_effect(samples, df):
    """
    Extract vaccine effect (first IV) expanded to observation level.

    Returns
    -------
    vaccine_effect : array (n_samples, n_observations)
        Vaccine coverage effect for each observation
    vaccine_coef_by_group : array (n_samples, n_states)
        Vaccine coefficient for each state
    """
    # Get samples (assuming pooled IVs)
    iv_global = samples["iv_global"]
    iv_group_offsets = samples["iv_group_offsets"]
    iv_group_variabilities = samples["iv_group_variabilities"]

    # Get state groups for each observation
    state_groups = (
        df["state"]
        .replace(
            df["state"].unique(maintain_order=True),
            pl.arange(0, df["state"].n_unique()),
        )
        .cast(pl.Int64)
        .to_numpy()
    )

    # Reconstruct vaccine coefficient by group (first IV, index 0)
    vaccine_coef_by_group = (
        iv_group_variabilities[:, 0][:, None] * iv_group_offsets[:, :, 0]
        + iv_global[:, 0][:, None]
    )

    # Z-score the coverage values (like you do before model fitting)
    vaccine_coverage = df["estimated_mmr_cov"].to_numpy()
    vaccine_coverage_zscore = (
        vaccine_coverage - np.mean(vaccine_coverage)
    ) / np.std(vaccine_coverage)

    # Multiply by z-scored coverage
    vaccine_effect = (
        vaccine_coef_by_group[:, state_groups]
        * vaccine_coverage_zscore[None, :]
    )

    return vaccine_effect, vaccine_coef_by_group


def plot_vaccine_vs_spatial(samples, df, adjacency, fig, ax):
    """
    Plot vaccine marginal effect by state vs average total population implicated in outbreaks.
    X-axis: Average total population (focal county + neighbors with cases same year) across all outbreaks in state,
            weighted by total outbreak size (focal + neighbors).
    Y-axis: Vaccine marginal effect with error bars showing posterior uncertainty.
    Color: Whether any neighboring county had an outbreak (≥3 cases) in same year as focal county.
    """

    # Get vaccine coefficients
    _, vaccine_coef_by_group = extract_vaccine_effect(
        samples, df
    )  # (n_samples, n_states)

    # Get coverage SD for conversion from z-score to raw scale
    sigma_coverage = np.std(df["estimated_mmr_cov"].to_numpy())

    # Map state names to state indices
    unique_states = df["state"].unique(maintain_order=True)
    state_to_idx = {state: idx for idx, state in enumerate(unique_states)}

    # Calculate vaccine marginal effect for each state (with uncertainty)
    state_vaccine_effects_median = {}
    state_vaccine_effects_lower = {}
    state_vaccine_effects_upper = {}

    for state in unique_states:
        state_idx = state_to_idx[state]
        beta_vax_state = vaccine_coef_by_group[:, state_idx]

        # Marginal effect: % change per 1pp coverage increase
        marginal_effect = (np.exp(beta_vax_state / sigma_coverage) - 1) * 100

        state_vaccine_effects_median[state] = np.median(marginal_effect)
        state_vaccine_effects_lower[state] = np.percentile(
            marginal_effect, 2.5
        )
        state_vaccine_effects_upper[state] = np.percentile(
            marginal_effect, 97.5
        )

    # Calculate outbreak statistics per state
    all_fips = df["fips"].unique(maintain_order=True).to_numpy()

    # Create mapping from FIPS to state
    fips_to_state = {}
    for fips in all_fips:
        fips_state = df.filter(pl.col("fips") == fips)["state"][0]
        fips_to_state[fips] = fips_state

    # Get which counties had cases in which years, and their populations
    county_year_with_cases = {}  # (fips, year) -> population
    for row in df.iter_rows(named=True):
        if row["cases_outbreak"] > 0:
            county_year_with_cases[(row["fips"], row["year"])] = row["pop"]

    # Get mu (predicted outbreak size) for weighting
    mu = samples["mu"]  # (n_samples, n_obs)
    mu_median = np.median(mu, axis=0)  # (n_obs,)

    # Create mapping from (fips, year) to observation index
    obs_to_mu = {}
    for obs_idx, row in enumerate(df.iter_rows(named=True)):
        obs_to_mu[(row["fips"], row["year"])] = mu_median[obs_idx]

    # Calculate average population implicated and whether neighbors had outbreaks
    state_avg_population = {}
    state_has_neighbor_outbreak = {}

    for state in unique_states:
        # Get all counties in this state
        state_fips = [
            fips for fips in all_fips if fips_to_state[fips] == state
        ]

        weighted_populations = []
        has_neighbor_outbreak = False

        for fips in state_fips:
            # Get all years this county had cases
            county_years = (
                df.filter(
                    (pl.col("fips") == fips) & (pl.col("cases_outbreak") > 0)
                )["year"]
                .unique()
                .to_list()
            )

            for year in county_years:
                # Get index of this FIPS in the adjacency matrix
                fips_idx = np.where(all_fips == fips)[0]
                if len(fips_idx) == 0:
                    continue
                fips_idx = fips_idx[0]

                # Get all neighbor indices
                neighbor_indices = np.where(adjacency[fips_idx, :] > 0)[0]

                # Count neighbors + focal county with cases, sum total population and outbreak size
                total_population = 0
                total_outbreak_size = 0
                neighbor_had_outbreak = False

                # Add focal county
                if (fips, year) in county_year_with_cases:
                    total_population += county_year_with_cases[(fips, year)]
                    if (fips, year) in obs_to_mu:
                        total_outbreak_size += obs_to_mu[(fips, year)]

                # Add neighbors
                for neighbor_idx in neighbor_indices:
                    neighbor_fips = all_fips[neighbor_idx]
                    if (neighbor_fips, year) in county_year_with_cases:
                        total_population += county_year_with_cases[
                            (neighbor_fips, year)
                        ]
                        if (neighbor_fips, year) in obs_to_mu:
                            neighbor_outbreak_size = obs_to_mu[
                                (neighbor_fips, year)
                            ]
                            total_outbreak_size += neighbor_outbreak_size
                            # Check if neighbor had outbreak (≥3 cases)
                            if neighbor_outbreak_size >= 3:
                                neighbor_had_outbreak = True
                                has_neighbor_outbreak = True

                # Record this outbreak
                if total_population > 0 and total_outbreak_size > 0:
                    weighted_populations.append(
                        (total_population, total_outbreak_size)
                    )

        # Calculate weighted average
        if len(weighted_populations) > 0:
            total_weight = sum(w for _, w in weighted_populations)
            if total_weight > 0:
                state_avg_population[state] = (
                    sum(pop * w for pop, w in weighted_populations)
                    / total_weight
                )
            else:
                state_avg_population[state] = np.mean(
                    [pop for pop, _ in weighted_populations]
                )
        else:
            state_avg_population[state] = 0

        state_has_neighbor_outbreak[state] = has_neighbor_outbreak

    # Create lists for plotting
    x_values = np.array(
        [state_avg_population[state] for state in unique_states]
    )
    y_values = np.array(
        [state_vaccine_effects_median[state] for state in unique_states]
    )
    y_lower = np.array(
        [state_vaccine_effects_lower[state] for state in unique_states]
    )
    y_upper = np.array(
        [state_vaccine_effects_upper[state] for state in unique_states]
    )
    has_neighbor = np.array(
        [state_has_neighbor_outbreak[state] for state in unique_states]
    )

    # Calculate error bar lengths
    yerr_lower = y_values - y_lower
    yerr_upper = y_upper - y_values

    # Split by neighbor outbreak status
    ax.scatter(
        x_values[~has_neighbor],
        y_values[~has_neighbor],
        c="maroon",
        s=100,
        alpha=0.7,
        label="Isolated outbreak",
        zorder=3,
    )

    ax.scatter(
        x_values[has_neighbor],
        y_values[has_neighbor],
        c="cornflowerblue",
        s=100,
        label="Outbreak also in\nneighboring county",
        zorder=3,
    )

    ax.errorbar(
        x_values,
        y_values,
        yerr=[yerr_lower, yerr_upper],
        fmt="none",
        ecolor="gray",
        alpha=0.5,
        zorder=2,
    )

    # print correlation
    print(stats.pearsonr(x_values[has_neighbor], y_values[has_neighbor]))
    print(stats.pearsonr(x_values[~has_neighbor], y_values[~has_neighbor]))
    print(stats.pearsonr(x_values, y_values))

    # print correlation
    print(stats.spearmanr(x_values[has_neighbor], y_values[has_neighbor]))
    print(stats.spearmanr(x_values[~has_neighbor], y_values[~has_neighbor]))
    print(stats.spearmanr(x_values, y_values))

    # what about theil slopes
    print(stats.theilslopes(y_values[has_neighbor], x_values[has_neighbor]))
    print(stats.theilslopes(y_values[~has_neighbor], x_values[~has_neighbor]))
    print(stats.theilslopes(y_values, x_values))

    # add theil slopes regression line for has neighbor with confidence interval shaded
    slope, intercept, low_slope, high_slope = stats.theilslopes(
        y_values[has_neighbor], x_values[has_neighbor]
    )

    # confidence band based on theil slopes
    x_fit = np.linspace(min(x_values), max(x_values), 100)
    y_fit = intercept + slope * x_fit
    ax.plot(x_fit, y_fit, color="k", linestyle="--", zorder=1)
    y_fit_low = intercept + low_slope * x_fit
    y_fit_high = intercept + high_slope * x_fit
    ax.fill_between(
        x_fit, y_fit_low, y_fit_high, color="grey", alpha=0.3, zorder=0
    )

    # Legend
    ax.legend(loc=(0.02, 0.85))

    # make the x axis look decent
    ax.set_xticks(
        [0, 1000000, 2000000, 3000000],
        ["0", "1,000,000", "2,000,000", "3,000,000"],
    )

    # Labels
    ax.set_xlabel("At risk population (including neighboring counties)")
    ax.set_ylabel("% change in incidence per\n1% MMR coverage increase")

    return fig, ax
