import os

from numpyro import enable_x64

enable_x64()

import geopandas as gpd
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import yaml
from matplotlib import rc
from scipy import stats
from shapely.geometry import MultiPolygon, Polygon

from outbreak_size import adjacency
from outbreak_size.adjacency import state_adjacency_matrix
from outbreak_size.coverage_effect import (
    effect_label,
    model_scale_column,
    percent_change_per_pp,
)
from outbreak_size.msr import spearman_msr_blocked
from outbreak_size.state_outbreaks import weighted_mean_population
from outbreak_size.waic import log_likelihood

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)
fig_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")

font = {"family": "Nimbus Roman", "size": 28}
rc("font", **font)

# read analysis params yaml
with open("params.yml", "r") as f:
    params = yaml.safe_load(f)
# the scale of the vaccination covariates of the fits (see params.yml)
COVARIATE_SCALE = params["covariate_scale"]
MMR_EFFECTIVENESS = params.get("mmr_effectiveness")

# the Beta draws below are seeded so the figure is reproducible
rng = np.random.default_rng(params["seed"])


def setup_plot(nrows, ncols):
    fig, ax = plt.subplots(nrows, ncols, figsize=(20, 22.5))
    if nrows * ncols == 1:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="out")
        # offset the spines
        for spine in ax.spines.values():
            spine.set_position(("outward", 5))
        # put the grid behind
        ax.set_axisbelow(True)
        return ax
    for temp in ax.flatten():  # type: ignore
        temp.spines["top"].set_visible(False)
        temp.spines["right"].set_visible(False)
        temp.get_xaxis().tick_bottom()
        temp.get_yaxis().tick_left()
        temp.tick_params(axis="x", direction="out")
        temp.tick_params(axis="y", direction="out")
        # offset the spines
        for spine in temp.spines.values():
            spine.set_position(("outward", 5))
        # put the grid behind
        temp.set_axisbelow(True)
    return fig, ax


def plot_car_effect_correlation(samples, fig, ax):
    # plot the effect
    ax.hist(
        samples,
        bins=100,
        density=True,
        color="green",
        alpha=0.7,
        label="Posterior",
    )

    ax.axvline(
        x=jnp.percentile(samples, 2.5),
        color="red",
    )
    print(jnp.percentile(samples, 2.5))
    print(jnp.median(samples))
    print(jnp.percentile(samples, 97.5))
    ax.axvline(
        x=jnp.percentile(samples, 97.5),
        color="red",
    )
    ax.axvline(
        x=jnp.median(samples),
        color="black",
    )
    ax.axvline(x=0, color="blue", linestyle="--")
    # plot the prior
    # beta(2, 2) scaled to -1 to 1
    x = np.linspace(-1, 1, 1000)
    prior = stats.beta.pdf((x + 1) / 2, 2, 2) / 2
    ax.plot(x, prior, color="grey", label="Prior")
    ax.fill_between(x, 0, prior, color="grey", alpha=0.3)
    ax.set_xlabel("Spatial correlation")
    ax.set_ylabel("Probability density")
    ax.legend(edgecolor="black")
    ax.set_xlim([-1, 1])


def map_fips_code_data(
    fips, vals, fig, ax, cbar_label, cbar_type, point_data=None
):
    """
    Create a static US county map using GeoPandas
    """
    # Load the county boundaries
    if not os.path.exists(os.path.join(data_dir, "cb_2020_us_county_20m.shp")):
        print("skipped panel: cb_2020_us_county_20m.shp not provided.")
        ax.set_axis_off()
        return fig, ax
    counties = gpd.read_file(
        os.path.join(data_dir, "cb_2020_us_county_20m.shp")
    )
    states = gpd.read_file(os.path.join(data_dir, "cb_2020_us_state_20m.shp"))

    exclude_fips = ["60", "66", "69", "72", "78"]  # AS, GU, MP, PR, VI
    counties = counties[~counties["STATEFP"].isin(exclude_fips)]
    states = states[~states["STATEFP"].isin(exclude_fips)]

    # Move Alaska and Hawaii to bottom left
    def reposition_alaska_hawaii(gdf):
        gdf = gdf.copy()

        # Move Alaska - handle longitude wraparound
        alaska_mask = gdf["STATEFP"] == "02"
        if alaska_mask.any():
            # Fix Alaska's longitude wraparound issue
            alaska_geoms = gdf.loc[alaska_mask, "geometry"].copy()

            # Function to fix coordinates that cross the dateline
            def fix_alaska_coords(geom):
                if geom.geom_type == "Polygon":
                    coords = list(geom.exterior.coords)
                    fixed_coords = []
                    for x, y in coords:
                        # Convert positive longitudes (0 to 180) to negative (-180 to 0)
                        if x > 0:
                            x = x - 360
                        fixed_coords.append((x, y))
                    return Polygon(fixed_coords)
                elif geom.geom_type == "MultiPolygon":
                    fixed_polygons = []
                    for poly in geom.geoms:
                        coords = list(poly.exterior.coords)
                        fixed_coords = []
                        for x, y in coords:
                            if x > 0:
                                x = x - 360
                            fixed_coords.append((x, y))
                        fixed_polygons.append(Polygon(fixed_coords))
                    return MultiPolygon(fixed_polygons)
                return geom

            # Apply coordinate fix
            alaska_geoms = alaska_geoms.apply(fix_alaska_coords)

            # Now get bounds after fixing coordinates
            alaska_bounds = alaska_geoms.total_bounds

            # Apply transformation
            alaska_transformed = alaska_geoms.translate(
                xoff=-alaska_bounds[0] - 600, yoff=-alaska_bounds[1] + 80
            ).scale(xfact=0.2, yfact=0.27, origin=(0, 0))

            gdf.loc[alaska_mask, "geometry"] = alaska_transformed

        # Move Hawaii (should be fine as-is)
        hawaii_mask = gdf["STATEFP"] == "15"
        if hawaii_mask.any():
            hawaii_bounds = gdf[hawaii_mask].total_bounds
            gdf.loc[hawaii_mask, "geometry"] = gdf.loc[
                hawaii_mask, "geometry"
            ].translate(
                xoff=-hawaii_bounds[0] - 110, yoff=-hawaii_bounds[1] + 25
            )

        return gdf

    # Reposition Alaska and Hawaii
    counties = reposition_alaska_hawaii(counties)
    states = reposition_alaska_hawaii(states)

    # Merge with the geographic data
    if fips is not None:
        df = pd.DataFrame({"GEOID": fips, "value": vals})
    else:
        df = pd.DataFrame(
            {
                "GEOID": pd.Series(dtype="str"),
                "value": pd.Series(dtype="float"),
            }
        )

    county_data = df[df["GEOID"].str.len() == 5]
    state_data = df[df["GEOID"].str.len() == 2]

    counties_with_data = counties.merge(county_data, how="left")

    # For state-level data, we need to identify counties that belong to states
    # where we have state-level data but no county-level data
    if not state_data.empty:
        # Create a mapping of state FIPS to values
        state_values = dict(zip(state_data["GEOID"], state_data["value"]))

        # For counties without data, check if their state has state-level data
        def fill_state_data(row):
            if pd.isna(row["value"]) and row["STATEFP"] in state_values:
                return state_values[row["STATEFP"]]
            return row["value"]

        counties_with_data["value"] = counties_with_data.apply(
            fill_state_data, axis=1
        )

    has_data_mask = counties_with_data["value"].notna()
    no_data_mask = counties_with_data["value"].isna()

    # geopandas cannot plot an empty frame (every county may have a value)
    if bool(no_data_mask.any()):
        counties_with_data[no_data_mask].plot(
            ax=ax, color="lightgrey", linewidth=0.3, edgecolor="grey"
        )

    # Create normalization for colorbar
    if cbar_type == "risk":
        norm = mcolors.Normalize(vmin=0, vmax=2300)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "custom", ["#67000d", "#fee0d2"], N=256
        )
    elif cbar_type == "exemptions":
        breakpoints = [-1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 100.0]
        norm = mcolors.BoundaryNorm(
            boundaries=breakpoints,
            ncolors=len(breakpoints) - 1,
        )
        # make the colormap
        cmap = mcolors.ListedColormap(
            [
                "#fee5d9",  # 0-2.5
                "#fcbba1",  # 2.5-5
                "#fc9272",  # 5-10
                "#fb6a4a",  # 10-15
                "#de2d26",  # 15-20
                "#a50f15",  # 20+
            ]
        )
    elif cbar_type == "probability":
        norm = mcolors.Normalize(vmin=0, vmax=1)
        breakpoints = [-1, 0.2, 0.4, 0.6, 0.8, 1.01]
        cmap = plt.get_cmap("Blues_r")
    elif cbar_type is None:
        cmap = None
        norm = None
    else:
        assert False, f"cbar_type {cbar_type} not recognized"

    # Plot counties with data
    if fips is not None:
        counties_with_data[has_data_mask].plot(
            column="value",
            cmap=cmap,
            norm=norm,
            linewidth=0.3,
            ax=ax,
            edgecolor="grey",
            legend=False,
            missing_kwds={"color": "lightgrey"},
        )

    # Create custom colorbar
    if cbar_type is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)  # type: ignore

        cbar = fig.colorbar(
            sm,
            ax=ax,
            shrink=0.7,  # Size of colorbar relative to axes
            aspect=20,  # Ratio of long to short dimensions
            pad=0.0,  # Distance between axes and colorbar
            extend="neither",  # Add arrow at top ('neither', 'both', 'min', 'max')
            orientation="horizontal",  # 'vertical' or 'horizontal'
        )

        # Customize the colorbar
        cbar.set_label(cbar_label, rotation=0, labelpad=0, fontsize=28)

        # Set ticks
        if cbar_type == "exemptions":
            # midpoint of each bucket (the bar spaces buckets evenly and
            # interpolates within them), so every label is centered on its
            # color block
            cbar.set_ticks([0.75, 3.75, 7.5, 12.5, 17.5, 60])
            cbar.set_ticklabels(
                [
                    "< 2.5%",
                    "2.5%-5%",
                    "5%-10%",
                    "10%-15%",
                    "15%-20%",
                    "≥20%",
                ]
            )
            cbar.ax.tick_params(labelsize=24, rotation=-60, length=0)

        if cbar_type == "probability":
            cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            cbar.set_ticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])

    # Add state borders
    states.boundary.plot(ax=ax, linewidth=1.0, edgecolor="black", alpha=0.8)

    if point_data is not None:
        # Ensure we have centroids
        counties_with_data["centroid"] = counties_with_data.geometry.centroid

    # Create a DataFrame with point information
    point_df = pd.DataFrame(
        point_data, columns=pd.Index(["GEOID", "point_value"])
    )
    counties_with_points = counties_with_data.merge(
        point_df, on="GEOID", how="inner"
    )

    if not counties_with_points.empty:
        # Scale the point sizes
        point_values = counties_with_points["point_value"]

        # Alternative: Square root scaling for better visual proportion
        sizes = np.array(list(point_values))

        # Plot the points
        ax.scatter(
            counties_with_points["centroid"].x,
            counties_with_points["centroid"].y,
            s=sizes,
            c="red",
            alpha=0.4,
            linewidth=0.8,
        )

    if cbar_type is None:
        # add a legend with point sizes
        size_values = [10, 50, 100, 500]  # Example sizes to show
        size_labels = ["10", "50", "100", "500"]

        legend_elements = []
        for size, label in zip(size_values, size_labels):
            legend_elements.append(
                plt.scatter([], [], s=size, c="red", alpha=0.8, label=label)
            )

        # Add the legend
        ax.legend(
            handles=legend_elements,
            title="Measles cases (N)",
            loc="upper right",
            frameon=False,
            scatterpoints=1,
            ncol=len(size_values),
            bbox_to_anchor=(1.15, 0.035),
        )

    # Remove axis
    ax.set_axis_off()

    return fig, ax


import numpy as np

from outbreak_size.samples_io import load_samples


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

    # Z-score the coverage values on the scale of the fit (like you do before
    # model fitting)
    vaccine_coverage = model_scale_column(
        df, "estimated_mmr_cov", COVARIATE_SCALE, MMR_EFFECTIVENESS
    )
    assert len(vaccine_coverage) == df.height
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

    # % change per 1pp of coverage of each state's coefficient, averaged over
    # the same county-years (all of df) for every state, so that the states
    # differ by their coefficient and not by the coverage of their counties
    effects_pct = percent_change_per_pp(
        vaccine_coef_by_group,
        df,
        "estimated_mmr_cov",
        df,
        COVARIATE_SCALE,
        MMR_EFFECTIVENESS,
    )

    # Map state names to state indices
    unique_states = df["state"].unique(maintain_order=True)
    state_to_idx = {state: idx for idx, state in enumerate(unique_states)}

    # Calculate vaccine marginal effect for each state (with uncertainty)
    state_vaccine_effects_median = {}
    state_vaccine_effects_lower = {}
    state_vaccine_effects_upper = {}

    for state in unique_states:
        state_idx = state_to_idx[state]
        marginal_effect = effects_pct[:, state_idx]

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
    county_year_cases = {}  # (fips, year) -> observed cases
    for row in df.iter_rows(named=True):
        if row["cases_outbreak"] > 0:
            county_year_with_cases[(row["fips"], row["year"])] = row["pop"]
            county_year_cases[(row["fips"], row["year"])] = row[
                "cases_outbreak"
            ]

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
    state_x_missing = {}
    # the same with the outbreaks weighted by their observed cases
    state_avg_population_observed = {}
    state_x_missing_observed = {}

    for state in unique_states:
        # Get all counties in this state
        state_fips = [
            fips for fips in all_fips if fips_to_state[fips] == state
        ]

        weighted_populations = []
        observed_populations = []
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
                total_observed_cases = 0

                # Add focal county
                if (fips, year) in county_year_with_cases:
                    total_population += county_year_with_cases[(fips, year)]
                    total_observed_cases += county_year_cases[(fips, year)]
                    if (fips, year) in obs_to_mu:
                        total_outbreak_size += obs_to_mu[(fips, year)]

                # Add neighbors
                for neighbor_idx in neighbor_indices:
                    neighbor_fips = all_fips[neighbor_idx]
                    if (neighbor_fips, year) in county_year_with_cases:
                        total_population += county_year_with_cases[
                            (neighbor_fips, year)
                        ]
                        total_observed_cases += county_year_cases[
                            (neighbor_fips, year)
                        ]
                        if (neighbor_fips, year) in obs_to_mu:
                            neighbor_outbreak_size = obs_to_mu[
                                (neighbor_fips, year)
                            ]
                            total_outbreak_size += neighbor_outbreak_size
                            # Check if neighbor had an outbreak (≥3 observed
                            # cases, not ≥3 predicted)
                            if county_year_cases[(neighbor_fips, year)] >= 3:
                                has_neighbor_outbreak = True

                # Record this outbreak
                if total_population > 0 and total_outbreak_size > 0:
                    weighted_populations.append(
                        (total_population, total_outbreak_size)
                    )
                if total_population > 0 and total_observed_cases > 0:
                    observed_populations.append(
                        (total_population, total_observed_cases)
                    )

        # Calculate weighted average. A state with no outbreak to average
        # over has no x: it is set to 0 here (the sensitivity analysis below
        # also leaves such states out)
        mean_population = weighted_mean_population(weighted_populations)
        state_x_missing[state] = mean_population is None
        state_avg_population[state] = (
            0 if mean_population is None else mean_population
        )
        mean_observed = weighted_mean_population(observed_populations)
        state_x_missing_observed[state] = mean_observed is None
        state_avg_population_observed[state] = (
            0 if mean_observed is None else mean_observed
        )
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
    x_missing = np.array([state_x_missing[state] for state in unique_states])
    x_values_observed = np.array(
        [state_avg_population_observed[state] for state in unique_states]
    )
    x_missing_observed = np.array(
        [state_x_missing_observed[state] for state in unique_states]
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

    # Create a state-level adjacency matrix for MSR
    # Build adjacency based on whether states share a border (via their counties)
    n_states = len(unique_states)
    state_adjacency = np.zeros((n_states, n_states))

    for i, state1 in enumerate(unique_states):
        for j, state2 in enumerate(unique_states):
            if i == j:
                continue
            # Check if any county from state1 is adjacent to any county from state2
            state1_fips = [
                fips for fips in all_fips if fips_to_state[fips] == state1
            ]
            state2_fips = [
                fips for fips in all_fips if fips_to_state[fips] == state2
            ]

            for fips1 in state1_fips:
                fips1_idx = np.where(all_fips == fips1)[0]
                if len(fips1_idx) == 0:
                    continue
                fips1_idx = fips1_idx[0]

                for fips2 in state2_fips:
                    fips2_idx = np.where(all_fips == fips2)[0]
                    if len(fips2_idx) == 0:
                        continue
                    fips2_idx = fips2_idx[0]

                    if adjacency[fips1_idx, fips2_idx] > 0:
                        state_adjacency[i, j] = 1
                        break
                if state_adjacency[i, j] == 1:
                    break

    # Create state indices (0-indexed)
    state_indices = np.arange(len(unique_states))

    # Print Pearson correlations (keeping original)
    print("\n=== Pearson correlations ===")
    print(
        "Has neighbor:",
        stats.pearsonr(x_values[has_neighbor], y_values[has_neighbor]),
    )
    print(
        "No neighbor:",
        stats.pearsonr(x_values[~has_neighbor], y_values[~has_neighbor]),
    )
    print("All:", stats.pearsonr(x_values, y_values))

    # Spearman with MSR
    print("\n=== Spearman correlations with MSR ===")

    # Has neighbor subset
    has_neighbor_states = np.where(has_neighbor)[0]
    n_has_neighbor = len(has_neighbor_states)

    rho_has, p_has, _ = spearman_msr_blocked(
        x_values[has_neighbor],
        y_values[has_neighbor],
        state_adjacency[np.ix_(has_neighbor, has_neighbor)],
        np.arange(
            n_has_neighbor
        ),  # Re-indexed to 0, 1, 2, ..., n_has_neighbor-1
        n_permutations=99999,
    )
    print(f"Has neighbor: rho={rho_has:.4f}, p={p_has:.4f}")

    # No neighbor subset
    no_neighbor_mask = ~has_neighbor
    no_neighbor_states = np.where(no_neighbor_mask)[0]
    n_no_neighbor = len(no_neighbor_states)

    if n_no_neighbor > 1:
        rho_no, p_no, _ = spearman_msr_blocked(
            x_values[no_neighbor_mask],
            y_values[no_neighbor_mask],
            state_adjacency[np.ix_(no_neighbor_mask, no_neighbor_mask)],
            np.arange(
                n_no_neighbor
            ),  # Re-indexed to 0, 1, 2, ..., n_no_neighbor-1
            n_permutations=99999,
        )
        print(f"No neighbor: rho={rho_no:.4f}, p={p_no:.4f}")

    # All states
    rho_all, p_all, _ = spearman_msr_blocked(
        x_values,
        y_values,
        state_adjacency,
        state_indices,  # This one is already 0, 1, 2, ..., n_states-1
        n_permutations=9999,
    )
    print(f"All: rho={rho_all:.4f}, p={p_all:.4f}")

    # Sensitivity of the tests above to the state graph and to the states
    # whose x is undefined. The graph above joins two states only where
    # counties that had outbreaks border each other; the real graph joins
    # states whose counties border, whatever happened in them.
    print("\n=== Sensitivity: state graph and states without x ===")
    print(
        "States with no outbreak to average over (x undefined): "
        f"{[st for st, gone in zip(unique_states, x_missing) if gone]} "
        "(predicted size), "
        f"{[st for st, gone in zip(unique_states, x_missing_observed) if gone]}"
        " (observed cases)"
    )
    county_adjacency_data = pl.read_csv(
        os.path.join(data_dir, "county_adjacencies.txt"),
        separator="|",
        schema_overrides={
            "County GEOID": pl.String,
            "Neighbor GEOID": pl.String,
        },
    )
    real_state_adjacency = state_adjacency_matrix(
        county_adjacency_data,
        [
            df.filter(pl.col("state") == state)["fips"][0][:2]
            for state in unique_states
        ],
    )
    everyone = np.ones(n_states, dtype=bool)
    for x_name, x_used, x_gone in [
        ("x weighted by predicted outbreak size", x_values, x_missing),
        (
            "x weighted by observed cases",
            x_values_observed,
            x_missing_observed,
        ),
    ]:
        for graph_name, graph in [
            ("figure's graph", state_adjacency),
            ("real state borders", real_state_adjacency),
        ]:
            for subset_name, subset in [
                ("all states", everyone),
                ("states with x", ~x_gone),
            ]:
                for group_name, group in [
                    ("has neighbor", has_neighbor),
                    ("no neighbor", ~has_neighbor),
                    ("all", everyone),
                ]:
                    keep = subset & group
                    rho_s, p_s, _ = spearman_msr_blocked(
                        x_used[keep],
                        y_values[keep],
                        graph[np.ix_(keep, keep)],
                        np.arange(keep.sum()),
                        n_permutations=9999,
                    )
                    n_borders = int(graph[np.ix_(keep, keep)].sum() / 2)
                    print(
                        f"  {x_name}, {graph_name}, {subset_name}, "
                        f"{group_name}: n={keep.sum()}, {n_borders} borders, "
                        f"rho={rho_s:.3f}, p={p_s:.4f}"
                    )

    # what about theil slopes
    print("\n=== Theil slopes ===")
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
    ax.legend(loc=(0.02, 0.85), edgecolor="black")

    # make the x axis look decent
    ax.set_xticks(
        [0, 2000000, 4000000],
        ["0", "2,000,000", "4,000,000"],
    )

    # Labels
    ax.set_xlabel("At risk population (including neighboring counties)")
    ax.set_ylabel(effect_label())

    return fig, ax


def plot_car_variance_across_years(
    samples,
    df,
    adjacency_matrix,
    two_year_bucket_map,
    fig,
    ax,
    show_longitudinal_variance=False,
):
    """
    Histogram of the posterior variance across year buckets of the mean CAR
    effect, where the mean is taken over counties observed in each bucket.
    Only CAR counties (those with at least one neighbor) are used.
    CAR effects are scaled by spatial_sigma before calculating variance.

    NOTE: In the new model structure:
    - raw_spatial_effect shape: (n_samples, n_year_buckets, n_car_counties)
    - spatial_sigma shape: (n_samples, n_year_buckets)

    Both the CAR effects AND the spatial_sigma now vary by year bucket.

    Parameters
    ----------
    samples : dict
        Posterior samples from the year-specific CAR model
    df : polars.DataFrame
        Analysis dataset
    adjacency_matrix : numpy.ndarray
        County adjacency matrix
    two_year_bucket_map : dict
        Mapping from years to bucket indices (e.g., {2021: 0, 2022: 0, 2023: 1, ...})
    fig, ax : matplotlib objects
        Figure and axis for plotting
    show_longitudinal_variance : bool
        If True, calculate and display variance for counties observed in multiple buckets

    Returns
    -------
    fig, ax, waic_dict
        waic_dict contains WAIC, lppd, p_waic, and n_obs
    """
    # Extract raw_spatial_effect from samples
    # NEW MODEL: This is now stored as separate samples per year
    # We need to reconstruct it into a (n_samples, n_year_buckets, n_car_counties) array

    # First, determine how many year buckets we have
    n_buckets = len(set(two_year_bucket_map.values()))

    # Get spatial_sigma - should be (n_samples, n_year_buckets)
    spatial_sigma = np.array(samples["spatial_sigma"])
    n_samples = spatial_sigma.shape[0]

    # Reconstruct raw_spatial_effect from year-specific samples
    # In your new model, these are stored as year_0/raw_spatial_effect, year_1/raw_spatial_effect, etc.
    raw_spatial_effects_list = []
    for year_idx in range(n_buckets):
        year_key = f"year_{year_idx}/raw_spatial_effect"
        if year_key in samples:
            raw_spatial_effects_list.append(np.array(samples[year_key]))
        else:
            raise KeyError(
                f"Expected key '{year_key}' not found in samples. Available keys: {list(samples.keys())}"
            )

    # Stack into (n_year_buckets, n_samples, n_car_counties), then transpose to (n_samples, n_year_buckets, n_car_counties)
    raw_spatial_effect = np.stack(raw_spatial_effects_list, axis=0).transpose(
        1, 0, 2
    )
    n_car_counties = raw_spatial_effect.shape[2]

    print(
        f"\nReconstructed raw_spatial_effect shape: {raw_spatial_effect.shape}"
    )
    print(f"spatial_sigma shape: {spatial_sigma.shape}")

    # Scale the raw spatial effect by spatial_sigma for each year bucket
    # spatial_sigma shape: (n_samples, n_year_buckets)
    # raw_spatial_effect shape: (n_samples, n_year_buckets, n_car_counties)
    # Need to broadcast spatial_sigma to match: add dimension for counties
    spatial_effect = raw_spatial_effect * spatial_sigma[:, :, None]

    # Identify which counties have neighbors (CAR counties)
    all_fips = df["fips"].unique(maintain_order=True).to_numpy()
    has_a_neighbor = adjacency_matrix.sum(axis=1) > 0
    car_fips = all_fips[has_a_neighbor]

    # Verify dimensions
    assert len(car_fips) == n_car_counties, (
        f"Number of CAR counties ({len(car_fips)}) doesn't match "
        f"spatial effect dimension ({n_car_counties})"
    )

    # For each bucket, find which CAR counties have at least one observation
    bucket_to_years = {}
    for year, bucket in two_year_bucket_map.items():
        bucket_to_years.setdefault(bucket, []).append(year)

    # Build a set of observed fips per bucket
    bucket_observed_fips = {}
    for bucket, years in bucket_to_years.items():
        observed = set(
            df.filter(pl.col("year").is_in(years))["fips"].unique().to_list()
        )
        bucket_observed_fips[bucket] = observed

    # Find conserved counties (observed in ALL buckets)
    conserved_fips = set(car_fips)
    for bucket in range(n_buckets):
        conserved_fips = conserved_fips.intersection(
            bucket_observed_fips[bucket]
        )

    conserved_mask = np.array([fips in conserved_fips for fips in car_fips])

    print(f"\n{'=' * 70}")
    print("CAR TEMPORAL VARIANCE ANALYSIS")
    print(f"{'=' * 70}")
    print(f"Total CAR counties: {len(car_fips)}")
    print(
        f"Conserved counties (in all {n_buckets} buckets): {len(conserved_fips)}"
    )
    for bucket, years in bucket_to_years.items():
        print(
            f"  Bucket {bucket} ({years}): {len(bucket_observed_fips[bucket])} counties"
        )

    # For each posterior sample, compute the mean CAR effect per bucket
    bucket_means = np.zeros((n_samples, n_buckets))
    for bucket in range(n_buckets):
        observed_fips = bucket_observed_fips[bucket]
        observed_mask = np.array([fips in observed_fips for fips in car_fips])

        if observed_mask.sum() == 0:
            bucket_means[:, bucket] = 0.0
        else:
            bucket_means[:, bucket] = spatial_effect[
                :, bucket, observed_mask
            ].mean(axis=1)

    # Variance across year buckets for each posterior sample
    variance_across_years = bucket_means.var(axis=1, ddof=1) ** 0.5

    # ===== Calculate longitudinal variance for counties in multiple buckets =====
    longitudinal_variances = None
    if show_longitudinal_variance:
        print(f"\n{'=' * 70}")
        print("LONGITUDINAL VARIANCE (Counties observed in >1 bucket)")
        print(f"{'=' * 70}")

        # For each CAR county, determine which buckets it appears in
        county_bucket_presence = np.zeros(
            (len(car_fips), n_buckets), dtype=bool
        )

        for bucket in range(n_buckets):
            observed_fips_in_bucket = bucket_observed_fips[bucket]
            for i, fips in enumerate(car_fips):
                if fips in observed_fips_in_bucket:
                    county_bucket_presence[i, bucket] = True

        # Count how many buckets each county appears in
        buckets_per_county = county_bucket_presence.sum(axis=1)

        # Identify counties observed in >1 bucket
        longitudinal_mask = buckets_per_county > 1
        n_longitudinal = longitudinal_mask.sum()

        print(
            f"Counties observed in >1 bucket: {n_longitudinal} / {len(car_fips)}"
        )
        for n_b in range(2, n_buckets + 1):
            n_counties_in_n_buckets = (buckets_per_county == n_b).sum()
            if n_counties_in_n_buckets > 0:
                print(
                    f"  In exactly {n_b} buckets: {n_counties_in_n_buckets} counties"
                )

        if n_longitudinal > 0:
            # For each posterior sample, calculate within-county temporal variance
            longitudinal_variances = np.zeros(n_samples)

            for sample_idx in range(n_samples):
                county_temporal_variances = []

                for county_idx in np.where(longitudinal_mask)[0]:
                    # Get which buckets this county appears in
                    buckets_present = np.where(
                        county_bucket_presence[county_idx]
                    )[0]

                    # Get CAR effects for this county in those buckets
                    county_effects = spatial_effect[
                        sample_idx, buckets_present, county_idx
                    ]

                    # Calculate variance across time for this county
                    if len(county_effects) >= 2:
                        county_var = np.var(county_effects, ddof=1)
                        county_temporal_variances.append(county_var)

                # Average variance across all longitudinal counties
                if len(county_temporal_variances) > 0:
                    longitudinal_variances[sample_idx] = (
                        np.mean(county_temporal_variances) ** 0.5
                    )
                else:
                    longitudinal_variances[sample_idx] = 0.0

            print("\nLongitudinal SD (within-county temporal variation):")
            print(f"  2.5%:  {np.percentile(longitudinal_variances, 2.5):.4f}")
            print(f"  50%:   {np.median(longitudinal_variances):.4f}")
            print(
                f"  97.5%: {np.percentile(longitudinal_variances, 97.5):.4f}"
            )

            print("\nComparison to aggregate temporal SD:")
            print(
                f"  Aggregate temporal SD: {np.median(variance_across_years):.4f}"
            )
            print(
                f"  Longitudinal SD: {np.median(longitudinal_variances):.4f}"
            )
            ratio = np.median(longitudinal_variances) / np.median(
                variance_across_years
            )
            print(f"  Ratio (Longitudinal / Aggregate): {ratio:.3f}")
            print("\nInterpretation:")
            print(
                "  Aggregate = variance of bucket-level means across all counties"
            )
            print("  Longitudinal = average within-county variance over time")
            print(
                "  Ratio < 1: Individual counties vary less than population average"
            )
            print(
                "  Ratio > 1: Individual counties vary more than population average"
            )
        else:
            print("No counties observed in >1 bucket")

    print(f"\n{'=' * 70}")
    print("TEMPORAL STANDARD DEVIATION (Primary Result)")
    print(f"{'=' * 70}")
    print(f"2.5%:  {np.percentile(variance_across_years, 2.5):.4f}")
    print(f"50%:   {np.median(variance_across_years):.4f}")
    print(f"97.5%: {np.percentile(variance_across_years, 97.5):.4f}")

    # ===== COMPARISON 1: Within-bucket spatial variance =====
    print(f"\n{'=' * 70}")
    print("COMPARISON 1: Temporal SD vs Spatial SD")
    print(f"{'=' * 70}")

    within_bucket_sd = np.zeros((n_samples, n_buckets))
    for bucket in range(n_buckets):
        observed_fips = bucket_observed_fips[bucket]
        observed_mask = np.array([fips in observed_fips for fips in car_fips])
        if observed_mask.sum() > 1:
            within_bucket_sd[:, bucket] = spatial_effect[
                :, bucket, observed_mask
            ].std(axis=1)
        else:
            within_bucket_sd[:, bucket] = 0.0

    avg_within_bucket_sd = within_bucket_sd.mean(axis=1)
    ratio_temporal_to_spatial = variance_across_years / avg_within_bucket_sd

    print("Average within-bucket spatial SD:")
    print(f"  2.5%:  {np.percentile(avg_within_bucket_sd, 2.5):.4f}")
    print(f"  50%:   {np.median(avg_within_bucket_sd):.4f}")
    print(f"  97.5%: {np.percentile(avg_within_bucket_sd, 97.5):.4f}")
    print("\nRatio (Temporal SD / Spatial SD):")
    print(f"  2.5%:  {np.percentile(ratio_temporal_to_spatial, 2.5):.4f}")
    print(f"  50%:   {np.median(ratio_temporal_to_spatial):.4f}")
    print(f"  97.5%: {np.percentile(ratio_temporal_to_spatial, 97.5):.4f}")
    print("\nInterpretation: Ratio < 1 means space matters more than time")
    print("                Ratio > 1 means time matters more than space")

    # ===== COMPARISON 2: Variance decomposition (conserved counties only) =====
    print(f"\n{'=' * 70}")
    print("COMPARISON 2: Variance Decomposition (Conserved Counties)")
    print(f"{'=' * 70}")

    if conserved_mask.sum() > 0:
        between_var = np.zeros(n_samples)
        within_var = np.zeros(n_samples)

        for i in range(n_samples):
            conserved_effects = spatial_effect[i, :, conserved_mask]
            bucket_means_conserved = conserved_effects.mean(axis=1)
            between_var[i] = np.var(bucket_means_conserved)
            within_bucket_vars = np.var(conserved_effects, axis=1)
            within_var[i] = np.mean(within_bucket_vars)

        print("Temporal variance (between buckets):")
        print(f"  2.5%:  {np.percentile(between_var, 2.5):.4f}")
        print(f"  50%:   {np.median(between_var):.4f}")
        print(f"  97.5%: {np.percentile(between_var, 97.5):.4f}")

        print("\nSpatial variance (within buckets, averaged):")
        print(f"  2.5%:  {np.percentile(within_var, 2.5):.4f}")
        print(f"  50%:   {np.median(within_var):.4f}")
        print(f"  97.5%: {np.percentile(within_var, 97.5):.4f}")

        ratio = between_var / within_var
        print("\nRatio (temporal / spatial):")
        print(f"  2.5%:  {np.percentile(ratio, 2.5):.4f}")
        print(f"  50%:   {np.median(ratio):.4f}")
        print(f"  97.5%: {np.percentile(ratio, 97.5):.4f}")
        print("\nInterpretation: Ratio < 1 means spatial variation dominates")
        print("                Ratio > 1 means temporal variation dominates")
    else:
        print("No conserved counties - cannot compute decomposition")

    # ===== COMPARISON 4: Compare to spatial_sigma =====
    print(f"\n{'=' * 70}")
    print("COMPARISON 4: Temporal SD vs spatial_sigma")
    print(f"{'=' * 70}")

    # spatial_sigma now has shape (n_samples, n_year_buckets)
    # ===== COMPARISON 4: Compare to spatial_sigma =====
    print(f"\n{'=' * 70}")
    print("COMPARISON 4: Temporal SD vs spatial_sigma")
    print(f"{'=' * 70}")

    # spatial_sigma now has shape (n_samples, n_year_buckets)
    # Calculate average across year buckets for each sample
    avg_spatial_sigma = spatial_sigma.mean(axis=1)

    # Also show the variance of spatial_sigma across buckets
    spatial_sigma_temporal_sd = spatial_sigma.std(axis=1)

    ratio_to_sigma = variance_across_years / avg_spatial_sigma

    print("Average spatial_sigma across buckets:")
    print(f"  2.5%:  {np.percentile(avg_spatial_sigma, 2.5):.4f}")
    print(f"  50%:   {np.median(avg_spatial_sigma):.4f}")
    print(f"  97.5%: {np.percentile(avg_spatial_sigma, 97.5):.4f}")

    print("\nTemporal SD of spatial_sigma (across buckets):")
    print(f"  2.5%:  {np.percentile(spatial_sigma_temporal_sd, 2.5):.4f}")
    print(f"  50%:   {np.median(spatial_sigma_temporal_sd):.4f}")
    print(f"  97.5%: {np.percentile(spatial_sigma_temporal_sd, 97.5):.4f}")

    print("\nRatio (Temporal SD / avg spatial_sigma):")
    print(f"  2.5%:  {np.percentile(ratio_to_sigma, 2.5):.4f}")
    print(f"  50%:   {np.median(ratio_to_sigma):.4f}")
    print(f"  97.5%: {np.percentile(ratio_to_sigma, 97.5):.4f}")
    print("\nInterpretation: Ratio << 1 means temporal changes are small")
    print("                relative to baseline spatial effect magnitude")

    # ===== COMPARISON 5: Coefficient of variation =====
    print(f"\n{'=' * 70}")
    print("COMPARISON 5: Coefficient of Variation")
    print(f"{'=' * 70}")

    # Mean absolute CAR effect across all buckets/counties
    mean_abs_effect_list = []
    for bucket in range(n_buckets):
        observed_fips = bucket_observed_fips[bucket]
        observed_mask = np.array([fips in observed_fips for fips in car_fips])
        mean_abs_effect_list.append(
            np.abs(spatial_effect[:, bucket, observed_mask]).mean(axis=1)
        )
    mean_abs_effect = np.stack(mean_abs_effect_list).mean(axis=0)

    cv = variance_across_years / mean_abs_effect

    print("Mean absolute CAR effect:")
    print(f"  2.5%:  {np.percentile(mean_abs_effect, 2.5):.4f}")
    print(f"  50%:   {np.median(mean_abs_effect):.4f}")
    print(f"  97.5%: {np.percentile(mean_abs_effect, 97.5):.4f}")
    print("\nCoefficient of variation (Temporal SD / Mean |effect|):")
    print(f"  2.5%:  {np.percentile(cv, 2.5):.2%}")
    print(f"  50%:   {np.median(cv):.2%}")
    print(f"  97.5%: {np.percentile(cv, 97.5):.2%}")

    # ===== WAIC CALCULATION =====
    print(f"\n{'=' * 70}")
    print("WAIC CALCULATION")
    print(f"{'=' * 70}")

    # Check if we have the necessary samples for WAIC
    if "mu" in samples:
        from scipy.special import logsumexp

        # mu has shape (n_samples, n_observations)
        mu = np.array(samples["mu"])

        # Get observed cases
        observed_cases = df["cases_outbreak"].to_numpy()

        # Verify dimensions match
        if mu.shape[1] != len(observed_cases):
            print(
                f"WARNING: mu shape {mu.shape} doesn't match observed cases {len(observed_cases)}"
            )
            waic_dict = None
        else:
            print(f"mu range: [{np.min(mu):.6f}, {np.max(mu):.6f}]")
            print(f"mu median: {np.median(mu):.6f}")
            print(f"Number of mu values < 1e-10: {np.sum(mu < 1e-10)}")
            print(f"Number of mu values == 0: {np.sum(mu == 0)}")

            # Calculate log pointwise predictive density: Poisson, or negative
            # binomial if the model has a concentration
            epsilon = 1e-10
            log_lik = log_likelihood(
                observed_cases,
                np.maximum(mu, epsilon),
                samples.get("nb_concentration"),
            )

            print(
                f"log_lik range: [{np.min(log_lik):.2f}, {np.max(log_lik):.2f}]"
            )
            print(f"Number of -inf in log_lik: {np.sum(np.isinf(log_lik))}")
            print(f"Number of nan in log_lik: {np.sum(np.isnan(log_lik))}")

            # WAIC calculation using logsumexp for numerical stability
            lppd_pointwise = np.zeros(len(observed_cases))
            for i in range(len(observed_cases)):
                lppd_pointwise[i] = logsumexp(log_lik[:, i]) - np.log(
                    mu.shape[0]
                )

            lppd = np.sum(lppd_pointwise)

            # p_waic (effective number of parameters)
            p_waic = np.sum(np.var(log_lik, axis=0))

            print(
                f"lppd_pointwise range: [{np.min(lppd_pointwise):.2f}, {np.max(lppd_pointwise):.2f}]"
            )
            print(
                f"Number of -inf in lppd_pointwise: {np.sum(np.isinf(lppd_pointwise))}"
            )
            print(
                f"Variance range: [{np.min(np.var(log_lik, axis=0)):.2f}, {np.max(np.var(log_lik, axis=0)):.2f}]"
            )

            # WAIC = -2 * (lppd - p_waic)
            waic = -2 * (lppd - p_waic)

            print(f"\nlppd (log pointwise predictive density): {lppd:.2f}")
            print(f"p_WAIC (effective number of parameters): {p_waic:.2f}")
            print(f"WAIC: {waic:.2f}")

            # Check for problematic observations
            if np.any(np.isinf(lppd_pointwise)) or np.any(
                np.isnan(lppd_pointwise)
            ):
                print(
                    "\nWARNING: Some observations have infinite or NaN lppd values"
                )
                bad_indices = np.where(
                    np.isinf(lppd_pointwise) | np.isnan(lppd_pointwise)
                )[0]
                print(
                    f"Number of problematic observations: {len(bad_indices)}"
                )
                if len(bad_indices) < 20:
                    for idx in bad_indices:
                        print(
                            f"  Obs {idx}: y={observed_cases[idx]}, "
                            f"mu range=[{mu[:, idx].min():.6f}, {mu[:, idx].max():.6f}], "
                            f"lppd={lppd_pointwise[idx]:.2f}"
                        )

            # Store WAIC for comparison
            waic_dict = {
                "waic": waic,
                "lppd": lppd,
                "p_waic": p_waic,
                "n_obs": len(observed_cases),
                "lppd_pointwise": lppd_pointwise,
                "log_lik": log_lik,
            }
    else:
        print("WARNING: 'mu' not found in samples. Cannot calculate WAIC.")
        print("Available keys:", list(samples.keys()))
        waic_dict = None

    print(f"\n{'=' * 70}\n")

    # ===== Plotting section =====
    ax.hist(
        variance_across_years,
        bins=100,
        density=True,
        color="steelblue",
        alpha=0.7,
        label="Aggregate temporal SD",
    )

    if show_longitudinal_variance and longitudinal_variances is not None:
        ax.hist(
            longitudinal_variances,
            bins=100,
            density=True,
            color="orange",
            alpha=0.7,
            label="Longitudinal SD",
        )

    ax.set_xlabel("Standard deviation of CAR effect over time")
    ax.set_ylabel("Density")
    if show_longitudinal_variance and longitudinal_variances is not None:
        ax.legend()

    return fig, ax, waic_dict


def map_spatial_effect_p_values(samples, df, adjacency_data, fig, ax):
    """
    Map the p-value of the spatial effect of each county of a period (the
    two-sided share of its posterior on the other side of 0) for the counties
    with a neighbor, which are the ones the spatial model has an effect for.
    `df` has the county-years of the period and `adjacency_data` is the table of
    county adjacencies. Returns the adjacency matrix of the period.
    """
    raw_spatial_effect = samples["raw_spatial_effect"]
    p_values = 2 * np.minimum(
        np.mean(raw_spatial_effect > 0, axis=0),
        np.mean(raw_spatial_effect < 0, axis=0),
    )
    all_fips = df["fips"].unique(maintain_order=True).to_list()
    adjacency_matrix = adjacency.assemble_adjacency_matrix(
        adjacency_data, all_fips
    )
    has_a_neighbor = adjacency_matrix.sum(axis=0) > 0
    counties_with_neighbors = np.array(all_fips)[has_a_neighbor]
    assert len(counties_with_neighbors) == len(p_values), (
        f"{len(counties_with_neighbors)} counties with a neighbor, "
        f"{len(p_values)} spatial effects"
    )
    map_fips_code_data(
        counties_with_neighbors.tolist(),
        p_values,
        fig,
        ax,
        cbar_label="Spatial effect p-value",
        cbar_type="probability",
    )
    return adjacency_matrix


if __name__ == "__main__":
    posterior_samples = load_samples(
        os.path.join(
            output_dir,
            "samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_[2021, 2022, 2023, 2024, 2025].npz",
        )
    )

    fig, ax = setup_plot(3, 2)  # type: ignore

    ax[0, 0].text(  # type: ignore
        -0.25,
        1.08,
        "a",
        transform=ax[0, 0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )

    plot_car_effect_correlation(
        2 * posterior_samples["spatial_correlation"] - 1,
        fig,
        ax[0, 0],  # type: ignore
    )

    posterior_samples_pre = load_samples(
        os.path.join(
            output_dir,
            "samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_[2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020].npz",
        )
    )

    ax[0, 1].text(  # type: ignore
        -0.25,
        1.08,
        "b",
        transform=ax[0, 1].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )

    plot_car_effect_correlation(
        2 * posterior_samples_pre["spatial_correlation"] - 1,
        fig,
        ax[0, 1],  # type: ignore
    )

    # the change in the spatial correlation from 2011-2020 to 2021-2025: every
    # draw of the 2021-2025 fit minus the same-numbered draw of the 2011-2020
    # fit (independent fits, so the draws are unrelated: the correlation of the
    # paired draws is printed as a check)
    correlation_pre = (
        2 * np.asarray(posterior_samples_pre["spatial_correlation"]) - 1
    )
    correlation_post = (
        2 * np.asarray(posterior_samples["spatial_correlation"]) - 1
    )
    for label, draws in [
        ("2011-2020", correlation_pre),
        ("2021-2025", correlation_post),
    ]:
        low, median, high = np.percentile(draws, [2.5, 50, 97.5])
        print(
            f"Spatial correlation {label}: {median:.3f} "
            f"[{low:.3f}, {high:.3f}]"
        )
    change = correlation_post - correlation_pre
    low, median, high = np.percentile(change, [2.5, 50, 97.5])
    print(
        "Change in the spatial correlation, 2021-2025 minus 2011-2020: "
        f"{median:.3f} [{low:.3f}, {high:.3f}], P(change > 0) = "
        f"{np.mean(change > 0):.3f} (correlation of the paired draws "
        f"{np.corrcoef(correlation_post, correlation_pre)[0, 1]:.3f})"
    )

    ax[1, 0].text(  # type: ignore
        -0.25,
        1.08,
        "c",
        transform=ax[1, 0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )

    adjacency_data = pl.read_csv(
        os.path.join(data_dir, "county_adjacencies.txt"),
        separator="|",
        schema_overrides={
            "County GEOID": pl.String,
            "Neighbor GEOID": pl.String,
        },
    )

    df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
    )

    # the adjacency matrix of 2021-2025 is used by the next panels too
    adjacency_matrix = map_spatial_effect_p_values(
        posterior_samples,
        df.filter(pl.col("year") >= 2021),
        adjacency_data,
        fig,
        ax[1, 0],  # type: ignore
    )

    # the same map for 2011-2020
    ax[1, 1].text(  # type: ignore
        -0.25,
        1.08,
        "d",
        transform=ax[1, 1].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    map_spatial_effect_p_values(
        posterior_samples_pre,
        df.filter(pl.col("year") < 2021),
        adjacency_data,
        fig,
        ax[1, 1],  # type: ignore
    )

    ax[2, 0].text(  # type: ignore
        -0.25,
        1.08,
        "e",
        transform=ax[2, 0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )

    df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
    )

    df = df.filter(pl.col("year") >= 2021)

    plot_vaccine_vs_spatial(
        posterior_samples,
        df,
        adjacency_matrix,
        fig,
        ax[2, 0],  # type: ignore
    )

    ax[2, 1].text(  # type: ignore
        -0.25,
        1.08,
        "f",
        transform=ax[2, 1].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )

    # load the year-specific CAR sensitivity analysis samples
    # In your main plotting code:

    # Load year-specific CAR model (with buckets)
    year_specific_samples = load_samples(
        os.path.join(
            output_dir,
            "samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_2021_2025_CAR_by_two_year_bucket.npz",
        )
    )

    # Load standard CAR model (without buckets), fit on the same one-year
    # covariates as the year-specific fit above
    standard_samples = load_samples(
        os.path.join(
            output_dir,
            "samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_[2021, 2022, 2023, 2024, 2025].npz",
        )
    )

    # Prepare data
    df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        schema_overrides={"fips": pl.String},
    )
    two_year_bucket_map = {
        2021: 0,
        2022: 0,
        2023: 1,
        2024: 1,
        2025: 2,
    }
    df_year_specific = df.filter(
        pl.col("year").is_in(list(two_year_bucket_map.keys()))
    )
    df_standard = df.filter(
        pl.col("year").is_in([2021, 2022, 2023, 2024, 2025])
    )

    adjacency_matrix_year_specific = adjacency.assemble_adjacency_matrix(
        pl.read_csv(
            os.path.join(data_dir, "county_adjacencies.txt"),
            separator="|",
            schema_overrides={
                "County GEOID": pl.String,
                "Neighbor GEOID": pl.String,
            },
        ),
        df_year_specific["fips"].unique(maintain_order=True).to_list(),
    )

    adjacency_matrix_standard = adjacency.assemble_adjacency_matrix(
        pl.read_csv(
            os.path.join(data_dir, "county_adjacencies.txt"),
            separator="|",
            schema_overrides={
                "County GEOID": pl.String,
                "Neighbor GEOID": pl.String,
            },
        ),
        df_standard["fips"].unique(maintain_order=True).to_list(),
    )

    # Plot and get WAIC for year-specific model
    fig_bucket, ax_bucket, waic_bucket = plot_car_variance_across_years(
        year_specific_samples,
        df_year_specific,
        adjacency_matrix_year_specific,
        two_year_bucket_map,
        fig,
        ax[2, 1],  # type: ignore
    )

    # For standard model, you'd need a simplified version that doesn't expect buckets
    # Or calculate WAIC separately:
    print(f"\n{'=' * 70}")
    print("MODEL COMPARISON")
    print(f"{'=' * 70}")

    # Calculate WAIC for standard model
    if "mu" in standard_samples:
        from scipy.special import logsumexp

        mu_standard = np.array(standard_samples["mu"])
        observed_cases_standard = df_standard["cases_outbreak"].to_numpy()

        print("\nStandard model diagnostics:")
        print(
            f"mu range: [{np.min(mu_standard):.6f}, {np.max(mu_standard):.6f}]"
        )

        epsilon = 1e-10
        log_lik_standard = log_likelihood(
            observed_cases_standard,
            np.maximum(mu_standard, epsilon),
            standard_samples.get("nb_concentration"),
        )

        # Use logsumexp for stability
        lppd_pointwise_standard = np.zeros(len(observed_cases_standard))
        for i in range(len(observed_cases_standard)):
            lppd_pointwise_standard[i] = logsumexp(
                log_lik_standard[:, i]
            ) - np.log(mu_standard.shape[0])

        lppd_standard = np.sum(lppd_pointwise_standard)
        p_waic_standard = np.sum(np.var(log_lik_standard, axis=0))
        waic_standard = -2 * (lppd_standard - p_waic_standard)

        print("\nStandard CAR model (no time buckets):")
        print(f"  WAIC: {waic_standard:.2f}")
        print(f"  p_WAIC: {p_waic_standard:.2f}")
        print(f"  lppd: {lppd_standard:.2f}")

        if waic_bucket is not None:
            print("\nYear-specific CAR model (two-year buckets):")
            print(f"  WAIC: {waic_bucket['waic']:.2f}")
            print(f"  p_WAIC: {waic_bucket['p_waic']:.2f}")
            print(f"  lppd: {waic_bucket['lppd']:.2f}")

            if not np.isinf(waic_bucket["waic"]) and not np.isinf(
                waic_standard
            ):
                diff = waic_standard - waic_bucket["waic"]
                print(f"\nDifference (Standard - Buckets): {diff:.2f}")
                print("  Negative = standard model is better (lower WAIC)")
                print("  Positive = buckets model is better (lower WAIC)")
                print("  |diff| > 10 suggests strong evidence")
                print("  |diff| > 4 suggests moderate evidence")
            else:
                print("\nWARNING: Cannot compare due to infinite WAIC values")

    fig.tight_layout()

    # save
    fig.savefig(
        os.path.join(fig_dir, "figure_supplemental_spatial_effect.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(fig_dir, "figure_supplemental_spatial_effect.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
