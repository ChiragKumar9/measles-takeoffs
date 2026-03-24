import os
import pickle

import geopandas as gpd
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib import rc
from scipy import stats
from shapely.geometry import MultiPolygon, Polygon

from outbreak_size import adjacency

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)
fig_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")

font = {"family": "Nimbus Roman", "size": 28}
rc("font", **font)


def setup_plot(nrows, ncols):
    fig, ax = plt.subplots(nrows, ncols, figsize=(30, 7.5))
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
    ax.legend()
    ax.set_xlim([-1, 1])


def map_fips_code_data(
    fips, vals, fig, ax, cbar_label, cbar_type, point_data=None
):
    """
    Create a static US county map using GeoPandas
    """
    # Load the county boundaries
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
            cbar.set_ticks([1.25, 4.0, 8.0, 12.5, 17.5, 50])
            cbar.set_ticklabels(
                [
                    "≤2.5%",
                    "2.5%-5%",
                    "5%-10%",
                    "10%-15%",
                    "15%-20%",
                    ">20%",
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
    point_df = pd.DataFrame(point_data, columns=["GEOID", "point_value"])
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


if __name__ == "__main__":
    with open(
        os.path.join(
            output_dir,
            "samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_[2021, 2022, 2023, 2024, 2025].pkl",
        ),
        "rb",
    ) as file:
        posterior_samples = pickle.load(file)

    fig, ax = setup_plot(1, 3)  # type: ignore

    ax[0].text(  # type: ignore
        -0.25,
        1.08,
        "a",
        transform=ax[0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )

    plot_car_effect_correlation(
        2 * posterior_samples["spatial_correlation"] - 1,
        fig,
        ax[0],  # type: ignore
    )

    ax[1].text(  # type: ignore
        -0.25,
        1.08,
        "b",
        transform=ax[1].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )

    # we have to link each sample's inferred spatial effect to the corresponding county
    # calculate p values
    raw_spatial_effect = posterior_samples["raw_spatial_effect"]
    # calculate the p-value for each spatial effect
    p_values = 2 * np.minimum(
        np.mean(raw_spatial_effect > 0, axis=0),
        np.mean(raw_spatial_effect < 0, axis=0),
    )

    adjacency_matrix = pl.read_csv(
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

    df = df.filter(pl.col("year") >= 2021)

    adjacency_matrix = adjacency.assemble_adjacency_matrix(
        adjacency_matrix,
        df["fips"].unique(maintain_order=True).to_list(),
    )

    # figure out who has a neighbor
    has_a_neighor = adjacency_matrix.sum(axis=0) > 0
    counties_with_neighbors = (
        df["fips"].unique(maintain_order=True).to_numpy()[has_a_neighor]
    )
    df = df.filter(pl.col("fips").is_in(counties_with_neighbors))
    df = df.select(pl.col("fips").unique(maintain_order=True))

    df = df.with_columns(p_values=pl.Series(p_values))

    map_fips_code_data(
        df["fips"].to_list(),
        df["p_values"].to_numpy(),
        fig,
        ax[1],  # type: ignore
        cbar_label="Spatial effect p-value",
        cbar_type="probability",
    )

    ax[2].text(  # type: ignore
        -0.25,
        1.08,
        "c",
        transform=ax[2].transAxes,  # type: ignore
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
        ax[2],  # type: ignore
    )

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
