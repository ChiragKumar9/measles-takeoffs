from numpyro import enable_x64

enable_x64()

import os
import pickle

import geopandas as gpd
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import yaml
from matplotlib import rc
from shapely.geometry import MultiPolygon, Polygon

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


def setup_plot(nrows, ncols):
    fig, ax = plt.subplots(nrows, ncols, figsize=(20, 15))
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


def plot_metric_over_values(df, samples, with_importation, x_label, fig, ax):
    colors = ["chartreuse", "forestgreen", "darkgreen"]
    shapes = ["o", "s", "^"]
    for idx, year in enumerate([2023, 2024, 2025]):
        year_mask = (df["year"] == year).to_jax()
        df_year = df.filter(pl.col("year") == year)
        if with_importation is None:
            outcome = "cases_outbreak"
        else:
            outcome = "cases"
        cases_sorted = df_year[outcome].sort().unique().to_list()
        metric_median = jnp.array([])
        metric_lower = jnp.array([])
        metric_upper = jnp.array([])
        metric_p_values_median = jnp.array([])
        for val in cases_sorted:
            mask = df_year[outcome].to_jax() >= val
            true = df_year.filter(pl.col(outcome) >= float(val))[
                outcome
            ].to_jax()
            samples_year = samples[idx][:, mask]
            # get the averages to match
            if with_importation is not None:
                samples_year *= with_importation[:, year_mask][:, mask]
            # samples_year = samples_year - jnp.mean(samples_year, axis=1, keepdims=True)
            # samples_year = samples_year / jnp.std(samples_year, axis=1, keepdims=True) * jnp.std(true)
            # samples_year = samples_year + jnp.mean(true)
            # metric_values = jnp.array([])
            # metric_p_values = jnp.array([])
            metric_values = jnp.array(
                [jnp.corrcoef(true, c)[0, 1] ** 2 * 100 for c in samples_year]
            )

            # metric_p_values_median = jnp.append(metric_p_values_median, jnp.nanmedian(metric_p_values))
            metric_median = jnp.append(
                metric_median, jnp.nanmedian(metric_values)
            )
            metric_lower = jnp.append(
                metric_lower, jnp.nanpercentile(metric_values, 2.5)
            )
            metric_upper = jnp.append(
                metric_upper, jnp.nanpercentile(metric_values, 97.5)
            )
        # truncate cases_sorted because some of the metric_values may have been nan
        # cases_sorted = cases_sorted[: len(metric_median)]
        ax.errorbar(
            cases_sorted,
            metric_median,
            yerr=np.array(
                [
                    [metric_median.__array__() - metric_lower.__array__()],
                    [metric_upper.__array__() - metric_median.__array__()],
                ]
            )[:, 0, :],
            label=f"{year}",
            capsize=0,
            fmt=shapes[idx],
            alpha=0.6,
            ecolor=colors[idx],
            color=colors[idx],
        )
        # ax.errorbar(
        #     cases_sorted,
        #     metric_median,
        #     yerr=np.array([[metric_median.__array__() - metric_lower.__array__()], [metric_upper.__array__() - metric_median.__array__()]])[:, 0, :],
        #     capsize=0,
        #     fmt = 'none',
        #     alpha = 0.6,
        #     ecolor = 'grey',
        # )
        # ax.scatter(
        #     cases_sorted,
        #     metric_median,
        #     label=f"{year}",
        #     marker=shapes[idx],
        #     c=metric_p_values_median,
        #     cmap = 'viridis',
        #     vmin = 0,
        #     vmax = 1,
        # )
    # add colorbar
    # norm = Normalize(vmin = 0, vmax = 1)
    # fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax, label='Median p-value')
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$R^2$ (%)")
    ax.legend()
    ax.set_ylim([0, 103])
    if with_importation is None:
        ax.set_xscale("symlog")
    else:
        ax.set_xscale("log")
        ax.set_xlim(0.9)


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

    if cbar_type == "percentile_risk":
        norm = mcolors.Normalize(vmin=0, vmax=100)
        breakpoints = [-1, 20, 40, 60, 80, 101]
        cmap = plt.get_cmap("Reds")
        cmap = mcolors.ListedColormap(
            [cmap(i) for i in np.linspace(0.3, 1.0, 5)]
        )
    elif cbar_type == "probability":
        norm = mcolors.Normalize(vmin=0, vmax=1)
        breakpoints = [-1, 0.2, 0.4, 0.6, 0.8, 1.01]
        cmap = plt.get_cmap("Reds")
        cmap = mcolors.ListedColormap(
            [cmap(i) for i in np.linspace(0.3, 1.0, 5)]
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

        if cbar_type == "percentile_risk":
            cbar.set_ticks([20, 40, 60, 80, 100])
            cbar.set_ticklabels(
                [
                    "Bottom 20%",
                    "20%-40%",
                    "40%-60%",
                    "60%-80%",
                    "Top 20%",
                ]
            )
            cbar.ax.tick_params(labelsize=24, rotation=-60, length=0)

        if cbar_type == "probability":
            cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            cbar.set_ticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
            cbar.ax.tick_params(labelsize=24, rotation=-60, length=0)

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


if __name__ == "__main__":
    fig, ax = setup_plot(2, 2)  # type: ignore

    ax[0, 0].text(  # type: ignore
        -0.05,
        1.08,
        "a",
        transform=ax[0, 0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    df_train = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
    )

    import_df = pl.read_csv(
        os.path.join(output_dir, "empirical_importation_probability.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
    )
    with open(
        os.path.join(
            output_dir,
            "empirical_importation_probability_samples.pkl",
        ),
        "rb",
    ) as f:
        samples = pickle.load(f)
    import_df = import_df.with_columns(
        importation_probability_samples=pl.Series(samples)
    )
    import_df = import_df.filter(pl.col("type") == "train")

    # join up with df
    df_train = df_train.join(
        import_df.select(
            [
                "county",  # rather than fips because we have some regions
                "state",
                "year",
                "importation_probability_samples",
            ]
        ),
        on=["county", "state", "year"],
        how="left",
    )
    importation_samples = jnp.array(
        np.vstack(df_train["importation_probability_samples"].to_numpy())  # type: ignore
    ).T

    samples = []
    for idx, year in enumerate([2023, 2024, 2025]):
        with open(
            os.path.join(
                output_dir,
                f"samples_partial_pooled_poisson_validation_testing_{year}_pool=True_CAR=False_All covariates.pkl",
            ),
            "rb",
        ) as f:
            m = pickle.load(
                f,
            )
        samples.append(m)

    plot_metric_over_values(
        df_train,
        samples,
        with_importation=importation_samples,
        x_label="Case threshold",
        fig=fig,
        ax=ax[0, 0],  # type: ignore
    )

    df = pl.read_csv(
        os.path.join(output_dir, "assembled_projection_results.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
    )

    ax[0, 1].text(  # type: ignore
        -0.05,
        1.08,
        "b",
        transform=ax[0, 1].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    map_fips_code_data(
        df["fips"],
        df["median_unconditioned_risk_rank"],
        fig,
        ax[0, 1],  # type: ignore
        cbar_label="Risk percentile",
        cbar_type="percentile_risk",
        point_data=None,
    )

    ax[1, 0].text(  # type: ignore
        -0.05,
        1.08,
        "c",
        transform=ax[1, 0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    map_fips_code_data(
        df["fips"],
        df["median_conditioned_risk_rank"],
        fig,
        ax[1, 0],  # type: ignore
        cbar_label="Import-conditioned risk percentile",
        cbar_type="percentile_risk",
        point_data=None,
    )

    ax[1, 1].text(  # type: ignore
        -0.05,
        1.08,
        "d",
        transform=ax[1, 1].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    map_fips_code_data(
        df["fips"],
        df["importation_probability_median"],
        fig,
        ax[1, 1],  # type: ignore
        cbar_label="Importation probability",
        cbar_type="probability",
        point_data=None,
    )

    plt.tight_layout()

    # save
    fig.savefig(
        os.path.join(fig_dir, "figure_validation_and_risk_projections.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(fig_dir, "figure_validation_and_risk_projections.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
