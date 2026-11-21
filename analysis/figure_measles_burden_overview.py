import os

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import yaml
from addfips import AddFIPS
from matplotlib import rc
from scipy.stats import linregress
from shapely.geometry import MultiPolygon, Polygon

from outbreak_size import vaccination_wrangling
from outbreak_size.baseline import (
    as_of_values,
    county_source_rows,
    panel_c_coverage,
    trend_to_year,
)
from outbreak_size.interventions import STATES_WITH_BANS
from outbreak_size.scenarios import project_coverage_scenarios

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
fig_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)

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


def map_fips_code_data(
    fips,
    vals,
    fig,
    ax,
    cbar_label,
    cbar_type,
    point_data=None,
    legend_bbox_to_anchor=None,
    cbar_orientation="horizontal",
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

    # geopandas/matplotlib errors on an empty GeoDataFrame (can't compute an
    # aspect ratio with no geometries), which happens when every county has
    # data, e.g. the R01 coverage map's backfilled version
    if bool(no_data_mask.any()):
        counties_with_data[no_data_mask].plot(
            ax=ax, color="lightgrey", linewidth=0.3, edgecolor="grey"
        )

    # Create normalization for colorbar
    if cbar_type == "continuous_cases":
        norm = mcolors.Normalize(vmin=0, vmax=500)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "custom", ["#fee5d9", "#a50f15"], N=256
        )
    elif cbar_type == "exemptions":
        breakpoints = [-1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 100.0]
        norm = mcolors.BoundaryNorm(
            boundaries=breakpoints,
            ncolors=len(breakpoints) - 1,
        )
        cmap = mcolors.ListedColormap(
            [
                "#fee5d9",
                "#fcbba1",
                "#fc9272",
                "#fb6a4a",
                "#de2d26",
                "#a50f15",
            ]
        )
    elif cbar_type == "continuous_coverage":
        norm = mcolors.Normalize(vmin=70, vmax=100)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "custom", ["#e0ecf4", "#4d004b"], N=256
        )
    elif cbar_type == "bucketed_coverage":
        breakpoints = [-1.0, 70.0, 80.0, 90.0, 95.0, 100.0]
        norm = mcolors.BoundaryNorm(
            boundaries=breakpoints,
            ncolors=len(breakpoints) - 1,
        )
        # make the colormap
        cmap = mcolors.ListedColormap(
            [
                "#d7191c",  # 0-70
                "#fdae61",  # 70-80
                "#ffffbf",  # 80-90
                "#a6d96a",  # 90-95
                "#1a9641",  # 95-100
            ]
        )
    elif cbar_type == "binary":
        breakpoints = [0, 1, 2]
        norm = mcolors.BoundaryNorm(
            boundaries=breakpoints, ncolors=len(breakpoints) - 1
        )
        # make the colormap
        cmap = mcolors.ListedColormap(
            [
                "#fee5d9",  # 0 to 10
                "#a50f15",  # 10 to inf
            ]
        )
    elif cbar_type == "discrete":
        breakpoints = [min(vals) - 1, -2, -1, -0.2, 0.2, 1, 2, max(vals) + 1]
        norm = mcolors.BoundaryNorm(
            boundaries=breakpoints, ncolors=len(breakpoints) - 1
        )
        # make the colormap
        cmap = mcolors.ListedColormap(
            [
                "#8c510a",  # -inf to -2
                "#d8b365",  # -2 to -1
                "#f6e8c3",  # -1 to -0.2
                "#f5f5f5",  # -0.2 to 0.2
                "#c7eae5",  # 0.2 to 1
                "#80cdc1",  # 1 to 2
                "#01665e",  # 2 to inf
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

        if cbar_orientation == "horizontal":
            cbar = fig.colorbar(
                sm,
                ax=ax,
                shrink=0.7,  # Size of colorbar relative to axes
                aspect=20,  # Ratio of long to short dimensions
                pad=0.0,  # Distance between axes and colorbar
                extend="neither",  # Add arrow at top ('neither', 'both', 'min', 'max')
                orientation="horizontal",
            )
        else:
            # fig.colorbar(ax=ax) sizes the bar from the axes box *before*
            # the equal-aspect map shrinks it, so the bar would be taller
            # than the map; place it explicitly at the map's own extent
            ax.apply_aspect()
            map_box = ax.get_position()
            # trim the axes' autoscale margin so the bar spans only the
            # drawn counties (top of the lower 48 to bottom of Alaska)
            y_lo, y_hi = ax.get_ylim()
            x_lo, x_hi = ax.get_xlim()
            data_x0, data_y0, data_x1, data_y1 = (
                counties_with_data.total_bounds
            )
            bar_y0 = map_box.y0 + map_box.height * (data_y0 - y_lo) / (
                y_hi - y_lo
            )
            bar_h = map_box.height * (data_y1 - data_y0) / (y_hi - y_lo)
            # start just right of the easternmost county, not the axes edge
            bar_x0 = (
                map_box.x0
                + map_box.width * (data_x1 - x_lo) / (x_hi - x_lo)
                + 0.008
            )
            cax = fig.add_axes((bar_x0, bar_y0, 0.022, bar_h))
            cbar = fig.colorbar(sm, cax=cax, extend="neither")

        # Customize the colorbar; a vertical bar keeps its tick labels
        # horizontal and its title rotated clockwise (-90) along the bar,
        # with enough labelpad to clear the (wider) tick labels
        if cbar_orientation == "horizontal":
            cbar.set_label(cbar_label, rotation=0, labelpad=0, fontsize=28)
            bucket_tick_rotation = -60
            bucket_tick_size = 24
        else:
            cbar.set_label(
                cbar_label,
                rotation=-90,
                labelpad=30,
                fontsize=24,
            )
            bucket_tick_rotation = 0
            bucket_tick_size = 24

        # Set ticks
        if cbar_type == "binary":
            cbar.set_ticks([0.5, 1.5])
            cbar.set_ticklabels(["<10", r"$\geq$10"])
            cbar.ax.tick_params(labelsize=24, rotation=0)
        elif cbar_type == "discrete":
            cbar.set_ticks([-9, -1.2, -0.2, 0.2, 0.8, 1.7, 20])
            cbar.set_ticklabels(
                [
                    "< -2",  # BoundaryNorm bins are left-closed
                    "-2 to -1",
                    "-1 to -0.2",
                    "-0.2 to 0.2",
                    "0.2 to 1",
                    "1 to 2",
                    "≥2",
                ]
            )
            cbar.ax.tick_params(labelsize=24, rotation=-60, length=0)
        elif cbar_type == "bucketed_coverage":
            cbar.set_ticks([35, 75, 85, 92.5, 97.5])
            tick_text = ["< 70%", "70%-80%", "80%-90%", "90%-95%", "≥95%"]
            # the unit is in the colorbar title instead; shorter labels let
            # the rotated text be larger within its segment
            tick_text = [t.replace("%", "") for t in tick_text]
            cbar.set_ticklabels(tick_text)
            cbar.ax.tick_params(
                labelsize=bucket_tick_size,
                rotation=bucket_tick_rotation,
                length=0,
            )
        elif cbar_type == "exemptions":
            # midpoint of each bucket (the bar spaces buckets evenly and
            # interpolates within them), so every label is centered on its
            # color block
            cbar.set_ticks([0.75, 3.75, 7.5, 12.5, 17.5, 60])
            tick_text = [
                "< 2.5%",
                "2.5%-5%",
                "5%-10%",
                "10%-15%",
                "15%-20%",
                "≥20%",
            ]
            if cbar_orientation == "vertical":
                tick_text = [t.replace("%", "") for t in tick_text]
            cbar.set_ticklabels(tick_text)
            cbar.ax.tick_params(
                labelsize=bucket_tick_size,
                rotation=bucket_tick_rotation,
                length=0,
            )

        if cbar_orientation == "vertical":
            for tick_label in cbar.ax.get_yticklabels():
                tick_label.set_ha("left")
                tick_label.set_va("center")

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
        size_values = [5, 100, 10, 50, 500]  # Example sizes to show
        size_labels = ["1-5", "100", "10", "50", "500"]

        legend_elements = []
        for i, (size, label) in enumerate(zip(size_values, size_labels)):
            if i == 3:
                legend_elements.append(
                    plt.scatter([], [], s=0, c="none", label=" ")
                )
            legend_elements.append(
                plt.scatter([], [], s=size, c="red", alpha=0.8, label=label)
            )

        # default: center the legend under the drawn counties (not the axes
        # box, whose autoscale margin is wider than the map)
        legend_loc = "upper right"
        if legend_bbox_to_anchor is None:
            ax.apply_aspect()
            x_lo, x_hi = ax.get_xlim()
            data_x0, _, data_x1, _ = counties_with_data.total_bounds
            map_center = ((data_x0 + data_x1) / 2 - x_lo) / (x_hi - x_lo)
            legend_loc = "upper center"
            legend_bbox_to_anchor = (map_center, 0.035)

        # Add the legend
        ax.legend(
            handles=legend_elements,
            title="Measles cases (N)",
            loc=legend_loc,
            frameon=False,
            scatterpoints=1,
            ncol=3,
            bbox_to_anchor=legend_bbox_to_anchor,
        )

    # Remove axis
    ax.set_axis_off()

    return fig, ax


# The county of the map that shows each place of the 2026 case data (a place
# of several counties is drawn on its most populous member, by the `pop` of the
# projection datasets)
DISPLAY_COUNTY_2026 = {
    "Coastal Health District, Georgia": "13051",  # Chatham
    "HHS District 3, Iowa": "19033",  # Cerro Gordo
    "HHS District 4, Iowa": "19155",  # Pottawattamie
    "HHS District 5, Iowa": "19153",  # Polk
    "HHS District 6, Iowa": "19179",  # Wapello
    "Northeast HCC Region, Kansas": "20177",  # Shawnee
    "TriCounty, Utah": "49047",  # Uintah
    "Weber-Morgan, Utah": "49057",  # Weber
}

# the last year of the case data is a part year (cases to September/October)
PART_YEAR = 2026


def plot_cases_over_time(df, fig, ax):
    df_overall = df.group_by("year").agg(pl.sum("cases"))
    df_overall = df_overall.sort("year")
    full = df_overall.filter(pl.col("year") < PART_YEAR)
    part = df_overall.filter(pl.col("year") >= PART_YEAR)
    ax.plot(
        full["year"],
        full["cases"],
        marker="^",
        markersize=10,
        color="black",
        linestyle="--",
        label="Cases",
    )
    if part.height:
        # the part year: an open marker, joined to the year before by a
        # dotted line
        ax.plot(
            [full["year"][-1], *part["year"]],
            [full["cases"][-1], *part["cases"]],
            color="black",
            linestyle=":",
        )
        ax.plot(
            part["year"],
            part["cases"],
            marker="^",
            markersize=10,
            markerfacecolor="white",
            color="black",
            linestyle="none",
        )
    ax.set_xlabel("Year")
    ax.set_ylabel("Measles cases (N)")
    ax.set_xticks([2014, 2017, 2020, 2023, 2026])
    ax.set_ylim(0)

    counts, bins = np.histogram(
        df["year"], bins=np.arange(2011, PART_YEAR + 2) - 0.5
    )
    ax2 = ax.twinx()
    is_part = bins[:-1] >= PART_YEAR - 0.5
    ax2.bar(
        bins[:-1][~is_part],
        counts[~is_part],
        width=np.diff(bins)[~is_part],
        alpha=0.2,
        align="edge",
        edgecolor="grey",
        label="Number of counties",
    )
    # the part year: an open bar, like its open marker
    ax2.bar(
        bins[:-1][is_part],
        counts[is_part],
        width=np.diff(bins)[is_part],
        align="edge",
        facecolor="none",
        edgecolor=mcolors.to_rgba("C0", 0.6),
        linestyle="--",
        linewidth=3,
    )
    ax2.set_ylabel(
        "Counties with measles cases (N)", rotation=-90, labelpad=30
    )
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.get_xaxis().tick_bottom()
    ax2.get_yaxis().tick_right()
    ax2.tick_params(axis="x", direction="out")
    ax2.tick_params(axis="y", direction="out")
    # offset the spines
    for spine in ax2.spines.values():
        spine.set_position(("outward", 5))
    # put the grid behind
    ax2.set_axisbelow(True)
    # the counties of a year can pass 250 (2026 has more than 280)
    ax2.set_ylim(0, max(250, 50 * np.ceil(counts.max() / 50)))

    l1, h1 = ax.get_legend_handles_labels()
    l2, h2 = ax2.get_legend_handles_labels()
    ax.legend([*l1, *l2], [*h1, *h2], edgecolor="black")

    return fig, ax


def county_universe(data_dir):
    """Every county of the 2020 shapefile: fips, county and state name."""
    counties = gpd.read_file(
        os.path.join(data_dir, "cb_2020_us_county_20m.shp")
    )
    states = gpd.read_file(
        os.path.join(data_dir, "cb_2020_us_state_20m.shp")
    ).rename(columns={"NAME": "state"})
    counties = counties[
        ~counties["STATEFP"].isin(["60", "66", "69", "72", "78"])
    ].merge(states[["STATEFP", "state"]], on="STATEFP")
    return pl.DataFrame(
        {
            "fips": counties["GEOID"].tolist(),
            "county": counties["NAME"].tolist(),
            "state": counties["state"].tolist(),
        }
    ).sort("fips")


if __name__ == "__main__":
    # the datasets that include the 2026 cases (the _with_2026 files of 00 and 01)
    df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset_with_2026.csv"),
        schema_overrides={"fips": pl.String},
    )
    df = df.with_columns(
        state_county_year=pl.concat_str(
            [
                pl.col("state"),
                pl.lit("_"),
                pl.col("county"),
                pl.lit("_"),
                pl.col("year").cast(pl.String),
            ]
        )
    )
    # plus add anything that may be missing from 2021-2026
    missing = pl.read_csv(
        os.path.join(data_dir, "missing_counties_with_2026.csv"),
        schema_overrides={"fips": pl.String},
    )
    missing = missing.with_columns(
        state_county_year=pl.concat_str(
            [
                pl.col("state"),
                pl.lit("_"),
                pl.col("county"),
                pl.lit("_"),
                pl.col("year").cast(pl.String),
            ]
        )
    )
    missing = missing.filter(
        ~pl.col("state_county_year").is_in(df["state_county_year"])
    )
    df = pl.concat([df, missing], how="diagonal_relaxed")

    # "New York City" cases were assigned to Kings (00_explore_data.py:176),
    # but the 2025 row still has fips 36061 (Manhattan) because there's no
    # 2024 NY vaccine data to join against for any county; fix it to Kings'
    # own fips (36047) so it isn't plotted on the wrong county
    df = df.with_columns(
        fips=pl.when(
            (pl.col("state") == "New York") & (pl.col("county") == "Kings")
        )
        .then(pl.lit("36047"))
        .otherwise(pl.col("fips"))
    )

    # let's just plot Fulton and Gwinnett counties over Fulton county only
    df = (
        df.with_columns(
            fips=pl.when(pl.col("county") == "Fulton, Gwinett")
            .then(pl.lit("13121"))
            .otherwise(pl.col("fips")),
        )
        .with_columns(
            fips=pl.when(pl.col("county") == "District 1")
            .then(pl.lit("22071"))
            .otherwise(pl.col("fips"))
        )
        .with_columns(
            fips=pl.when(pl.col("county") == "Forsyth, Fulton, Paulding")
            .then(pl.lit("13117"))
            .otherwise(pl.col("fips")),
        )
        .with_columns(
            fips=pl.when(pl.col("county") == "Blue Ridge")
            .then(pl.lit("51023"))
            .otherwise(pl.col("fips")),
        )
        .with_columns(
            fips=pl.when(pl.col("county") == "East tennessee?")
            .then(pl.lit("47009"))
            .otherwise(pl.col("fips")),
        )
        .with_columns(
            fips=pl.when(pl.col("county") == "Central Utah, Utah")
            .then(pl.lit("49039"))
            .otherwise(pl.col("fips"))
        )
        .with_columns(
            fips=pl.when(pl.col("county") == "Southwest Health District, Utah")
            .then(pl.lit("49001"))
            .otherwise(pl.col("fips"))
        )
        .with_columns(
            fips=pl.when(pl.col("county") == "Western, Iowa")
            .then(pl.lit("19193"))
            .otherwise(pl.col("fips"))
        )
        .with_columns(
            fips=pl.when(pl.col("county") == "Eastern, Iowa")
            .then(pl.lit("19061"))
            .otherwise(pl.col("fips"))
        )
        .with_columns(
            fips=pl.when(
                pl.col("county") == "Mid-Cumberland Region, Tennessee"
            )
            .then(pl.lit("47043"))
            .otherwise(pl.col("fips"))
        )
        .with_columns(
            fips=pl.when(pl.col("county") == "Bear River, Utah")
            .then(pl.lit("49003"))
            .otherwise(pl.col("fips"))
        )
        .with_columns(
            fips=pl.when(pl.col("county") == "Southeast Health District, Utah")
            .then(pl.lit("49007"))
            .otherwise(pl.col("fips"))
        )
        .with_columns(
            fips=pl.when(pl.col("county") == "North, Alabama")
            .then(pl.lit("01089"))
            .otherwise(pl.col("fips"))
        )
        .with_columns(
            fips=pl.when(
                pl.col("county") == "Upper Cumberland Region, Tennessee"
            )
            .then(pl.lit("47015"))
            .otherwise(pl.col("fips"))
        )
        .with_columns(
            fips=pl.when(
                pl.col("county")
                == "Nashville-Davidson County Region, Tennessee"
            )
            .then(pl.lit("47037"))
            .otherwise(pl.col("fips"))
        )
        .with_columns(
            # Carver: a county of the region that never has a case row of its
            # own (Anoka, the county this used to be, has one in 2025)
            fips=pl.when(
                pl.col("county") == "Twin Cities Metro Area, Minnesota"
            )
            .then(pl.lit("27019"))
            .otherwise(pl.col("fips"))
        )
        .with_columns(
            fips=pl.when(pl.col("county") == "Central, Iowa")
            .then(pl.lit("19179"))
            .otherwise(pl.col("fips"))
        )
        .with_columns(
            fips=pl.when(pl.col("county") == "Region 9, Louisiana")
            .then(pl.lit("22063"))
            .otherwise(pl.col("fips"))
        )
        .with_columns(
            fips=pl.when(pl.col("county") == "Northwest Region, Virginia")
            .then(pl.lit("51015"))
            .otherwise(pl.col("fips"))
        )
        .with_columns(
            fips=pl.when(pl.col("county") == "Eastern Region, Virginia")
            .then(pl.lit("51550"))
            .otherwise(pl.col("fips"))
        )
        .with_columns(
            fips=pl.when(pl.col("county") == "Upstate, South Carolina")
            .then(pl.lit("45045"))
            .otherwise(pl.col("fips"))
        )
        .with_columns(
            fips=pl.when(pl.col("county") == "Piedmont, Virginia")
            .then(pl.lit("51007"))
            .otherwise(pl.col("fips"))
        )
        .with_columns(
            fips=pl.when(pl.col("county") == "Central Region, Virginia")
            .then(pl.lit("51025"))
            .otherwise(pl.col("fips"))
        )
        .with_columns(
            fips=pl.when(pl.col("county") == "Region 1, Louisiana")
            .then(pl.lit("22051"))
            .otherwise(pl.col("fips"))
        )
        # the places of the 2026 case data
        .with_columns(
            fips=pl.col("county").replace_strict(
                DISPLAY_COUNTY_2026, default=pl.col("fips")
            )
        )
    )

    df_by_fips = df.group_by("fips")
    overall_cases = df_by_fips.agg(cases=pl.sum("cases"))

    # map of total cases
    fig, ax = setup_plot(2, 2)  # type: ignore
    ax[0, 0].text(  # type: ignore
        -0.05,
        1.23,
        "a",
        transform=ax[0, 0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    map_fips_code_data(
        None,
        None,
        fig,
        ax[0, 0],  # type: ignore
        cbar_label=None,
        cbar_type=None,
        point_data=overall_cases[["fips", "cases"]].with_columns(
            cases=pl.when(pl.col("cases") < 5)
            .then(pl.lit(5))
            .otherwise(pl.col("cases"))
        ),
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

    plot_cases_over_time(df, fig, ax[0, 1])  # type: ignore

    vaccination_data = pl.read_excel(
        os.path.join(
            data_dir,
            "Compiled_Vaccine_Exemption_Data_R1_110225_Hawaiied_AK_CA_Updated.xlsx",
        ),
        schema_overrides={"FIPS": pl.String},
    )
    # the 2025-26 school year ("year" 2025) the spreadsheet lacks: it feeds the
    # projections (baselines, scenarios) and panel d; panel c and the R01
    # coverage map stay on 2024-25
    vaccination_data = vaccination_wrangling.add_2025_vaccination_rows(
        vaccination_data, data_dir
    )
    # Tennessee's 2025-26 rows, from its own file
    vaccination_data = vaccination_wrangling.add_tennessee_2025_rows(
        vaccination_data, data_dir
    )

    # cells of the spreadsheet the data owner's revised file (or the raw
    # state files) have right, before the places that failed the QC are removed
    vaccination_data = vaccination_wrangling.fix_known_values(
        vaccination_data, data_dir
    )

    # remove the places that failed the data QC (params.yml)
    vaccination_data = vaccination_wrangling.apply_qc_exclusions(
        vaccination_data, params["qc_exclusions"]
    )
    # Wyoming's year labels (recording year, not school year) and the 2024
    # Hawaii exemptions of the data owner's revised file, as in 00
    vaccination_data = vaccination_wrangling.fix_wyoming_year_labels(
        vaccination_data, data_dir, mmr_column="MMR_cov", fips_column="FIPS"
    )
    vaccination_data = vaccination_wrangling.fill_hawaii_2024_exemptions(
        vaccination_data, data_dir
    )
    vaccination_data = vaccination_data.with_columns(
        MMR_cov=pl.when(
            (pl.col("county") == "Cocke")
            & (pl.col("state") == "Tennessee")
            & (pl.col("year") == 2024)
        )
        .then(pl.lit(94.5))
        # the sheet has the text ".89%" for Faulkner County, Arkansas, 2024; the
        # raw data give 89.89
        .when(
            (pl.col("county") == "Faulkner")
            & (pl.col("state") == "Arkansas")
            & (pl.col("year") == 2024)
        )
        .then(pl.lit(89.89))
        .otherwise(pl.col("MMR_cov"))
    )
    vaccination_data = vaccination_data.with_columns(
        med_exempt_pct=pl.when(
            (
                pl.col("state").is_in(
                    ["Wyoming", "Louisiana", "Vermont", "Mississippi"]
                )
            )
            & pl.col("med_exempt_pct").is_null()
        )
        .then(pl.lit(0))
        .otherwise(pl.col("med_exempt_pct"))
    )

    vaccination_data = vaccination_data.with_columns(
        FIPS=pl.when(pl.col("FIPS") == "510591")
        .then(pl.lit("51059"))
        .otherwise(pl.col("FIPS")),
        # no MMR coverage (missing or 0): use the all-vaccine coverage, as
        # 00_explore_data.py does
        estimated_mmr_cov=pl.when(
            pl.col("MMR_cov").cast(pl.Float64, strict=False).is_not_null()
            & (pl.col("MMR_cov").cast(pl.Float64, strict=False) != 0.0)
        )
        .then(pl.col("MMR_cov"))
        .otherwise(pl.col("all_vacc_cov")),
    )

    vaccination_data = vaccination_data.with_columns(
        estimated_mmr_cov=pl.col("estimated_mmr_cov").cast(
            pl.Float64, strict=False
        )
    )

    vaccination_data = vaccination_data.rename({"FIPS": "fips"})

    vaccination_data = vaccination_data.with_columns(
        fips=pl.when(pl.col("fips") == "11")
        .then(pl.lit("11001"))
        .otherwise(pl.col("fips"))
    )

    state_fips_to_expand = {
        "higher_level_fips": [
            *["10"] * 3,
            *["44"] * 5,
            *["54"] * 55,
        ],
        "county_fips": [
            *["10001", "10003", "10005"],
            *[f"44{str(2 * i + 1).zfill(3)}" for i in range(5)],
            *[f"54{str(2 * i + 1).zfill(3)}" for i in range(55)],
        ],
    }

    state_fips_to_expand = pl.DataFrame(state_fips_to_expand)
    vaccination_data = vaccination_data.join(
        state_fips_to_expand,
        left_on="fips",
        right_on="higher_level_fips",
        how="left",
        maintain_order="left",
    )
    vaccination_data = vaccination_data.with_columns(
        fips=pl.when(pl.col("county_fips").is_not_null())
        .then(pl.col("county_fips"))
        .otherwise(pl.col("fips"))
    )

    alaska_fips = {
        "old_alaska": [
            "20200",
            *["21220"] * 4,
            *["22900"] * 4,
            "21700",
            *["21850"] * 3,
            *["22820"] * 10,
            *["21640"] * 7,
        ],
        "new_alaska": [
            "02020",
            "02150",
            "02122",
            "02063",
            "02066",
            "02290",
            "02068",
            "02090",
            "02240",
            "02170",
            "02180",
            "02185",
            "02188",
            "02100",
            "02282",
            "02105",
            "02220",
            "02198",
            "02130",
            "02275",
            "02195",
            "02110",
            "02230",
            "02013",
            "02016",
            "02164",
            "02060",
            "02070",
            "02050",
            "02158",
        ],
    }

    alaska_fips = pl.DataFrame(alaska_fips)
    vaccination_data = vaccination_data.join(
        alaska_fips,
        left_on="fips",
        right_on="old_alaska",
        how="left",
        maintain_order="left",
    )
    vaccination_data = vaccination_data.with_columns(
        fips=pl.when(pl.col("new_alaska").is_not_null())
        .then(pl.col("new_alaska"))
        .otherwise(pl.col("fips"))
    )

    nebraska_fips = {
        "old_nebraska": [
            *["310012"] * 3,
            "310021",
            "310002",
            *["310005"] * 4,
            *["310008"] * 4,
            *["310007"] * 4,
            "310003",
            *["310015"] * 9,
            *["310020"] * 9,
            *["310010"] * 4,
            *["310018"] * 12,
            *["310011"] * 5,
            *["310006"] * 2,
            *["310013"] * 4,
            *["310009"] * 5,
            *["310017"] * 9,
            *["310004"] * 3,
            *["310014"] * 7,
            *["310016"] * 6,
        ],
        "new_nebraska": [
            "31079",
            "31081",
            "31121",
            "31043",
            "31055",
            "31011",
            "31037",
            "31125",
            "31141",
            "31021",
            "31039",
            "31119",
            "31167",
            "31023",
            "31143",
            "31159",
            "31185",
            "31109",
            "31009",
            "31041",
            "31071",
            "31077",
            "31093",
            "31115",
            "31163",
            "31175",
            "31183",
            "31003",
            "31015",
            "31017",
            "31031",
            "31089",
            "31103",
            "31107",
            "31139",
            "31149",
            "31027",
            "31051",
            "31173",
            "31179",
            "31007",
            "31013",
            "31033",
            "31045",
            "31049",
            "31069",
            "31075",
            "31165",
            "31157",
            "31123",
            "31161",
            "31105",
            "31059",
            "31067",
            "31095",
            "31151",
            "31169",
            "31153",
            "31025",
            "31001",
            "31035",
            "31129",
            "31181",
            "31097",
            "31127",
            "31131",
            "31133",
            "31147",
            "31029",
            "31057",
            "31063",
            "31065",
            "31085",
            "31087",
            "31101",
            "31135",
            "31145",
            "31053",
            "31177",
            "31155",
            "31019",
            "31047",
            "31061",
            "31073",
            "31083",
            "31099",
            "31137",
            "31005",
            "31091",
            "31111",
            "31113",
            "31117",
            "31171",
        ],
    }

    nebraska_fips = pl.DataFrame(nebraska_fips)
    vaccination_data = vaccination_data.join(
        nebraska_fips,
        left_on="fips",
        right_on="old_nebraska",
        how="left",
        maintain_order="left",
    )
    vaccination_data = vaccination_data.with_columns(
        fips=pl.when(pl.col("new_nebraska").is_not_null())
        .then(pl.col("new_nebraska"))
        .otherwise(pl.col("fips"))
    )

    # 4-character county FIPS lost their leading zero (AL, AK, AZ, ... ), and
    # so did the 1-character state FIPS, which the map needs to fill counties
    # from a state-level value
    vaccination_data = vaccination_data.with_columns(
        fips=pl.when(pl.col("fips").str.len_chars() == 4)
        .then(pl.col("fips").str.zfill(5))
        .when(pl.col("fips").str.len_chars() == 1)
        .then(pl.col("fips").str.zfill(2))
        .otherwise(pl.col("fips"))
    )

    # San Benito (CA) and Lincoln (MT) carry the FIPS of Sacramento and Madison
    vaccination_data = vaccination_wrangling.fix_wrong_fips(vaccination_data)

    vaccination_data = vaccination_data.with_columns(
        fips=pl.when(
            (pl.col("fips").str.len_chars() > 5)
            & (pl.col("state") != "Kansas")
        )
        .then(pl.col("fips").str.slice(0, 5))
        .otherwise(pl.col("fips"))
    )

    kansas = vaccination_wrangling.read_kansas_data(data_dir)
    kansas = kansas.rename({"mmr_cov": "estimated_mmr_cov"})
    vaccination_data = pl.concat(
        [vaccination_data, kansas], how="diagonal_relaxed"
    )

    # replace the spreadsheet's New York rows with New York.xlsx (as
    # 00_explore_data.py does)
    vaccination_data = vaccination_wrangling.replace_state_rows(
        vaccination_data,
        "New York",
        vaccination_wrangling.read_new_york_data(data_dir=data_dir),
    )

    montana_vacc = vaccination_wrangling.read_montana_data(data_dir=data_dir)
    vaccination_data = vaccination_data.join(
        montana_vacc,
        on=["state", "county", "year"],
        how="left",
        suffix="_montana",
        maintain_order="left",
    )
    vaccination_data = vaccination_data.with_columns(
        all_vacc_cov=pl.when(
            (pl.col("state") == "Montana") & pl.col("all_vacc_cov").is_null()
        )
        .then(pl.col("all_vacc_cov_montana"))
        .otherwise(pl.col("all_vacc_cov")),
        MMR_cov=pl.when(
            (pl.col("state") == "Montana") & pl.col("MMR_cov").is_null()
        )
        .then(pl.col("MMR_cov_montana"))
        .otherwise(pl.col("MMR_cov")),
        nonmed_exempt_pct=pl.when(
            (pl.col("state") == "Montana")
            & pl.col("nonmed_exempt_pct").is_null()
        )
        .then(pl.col("nonmed_exempt_pct_montana"))
        .otherwise(pl.col("nonmed_exempt_pct")),
        med_exempt_pct=pl.when(
            (pl.col("state") == "Montana") & pl.col("med_exempt_pct").is_null()
        )
        .then(pl.col("med_exempt_pct_montana"))
        .otherwise(pl.col("med_exempt_pct")),
        estimated_mmr_cov=pl.when(
            (pl.col("state") == "Montana")
            & pl.col("estimated_mmr_cov").is_null()
        )
        .then(pl.col("estimated_mmr_cov_montana"))
        .otherwise(pl.col("estimated_mmr_cov")),
    ).drop(
        [
            "MMR_cov_montana",
            "all_vacc_cov_montana",
            "nonmed_exempt_pct_montana",
            "med_exempt_pct_montana",
            "estimated_mmr_cov_montana",
        ]
    )

    vaccination_data = vaccination_data.with_columns(
        fips=pl.when((pl.col("state") == "Montana") & pl.col("fips").is_null())
        .then(pl.col("fips").drop_nulls().first().over("state", "county"))
        .otherwise(pl.col("fips"))
    )

    oklahoma_vacc = vaccination_wrangling.read_oklahoma_data(data_dir=data_dir)
    vaccination_data = vaccination_data.join(
        oklahoma_vacc,
        on=["state", "county", "year"],
        how="left",
        suffix="_oklahoma",
        maintain_order="left",
    )
    vaccination_data = vaccination_data.with_columns(
        all_vacc_cov=pl.when(
            (pl.col("state") == "Oklahoma") & pl.col("all_vacc_cov").is_null()
        )
        .then(pl.col("all_vacc_cov_oklahoma"))
        .otherwise(pl.col("all_vacc_cov")),
        MMR_cov=pl.when(
            (pl.col("state") == "Oklahoma") & pl.col("MMR_cov").is_null()
        )
        .then(pl.col("MMR_cov_oklahoma"))
        .otherwise(pl.col("MMR_cov")),
        nonmed_exempt_pct=pl.when(
            (pl.col("state") == "Oklahoma")
            & pl.col("nonmed_exempt_pct").is_null()
        )
        .then(pl.col("nonmed_exempt_pct_oklahoma"))
        .otherwise(pl.col("nonmed_exempt_pct")),
        med_exempt_pct=pl.when(
            (pl.col("state") == "Oklahoma")
            & pl.col("med_exempt_pct").is_null()
        )
        .then(pl.col("med_exempt_pct_oklahoma"))
        .otherwise(pl.col("med_exempt_pct")),
        estimated_mmr_cov=pl.when(
            (pl.col("state") == "Oklahoma")
            & pl.col("estimated_mmr_cov").is_null()
        )
        .then(pl.col("estimated_mmr_cov_oklahoma"))
        .otherwise(pl.col("estimated_mmr_cov")),
    ).drop(
        [
            "MMR_cov_oklahoma",
            "all_vacc_cov_oklahoma",
            "nonmed_exempt_pct_oklahoma",
            "med_exempt_pct_oklahoma",
            "estimated_mmr_cov_oklahoma",
        ]
    )

    # Oklahoma's 2024 rows come without a FIPS: take it from the same county's
    # other years (before the state imputation below, which needs it)
    ok_2024 = (pl.col("state") == "Oklahoma") & (pl.col("year") == 2024)
    print(
        "Oklahoma 2024 rows with a FIPS before the fill: "
        f"{vaccination_data.filter(ok_2024 & pl.col('fips').is_not_null()).height}"
    )
    vaccination_data = vaccination_data.with_columns(
        fips=pl.when(
            (pl.col("state") == "Oklahoma") & pl.col("fips").is_null()
        )
        .then(pl.col("fips").drop_nulls().first().over("state", "county"))
        .otherwise(pl.col("fips"))
    )
    print(
        "Oklahoma 2024 rows with a FIPS after the fill: "
        f"{vaccination_data.filter(ok_2024 & pl.col('fips').is_not_null()).height}"
    )

    # the rows that pool counties (Issaquena/Sharkey) go to their members, as
    # in 00_explore_data.py
    vaccination_data = vaccination_wrangling.expand_pooled_rows(
        vaccination_data,
        [
            c
            for c in (
                "nonmed_exempt_pct",
                "med_exempt_pct",
                "MMR_cov",
                "all_vacc_cov",
                "estimated_mmr_cov",
            )
            if c in vaccination_data.columns
        ],
    )

    # Snapshot of the rows as reported, before the interpolation and the
    # state fill below. A row is county-reported when the county's own value
    # was reported: not a state value (county "State") copied to each county
    # (DE, RI, WV), not a region value copied to its counties (AK, NE), and
    # not the Oregon "North Central" region, which stands for Wasco
    reported_rows = vaccination_data.select(
        "state",
        "county",
        "fips",
        "year",
        pl.col("estimated_mmr_cov").fill_nan(None),
        pl.col("nonmed_exempt_pct")
        .cast(pl.Float64, strict=False)
        .fill_nan(None),
        pl.col("med_exempt_pct").cast(pl.Float64, strict=False).fill_nan(None),
        county_reported=(
            (pl.col("fips").str.len_chars() == 5)
            & (pl.col("county") != "State")
            & pl.col("county_fips").is_null()
            # one value copied to several counties (a region or a state)
            # appears under several FIPS
            & (pl.col("fips").n_unique().over("state", "county", "year") == 1)
            & ~(
                (pl.col("state") == "Oregon")
                & (pl.col("county") == "North Central")
            )
        ).fill_null(False),
    )

    # a series is a county (its fips): the counties of a state value copied to
    # all of them (DE, RI, WV) share the name "State", and interpolating over
    # that name steps through the rows in their (arbitrary) order, not through
    # the years
    vaccination_data = vaccination_data.sort(
        "state", "county", "fips", "year", maintain_order=True
    ).with_columns(
        [
            pl.col(col).interpolate().over("state", "county", "fips")
            for col in [
                "estimated_mmr_cov",
                "nonmed_exempt_pct",
                "med_exempt_pct",
            ]
        ]
    )

    vaccination_data = (
        vaccination_data.with_columns(
            estimated_mmr_cov_state=pl.col("estimated_mmr_cov")
            .filter(pl.col("county") == "State")
            .first()
            .over(["state", "year"]),
            nonmed_exempt_pct_state=pl.col("nonmed_exempt_pct")
            .filter(pl.col("county") == "State")
            .first()
            .over(["state", "year"]),
            med_exempt_pct_state=pl.col("med_exempt_pct")
            .filter(pl.col("county") == "State")
            .first()
            .over(["state", "year"]),
        )
        .with_columns(
            estimated_mmr_cov=pl.col("estimated_mmr_cov").fill_null(
                pl.col("estimated_mmr_cov_state")
            ),
            nonmed_exempt_pct=pl.col("nonmed_exempt_pct").fill_null(
                pl.col("nonmed_exempt_pct_state")
            ),
            med_exempt_pct=pl.col("med_exempt_pct").fill_null(
                pl.col("med_exempt_pct_state")
            ),
        )
        .drop(
            [
                "estimated_mmr_cov_state",
                "nonmed_exempt_pct_state",
                "med_exempt_pct_state",
            ]
        )
    )

    print(vaccination_data.filter(pl.col("estimated_mmr_cov") == 0.0))
    vaccination_data = vaccination_data.filter(
        pl.col("estimated_mmr_cov") != 0.0
    )
    print(vaccination_data.filter(pl.col("estimated_mmr_cov").is_null()))
    print(vaccination_data.filter(pl.col("estimated_mmr_cov").is_nan()))
    vaccination_data = vaccination_data.filter(
        pl.col("estimated_mmr_cov").is_not_null()
        & ~pl.col("estimated_mmr_cov").is_nan()
    )

    # panel c: 2024-2025 school-year coverage, with the 2023-2024 value carried
    # forward where 2024-2025 was not reported
    vaccination_data_latest = panel_c_coverage(reported_rows)

    print("\n" + "=" * 80)
    print("IMPUTING MISSING COUNTY FIPS FOR TARGET STATES")
    print("=" * 80)
    # the baselines use the rows before this imputation: it adds copies of
    # state values for years that have no state value
    vaccination_data_before_state_imputation = vaccination_data
    vaccination_data = vaccination_wrangling.impute_from_state_level_coverages(
        vaccination_data, data_dir
    )

    print("\n" + "=" * 80)
    print("POST-IMPUTATION DIAGNOSTICS")
    print("=" * 80)

    nebraska_all = vaccination_data.filter(pl.col("state") == "Nebraska")
    print(f"Total Nebraska rows after imputation: {nebraska_all.height}")

    nebraska_counties = nebraska_all.filter(
        pl.col("fips").str.len_chars() == 5
    )
    print(f"Nebraska county-level rows: {nebraska_counties.height}")
    print(f"Unique Nebraska counties: {nebraska_counties['fips'].n_unique()}")

    brown_county = vaccination_data.filter(pl.col("fips") == "31017")
    print(f"\nBrown County (31017) rows: {brown_county.height}")
    if brown_county.height > 0:
        print(f"Brown County years: {sorted(brown_county['year'].to_list())}")
        print(
            f"Brown County coverage values: {brown_county[['year', 'estimated_mmr_cov']].to_pandas()}"
        )
    else:
        print("Brown County NOT FOUND after imputation!")

    # Check if it's in the shapefile
    counties_gdf = gpd.read_file(
        os.path.join(data_dir, "cb_2020_us_county_20m.shp")
    )
    brown_in_shapefile = counties_gdf[counties_gdf["GEOID"] == "31017"]
    print(f"\nBrown County in shapefile: {len(brown_in_shapefile) > 0}")
    if len(brown_in_shapefile) > 0:
        print(
            f"Shapefile info: {brown_in_shapefile[['STATEFP', 'GEOID', 'NAME']].to_dict('records')}"  # type: ignore
        )

    print("=" * 80 + "\n")

    # the state values and every county, for the counties that have no value of
    # their own in some field
    state_series = vaccination_wrangling.state_level_series(data_dir)
    universe = county_universe(data_dir)

    validation_years = [2021, 2022, 2023, 2024, 2025, 2026, 2027]
    for year in validation_years:
        # every county as of the data before `year` (see as_of_values): the
        # three fields are brought to one school year (`reference_year`, the
        # latest any of them has before `year`; it is the export's `year`),
        # the *_year columns are the school years of the observations and the
        # *_observed columns the observed values (before they are brought to
        # the reference year)
        values = as_of_values(
            reported_rows,
            vaccination_data_before_state_imputation,
            state_series,
            universe,
            year,
            states_with_bans=STATES_WITH_BANS,
        )
        # the school year an outbreak year uses (year - 1): a value observed
        # earlier is trended to it, as the scenarios do from the last known
        # school year, so that the files of 2026 and the baseline scenario of
        # 2027 are on the same footing
        values = trend_to_year(values, year - 1)
        vaccination_data_export = values.select(
            "fips",
            pl.col("reference_year").alias("year"),
            "estimated_mmr_cov",
            "nonmed_exempt_pct",
            "med_exempt_pct",
            "state",
            "county",
            "estimated_mmr_cov_year",
            "nonmed_exempt_pct_year",
            "med_exempt_pct_year",
            "estimated_mmr_cov_observed",
            "nonmed_exempt_pct_observed",
            "med_exempt_pct_observed",
            "estimated_mmr_cov_source",
            "nonmed_exempt_pct_source",
            "med_exempt_pct_source",
        ).sort("fips", descending=True)
        for field in (
            "estimated_mmr_cov",
            "nonmed_exempt_pct",
            "med_exempt_pct",
        ):
            by_source = (
                vaccination_data_export.group_by(f"{field}_source")
                .len()
                .sort(f"{field}_source")
            )
            print(
                f"Year {year}, {field} from: "
                f"{dict(zip(by_source[f'{field}_source'], by_source['len']))}"
            )

        brown_in_export = vaccination_data_export.filter(
            pl.col("fips") == "31017"
        )
        print(
            f"Year {year}: Brown County in export = {brown_in_export.height > 0}"
        )
        print(f"  Total counties exported: {vaccination_data_export.height}")

        vaccination_data_export.write_csv(
            os.path.join(data_dir, f"{year}_vaccination_data.csv")
        )
        if year == 2026:
            # keep this for the R01-only NME map below
            nme_2026 = vaccination_data_export

    # map
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
        vaccination_data_latest["fips"].to_list(),
        vaccination_data_latest["estimated_mmr_cov"].to_list(),
        fig,
        ax[1, 0],  # type: ignore
        cbar_label="2024-2025 MMR vaccine coverage (%)",
        cbar_type="bucketed_coverage",
    )

    # Flag how every value in panel c was obtained: reported for 2024-2025,
    # carried forward from 2023-2024, a region/state value copied to its
    # counties, or not available (grey). Counties of the map without any value
    # are added to the list.
    county_shapes = gpd.read_file(
        os.path.join(data_dir, "cb_2020_us_county_20m.shp")
    )
    state_shapes = gpd.read_file(
        os.path.join(data_dir, "cb_2020_us_state_20m.shp")
    ).rename(columns={"NAME": "state"})
    county_shapes = county_shapes[
        ~county_shapes["STATEFP"].isin(["60", "66", "69", "72", "78"])
    ].merge(state_shapes[["STATEFP", "state"]], on="STATEFP")
    no_county_value = (
        pl.DataFrame(
            {
                column: county_shapes[column].tolist()
                for column in ["GEOID", "STATEFP", "NAME", "state"]
            }
        )
        .filter(
            ~pl.col("GEOID").is_in(vaccination_data_latest["fips"].implode())
        )
        .select(
            level=pl.lit("county"),
            fips=pl.col("GEOID"),
            county=pl.col("NAME"),
            state=pl.col("state"),
            category=pl.lit("no_coverage"),
        )
    )
    panel_c_sources = pl.concat(
        [
            vaccination_data_latest.select(
                "fips",
                "county",
                "state",
                "source_year",
                "category",
                "estimated_mmr_cov",
            ),
            no_county_value,
        ],
        how="diagonal_relaxed",
    ).sort(["state", "fips"])
    panel_c_sources.write_csv(
        os.path.join(
            output_dir,
            "figure_measles_burden_overview_panel_c_coverage_sources.csv",
        )
    )
    print("\npanel c coverage sources:")
    print(panel_c_sources.group_by("category").len().sort("category"))
    for category in [
        "county_2023_carried_forward",
        "region_or_state_2023_carried_forward",
        "no_coverage",
    ]:
        by_state = (
            panel_c_sources.filter(pl.col("category") == category)
            .group_by("state")
            .len()
            .sort("state")
        )
        print(f"{category}: {dict(zip(by_state['state'], by_state['len']))}")

    # counties with a coverage value in the latest year (a region value counts
    # for each county it fills)
    n_below_95 = vaccination_data_latest.filter(
        pl.col("estimated_mmr_cov") < 95
    ).height
    print(
        f"counties below 95%: {n_below_95} of {vaccination_data_latest.height} "
        f"counties with a coverage value "
        f"({n_below_95 / vaccination_data_latest.height:.3f})"
    )

    # slopes of the counties' own reported values only; all of these are
    # counties, so the average is over counties
    def coverage_slopes(rows):
        slopes = (
            county_source_rows(rows)
            .group_by("fips", maintain_order=True)
            .map_groups(
                lambda group_df: pl.DataFrame(
                    {
                        "fips": [group_df["fips"][0]],
                        "slope": [
                            linregress(
                                group_df["year"], group_df["estimated_mmr_cov"]
                            ).slope  # type: ignore
                        ],
                    }
                )
            )
        )
        return slopes.filter(pl.col("slope").is_not_nan())

    # panel d and the national average slope of the scenarios: every year
    # there is (through school year 2025-26)
    vaccination_data_roc = coverage_slopes(reported_rows)

    # --- Summary statistics ---
    # averages over the reported values (no interpolated or state-filled
    # values): each county's own value counts once, and a region value (AK,
    # NE) or a state value (DE, RI, WV), which is copied to each of its
    # counties under several FIPS, counts once per region or state. By school
    # year: the school year starting in year y is outbreak year y + 1 (2010
    # is "2011", 2024 is "2025")
    has_value = pl.col("estimated_mmr_cov").is_not_null() & (
        pl.col("estimated_mmr_cov") != 0.0
    )
    reported_counties = reported_rows.filter(
        pl.col("county_reported") & has_value
    )
    reported_regions = reported_rows.filter(
        ~pl.col("county_reported")
        & (pl.col("fips").str.len_chars() == 5)
        & (pl.col("fips").n_unique().over("state", "county", "year") > 1)
        & has_value
    ).unique(
        subset=["state", "county", "year"], keep="first", maintain_order=True
    )
    reported_units = pl.concat([reported_counties, reported_regions])

    def average_coverage(first_year, last_year):
        in_years = reported_units.filter(
            pl.col("year").is_between(first_year, last_year)
        )
        n_regions = in_years.filter(~pl.col("county_reported")).height
        print(
            f"  school years {first_year}-{last_year}: "
            f"{in_years.height} reporting units "
            f"({in_years.height - n_regions} county-years, "
            f"{n_regions} region-or-state-years)"
        )
        return in_years["estimated_mmr_cov"].mean()

    avg_2011 = average_coverage(2010, 2010)
    avg_2011_15 = average_coverage(2010, 2014)
    avg_2011_16 = average_coverage(2010, 2015)
    avg_2025 = average_coverage(2024, 2024)
    avg_2026 = average_coverage(2025, 2025)
    avg_2021_26 = average_coverage(2020, 2025)

    avg_roc = vaccination_data_roc["slope"].mean()

    print(f"Average MMR coverage in 2011 (reporting units): {avg_2011}%")
    print(f"Average MMR coverage in 2025 (reporting units): {avg_2025}%")
    print(f"Average MMR coverage in 2026 (reporting units): {avg_2026}%")
    print(
        f"Average annual rate of change since 2020 (across counties): {avg_roc} percentage points/year"
    )
    print(f"Average MMR coverage in 2011-15 (reporting units): {avg_2011_15}%")
    print(f"Average MMR coverage in 2011-16 (reporting units): {avg_2011_16}%")
    print(f"Average MMR coverage in 2021-26 (reporting units): {avg_2021_26}%")

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
        vaccination_data_roc["fips"],
        vaccination_data_roc["slope"],
        fig,
        ax[1, 1],  # type: ignore
        cbar_label="Average annual MMR vaccine coverage change\nsince 2020 (percentage points)",
        cbar_type="discrete",
    )

    vaccination_data = vaccination_data.sort(
        ["fips", "year"], maintain_order=True
    ).with_columns(
        year_over_year_change=pl.col("estimated_mmr_cov")
        - pl.col("estimated_mmr_cov").shift(1).over("fips")
    )

    vaccination_data.write_csv(
        os.path.join(data_dir, "year_over_year_change_vaccination_data.csv")
    )

    # ------------------------------------------------------------------
    # Projections: from the rows before the state imputation
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("GENERATING VACCINE COVERAGE PROJECTIONS")
    print("=" * 80)

    projections = project_coverage_scenarios(
        vaccination_data_before_state_imputation,
        vaccination_data_roc,
        reported_rows,
        state_series,
        universe,
        latest_year=2026,
        end_year=2030,
    )

    # Save combined file
    projections.write_csv(
        os.path.join(
            data_dir, "vaccine_coverage_projections_all_scenarios.csv"
        )
    )

    # Save individual scenario files
    for scenario in projections["scenario"].unique():
        scenario_df = projections.filter(pl.col("scenario") == scenario)
        brown_in_proj = scenario_df.filter(pl.col("fips") == "31017")
        print(
            f"Scenario {scenario}: Brown County rows = {brown_in_proj.height}"
        )
        scenario_df.write_csv(
            os.path.join(
                data_dir, f"vaccine_coverage_projections_{scenario}.csv"
            )
        )
        print(f"Saved: vaccine_coverage_projections_{scenario}.csv")

    print("\nProjection files saved successfully!")
    print("=" * 80 + "\n")

    plt.tight_layout(rect=(0, 0, 1, 1))
    fig.savefig(
        os.path.join(fig_dir, "figure_measles_burden_overview.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(fig_dir, "figure_measles_burden_overview.pdf"),
        dpi=300,
        bbox_inches="tight",
    )

    # ------------------------------------------------------------------
    # R01-only standalone figures. Built on their own figure/axes, saved
    # after the main figure above, so they don't affect it.
    # ------------------------------------------------------------------

    # --- case map, with as much case data placed as possible ---
    addfips = AddFIPS()
    case_map_df = df.with_columns(
        fips=pl.when(pl.col("fips").is_null())
        .then(
            pl.struct(["county", "state"]).map_elements(
                lambda row: addfips.get_county_fips(
                    row["county"], row["state"]
                ),
                return_dtype=pl.String,
            )
        )
        .otherwise(pl.col("fips"))
    ).with_columns(
        # a few remaining region-level/state-level rows that addfips can't
        # resolve by name; R01-map-only placements (see plan)
        fips=pl.when(
            (pl.col("state") == "Iowa") & (pl.col("county") == "Eastern")
        )
        .then(pl.lit("19061"))
        .when((pl.col("state") == "Iowa") & (pl.col("county") == "Western"))
        .then(pl.lit("19193"))
        .when((pl.col("state") == "Iowa") & (pl.col("county") == "Central"))
        .then(pl.lit("19179"))
        .when(
            (pl.col("state") == "Alaska") & (pl.col("county") == "Gulf Coast")
        )
        .then(pl.lit("02020"))
        .when((pl.col("state") == "Delaware") & (pl.col("county") == "State"))
        .then(pl.lit("10001"))
        .otherwise(pl.col("fips"))
    )

    case_map_totals = case_map_df.group_by("fips").agg(cases=pl.sum("cases"))
    still_unplaced = case_map_df.filter(pl.col("fips").is_null())
    print(
        f"R01 case map: still-unplaced rows = {still_unplaced.height}, "
        f"cases = {still_unplaced['cases'].sum() if still_unplaced.height else 0}"
    )
    if still_unplaced.height:
        print(still_unplaced.select("state", "county", "year", "cases"))

    placed_totals = case_map_totals.filter(pl.col("fips").is_not_null())
    print(f"R01 case map: total cases = {placed_totals['cases'].sum()}")
    print(f"R01 case map: counties with >=1 case = {placed_totals.height}")
    print(
        f"R01 case map: county-year rows (state/county/year, pre-aggregation) "
        f"= {case_map_df.height}"
    )

    fig_r01_case, ax_r01_case = plt.subplots(figsize=(12, 9))
    map_fips_code_data(
        None,
        None,
        fig_r01_case,
        ax_r01_case,
        cbar_label=None,
        cbar_type=None,
        point_data=placed_totals[["fips", "cases"]].with_columns(
            cases=pl.when(pl.col("cases") < 5)
            .then(pl.lit(5))
            .otherwise(pl.col("cases"))
        ),
        # this map is its own figure rather than one panel of a 2x2 grid, so
        # the default anchor (tuned for that grid) sits well below/right of
        # the visible map; pull it back in
        legend_bbox_to_anchor=(0.865, 0.05),
    )
    fig_r01_case.savefig(
        os.path.join(fig_dir, "figure_r01_case_map.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig_r01_case)

    # --- MMR coverage map (R01-only, as of year 2024 (2024-2025 school
    # year): counties missing a 2024 row are filled with their latest
    # available year <= 2024 instead of being left grey) ---
    vaccination_data_latest_filled = (
        vaccination_data.filter(pl.col("year") <= 2024)
        .filter(pl.col("estimated_mmr_cov").is_not_null())
        .sort(["fips", "year"], descending=True, maintain_order=True)
        .group_by("fips", maintain_order=True)
        .agg(pl.first("estimated_mmr_cov").alias("estimated_mmr_cov"))
    )

    # --- R01-ONLY OVERRIDE (revert by deleting this block) ---
    # Montana's real county-level coverage only exists for 2019; every other
    # year is a statewide-imputed placeholder (see
    # vaccination_wrangling.impute_from_state_level_coverages), which the
    # "latest year <= 2024" rule above would otherwise prefer over the real,
    # county-varying 2019 numbers. Force Montana counties to their real 2019
    # values here instead.
    montana_2019 = vaccination_data.filter(
        (pl.col("state") == "Montana") & (pl.col("year") == 2019)
    ).select("fips", "estimated_mmr_cov")
    vaccination_data_latest_filled = (
        vaccination_data_latest_filled.join(
            montana_2019, on="fips", how="left", suffix="_montana_2019"
        )
        .with_columns(
            estimated_mmr_cov=pl.when(
                pl.col("estimated_mmr_cov_montana_2019").is_not_null()
            )
            .then(pl.col("estimated_mmr_cov_montana_2019"))
            .otherwise(pl.col("estimated_mmr_cov"))
        )
        .drop("estimated_mmr_cov_montana_2019")
    )
    # --- END R01-ONLY OVERRIDE ---

    fig_r01_cov, ax_r01_cov = plt.subplots(figsize=(12, 9))
    map_fips_code_data(
        vaccination_data_latest_filled["fips"].to_list(),
        vaccination_data_latest_filled["estimated_mmr_cov"].to_list(),
        fig_r01_cov,
        ax_r01_cov,
        cbar_label="MMR vaccine coverage (%)",
        cbar_type="bucketed_coverage",
        cbar_orientation="vertical",
    )
    fig_r01_cov.savefig(
        os.path.join(fig_dir, "figure_r01_mmr_coverage_map.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig_r01_cov)

    # --- nonmedical exemption rate map: the latest observed value of each
    # county (a county with no value of its own gets its state's value), not
    # the value brought forward to the reference year ---
    # the in-memory export column may not be Float64, so cast it explicitly
    nme_2026_map = nme_2026.filter(  # type: ignore
        pl.col("fips").str.len_chars() == 5
    ).with_columns(pl.col("nonmed_exempt_pct_observed").cast(pl.Float64))
    fig_r01_nme, ax_r01_nme = plt.subplots(figsize=(12, 9))
    map_fips_code_data(
        nme_2026_map["fips"],
        nme_2026_map["nonmed_exempt_pct_observed"],
        fig_r01_nme,
        ax_r01_nme,
        cbar_label="Nonmedical exemption rate (%)",
        cbar_type="exemptions",
        cbar_orientation="vertical",
    )
    fig_r01_nme.savefig(
        os.path.join(fig_dir, "figure_r01_nme_map.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig_r01_nme)
