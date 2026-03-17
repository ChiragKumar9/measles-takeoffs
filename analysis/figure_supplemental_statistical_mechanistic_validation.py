import matplotlib.lines as mlines
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
from scipy import stats
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


def setup_plot(nrows, ncols, figsize=None):
    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize=(
            figsize if figsize is not None else (10 * ncols, 7.5 * nrows)
        ),
    )
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
        return fig, ax
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


def plot_total_cases_over_time(
    df_observed, all_counties, samples, naive_samples, fig, ax
):
    years = [2021, 2022, 2023, 2024, 2025]

    offset = 0.3
    widths = 0.3

    for idx, year in enumerate(years):
        mask = df_observed["validation_year"] == year
        true = df_observed.filter(pl.col("validation_year") == year)[
            "cases"
        ].sum()
        mask = all_counties["validation_year"] == year
        samples_year = samples[:, mask.to_numpy()]
        samples_year = jnp.sum(samples_year, axis=1)
        naive_samples_year = naive_samples[:, mask.to_numpy()]
        naive_samples_year = jnp.sum(naive_samples_year, axis=1)
        # only violin the middle 95th percentile
        lower = jnp.percentile(samples_year, 2.5)
        upper = jnp.percentile(samples_year, 97.5)
        samples_year_vln = samples_year[
            jnp.logical_and(samples_year >= lower, samples_year <= upper)
        ]
        naive_samples_vln = naive_samples_year[
            jnp.logical_and(
                naive_samples_year >= jnp.percentile(naive_samples_year, 2.5),
                naive_samples_year <= jnp.percentile(naive_samples_year, 97.5),
            )
        ]
        # plot a violin of samples_year and a point for true
        vln1 = ax.violinplot(
            samples_year_vln,
            positions=[idx - offset],
            widths=widths,
            showextrema=False,
        )
        # set the color of the violin to be grey with some transparency
        for pc in vln1["bodies"]:
            pc.set_facecolor("green")
            pc.set_edgecolor("green")
            pc.set_alpha(0.3)

        # add an error bar showing the median and 25-75th percentile
        ax.errorbar(
            idx - offset,
            jnp.median(samples_year),
            yerr=np.array(
                [
                    [
                        jnp.median(samples_year)
                        - jnp.percentile(samples_year, 25)
                    ],
                    [
                        jnp.percentile(samples_year, 75)
                        - jnp.median(samples_year)
                    ],
                ]
            ),
            fmt="o",
            markersize=7,
            ecolor="green",
            color="green",
            # add bars at top and bottom of error bar
            capsize=5,
            capthick=2,
        )

        # plot a violin of samples_year and a point for true
        vln2 = ax.violinplot(
            naive_samples_vln,
            positions=[idx + offset],
            widths=widths,
            showextrema=False,
        )
        # set the color of the violin to be grey with some transparency
        for pc in vln2["bodies"]:
            pc.set_facecolor("red")
            pc.set_edgecolor("red")
            pc.set_alpha(0.3)

        # add an error bar showing the median and 25-75th percentile
        ax.errorbar(
            idx + offset,
            jnp.median(naive_samples_vln),
            yerr=np.array(
                [
                    [
                        jnp.median(naive_samples_vln)
                        - jnp.percentile(naive_samples_vln, 25)
                    ],
                    [
                        jnp.percentile(naive_samples_vln, 75)
                        - jnp.median(naive_samples_vln)
                    ],
                ]
            ),
            fmt="o",
            markersize=7,
            ecolor="red",
            color="red",
            # add bars at top and bottom of error bar
            capsize=5,
            capthick=2,
        )

        ax.scatter(
            idx,
            true,
            color="purple",
            marker="*",
            s=200,
            zorder=10,
            label="Observed" if idx == 0 else "",
        )

    ax.legend(
        [
            vln1["bodies"][0],  # type: ignore
            vln2["bodies"][0],  # type: ignore
            mlines.Line2D(
                [], [], color="purple", marker="*", linestyle="", markersize=15
            ),
        ],
        ["Statistical", "Mechanistic", "Observed"],
    )

    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years)
    ax.set_ylabel("Forecasted cases (N)")
    ax.set_yscale("log")
    ax.set_xlabel("Year")
    ax.set_yticks([1, 10, 100, 1000, 10000])
    ax.set_yticklabels(["1", "10", "100", "1,000", "10,000"])


def plot_risk_predictions_by_county(
    df, samples, naive_samples, background_samples, background_naive, fig, ax
):
    # box plots of percentiles by buckets
    df = df.with_columns(
        statistical=pl.Series(samples.__array__()),
        naive=pl.Series(naive_samples.__array__()),
        rank=pl.col("cases").rank(method="average").over("year")
        / pl.col("cases").count().over("year")
        * 100,
        bucket=pl.when(pl.col("cases") < 4)
        .then(1)
        .when(pl.col("cases") < 9)
        .then(2)
        .otherwise(3),
    )
    # df = df.with_columns(
    #     bucket = pl.when(pl.col('rank') < 33.3).then(0).when(pl.col("rank") < 66.6).then(1).otherwise(2)
    # )

    bin_width = 0.3
    offset = 0.2

    bp = ax.boxplot(
        background_samples,
        positions=[0 - offset],
        widths=bin_width,
        whis=(2.5, 97.5),
        patch_artist=True,
        showfliers=True,
        showmeans=True,
        meanprops=dict(
            marker="*",
            markerfacecolor="green",
            markeredgecolor="green",
            markersize=8,
        ),
        flierprops=dict(
            marker="o",
            markerfacecolor="none",
            markeredgecolor="green",
            markersize=4,
        ),
        medianprops=dict(color="black", linewidth=3),
        whiskerprops=dict(linestyle="--", color="green"),
        capprops=dict(color="green"),
        notch=False,
    )

    # Color the box
    for patch in bp["boxes"]:
        patch.set_facecolor("white")  # or 'white' for white fill
        patch.set_edgecolor("green")
        patch.set_linewidth(2)
        patch.set_label("Statistical")

    bp = ax.boxplot(
        background_naive,
        positions=[0 + offset],
        widths=bin_width,
        patch_artist=True,
        showfliers=True,
        showmeans=True,
        meanprops=dict(
            marker="*",
            markerfacecolor="red",
            markeredgecolor="red",
            markersize=8,
        ),
        flierprops=dict(
            marker="o",
            markerfacecolor="none",
            markeredgecolor="red",
            markersize=4,
        ),
        medianprops=dict(color="black", linewidth=3),
        whiskerprops=dict(linestyle="--", color="red"),
        capprops=dict(color="red"),
        notch=False,
    )

    # Color the box
    for patch in bp["boxes"]:
        patch.set_facecolor("white")  # or 'white' for white fill
        patch.set_edgecolor("red")
        patch.set_linewidth(2)
        patch.set_label("Mechanistic")

    # box plot of statistical and naive predictions by bucket
    for bin in df["bucket"].unique():
        bin_data = df.filter(pl.col("bucket") == bin)["statistical"].to_list()
        bp = ax.boxplot(
            bin_data,
            positions=[bin - offset],
            widths=bin_width,
            whis=(2.5, 97.5),
            patch_artist=True,
            showfliers=True,
            showmeans=True,
            meanprops=dict(
                marker="*",
                markerfacecolor="green",
                markeredgecolor="green",
                markersize=8,
            ),
            flierprops=dict(
                marker="o",
                markerfacecolor="none",
                markeredgecolor="green",
                markersize=4,
            ),
            medianprops=dict(color="black", linewidth=3),
            whiskerprops=dict(linestyle="--", color="green"),
            capprops=dict(color="green"),
            notch=False,
        )

        # Color the box
        for patch in bp["boxes"]:
            patch.set_facecolor("white")  # or 'white' for white fill
            patch.set_edgecolor("green")
            patch.set_linewidth(2)

        bin_data = df.filter(pl.col("bucket") == bin)["naive"].to_list()
        bp = ax.boxplot(
            bin_data,
            positions=[bin + offset],
            widths=bin_width,
            patch_artist=True,
            showfliers=True,
            showmeans=True,
            meanprops=dict(
                marker="*",
                markerfacecolor="red",
                markeredgecolor="red",
                markersize=8,
            ),
            flierprops=dict(
                marker="o",
                markerfacecolor="none",
                markeredgecolor="red",
                markersize=4,
            ),
            medianprops=dict(color="black", linewidth=3),
            whiskerprops=dict(linestyle="--", color="red"),
            capprops=dict(color="red"),
            notch=False,
        )

        # Color the box
        for patch in bp["boxes"]:
            patch.set_facecolor("white")  # or 'white' for white fill
            patch.set_edgecolor("red")
            patch.set_linewidth(2)

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["0", "1-4", "5-9", "≥10"])
    ax.set_xlabel("Observed cases (N)")
    ax.set_ylabel("Forecasted outbreak size percentile")

    # add a legend
    ax.legend()

    ax.set_ylim([0, 100])

    return df


# def plot_metric_over_values(df_year,
#                             samples,
#                             naive_samples,
#                             with_importation, x_label, fig, ax):
#     if with_importation is None:
#         outcome = "cases_outbreak"
#     else:
#         outcome = "cases"
#     cases_sorted = df_year[outcome].sort().unique().to_list()
#     n_cases = len(cases_sorted)
#     years = df_year['year'].to_jax()
#     n_years = len(jnp.unique(years))
#     metric_median = np.empty((n_cases, n_years)) * np.nan
#     metric_lower = np.empty((n_cases, n_years)) * np.nan
#     metric_upper = np.empty((n_cases, n_years)) * np.nan
#     metric_mean = np.empty((n_cases, n_years)) * np.nan

#     metric_median_naive = np.empty((n_cases, n_years)) * np.nan
#     metric_lower_naive = np.empty((n_cases, n_years)) * np.nan
#     metric_upper_naive = np.empty((n_cases, n_years)) * np.nan
#     metric_mean_naive = np.empty((n_cases, n_years)) * np.nan

#     if with_importation is not None:
#         samples *= with_importation
#         naive_samples *= with_importation

#     for year_idx, year in tqdm(enumerate(jnp.unique(years))):
#         # get the samples for this year and the true values for this year
#         mask = years == year
#         true = df_year.filter(pl.col("year") == year)[outcome].to_jax()
#         samples_year = samples[:, mask]
#         naive_samples_year = naive_samples[:, mask]

#         for idx, val in enumerate(cases_sorted):
#             if val < jnp.sort(true)[-2]:
#                 baseline = (true > val).sum() / len(true)
#                 metric_values = np.array(
#                     [
#                         (average_precision_score(
#                             true > val,
#                             c
#                         ) - baseline) / (1 - baseline) * 50 + 50
#                         for c in tqdm(jnp.sort(samples_year, axis = 0))
#                     ]
#                 )

#                 print(f"Year {year}, cases {val}: metric median={np.nanmedian(metric_values)}, mean={np.nanmean(metric_values)}, lower={np.nanpercentile(metric_values, 2.5)}, upper={np.nanpercentile(metric_values, 97.5)}")
#                 metric_median[idx, year_idx] = float(np.nanmedian(metric_values))
#                 metric_lower[idx, year_idx] = float(np.nanpercentile(metric_values, 2.5))
#                 metric_upper[idx, year_idx] = float(np.nanpercentile(metric_values, 97.5))
#                 metric_mean[idx, year_idx] = float(np.nanmean(metric_values))

#                 metric_values_naive = np.array(
#                     [
#                         (average_precision_score(
#                             true > val,
#                             c
#                         ) - baseline) / (1 - baseline) * 50 + 50
#                         for c in tqdm(jnp.sort(naive_samples_year, axis = 0))
#                     ]
#                 )

#                 print(f"Year {year}, cases {val}: metric median={np.nanmedian(metric_values_naive)}, mean={np.nanmean(metric_values)}, lower={np.nanpercentile(metric_values, 2.5)}, upper={np.nanpercentile(metric_values, 97.5)}")
#                 metric_median_naive[idx, year_idx] = float(np.nanmedian(metric_values_naive))
#                 metric_lower_naive[idx, year_idx] = float(np.nanpercentile(metric_values_naive, 2.5))
#                 metric_upper_naive[idx, year_idx] = float(np.nanpercentile(metric_values_naive, 97.5))
#                 metric_mean_naive[idx, year_idx] = float(np.nanmean(metric_values_naive))

#     metric_median = np.nanmean(metric_median, axis = 1)
#     metric_lower = np.nanmean(metric_lower, axis = 1)
#     metric_upper = np.nanmean(metric_upper, axis = 1)
#     metric_mean = np.nanmean(metric_mean, axis = 1)

#     metric_median_naive = np.nanmean(metric_median_naive, axis = 1)
#     metric_lower_naive = np.nanmean(metric_lower_naive, axis = 1)
#     metric_upper_naive = np.nanmean(metric_upper_naive, axis = 1)
#     metric_mean_naive = np.nanmean(metric_mean_naive, axis = 1)

#     ax.errorbar(
#         cases_sorted,
#         metric_median,
#         yerr=np.array(
#             [
#                 metric_median - metric_lower,
#                 metric_upper - metric_median,
#             ]
#         ),
#         capsize=0,
#         fmt="o",
#         markersize=7,
#         ecolor="green",
#         color="green",
#         label = 'Statistical'
#     )

#     ax.errorbar(
#         cases_sorted,
#         metric_median_naive,
#         yerr=np.array(
#             [
#                 metric_median_naive - metric_lower_naive,
#                 metric_upper_naive - metric_median_naive,
#             ]
#         ),
#         capsize=0,
#         fmt="o",
#         markersize=7,
#         ecolor="purple",
#         color="purple",
#         label = 'Mechanistic'
#     )
#     # ax.scatter(
#     #     cases_sorted,
#     #     metric_mean,
#     #     marker="*",
#     #     s = 30,
#     #     color = 'grey',
#     # )

#     # # add a horizontal line at the 80th percentile to mark the top 20% of risk
#     # ax.axvline(
#     #     np.percentile(df_year[outcome].to_numpy(), 80),
#     #     color="red",
#     #     linestyle="--",
#     #     zorder=0
#     # )

#     ax.axhline(
#         50,
#         color = 'grey',
#         linestyle = '--',
#         zorder = 0
#     )

#     ax.legend()

#     ax.set_xlabel(x_label)
#     ax.set_ylabel("Normalized average precision")
#     # ax.set_ylim(0, 100)
#     if with_importation is None:
#         ax.set_xscale("symlog")
#     else:
#         ax.set_xscale("log")
#         ax.set_xlim(0.9)
#         ax.xaxis.set_major_formatter(ScalarFormatter())


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
        norm = mcolors.LogNorm(vmin=counties_with_data["value"].min(), vmax=1)
        breakpoints = [-1, 0.2, 0.4, 0.6, 0.8, 1.01]
        cmap = plt.get_cmap("Blues")
        # cmap = mcolors.ListedColormap(
        #     [cmap(i) for i in np.linspace(0.3, 1.0, 5)]
        # )

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
            cbar.set_ticks([1e-4, 0.001, 0.01, 0.1, 1.0])
            cbar.set_ticklabels(["0.0001", "0.001", "0.01", "0.1", "1.0"])
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

    # but we also need to expand df to include all counties
    validation_years = [2021, 2022, 2023, 2024, 2025]
    all_counties = []
    for year in validation_years:
        temp = pl.read_csv(
            os.path.join(output_dir, f"projection_dataset_{year}.csv"),
            schema_overrides={"fips": pl.String},
        )
        temp = temp.with_columns(validation_year=pl.lit(year))
        all_counties.append(temp)
    all_counties = pl.concat(all_counties)
    # the only things we need from all_counties are the county, state, year, and cases
    all_counties = all_counties.select(
        [
            "county",
            "state",
            "year",
            "validation_year",
            "cases_outbreak",
            "type",
        ]
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
    # each place can only occur once per year
    # we want to drop duplicates, deferring to projection rather than train
    # type values
    # to do this, sort by type
    import_df = import_df.sort("type").unique(
        subset=["state", "county", "year"], keep="last"
    )

    # join up with df
    all_counties = all_counties.join(
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
        np.vstack(all_counties["importation_probability_samples"].to_numpy())  # type: ignore
    ).T

    samples = []
    for idx, year in enumerate(
        [
            [2019, 2020],
            [2019, 2020, 2021],
            [2019, 2020, 2021, 2022],
            [2019, 2020, 2021, 2022, 2023],
            [2019, 2020, 2021, 2022, 2023, 2024],
        ]
    ):
        with open(
            os.path.join(
                output_dir,
                f"samples_partial_pooled_poisson_validation_testing_{year}_pool=True_CAR=True_All covariates.pkl",
            ),
            "rb",
        ) as f:
            m = pickle.load(
                f,
            )
        samples.append(m)

    samples = jnp.concat(samples, axis=1)

    naive_samples = []
    for idx, year in enumerate(
        [
            [2019, 2020],
            [2019, 2020, 2021],
            [2019, 2020, 2021, 2022],
            [2019, 2020, 2021, 2022, 2023],
            [2019, 2020, 2021, 2022, 2023, 2024],
        ]
    ):
        with open(
            os.path.join(
                output_dir,
                f"predictive_final_outbreak_size_predictions_{year}.pkl",
            ),
            "rb",
        ) as f:
            m = pickle.load(
                f,
            )
        naive_samples.append(m["cases"])

    naive_samples = jnp.concat(naive_samples, axis=1)

    # need to subset to the places where we have observed takeoffs so we can
    # truly validate the two models combined
    mask = (all_counties["type"] == "train").to_numpy()
    samples_observed = samples[:, mask]
    naive_samples_observed = naive_samples[:, mask]
    importation_samples_observed = importation_samples[:, mask]
    df_train_observed = all_counties.filter(pl.col("type") == "train")

    # let's get the actual case counts from the train data
    analysis = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        schema_overrides={"fips": pl.String},
    )

    df_train_observed = df_train_observed.join(
        analysis[["year", "state", "county", "cases"]],
        on=["year", "state", "county"],
        maintain_order="left",
    )

    # incorporate improper CAR prior as scale term
    observed = np.array([])
    predicted = np.array([])
    predicted_full = np.array([])

    for year in validation_years:
        mask = df_train_observed["validation_year"] == year
        samples_year = samples_observed[:, mask.to_numpy()]
        importation_samples_year = importation_samples_observed[
            :, mask.to_numpy()
        ]
        # multiply
        samples_year = samples_year * importation_samples_year
        # sum cases
        true = df_train_observed.filter(pl.col("validation_year") == year)[
            "cases"
        ].sum()
        # sum samples
        samples_year = jnp.sum(samples_year, axis=1)
        # print out true, 2.5th 25th, 50th, 75th, and 97.5th percentiles of samples_year
        print(
            f"Year {year}: true={true}, 2.5th={jnp.percentile(samples_year, 2.5)}, 25th={jnp.percentile(samples_year, 25)}, 50th={jnp.percentile(samples_year, 50)}, 75th={jnp.percentile(samples_year, 75)}, 97.5th={jnp.percentile(samples_year, 97.5)}"
        )

        observed = np.append(observed, true)
        predicted = np.append(predicted, jnp.median(samples_year))

        # print total cases predicted in that year
        mask = all_counties["validation_year"] == year
        samples_year = samples[:, mask.to_numpy()]
        importation_samples_year = importation_samples[:, mask.to_numpy()]
        samples_year = samples_year * importation_samples_year
        samples_year = jnp.sum(samples_year, axis=1)
        print(
            f"Year {year} total predicted cases: 2.5th={jnp.percentile(samples_year, 2.5)}, 25th={jnp.percentile(samples_year, 25)}, 50th={jnp.percentile(samples_year, 50)}, 75th={jnp.percentile(samples_year, 75)}, 97.5th={jnp.percentile(samples_year, 97.5)}"
        )
        predicted_full = np.append(predicted_full, jnp.median(samples_year))

        # how many cases does the naive model predict
        print(
            f"Year {year} total predicted cases (naive): 2.5th={jnp.percentile(jnp.sum(importation_samples_year * naive_samples[:, mask.to_numpy()], axis=1), 2.5)}, 25th={jnp.percentile(jnp.sum(importation_samples_year * naive_samples[:, mask.to_numpy()], axis=1), 25)}, 50th={jnp.percentile(jnp.sum(importation_samples_year * naive_samples[:, mask.to_numpy()], axis=1), 50)}, 75th={jnp.percentile(jnp.sum(importation_samples_year * naive_samples[:, mask.to_numpy()], axis=1), 75)}, 97.5th={jnp.percentile(jnp.sum(importation_samples_year * naive_samples[:, mask.to_numpy()], axis=1), 97.5)}"
        )

    # fit a scale factor to adjust the predicted values to match the observed
    # values
    scale = predicted_full / observed
    scale = np.mean(scale)

    samples_unconditioned = samples * importation_samples
    samples_observed_unconditioned = (
        samples_observed * importation_samples_observed
    )
    samples_scaled = samples_unconditioned / scale
    samples_observed_scaled = samples_observed_unconditioned / scale

    naive_samples_unconditioned = naive_samples * importation_samples
    naive_samples_observed_unconditioned = (
        naive_samples_observed * importation_samples_observed
    )

    # once again print out the total predicted values after rescaling
    for year in validation_years:
        mask = all_counties["validation_year"] == year
        samples_year = samples_scaled[:, mask.to_numpy()]
        # sum samples
        samples_year = jnp.sum(samples_year, axis=1)
        # print out true, 2.5th 25th, 50th, 75th, and 97.5th percentiles of samples_year
        print(
            f"Year {year}: 2.5th={jnp.percentile(samples_year, 2.5)}, 25th={jnp.percentile(samples_year, 25)}, 50th={jnp.percentile(samples_year, 50)}, 75th={jnp.percentile(samples_year, 75)}, 97.5th={jnp.percentile(samples_year, 97.5)}"
        )

    plot_total_cases_over_time(
        df_train_observed,
        all_counties,
        samples_scaled,
        naive_samples_unconditioned,
        fig,
        ax[0, 0],  # type: ignore
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

    # different ways to rank -- we can rank first, then median or mean
    # of these, mean first looks better
    # we can rank median, we can rank mean

    # let's rank first and then mean

    samples_2021 = samples_scaled[
        :, all_counties["validation_year"].to_jax() == 2021
    ]
    samples_2022 = samples_scaled[
        :, all_counties["validation_year"].to_jax() == 2022
    ]
    samples_2023 = samples_scaled[
        :, all_counties["validation_year"].to_jax() == 2023
    ]
    samples_2024 = samples_scaled[
        :, all_counties["validation_year"].to_jax() == 2024
    ]
    samples_2025 = samples_scaled[
        :, all_counties["validation_year"].to_jax() == 2025
    ]

    ranking_method = "average"
    ranked_2021 = stats.rankdata(
        jnp.median(samples_2021, axis=0),
        # samples_2021,
        # axis = 1,
        method=ranking_method,
    )
    ranked_2022 = stats.rankdata(
        jnp.median(samples_2022, axis=0),
        # samples_2022,
        # axis = 1,
        method=ranking_method,
    )
    ranked_2023 = stats.rankdata(
        jnp.median(samples_2023, axis=0),
        # samples_2023,
        # axis = 1,
        method=ranking_method,
    )
    ranked_2024 = stats.rankdata(
        jnp.median(samples_2024, axis=0),
        # samples_2024,
        # axis = 1,
        method=ranking_method,
    )
    ranked_2025 = stats.rankdata(
        jnp.median(samples_2025, axis=0),
        # samples_2025,
        # axis = 1,
        method=ranking_method,
    )

    # summarized_ranked_2021 = jnp.median(
    #     ranked_2021, axis = 0
    # )
    # summarized_ranked_2022 = jnp.median(
    #     ranked_2022, axis = 0
    # )
    # summarized_ranked_2023 = jnp.median(
    #     ranked_2023, axis = 0
    # )
    # summarized_ranked_2024 = jnp.median(
    #     ranked_2024, axis = 0
    # )
    # summarized_ranked_2025 = jnp.median(
    #     ranked_2025, axis = 0
    # )

    # turn into percentiles
    summarized_ranked_2021 = ranked_2021 / ranked_2021.max() * 100
    summarized_ranked_2022 = ranked_2022 / ranked_2022.max() * 100
    summarized_ranked_2023 = ranked_2023 / ranked_2023.max() * 100
    summarized_ranked_2024 = ranked_2024 / ranked_2024.max() * 100
    summarized_ranked_2025 = ranked_2025 / ranked_2025.max() * 100

    # summarized_ranked_2021 = summarized_ranked_2021 / ranked_2021.shape[1] * 100
    # summarized_ranked_2022 = summarized_ranked_2022 / ranked_2022.shape[1] * 100
    # summarized_ranked_2023 = summarized_ranked_2023 / ranked_2023.shape[1] * 100
    # summarized_ranked_2024 = summarized_ranked_2024 / ranked_2024.shape[1] * 100
    # summarized_ranked_2025 = summarized_ranked_2025 / ranked_2025.shape[1] * 100

    summarized_observed_ranked = []
    summarized_unobserved_ranked = []
    for t, y in zip(
        [
            summarized_ranked_2021,
            summarized_ranked_2022,
            summarized_ranked_2023,
            summarized_ranked_2024,
            summarized_ranked_2025,
        ],
        [2021, 2022, 2023, 2024, 2025],
    ):
        observed_ranked = t[
            all_counties.filter(pl.col("validation_year") == y)[
                "year"
            ].to_jax()
            == y
        ]
        unobserved_ranked = t[
            all_counties.filter(pl.col("validation_year") == y)[
                "year"
            ].to_jax()
            != y
        ]
        summarized_observed_ranked.append(observed_ranked)
        summarized_unobserved_ranked.append(unobserved_ranked)

    # same deal for naive samples
    naive_samples_2021 = naive_samples_unconditioned[
        :, all_counties["validation_year"].to_jax() == 2021
    ]
    naive_samples_2022 = naive_samples_unconditioned[
        :, all_counties["validation_year"].to_jax() == 2022
    ]
    naive_samples_2023 = naive_samples_unconditioned[
        :, all_counties["validation_year"].to_jax() == 2023
    ]
    naive_samples_2024 = naive_samples_unconditioned[
        :, all_counties["validation_year"].to_jax() == 2024
    ]
    naive_samples_2025 = naive_samples_unconditioned[
        :, all_counties["validation_year"].to_jax() == 2025
    ]

    ranking_method = "average"
    naive_ranked_2021 = stats.rankdata(
        jnp.median(naive_samples_2021, axis=0),
        # naive_samples_2021,
        # axis = 1,
        method=ranking_method,
    )
    naive_ranked_2022 = stats.rankdata(
        jnp.median(naive_samples_2022, axis=0),
        # naive_samples_2022,
        # axis = 1,
        method=ranking_method,
    )
    naive_ranked_2023 = stats.rankdata(
        jnp.median(naive_samples_2023, axis=0),
        # naive_samples_2023,
        # axis = 1,
        method=ranking_method,
    )
    naive_ranked_2024 = stats.rankdata(
        jnp.median(naive_samples_2024, axis=0),
        # naive_samples_2024,
        # axis = 1,
        method=ranking_method,
    )
    naive_ranked_2025 = stats.rankdata(
        jnp.median(naive_samples_2025, axis=0),
        # naive_samples_2025,
        # axis = 1,
        method=ranking_method,
    )

    # summarized_naive_ranked_2021 = jnp.median(
    #     naive_ranked_2021, axis = 0
    # )
    # summarized_naive_ranked_2022 = jnp.median(
    #     naive_ranked_2022, axis = 0
    # )
    # summarized_naive_ranked_2023 = jnp.median(
    #     naive_ranked_2023, axis = 0
    # )
    # summarized_naive_ranked_2024 = jnp.median(
    #     naive_ranked_2024, axis = 0
    # )
    # summarized_naive_ranked_2025 = jnp.median(
    #     naive_ranked_2025, axis = 0
    # )

    # turn into percentiles
    summarized_naive_ranked_2021 = (
        naive_ranked_2021 / naive_ranked_2021.max() * 100
    )
    summarized_naive_ranked_2022 = (
        naive_ranked_2022 / naive_ranked_2022.max() * 100
    )
    summarized_naive_ranked_2023 = (
        naive_ranked_2023 / naive_ranked_2023.max() * 100
    )
    summarized_naive_ranked_2024 = (
        naive_ranked_2024 / naive_ranked_2024.max() * 100
    )
    summarized_naive_ranked_2025 = (
        naive_ranked_2025 / naive_ranked_2025.max() * 100
    )

    # summarized_naive_ranked_2021 = summarized_naive_ranked_2021 / naive_ranked_2021.shape[1] * 100
    # summarized_naive_ranked_2022 = summarized_naive_ranked_2022 / naive_ranked_2022.shape[1] * 100
    # summarized_naive_ranked_2023 = summarized_naive_ranked_2023 / naive_ranked_2023.shape[1] * 100
    # summarized_naive_ranked_2024 = summarized_naive_ranked_2024 / naive_ranked_2024.shape[1] * 100
    # summarized_naive_ranked_2025 = summarized_naive_ranked_2025 / naive_ranked_2025.shape[1] * 100

    summarized_naive_observed_ranked = []
    summarized_naive_unobserved_ranked = []
    for t, y in zip(
        [
            summarized_naive_ranked_2021,
            summarized_naive_ranked_2022,
            summarized_naive_ranked_2023,
            summarized_naive_ranked_2024,
            summarized_naive_ranked_2025,
        ],
        [2021, 2022, 2023, 2024, 2025],
    ):
        observed_ranked = t[
            all_counties.filter(pl.col("validation_year") == y)[
                "year"
            ].to_jax()
            == y
        ]
        unobserved_ranked = t[
            all_counties.filter(pl.col("validation_year") == y)[
                "year"
            ].to_jax()
            != y
        ]
        summarized_naive_observed_ranked.append(observed_ranked)
        summarized_naive_unobserved_ranked.append(unobserved_ranked)

    plot_risk_predictions_by_county(
        df_train_observed,
        jnp.concat(summarized_observed_ranked),
        jnp.concat(summarized_naive_observed_ranked),
        jnp.concat(summarized_unobserved_ranked),
        jnp.concat(summarized_naive_unobserved_ranked),
        fig,
        ax[0, 1],  # type: ignore
    )

    # plot_metric_over_values(
    #     df_train_observed,
    #     samples_observed_scaled,
    #     naive_samples_observed_unconditioned,
    #     with_importation=importation_samples_observed,
    #     x_label="Case classification threshold (N)",
    #     fig=fig,
    #     ax=ax[0, 1],  # type: ignore
    # )

    df = pl.read_csv(
        os.path.join(output_dir, "assembled_projection_results.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
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
        df["importation_probability_median"],
        fig,
        ax[1, 0],  # type: ignore
        cbar_label="Probability of ≥1 international importation",
        cbar_type="probability",
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
        df["median_unconditioned_risk_rank"],
        fig,
        ax[1, 1],  # type: ignore
        cbar_label="Outbreak size percentile",
        cbar_type="percentile_risk",
        point_data=None,
    )

    fig.tight_layout()

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
