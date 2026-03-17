import os

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import yaml
from matplotlib import rc
from scipy.stats import linregress
from shapely.geometry import MultiPolygon, Polygon

from outbreak_size import vaccination_wrangling

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
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
    if cbar_type == "continuous_cases":
        norm = mcolors.Normalize(vmin=0, vmax=500)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "custom", ["#fee5d9", "#a50f15"], N=256
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
        if cbar_type == "binary":
            cbar.set_ticks([0.5, 1.5])
            cbar.set_ticklabels(["<10", r"$\geq$10"])
            cbar.ax.tick_params(labelsize=24, rotation=0)
        elif cbar_type == "discrete":
            cbar.set_ticks([-9, -1.2, -0.2, 0.2, 0.8, 1.7, 20])
            cbar.set_ticklabels(
                [
                    "≤-2%",  # less than equal
                    "-2% to -1%",
                    "-1% to -0.2%",
                    "-0.2% to 0.2%",
                    "0.2% to 1%",
                    "1% to 2%",
                    ">2%",
                ]
            )
            cbar.ax.tick_params(labelsize=24, rotation=-60, length=0)
        elif cbar_type == "bucketed_coverage":
            cbar.set_ticks([35, 75, 85, 92.5, 97.5])
            cbar.set_ticklabels(
                [
                    "≤70%",
                    "70%-80%",
                    "80%-90%",
                    "90%-95%",
                    ">95%",
                ]
            )
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


def plot_cases_over_time(df, fig, ax):
    df_overall = df.group_by("year").agg(pl.sum("cases"))
    df_overall = df_overall.sort("year")
    ax.plot(
        df_overall["year"],
        df_overall["cases"],
        marker="^",
        markersize=10,
        color="black",
        linestyle="--",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Measles cases (N)")
    ax.set_xticks([2013, 2016, 2019, 2022, 2025])
    ax.set_ylim(0)

    counts, bins = np.histogram(df["year"], bins=np.arange(2011, 2027) - 0.5)
    ax2 = ax.twinx()
    ax2.bar(
        bins[:-1],
        counts,
        width=np.diff(bins),
        alpha=0.2,
        align="edge",
        edgecolor="grey",
    )
    ax2.set_ylabel("County-year outbreaks (N)", rotation=-90, labelpad=30)
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

    return fig, ax


def exclude_county_state_year(df, county, state, year):
    return df.filter(
        ~(
            (pl.col("county") == county)
            & (pl.col("state") == state)
            & (pl.col("year") == year)
        )
    )


def project_coverage_scenarios(
    vaccination_data, vaccination_data_roc, latest_year=2025, end_year=2030
):
    """
    Project vaccine coverage, nonmedical exemption, and medical exemption rates
    under 5 scenarios.

    Parameters:
    -----------
    vaccination_data : pl.DataFrame
        Full vaccination dataset with year, fips, estimated_mmr_cov,
        nonmed_exempt_pct, med_exempt_pct, etc.
    vaccination_data_roc : pl.DataFrame
        DataFrame with fips and slope (from linear regression on coverage)
    latest_year : int
        Most recent year to use as baseline (default 2025)
    end_year : int
        Final year to project to (default 2030)

    Returns:
    --------
    pl.DataFrame with projections in long format
    """

    # 1. Get baseline data (most recent per county up to latest_year)
    baseline = (
        vaccination_data.filter(pl.col("year") <= latest_year)
        .filter(pl.col("fips").is_not_null())
        .filter(pl.col("fips").str.len_chars() > 2)  # County-level only
        .sort(["fips", "year"], descending=True)
        .group_by("fips", maintain_order=True)
        .agg(
            pl.first("year").alias("baseline_year"),
            pl.first("estimated_mmr_cov").alias("baseline_coverage"),
            pl.first("nonmed_exempt_pct").alias("baseline_nonmed_exempt"),
            pl.first("med_exempt_pct").alias("baseline_med_exempt"),
            pl.first("state"),
            pl.first("county"),
        )
    )

    # 2. Determine which counties have sufficient data (>=2 years since 2020)
    county_data_counts = (
        vaccination_data.filter(pl.col("year") >= 2020)
        .group_by("fips")
        .agg(pl.count("year").alias("n_years"))
    )

    # 3. Calculate slopes for nonmed and med exemptions via linear regression
    def calc_slopes(df, outcome_col, alias):
        return (
            df.filter(pl.col("year") >= 2020)
            .filter(pl.col(outcome_col).is_not_null())
            .group_by("fips")
            .map_groups(
                lambda g: pl.DataFrame(
                    {
                        "fips": [g["fips"][0]],
                        alias: [
                            linregress(
                                g["year"].cast(pl.Float64).to_numpy(),
                                g[outcome_col].cast(pl.Float64).to_numpy(),
                            ).slope
                            if g.height >= 2
                            else float("nan")
                        ],
                    }
                )
            )
            .filter(~pl.col(alias).is_nan())
        )

    nonmed_roc = calc_slopes(
        vaccination_data, "nonmed_exempt_pct", "slope_nonmed"
    )
    med_roc = calc_slopes(vaccination_data, "med_exempt_pct", "slope_med")

    # 4. Calculate national average slopes
    valid_slopes = vaccination_data_roc.filter(
        ~pl.col("slope").is_nan() & ~pl.col("slope").is_null()
    )
    national_avg_slope = valid_slopes["slope"].mean()

    valid_nonmed = nonmed_roc.filter(
        ~pl.col("slope_nonmed").is_nan() & ~pl.col("slope_nonmed").is_null()
    )
    national_avg_slope_nonmed = valid_nonmed["slope_nonmed"].mean()

    valid_med = med_roc.filter(
        ~pl.col("slope_med").is_nan() & ~pl.col("slope_med").is_null()
    )
    national_avg_slope_med = valid_med["slope_med"].mean()

    print(f"National average slope (coverage):     {national_avg_slope:.4f}")
    print(
        f"National average slope (nonmed exempt): {national_avg_slope_nonmed:.4f}"
    )
    print(
        f"National average slope (med exempt):    {national_avg_slope_med:.4f}"
    )
    print(
        f"Number of counties with coverage slope data: {valid_slopes.height}"
    )
    print(
        f"Number of counties with nonmed slope data:   {valid_nonmed.height}"
    )
    print(f"Number of counties with med slope data:      {valid_med.height}")

    # 5. Join all slopes onto baseline
    projection_base = (
        baseline.join(county_data_counts, on="fips", how="left")
        .join(
            vaccination_data_roc.select(["fips", "slope"]),
            on="fips",
            how="left",
        )
        .join(nonmed_roc, on="fips", how="left")
        .join(med_roc, on="fips", how="left")
        .with_columns(
            [
                pl.when(pl.col("n_years") >= 2)
                .then(pl.col("slope"))
                .otherwise(national_avg_slope)
                .alias("slope_baseline"),
                pl.when(pl.col("n_years") >= 2)
                .then(pl.col("slope_nonmed"))
                .otherwise(national_avg_slope_nonmed)
                .alias("slope_nonmed_baseline"),
                pl.when(pl.col("n_years") >= 2)
                .then(pl.col("slope_med"))
                .otherwise(national_avg_slope_med)
                .alias("slope_med_baseline"),
            ]
        )
    )

    # 6. Identify bottom 20% of declining counties by coverage slope
    declining = projection_base.filter(pl.col("slope_baseline") < 0)
    n_bottom_30 = int(np.ceil(declining.height * 0.3))

    bottom_30_fips = (
        declining.sort("slope_baseline").head(n_bottom_30).select("fips")
    )

    print(f"\nNumber of declining counties: {declining.height}")
    print(f"Number in bottom 30%: {n_bottom_30}")
    print(
        f"Worst slope in bottom 30%: {declining.sort('slope_baseline').head(1)['slope_baseline'][0]:.4f}"
    )
    print(
        f"Best slope in bottom 30%: {declining.sort('slope_baseline').head(n_bottom_30).tail(1)['slope_baseline'][0]:.4f}"
    )

    # 7. Define scenario modifications
    # Coverage: declining = worsening; improving = coverage going up
    # Exemptions: increasing = worsening; decreasing = exemptions going down (improving)

    def apply_scenario_a(df):
        """(a) Baseline - current trends"""
        return df.with_columns(
            [
                pl.col("slope_baseline").alias("slope_scenario"),
                pl.col("slope_nonmed_baseline").alias("slope_nonmed_scenario"),
                pl.col("slope_med_baseline").alias("slope_med_scenario"),
            ]
        )

    def apply_scenario_b(df):
        """(b) Targeted worsening - worst 30% coverage decline 30% faster;
        nonmed/med exemptions increase 30% faster in those counties"""
        return df.with_columns(
            [
                # Coverage: more negative slope = faster decline
                pl.when(pl.col("fips").is_in(bottom_30_fips["fips"]))
                .then(pl.col("slope_baseline") * 1.3)
                .otherwise(pl.col("slope_baseline"))
                .alias("slope_scenario"),
                # Nonmed: more positive slope = faster increase = worsening
                pl.when(pl.col("fips").is_in(bottom_30_fips["fips"]))
                .then(
                    pl.when(pl.col("slope_nonmed_baseline") > 0)
                    .then(pl.col("slope_nonmed_baseline") * 1.3)
                    .otherwise(pl.col("slope_nonmed_baseline") * 0.7)
                )
                .otherwise(pl.col("slope_nonmed_baseline"))
                .alias("slope_nonmed_scenario"),
                # Med: same logic
                pl.when(pl.col("fips").is_in(bottom_30_fips["fips"]))
                .then(
                    pl.when(pl.col("slope_med_baseline") > 0)
                    .then(pl.col("slope_med_baseline") * 1.3)
                    .otherwise(pl.col("slope_med_baseline") * 0.7)
                )
                .otherwise(pl.col("slope_med_baseline"))
                .alias("slope_med_scenario"),
            ]
        )

    def apply_scenario_c(df):
        """(c) Universal worsening - coverage declines 20% faster (improving -> national avg);
        exemptions increase 20% faster (decreasing -> national avg)"""
        return df.with_columns(
            [
                pl.when(pl.col("slope_baseline") < 0)
                .then(pl.col("slope_baseline") * 1.2)
                .otherwise(national_avg_slope)
                .alias("slope_scenario"),
                # Nonmed: worsening = increasing; if already decreasing, revert to national avg
                pl.when(pl.col("slope_nonmed_baseline") > 0)
                .then(pl.col("slope_nonmed_baseline") * 1.2)
                .otherwise(national_avg_slope_nonmed)
                .alias("slope_nonmed_scenario"),
                # Med: same logic
                pl.when(pl.col("slope_med_baseline") > 0)
                .then(pl.col("slope_med_baseline") * 1.2)
                .otherwise(national_avg_slope_med)
                .alias("slope_med_scenario"),
            ]
        )

    def apply_scenario_d(df):
        """(d) Targeted improvement - worst 30% coverage decline 30% slower;
        nonmed/med exemptions increase 30% slower in those counties"""
        return df.with_columns(
            [
                pl.when(pl.col("fips").is_in(bottom_30_fips["fips"]))
                .then(pl.col("slope_baseline") * 0.7)
                .otherwise(pl.col("slope_baseline"))
                .alias("slope_scenario"),
                pl.when(pl.col("fips").is_in(bottom_30_fips["fips"]))
                .then(
                    pl.when(pl.col("slope_nonmed_baseline") > 0)
                    .then(pl.col("slope_nonmed_baseline") * 0.7)
                    .otherwise(pl.col("slope_nonmed_baseline") * 1.7)
                )
                .otherwise(pl.col("slope_nonmed_baseline"))
                .alias("slope_nonmed_scenario"),
                pl.when(pl.col("fips").is_in(bottom_30_fips["fips"]))
                .then(
                    pl.when(pl.col("slope_med_baseline") > 0)
                    .then(pl.col("slope_med_baseline") * 0.7)
                    .otherwise(pl.col("slope_med_baseline") * 1.3)
                )
                .otherwise(pl.col("slope_med_baseline"))
                .alias("slope_med_scenario"),
            ]
        )

    def apply_scenario_e(df):
        """(e) Universal improvement - coverage declines 20% slower / improving accelerates 20%;
        exemptions increase 20% slower / decreasing accelerates 20%"""
        return df.with_columns(
            [
                pl.when(pl.col("slope_baseline") < 0)
                .then(pl.col("slope_baseline") * 0.8)
                .otherwise(pl.col("slope_baseline") * 1.2)
                .alias("slope_scenario"),
                # Nonmed: worsening = increasing; improvement = decreasing
                pl.when(pl.col("slope_nonmed_baseline") > 0)
                .then(pl.col("slope_nonmed_baseline") * 0.8)
                .otherwise(pl.col("slope_nonmed_baseline") * 1.2)
                .alias("slope_nonmed_scenario"),
                # Med: same logic
                pl.when(pl.col("slope_med_baseline") > 0)
                .then(pl.col("slope_med_baseline") * 0.8)
                .otherwise(pl.col("slope_med_baseline") * 1.2)
                .alias("slope_med_scenario"),
            ]
        )

    scenarios = [
        ("baseline", apply_scenario_a),
        ("targeted_worsening", apply_scenario_b),
        ("universal_worsening", apply_scenario_c),
        ("targeted_improvement", apply_scenario_d),
        ("universal_improvement", apply_scenario_e),
    ]

    # 8. Create projections for each scenario
    all_scenarios = []

    for scenario_name, modifier_func in scenarios:
        print(f"\nProcessing scenario: {scenario_name}")

        scenario_df = modifier_func(projection_base.clone())

        projection_years = list(range(latest_year + 1, end_year + 1))

        for proj_year in projection_years:
            years_ahead = proj_year - scenario_df["baseline_year"].cast(
                pl.Int32
            )

            scenario_df = scenario_df.with_columns(
                [
                    (
                        pl.col("baseline_coverage").cast(pl.Float64)
                        + pl.col("slope_scenario").cast(pl.Float64)
                        * years_ahead
                    )
                    .clip(0.0, 100.0)
                    .alias(f"cov_{proj_year}"),
                    (
                        pl.col("baseline_nonmed_exempt").cast(pl.Float64)
                        + pl.col("slope_nonmed_scenario").cast(pl.Float64)
                        * years_ahead
                    )
                    .clip(0.0, 100.0)
                    .alias(f"nonmed_{proj_year}"),
                    (
                        pl.col("baseline_med_exempt").cast(pl.Float64)
                        + pl.col("slope_med_scenario").cast(pl.Float64)
                        * years_ahead
                    )
                    .clip(0.0, 100.0)
                    .alias(f"med_{proj_year}"),
                ]
            )

        # Reshape coverage to long
        cov_long = (
            scenario_df.select(
                [
                    "fips",
                    "county",
                    "state",
                    "baseline_year",
                    "baseline_coverage",
                    "slope_baseline",
                    "slope_scenario",
                    *[f"cov_{y}" for y in projection_years],
                ]
            )
            .unpivot(
                index=[
                    "fips",
                    "county",
                    "state",
                    "baseline_year",
                    "baseline_coverage",
                    "slope_baseline",
                    "slope_scenario",
                ],
                on=[f"cov_{y}" for y in projection_years],
                variable_name="year_col",
                value_name="projected_coverage",
            )
            .with_columns(
                pl.col("year_col")
                .str.replace("cov_", "")
                .cast(pl.Int32)
                .alias("year")
            )
            .drop("year_col")
        )

        # Reshape nonmed exemptions to long
        nonmed_long = (
            scenario_df.select(
                [
                    "fips",
                    "baseline_nonmed_exempt",
                    "slope_nonmed_baseline",
                    "slope_nonmed_scenario",
                    *[f"nonmed_{y}" for y in projection_years],
                ]
            )
            .unpivot(
                index=[
                    "fips",
                    "baseline_nonmed_exempt",
                    "slope_nonmed_baseline",
                    "slope_nonmed_scenario",
                ],
                on=[f"nonmed_{y}" for y in projection_years],
                variable_name="year_col",
                value_name="projected_nonmed_exempt",
            )
            .with_columns(
                pl.col("year_col")
                .str.replace("nonmed_", "")
                .cast(pl.Int32)
                .alias("year")
            )
            .drop("year_col")
        )

        # Reshape med exemptions to long
        med_long = (
            scenario_df.select(
                [
                    "fips",
                    "baseline_med_exempt",
                    "slope_med_baseline",
                    "slope_med_scenario",
                    *[f"med_{y}" for y in projection_years],
                ]
            )
            .unpivot(
                index=[
                    "fips",
                    "baseline_med_exempt",
                    "slope_med_baseline",
                    "slope_med_scenario",
                ],
                on=[f"med_{y}" for y in projection_years],
                variable_name="year_col",
                value_name="projected_med_exempt",
            )
            .with_columns(
                pl.col("year_col")
                .str.replace("med_", "")
                .cast(pl.Int32)
                .alias("year")
            )
            .drop("year_col")
        )

        # Join all three into one long frame
        scenario_long = (
            cov_long.join(nonmed_long, on=["fips", "year"], how="left")
            .join(med_long, on=["fips", "year"], how="left")
            .with_columns(pl.lit(scenario_name).alias("scenario"))
        )

        all_scenarios.append(scenario_long)

        # Print summary statistics
        for col, label in [
            ("projected_coverage", "coverage"),
            ("projected_nonmed_exempt", "nonmed exempt"),
            ("projected_med_exempt", "med exempt"),
        ]:
            avg = scenario_long.filter(pl.col("year") == end_year)[col].mean()
            mn = scenario_long.filter(pl.col("year") == end_year)[col].min()
            print(f"  Avg {label} in {end_year}: {avg:.2f}%   Min: {mn:.2f}%")

    combined = pl.concat(all_scenarios)
    return combined


if __name__ == "__main__":
    df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
    )
    # let's just plot Fulton and Gwinnett counties over Fulton county only
    df = (
        df.with_columns(
            fips=pl.when(pl.col("county") == "Fulton and Gwynett")
            .then(pl.lit("13121"))
            .otherwise(pl.col("fips")),
            # east tennessee over knox county
        )
        .with_columns(
            fips=pl.when(pl.col("county") == "East tennessee?")
            .then(pl.lit("47093"))
            .otherwise(pl.col("fips")),
        )
        .with_columns(
            fips=pl.when(pl.col("county") == "Central, Utah")
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
            .then(pl.lit("01019"))
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
            fips=pl.when(
                pl.col("county") == "Twin Cities Metro Area, Minnesota"
            )
            .then(pl.lit("27003"))
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
        point_data=overall_cases[["fips", "cases"]],
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
            data_dir, "Compiled_Vaccine_Exemption_Data_R1_110225_Hawaiied.xlsx"
        ),
        schema_overrides={"FIPS": pl.String},
    )

    # remove the funky places lauren found in her QC
    # exclude Butts, Georgia, 2024
    vaccination_data = exclude_county_state_year(
        vaccination_data, "Butts", "Georgia", 2024
    )
    # exclude Jasper, Georgia 2024
    vaccination_data = exclude_county_state_year(
        vaccination_data, "Jasper", "Georgia", 2024
    )
    # exclude Lumpkin, Georgia 2024
    vaccination_data = exclude_county_state_year(
        vaccination_data, "Lumpkin", "Georgia", 2024
    )
    # exclude Richmond, Georgia 2024
    vaccination_data = exclude_county_state_year(
        vaccination_data, "Richmond", "Georgia", 2024
    )
    # exclude schley, georgia 2024
    vaccination_data = exclude_county_state_year(
        vaccination_data, "Schley", "Georgia", 2024
    )
    # exclude baltimore city, maryland 2019
    vaccination_data = exclude_county_state_year(
        vaccination_data, "Baltimore City", "Maryland", 2019
    )
    # drop slope county 2021
    vaccination_data = exclude_county_state_year(
        vaccination_data, "Slope", "North Dakota", 2021
    )
    # for Cocke, Tennessee 2024, number should be corrected from 4.5 to 94.5
    vaccination_data = vaccination_data.with_columns(
        MMR_cov=pl.when(
            (pl.col("county") == "Cocke")
            & (pl.col("state") == "Tennessee")
            & (pl.col("year") == 2024)
        )
        .then(pl.lit(94.5))
        .otherwise(pl.col("MMR_cov"))
    )
    # fill in states that did not report medical exemptions separately
    # as having medical exemptions of 0
    # these two states are wyoming and louisiana
    # we also do this for vermont and mississippi, which censored medical exemptions
    # whenever medical exemptions for these states are null, fill them with 0
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

    # Ad hoc adjustment to some odd fips codes
    # There is an instance of a six-digit fips code
    # it looks to me like this just fairfax county, VA (fips 51059), so truncate
    # just for now while playing around
    vaccination_data = vaccination_data.with_columns(
        FIPS=pl.when(pl.col("FIPS") == "510591")
        .then(pl.lit("51059"))
        .otherwise(pl.col("FIPS")),
        # combine vaccination coverages to make a single estimate
        estimated_mmr_cov=pl.when(pl.col("MMR_cov").is_not_null())
        .then(pl.col("MMR_cov"))
        .otherwise(pl.col("all_vacc_cov")),
    )

    # Make vaccination coverage a number and drop anything that doesn't exist
    vaccination_data = vaccination_data.with_columns(
        estimated_mmr_cov=pl.col("estimated_mmr_cov").cast(
            pl.Float64, strict=False
        )
    )

    # rename "FIPS" for "fips"
    vaccination_data = vaccination_data.rename({"FIPS": "fips"})

    # Some fips code editing
    # edit the washington dc fips code to be 11001 because DC has one county
    vaccination_data = vaccination_data.with_columns(
        fips=pl.when(pl.col("fips") == "11")
        .then(pl.lit("11001"))
        .otherwise(pl.col("fips"))
    )

    # There are some cases where we have a higher spatial unit than the county
    # We want to expand that state/region level data to all available counties
    # This happens for places where we only have state-level data

    # West Virginia, Rhode Island, and Delaware only report state-level data
    # Expand the data to include all fips codes in these states
    # Alaska is funky where the data are provided by economic zone but those
    # encompass multiple census fips code level units
    state_fips_to_expand = {
        "higher_level_fips": [
            *["10"] * 3,  # Delaware has 3 counties
            *["44"] * 5,  # Rhode Island has 5
            *["54"] * 55,  # WV has 55
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
    )
    # for cases where there are county fips available, use those as the state fips
    vaccination_data = vaccination_data.with_columns(
        fips=pl.when(pl.col("county_fips").is_not_null())
        .then(pl.col("county_fips"))
        .otherwise(pl.col("fips"))
    )

    # The Alaska fips codes are largely just wrong
    # Anchorage is 02020 not 20200
    # Gulf Coast (21220) comprises Kodiak Island, Kenai Peninsular, Chugach, and Copper
    # Interior (22900) comprises Yukon-Koyukuk, Denali, Fairbanks North Star, Southeast Fairbanks
    # Matanuska-Susitana is not 21700 but 02170
    # Northern (21850) comprises Nome, North Slope, and Northwest Arctic
    # Southeast (22820) comprises Haines, Yakutat, Hoonah-Angoon, Sitka,
    # Prince of Wales-Hyder, Ketchikan, Wrangell, Petersburg, Juneau, Skagway
    # Southwest (21640) consists of Aleutians East, Aleutians West, Lake and Peninsula,
    # Bristol Bay, Dilingham, Bethel, Kusilvak
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
        alaska_fips, left_on="fips", right_on="old_alaska", how="left"
    )
    vaccination_data = vaccination_data.with_columns(
        fips=pl.when(pl.col("new_alaska").is_not_null())
        .then(pl.col("new_alaska"))
        .otherwise(pl.col("fips"))
    )

    # Nebraska is similar

    nebraska_fips = {
        "old_nebraska": [
            *["310012"] * 3,  # Central
            "310021",  # Dakota
            "310002",  # Douglas
            *["310005"] * 4,  # East Central
            *["310008"] * 4,  # Elkhorn Logan Valley
            *["310007"] * 4,  # Four Corners
            "310003",  # Lincoln-Lancaster
            *["310015"] * 9,  # Loup Basin
            *["310020"] * 9,  # North Central
            *["310010"] * 4,  # Northeast
            *["310018"] * 12,  # Panhandle
            *["310011"] * 5,  # Public Health Solutions
            *["310006"] * 2,  # Sarpy/Cas
            *["310013"] * 4,  # South Heartland
            *["310009"] * 5,  # Southeast
            *["310017"] * 9,  # Southwest
            *["310004"] * 3,  # Three Rivers
            *["310014"] * 7,  # Two Rivers
            *["310016"] * 6,  # West Central
        ],
        "new_nebraska": [
            "31079",
            "31081",
            "31121",  # Hall, Hamilton, and Merrick
            "31043",  # Dakota
            "31055",  # Douglas
            "31011",
            "31037",
            "31125",
            "31141",  # Boone, Colfax, Nance, and Platte
            "31021",
            "31039",
            "31119",
            "31167",  # Burt, Cuming, Madison, Stanton
            "31023",
            "31143",
            "31159",
            "31185",  # Bulter, Polk, Seward, York
            "31109",  # Lancaster
            "31009",
            "31041",
            "31071",
            "31077",
            "31093",
            "31115",
            "31163",
            "31175",
            "31183",  # Blaine, Custer, Garfield, Greeley, Howard, Loup, Sherman, Valley, Wheeler
            "31003",
            "31015",
            "31017",
            "31031",
            "31089",
            "31103",
            "31107",
            "31139",
            "31149",  # Antelope, Boyd, Brown, Cherry, Holt, Keya Paha, Knox, Pierce, Rock
            "31027",
            "31051",
            "31173",
            "31179",  # Cedar, Dixon, Thurston, Wayne
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
            "31105",  # Banner, Box Butte, Cheyenne, Dawes, Deuel, Garden, Grant, Kimball, Sioux, Scotts Bluff, Morrill, Sheridan
            "31059",
            "31067",
            "31095",
            "31151",
            "31169",  # Fillmore, Gage, Jefferson, Saline, Thayer
            "31153",
            "31025",  # Sarpy, Cass
            "31001",
            "31035",
            "31129",
            "31181",  # Adams, Clay, Nuckolls, Webster
            "31097",
            "31127",
            "31131",
            "31133",
            "31147",  # Johnson, Nemaha, Otoe, Pawnee, Richardson
            "31029",
            "31057",
            "31063",
            "31065",
            "31085",
            "31087",
            "31101",
            "31135",
            "31145",  # Chase, Dundy, Frontier, Furnas, Hayes, Hitchcock, Keith, Perkins, Red Willow
            "31053",
            "31177",
            "31155",  # Dodge, Washington, Saunders
            "31019",
            "31047",
            "31061",
            "31073",
            "31083",
            "31099",
            "31137",  # Buffalo, Dawson, Franklin, Gosper, Harlan, kearney, Phelps
            "31005",
            "31091",
            "31111",
            "31113",
            "31117",
            "31171",  # Arthur, Hooker, Lincoln, Logan, McPherson, Thomas
        ],
    }

    nebraska_fips = pl.DataFrame(nebraska_fips)
    vaccination_data = vaccination_data.join(
        nebraska_fips, left_on="fips", right_on="old_nebraska", how="left"
    )
    vaccination_data = vaccination_data.with_columns(
        fips=pl.when(pl.col("new_nebraska").is_not_null())
        .then(pl.col("new_nebraska"))
        .otherwise(pl.col("fips"))
    )

    # Hawaii also only reports state level data mainly, but that too for only one year
    # so we can't even use it in this analysis
    # Indiana largely also only reports state-level data, but that too only for
    # exemptions, so we don't want to assume something stringently about coverage
    # in those cases, so we just don't touch those data

    # The four digit fips codes need to be expanded to five digit codes
    # but don't pad blindly because otherwise we also pad the states
    vaccination_data = vaccination_data.with_columns(
        fips=pl.when(pl.col("fips").str.len_chars() == 4)
        .then(pl.col("fips").str.zfill(5))
        .otherwise(pl.col("fips"))
    )

    # in some cases, we have longer than we need fips details
    vaccination_data = vaccination_data.with_columns(
        # Kansas fips codes are weird because NBC team created their own effectively
        fips=pl.when(
            (pl.col("fips").str.len_chars() > 5)
            & (pl.col("state") != "Kansas")
        )
        .then(pl.col("fips").str.slice(0, 5))
        .otherwise(pl.col("fips"))
    )

    # add in the kansas data
    kansas = vaccination_wrangling.read_kansas_data(data_dir)
    kansas = kansas.rename({"mmr_cov": "estimated_mmr_cov"})
    vaccination_data = pl.concat(
        [vaccination_data, kansas], how="diagonal_relaxed"
    )

    montana_vacc = vaccination_wrangling.read_montana_data(data_dir=data_dir)
    # join with regular vaccination data on state, county, and year
    vaccination_data = vaccination_data.join(
        montana_vacc,
        on=["state", "county", "year"],
        how="left",
        suffix="_montana",
    )
    # if we have montana data, use it to fill in missing values in the main vaccination data
    # only do this if the state is montana
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

    # for montana, the fips codes in later years are missing
    # bring them down
    vaccination_data = vaccination_data.with_columns(
        fips=pl.when((pl.col("state") == "Montana") & pl.col("fips").is_null())
        .then(pl.col("fips").forward_fill().over("county"))
        .otherwise(pl.col("fips"))
    )

    # if we have data for the county in other years but are missing one year,
    # fill it in with the average of the neighboring years for that county
    vaccination_data = vaccination_data.sort(
        "state", "county", "year"
    ).with_columns(
        [
            pl.col(col).interpolate().over("state", "county")
            for col in [
                "estimated_mmr_cov",
                "nonmed_exempt_pct",
                "med_exempt_pct",
            ]
        ]
    )

    # if we have state-level reported values for a state but not county-level,
    # fill nulls with the mean of the non-null county-level values for that state and year
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

    # find any places where estimated mmr cov is 0
    print(vaccination_data.filter(pl.col("estimated_mmr_cov") == 0.0))
    # drop these places
    vaccination_data = vaccination_data.filter(
        pl.col("estimated_mmr_cov") != 0.0
    )
    # also ensure estimated mmr cov is not null or nan
    print(vaccination_data.filter(pl.col("estimated_mmr_cov").is_null()))
    print(vaccination_data.filter(pl.col("estimated_mmr_cov").is_nan()))
    vaccination_data = vaccination_data.filter(
        pl.col("estimated_mmr_cov").is_not_null()
        & ~pl.col("estimated_mmr_cov").is_nan()
    )

    # filter
    vaccination_data_latest = vaccination_data.filter(pl.col("year") == 2023)

    # what we want to do is really grab the latest available data not to exceed
    # a given year for all counties
    # we validate on 2021, 2022, 2023, 2024, and 2025
    validation_years = [2021, 2022, 2023, 2024, 2025, 2026]
    for year in validation_years:
        # grab the latest available vaccination data
        vaccination_data_export = (
            vaccination_data.filter(pl.col("estimated_mmr_cov") != 0.0)
            .filter(pl.col("fips").is_not_null())
            .filter(
                (pl.col("estimated_mmr_cov").is_not_null())
                & (pl.col("nonmed_exempt_pct").is_not_null())
                & (pl.col("med_exempt_pct").is_not_null())
            )
            .filter(pl.col("year") < year)
            .sort(["fips", "year"], descending=True)
            .group_by("fips", maintain_order=True)
            .agg(
                pl.first("year"),
                pl.first("estimated_mmr_cov"),
                pl.first("nonmed_exempt_pct"),
                pl.first("med_exempt_pct"),
                pl.first("state"),
                pl.first("county"),
            )
            .filter(pl.col("fips").str.len_chars() > 2)
        )

        vaccination_data_export.write_csv(
            os.path.join(data_dir, f"{year}_vaccination_data.csv")
        )

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
        cbar_label="2023-2024 MMR vaccine coverage (%)",
        cbar_type="bucketed_coverage",
    )

    # how many counties are below 95%?
    print("counties below 95%:")
    print(
        vaccination_data_latest.filter(
            pl.col("estimated_mmr_cov") < 95
        ).count()
    )
    # what is that as a proportion of all counties reporting data?
    print(
        vaccination_data_latest.filter(
            pl.col("estimated_mmr_cov") < 95
        ).count()
        / vaccination_data_latest.count()
    )

    # Calculate the average change in coverage
    # subset to 2021 onwards
    vaccination_data_roc = vaccination_data.filter(pl.col("year") >= 2020)
    vaccination_data_roc = vaccination_data_roc.group_by("fips").map_groups(
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

    vaccination_data_roc = vaccination_data_roc.filter(
        pl.col("slope").is_not_nan()
    )

    # map of the change in vaccination coverage
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
        cbar_label="Average annual MMR vaccine coverage change\nsince 2020 (%)",
        cbar_type="discrete",
    )

    # While we're here, calculate how vaccination coverage has changed over
    # year over year
    vaccination_data = vaccination_data.sort(["fips", "year"]).with_columns(
        # by fips code, calculate year over year change
        year_over_year_change=pl.col("estimated_mmr_cov")
        - pl.col("estimated_mmr_cov").shift(1).over("fips")
    )

    # export this
    vaccination_data.write_csv(
        os.path.join(data_dir, "year_over_year_change_vaccination_data.csv")
    )

    print("\n" + "=" * 80)
    print("GENERATING VACCINE COVERAGE PROJECTIONS")
    print("=" * 80)

    projections = project_coverage_scenarios(
        vaccination_data, vaccination_data_roc, latest_year=2025, end_year=2030
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
        scenario_df.write_csv(
            os.path.join(
                data_dir, f"vaccine_coverage_projections_{scenario}.csv"
            )
        )
        print(f"Saved: vaccine_coverage_projections_{scenario}.csv")

    print("\nProjection files saved successfully!")
    print("=" * 80 + "\n")

    plt.tight_layout(rect=(0, 0, 1, 1))
    # save
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
