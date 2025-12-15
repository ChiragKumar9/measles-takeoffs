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


if __name__ == "__main__":
    df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
    )
    # let's just plot Fulton and Gwinnett counties over Fulton county only
    df = df.with_columns(
        fips=pl.when(pl.col("county") == "Fulton and Gwynett")
        .then(pl.lit("13121"))
        .otherwise(pl.col("fips")),
        # east tennessee over knox county
    ).with_columns(
        fips=pl.when(pl.col("county") == "East tennessee?")
        .then(pl.lit("47093"))
        .otherwise(pl.col("fips")),
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
            data_dir, "Compiled_Vaccine_Exemption_Data_Final_v1_092425.xlsx"
        ),
        schema_overrides={"FIPS": pl.String},
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
    ).filter(pl.col("FIPS").is_not_null())

    # Make vaccination coverage a number and drop anything that doesn't exist
    vaccination_data = vaccination_data.with_columns(
        estimated_mmr_cov=pl.col("estimated_mmr_cov").cast(
            pl.Float64, strict=False
        )
    ).filter(pl.col("estimated_mmr_cov").is_not_null())

    # correct the baltimore oddity
    vaccination_data = vaccination_data.with_columns(
        estimated_mmr_cov=pl.when(
            (pl.col("state") == "Maryland")
            & (pl.col("county") == "Baltimore City")
            & (pl.col("year") == 2019)
        )
        .then(
            pl.lit(
                vaccination_data.filter(
                    (pl.col("state") == "Maryland")
                    & (pl.col("county") == "Baltimore City")
                    & ((pl.col("year") == 2018) | (pl.col("year") == 2020))
                )["estimated_mmr_cov"].mean()
            )
        )
        .otherwise(pl.col("estimated_mmr_cov"))
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

    # filter
    vaccination_data_latest = vaccination_data.filter(pl.col("year") == 2023)

    # grab the latest available vaccination data
    vaccination_data_export = (
        vaccination_data.filter(
            (pl.col("estimated_mmr_cov").is_not_null())
            & (pl.col("nonmed_exempt_pct").is_not_null())
            & (pl.col("med_exempt_pct").is_not_null())
        )
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
    )

    vaccination_data_export.write_csv(
        os.path.join(data_dir, "latest_vaccination_data.csv")
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
