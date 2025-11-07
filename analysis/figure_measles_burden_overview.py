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

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
fig_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")

font = {"family": "Nimbus Roman", "size": 28}
rc("font", **font)

# read analysis params yaml
with open("params.yml", "r") as f:
    params = yaml.safe_load(f)


def setup_plot(nrows, ncols):
    fig, ax = plt.subplots(nrows, ncols, figsize=(27, 8))
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
    df = pd.DataFrame({"GEOID": fips, "value": vals})

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
        breakpoints = [min(vals), -1, -0.5, -0.1, 0, 0.1, 0.5, 1, max(vals)]
        norm = mcolors.BoundaryNorm(
            boundaries=breakpoints, ncolors=len(breakpoints) - 1
        )
        # make the colormap
        cmap = mcolors.ListedColormap(
            [
                "#8c510a",  # -inf to -1
                "#bf812d",  # -1 to -0.5
                "#dfc27d",  # -0.5 to -0.1
                "#f6e8c3",  # -0.1 to 0
                "#c7eae5",  # 0 to 0.1
                "#80cdc1",  # 0.1 to 0.5
                "#35978f",  # 0.5 to 1
                "#01665e",  # 1 to inf
            ]
        )
    else:
        assert False, f"cbar_type {cbar_type} not recognized"

    # Plot counties with data
    counties_with_data[has_data_mask].plot(
        column="value",
        cmap=cmap,
        linewidth=0.3,
        ax=ax,
        edgecolor="grey",
        legend=False,
        missing_kwds={"color": "lightgrey"},
    )

    # Create custom colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    cbar = fig.colorbar(
        sm,
        ax=ax,
        shrink=0.7,  # Size of colorbar relative to axes
        aspect=20,  # Ratio of long to short dimensions
        pad=0.0,  # Distance between axes and colorbar
        extend="both"
        if cbar_type == "discrete"
        else "neither",  # Add arrow at top ('neither', 'both', 'min', 'max')
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
        cbar.set_ticks([-9, -0.4, -0.05, 0.03, 0.2, 0.7, 3.5, 20])
        cbar.set_ticklabels(
            [
                "<-1%",
                "-1% to -0.5%",
                "-0.5% to -0.1%",
                "-0.1% to 0%",
                "0% to 0.1%",
                "0.1% to 0.5%",
                "0.5% to 1%",
                ">1%",
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
        sizes = np.sqrt(list(point_values)) * 10

        # Plot the points
        ax.scatter(
            counties_with_points["centroid"].x,
            counties_with_points["centroid"].y,
            s=sizes,
            c="blue",
            alpha=0.2,
            linewidth=0.8,
        )

    # Remove axis
    ax.set_axis_off()

    return fig, ax


if __name__ == "__main__":
    df = pl.read_csv(
        os.path.join(data_dir, "raw_merged_outbreaks_vaccinations.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
    )

    # Some places are missing fips codes -- this happens when the vaccination
    # data doesn't exist so we didn't bother to fill the fips codes in
    af = AddFIPS()
    df = df.with_columns(
        fips=pl.when(pl.col("fips").is_null())
        .then(
            pl.struct(pl.col("county"), pl.col("state")).map_elements(
                lambda x: af.get_county_fips(x["county"], x["state"]),
                return_dtype=pl.String,
            )
        )
        .otherwise(pl.col("fips"))
    )

    df = df.with_columns(
        cases_2025=pl.when(pl.col("year") == 2025)
        .then(pl.col("cases"))
        .otherwise(0)
    )

    df_by_fips = df.group_by("fips")
    overall_cases = df_by_fips.agg(
        cases=pl.sum("cases") - pl.sum("cases_2025"),
        cases_2025=pl.sum("cases_2025"),
        outbreak=(pl.col("cases") >= 10).any().cast(pl.Int64),
    )

    # map of total cases
    fig, ax = setup_plot(1, 3)  # type: ignore
    ax[0].text(  # type: ignore
        -0.05,
        1.08,
        "a",
        transform=ax[0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    map_fips_code_data(
        overall_cases["fips"],
        overall_cases["cases_2025"],
        fig,
        ax[0],  # type: ignore
        cbar_label="Reported cases in 2025",
        cbar_type="continuous_cases",
        point_data=overall_cases[["fips", "cases"]],
    )

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
            *["310018"] * 8,  # Panhandle
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
            "31105",  # Banner, Box Butte, Cheyenne, Dawes, Deuel, Garden, Grant, Kimball
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
        fips=pl.when(pl.col("fips").str.len_chars() > 5)
        .then(pl.col("fips").str.slice(0, 5))
        .otherwise(pl.col("fips"))
    )

    # take the last year of vaccination data as long as that year does not
    # exceed 2025 (because 2024 coverage determines 2025 outbreaks)
    vaccination_data = vaccination_data.with_columns(
        latest_cov=pl.when(pl.col("year") <= 2024)
        .then(pl.col("estimated_mmr_cov"))
        .otherwise(pl.lit(0))
    )

    vaccination_data_latest = vaccination_data.group_by("fips").agg(
        latest_cov=pl.col("latest_cov").max()
    )

    vaccination_data_latest = vaccination_data_latest.filter(
        pl.col("latest_cov") > 50
    )

    # map of the last year's of vaccination data
    ax[1].text(  # type: ignore
        -0.05,
        1.08,
        "b",
        transform=ax[1].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    map_fips_code_data(
        vaccination_data_latest["fips"],
        vaccination_data_latest["latest_cov"],
        fig,
        ax[1],  # type: ignore
        cbar_label="Last reported MMR vaccine coverage (%)",
        cbar_type="continuous_coverage",
    )

    # Calculate the average change in coverage
    vaccination_data = vaccination_data.group_by("fips").map_groups(
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

    vaccination_data = vaccination_data.filter(pl.col("slope").is_not_nan())

    # map of the change in vaccination coverage
    ax[2].text(  # type: ignore
        -0.05,
        1.08,
        "c",
        transform=ax[2].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    map_fips_code_data(
        vaccination_data["fips"],
        vaccination_data["slope"],
        fig,
        ax[2],  # type: ignore
        cbar_label="Average yearly change in MMR\nvaccine coverage (pp/year)",
        cbar_type="discrete",
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
