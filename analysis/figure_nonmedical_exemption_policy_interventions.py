import os

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import yaml
from matplotlib import rc
from scipy import stats
from shapely.geometry import MultiPolygon, Polygon
from sklearn.linear_model import TheilSenRegressor

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


def coverage_increase_nme_policy_change(df, fig, ax):
    states = {
        "California": 2016,
        "Maine": 2021,
        "New York": 2019,
        "Connecticut": 2022,
    }

    all_dfs = []
    regression_df = []
    for state, year in states.items():
        t1 = df.filter(
            (pl.col("state") == state) & (pl.col("year") == year - 1)
        ).select(
            [
                "state",
                "year",
                "county",
                "nonmed_exempt_pct",
                "estimated_mmr_cov",
            ]
        )

        t2 = df.filter(
            (pl.col("state") == state) & (pl.col("year") == year)
        ).select(
            [
                "state",
                "year",
                "county",
                "nonmed_exempt_pct",
                "estimated_mmr_cov",
            ]
        )

        temp = t1.join(
            t2, on=["state", "county"], suffix="_policy_year"
        ).with_columns(
            estimated_mmr_cov_change_policy_year=pl.col(
                "estimated_mmr_cov_policy_year"
            )
            - pl.col("estimated_mmr_cov"),
        )

        # do a regression between nonmed_exempt_pct and estimated_mmr_cov_change_policy_year
        temp_regression = temp[
            ["nonmed_exempt_pct", "estimated_mmr_cov_change_policy_year"]
        ].drop_nulls()
        estimator = stats.theilslopes(
            temp_regression["estimated_mmr_cov_change_policy_year"].to_numpy(),
            temp_regression["nonmed_exempt_pct"].to_numpy(),
        )
        temp_regression = pl.DataFrame().with_columns(
            State=pl.lit(state + " only"),
            Year=pl.lit("Policy change year: " + str(year)),
            Slope=pl.lit(estimator[0]),
            Slope_lower=pl.lit(estimator[2]),
            Slope_upper=pl.lit(estimator[3]),
            Intercept=pl.lit(estimator[1]),
            N=pl.lit(temp_regression.height),
        )
        regression_df.append(temp_regression)

        t3 = df.filter(
            (pl.col("state") == state) & (pl.col("year") == year + 1)
        ).select(
            [
                "state",
                "year",
                "county",
                "nonmed_exempt_pct",
                "estimated_mmr_cov",
            ]
        )

        temp = temp.join(
            t3, on=["state", "county"], suffix="_1yr"
        ).with_columns(
            estimated_mmr_cov_change_1yr=pl.col("estimated_mmr_cov_1yr")
            - pl.col("estimated_mmr_cov"),
        )

        temp_regression = temp[
            ["nonmed_exempt_pct", "estimated_mmr_cov_change_1yr"]
        ].drop_nulls()
        estimator = stats.theilslopes(
            temp_regression["estimated_mmr_cov_change_1yr"].to_numpy(),
            temp_regression["nonmed_exempt_pct"].to_numpy(),
        )
        temp_regression = pl.DataFrame().with_columns(
            State=pl.lit(state + " only"),
            Year=pl.lit("1 year after: " + str(year + 1)),
            Slope=pl.lit(estimator[0]),
            Slope_lower=pl.lit(estimator[2]),
            Slope_upper=pl.lit(estimator[3]),
            Intercept=pl.lit(estimator[1]),
            N=pl.lit(temp_regression.height),
        )
        regression_df.append(temp_regression)

        t4 = df.filter(
            (pl.col("state") == state) & (pl.col("year") == year + 2)
        ).select(
            [
                "state",
                "year",
                "county",
                "nonmed_exempt_pct",
                "estimated_mmr_cov",
            ]
        )

        temp = temp.join(
            t4, on=["state", "county"], suffix="_2yr"
        ).with_columns(
            estimated_mmr_cov_change_2yr=pl.col("estimated_mmr_cov_2yr")
            - pl.col("estimated_mmr_cov"),
        )

        if state != "Connecticut":
            # connecticut reported state-level data in 2024
            temp_regression = temp[
                ["nonmed_exempt_pct", "estimated_mmr_cov_change_2yr"]
            ].drop_nulls()
            estimator = stats.theilslopes(
                temp_regression["estimated_mmr_cov_change_2yr"].to_numpy(),
                temp_regression["nonmed_exempt_pct"].to_numpy(),
            )
            temp_regression = pl.DataFrame().with_columns(
                State=pl.lit(state + " only"),
                Year=pl.lit("2 years after: " + str(year + 2)),
                Slope=pl.lit(estimator[0]),
                Slope_lower=pl.lit(estimator[2]),
                Slope_upper=pl.lit(estimator[3]),
                Intercept=pl.lit(estimator[1]),
                N=pl.lit(temp_regression.height),
            )
            regression_df.append(temp_regression)

        all_dfs.append(temp)

    result_df = pl.concat(all_dfs)

    # plot
    state_colors = {
        "California": "lightblue",
        "Maine": "purple",
        "New York": "darkgreen",
        "Connecticut": "orange",
    }

    for state in states.keys():
        state_df = result_df.filter(pl.col("state") == state)
        ax.scatter(
            state_df["nonmed_exempt_pct"].to_numpy(),
            state_df["estimated_mmr_cov_change_1yr"].to_numpy(),
            label=state,
            alpha=0.7,
            s=150,
            color=state_colors[state],
        )

    ax.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax.legend()
    ax.set_xlabel("Pre-policy nonmedical exemption rate (%)")
    ax.set_ylabel("Change in MMR coverage (%)")
    ax.set_xlim(0)

    # do a regression for all states combined
    for year_mode in ["policy_year", "1yr", "2yr"]:
        temp_regression = result_df[
            ["nonmed_exempt_pct", f"estimated_mmr_cov_change_{year_mode}"]
        ].drop_nulls()
        estimator = stats.theilslopes(
            temp_regression[
                f"estimated_mmr_cov_change_{year_mode}"
            ].to_numpy(),
            temp_regression["nonmed_exempt_pct"].to_numpy(),
        )
        temp_regression = pl.DataFrame().with_columns(
            State=pl.lit("All states"),
            Year=pl.lit(
                "Policy change year"
                if year_mode == "policy_year"
                else "1 year after"
                if year_mode == "1yr"
                else "2 years after"
            ),
            Slope=pl.lit(estimator[0]),
            Slope_lower=pl.lit(estimator[2]),
            Slope_upper=pl.lit(estimator[3]),
            Intercept=pl.lit(estimator[1]),
            N=pl.lit(temp_regression.height),
        )
        regression_df.append(temp_regression)

    # do the same but add in state fixed effects
    for year_mode in ["policy_year", "1yr", "2yr"]:
        temp_regression = result_df[
            [
                "state",
                "nonmed_exempt_pct",
                f"estimated_mmr_cov_change_{year_mode}",
            ]
        ].drop_nulls()
        temp_regression = temp_regression.to_dummies("state", drop_first=True)
        x_cols = ["nonmed_exempt_pct"] + [
            col for col in temp_regression.columns if col.startswith("state_")
        ]
        X = temp_regression.select(x_cols).to_numpy()
        estimator = TheilSenRegressor(random_state=108).fit(
            X,
            temp_regression[
                f"estimated_mmr_cov_change_{year_mode}"
            ].to_numpy(),
        )
        temp_regression = pl.DataFrame().with_columns(
            State=pl.lit("All states with state fixed effects"),
            Year=pl.lit(
                "Policy change year"
                if year_mode == "policy_year"
                else "1 year after"
                if year_mode == "1yr"
                else "2 years after"
            ),
            Slope=pl.lit(estimator.coef_[0]),
            Slope_lower=pl.lit(None),
            Slope_upper=pl.lit(None),
            Intercept=pl.lit(estimator.intercept_),
            N=pl.lit(temp_regression.height),
        )
        regression_df.append(temp_regression)

    # merge slopes
    regression_df = pl.concat(regression_df)

    # add a line to the plot for the regression of all states combined
    x_vals = np.array(ax.get_xlim())
    slope = regression_df.filter(
        (pl.col("State") == "All states") & (pl.col("Year") == "1 year after")
    )["Slope"][0]
    intercept = regression_df.filter(
        (pl.col("State") == "All states") & (pl.col("Year") == "1 year after")
    )["Intercept"][0]
    y_vals = intercept + slope * x_vals
    ax.plot(
        x_vals,
        y_vals,
        color="black",
        linewidth=2,
    )

    regression_df = regression_df.with_columns(
        Slope=pl.when(pl.col("Slope_lower").is_not_null())
        .then(
            pl.col("Slope").round(3).cast(pl.String)
            + " ["
            + pl.col("Slope_lower").round(3).cast(pl.String)
            + ", "
            + pl.col("Slope_upper").round(3).cast(pl.String)
            + "]"
        )
        .otherwise(pl.col("Slope").round(3).cast(pl.String)),
        # round the intercept values
        Intercept=pl.col("Intercept").round(3),
    ).drop(["Slope_lower", "Slope_upper"])

    return regression_df


if __name__ == "__main__":
    df = pl.read_csv(
        os.path.join(data_dir, "latest_vaccination_data.csv"),
        schema_overrides={"fips": pl.String},
    )

    # map of risk
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
    map_fips_code_data(
        df["fips"],
        df["nonmed_exempt_pct"],
        fig,
        ax[0, 0],  # type: ignore
        cbar_label="Nonmedical exemption rate (%)",
        cbar_type="exemptions",
        point_data=None,
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
    vac_df = pl.read_csv(
        os.path.join(data_dir, "processed_vaccinations_data.csv"),
        schema_overrides={
            "fips": pl.String,
            "estimated_mmr_cov": pl.Float64,
            "nonmed_exempt_pct": pl.Float64,
        },
    )
    regression_df = coverage_increase_nme_policy_change(vac_df, fig, ax[0, 1])  # type: ignore
    regression_df.write_csv(
        os.path.join(
            fig_dir,
            "table_supplemental_mmr_coverage_change_nme_policy_interventions.csv",
        )
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

    fig.tight_layout()
    # save
    fig.savefig(
        os.path.join(
            fig_dir, "figure_nonmedical_exemption_policy_interventions.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(
            fig_dir, "figure_nonmedical_exemption_policy_interventions.pdf"
        ),
        dpi=300,
        bbox_inches="tight",
    )
