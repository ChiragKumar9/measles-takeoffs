import os

from numpyro import enable_x64

enable_x64()
import pickle

import geopandas as gpd
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import yaml
from matplotlib import rc, ticker
from matplotlib.patches import Patch
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
    fig, ax = plt.subplots(nrows, ncols, figsize=(20, 17))
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
            R2=pl.lit(
                np.corrcoef(
                    temp_regression["nonmed_exempt_pct"].to_numpy()
                    * estimator[0]
                    + estimator[1],
                    temp_regression[
                        "estimated_mmr_cov_change_policy_year"
                    ].to_numpy(),
                )[0, 1]
                ** 2
            ),
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
            R2=pl.lit(
                np.corrcoef(
                    temp_regression["nonmed_exempt_pct"].to_numpy()
                    * estimator[0]
                    + estimator[1],
                    temp_regression["estimated_mmr_cov_change_1yr"].to_numpy(),
                )[0, 1]
                ** 2
            ),
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
                R2=pl.lit(
                    np.corrcoef(
                        temp_regression["nonmed_exempt_pct"].to_numpy()
                        * estimator[0]
                        + estimator[1],
                        temp_regression[
                            "estimated_mmr_cov_change_2yr"
                        ].to_numpy(),
                    )[0, 1]
                    ** 2
                ),
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

    ax.axhline(0, color="black", linewidth=ax.spines["bottom"].get_linewidth())
    ax.legend(loc=(0.6, 0.0))
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
            R2=pl.lit(
                np.corrcoef(
                    temp_regression["nonmed_exempt_pct"].to_numpy()
                    * estimator[0]
                    + estimator[1],
                    temp_regression[
                        f"estimated_mmr_cov_change_{year_mode}"
                    ].to_numpy(),
                )[0, 1]
                ** 2
            ),
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
            R2=pl.lit(
                estimator.score(
                    X,
                    temp_regression[
                        f"estimated_mmr_cov_change_{year_mode}"
                    ].to_numpy(),
                )
            ),
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
    slope_lower = regression_df.filter(
        (pl.col("State") == "All states") & (pl.col("Year") == "1 year after")
    )["Slope_lower"][0]
    slope_upper = regression_df.filter(
        (pl.col("State") == "All states") & (pl.col("Year") == "1 year after")
    )["Slope_upper"][0]
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
    # put some confidence bounds
    y_lower = intercept + slope_lower * x_vals
    y_upper = intercept + slope_upper * x_vals
    ax.fill_between(
        x_vals,
        y_lower,
        y_upper,
        color="grey",
        alpha=0.3,
    )

    # annotate the figure with the regression equation
    ax.text(
        0.05,
        0.95,
        f"Slope: {slope:.3f} ({slope_lower:.3f}, {slope_upper:.3f})",
        transform=ax.transAxes,
        fontsize=32,
        verticalalignment="top",
    )
    # annotate the figure with the R2
    r2 = regression_df.filter(
        (pl.col("State") == "All states") & (pl.col("Year") == "1 year after")
    )["R2"][0]
    ax.text(
        0.05,
        0.85,
        rf"$R^2$: {r2 * 100:.2f}%",
        transform=ax.transAxes,
        fontsize=32,
        verticalalignment="top",
    )

    # annotate the figure with the 1-1 line
    ax.plot(
        x_vals,
        x_vals,
        color="grey",
        linestyle="--",
        linewidth=1,
    )

    regression_df = (
        regression_df.with_columns(
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
            # round R2
            R2=(pl.col("R2") * 100).round(3),
        )
        .drop(["Slope_lower", "Slope_upper"])
        .rename(
            {
                "Slope": "Slope (95% CI)",
                "Intercept": "Intercept",
                "R2": "R² (%)",
            }
        )
    )

    return regression_df


def scenario_results(df, fig, ax):
    # read the risk projections
    with open(
        os.path.join(
            output_dir,
            "risk_projections.pkl",
        ),
        "rb",
    ) as f:
        projections = pickle.load(f)

    # read the df used to make the risk projections
    df = pl.read_csv(
        os.path.join(output_dir, "scaled_projection_dataset.csv"),
        # make sure we read fips codes as strings
        schema_overrides={
            "fips": pl.String,
            "estimated_mmr_cov_raw": pl.Float64,
        },
    )
    df = df.filter(pl.col("in_train") == 0)

    # make into an incidence measure
    # clip at population size
    projections = jnp.minimum(projections, jnp.array(df["pop"]).reshape(1, -1))
    incidence = projections  # / df["pop"].to_numpy()

    # concatenate to dataframe so we can print out counties by risk
    df = df.with_columns(
        projections=pl.Series(incidence.T.__array__()),
    )

    # add in import probability
    with open(
        os.path.join(
            output_dir, "empirical_importation_probability_samples.pkl"
        ),
        "rb",
    ) as f:
        samples = pickle.load(f)
    import_df = pl.read_csv(
        os.path.join(output_dir, "empirical_importation_probability.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
    )
    import_df = import_df.with_columns(
        importation_probability_samples=pl.Series(
            samples["importation_probability_samples"].__array__()
        )
    )
    # join with main dataframe
    df = df.join(
        import_df.select(
            [
                "fips",
                "year",
                "type",
                "importation_probability_samples",
            ]
        ),
        on=["fips", "year", "type"],
        how="left",
    )
    # multiply the importation probability samples with the incidence projections
    importation_probability_samples = np.vstack(
        df["importation_probability_samples"].to_numpy()  # type: ignore
    ).T
    unconditioned_risk_samples = importation_probability_samples * incidence

    # sum over all counties to get expected number of yearly cases
    expected_cases_samples = unconditioned_risk_samples.sum(axis=1)
    # violin plot this
    # violinparts = ax.violinplot(
    # expected_cases_samples,
    # positions=[0],
    # showextrema=False,
    # widths=0.6,
    # )
    # for pc in violinparts["bodies"]:
    # pc.set(facecolor="grey", alpha=0.4)
    # print(expected_cases_samples.min(), expected_cases_samples.max())

    # add error bars for the median and interquartile range
    median = np.median(expected_cases_samples)
    lower = np.percentile(expected_cases_samples, 25)
    upper = np.percentile(expected_cases_samples, 75)
    ax.errorbar(
        0,
        median,
        yerr=[[median - lower], [upper - median]],
        fmt="o",
        color="grey",
        capsize=5,
        capthick=2,
    )
    # dashed grey line at this median
    # ax.axhline(
    # median,
    # color="grey",
    # linestyle="--",
    # linewidth=1,
    # )

    # add the runs for each of the scenarios
    scenarios = {
        "Remove existing NME bans": "add_nmes_back",
        "NME bans in highest risk states": "top_risk_state_unconditioned",
        "NME bans in highest NME states": "top_nme_state",
        "NME bans in highest risk counties": "top_ten_pct_counties",
    }
    # read sims
    with open(
        os.path.join(
            output_dir,
            f"nme_policy_analysis_median.pkl",
        ),
        "rb",
    ) as f:
        results = pickle.load(f)

    no_nme_df = pl.read_csv(
        os.path.join(
            output_dir, "nme_policy_analysis_broken_up_by_scenario.csv"
        ),
        schema_overrides={
            "fips": pl.String,
            "estimated_mmr_cov_raw": pl.Float64,
            "nonmed_exempt_pct_raw": pl.Float64,
        },
    )

    results = jnp.minimum(results, jnp.array(no_nme_df["pop"]).reshape(1, -1))
    no_nme_df = no_nme_df.with_columns(
        projections=pl.Series(results.T.__array__()),
    )

    # join no nme_df with results and import probability samples
    no_nme_df = no_nme_df.join(
        import_df.select(
            [
                "fips",
                "year",
                "importation_probability_samples",
            ]
        ),
        on=["fips", "year"],
        how="left",
    )

    # join with the baseline
    no_nme_df = no_nme_df.join(
        df.select(["fips", "year", "projections"]).rename(
            {"projections": "baseline_projections"}
        ),
        on=["fips", "year"],
        how="left",
    )

    # make violin plots for each scenario
    num_per_scenario = no_nme_df.shape[0] // len(scenarios)
    for idx, (scenario, column) in enumerate(scenarios.items(), start=1):
        # first we want to plot the baseline without including neighbors
        temp = no_nme_df[
            (idx - 1) * num_per_scenario : idx * num_per_scenario, :
        ]
        # we need to make two violins
        # one for the scenario itself
        scenario_df = temp.with_columns(
            projections_direct=pl.when(pl.col(column) == True)
            .then(pl.col("projections"))
            .otherwise(pl.col("baseline_projections")),
        )
        # extract the projections
        scenario_projections = jnp.vstack(
            scenario_df["projections_direct"].to_numpy()
        ).T
        # multiply with importation probability samples
        scenario_importation_probability_samples = np.vstack(
            scenario_df["importation_probability_samples"].to_numpy()  # type: ignore
        ).T
        scenario_risk_samples = (
            scenario_importation_probability_samples * scenario_projections
        )
        # sum over all counties to get expected number of yearly cases
        scenario_expected_cases_samples = scenario_risk_samples.sum(axis=1)
        # make a violin plot of this
        # violinparts = ax.violinplot(
        # scenario_expected_cases_samples,
        # positions=[idx - 0.2],
        # showextrema=False,
        # widths=0.3,
        # )
        # for pc in violinparts["bodies"]:
        # pc.set(facecolor="#298C8C", alpha=0.4)
        # add error bars for the median and interquartile range
        median = np.median(scenario_expected_cases_samples)
        lower = np.percentile(scenario_expected_cases_samples, 25)
        upper = np.percentile(scenario_expected_cases_samples, 75)
        print(
            scenario_expected_cases_samples.min(),
            scenario_expected_cases_samples.max(),
        )
        print(scenario, median, lower, upper)
        ax.errorbar(
            idx - 0.2,
            median,
            yerr=[[median - lower], [upper - median]],
            fmt="o",
            color="#298C8C",
            capsize=5,
            capthick=2,
        )

        # now similar thing but for the whole scenario including neighbors
        # here we just use the full projections since the neighbors are already included
        scenario_projections = jnp.vstack(temp["projections"].to_numpy()).T
        # multiply with importation probability samples
        scenario_importation_probability_samples = np.vstack(
            temp["importation_probability_samples"].to_numpy()  # type: ignore
        ).T
        scenario_risk_samples = (
            scenario_importation_probability_samples * scenario_projections
        )
        # sum over all counties to get expected number of yearly cases
        scenario_expected_cases_samples = scenario_risk_samples.sum(axis=1)
        # make a violin plot of this
        # violinparts = ax.violinplot(
        # scenario_expected_cases_samples,
        # positions=[idx + 0.2],
        # showextrema=False,
        # widths=0.3,
        # )
        # for pc in violinparts["bodies"]:
        # pc.set(facecolor="#800074", alpha=0.4)
        # add error bars for the median and interquartile range
        print(
            scenario_expected_cases_samples.min(),
            scenario_expected_cases_samples.max(),
        )
        median = np.median(scenario_expected_cases_samples)
        lower = np.percentile(scenario_expected_cases_samples, 25)
        upper = np.percentile(scenario_expected_cases_samples, 75)
        ax.errorbar(
            idx + 0.2,
            median,
            yerr=[[median - lower], [upper - median]],
            fmt="o",
            color="#800074",
            capsize=5,
            capthick=2,
        )

    # add a legend
    legend_elements = [
        Patch(
            facecolor="grey",
            edgecolor="grey",
            alpha=0.4,
            label="No intervention",
        ),
        Patch(
            facecolor="#298C8C",
            edgecolor="#298C8C",
            alpha=0.4,
            label="Direct effects",
        ),
        Patch(
            facecolor="#800074",
            edgecolor="#800074",
            alpha=0.4,
            label="Spillover effects",
        ),
    ]
    ax.legend(
        handles=legend_elements,
    )

    ax.set_xticks(range(len(scenarios) + 1))
    ax.set_xticklabels(
        ["No intervention"] + list(scenarios.keys()), rotation=-45, ha="left"
    )
    ax.set_ylabel("Annual measles cases (N)")
    # ax.set_ylim(2000, 15000)
    # ax.set_yscale("log")
    # set y minor ticks
    # ax.yaxis.set_minor_locator(
    #     ticker.LogLocator(
    #         base=10.0,
    #         subs=np.arange(1.0, 10.0),  # type: ignore
    #                                              numticks=10))
    # ax.set_ylim(top = 100000)


def plot_who_benefits_most(df, fig, ax):
    df = df.filter(
        (
            pl.col("top_risk_state_unconditioned") == True
            # | (pl.col("bordering_high_risk_state") == True)
        )
    )
    df = df.filter(pl.col("median_percent_change") <= 0)
    ax.scatter(
        x=df["estimated_mmr_cov_not_scaled"],
        # df["change_in_mmr_cov"],
        alpha=0.7,
        y=df["median_percent_change"],
    )
    # cbar = fig.colorbar(
    # plt.cm.ScalarMappable(cmap="viridis"),
    # ax=ax,
    # shrink=0.7,
    # aspect=20,
    # pad=0.0,
    # extend="neither",
    # orientation="vertical",
    # )
    # cbar.set_label(
    # "Percent change in outbreak size (%)",
    # rotation=90,
    # labelpad=15,
    # fontsize=28,
    # )
    # cbar.ax.tick_params(labelsize=24)
    ax.set_xlabel("Initial MMR coverage (%)")
    ax.set_ylabel("Change in median outbreak size (%)")
    ax.set_xlim(50)


if __name__ == "__main__":
    df = pl.read_csv(
        os.path.join(data_dir, "latest_vaccination_data.csv"),
        schema_overrides={
            "fips": pl.String,
            "estimated_mmr_cov_raw": pl.Float64,
        },
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

    df = pl.read_csv(
        os.path.join(output_dir, "nme_policy_analysis_assembled_median.csv"),
        schema_overrides={
            "estimated_mmr_cov_raw": pl.Float64,
            "nonmed_exempt_pct_raw": pl.Float64,
            "change_in_mmr_cov": pl.Float64,
            "fips": pl.String,
        },
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
    plot_data = scenario_results(df, fig, ax[1, 0])  # type: ignore

    # read the df used to make the risk projections
    df_raw = pl.read_csv(
        os.path.join(output_dir, "projection_dataset.csv"),
        # make sure we read fips codes as strings
        schema_overrides={
            "fips": pl.String,
            "estimated_mmr_cov_raw": pl.Float64,
        },
    )

    df_raw = df_raw.drop_nans()

    # read the risk projections
    with open(
        os.path.join(
            output_dir,
            "risk_projections.pkl",
        ),
        "rb",
    ) as f:
        projections = pickle.load(f)

    df_raw = df_raw.with_columns(
        baseline=pl.Series(projections.T.__array__()),
    )

    # merge
    df = df.join(
        df_raw.select(["fips", "estimated_mmr_cov", "baseline"]),
        on="fips",
        how="left",
        suffix="_not_scaled",
    )

    # read sims
    with open(
        os.path.join(
            output_dir,
            f"nme_policy_analysis_median.pkl",
        ),
        "rb",
    ) as f:
        results = pickle.load(f)

    results = jnp.minimum(results, jnp.array(df["pop"]).reshape(1, -1))
    df = df.with_columns(
        projections=pl.Series(results.T.__array__()),
    )

    df = df[2765 : (2765 * 2), :]

    # extract the median percent change in outbreak size for the top risk state scenario
    df = df.with_columns(
        median_percent_change=(pl.col("projections") - pl.col("baseline"))
        / pl.col("baseline")
        * 100
    )
    # summarize by taking the median
    # extract the values
    median_percent_change = df["median_percent_change"].to_numpy()
    # take the median
    median_percent_change = np.median(median_percent_change, axis=1)
    df = df.with_columns(
        median_percent_change=pl.Series(median_percent_change)
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
    plot_who_benefits_most(df, fig, ax[1, 1])  # type: ignore

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
