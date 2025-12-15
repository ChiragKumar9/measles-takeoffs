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

state_abbreviations = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}


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


def plot_validation(df, fig, ax):
    # we have validation data for 2023, 2024, and 2025
    projection_years = [2023, 2024, 2025]
    df = df.filter(pl.col("year").is_in(projection_years))
    df = df.with_columns(
        # must be either "min", "ordinal", "random" to be true percentiles
        overall_rank_order=pl.col("cases_outbreak").rank(
            "random", descending=False, seed=108
        )
        - 1
    ).with_columns(
        overall_percentile=pl.col("overall_rank_order")
        / pl.col("overall_rank_order").max()
        * 100
    )
    dfs_by_year = {}
    for year in projection_years:
        df_year = df.filter(pl.col("year") == year)
        # calculate the rank order of the samples
        with open(
            os.path.join(
                output_dir,
                f"samples_partial_pooled_poisson_validation_testing_{year}.pkl",
            ),
            "rb",
        ) as file:
            projections = pickle.load(file)

        ranked_projections = np.argsort(
            np.argsort(-projections, axis=1, kind="stable"),
            axis=1,
            kind="stable",
        )

        df_year = df_year.with_columns(
            median_risk_rank=pl.Series(
                jnp.median(ranked_projections, axis=0).__array__()
            ),
            lower_risk_rank=pl.Series(
                jnp.percentile(ranked_projections, 25, axis=0).__array__()
            ),
            upper_risk_rank=pl.Series(
                jnp.percentile(ranked_projections, 75, axis=0).__array__()
            ),
        ).with_columns(
            # turn into percentiles
            median_risk_rank=pl.col("median_risk_rank") * 100 / df_year.height,
            lower_risk_rank=pl.col("lower_risk_rank") * 100 / df_year.height,
            upper_risk_rank=pl.col("upper_risk_rank") * 100 / df_year.height,
        )
        df_year = df_year.sort("median_risk_rank")
        dfs_by_year[year] = df_year

    # want to plot overall rank vs predicted rank
    df_ranked = df.sort("overall_rank_order")
    year_color = {
        2023: "#78c679",
        2024: "#238443",
        2025: "#004529",
    }
    for year in projection_years:
        ax.errorbar(
            dfs_by_year[year]["cases_outbreak"],
            dfs_by_year[year]["median_risk_rank"],
            yerr=np.array(
                [
                    np.array(
                        [
                            dfs_by_year[year]["median_risk_rank"].to_numpy()
                            - dfs_by_year[year]["lower_risk_rank"].to_numpy()
                        ]
                    ),
                    np.array(
                        [
                            dfs_by_year[year]["upper_risk_rank"].to_numpy()
                            - dfs_by_year[year]["median_risk_rank"].to_numpy()
                        ]
                    ),
                ]
            )[:, 0, :],
            fmt="o",
            color=year_color[year],
            ecolor=year_color[year],
            label=str(year),
            alpha=0.7,
        )
    # counter_by_year = {2023: 0, 2024: 0, 2025: 0}
    # for outbreak_size, year in zip(
    #     df_ranked['cases_outbreak'],
    #     df_ranked['year']
    # ):
    #     # get the relevant value for the year
    #     predicted_rank = dfs_by_year[year]['median_risk_rank'][counter_by_year[year]]
    #     lower_rank = dfs_by_year[year]['lower_risk_rank'][counter_by_year[year]]
    #     upper_rank = dfs_by_year[year]['upper_risk_rank'][counter_by_year[year]]
    #     # augment the counter for the given year
    #     counter_by_year[year] += 1

    #     # plot
    #     if counter_by_year[year] == 1:
    #         label = str(year)
    #     else:
    #         label = None
    #     ax.errorbar(
    #         outbreak_size,
    #         predicted_rank,
    #         yerr = [[predicted_rank - lower_rank],
    #                 [upper_rank - predicted_rank]],
    #         fmt='o',
    #         color=year_color[year],
    #         ecolor=year_color[year],
    #         label=label,
    #         alpha = 0.7
    #     )

    ax.set_xlabel("Reported cases (N)")
    ax.set_xscale("symlog")
    ax.set_ylabel("Predicted outbreak size percentile")
    ax.legend()

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

    if cbar_type == "percentile_risk":
        norm = mcolors.Normalize(vmin=0, vmax=100)
        breakpoints = [-1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 101]
        cmap = mcolors.ListedColormap(
            [
                "#fff5f0",
                "#fee0d2",
                "#fcbba1",
                "#fc9272",
                "#fb6a4a",
                "#ef3b2c",
                "#cb181d",
                "#a50f15",
                "#67000d",
                "#3f0000",
            ]
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
            cbar.set_ticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            cbar.set_ticklabels(
                [
                    "Bottom 10%",
                    "10%-20%",
                    "20%-30%",
                    "30%-40%",
                    "40%-50%",
                    "50%-60%",
                    "60%-70%",
                    "70%-80%",
                    "80%-90%",
                    "Top 10%",
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


def plot_county_risk_fcn_of_coverage(samples, df, fig, ax):
    df = df.with_columns(projections=samples.T.__array__())

    # make an incidence measure
    df = df.with_columns(incidence=pl.col("projections") / pl.col("pop"))

    # extract the incidence data to do numpy rank ordering
    incidence = np.vstack(df["incidence"].to_numpy())
    percentile_projections = np.argsort(
        np.argsort(incidence, axis=0, kind="stable"), axis=0, kind="stable"
    )
    percentile_projections = (
        percentile_projections / percentile_projections.max() * 100
    )

    # now calculate median and 95% CrI across samples
    df = df.with_columns(
        median_risk_rank=pl.Series(
            jnp.median(percentile_projections, axis=1).__array__()
        ),
        lower_risk_rank=pl.Series(
            jnp.percentile(percentile_projections, 2.5, axis=1).__array__()
        ),
        upper_risk_rank=pl.Series(
            jnp.percentile(percentile_projections, 97.5, axis=1).__array__()
        ),
        # median incidences
        median_projection=pl.Series(jnp.median(incidence, axis=1).__array__()),
        lower_projection=pl.Series(
            jnp.percentile(incidence, 25, axis=1).__array__()
        ),
        upper_projection=pl.Series(
            jnp.percentile(incidence, 75, axis=1).__array__()
        ),
    )

    # now plot rank vs MMR coverage
    ax.errorbar(
        df["estimated_mmr_cov"].to_numpy(),
        df["median_risk_rank"].to_numpy(),
        yerr=0,
        # [[df['median_risk_rank'].to_numpy() - df['lower_risk_rank'].to_numpy()],
        #         [df['upper_risk_rank'].to_numpy() - df['median_risk_rank'].to_numpy()]],
        fmt="o",
        alpha=0.7,
        ecolor="black",
        color="black",
    )

    ax.set_ylabel("Risk percentile")
    ax.set_xlabel("MMR coverage (%)")

    return fig, ax


def plot_state_risk_fcn_of_coverage(samples, df, fig, ax):
    df = df.with_columns(projections=samples.T.__array__())
    # group by state and add projections
    state_df = df.group_by("state").agg(
        pl.col("projections").map_elements(
            lambda x: np.sum(np.stack([x]), axis=0).tolist(),
            return_dtype=pl.List(pl.Int64),
        ),
        # count number of counties per state
        n_counties=pl.col("fips").count(),
        mmr_coverage=pl.col("estimated_mmr_cov").mean(),
        total_population=pl.col("pop").sum(),
    )

    # now sum across all counties
    state_df = state_df.with_columns(
        projections=pl.col("projections").map_elements(
            lambda x: np.array(x).sum(axis=0).tolist(),
            return_dtype=pl.List(pl.Int64),
        )
    )

    # make an incidence measure
    state_df = state_df.with_columns(
        incidence=pl.col("projections") / pl.col("total_population")
    )

    # extract the incidence data to do numpy rank ordering
    incidence = np.vstack(state_df["incidence"].to_numpy())
    percentile_projections = np.argsort(
        np.argsort(incidence, axis=0, kind="stable"), axis=0, kind="stable"
    )
    percentile_projections = (
        percentile_projections / percentile_projections.max() * 100
    )

    # now calculate median and 95% CrI across samples
    state_df = state_df.with_columns(
        median_risk_rank=pl.Series(
            jnp.median(percentile_projections, axis=1).__array__()
        ),
        lower_risk_rank=pl.Series(
            jnp.percentile(percentile_projections, 2.5, axis=1).__array__()
        ),
        upper_risk_rank=pl.Series(
            jnp.percentile(percentile_projections, 97.5, axis=1).__array__()
        ),
        # median incidences
        median_projection=pl.Series(jnp.median(incidence, axis=1).__array__()),
        lower_projection=pl.Series(
            jnp.percentile(incidence, 2.5, axis=1).__array__()
        ),
        upper_projection=pl.Series(
            jnp.percentile(incidence, 97.5, axis=1).__array__()
        ),
    )

    # get state abbreviations
    state_df = state_df.with_columns(
        state_abbr=pl.col("state").replace(state_abbreviations)
    )

    # now plot rank vs MMR coverage
    for cov, median_risk, lower_risk, upper_risk, state in zip(
        state_df["mmr_coverage"],
        state_df["median_risk_rank"],
        state_df["lower_risk_rank"],
        state_df["upper_risk_rank"],
        state_df["state_abbr"],
    ):
        # ax.errorbar(
        #     cov,
        #     median_risk,
        #     yerr = [[median_risk - lower_risk],
        #             [upper_risk - median_risk]],
        #     fmt='none',
        #     alpha=0.7,
        #     ecolor='black',
        # )

        ax.scatter(cov, median_risk, s=0, alpha=0.0)

        ax.text(
            cov,
            median_risk,
            state,
            fontsize=18,
            va="center",
            ha="center",
        )

    ax.set_ylabel("Risk percentile")
    ax.set_xlabel("MMR coverage (%)")

    return fig, ax


if __name__ == "__main__":
    df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
    )

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

    plot_validation(df, fig, ax[0, 0])  # type: ignore

    with open(
        os.path.join(
            output_dir,
            "risk_projections.pkl",
        ),
        "rb",
    ) as f:
        projections = pickle.load(f)

    df = pl.read_csv(
        os.path.join(output_dir, "projection_dataset.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
    )

    incidence = projections / df["pop"].to_numpy()

    percentile_projections = np.argsort(
        np.argsort(incidence, axis=1, kind="stable"), axis=1, kind="stable"
    )
    percentile_projections = (
        percentile_projections / percentile_projections.max() * 100
    )

    # concatenate to dataframe so we can print out counties by risk
    df = df.with_columns(
        median_risk_rank=pl.Series(
            jnp.median(percentile_projections, axis=0).__array__()
        ),
        lower_risk_rank=pl.Series(
            jnp.percentile(percentile_projections, 2.5, axis=0).__array__()
        ),
        upper_risk_rank=pl.Series(
            jnp.percentile(percentile_projections, 97.5, axis=0).__array__()
        ),
        # median projections
        median_projection=pl.Series(jnp.median(incidence, axis=0).__array__()),
        lower_projection=pl.Series(
            jnp.percentile(incidence, 2.5, axis=0).__array__()
        ),
        upper_projection=pl.Series(
            jnp.percentile(incidence, 97.5, axis=0).__array__()
        ),
    )
    print(
        df.sort("median_risk_rank")
        .select(
            [
                "state",
                "county",
                "fips",
                "median_risk_rank",
                "median_projection",
            ]
        )
        .to_pandas()
        .head(60)
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
        df["median_risk_rank"],
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
    plot_county_risk_fcn_of_coverage(projections, df, fig, ax[1, 0])  # type: ignore

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

    plot_state_risk_fcn_of_coverage(projections, df, fig, ax[1, 1])  # type: ignore

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
