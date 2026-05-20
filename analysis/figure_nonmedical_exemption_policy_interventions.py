import os

from numpyro import enable_x64

enable_x64()
import pickle

import geopandas as gpd
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

with open("params.yml", "r") as f:
    params = yaml.safe_load(f)

FUTURE_YEARS = [2026, 2027, 2028, 2029, 2030]
INTERVENTIONS = [
    # "remove_bans",
    "improve_bottom_decile_5pct",
    "improve_cases_previously_counties",
    "ban_top_nme_states",
    "ban_top_risk_states",
    # "improve_top_risk_states_5pct",
    "ban_top_risk_counties",
]

INTERVENTION_LABELS = {
    "remove_bans": "Remove existing\nNME policies",
    "ban_top_risk_states": "Top risk states",
    "ban_top_nme_states": "Top NME states",
    "ban_top_risk_counties": "Top risk counties",
    "improve_cases_previously_counties": "Counties with\n           previous outbreaks",
    "improve_bottom_decile_5pct": "Lowest coverage counties",
    "improve_top_risk_states_5pct": "Top risk states",
}

DIRECT_COLOR = "#298C8C"
TOTAL_COLOR = "#800074"


def setup_plot(nrows, ncols):
    fig, ax = plt.subplots(nrows, ncols, figsize=(16, 20))
    if nrows * ncols == 1:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="out")
        for spine in ax.spines.values():
            spine.set_position(("outward", 5))
        ax.set_axisbelow(True)
        return ax
    for temp in ax.flatten():  # type: ignore
        temp.spines["top"].set_visible(False)
        temp.spines["right"].set_visible(False)
        temp.get_xaxis().tick_bottom()
        temp.get_yaxis().tick_left()
        temp.tick_params(axis="x", direction="out")
        temp.tick_params(axis="y", direction="out")
        for spine in temp.spines.values():
            spine.set_position(("outward", 5))
        temp.set_axisbelow(True)
    return fig, ax


def map_fips_code_data(
    fips, vals, fig, ax, cbar_label, cbar_type, point_data=None
):
    counties = gpd.read_file(
        os.path.join(data_dir, "cb_2020_us_county_20m.shp")
    )
    states = gpd.read_file(os.path.join(data_dir, "cb_2020_us_state_20m.shp"))

    exclude_fips = ["60", "66", "69", "72", "78"]
    counties = counties[~counties["STATEFP"].isin(exclude_fips)]
    states = states[~states["STATEFP"].isin(exclude_fips)]

    def reposition_alaska_hawaii(gdf):
        gdf = gdf.copy()
        alaska_mask = gdf["STATEFP"] == "02"
        if alaska_mask.any():
            alaska_geoms = gdf.loc[alaska_mask, "geometry"].copy()

            def fix_alaska_coords(geom):
                if geom.geom_type == "Polygon":
                    coords = list(geom.exterior.coords)
                    fixed_coords = []
                    for x, y in coords:
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

            alaska_geoms = alaska_geoms.apply(fix_alaska_coords)
            alaska_bounds = alaska_geoms.total_bounds
            alaska_transformed = alaska_geoms.translate(
                xoff=-alaska_bounds[0] - 600, yoff=-alaska_bounds[1] + 80
            ).scale(xfact=0.2, yfact=0.27, origin=(0, 0))
            gdf.loc[alaska_mask, "geometry"] = alaska_transformed

        hawaii_mask = gdf["STATEFP"] == "15"
        if hawaii_mask.any():
            hawaii_bounds = gdf[hawaii_mask].total_bounds
            gdf.loc[hawaii_mask, "geometry"] = gdf.loc[
                hawaii_mask, "geometry"
            ].translate(
                xoff=-hawaii_bounds[0] - 110, yoff=-hawaii_bounds[1] + 25
            )
        return gdf

    counties = reposition_alaska_hawaii(counties)
    states = reposition_alaska_hawaii(states)

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

    if not state_data.empty:
        state_values = dict(zip(state_data["GEOID"], state_data["value"]))

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
    elif cbar_type is None:
        cmap = None
        norm = None
    else:
        assert False, f"cbar_type {cbar_type} not recognized"

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

    if cbar_type is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(
            sm,
            ax=ax,
            shrink=0.7,
            aspect=20,
            pad=0.0,
            extend="neither",
            orientation="horizontal",
        )
        cbar.set_label(cbar_label, rotation=0, labelpad=0, fontsize=28)
        if cbar_type == "exemptions":
            cbar.set_ticks([1.25, 4.0, 8.0, 12.5, 17.5, 50])
            cbar.set_ticklabels(
                ["≤2.5%", "2.5%-5%", "5%-10%", "10%-15%", "15%-20%", ">20%"]
            )
            cbar.ax.tick_params(labelsize=24, rotation=-60, length=0)

    states.boundary.plot(ax=ax, linewidth=1.0, edgecolor="black", alpha=0.8)

    if point_data is not None:
        counties_with_data["centroid"] = counties_with_data.geometry.centroid

    point_df = pd.DataFrame(point_data, columns=["GEOID", "point_value"])
    counties_with_points = counties_with_data.merge(
        point_df, on="GEOID", how="inner"
    )

    if not counties_with_points.empty:
        point_values = counties_with_points["point_value"]
        sizes = np.array(list(point_values))
        ax.scatter(
            counties_with_points["centroid"].x,
            counties_with_points["centroid"].y,
            s=sizes,
            c="red",
            alpha=0.4,
            linewidth=0.8,
        )

    if cbar_type is None:
        size_values = [10, 50, 100, 500]
        size_labels = ["10", "50", "100", "500"]
        legend_elements = []
        for size, label in zip(size_values, size_labels):
            legend_elements.append(
                plt.scatter([], [], s=size, c="red", alpha=0.8, label=label)
            )
        ax.legend(
            handles=legend_elements,
            title="Measles cases (N)",
            loc="upper right",
            frameon=False,
            scatterpoints=1,
            ncol=len(size_values),
            bbox_to_anchor=(1.15, 0.035),
        )

    ax.set_axis_off()
    return fig, ax


def coverage_increase_nme_policy_change(df, fig, ax):
    states = {
        "California": 2016,
        "Connecticut": 2022,
        "Maine": 2021,
        "New York": 2019,
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

    # Calculate and print average bumps for each state
    print("\nAverage MMR Coverage Changes by State:")
    print("=" * 60)
    for state, year in states.items():
        state_df = all_dfs[list(states.keys()).index(state)]

        # Policy year change
        policy_change = (
            state_df["estimated_mmr_cov_change_policy_year"]
            .drop_nulls()
            .mean()
        )
        print(f"\n{state} (Policy year: {year}):")
        print(f"  Policy year change: {policy_change:.2f}%")

        # 1 year after change
        yr1_change = (
            state_df["estimated_mmr_cov_change_1yr"].drop_nulls().mean()
        )
        print(f"  1 year after: {yr1_change:.2f}%")

    result_df = pl.concat(all_dfs)

    result_df = pl.concat(all_dfs)

    state_colors = {
        "California": "lightblue",
        "Connecticut": "orange",
        "Maine": "purple",
        "New York": "darkgreen",
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

    ax.axhline(0, color="k", ls="--")
    ax.set_ylim(-9)
    ax.legend(
        loc=(0.57, 0.02), edgecolor="black", title="Counties in:           "
    )
    ax.set_xlabel("Pre-policy nonmedical exemption rate (%)")
    ax.set_ylabel(
        "Change in MMR coverage after removal of\nnonmedical exemptions (%)"
    )
    ax.set_xlim(0)

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

    regression_df = pl.concat(regression_df)

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
    ax.plot(x_vals, y_vals, color="black", linewidth=2)
    y_lower = intercept + slope_lower * x_vals
    y_upper = intercept + slope_upper * x_vals
    ax.fill_between(x_vals, y_lower, y_upper, color="grey", alpha=0.3)

    ax.text(
        0.02,
        0.95,
        f"Slope: {slope:.2f} ({slope_lower:.2f}, {slope_upper:.2f})",
        transform=ax.transAxes,
        fontsize=32,
        verticalalignment="top",
    )
    r2 = regression_df.filter(
        (pl.col("State") == "All states") & (pl.col("Year") == "1 year after")
    )["R2"][0]
    ax.text(
        0.02,
        0.85,
        rf"$R^2$: {r2 * 100:.2f}%",
        transform=ax.transAxes,
        fontsize=32,
        verticalalignment="top",
    )
    ax.plot(x_vals, x_vals, color="grey", linestyle="--", linewidth=1)

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
            Intercept=pl.col("Intercept").round(3),
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


def load_importation_samples_future(output_dir):
    future_base = pl.read_csv(
        os.path.join(output_dir, "projection_dataset_2026.csv"),
        schema_overrides={"fips": pl.String},
    ).filter(pl.col("type") == "projection")

    reference = pl.read_csv(
        os.path.join(output_dir, "projection_dataset_baseline_2026.csv"),
        schema_overrides={"fips": pl.String},
    )
    future_base = future_base.join(
        reference.select("fips"), on="fips", how="inner"
    )

    import_df = pl.read_csv(
        os.path.join(output_dir, "empirical_importation_probability.csv"),
        schema_overrides={"fips": pl.String},
    )
    with open(
        os.path.join(
            output_dir, "empirical_importation_probability_samples.pkl"
        ),
        "rb",
    ) as f:
        imp_raw = pickle.load(f)
    with open(
        os.path.join(output_dir, "empirical_importation_count_samples.pkl"),
        "rb",
    ) as f:
        count_raw = pickle.load(f)

    import_df = import_df.with_columns(
        importation_probability_samples=pl.Series(imp_raw),
        expected_importations_samples=pl.Series(count_raw),
    )
    import_df = import_df.sort("type").unique(
        subset=["state", "county", "year", "fips"], keep="last"
    )
    import_df = import_df.sort("year", descending=True).unique(
        subset=["fips"], keep="first"
    )

    future_base = future_base.join(
        import_df.select(
            [
                "fips",
                "importation_probability_samples",
                "expected_importations_samples",
            ]
        ),
        on="fips",
        how="left",
    )
    assert future_base["importation_probability_samples"].null_count() == 0, (
        "Some future counties missing importation probability samples"
    )
    assert future_base["expected_importations_samples"].null_count() == 0, (
        "Some future counties missing importation count samples"
    )

    importation_samples_future = jnp.array(
        np.vstack(future_base["importation_probability_samples"].to_numpy())  # type: ignore
    ).T
    importation_count_samples_future = jnp.array(
        np.vstack(future_base["expected_importations_samples"].to_numpy())  # type: ignore
    ).T
    return (
        importation_samples_future,
        importation_count_samples_future,
        future_base["fips"].to_list(),
    )


def plot_intervention_effects(
    interventions,
    intervention_labels,
    future_years,
    baseline_samples_by_year,
    intervention_data,
    importation_samples,
    importation_count_samples,
    baseline_fips,
    scale,
    fig,
    ax,
):
    violin_width = 0.12
    pair_gap = 0.35

    # In plot_intervention_effects, update INTERVENTION_COLORS:
    INTERVENTION_COLORS = {
        "ban_top_risk_states": DIRECT_COLOR,
        "ban_top_nme_states": TOTAL_COLOR,
        "ban_top_risk_counties": "#8B5CF6",
        "improve_cases_previously_counties": "#F59E0B",
        "improve_bottom_decile_5pct": "#10B981",
        "improve_bottom_decile_10pct": "#10B981",
        "improve_top_risk_states_5pct": "#EF4444",
        "improve_top_risk_states_10pct": "#EF4444",
        "improve_top_risk_counties_5pct": "#8B5CF6",
        "improve_top_risk_counties_10pct": "#8B5CF6",
    }

    fips_to_idx = {f: i for i, f in enumerate(baseline_fips)}

    BAN_INTERVENTIONS = {
        "ban_top_risk_states",
        "ban_top_nme_states",
        "ban_top_risk_counties",
    }
    REMOVE_INTERVENTIONS = {"remove_bans"}

    PLOT_INTERVENTIONS = [
        iv for iv in interventions if iv not in REMOVE_INTERVENTIONS
    ]
    n_interventions = len(PLOT_INTERVENTIONS)

    offsets = np.linspace(-pair_gap, pair_gap, n_interventions)

    vln_handles = []

    for i_idx, intervention in enumerate(PLOT_INTERVENTIONS):
        color = INTERVENTION_COLORS[intervention]

        # Build county index mappings from first year (consistent across years)
        first_pkg = intervention_data[intervention][future_years[0]]
        first_fips_list = first_pkg["fips"]
        first_int_applied = first_pkg["intervention_applied"]

        proj_fips_to_idx = {f: j for j, f in enumerate(first_fips_list)}
        common_fips = [f for f in baseline_fips if f in proj_fips_to_idx]
        baseline_col_idx = jnp.array([fips_to_idx[f] for f in common_fips])
        proj_col_idx = jnp.array([proj_fips_to_idx[f] for f in common_fips])

        int_fips = [f for f, a in zip(first_fips_list, first_int_applied) if a]
        int_fips_set = set(int_fips)
        int_in_common_mask = jnp.array(
            [f in int_fips_set for f in common_fips], dtype=bool
        )

        imp = importation_samples
        imp_count = importation_count_samples

        cumulative_total_by_county = None
        cumulative_baseline_by_county = None

        for y_idx, year in enumerate(future_years):
            pkg = intervention_data[intervention][year]
            pred = pkg["predictions"]

            # --- Total: (n_samples, n_common_counties) ---
            b_county = (
                baseline_samples_by_year[year][:, baseline_col_idx]
                * imp[:, baseline_col_idx]
                + imp_count[:, baseline_col_idx]
            ) / scale
            i_county = (
                pred[:, proj_col_idx] * imp[:, baseline_col_idx]
                + imp_count[:, baseline_col_idx]
            ) / scale

            if intervention not in REMOVE_INTERVENTIONS:
                i_county = jnp.where(
                    True,  # int_in_common_mask,
                    jnp.minimum(i_county, b_county),
                    i_county,
                )
            elif intervention in REMOVE_INTERVENTIONS:
                i_county = jnp.where(
                    True,  # int_in_common_mask,
                    jnp.maximum(i_county, b_county),
                    i_county,
                )

            cumulative_baseline_by_county = (
                b_county
                if cumulative_baseline_by_county is None
                else cumulative_baseline_by_county + b_county
            )
            cumulative_total_by_county = (
                i_county
                if cumulative_total_by_county is None
                else cumulative_total_by_county + i_county  # type: ignore
            )

            # cumulative_baseline_by_county = b_county
            # cumulative_total_by_county = i_county

            # Cumulative diff summed across counties: (n_samples,)
            cumulative_diffs = jnp.sum(
                cumulative_total_by_county - cumulative_baseline_by_county,
                axis=1,
            )

            # --- Print % cumulative cases averted ---
            baseline_total = jnp.sum(
                cumulative_baseline_by_county, axis=1
            )  # (n_samples,)

            # Cases averted = baseline - intervention (negative diffs = averted)
            pct_averted = (
                -cumulative_diffs / baseline_total
            ) * 100  # (n_samples,)

            med_pct = float(jnp.median(pct_averted))
            p25_pct = float(jnp.percentile(pct_averted, 2.5))
            p75_pct = float(jnp.percentile(pct_averted, 97.5))

            print(
                f"{intervention} | Year {year}: "
                f"median {med_pct:.2f}% averted "
                f"[CrI: {p25_pct:.2f}% – {p75_pct:.2f}%]"
            )

            x_pos = y_idx + offsets[i_idx]

            lo = float(jnp.percentile(cumulative_diffs, 2.5))
            hi = float(jnp.percentile(cumulative_diffs, 97.5))
            vln_data = np.array(cumulative_diffs)[
                (np.array(cumulative_diffs) >= lo)
                & (np.array(cumulative_diffs) <= hi)
            ]
            vln = ax.violinplot(
                vln_data,
                positions=[x_pos],
                widths=violin_width,
                showextrema=False,
            )
            for pc in vln["bodies"]:
                pc.set_facecolor(color)
                pc.set_edgecolor(color)
                pc.set_alpha(0.3)

            med = float(jnp.median(cumulative_diffs))
            p25 = float(jnp.percentile(cumulative_diffs, 25))
            p75 = float(jnp.percentile(cumulative_diffs, 75))
            ax.errorbar(
                x_pos,
                med,
                yerr=np.array([[med - p25], [p75 - med]]),
                fmt="o",
                markersize=6,
                color=color,
                ecolor=color,
                capsize=4,
                capthick=2,
            )

        vln_handles.append(
            (
                vln["bodies"][0],  # type: ignore
                intervention_labels[intervention],
            )
        )

    ax.axhline(0, color="black", ls="--")
    ax.set_xticks(range(len(future_years)))
    ax.set_xticklabels(future_years)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative change in projected cases")

    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))

    # Calculate average coverage changes for NME ban interventions
    avg_cov_changes = {}

    NME_BAN_INTERVENTIONS = {
        "ban_top_risk_states",
        "ban_top_nme_states",
        "ban_top_risk_counties",
    }

    for intervention in interventions:
        if intervention in NME_BAN_INTERVENTIONS:
            # Load intervention base file to get coverage changes
            int_base = pl.read_csv(
                os.path.join(
                    output_dir, f"intervention_base_{intervention}.csv"
                ),
                schema_overrides={"fips": pl.String},
            )
            base_2026 = pl.read_csv(
                os.path.join(output_dir, "projection_dataset_2026.csv"),
                schema_overrides={"fips": pl.String},
            ).filter(pl.col("type") == "projection")

            # Join to get baseline coverage
            coverage_comparison = (
                int_base.join(
                    base_2026.select(
                        [
                            "fips",
                            pl.col("estimated_mmr_cov").alias("baseline_cov"),
                        ]
                    ),
                    on="fips",
                    how="left",
                )
                .with_columns(
                    [
                        (
                            pl.col("estimated_mmr_cov")
                            - pl.col("baseline_cov")
                        ).alias("cov_change")
                    ]
                )
                .filter(pl.col("intervention_applied"))
            )

            # Calculate average coverage change across all intervened counties
            avg_change = coverage_comparison["cov_change"].mean()
            avg_cov_changes[intervention] = avg_change

            print(
                f"{intervention}: Average coverage change = {avg_change:.2f}pp across {coverage_comparison.height} counties"
            )

    from matplotlib.legend_handler import HandlerBase
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    class HeaderHandler(HandlerBase):
        def create_artists(
            self,
            legend,
            orig_handle,
            xdescent,
            ydescent,
            width,
            height,
            fontsize,
            trans,
        ):
            line = Line2D([0], [0], linestyle="none", marker="none", alpha=0)
            line.set_transform(trans)
            return [line]

    dummy = Rectangle(
        (0, 0), 1, 1, fc="white", ec="white", linewidth=0, alpha=0
    )

    nme_items = []
    alt_items = []

    for h, label in vln_handles:
        intervention_name = [
            k for k, v in intervention_labels.items() if v == label
        ][0]
        if intervention_name in NME_BAN_INTERVENTIONS:
            # Add average coverage change to label
            avg_change = avg_cov_changes.get(intervention_name, 0)
            updated_label = f"{label} (Δ coverage: {avg_change:+.1f}pp)"
            nme_items.append((h, updated_label))
        else:
            alt_items.append((h, label))

    combined_handles = [
        dummy,
        alt_items[0][0],
        alt_items[1][0],
        dummy,  # col 1: header, item, item
        dummy,
        nme_items[0][0],
        nme_items[1][0],
        nme_items[2][0],  # col 2: header, item, item, item
    ]
    spaces = "           "
    combined_labels = [
        "5pp coverage increase in:",
        spaces + alt_items[0][1],
        spaces + alt_items[1][1],
        "",
        "Remove nonmedical exemptions in:",
        spaces + nme_items[0][1],
        spaces + nme_items[1][1],
        spaces + nme_items[2][1],
    ]

    ax.legend(
        handles=combined_handles,
        labels=combined_labels,
        handler_map={dummy: HeaderHandler()},
        loc=(0.001, 0.005),
        edgecolor="black",
        frameon=True,
        ncol=2,
        columnspacing=2.0,
        handletextpad=-2,
    )


def plot_coverage_change_vs_cases(
    intervention,
    future_years,
    baseline_samples_by_year,
    intervention_data,
    importation_samples,
    importation_count_samples,
    baseline_fips,
    scale,
    adjacency_data_path,
    output_dir,
    fig,
    ax,
    year=2026,
):
    BAN_INTERVENTIONS = {
        "ban_top_risk_states",
        "ban_top_nme_states",
        "ban_top_risk_counties",
    }
    REMOVE_INTERVENTIONS = {"remove_bans"}

    int_base = pl.read_csv(
        os.path.join(output_dir, f"intervention_base_{intervention}.csv"),
        schema_overrides={"fips": pl.String},
    )
    base_2026 = pl.read_csv(
        os.path.join(output_dir, "projection_dataset_2026.csv"),
        schema_overrides={"fips": pl.String},
    ).filter(pl.col("type") == "projection")

    intervened_fips = set(
        int_base.filter(pl.col("intervention_applied"))["fips"].to_list()
    )

    adj = pl.read_csv(
        adjacency_data_path,
        separator="|",
        schema_overrides={
            "County GEOID": pl.String,
            "Neighbor GEOID": pl.String,
        },
    )
    neighbors = set()
    for fip in intervened_fips:
        nbrs = adj.filter(pl.col("County GEOID") == fip)[
            "Neighbor GEOID"
        ].to_list()
        neighbors.update(nbrs)
    neighbor_fips = neighbors - intervened_fips
    plot_fips = intervened_fips | neighbor_fips

    fips_to_idx = {f: i for i, f in enumerate(baseline_fips)}
    imp = importation_samples
    imp_count = importation_count_samples

    # Load population data
    pop_df = (
        pl.read_csv(
            os.path.join(output_dir, "projection_dataset_2026.csv"),
            schema_overrides={"fips": pl.String},
        )
        .with_columns(
            pop=pl.col("pop")
            * (pl.col("fraction_school_aged") + pl.col("fraction_under_5"))
        )
        .select(["fips", "pop"])
    )
    pop_lookup = dict(zip(pop_df["fips"].to_list(), pop_df["pop"].to_list()))

    # Calculate cumulative incidence change across all years
    cumulative_incidence_change = {}

    for fip in plot_fips:
        if fip not in fips_to_idx:
            continue
        if fip not in pop_lookup:
            continue

        b_idx = fips_to_idx[fip]
        pop = pop_lookup[fip]

        cumulative_baseline_incidence = None
        cumulative_intervention_incidence = None

        # Loop through all future years
        for yr in future_years:
            pkg = intervention_data[intervention][yr]
            pred = pkg["predictions"]
            fips_list = pkg["fips"]

            if fip not in fips_list:
                continue

            proj_fips_to_idx = {f: j for j, f in enumerate(fips_list)}
            p_idx = proj_fips_to_idx[fip]

            baseline_samps = np.array(
                baseline_samples_by_year[yr][:, b_idx] / scale
            )
            intervention_samps = np.array(pred[:, p_idx] / scale)

            # Apply constraint only to intervened counties, not neighbors
            if fip in intervened_fips:
                if intervention in BAN_INTERVENTIONS:
                    intervention_samps = np.minimum(
                        intervention_samps, baseline_samps
                    )
                elif intervention in REMOVE_INTERVENTIONS:
                    intervention_samps = np.maximum(
                        intervention_samps, baseline_samps
                    )

            # Apply importation to both
            baseline_samps = (
                baseline_samps * imp[:, b_idx] + imp_count[:, b_idx]
            )
            intervention_samps = (
                intervention_samps * imp[:, b_idx] + imp_count[:, b_idx]
            )

            # Calculate incidence (cases per 100,000 population)
            baseline_incidence = (baseline_samps / pop) * 100000
            intervention_incidence = (intervention_samps / pop) * 100000

            # Accumulate across years
            if cumulative_baseline_incidence is None:
                cumulative_baseline_incidence = baseline_incidence
                cumulative_intervention_incidence = intervention_incidence
            else:
                cumulative_baseline_incidence += baseline_incidence
                cumulative_intervention_incidence += intervention_incidence

        if cumulative_baseline_incidence is not None:
            cumulative_incidence_change[fip] = float(
                np.mean(
                    cumulative_intervention_incidence
                    - cumulative_baseline_incidence
                )
            )

    base_cov_lookup = dict(
        zip(
            base_2026["fips"].to_list(),
            base_2026["estimated_mmr_cov"].to_list(),
        )
    )
    int_cov_lookup = dict(
        zip(
            int_base["fips"].to_list(), int_base["estimated_mmr_cov"].to_list()
        )
    )

    rows = []
    for fip in plot_fips:
        if fip not in cumulative_incidence_change:
            continue
        if fip not in base_cov_lookup:
            continue

        initial_cov = base_cov_lookup[fip]

        if fip in intervened_fips:
            if fip not in int_cov_lookup:
                continue
            cov_change = int_cov_lookup[fip] - initial_cov
            is_intervened = True
        else:
            cov_change = 0.0
            is_intervened = False

        rows.append(
            {
                "fips": fip,
                "initial_cov": initial_cov,
                "cov_change": cov_change,
                "incidence_change": cumulative_incidence_change[fip],
                "is_intervened": is_intervened,
            }
        )

    scatter_df = pd.DataFrame(rows)
    if scatter_df.empty:
        print(f"Warning: no data to plot for intervention {intervention}")
        return

    scatter_df = scatter_df.dropna(subset=["incidence_change"])

    # Get actual data range before clipping
    vmin_original = scatter_df["incidence_change"].min()
    vmax_original = scatter_df["incidence_change"].max()

    print(
        f"Cumulative incidence change range (before clipping): min={vmin_original:.2f}, max={vmax_original:.2f}"
    )

    # Clip values below -100 and above 0
    scatter_df["incidence_change"] = scatter_df["incidence_change"].clip(
        lower=-100, upper=0
    )

    # Set colorbar range from -100 to 0
    vmin = -100
    vmax = 0

    print(
        f"Cumulative incidence change range (after clipping): min={vmin:.2f}, max={vmax:.2f}"
    )
    print(
        f"Median cumulative incidence change: {scatter_df['incidence_change'].median():.2f}"
    )
    print(
        f"Mean cumulative incidence change: {scatter_df['incidence_change'].mean():.2f}"
    )

    # Create a blue-to-grey colormap (no red needed since vmax=0)
    color_positions = [
        0.0,  # -100 (dark blue)
        0.25,  # -75
        0.5,  # -50
        0.75,  # -25
        1.0,  # 0 (grey)
    ]

    colors = [
        "#08519c",  # dark blue (-100)
        "#3182bd",  # medium blue
        "#6baed6",  # light blue
        "#c6dbef",  # very light blue
        "#f0f0f0",  # light grey (0)
    ]

    blue_to_grey = mcolors.LinearSegmentedColormap.from_list(
        "Blue_to_Grey",
        list(zip(color_positions, colors)),
        N=256,
    )

    # Use simple linear normalization from -100 to 0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    neighbors_df = scatter_df[~scatter_df["is_intervened"]]
    intervened_df = scatter_df[scatter_df["is_intervened"]]

    sc = ax.scatter(
        intervened_df["initial_cov"],
        intervened_df["cov_change"],
        c=intervened_df["incidence_change"],
        cmap=blue_to_grey,
        norm=norm,
        s=120,
        alpha=0.5,
        edgecolors="black",
        linewidths=0.5,
        marker="o",
        zorder=2,
    )

    sm = plt.cm.ScalarMappable(cmap=blue_to_grey, norm=norm)
    # Add extend='min' to show arrow at bottom indicating values below -100
    cbar = fig.colorbar(
        sm, ax=ax, shrink=0.7, pad=0.02, orientation="vertical", extend="min"
    )
    cbar.set_label(
        "Change in incidence\n(cases per 100,000)", rotation=270, labelpad=44
    )

    # Create tick positions from -100 to 0
    tick_positions = [-100, -75, -50, -25, 0]
    tick_labels = ["-100", "-75", "-50", "-25", "0"]

    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels)

    ax.set_xlabel("Initial MMR coverage (%)")
    ax.set_ylabel(
        "Change in MMR coverage after removal\nof nonmedical exemptions (%)"
    )

    ax.set_xlim(50, 100)
    ax.set_ylim(0)


def create_supplemental_nme_ban_interventions_figure(
    future_years,
    vaccination_scenarios,
    vaccination_scenario_labels,
    all_scenario_baseline_samples,
    all_scenario_intervention_data,
    importation_samples,
    importation_count_samples,
    baseline_fips,
    scale,
    output_path,
):
    """
    Create supplemental figure showing NME ban intervention effects
    across vaccination scenarios in a 2x2 grid.
    """
    interventions = [
        "ban_top_risk_states",
        "ban_top_nme_states",
        "ban_top_risk_counties",
    ]

    intervention_labels = {
        "ban_top_risk_states": "Top risk states",
        "ban_top_nme_states": "Top NME states",
        "ban_top_risk_counties": "Top risk counties",
    }

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 17))

    # Apply standard styling to all subplots
    for ax_row in axes:
        for ax in ax_row:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            ax.tick_params(axis="x", direction="out")
            ax.tick_params(axis="y", direction="out")
            for spine in ax.spines.values():
                spine.set_position(("outward", 5))
            ax.set_axisbelow(True)

    # Add subplot labels
    labels = ["a", "b", "c", "d"]
    for idx, (ax_row, label_row) in enumerate(
        zip(axes, [labels[:2], labels[2:]])
    ):
        for jdx, (ax, label) in enumerate(zip(ax_row, label_row)):
            ax.text(
                -0.05,
                1.08,
                label,
                transform=ax.transAxes,
                fontsize=24,
                fontweight="bold",
                va="top",
                ha="right",
            )

    # Plot each vaccination scenario
    for idx, scenario in enumerate(vaccination_scenarios):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        plot_intervention_effects(
            interventions=interventions,
            intervention_labels=intervention_labels,
            future_years=future_years,
            baseline_samples_by_year=all_scenario_baseline_samples[scenario],
            intervention_data=all_scenario_intervention_data[scenario],
            importation_samples=importation_samples,
            importation_count_samples=importation_count_samples,
            baseline_fips=baseline_fips,
            scale=scale,
            fig=fig,
            ax=ax,
        )

        # Add scenario title to each subplot
        ax.set_title(
            vaccination_scenario_labels[scenario], fontsize=28, pad=20
        )

    fig.tight_layout()
    fig.savefig(f"{output_path}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{output_path}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(
        f"NME ban interventions figure saved to {output_path}.png and {output_path}.pdf"
    )


def create_supplemental_5pp_improvement_interventions_figure(
    future_years,
    vaccination_scenarios,
    vaccination_scenario_labels,
    all_scenario_baseline_samples,
    all_scenario_intervention_data,
    importation_samples,
    importation_count_samples,
    baseline_fips,
    scale,
    output_path,
):
    """
    Create supplemental figure showing 5pp coverage improvement intervention effects
    across vaccination scenarios in a 2x2 grid.
    """
    interventions = [
        "improve_cases_previously_counties",
        "improve_bottom_decile_5pct",
        "improve_top_risk_states_5pct",
        "improve_top_risk_counties_5pct",
    ]

    intervention_labels = {
        "improve_cases_previously_counties": "Counties with\noutbreaks",
        "improve_bottom_decile_5pct": "Lowest coverage\ncounties",
        "improve_top_risk_states_5pct": "Top risk states",
        "improve_top_risk_counties_5pct": "Top risk counties",
    }

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 17))

    # Apply standard styling to all subplots
    for ax_row in axes:
        for ax in ax_row:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            ax.tick_params(axis="x", direction="out")
            ax.tick_params(axis="y", direction="out")
            for spine in ax.spines.values():
                spine.set_position(("outward", 5))
            ax.set_axisbelow(True)

    # Add subplot labels
    labels = ["a", "b", "c", "d"]
    for idx, (ax_row, label_row) in enumerate(
        zip(axes, [labels[:2], labels[2:]])
    ):
        for jdx, (ax, label) in enumerate(zip(ax_row, label_row)):
            ax.text(
                -0.05,
                1.08,
                label,
                transform=ax.transAxes,
                fontsize=24,
                fontweight="bold",
                va="top",
                ha="right",
            )

    # Plot each vaccination scenario
    for idx, scenario in enumerate(vaccination_scenarios):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        plot_intervention_effects(
            interventions=interventions,
            intervention_labels=intervention_labels,
            future_years=future_years,
            baseline_samples_by_year=all_scenario_baseline_samples[scenario],
            intervention_data=all_scenario_intervention_data[scenario],
            importation_samples=importation_samples,
            importation_count_samples=importation_count_samples,
            baseline_fips=baseline_fips,
            scale=scale,
            fig=fig,
            ax=ax,
        )

        # Add scenario title to each subplot
        ax.set_title(
            vaccination_scenario_labels[scenario], fontsize=28, pad=20
        )

    fig.tight_layout()
    fig.savefig(f"{output_path}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{output_path}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(
        f"5pp improvement interventions figure saved to {output_path}.png and {output_path}.pdf"
    )


def create_supplemental_10pp_improvement_interventions_figure(
    future_years,
    vaccination_scenarios,
    vaccination_scenario_labels,
    all_scenario_baseline_samples,
    all_scenario_intervention_data,
    importation_samples,
    importation_count_samples,
    baseline_fips,
    scale,
    output_path,
):
    """
    Create supplemental figure showing 10pp coverage improvement intervention effects
    across vaccination scenarios in a 2x2 grid.
    """
    interventions = [
        "improve_bottom_decile_10pct",
        "improve_top_risk_states_10pct",
        "improve_top_risk_counties_10pct",
    ]

    intervention_labels = {
        "improve_bottom_decile_10pct": "Lowest coverage\ncounties",
        "improve_top_risk_states_10pct": "Top risk states",
        "improve_top_risk_counties_10pct": "Top risk counties",
    }

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 17))

    # Apply standard styling to all subplots
    for ax_row in axes:
        for ax in ax_row:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            ax.tick_params(axis="x", direction="out")
            ax.tick_params(axis="y", direction="out")
            for spine in ax.spines.values():
                spine.set_position(("outward", 5))
            ax.set_axisbelow(True)

    # Add subplot labels
    labels = ["a", "b", "c", "d"]
    for idx, (ax_row, label_row) in enumerate(
        zip(axes, [labels[:2], labels[2:]])
    ):
        for jdx, (ax, label) in enumerate(zip(ax_row, label_row)):
            ax.text(
                -0.05,
                1.08,
                label,
                transform=ax.transAxes,
                fontsize=24,
                fontweight="bold",
                va="top",
                ha="right",
            )

    # Plot each vaccination scenario
    for idx, scenario in enumerate(vaccination_scenarios):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        plot_intervention_effects(
            interventions=interventions,
            intervention_labels=intervention_labels,
            future_years=future_years,
            baseline_samples_by_year=all_scenario_baseline_samples[scenario],
            intervention_data=all_scenario_intervention_data[scenario],
            importation_samples=importation_samples,
            importation_count_samples=importation_count_samples,
            baseline_fips=baseline_fips,
            scale=scale,
            fig=fig,
            ax=ax,
        )

        # Add scenario title to each subplot
        ax.set_title(
            vaccination_scenario_labels[scenario], fontsize=28, pad=20
        )

    fig.tight_layout()
    fig.savefig(f"{output_path}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{output_path}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(
        f"10pp improvement interventions figure saved to {output_path}.png and {output_path}.pdf"
    )


if __name__ == "__main__":
    # Create figure with 2 rows: top row has 2 subplots, bottom row has 1 wide subplot
    fig = plt.figure(figsize=(20, 17))

    # Create GridSpec for custom layout
    import matplotlib.gridspec as gridspec

    gs = gridspec.GridSpec(2, 2, figure=fig)

    # Top row: two subplots
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # Bottom row: one wide subplot spanning both columns
    ax_c = fig.add_subplot(gs[1, :])

    # Apply styling to all axes
    for ax in [ax_a, ax_b, ax_c]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="out")
        for spine in ax.spines.values():
            spine.set_position(("outward", 5))
        ax.set_axisbelow(True)

    # -------------------------------------------------------------------------
    # Subplot a: map of NME rates
    # -------------------------------------------------------------------------
    ax_a.text(
        -0.05,
        1.28,
        "a",
        transform=ax_a.transAxes,
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )

    # load the NME data for the map — using the assembled projection results
    # which has the latest NME values per county
    df = pl.read_csv(
        os.path.join(output_dir, "projection_dataset_2026.csv"),
        schema_overrides={"fips": pl.String},
    )

    print(df["nonmed_exempt_pct"].describe())
    print(
        df.filter(
            ~pl.col("state").is_in(
                ["California", "Connecticut", "Maine", "New York"]
            )
        )["nonmed_exempt_pct"].describe()
    )

    map_fips_code_data(
        df["fips"],
        df["nonmed_exempt_pct"],
        fig,
        ax_a,
        cbar_label="Nonmedical exemption rate (%)",
        cbar_type="exemptions",
        point_data=None,
    )

    # -------------------------------------------------------------------------
    # Subplot b: coverage change vs NME policy
    # -------------------------------------------------------------------------
    ax_b.text(
        -0.05,
        1.08,
        "b",
        transform=ax_b.transAxes,
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
    regression_df = coverage_increase_nme_policy_change(
        vac_df,
        fig,
        ax_b,
    )
    regression_df.write_csv(
        os.path.join(
            fig_dir,
            "table_supplemental_mmr_coverage_change_nme_policy_interventions.csv",
        )
    )

    # -------------------------------------------------------------------------
    # Load shared data for subplot c
    # -------------------------------------------------------------------------
    with open(os.path.join(output_dir, "scale_2025.pkl"), "rb") as f:
        scale_2025 = pickle.load(f)

    (
        importation_samples_future,
        importation_count_samples_future,
        baseline_fips,
    ) = load_importation_samples_future(output_dir)

    baseline_samples_by_year = {}
    for year in FUTURE_YEARS:
        with open(
            os.path.join(output_dir, f"risk_projections_baseline_{year}.pkl"),
            "rb",
        ) as f:
            baseline_samples_by_year[year] = pickle.load(f)

    intervention_data = {}
    for intervention in INTERVENTIONS:
        intervention_data[intervention] = {}
        for year in FUTURE_YEARS:
            with open(
                os.path.join(
                    output_dir,
                    f"risk_projections_{intervention}_baseline_{year}.pkl",
                ),
                "rb",
            ) as f:
                intervention_data[intervention][year] = pickle.load(f)

    # -------------------------------------------------------------------------
    # Subplot c: intervention effects (now spanning full width)
    # -------------------------------------------------------------------------
    ax_c.text(
        -0.05,
        1.08,
        "c",
        transform=ax_c.transAxes,
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )

    with open(
        os.path.join(
            output_dir,
            "import_scaler_2025.pkl",
        ),
        "rb",
    ) as f:
        import_scaler_2025 = pickle.load(f)

    plot_intervention_effects(
        interventions=INTERVENTIONS,
        intervention_labels=INTERVENTION_LABELS,
        future_years=FUTURE_YEARS,
        baseline_samples_by_year=baseline_samples_by_year,
        intervention_data=intervention_data,
        importation_samples=jnp.minimum(
            importation_samples_future * import_scaler_2025, 1
        ),
        importation_count_samples=(
            importation_count_samples_future * import_scaler_2025
        ),
        baseline_fips=baseline_fips,
        scale=scale_2025,
        fig=fig,
        ax=ax_c,
    )

    fig.tight_layout()
    fig.savefig(
        os.path.join(
            fig_dir,
            "figure_nonmedical_exemption_policy_interventions.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(
            fig_dir,
            "figure_nonmedical_exemption_policy_interventions.pdf",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    # Define the 4 vaccination scenarios
    vaccination_scenarios_to_plot = [
        "targeted_worsening",
        "universal_worsening",
        "targeted_improvement",
    ]

    vaccination_scenario_labels = {
        "targeted_worsening": "Targeted worsening",
        "universal_worsening": "Universal worsening",
        "targeted_improvement": "Targeted improvement",
        "universal_improvement": "Universal improvement",
    }

    # Load baseline samples for each vaccination scenario
    all_scenario_baseline_samples = {}
    for vac_scenario in vaccination_scenarios_to_plot:
        all_scenario_baseline_samples[vac_scenario] = {}
        for year in FUTURE_YEARS:
            with open(
                os.path.join(
                    output_dir, f"risk_projections_{vac_scenario}_{year}.pkl"
                ),
                "rb",
            ) as f:
                all_scenario_baseline_samples[vac_scenario][year] = (
                    pickle.load(f)
                )

    # Load ALL intervention data for each vaccination scenario
    all_interventions = [
        "ban_top_risk_states",
        "ban_top_nme_states",
        "ban_top_risk_counties",
        "improve_cases_previously_counties",
        "improve_bottom_decile_5pct",
        "improve_bottom_decile_10pct",
        "improve_top_risk_states_5pct",
        "improve_top_risk_states_10pct",
        "improve_top_risk_counties_5pct",
        "improve_top_risk_counties_10pct",
    ]

    all_scenario_intervention_data = {}
    for vac_scenario in vaccination_scenarios_to_plot:
        all_scenario_intervention_data[vac_scenario] = {}
        for intervention in all_interventions:
            all_scenario_intervention_data[vac_scenario][intervention] = {}
            for year in FUTURE_YEARS:
                try:
                    with open(
                        os.path.join(
                            output_dir,
                            f"risk_projections_{intervention}_{vac_scenario}_{year}.pkl",
                        ),
                        "rb",
                    ) as f:
                        all_scenario_intervention_data[vac_scenario][
                            intervention
                        ][year] = pickle.load(f)
                except FileNotFoundError:
                    print(
                        f"Warning: Could not find data for {intervention} in {vac_scenario} {year}"
                    )

    # Create three supplemental figures
    create_supplemental_nme_ban_interventions_figure(
        future_years=FUTURE_YEARS,
        vaccination_scenarios=vaccination_scenarios_to_plot,
        vaccination_scenario_labels=vaccination_scenario_labels,
        all_scenario_baseline_samples=all_scenario_baseline_samples,
        all_scenario_intervention_data=all_scenario_intervention_data,
        importation_samples=jnp.minimum(
            importation_samples_future * import_scaler_2025, 1
        ),
        importation_count_samples=(
            importation_count_samples_future * import_scaler_2025
        ),
        baseline_fips=baseline_fips,
        scale=scale_2025,
        output_path=os.path.join(
            fig_dir,
            "figure_supplemental_nme_ban_interventions",
        ),
    )

    create_supplemental_5pp_improvement_interventions_figure(
        future_years=FUTURE_YEARS,
        vaccination_scenarios=vaccination_scenarios_to_plot,
        vaccination_scenario_labels=vaccination_scenario_labels,
        all_scenario_baseline_samples=all_scenario_baseline_samples,
        all_scenario_intervention_data=all_scenario_intervention_data,
        importation_samples=jnp.minimum(
            importation_samples_future * import_scaler_2025, 1
        ),
        importation_count_samples=(
            importation_count_samples_future * import_scaler_2025
        ),
        baseline_fips=baseline_fips,
        scale=scale_2025,
        output_path=os.path.join(
            fig_dir,
            "figure_supplemental_5pp_improvement_interventions",
        ),
    )

    create_supplemental_10pp_improvement_interventions_figure(
        future_years=FUTURE_YEARS,
        vaccination_scenarios=vaccination_scenarios_to_plot,
        vaccination_scenario_labels=vaccination_scenario_labels,
        all_scenario_baseline_samples=all_scenario_baseline_samples,
        all_scenario_intervention_data=all_scenario_intervention_data,
        importation_samples=jnp.minimum(
            importation_samples_future * import_scaler_2025, 1
        ),
        importation_count_samples=(
            importation_count_samples_future * import_scaler_2025
        ),
        baseline_fips=baseline_fips,
        scale=scale_2025,
        output_path=os.path.join(
            fig_dir,
            "figure_supplemental_10pp_improvement_interventions",
        ),
    )
