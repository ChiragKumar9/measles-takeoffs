import os

from numpyro import enable_x64

enable_x64()
import pickle

import geopandas as gpd
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import polars as pl
import yaml
from matplotlib import rc
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy import stats
from shapely.geometry import MultiPolygon, Polygon
from sklearn.linear_model import TheilSenRegressor

from outbreak_size import scenario_tables
from outbreak_size.doses import kindergarten_cohorts
from outbreak_size.importation import (
    future_importation_draws,
    scale_importation_probability,
)
from outbreak_size.interventions import (
    build_projection_rows,
    coverage_change_fit,
    state_coverage_changes,
)
from outbreak_size.samples_io import load_samples
from outbreak_size.scenarios import VACCINATION_SCENARIO_LABELS
from outbreak_size.utils import CENSUS_CACHE_PATH, CensusCache
from outbreak_size.validation import require_same_fips, require_width

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)
fig_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")

font = {"family": "Nimbus Roman", "size": 28}
rc("font", **font)

with open("params.yml", "r") as f:
    params = yaml.safe_load(f)

GLOBALS_PRIOR_SD = params["validation_globals_prior_sd"]

FUTURE_YEARS = [2027, 2028, 2029, 2030]
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

INTERVENTION_TABLE_LABELS = {
    "ban_top_risk_states": "Remove NMEs from top risk states",
    "ban_top_nme_states": "Remove NMEs from top NME states",
    "ban_top_risk_counties": "Remove NMEs from top risk counties",
    "improve_cases_previously_counties": "Increase vaccination coverage by 5pp in counties with previous outbreaks",
    "improve_bottom_decile_5pct": "Increase vaccination coverage by 5pp in lowest coverage counties",
    "improve_bottom_decile_10pct": "Increase vaccination coverage by 10pp in lowest coverage counties",
    "improve_top_risk_states_5pct": "Increase vaccination coverage by 5pp in top risk states",
    "improve_top_risk_states_10pct": "Increase vaccination coverage by 10pp in top risk states",
    "improve_top_risk_counties_5pct": "Increase vaccination coverage by 5pp in top risk counties",
    "improve_top_risk_counties_10pct": "Increase vaccination coverage by 10pp in top risk counties",
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

    # geopandas cannot plot an empty frame (every county has a value)
    if bool(no_data_mask.any()):
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
            # midpoint of each bucket (the bar spaces buckets evenly and
            # interpolates within them), so every label is centered on its
            # color block
            cbar.set_ticks([0.75, 3.75, 7.5, 12.5, 17.5, 60])
            cbar.set_ticklabels(
                ["< 2.5%", "2.5%-5%", "5%-10%", "10%-15%", "15%-20%", "≥20%"]
            )
            cbar.ax.tick_params(labelsize=24, rotation=-60, length=0)

    states.boundary.plot(ax=ax, linewidth=1.0, edgecolor="black", alpha=0.8)

    if point_data is not None:
        counties_with_data["centroid"] = counties_with_data.geometry.centroid

    point_df = pd.DataFrame(
        point_data, columns=pd.Index(["GEOID", "point_value"])
    )
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
        # each county's coverage change after the ban (the same function
        # that 10_nme_policy_scenarios.py uses for the ban interventions)
        temp = state_coverage_changes(df, state, year)

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
        print(f"  Policy year change: {policy_change:.2f} pp")

        # 1 year after change
        yr1_change = (
            state_df["estimated_mmr_cov_change_1yr"].drop_nulls().mean()
        )
        print(f"  1 year after: {yr1_change:.2f} pp")

    result_df = pl.concat(all_dfs)

    print("\n" + "=" * 60)
    print("Average MMR Coverage Changes Across All Counties:")
    print("=" * 60)

    overall_policy_change = (
        result_df["estimated_mmr_cov_change_policy_year"].drop_nulls().mean()
    )
    print(f"  Policy year change: {overall_policy_change:.2f} pp")

    overall_yr1_change = (
        result_df["estimated_mmr_cov_change_1yr"].drop_nulls().mean()
    )
    print(f"  1 year after: {overall_yr1_change:.2f} pp")

    overall_yr2_change = (
        result_df["estimated_mmr_cov_change_2yr"].drop_nulls().mean()
    )
    print(f"  2 years after: {overall_yr2_change:.2f} pp")

    # Pre-policy NME rates
    print("\n" + "=" * 60)
    print("Pre-Policy Nonmedical Exemption Rates:")
    print("=" * 60)

    for state, year in states.items():
        state_df = all_dfs[list(states.keys()).index(state)]

        # Pre-policy NME rate (from year before policy)
        pre_policy_nme = state_df["nonmed_exempt_pct"].drop_nulls().mean()
        print(f"\n{state} (Pre-policy year: {year - 1}):")
        print(f"  Average NME rate: {pre_policy_nme:.2f}%")

    # Overall pre-policy NME rate across all counties
    overall_pre_policy_nme = result_df["nonmed_exempt_pct"].drop_nulls().mean()
    print("\nAcross all counties in states with policies:")
    print(f"  Average pre-policy NME rate: {overall_pre_policy_nme:.2f}%")

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
        "Change in MMR coverage after removal of\nnonmedical exemptions (pp)"
    )
    ax.set_xlim(0)

    for year_mode in ["policy_year", "1yr", "2yr"]:
        temp_regression = result_df[
            ["nonmed_exempt_pct", f"estimated_mmr_cov_change_{year_mode}"]
        ].drop_nulls()
        # the pooled line; 10_nme_policy_scenarios.py fits the "1yr" one
        # with the same function to set the coverage gain of a ban
        estimator = coverage_change_fit(
            result_df, f"estimated_mmr_cov_change_{year_mode}"
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
        os.path.join(output_dir, "projection_dataset_baseline_2027.csv"),
        schema_overrides={"fips": pl.String},
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
    probability, counts = future_importation_draws(import_df, future_base)
    return (
        jnp.array(probability),
        jnp.array(counts),
        future_base["fips"].to_list(),
    )


def plot_intervention_effects(
    interventions,
    intervention_labels,
    future_years,
    load_baseline,
    load_intervention,
    importation_samples,
    importation_count_samples,
    baseline_fips,
    fig,
    ax,
):
    # the draws are loaded one at a time (load_baseline(year) and
    # load_intervention(intervention, year)): the ones of all the interventions
    # and years together are several GB
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
        first_pkg = load_intervention(intervention, future_years[0])
        first_fips_list = first_pkg["fips"]
        del first_pkg

        proj_fips_to_idx = {f: j for j, f in enumerate(first_fips_list)}
        common_fips = [f for f in baseline_fips if f in proj_fips_to_idx]
        baseline_col_idx = jnp.array([fips_to_idx[f] for f in common_fips])
        proj_col_idx = jnp.array([proj_fips_to_idx[f] for f in common_fips])

        imp = importation_samples
        imp_count = importation_count_samples

        cumulative_total_by_county = None
        cumulative_baseline_by_county = None

        for y_idx, year in enumerate(future_years):
            pkg = load_intervention(intervention, year)
            require_same_fips(
                pkg["fips"], first_fips_list, f"{intervention} {year}"
            )
            pred = pkg["predictions"]

            # --- Total: (n_samples, n_common_counties) ---
            b_county = (
                load_baseline(year)[:, baseline_col_idx]
                * imp[:, baseline_col_idx]
                + imp_count[:, baseline_col_idx]
            )
            i_county = (
                pred[:, proj_col_idx] * imp[:, baseline_col_idx]
                + imp_count[:, baseline_col_idx]
            )
            # NOTE: no intervention bound here (INTERVENTION BOUND in
            # SCENARIO BOUNDS at the top of analysis/figure_validation_and_
            # risk_projections.py): i_county used to be capped at b_county
            # with jnp.minimum

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
            base_values = pl.read_csv(
                os.path.join(
                    output_dir, "projection_dataset_baseline_2027.csv"
                ),
                schema_overrides={"fips": pl.String},
            )

            # Join to get baseline coverage
            coverage_comparison = (
                int_base.join(
                    base_values.select(
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

    # Build legend handles and labels safely
    combined_handles = [dummy]  # Start with header
    combined_labels = ["5pp coverage increase in:"]
    spaces = "           "

    # Add alt_items (coverage improvements)
    for h, label in alt_items:
        combined_handles.append(h)
        combined_labels.append(spaces + label)

    # Add padding if needed to align columns
    if len(alt_items) < len(nme_items):
        for _ in range(len(nme_items) - len(alt_items)):
            combined_handles.append(dummy)
            combined_labels.append("")

    # Add NME header
    combined_handles.append(dummy)
    combined_labels.append("Remove nonmedical exemptions in:")

    # Add nme_items
    for h, label in nme_items:
        combined_handles.append(h)
        combined_labels.append(spaces + label)

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


def create_supplemental_nme_ban_interventions_figure(
    future_years,
    vaccination_scenarios,
    vaccination_scenario_labels,
    load_baseline,
    load_intervention,
    importation_samples,
    importation_count_samples,
    baseline_fips,
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
    fig, axes = plt.subplots(
        2, 2, figsize=(24, 20)
    )  # Increased width for better spacing

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
                -0.08,
                1.05,
                label,
                transform=ax.transAxes,
                fontsize=28,
                fontweight="bold",
                va="top",
                ha="right",
            )

    # Plot each scenario, with a single shared legend added below
    for idx, scenario in enumerate(vaccination_scenarios):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        # Temporarily disable legend for plot_intervention_effects
        # We'll add a single shared legend later
        plot_intervention_effects(
            interventions=interventions,
            intervention_labels=intervention_labels,
            future_years=future_years,
            load_baseline=lambda year: load_baseline(scenario, year),
            load_intervention=lambda intervention, year: load_intervention(
                intervention, scenario, year
            ),
            importation_samples=importation_samples,
            importation_count_samples=importation_count_samples,
            baseline_fips=baseline_fips,
            fig=fig,
            ax=ax,
        )

        # Remove the legend created by plot_intervention_effects
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

        # Add scenario title to each subplot
        ax.set_title(
            vaccination_scenario_labels[scenario],
            fontsize=32,
            pad=15,
            fontweight="bold",
        )

        # Only show x-label on bottom row
        if row == 0:
            ax.set_xlabel("")

        # Only show y-label on left column
        if col == 1:
            ax.set_ylabel("")

    # Create a single shared legend in the bottom right subplot
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

    # Create legend elements
    INTERVENTION_COLORS = {
        "ban_top_risk_states": DIRECT_COLOR,
        "ban_top_nme_states": TOTAL_COLOR,
        "ban_top_risk_counties": "#8B5CF6",
    }

    dummy = Rectangle(
        (0, 0), 1, 1, fc="white", ec="white", linewidth=0, alpha=0
    )

    handles = [dummy]
    labels_list = ["Remove nonmedical\nexemptions in:"]

    for intervention in interventions:
        color = INTERVENTION_COLORS[intervention]
        # Create a rectangle patch for the violin plot color
        handle = Rectangle((0, 0), 1, 1, fc=color, ec=color, alpha=0.5)
        handles.append(handle)
        labels_list.append("  " + intervention_labels[intervention])

    # Add legend to bottom right subplot only
    axes[1, 1].legend(
        handles=handles,
        labels=labels_list,
        handler_map={dummy: HeaderHandler()},
        loc="lower right",
        edgecolor="black",
        frameon=True,
        fontsize=24,
        handlelength=1.5,
        handleheight=1.5,
    )

    fig.tight_layout()
    fig.savefig(f"{output_path}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{output_path}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(
        f"NME ban interventions figure saved to {output_path}.png and {output_path}.pdf"
    )


def _draw_loaders(output_dir, baseline_fips):
    """The loaders of the baseline draws of a scenario and year (one column
    per county of `baseline_fips`) and of an intervention's draws (its fips and
    predictions), from the files in `output_dir`."""

    def load_baseline(vac_scenario, year):
        return require_width(
            load_samples(
                os.path.join(
                    output_dir,
                    f"risk_projections_{vac_scenario}_{year}_globals_prior_sd={GLOBALS_PRIOR_SD}.npz",
                )
            ),
            len(baseline_fips),
            "baseline draws",
        )

    def load_intervention(intervention, vac_scenario, year):
        return load_samples(
            os.path.join(
                output_dir,
                f"risk_projections_{intervention}_{vac_scenario}_{year}.npz",
            )
        )

    return load_baseline, load_intervention


LOCATIONS_YEAR = 2027


def _locations(output_dir, columns):
    """The state, county and population of each county, from the 2027
    projection dataset."""
    return (
        pl.read_csv(
            os.path.join(
                output_dir, f"projection_dataset_baseline_{LOCATIONS_YEAR}.csv"
            ),
            schema_overrides={"fips": pl.String},
        )
        .select(columns)
        .unique()
    )


def _county_locations(output_dir):
    """The state, county, population and kindergarten cohort of each county:
    the cohort is a fifth of the population aged 5-9 of the Census vintage of
    the population (outbreak_size.doses), read from the saved Census answers
    (outputs/census_cache.sqlite, a required input of this table)."""
    locations = _locations(output_dir, ["fips", "state", "county", "pop"])
    cohorts = kindergarten_cohorts(
        locations["fips"].to_list(),
        locations["pop"].to_list(),
        LOCATIONS_YEAR,
        CensusCache(CENSUS_CACHE_PATH),
    )
    return locations.with_columns(kindergarten_cohort=pl.Series(cohorts))


def _coverage_change_loader(output_dir):
    """`load(intervention, scenario, year)`: {fips: coverage of the
    intervention minus the scenario's baseline coverage (percentage points)},
    from the same projection rows that script 11 gave the model
    (build_projection_rows of the scenario's projection dataset and the
    intervention base). Each file is read once."""
    projections, bases = {}, {}

    def load(intervention, scenario, year):
        if (scenario, year) not in projections:
            projections[(scenario, year)] = pl.read_csv(
                os.path.join(
                    output_dir, f"projection_dataset_{scenario}_{year}.csv"
                ),
                schema_overrides={"fips": pl.String},
            )
        if intervention not in bases:
            bases[intervention] = pl.read_csv(
                os.path.join(
                    output_dir, f"intervention_base_{intervention}.csv"
                ),
                schema_overrides={"fips": pl.String},
            )
        baseline = projections[(scenario, year)]
        rows = build_projection_rows(
            baseline, bases[intervention], intervention
        )
        change = rows["estimated_mmr_cov"] - baseline["estimated_mmr_cov"]
        return dict(zip(baseline["fips"].to_list(), change.to_list()))

    return load


def create_intervention_effects_table(
    all_interventions,
    vaccination_scenarios,
    future_years,
    output_dir,
    importation_samples,
    importation_count_samples,
    baseline_fips,
    output_path,
):
    """
    Create a supplemental table with % change in outbreak size by county
    for each intervention, vaccination scenario, and year (one row per
    county-year-scenario-intervention combination), and save it.
    RAM-efficient: loads the files one at a time.
    """
    load_baseline, load_intervention = _draw_loaders(output_dir, baseline_fips)
    results_df = scenario_tables.county_intervention_effects(
        all_interventions,
        vaccination_scenarios,
        future_years,
        load_baseline,
        load_intervention,
        _county_locations(output_dir),
        importation_samples,
        importation_count_samples,
        baseline_fips,
        VACCINATION_SCENARIO_LABELS,
        INTERVENTION_TABLE_LABELS,
        _coverage_change_loader(output_dir),
    )
    results_df.write_csv(output_path)
    print(f"\nIntervention effects table saved to {output_path}")
    print(f"Total rows: {len(results_df)}")

    return results_df


def create_state_intervention_effects_table(
    all_interventions,
    vaccination_scenarios,
    future_years,
    output_dir,
    importation_samples,
    importation_count_samples,
    baseline_fips,
    output_path,
):
    """
    Create state-level intervention effects table and save it.

    RAM-efficient: processes one scenario-intervention-year at a time,
    aggregates counties to states, then discards samples.
    """
    load_baseline, load_intervention = _draw_loaders(output_dir, baseline_fips)
    results_df = scenario_tables.state_intervention_effects(
        all_interventions,
        vaccination_scenarios,
        future_years,
        load_baseline,
        load_intervention,
        _county_locations(output_dir),
        importation_samples,
        importation_count_samples,
        baseline_fips,
        VACCINATION_SCENARIO_LABELS,
        INTERVENTION_TABLE_LABELS,
        _coverage_change_loader(output_dir),
    )
    results_df.write_csv(output_path)
    print(f"\nState-level intervention effects table saved to {output_path}")
    print(f"Total rows: {len(results_df)}")

    return results_df


def create_nationwide_summary_table(
    all_interventions,
    vaccination_scenarios,
    future_years,
    output_dir,
    importation_samples,
    importation_count_samples,
    baseline_fips,
    output_path,
):
    """
    Create nationwide summary table with total cases across all counties and
    save it: one row per (vaccination_scenario, intervention, year)
    combination with mean, median, and credible intervals for both single-year
    and cumulative nationwide cases.
    """
    load_baseline, load_intervention = _draw_loaders(output_dir, baseline_fips)
    results_df = scenario_tables.nationwide_summary(
        all_interventions,
        vaccination_scenarios,
        future_years,
        load_baseline,
        load_intervention,
        importation_samples,
        importation_count_samples,
        baseline_fips,
        VACCINATION_SCENARIO_LABELS,
        INTERVENTION_TABLE_LABELS,
    )
    results_df.write_csv(output_path)
    print(f"\nNationwide summary table saved to {output_path}")
    print(f"Total rows: {len(results_df)}")

    return results_df


if __name__ == "__main__":
    # Create figure with 2 rows: top row has 2 subplots, bottom row has 1 wide subplot
    fig = plt.figure(figsize=(20, 17))

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

    # load the NME data for the map: the latest observed value of each county
    # (a county with no value of its own gets its state's), not the value
    # brought forward to 2026 that the model uses
    df = pl.read_csv(
        os.path.join(data_dir, "2026_vaccination_data.csv"),
        schema_overrides={"fips": pl.String},
    ).filter(pl.col("fips").str.len_chars() == 5)

    print(df["nonmed_exempt_pct_observed"].describe())
    print(
        df.filter(
            ~pl.col("state").is_in(
                ["California", "Connecticut", "Maine", "New York"]
            )
        )["nonmed_exempt_pct_observed"].describe()
    )

    map_fips_code_data(
        df["fips"],
        df["nonmed_exempt_pct_observed"],
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
    # the state values the imputation filled in are left out of the ban estimate
    vac_df = pl.read_csv(
        os.path.join(data_dir, "processed_vaccinations_data.csv"),
        schema_overrides={
            "fips": pl.String,
            "estimated_mmr_cov": pl.Float64,
            "nonmed_exempt_pct": pl.Float64,
        },
    ).filter(~pl.col("_is_imputed"))
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
    (
        importation_probability,
        importation_counts,
        baseline_fips,
    ) = load_importation_samples_future(output_dir)
    # conditioned on the level of 2025 once, here: every figure and table below
    # uses these two arrays
    with open(os.path.join(output_dir, "import_scaler_2025.pkl"), "rb") as f:
        import_scaler_2025 = pickle.load(f)
    importation_samples_scaled = scale_importation_probability(
        importation_probability, import_scaler_2025
    )
    importation_count_samples_scaled = importation_counts * import_scaler_2025
    del importation_probability, importation_counts

    # the draws are loaded one at a time where they are used (several GB
    # together): the baseline of a scenario and year, and the draws of an
    # intervention
    load_baseline, load_intervention = _draw_loaders(output_dir, baseline_fips)

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

    plot_intervention_effects(
        interventions=INTERVENTIONS,
        intervention_labels=INTERVENTION_LABELS,
        future_years=FUTURE_YEARS,
        load_baseline=lambda year: load_baseline("baseline", year),
        load_intervention=lambda intervention, year: load_intervention(
            intervention, "baseline", year
        ),
        importation_samples=importation_samples_scaled,
        importation_count_samples=importation_count_samples_scaled,
        baseline_fips=baseline_fips,
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
        "no_change",
    ]

    create_supplemental_nme_ban_interventions_figure(
        future_years=FUTURE_YEARS,
        vaccination_scenarios=vaccination_scenarios_to_plot,
        vaccination_scenario_labels=VACCINATION_SCENARIO_LABELS,
        load_baseline=load_baseline,
        load_intervention=load_intervention,
        importation_samples=importation_samples_scaled,
        importation_count_samples=importation_count_samples_scaled,
        baseline_fips=baseline_fips,
        output_path=os.path.join(
            fig_dir,
            "figure_supplemental_nme_ban_interventions",
        ),
    )

    # Create supplemental table with intervention effects by county
    interventions_for_table = [
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

    intervention_effects_table = create_intervention_effects_table(
        all_interventions=interventions_for_table,
        vaccination_scenarios=[*vaccination_scenarios_to_plot, "baseline"],
        future_years=FUTURE_YEARS,
        output_dir=output_dir,  # Changed: pass output_dir instead of loaded data
        importation_samples=importation_samples_scaled,
        importation_count_samples=importation_count_samples_scaled,
        baseline_fips=baseline_fips,
        output_path=os.path.join(
            fig_dir,
            "supplemental_data_s5a_intervention_effects_by_county.csv",
        ),
    )

    state_intervention_effects_table = create_state_intervention_effects_table(
        all_interventions=interventions_for_table,
        vaccination_scenarios=[*vaccination_scenarios_to_plot, "baseline"],
        future_years=FUTURE_YEARS,
        output_dir=output_dir,
        importation_samples=importation_samples_scaled,
        importation_count_samples=importation_count_samples_scaled,
        baseline_fips=baseline_fips,
        output_path=os.path.join(
            fig_dir,
            "supplemental_data_s5b_intervention_effects_by_state.csv",
        ),
    )

    # Create nationwide summary table
    nationwide_summary = create_nationwide_summary_table(
        all_interventions=interventions_for_table,
        vaccination_scenarios=[*vaccination_scenarios_to_plot, "baseline"],
        future_years=FUTURE_YEARS,
        output_dir=output_dir,
        importation_samples=importation_samples_scaled,
        importation_count_samples=importation_count_samples_scaled,
        baseline_fips=baseline_fips,
        output_path=os.path.join(
            fig_dir,
            "supplemental_data_s4_nationwide_vaccine_scenarios_interventions_summary.csv",
        ),
    )
