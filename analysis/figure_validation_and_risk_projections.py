import matplotlib.lines as mlines
from numpyro import enable_x64

enable_x64()

import os
import pickle

import geopandas as gpd
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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

GLOBALS_PRIOR_SD = 1

# read analysis params yaml
with open("params.yml", "r") as f:
    params = yaml.safe_load(f)

VACCINATION_SCENARIO_LABELS = {
    "no_change": "No changes in vaccination",
    "baseline": "Vaccination trends continue",
    "targeted_worsening": "Vaccinations decline 2x faster",
    "universal_worsening": "Vaccinations decline 1.5x faster",
    "targeted_improvement": "Universal 95% coverage by 2030",
    "universal_improvement": "1pp improvement year-over-year",
}


def mean_percentile_rank(samples_2d: np.ndarray) -> np.ndarray:
    """
    For each posterior draw, rank units, shift to 0-based, convert to percentiles,
    then average percentiles across draws.

    Parameters
    ----------
    samples_2d : np.ndarray, shape (n_draws, n_units)

    Returns
    -------
    np.ndarray, shape (n_units,), values in [0, 100]
    """
    from scipy.stats import rankdata

    percentile_method = "average"

    # rank each draw across units; shape stays (n_draws, n_units)
    ranked = np.apply_along_axis(
        lambda row: rankdata(row, method=percentile_method),
        axis=1,
        arr=samples_2d,
    )
    # shift to 0-based so the lowest unit gets exactly 0
    ranked = ranked - 1
    # divide each draw by that draw's max rank → percentiles in [0, 1]
    max_per_draw = ranked.max(axis=1, keepdims=True)  # shape (n_draws, 1)
    percentiles = ranked / max_per_draw  # shape (n_draws, n_units)
    # average across draws, scale to [0, 100]
    return percentiles.mean(axis=0) * 100


def plot_total_cases_over_time(df_observed, all_counties, samples, fig, ax):
    years = [2021, 2022, 2023, 2024, 2025]

    offset = 0.0
    widths = 0.6

    for idx, year in enumerate(years):
        mask = df_observed["validation_year"] == year
        if year == 2025:
            true_without_sc = (
                df_observed.filter(pl.col("state") != "South Carolina")
                .filter(pl.col("validation_year") == year)["cases"]
                .sum()
            )
            # South Carolina had 333 cases in 2025 according to CDC
            true = true_without_sc + 333
        else:
            true = df_observed.filter(pl.col("validation_year") == year)[
                "cases"
            ].sum()
        mask = all_counties["validation_year"] == year
        samples_year = samples[:, mask.to_numpy()]
        samples_year = jnp.sum(samples_year, axis=1)
        lower = jnp.percentile(samples_year, 2.5)
        upper = jnp.percentile(samples_year, 97.5)
        samples_year_vln = samples_year[
            jnp.logical_and(samples_year >= lower, samples_year <= upper)
        ]
        vln1 = ax.violinplot(
            samples_year_vln,
            positions=[idx - offset],
            widths=widths,
            showextrema=False,
        )
        for pc in vln1["bodies"]:
            pc.set_facecolor("green")
            pc.set_edgecolor("green")
            pc.set_alpha(0.3)

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
            capsize=5,
            capthick=2,
        )

        ax.scatter(
            idx,
            true,
            color="k",
            marker="*",
            s=200,
            zorder=10,
            label="Observed" if idx == 0 else "",
        )

    ax.legend(
        [
            vln1["bodies"][0],  # type: ignore
            mlines.Line2D(
                [], [], color="k", marker="*", linestyle="", markersize=15
            ),
        ],
        ["Validation-\npredicted", "Observed"],
        loc=(0.02, 0.75),
        edgecolor="black",
    )

    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years)
    ax.set_ylabel("Nationwide cases (N)")
    ax.set_yscale("log")
    ax.set_xlabel("Year")
    ax.set_yticks([10, 100, 1000, 10000])
    ax.set_yticklabels(["10", "100", "1,000", "10,000"])


def plot_risk_predictions_by_county(df, samples, background_samples, fig, ax):
    df = df.with_columns(
        statistical=pl.Series(samples.__array__()),
        rank=pl.col("cases").rank(method="average").over("year")
        / pl.col("cases").count().over("year")
        * 100,
        bucket=pl.when(pl.col("cases") <= 9).then(1).otherwise(2),
    )

    bin_width = 0.5
    offset = 0.0

    background_samples_vln = background_samples[
        jnp.logical_and(
            background_samples >= jnp.percentile(background_samples, 2.5),
            background_samples <= jnp.percentile(background_samples, 97.5),
        )
    ]
    vln = ax.violinplot(
        background_samples_vln,
        positions=[0 - offset],
        widths=bin_width,
        showextrema=False,
    )
    for pc in vln["bodies"]:
        pc.set_facecolor("green")
        pc.set_edgecolor("green")
        pc.set_alpha(0.3)

    ax.errorbar(
        0 - offset,
        jnp.median(background_samples),
        yerr=np.array(
            [
                [
                    jnp.median(background_samples)
                    - jnp.percentile(background_samples, 25)
                ],
                [
                    jnp.percentile(background_samples, 75)
                    - jnp.median(background_samples)
                ],
            ]
        ),
        fmt="o",
        markersize=7,
        ecolor="green",
        color="green",
        capsize=5,
        capthick=2,
        label="Statistical",
    )

    for bin in sorted(df["bucket"].unique()):
        bin_data = df.filter(pl.col("bucket") == bin)["statistical"].to_list()

        bin_data_vln = np.array(bin_data)[
            np.logical_and(
                bin_data >= np.percentile(bin_data, 2.5),
                bin_data <= np.percentile(bin_data, 97.5),
            )
        ]
        vln = ax.violinplot(
            bin_data_vln,
            positions=[bin - offset],
            widths=bin_width,
            showextrema=False,
        )
        for pc in vln["bodies"]:
            pc.set_facecolor("green")
            pc.set_edgecolor("green")
            pc.set_alpha(0.3)

        ax.errorbar(
            bin - offset,
            np.median(bin_data),
            yerr=np.array(
                [
                    [np.median(bin_data) - np.percentile(bin_data, 25)],
                    [np.percentile(bin_data, 75) - np.median(bin_data)],
                ]
            ),
            fmt="o",
            markersize=7,
            ecolor="green",
            color="green",
            capsize=5,
            capthick=2,
        )

    ax.set_xticks([0, *df["bucket"].sort().unique().to_list()])
    ax.set_xticklabels(["0", "1-9", "≥10"])
    ax.set_xlabel("Observed cases (N)")
    ax.set_ylabel("Validation-predicted county\noutbreak size percentile")
    ax.set_ylim([0, 100])

    return df


def plot_future_scenario_cases(
    future_years,
    plot_scenarios,
    scenario_colors,
    all_future_samples,
    importation_samples,
    importation_count_samples,
    scale,
    fig,
    ax,
    validation_2025_samples=None,
    validation_2025_mask=None,
):
    scenario_labels = {
        "no_change": "No changes in vaccination",
        "baseline": "Vaccination trends continue",
        "targeted_worsening": "Vaccinations universally decline",
        "universal_worsening": "Vaccinations universally decline by 50% more",
        "targeted_improvement": "Universal 95% coverage by 2030",
        "universal_improvement": "1% improvement year-over-year",
    }

    WORSENING_SCENARIOS = {"targeted_worsening", "universal_worsening"}
    IMPROVING_SCENARIOS = {"targeted_improvement", "universal_improvement"}

    n_scenarios = len(plot_scenarios)

    width = 0.2
    gap = 0.4
    offsets = np.linspace(-gap, gap, n_scenarios)  # type: ignore

    violin_tops = {}

    vlns = []
    labels = []
    for s_idx, scenario in enumerate(plot_scenarios):
        color = scenario_colors[scenario]
        for y_idx, year in enumerate(future_years):
            x_pos = y_idx + offsets[s_idx]

            raw = all_future_samples[
                (scenario, year)
            ]  # (n_samples, n_counties)

            # NEW: Apply no_change constraint - must be >= 2025 validation
            if (
                scenario == "no_change"
                and validation_2025_samples is not None
                and validation_2025_mask is not None
            ):
                # Lower bound: must be >= 2025
                constrained = jnp.where(
                    validation_2025_mask,
                    jnp.maximum(raw, validation_2025_samples),
                    raw,
                )
                raw = constrained

                # Upper bound: must be <= baseline 2026
                if ("targeted_worsening", 2026) in all_future_samples:
                    baseline_raw_2026 = all_future_samples[
                        ("targeted_worsening", 2026)
                    ]
                    raw = jnp.minimum(raw, baseline_raw_2026)

            # Apply monotonicity constraints at county level, sample-by-sample,
            # relative to 2026 baseline, before multiplying by importation samples
            if (
                scenario in WORSENING_SCENARIOS
                and ("baseline", 2026) in all_future_samples
            ):
                baseline_raw_2026 = all_future_samples[("baseline", 2026)]
                raw = jnp.maximum(raw, baseline_raw_2026)
            elif (
                scenario in IMPROVING_SCENARIOS
                and ("baseline", 2026) in all_future_samples
            ):
                baseline_raw_2026 = all_future_samples[("baseline", 2026)]
                raw = jnp.minimum(raw, baseline_raw_2026)

            combined_raw = (
                raw * importation_samples + importation_count_samples
            )
            combined = jnp.sum(combined_raw, axis=1) / scale
            combined_raw = combined_raw / scale

            lower = jnp.percentile(combined, 2.5)
            upper = jnp.percentile(combined, 97.5)
            violin_tops[(scenario, year)] = float(upper)

            combined_vln = combined[
                jnp.logical_and(
                    combined >= lower,
                    combined <= upper,
                )
            ]

            vln = ax.violinplot(
                combined_vln,
                positions=[x_pos],
                widths=width,
                showextrema=False,
            )
            for pc in vln["bodies"]:
                pc.set_facecolor(color)
                pc.set_edgecolor(color)
                pc.set_alpha(0.3)

            ax.errorbar(
                x_pos,
                jnp.median(combined),
                yerr=np.array(
                    [
                        [
                            jnp.median(combined)
                            - jnp.percentile(combined, 25),
                            jnp.percentile(combined, 75)
                            - jnp.median(combined),
                        ]
                    ]
                ).T,
                fmt="o",
                markersize=7,
                ecolor=color,
                color=color,
                capsize=5,
                capthick=2,
            )

            # print(
            #     f"Year {year}, Scenario {scenario}: "
            #     f"2.5th={lower:.1f}, 25th={jnp.percentile(combined, 25):.1f}, "
            #     f"50th={jnp.median(combined):.1f}, mean={jnp.mean(combined):.1f}, 75th={jnp.percentile(combined, 75):.1f}, "
            #     f"97.5th={upper:.1f}"
            #     f"\nCounties with over 20 cases in median: {jnp.sum(jnp.median(combined_raw, axis=0) >= 20) / combined_raw.shape[1]}"
            #     f"\nCounties with over 20 cases in 2.5th percentile: {jnp.sum(jnp.percentile(combined_raw, 2.5, axis=0) >= 20) / combined_raw.shape[1]}"
            #     f"\nCounties with over 20 cases in 97.5th percentile: {jnp.sum(jnp.percentile(combined_raw, 97.5, axis=0) >= 20) / combined_raw.shape[1]}"
            #     f"\nCounties with over 10 cases in median: {jnp.sum(jnp.median(combined_raw, axis=0) >= 10) / combined_raw.shape[1]}"
            #     f"\nCounties with over 10 cases in 2.5th percentile: {jnp.sum(jnp.percentile(combined_raw, 2.5, axis=0) >= 10) / combined_raw.shape[1]}"
            #     f"\nCounties with over 10 cases in 97.5th percentile: {jnp.sum(jnp.percentile(combined_raw, 97.5, axis=0) >= 10) / combined_raw.shape[1]}"
            # )

            # def calculate_top_10_percent_cases(sample_row):
            #     """For a single posterior sample, calculate % of cases in top 10% of counties"""
            #     threshold = jnp.percentile(sample_row, 90)  # 90th percentile for this sample
            #     cases_in_top = jnp.sum(sample_row[sample_row >= threshold])  # sum cases >= threshold
            #     total_cases = jnp.sum(sample_row)  # total cases this sample
            #     return (cases_in_top / total_cases) * 100 if total_cases > 0 else 0.0

            # # Apply to each posterior sample (each row of combined_raw)
            # percent_in_top_10 = jnp.array([calculate_top_10_percent_cases(combined_raw[i, :])
            #                                 for i in range(combined_raw.shape[0])])

            # print(
            #     f"\nPercent of cases in top 10% of counties (sample-by-sample):"
            #     f"\n  Median: {jnp.median(percent_in_top_10):.1f}%"
            #     f"\n  Mean: {jnp.mean(percent_in_top_10):.1f}%"
            #     f"\n  95% CI: [{jnp.percentile(percent_in_top_10, 2.5):.1f}%, {jnp.percentile(percent_in_top_10, 97.5):.1f}%]"
            # )

        labels.append(scenario_labels[scenario])
        vlns.append(vln)  # type: ignore

    ax.set_ylim(0)
    ax.legend(
        handles=[
            vlns[0]["bodies"][0],
            vlns[1]["bodies"][0],
            vlns[2]["bodies"][0],
            vlns[3]["bodies"][0],
        ],
        labels=labels,
        loc=(0.02, 0.8),
        edgecolor="black",
    )
    ax.set_xticks(range(len(future_years)))
    ax.set_xticklabels(future_years)
    ax.set_xlabel("Year")
    ax.set_ylabel("Projected cases (N)")
    ax.set_yscale("log")
    ax.set_ylim(50, None)
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))

    # Add this section here - after all scenarios and years have been processed
    print("\n" + "=" * 60)
    print("CUMULATIVE CASES BY SCENARIO (2026-2030)")
    print("=" * 60)
    for scenario in plot_scenarios:
        cumulative_samples = None
        for year in future_years:
            raw = all_future_samples[(scenario, year)]

            # Apply no_change constraint - must be >= 2025 validation
            if (
                scenario == "no_change"
                and validation_2025_samples is not None
                and validation_2025_mask is not None
            ):
                # Lower bound: must be >= 2025
                constrained = jnp.where(
                    validation_2025_mask,
                    jnp.maximum(raw, validation_2025_samples),
                    raw,
                )
                raw = constrained

                # Upper bound: must be <= baseline 2026
                if ("targeted_worsening", 2026) in all_future_samples:
                    baseline_raw_2026 = all_future_samples[
                        ("targeted_worsening", 2026)
                    ]
                    raw = jnp.minimum(raw, baseline_raw_2026)

            # Apply same monotonicity constraints
            if (
                scenario in WORSENING_SCENARIOS
                and ("baseline", 2026) in all_future_samples
            ):
                baseline_raw_2026 = all_future_samples[("baseline", 2026)]
                raw = jnp.maximum(raw, baseline_raw_2026)
            elif (
                scenario in IMPROVING_SCENARIOS
                and ("baseline", 2026) in all_future_samples
            ):
                baseline_raw_2026 = all_future_samples[("baseline", 2026)]
                raw = jnp.minimum(raw, baseline_raw_2026)

            combined_raw = (
                raw * importation_samples + importation_count_samples
            )
            combined = jnp.sum(combined_raw, axis=1) / scale

            if cumulative_samples is None:
                cumulative_samples = combined
            else:
                cumulative_samples += combined

        print(f"\n{scenario_labels[scenario]}:")
        print(f"  Median: {jnp.median(cumulative_samples):,.1f}")  # type: ignore
        print(f"  Mean: {jnp.mean(cumulative_samples):,.1f}")  # type: ignore
        print(
            f"  95% CI: [{jnp.percentile(cumulative_samples, 2.5):,.1f}, {jnp.percentile(cumulative_samples, 97.5):,.1f}]"  # type: ignore
        )
    print("=" * 60 + "\n")


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
    if cbar_type == "percentile_risk":
        norm = mcolors.Normalize(vmin=0, vmax=100)
        breakpoints = [-1, 20, 40, 60, 80, 101]
        cmap = plt.get_cmap("Reds")
        cmap = mcolors.ListedColormap(
            [
                "#fcbba1",
                "#fc9272",
                "#fb6a4a",
                "#de2d26",
                "#a50f15",
            ]
        )

        # cmap = mcolors.ListedColormap(
        #     [cmap(i) for i in np.linspace(0.3, 1.0, 5)]
        # )
    elif cbar_type == "probability":
        norm = mcolors.LogNorm(vmin=counties_with_data["value"].min(), vmax=1)
        breakpoints = [-1, 0.2, 0.4, 0.6, 0.8, 1.01]
        cmap = plt.get_cmap("Blues")
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
    elif cbar_type == "count_buckets":
        norm = mcolors.BoundaryNorm(
            boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], ncolors=5
        )
        cmap = mcolors.ListedColormap(
            ["#fee5d9", "#fcae91", "#fb6a4a", "#de2d26", "#a50f15"]
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
        if cbar_type == "percentile_risk":
            cbar.set_ticks([20, 40, 60, 80, 100])
            cbar.set_ticklabels(
                ["Bottom 20%", "20%-40%", "40%-60%", "60%-80%", "Top 20%"]
            )
            cbar.ax.tick_params(labelsize=24, rotation=-60, length=0)
        if cbar_type == "probability":
            cbar.set_ticks([1e-4, 0.001, 0.01, 0.1, 1.0])
            cbar.set_ticklabels(["0.0001", "0.001", "0.01", "0.1", "1.0"])
            cbar.ax.tick_params(labelsize=24, rotation=-60, length=0)

        if cbar_type == "count_buckets":
            cbar.set_ticks([0, 1, 2, 3, 4])
            cbar.set_ticklabels(["<1", "1–9", "10–49", "50–99", "≥100"])
            cbar.ax.tick_params(labelsize=24, rotation=0, length=0)

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


def create_supplemental_future_projections_figure(
    future_years,
    scenarios,
    scenario_colors,
    all_future_samples,
    importation_samples,
    importation_count_samples,
    scale,
    output_path,
    validation_2025_samples=None,
    validation_2025_mask=None,
):
    """
    Create a standalone figure showing alternative future projection scenarios.

    Parameters
    ----------
    future_years : list
        Years to plot (e.g., [2026, 2027, 2028, 2029, 2030])
    scenarios : list
        Scenario names to include in the plot
    scenario_colors : dict
        Mapping of scenario names to colors
    all_future_samples : dict
        Dictionary with (scenario, year) keys containing sample arrays
    importation_samples : array
        Importation probability samples
    importation_count_samples : array
        Importation count samples
    scale : float
        Scaling factor for predictions
    output_path : str
        Path to save the figure (without extension)
    """
    # Create figure with single subplot
    fig, ax = plt.subplots(1, 1, figsize=(15, 7.5))

    # Apply standard styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis="x", direction="out")
    ax.tick_params(axis="y", direction="out")
    for spine in ax.spines.values():
        spine.set_position(("outward", 5))
    ax.set_axisbelow(True)

    # Plot the scenarios
    plot_future_scenario_cases(
        future_years=future_years,
        plot_scenarios=scenarios,
        scenario_colors=scenario_colors,
        all_future_samples=all_future_samples,
        importation_samples=importation_samples,
        importation_count_samples=importation_count_samples,
        scale=scale,
        fig=fig,
        ax=ax,
        validation_2025_samples=validation_2025_samples,
        validation_2025_mask=validation_2025_mask,
    )

    fig.tight_layout()

    # Save in both formats
    fig.savefig(
        f"{output_path}_constrained.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        f"{output_path}.pdf",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Supplemental figure saved to {output_path}.png and {output_path}.pdf"
    )


def plot_excess_cases_vs_no_change(
    future_years,
    plot_scenarios,
    scenario_colors,
    all_future_samples,
    importation_samples,
    importation_count_samples,
    scale,
    fig,
    ax,
):
    """
    Plot excess cases relative to 'no_change' scenario using paired sample differences.

    Positive values indicate more cases than no_change scenario.
    Negative values indicate fewer cases than no_change scenario.
    """
    scenario_labels = {
        "baseline": "Vaccination trends continue",
        "targeted_worsening": "Vaccinations universally decline",
        "universal_worsening": "Vaccinations universally decline by 50% more",
        "targeted_improvement": "Universal 95% coverage by 2030",
        "universal_improvement": "1% improvement year-over-year",
    }

    WORSENING_SCENARIOS = {"targeted_worsening", "universal_worsening"}
    IMPROVING_SCENARIOS = {"targeted_improvement", "universal_improvement"}

    # Remove 'no_change' from plot_scenarios if it's there
    plot_scenarios = [s for s in plot_scenarios if s != "no_change"]
    n_scenarios = len(plot_scenarios)

    width = 0.2
    gap = 0.4
    offsets = np.linspace(-gap, gap, n_scenarios)

    vlns = []
    labels = []

    for s_idx, scenario in enumerate(plot_scenarios):
        color = scenario_colors[scenario]

        for y_idx, year in enumerate(future_years):
            x_pos = y_idx + offsets[s_idx]

            # Get scenario samples
            raw_scenario = all_future_samples[(scenario, year)]

            # Apply monotonicity constraints
            if (
                scenario in WORSENING_SCENARIOS
                and ("baseline", 2026) in all_future_samples
            ):
                baseline_raw_2026 = all_future_samples[("baseline", 2026)]
                raw_scenario = jnp.maximum(raw_scenario, baseline_raw_2026)
            elif (
                scenario in IMPROVING_SCENARIOS
                and ("baseline", 2026) in all_future_samples
            ):
                baseline_raw_2026 = all_future_samples[("baseline", 2026)]
                raw_scenario = jnp.minimum(raw_scenario, baseline_raw_2026)

            combined_scenario = (
                raw_scenario * importation_samples + importation_count_samples
            )
            total_scenario = jnp.sum(combined_scenario, axis=1) / scale

            # Get no_change samples (with same constraints for fair comparison)
            raw_no_change = all_future_samples[("no_change", year)]
            combined_no_change = (
                raw_no_change * importation_samples + importation_count_samples
            )
            total_no_change = jnp.sum(combined_no_change, axis=1) / scale

            # Calculate paired difference (excess cases)
            excess = total_scenario - total_no_change

            lower = jnp.percentile(excess, 2.5)
            upper = jnp.percentile(excess, 97.5)

            excess_vln = excess[
                jnp.logical_and(
                    excess >= lower,
                    excess <= upper,
                )
            ]

            vln = ax.violinplot(
                excess_vln,
                positions=[x_pos],
                widths=width,
                showextrema=False,
            )
            for pc in vln["bodies"]:
                pc.set_facecolor(color)
                pc.set_edgecolor(color)
                pc.set_alpha(0.3)

            ax.errorbar(
                x_pos,
                jnp.median(excess),
                yerr=np.array(
                    [
                        [
                            jnp.median(excess) - jnp.percentile(excess, 25),
                            jnp.percentile(excess, 75) - jnp.median(excess),
                        ]
                    ]
                ).T,
                fmt="o",
                markersize=7,
                ecolor=color,
                color=color,
                capsize=5,
                capthick=2,
            )

        labels.append(scenario_labels[scenario])
        vlns.append(vln)  # type: ignore

    # Add horizontal line at zero
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)

    ax.legend(
        handles=[
            vlns[0]["bodies"][0],
            vlns[1]["bodies"][0],
            vlns[2]["bodies"][0],
            vlns[3]["bodies"][0],
        ],
        labels=labels,
        loc="best",
        edgecolor="black",
    )

    ax.set_xticks(range(len(future_years)))
    ax.set_xticklabels(future_years)
    ax.set_xlabel("Year")
    ax.set_ylabel("Excess cases vs. no change scenario (N)")
    ax.set_yscale("symlog", linthresh=100)
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))

    # Print cumulative statistics
    print("\n" + "=" * 60)
    print("CUMULATIVE EXCESS CASES VS NO CHANGE (2026-2030)")
    print("=" * 60)
    for scenario in plot_scenarios:
        cumulative_excess = None
        for year in future_years:
            raw_scenario = all_future_samples[(scenario, year)]

            if (
                scenario in WORSENING_SCENARIOS
                and ("baseline", 2026) in all_future_samples
            ):
                baseline_raw_2026 = all_future_samples[("baseline", 2026)]
                raw_scenario = jnp.maximum(raw_scenario, baseline_raw_2026)
            elif (
                scenario in IMPROVING_SCENARIOS
                and ("baseline", 2026) in all_future_samples
            ):
                baseline_raw_2026 = all_future_samples[("baseline", 2026)]
                raw_scenario = jnp.minimum(raw_scenario, baseline_raw_2026)

            combined_scenario = (
                raw_scenario * importation_samples + importation_count_samples
            )
            total_scenario = jnp.sum(combined_scenario, axis=1) / scale

            raw_no_change = all_future_samples[("no_change", year)]
            combined_no_change = (
                raw_no_change * importation_samples + importation_count_samples
            )
            total_no_change = jnp.sum(combined_no_change, axis=1) / scale

            excess = total_scenario - total_no_change

            if cumulative_excess is None:
                cumulative_excess = excess
            else:
                cumulative_excess += excess

        print(f"\n{scenario_labels[scenario]}:")
        print(f"  Median: {jnp.median(cumulative_excess):,.1f}")  # type: ignore
        print(f"  Mean: {jnp.mean(cumulative_excess):,.1f}")  # type: ignore
        print(
            f"  95% CI: [{jnp.percentile(cumulative_excess, 2.5):,.1f}, {jnp.percentile(cumulative_excess, 97.5):,.1f}]"  # type: ignore
        )
    print("=" * 60 + "\n")


def create_supplemental_excess_cases_figure(
    future_years,
    scenarios,
    scenario_colors,
    all_future_samples,
    importation_samples,
    importation_count_samples,
    scale,
    output_path,
):
    """
    Create a supplemental figure showing excess cases relative to no_change scenario.

    Parameters
    ----------
    future_years : list
        Years to plot (e.g., [2026, 2027, 2028, 2029, 2030])
    scenarios : list
        Scenario names to compare against no_change (should not include 'no_change')
    scenario_colors : dict
        Mapping of scenario names to colors
    all_future_samples : dict
        Dictionary with (scenario, year) keys containing sample arrays
    importation_samples : array
        Importation probability samples
    importation_count_samples : array
        Importation count samples
    scale : float
        Scaling factor for predictions
    output_path : str
        Path to save the figure (without extension)
    """
    # Verify no_change is in all_future_samples
    if not any(k[0] == "no_change" for k in all_future_samples.keys()):
        raise ValueError(
            "all_future_samples must contain 'no_change' scenario"
        )

    # Create figure with single subplot
    fig, ax = plt.subplots(1, 1, figsize=(15, 7.5))

    # Apply standard styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis="x", direction="out")
    ax.tick_params(axis="y", direction="out")
    for spine in ax.spines.values():
        spine.set_position(("outward", 5))
    ax.set_axisbelow(True)

    # Plot the excess cases
    plot_excess_cases_vs_no_change(
        future_years=future_years,
        plot_scenarios=scenarios,
        scenario_colors=scenario_colors,
        all_future_samples=all_future_samples,
        importation_samples=importation_samples,
        importation_count_samples=importation_count_samples,
        scale=scale,
        fig=fig,
        ax=ax,
    )

    fig.tight_layout()

    # Save in both formats
    fig.savefig(
        f"{output_path}.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        f"{output_path}.pdf",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Supplemental excess cases figure saved to {output_path}.png and {output_path}.pdf"
    )


def create_vaccination_scenario_effects_table(
    vaccination_scenarios,
    future_years,
    future_base_df,
    all_future_samples,
    importation_samples,
    importation_count_samples,
    scale,
    output_path,
):
    """
    Create a table with risk percentiles for each vaccination scenario by county and year.

    Returns one row per (county, year, vaccination_scenario) combination.

    Parameters
    ----------
    vaccination_scenarios : list
        List of scenario names
    future_years : list
        List of years to process
    future_base_df : pl.DataFrame
        DataFrame with fips, state, county, pop columns
    all_future_samples : dict
        Dictionary with (scenario, year) keys containing sample arrays (n_samples, n_counties)
    importation_samples : jnp.ndarray
        Importation probability samples (n_samples, n_counties)
    importation_count_samples : jnp.ndarray
        Importation count samples (n_samples, n_counties)
    scale : float
        Scaling factor
    output_path : str
        Path to save CSV
    """

    # Get county metadata
    fips_to_location = {}
    for row in future_base_df.select(
        ["fips", "state", "county", "pop"]
    ).iter_rows(named=True):
        fips_to_location[row["fips"]] = {
            "state": row["state"],
            "county": row["county"],
            "pop": row["pop"],
        }

    common_fips = future_base_df["fips"].to_list()
    common_pop = jnp.array(future_base_df["pop"].to_list())

    # Calculate importation probability statistics per county
    # Shape: (n_counties,)
    importation_prob_mean = jnp.mean(importation_samples, axis=0)
    importation_prob_median = jnp.median(importation_samples, axis=0)
    importation_prob_p025 = jnp.percentile(importation_samples, 2.5, axis=0)
    importation_prob_p975 = jnp.percentile(importation_samples, 97.5, axis=0)

    WORSENING_SCENARIOS = {"targeted_worsening", "universal_worsening"}
    IMPROVING_SCENARIOS = {"targeted_improvement", "universal_improvement"}

    all_rows = []

    # Loop through all scenarios
    for vac_scenario in vaccination_scenarios:
        print(f"Processing vaccination scenario: {vac_scenario}")

        # Get baseline 2026 for monotonicity constraints
        baseline_raw_2026 = None
        if (
            vac_scenario in WORSENING_SCENARIOS
            or vac_scenario in IMPROVING_SCENARIOS
        ):
            if ("baseline", 2026) in all_future_samples:
                baseline_raw_2026 = all_future_samples[("baseline", 2026)]

        for year in future_years:
            # Load scenario samples for this year
            raw = all_future_samples[(vac_scenario, year)]

            # Verify shape matches
            assert raw.shape[1] == len(common_fips), (
                f"Shape mismatch for {vac_scenario} {year}: "
                f"expected {len(common_fips)} counties, got {raw.shape[1]}"
            )

            # Apply monotonicity constraints at county level
            if (
                vac_scenario in WORSENING_SCENARIOS
                and baseline_raw_2026 is not None
            ):
                raw = jnp.maximum(raw, baseline_raw_2026)
            elif (
                vac_scenario in IMPROVING_SCENARIOS
                and baseline_raw_2026 is not None
            ):
                raw = jnp.minimum(raw, baseline_raw_2026)

            # Calculate cases for this year
            # Shape: (n_samples, n_counties)
            cases_county = (
                raw * importation_samples + importation_count_samples
            ) / scale

            # --- SINGLE-YEAR risk percentiles ---
            # Count-based rank
            count_rank = mean_percentile_rank(np.array(cases_county))

            # Incidence-based rank: cases / population
            cases_incidence = cases_county / common_pop.reshape(1, -1)
            incidence_rank = mean_percentile_rank(np.array(cases_incidence))

            for i, fips in enumerate(common_fips):
                location = fips_to_location[fips]
                all_rows.append(
                    {
                        "state": location["state"],
                        "county": location["county"],
                        "fips": fips,
                        "year": year,
                        "vaccination_scenario": VACCINATION_SCENARIO_LABELS[
                            vac_scenario
                        ],  # Changed
                        "importation_probability_mean": float(
                            importation_prob_mean[i]
                        ),
                        "importation_probability_median": float(
                            importation_prob_median[i]
                        ),
                        "importation_probability_p025": float(
                            importation_prob_p025[i]
                        ),
                        "importation_probability_p975": float(
                            importation_prob_p975[i]
                        ),
                        "count_risk_percentile": float(count_rank[i]),
                        "incidence_risk_percentile": float(incidence_rank[i]),
                    }
                )

            # Free memory
            del cases_county, cases_incidence, count_rank, incidence_rank

    # Convert to DataFrame and save
    results_df = pl.DataFrame(all_rows)

    # Sort for easier reading
    results_df = results_df.sort(
        ["vaccination_scenario", "year", "state", "county"]
    )

    results_df.write_csv(output_path)
    print(f"\nVaccination scenario effects table saved to {output_path}")
    print(f"Total rows: {len(results_df)}")

    return results_df


def create_state_vaccination_scenario_effects_table(
    vaccination_scenarios,
    future_years,
    future_base_df,
    all_future_samples,
    importation_samples,
    importation_count_samples,
    scale,
    output_path,
):
    """
    Create state-level vaccination scenario effects table.

    Returns one row per (state, year, vaccination_scenario) combination.

    Parameters
    ----------
    vaccination_scenarios : list
        List of scenario names
    future_years : list
        List of years to process
    future_base_df : pl.DataFrame
        DataFrame with fips, state, county, pop columns
    all_future_samples : dict
        Dictionary with (scenario, year) keys containing sample arrays (n_samples, n_counties)
    importation_samples : jnp.ndarray
        Importation probability samples (n_samples, n_counties)
    importation_count_samples : jnp.ndarray
        Importation count samples (n_samples, n_counties)
    scale : float
        Scaling factor
    output_path : str
        Path to save CSV
    """

    # Get state mapping and populations
    fips_to_state = dict(zip(future_base_df["fips"], future_base_df["state"]))

    # Calculate state populations
    state_pop_df = future_base_df.group_by("state").agg(
        pl.col("pop").sum().alias("state_pop")
    )
    state_to_pop = dict(zip(state_pop_df["state"], state_pop_df["state_pop"]))

    common_fips = future_base_df["fips"].to_list()
    fips_states = [fips_to_state[f] for f in common_fips]
    unique_states = sorted(list(set(fips_states)))

    # Pre-compute which column indices belong to which state
    state_to_col_indices = {}
    for state in unique_states:
        state_to_col_indices[state] = jnp.array(
            [i for i, s in enumerate(fips_states) if s == state]
        )

    # Calculate state-level importation probabilities by summing county probabilities
    # Shape: (n_samples, n_counties) -> aggregate to (n_samples, n_states)
    state_importation_samples = {}
    for state in unique_states:
        col_indices = state_to_col_indices[state]
        # Sum importation probabilities across counties in this state
        state_importation_samples[state] = jnp.sum(
            importation_samples[:, col_indices], axis=1
        )

    # Calculate summary statistics for each state
    state_importation_stats = {}
    for state in unique_states:
        samples = state_importation_samples[state]
        state_importation_stats[state] = {
            "mean": float(jnp.mean(samples)),
            "median": float(jnp.median(samples)),
            "p025": float(jnp.percentile(samples, 2.5)),
            "p975": float(jnp.percentile(samples, 97.5)),
        }

    WORSENING_SCENARIOS = {"targeted_worsening", "universal_worsening"}
    IMPROVING_SCENARIOS = {"targeted_improvement", "universal_improvement"}

    all_rows = []

    # Loop through all scenarios
    for vac_scenario in vaccination_scenarios:
        print(f"Processing state-level vaccination scenario: {vac_scenario}")

        # Get baseline 2026 for monotonicity constraints
        baseline_raw_2026 = None
        if (
            vac_scenario in WORSENING_SCENARIOS
            or vac_scenario in IMPROVING_SCENARIOS
        ):
            if ("baseline", 2026) in all_future_samples:
                baseline_raw_2026 = all_future_samples[("baseline", 2026)]

        for year in future_years:
            # Load scenario samples for this year
            raw = all_future_samples[(vac_scenario, year)]

            # Verify shape matches
            assert raw.shape[1] == len(common_fips), (
                f"Shape mismatch for {vac_scenario} {year}: "
                f"expected {len(common_fips)} counties, got {raw.shape[1]}"
            )

            # Apply monotonicity constraints at county level
            if (
                vac_scenario in WORSENING_SCENARIOS
                and baseline_raw_2026 is not None
            ):
                raw = jnp.maximum(raw, baseline_raw_2026)
            elif (
                vac_scenario in IMPROVING_SCENARIOS
                and baseline_raw_2026 is not None
            ):
                raw = jnp.minimum(raw, baseline_raw_2026)

            # Calculate cases for this year at county level
            # Shape: (n_samples, n_counties)
            cases_county = (
                raw * importation_samples + importation_count_samples
            ) / scale

            # Aggregate to state level for this year
            cases_state_year = {}
            for state in unique_states:
                col_indices = state_to_col_indices[state]
                # Sum across counties in this state for each sample
                # Shape: (n_samples,)
                cases_state_year[state] = jnp.sum(
                    cases_county[:, col_indices], axis=1
                )

            del cases_county

            # --- SINGLE-YEAR risk percentiles at state level ---
            # Stack all states' samples for this year
            # Shape: (n_samples, n_states)
            cases_state_year_array = jnp.stack(
                [cases_state_year[state] for state in unique_states], axis=1
            )

            # Count-based rank
            count_rank_states = mean_percentile_rank(
                np.array(cases_state_year_array)
            )

            # Incidence-based rank: cases / state population
            state_pops = jnp.array(
                [state_to_pop[state] for state in unique_states]
            )
            cases_state_incidence = (
                cases_state_year_array / state_pops.reshape(1, -1)
            )
            incidence_rank_states = mean_percentile_rank(
                np.array(cases_state_incidence)
            )

            # Create rows for each state
            for idx, state in enumerate(unique_states):
                all_rows.append(
                    {
                        "state": state,
                        "year": year,
                        "vaccination_scenario": VACCINATION_SCENARIO_LABELS[
                            vac_scenario
                        ],  # Changed
                        "importation_probability_mean": state_importation_stats[
                            state
                        ]["mean"],
                        "importation_probability_median": state_importation_stats[
                            state
                        ]["median"],
                        "importation_probability_p025": state_importation_stats[
                            state
                        ]["p025"],
                        "importation_probability_p975": state_importation_stats[
                            state
                        ]["p975"],
                        "count_risk_percentile": float(count_rank_states[idx]),
                        "incidence_risk_percentile": float(
                            incidence_rank_states[idx]
                        ),
                    }
                )

            # Free memory
            del cases_state_year, cases_state_year_array, cases_state_incidence
            del count_rank_states, incidence_rank_states

    # Convert to DataFrame and save
    results_df = pl.DataFrame(all_rows)

    # Sort for easier reading
    results_df = results_df.sort(["vaccination_scenario", "year", "state"])

    results_df.write_csv(output_path)
    print(
        f"\nState-level vaccination scenario effects table saved to {output_path}"
    )
    print(f"Total rows: {len(results_df)}")

    return results_df


if __name__ == "__main__":
    # Build figure with subplots a-d in a 2x2 grid and e spanning the full bottom row
    fig = plt.figure(figsize=(20, 22.5))
    ax = {}
    ax[0, 0] = fig.add_subplot(3, 2, 1)
    ax[0, 1] = fig.add_subplot(3, 2, 2)
    ax[1, 0] = fig.add_subplot(3, 2, 3)
    ax[1, 1] = fig.add_subplot(3, 2, 4)
    ax_e = fig.add_subplot(3, 1, 3)

    for temp in [ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1], ax_e]:
        temp.spines["top"].set_visible(False)
        temp.spines["right"].set_visible(False)
        temp.get_xaxis().tick_bottom()
        temp.get_yaxis().tick_left()
        temp.tick_params(axis="x", direction="out")
        temp.tick_params(axis="y", direction="out")
        for spine in temp.spines.values():
            spine.set_position(("outward", 5))
        temp.set_axisbelow(True)

    ax[0, 0].text(
        -0.05,
        1.08,
        "a",
        transform=ax[0, 0].transAxes,
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )

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
    all_counties = all_counties.select(
        [
            "county",
            "state",
            "year",
            "validation_year",
            "cases_outbreak",
            "type",
            "fips",
        ]
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
        samples = pickle.load(f)
    with open(
        os.path.join(output_dir, "empirical_importation_count_samples.pkl"),
        "rb",
    ) as f:
        count_samples = pickle.load(f)
    import_df = import_df.with_columns(
        importation_probability_samples=pl.Series(samples),
        expected_importations_samples=pl.Series(count_samples),
    )
    import_df = import_df.sort("type").unique(
        subset=["state", "county", "year"], keep="last"
    )

    all_counties = all_counties.join(
        import_df.select(
            [
                "county",
                "state",
                "year",
                "importation_probability_samples",
                "expected_importations_samples",
                "type",
            ]
        ),
        on=["county", "state", "year"],
        how="left",
        maintain_order="left",
    )
    importation_samples = jnp.array(
        np.vstack(all_counties["importation_probability_samples"].to_numpy())
    ).T
    importation_count_samples = jnp.array(
        np.vstack(all_counties["expected_importations_samples"].to_numpy())
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
                f"samples_partial_pooled_poisson_validation_testing_{year}_pool=True_CAR=True_All covariates_globals_prior_sd={GLOBALS_PRIOR_SD}.pkl",
            ),
            "rb",
        ) as f:
            m = pickle.load(f)
        samples.append(m)

    samples = jnp.concat(samples, axis=1)

    # read in counties with missing vaccine data
    missing = pl.read_csv(os.path.join(data_dir, "missing_counties.csv"))
    # adjust all counties to have the type of these places be train
    all_counties = all_counties.with_columns(
        state_county_year=pl.concat_str(
            [
                pl.col("state"),
                pl.lit("_"),
                pl.col("county"),
                pl.lit("_"),
                pl.col("validation_year").cast(pl.String),
            ]
        )
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
    all_counties = all_counties.with_columns(
        type=pl.when(
            pl.col("state_county_year").is_in(missing["state_county_year"])
        )
        .then(pl.lit("train"))
        .otherwise(pl.col("type")),
        year=pl.when(
            pl.col("state_county_year").is_in(missing["state_county_year"])
        )
        .then(pl.col("validation_year"))
        .otherwise(pl.col("year")),
    )

    df_train_observed = all_counties.filter(pl.col("type") == "train")

    analysis = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        schema_overrides={"fips": pl.String},
    )[["year", "state", "county", "cases"]]
    analysis = pl.concat(
        [analysis, missing[["year", "state", "county", "cases"]]],
        how="diagonal_relaxed",
    )

    df_train_observed = df_train_observed.join(
        analysis,
        on=["year", "state", "county"],
        maintain_order="left",
    )

    observed = np.array([])
    predicted = np.array([])
    predicted_full = np.array([])

    mask = (all_counties["type"] == "train").to_numpy()
    samples_observed = samples[:, mask]
    importation_samples_observed = importation_samples[:, mask]
    importation_count_samples_observed = importation_count_samples[:, mask]

    for year in validation_years:
        mask = df_train_observed["validation_year"] == year
        samples_year = samples_observed[:, mask.to_numpy()]
        importation_samples_year = importation_samples_observed[
            :, mask.to_numpy()
        ]
        importation_count_samples_year = importation_count_samples_observed[
            :, mask.to_numpy()
        ]
        samples_year = (
            samples_year * importation_samples_year
            + importation_count_samples_year
        )
        if year != 2025:
            true = df_train_observed.filter(pl.col("validation_year") == year)[
                "cases"
            ].sum()
        else:
            # 2025 df_train_observed for SC contains 25-26 cases
            true = df_train_observed.filter(
                (pl.col("validation_year") == year)
                & (pl.col("state") != "South Carolina")
            )["cases"].sum()
            true += 333  # that is the 2025 case count for SC
        samples_year = jnp.sum(samples_year, axis=1)
        print(
            f"Year {year}: true={true}, 2.5th={jnp.percentile(samples_year, 2.5)}, "
            f"25th={jnp.percentile(samples_year, 25)}, 50th={jnp.percentile(samples_year, 50)}, "
            f"75th={jnp.percentile(samples_year, 75)}, 97.5th={jnp.percentile(samples_year, 97.5)}"
        )
        observed = np.append(observed, true)
        predicted = np.append(predicted, jnp.median(samples_year))

        mask = all_counties["validation_year"] == year
        samples_year = samples[:, mask.to_numpy()]
        importation_samples_year = importation_samples[:, mask.to_numpy()]
        importation_count_samples_year = importation_count_samples[
            :, mask.to_numpy()
        ]
        samples_year = (
            samples_year * importation_samples_year
            + importation_count_samples_year
        )
        samples_year = jnp.sum(samples_year, axis=1)
        print(
            f"Year {year} total predicted cases: 2.5th={jnp.percentile(samples_year, 2.5)}, "
            f"25th={jnp.percentile(samples_year, 25)}, 50th={jnp.percentile(samples_year, 50)}, "
            f"75th={jnp.percentile(samples_year, 75)}, 97.5th={jnp.percentile(samples_year, 97.5)}"
        )
        predicted_full = np.append(predicted_full, jnp.median(samples_year))

    scale = predicted_full / observed
    print(scale)
    scale = np.mean(scale)

    scale_2025 = predicted_full[-1] / observed[-1]

    # this is an important value, so let's save it
    with open(
        os.path.join(output_dir, "scale_2025.pkl"),
        "wb",
    ) as f:
        pickle.dump(scale_2025, f)

    samples_unconditioned = (
        samples * importation_samples + importation_count_samples
    )
    samples_observed_unconditioned = (
        samples_observed * importation_samples_observed
        + importation_count_samples_observed
    )
    samples_scaled = samples_unconditioned / scale
    samples_observed_scaled = samples_observed_unconditioned / scale

    for year in validation_years:
        mask = all_counties["validation_year"] == year
        samples_year = samples_scaled[:, mask.to_numpy()]
        samples_year = jnp.sum(samples_year, axis=1)
        print(
            f"Year {year}: 2.5th={jnp.percentile(samples_year, 2.5)}, "
            f"25th={jnp.percentile(samples_year, 25)}, 50th={jnp.percentile(samples_year, 50)}, "
            f"75th={jnp.percentile(samples_year, 75)}, 97.5th={jnp.percentile(samples_year, 97.5)}"
        )

    plot_total_cases_over_time(
        df_train_observed,
        all_counties,
        samples_scaled,
        fig,
        ax[0, 0],
    )

    ax[0, 1].text(
        -0.05,
        1.12,
        "b",
        transform=ax[0, 1].transAxes,
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )

    ranking_method = "average"
    samples_ranked = stats.rankdata(
        samples_scaled,
        axis=1,
        method=ranking_method,
    )
    samples_ranked = samples_ranked / samples_ranked.max(keepdims=True) * 100

    samples_ranked = np.mean(samples_ranked, axis=0)

    observed_ranked = samples_ranked[all_counties["type"] == "train"]
    unobserved_ranked = samples_ranked[all_counties["type"] == "projection"]

    plot_risk_predictions_by_county(
        df_train_observed,
        observed_ranked,
        unobserved_ranked,
        fig,
        ax[0, 1],
    )

    df = pl.read_csv(
        os.path.join(output_dir, "assembled_projection_results.csv"),
        schema_overrides={"fips": pl.String},
    )

    ax[1, 0].text(
        -0.05,
        1.08,
        "c",
        transform=ax[1, 0].transAxes,
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    map_fips_code_data(
        df["fips"],
        df["importation_probability_median"],
        fig,
        ax[1, 0],
        cbar_label="Probability of ≥1 international importation",
        cbar_type="probability",
        point_data=None,
    )

    ax[1, 1].text(
        -0.05,
        1.08,
        "d",
        transform=ax[1, 1].transAxes,
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )

    # --- subplot e: future scenario projections ---
    ax_e.text(
        -0.025,
        1.1,
        "e",
        transform=ax_e.transAxes,
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )

    future_years_plot = [2026, 2027, 2028, 2029, 2030]
    scenarios_ordered = [
        "no_change",
        "baseline",
        "targeted_worsening",
        "universal_worsening",
        "targeted_improvement",
        "universal_improvement",
    ]
    scenario_colors = {
        "no_change": "#525252",
        "baseline": "#8B4513",
        "targeted_worsening": "#FA2D2D",
        "universal_worsening": "#d94701",
        "targeted_improvement": "#006d2c",
        "universal_improvement": "#41ab5d",
    }

    # Load future_base and restrict to counties present in the scenario pkl
    # files. The scenario CSVs were built via an inner join against the
    # vaccine projections, so one county present in projection_dataset_2026
    # may have been dropped. We use projection_dataset_baseline_2026 as the
    # reference county set since all scenarios share the same counties.
    future_base = pl.read_csv(
        os.path.join(output_dir, "projection_dataset_2026.csv"),
        schema_overrides={"fips": pl.String},
    ).filter(pl.col("type") == "projection")

    reference_scenario_df = pl.read_csv(
        os.path.join(output_dir, "projection_dataset_baseline_2026.csv"),
        schema_overrides={"fips": pl.String},
    )
    future_base = future_base.join(
        reference_scenario_df.select("fips"),
        on="fips",
        how="inner",
        maintain_order="left",
    )

    # Load importation probabilities fresh and attach samples before
    # any filtering so row counts are guaranteed to match
    import_df_latest = pl.read_csv(
        os.path.join(output_dir, "empirical_importation_probability.csv"),
        schema_overrides={"fips": pl.String},
    )
    with open(
        os.path.join(
            output_dir, "empirical_importation_probability_samples.pkl"
        ),
        "rb",
    ) as f:
        importation_samples_raw = pickle.load(f)
    with open(
        os.path.join(output_dir, "empirical_importation_count_samples.pkl"),
        "rb",
    ) as f:
        importation_count_samples_raw = pickle.load(f)
    import_df_latest = import_df_latest.with_columns(
        importation_probability_samples=pl.Series(importation_samples_raw),
        expected_importations_samples=pl.Series(importation_count_samples_raw),
    )
    # Defer to projection over train where both exist
    import_df_latest = import_df_latest.sort("type").unique(
        subset=["state", "county", "year", "fips"], keep="last"
    )
    # Keep only the latest year per fips
    import_df_latest = import_df_latest.sort("year", descending=True).unique(
        subset=["fips"], keep="first"
    )

    assert import_df_latest["fips"].is_duplicated().sum() == 0, (
        "import_df_latest still has duplicate fips rows after dedup"
    )

    future_base = future_base.join(
        import_df_latest.select(
            [
                "fips",
                "importation_probability_samples",
                "expected_importations_samples",
            ]
        ),
        on="fips",
        how="left",
        maintain_order="left",
    )

    assert future_base["fips"].is_duplicated().sum() == 0, (
        "Join expanded future_base — duplicate fips in import_df_latest"
    )
    assert future_base["importation_probability_samples"].null_count() == 0, (
        "Some future counties are missing importation probability samples"
    )
    assert future_base["expected_importations_samples"].null_count() == 0, (
        "Some future counties are missing importation count samples"
    )

    importation_samples_future = jnp.array(
        np.vstack(future_base["importation_probability_samples"].to_numpy())  # type: ignore
    ).T
    importation_count_samples_future = jnp.array(
        np.vstack(future_base["expected_importations_samples"].to_numpy())  # type: ignore
    ).T

    # Load 2025 validation samples for no_change constraint
    validation_2025_df = pl.read_csv(
        os.path.join(output_dir, "projection_dataset_2025.csv"),
        schema_overrides={"fips": pl.String},
    )

    # Load the validation samples (already done earlier, but we'll extract 2025 specifically)
    # Get mask for 2025 validation year from all_counties
    mask_2025 = all_counties["validation_year"] == 2025
    samples_2025_raw = samples[
        :, mask_2025.to_numpy()
    ]  # (n_samples, n_counties_2025)

    # Get the fips codes for 2025 in the same order as samples_2025_raw
    fips_2025 = all_counties.filter(pl.col("validation_year") == 2025)[
        "fips"
    ].to_list()

    # Create a mapping from fips to column index in samples_2025_raw
    fips_2025_to_col = {fips: i for i, fips in enumerate(fips_2025)}

    # Get fips from future_base (in its current order)
    fips_future = future_base["fips"].to_list()

    # Align 2025 samples to match future_base county order
    # For counties not in 2025 validation, we'll use None as a sentinel
    aligned_2025_samples = []
    alignment_mask = []  # True where we have 2025 data, False otherwise

    for fips in fips_future:
        if fips in fips_2025_to_col:
            col_idx = fips_2025_to_col[fips]
            aligned_2025_samples.append(samples_2025_raw[:, col_idx])
            alignment_mask.append(True)
        else:
            # No 2025 data for this county - create dummy column of zeros
            aligned_2025_samples.append(jnp.zeros(samples_2025_raw.shape[0]))
            alignment_mask.append(False)

    # Stack into array: (n_samples, n_counties_future)
    validation_2025_samples_aligned = jnp.stack(aligned_2025_samples, axis=1)
    alignment_mask = jnp.array(alignment_mask)

    print(
        f"\n2025 validation alignment: {alignment_mask.sum()}/{len(alignment_mask)} counties have 2025 data"
    )

    all_future_samples = {}
    for scenario in scenarios_ordered:
        for future_year in future_years_plot:
            with open(
                os.path.join(
                    output_dir,
                    f"risk_projections_{scenario}_{future_year}_globals_prior_sd={GLOBALS_PRIOR_SD}.pkl",
                ),
                "rb",
            ) as f:
                loaded = pickle.load(f)
            assert loaded.shape[1] == future_base.height, (
                f"Shape mismatch for {scenario} {future_year}: pkl has {loaded.shape[1]} counties but future_base has {future_base.height}"
            )
            all_future_samples[(scenario, future_year)] = loaded

    with open(
        os.path.join(
            output_dir,
            "import_scaler_2025.pkl",
        ),
        "rb",
    ) as f:
        import_scaler_2025 = pickle.load(f)

    plot_scenarios = [
        "targeted_worsening",
        "baseline",
        "no_change",
        "targeted_improvement",
    ]

    plot_future_scenario_cases(
        future_years=future_years_plot,
        plot_scenarios=plot_scenarios,
        scenario_colors=scenario_colors,
        all_future_samples=all_future_samples,
        importation_samples=jnp.minimum(
            importation_samples_future * import_scaler_2025, 1
        ),
        importation_count_samples=importation_count_samples_future,
        scale=scale_2025,
        fig=fig,
        ax=ax_e,
        validation_2025_samples=validation_2025_samples_aligned,
        validation_2025_mask=alignment_mask,
    )

    map_fips_code_data(
        df["fips"],
        df["unconditioned_count_rank"],
        fig,
        ax[1, 1],
        cbar_label="Outbreak risk percentile",
        cbar_type="percentile_risk",
        point_data=None,
    )

    supplemental_df = df.select(
        ["fips", "state", "county", "unconditioned_count_rank"]
    )
    supplemental_df.write_csv(
        os.path.join(fig_dir, "table_supplemental_model_projections.csv")
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            fig_dir, "figure_validation_and_risk_projections_constrained.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(fig_dir, "figure_validation_and_risk_projections.pdf"),
        dpi=300,
        bbox_inches="tight",
    )

    # Create supplemental figure showing excess cases vs no_change
    excess_scenarios = [
        "universal_worsening",
        "baseline",
        "targeted_improvement",
        "universal_improvement",
    ]

    create_supplemental_excess_cases_figure(
        future_years=future_years_plot,
        scenarios=excess_scenarios,
        scenario_colors=scenario_colors,
        all_future_samples=all_future_samples,
        importation_samples=jnp.minimum(
            importation_samples_future * import_scaler_2025, 1
        ),
        importation_count_samples=importation_count_samples_future,
        scale=scale_2025,
        output_path=os.path.join(
            fig_dir, "figure_supplemental_excess_cases_vs_no_change"
        ),
    )

    # Create supplemental figure with alternative scenarios
    alternative_scenarios = [
        "universal_worsening",
        "baseline",
        "no_change",
        "universal_improvement",
    ]

    create_supplemental_future_projections_figure(
        future_years=future_years_plot,
        scenarios=alternative_scenarios,
        scenario_colors=scenario_colors,
        all_future_samples=all_future_samples,
        importation_samples=jnp.minimum(
            importation_samples_future * import_scaler_2025, 1
        ),
        importation_count_samples=importation_count_samples_future,
        scale=scale_2025,
        output_path=os.path.join(
            fig_dir, "figure_supplemental_future_projections"
        ),
        validation_2025_samples=validation_2025_samples_aligned,
        validation_2025_mask=alignment_mask,
    )

    vaccination_scenario_effects = create_vaccination_scenario_effects_table(
        vaccination_scenarios=scenarios_ordered,
        future_years=future_years_plot,
        future_base_df=future_base,
        all_future_samples=all_future_samples,
        importation_samples=jnp.minimum(
            importation_samples_future * import_scaler_2025, 1
        ),
        importation_count_samples=importation_count_samples_future
        * import_scaler_2025,
        scale=scale_2025,
        output_path=os.path.join(
            fig_dir,
            "supplemental_data_s4a_vaccination_scenario_effects_by_county_constrained.csv",
        ),
    )

    state_vaccination_scenario_effects = create_state_vaccination_scenario_effects_table(
        vaccination_scenarios=scenarios_ordered,
        future_years=future_years_plot,
        future_base_df=future_base,
        all_future_samples=all_future_samples,
        importation_samples=jnp.minimum(
            importation_samples_future * import_scaler_2025, 1
        ),
        importation_count_samples=importation_count_samples_future
        * import_scaler_2025,
        scale=scale_2025,
        output_path=os.path.join(
            fig_dir,
            "supplemental_data_s4b_vaccination_scenario_effects_by_state_constrained.csv",
        ),
    )
