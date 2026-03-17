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
        for spine in ax.spines.values():
            spine.set_position(("outward", 5))
        ax.set_axisbelow(True)
        return fig, ax
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


def plot_total_cases_over_time(df_observed, all_counties, samples, fig, ax):
    years = [2021, 2022, 2023, 2024, 2025]

    offset = 0.0
    widths = 0.6

    for idx, year in enumerate(years):
        mask = df_observed["validation_year"] == year
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
            color="purple",
            marker="*",
            s=200,
            zorder=10,
            label="Observed" if idx == 0 else "",
        )

    ax.legend(
        [
            vln1["bodies"][0],  # type: ignore
            mlines.Line2D(
                [], [], color="purple", marker="*", linestyle="", markersize=15
            ),
        ],
        ["Forecasted", "Observed"],
    )

    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years)
    ax.set_ylabel("Cases (N)")
    ax.set_yscale("log")
    ax.set_xlabel("Year")
    ax.set_yticks([1, 10, 100, 1000, 10000])
    ax.set_yticklabels(["1", "10", "100", "1,000", "10,000"])


def plot_risk_predictions_by_county(df, samples, background_samples, fig, ax):
    df = df.with_columns(
        statistical=pl.Series(samples.__array__()),
        rank=pl.col("cases").rank(method="average").over("year")
        / pl.col("cases").count().over("year")
        * 100,
        bucket=pl.when(pl.col("cases") <= 4)
        .then(1)
        .when(pl.col("cases") <= 9)
        .then(2)
        .when(pl.col("cases") <= 19)
        .then(3)
        .otherwise(4),
    )

    bin_width = 0.6
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
    ax.set_xticklabels(["0", "1-4", "5-9", "10-19", "≥20"])
    ax.set_xlabel("Observed cases (N)")
    ax.set_ylabel("Forecasted outbreak size percentile")
    ax.set_ylim([0, 100])

    return df


def plot_future_scenario_cases(
    future_years,
    scenarios,
    scenario_colors,
    all_future_samples,
    importation_samples,
    scale,
    fig,
    ax,
):
    scenario_labels = {
        "baseline": "Vaccination trends continue",
        "targeted_worsening": "Vaccination declines accelerate",
        "universal_worsening": "Universal declines",
        "targeted_improvement": "Vaccination declines stagnate",
        "universal_improvement": "Universal improvement",
    }

    # Fixed order: worsening left, baseline center, improvement right
    plot_scenarios = ["targeted_worsening", "baseline", "targeted_improvement"]
    n_scenarios = len(plot_scenarios)

    # Wider violins with more separation
    width = 0.25
    gap = 0.30  # total spread per year group
    offsets = np.linspace(-gap, gap, n_scenarios)

    # Store combined samples and violin tops per (scenario, year) for annotation
    combined_samples = {}
    violin_tops = {}  # 97.5th percentile y value for each violin

    vlns = []
    labels = []
    for s_idx, scenario in enumerate(plot_scenarios):
        color = scenario_colors[scenario]
        for y_idx, year in enumerate(future_years):
            x_pos = y_idx + offsets[s_idx]

            raw = all_future_samples[(scenario, year)]
            combined = raw * importation_samples
            combined = jnp.sum(combined, axis=1) / scale
            combined_samples[(scenario, year)] = combined

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

        labels.append(scenario_labels[scenario])
        vlns.append(vln)  # type: ignore

    # Annotation brackets: sit just above the tops of the two violins being compared
    # Bracket height is the max of the two violin tops plus a small padding
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    tick_len = y_range * 0.02  # length of downward ticks
    pad = y_range * 0.05  # gap between violin top and bracket
    text_pad = y_range * 0.015  # gap between bracket and text

    for y_idx, year in enumerate(future_years):
        baseline_samples = combined_samples[("baseline", year)]
        worsening_samples = combined_samples[("targeted_worsening", year)]
        improvement_samples = combined_samples[("targeted_improvement", year)]

        # Paired differences (same posterior draw index, no mixing)
        diff_worsening = worsening_samples - baseline_samples
        diff_improvement = improvement_samples - baseline_samples

        for tier, (
            diff,
            left_scenario,
            right_scenario,
            label_color,
        ) in enumerate(
            [
                (
                    diff_worsening,
                    "targeted_worsening",
                    "baseline",
                    scenario_colors["targeted_worsening"],
                ),
                (
                    diff_improvement,
                    "baseline",
                    "targeted_improvement",
                    scenario_colors["targeted_improvement"],
                ),
            ]
        ):
            med = int(round(float(jnp.median(diff))))

            s_idx_left = plot_scenarios.index(left_scenario)
            s_idx_right = plot_scenarios.index(right_scenario)
            x_left = y_idx + offsets[s_idx_left]
            x_right = y_idx + offsets[s_idx_right]
            x_mid = (x_left + x_right) / 2

            # Tier 0 (worsening) higher, tier 1 (improvement) lower
            tier_spacing = y_range * 0.10
            base_height = max(
                violin_tops[(left_scenario, year)],
                violin_tops[(right_scenario, year)],
            )
            y_bracket = base_height + pad + (1 - tier) * tier_spacing

            # Colored bracket
            ax.plot(
                [x_left, x_right],
                [y_bracket, y_bracket],
                color=label_color,
                lw=1.5,
                zorder=5,
                alpha=0.8,
            )
            ax.plot(
                [x_left, x_left],
                [y_bracket, y_bracket - tick_len],
                color=label_color,
                lw=1.5,
                zorder=5,
                alpha=0.8,
            )
            ax.plot(
                [x_right, x_right],
                [y_bracket, y_bracket - tick_len],
                color=label_color,
                lw=1.5,
                zorder=5,
                alpha=0.8,
            )

            ax.text(
                x_mid,
                y_bracket + text_pad,
                f"Δ={med:d}",
                ha="center",
                va="bottom",
                color=label_color,
            )

        # Update y-limit calculation
        highest_bracket = max(
            max(
                violin_tops[("targeted_worsening", y)],
                violin_tops[("baseline", y)],
            )
            + pad
            + tier_spacing  # type: ignore
            for y in future_years
        )
    ax.set_ylim(
        0,
        highest_bracket + pad + text_pad + y_range * 0.12,  # type: ignore
    )

    ax.legend(
        handles=[
            vlns[0]["bodies"][0],
            vlns[1]["bodies"][0],
            vlns[2]["bodies"][0],
        ],
        labels=labels,
    )
    ax.set_xticks(range(len(future_years)))
    ax.set_xticklabels(future_years)
    ax.set_xlabel("Year")
    ax.set_ylabel("Projected cases (N)")


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
            [cmap(i) for i in np.linspace(0.3, 1.0, 5)]
        )
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
    import_df = import_df.with_columns(
        importation_probability_samples=pl.Series(samples)
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
                "type",
            ]
        ),
        on=["county", "state", "year"],
        how="left",
    )
    importation_samples = jnp.array(
        np.vstack(all_counties["importation_probability_samples"].to_numpy())
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
            m = pickle.load(f)
        samples.append(m)

    samples = jnp.concat(samples, axis=1)

    mask = (all_counties["type"] == "train").to_numpy()
    samples_observed = samples[:, mask]
    importation_samples_observed = importation_samples[:, mask]
    df_train_observed = all_counties.filter(pl.col("type") == "train")

    analysis = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        schema_overrides={"fips": pl.String},
    )
    df_train_observed = df_train_observed.join(
        analysis[["year", "state", "county", "cases"]],
        on=["year", "state", "county"],
        maintain_order="left",
    )

    observed = np.array([])
    predicted = np.array([])
    predicted_full = np.array([])

    for year in validation_years:
        mask = df_train_observed["validation_year"] == year
        samples_year = samples_observed[:, mask.to_numpy()]
        importation_samples_year = importation_samples_observed[
            :, mask.to_numpy()
        ]
        samples_year = samples_year * importation_samples_year
        true = df_train_observed.filter(pl.col("validation_year") == year)[
            "cases"
        ].sum()
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
        samples_year = samples_year * importation_samples_year
        samples_year = jnp.sum(samples_year, axis=1)
        print(
            f"Year {year} total predicted cases: 2.5th={jnp.percentile(samples_year, 2.5)}, "
            f"25th={jnp.percentile(samples_year, 25)}, 50th={jnp.percentile(samples_year, 50)}, "
            f"75th={jnp.percentile(samples_year, 75)}, 97.5th={jnp.percentile(samples_year, 97.5)}"
        )
        predicted_full = np.append(predicted_full, jnp.median(samples_year))

    scale = predicted_full / observed
    scale = np.mean(scale)

    scale_2025 = (
        predicted_full[validation_years.index(2025)]
        / observed[validation_years.index(2025)]
    )

    samples_unconditioned = samples * importation_samples
    samples_observed_unconditioned = (
        samples_observed * importation_samples_observed
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
        1.08,
        "b",
        transform=ax[0, 1].transAxes,
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )

    ranking_method = "average"
    samples_ranked = stats.rankdata(
        jnp.mean(samples_scaled, axis=0),
        method=ranking_method,
    )
    samples_ranked = samples_ranked / samples_ranked.max() * 100

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
    map_fips_code_data(
        df["fips"],
        df["median_unconditioned_risk_rank"],
        fig,
        ax[1, 1],
        cbar_label="Outbreak size percentile",
        cbar_type="percentile_risk",
        point_data=None,
    )

    # --- subplot e: future scenario projections ---
    ax_e.text(
        -0.025,
        1.08,
        "e",
        transform=ax_e.transAxes,
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )

    future_years_plot = [2026, 2027, 2028, 2029, 2030]
    scenarios_ordered = [
        "baseline",
        "targeted_worsening",
        "universal_worsening",
        "targeted_improvement",
        "universal_improvement",
    ]
    scenario_colors = {
        "baseline": "#525252",
        "targeted_worsening": "#d94701",
        "universal_worsening": "#8c2d04",
        "targeted_improvement": "#41ab5d",
        "universal_improvement": "#006d2c",
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
    import_df_latest = import_df_latest.with_columns(
        importation_probability_samples=pl.Series(importation_samples_raw)
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
        import_df_latest.select(["fips", "importation_probability_samples"]),
        on="fips",
        how="left",
    )

    assert future_base["fips"].is_duplicated().sum() == 0, (
        "Join expanded future_base — duplicate fips in import_df_latest"
    )
    assert future_base["importation_probability_samples"].null_count() == 0, (
        "Some future counties are missing importation probability samples"
    )

    importation_samples_future = jnp.array(
        np.vstack(future_base["importation_probability_samples"].to_numpy())  # type: ignore
    ).T

    all_future_samples = {}
    for scenario in scenarios_ordered:
        for future_year in future_years_plot:
            with open(
                os.path.join(
                    output_dir,
                    f"risk_projections_{scenario}_{future_year}.pkl",
                ),
                "rb",
            ) as f:
                loaded = pickle.load(f)
            assert loaded.shape[1] == future_base.height, (
                f"Shape mismatch for {scenario} {future_year}: pkl has {loaded.shape[1]} counties but future_base has {future_base.height}"
            )
            all_future_samples[(scenario, future_year)] = loaded

    plot_future_scenario_cases(
        future_years=future_years_plot,
        scenarios=scenarios_ordered,
        scenario_colors=scenario_colors,
        all_future_samples=all_future_samples,
        importation_samples=importation_samples_future,
        scale=scale_2025,
        fig=fig,
        ax=ax_e,
    )

    fig.tight_layout()

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
