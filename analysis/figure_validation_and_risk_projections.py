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
from matplotlib.patches import Patch
from shapely.geometry import MultiPolygon, Polygon

from outbreak_size import scenario_tables
from outbreak_size.concentration import (
    COVERAGE,
    GRID,
    IMPORTS,
    MECHANISTIC,
    OBSERVED,
    POPULATION,
    RANKINGS,
    STATISTICAL,
    mean_curve,
    rankings,
    scenario_curve,
    validation_curves,
)
from outbreak_size.flag_discordance import COVARIATES
from outbreak_size.headline_models import (
    HEADLINE_FILE_NAME,
    read_headline_models,
)
from outbreak_size.headline_panels import (
    BUCKET_LABELS,
    PERFORMANCE_METRICS,
    case_bucket,
    conditioned_national_totals,
    performance_values,
)
from outbreak_size.importation import (
    future_importation_draws,
    scale_importation_probability,
)
from outbreak_size.model_colors import (
    COVERAGE_COLOR,
    IMPORTS_COLOR,
    MECHANISTIC_COLOR,
    POPULATION_COLOR,
    STATISTICAL_COLOR,
)
from outbreak_size.model_designs import training_years
from outbreak_size.model_selection import BEST_MODEL_YEARS
from outbreak_size.ranking import percentile_of_scores, risk_percentile
from outbreak_size.samples_io import LazyDraws, load_samples
from outbreak_size.scenarios import VACCINATION_SCENARIO_LABELS
from outbreak_size.validation import keep_rows, validation_file_name
from outbreak_size.year_rows import observed_county_years, validation_year_rows

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

GLOBALS_PRIOR_SD = params["validation_globals_prior_sd"]

# SCENARIO BOUNDS (removed; they can be put back if they are needed again)
# The figure and the tables used to post-process each county's draws of the
# outbreak size, draw by draw, before multiplying by P(import). There were two
# separate rules, applied at every place marked "no scenario bounds" below
# (the violins and the cumulative print of plot_future_scenario_cases, the two
# loops of plot_excess_cases_vs_no_change, the concentration curves, and
# scenario_tables._scenario_samples):
#   1. ordering: first = the first projected year; a worsening scenario
#      (targeted_/universal_worsening) was raised to at least the baseline's
#      draw of `first`, county by county (raw = jnp.maximum(raw,
#      all_future_samples[("baseline", first)])); an improving scenario
#      (targeted_/universal_improvement) was lowered to at most it (jnp.minimum)
#   2. no_change: a cap, jnp.minimum(raw, all_future_samples[
#      ("targeted_worsening", first)]), and a floor, jnp.maximum(raw, the draws
#      of the 2026 fold of the headline model aligned to the counties of the
#      projections by align_samples_to_counties, where a county without a row
#      was not bounded)
#   3. INTERVENTION BOUND (the intervention tables and figures): right after
#      a county's cases with the intervention (i_county) and under the
#      baseline (b_county) were computed per draw, i_county = jnp.minimum(
#      i_county, b_county), so an intervention could never raise a county's
#      cases. It sat at four places that are now marked "no intervention
#      bound": scenario_tables.county_intervention_effects, state_
#      intervention_effects and nationwide_summary (before the sums, the risk
#      percentiles and the percent changes), and plot_intervention_effects of
#      figure_nonmedical_exemption_policy_interventions.py (before the
#      cumulative sums; the supplemental NME ban figure draws with it). At
#      HEAD the supplemental figure also had its own copy of the clip in an
#      unused first pass (it collected y-values that nothing read), which has
#      been deleted. The HEAD tables also divided the cases by the `scale` of
#      2025, which is no longer computed.
# Why they were taken out: with one seed for every scenario the draws are
# paired, so a scenario that lands on the wrong side of the baseline does so
# because of what the model learned (for example the negative nonmedical
# exemption slope of Ohio and the negative coverage slope of New York), not
# because of sampling noise. The bounds hid that and made the improving
# scenarios look more protective than the model says. Rule 3 hid it the
# same way: in a draw where the learned slope has the wrong sign (Ohio's
# nonmedical exemption slope) an intervention does raise cases, and the
# clip turned that into 0%. To put them back, add the rules at the marked
# places, and the first-year keys that rules 1 and 2 look up
# ("baseline" and "targeted_worsening" of `first`) must be among
# all_future_samples or a bound is silently not applied: assert that they are.

# the headline models of the panels (chosen with 06b_choose_headline_models.py)
# are read from outputs/headline_models.json where the main program starts
# panel a: a place with 10+ cases is an outbreak; S1's window of years it
# shows, and the later year, scored by table_headline_2026_validation.py
PERFORMANCE_LEVEL = 10
EARLIER_WINDOW = "2022-2025"
LATER_YEAR = 2026
HEADLINE_2026_TABLE = os.path.join(
    fig_dir, "table_supplemental_headline_2026_validation.csv"
)
# the cases of 2026 observed so far (the data run to September)
OBSERVED_2026_LABEL = "Observed\n(September, 2026)"


def panel_letter(ax, letter, x=-0.05, y=1.08):
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )


def plot_violin(ax, values, x, color, width):
    """A violin of the central 95% of `values` at x, with their median and
    interquartile range. Fewer than three different values have no density to
    draw: only the median is."""
    values = np.asarray(values)
    median = np.median(values)
    if np.unique(values).size < 3:
        ax.scatter([x], [median], color=color, s=60, zorder=5)
        return
    lower, upper = np.percentile(values, [2.5, 97.5])
    body = ax.violinplot(
        values[(values >= lower) & (values <= upper)],
        positions=[x],
        widths=width,
        showextrema=False,
    )
    for pc in body["bodies"]:
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        pc.set_alpha(0.3)
    q1, q3 = np.percentile(values, [25, 75])
    ax.errorbar(
        x,
        median,
        yerr=[[median - q1], [q3 - median]],
        fmt="o",
        markersize=7,
        ecolor=color,
        color=color,
        capsize=5,
        capthick=2,
    )


# the transparency of the bars of the earlier years and of the later year
EARLIER_ALPHA, LATER_ALPHA = 0.9, 0.4
# the lines of the hatch of the outbreak cases: heavy, to be seen
HATCH_LINEWIDTH = 2.4


def plot_performance_bars(ax, values, earlier_period, later_period):
    """
    Panel a: the performance of the two models (performance_values) in each
    metric. A metric has eight bars, the two models alternating: all cases in
    the earlier years, all cases in the later year, then the outbreak cases in
    the same two periods. The later period is lighter and outbreak cases are
    hatched.
    """
    colors = {
        "Statistical": STATISTICAL_COLOR,
        "Mechanistic": MECHANISTIC_COLOR,
    }
    bars = [
        (condition, period, model)
        for condition in ("All cases", "Outbreak cases")
        for period in (earlier_period, later_period)
        for model in colors
    ]
    width = 0.1
    with plt.rc_context({"hatch.linewidth": HATCH_LINEWIDTH}):
        for i, metric in enumerate(PERFORMANCE_METRICS):
            for j, (condition, period, model) in enumerate(bars):
                row = values.filter(
                    (pl.col("metric") == metric)
                    & (pl.col("model") == model)
                    & (pl.col("condition") == condition)
                    & (pl.col("period") == period)
                )
                assert row.height == 1, (metric, model, condition, period)
                value = row["value"][0]
                lower, upper = row["lower"][0], row["upper"][0]
                color = colors[model]
                x = i + (j - (len(bars) - 1) / 2) * width
                face = mcolors.to_rgba(
                    color,
                    EARLIER_ALPHA if period == earlier_period else LATER_ALPHA,
                )
                if condition == "Outbreak cases":
                    # heavy hatch: white lines on the dark bars, lines of the
                    # model's colour on the light ones, outline on top
                    ax.bar(
                        x,
                        value,
                        width=width,
                        facecolor=face,
                        edgecolor="white"
                        if period == earlier_period
                        else color,
                        linewidth=0,
                        hatch="///",
                    )
                    ax.bar(
                        x,
                        value,
                        width=width,
                        fill=False,
                        edgecolor=color,
                        linewidth=1.5,
                    )
                else:
                    ax.bar(
                        x,
                        value,
                        width=width,
                        facecolor=face,
                        edgecolor=color,
                        linewidth=1.2,
                    )
                if not (np.isnan(lower) or np.isnan(upper)):
                    ax.errorbar(
                        x,
                        value,
                        yerr=[[value - lower], [upper - value]],
                        fmt="none",
                        ecolor="black",
                        capsize=3,
                        elinewidth=1.5,
                        alpha=0.8,
                    )
    ax.set_xticks(range(len(PERFORMANCE_METRICS)))
    ax.set_xticklabels(list(PERFORMANCE_METRICS))
    ax.set_ylim(0, 100)
    ax.set_ylabel("Performance (%)")
    # one legend for the models, one for what the shade and the hatch mean (the
    # hatch of its swatch is as heavy as that of the bars)
    with plt.rc_context({"hatch.linewidth": HATCH_LINEWIDTH}):
        models_legend = ax.legend(
            handles=[Patch(facecolor=c, label=m) for m, c in colors.items()],
            loc=(0.0, 1.05),
            edgecolor="black",
            fontsize=22,
        )
        ax.add_artist(models_legend)
        grey = "gray"
        ax.legend(
            handles=[
                Patch(
                    facecolor=mcolors.to_rgba(grey, EARLIER_ALPHA),
                    edgecolor=grey,
                    label=earlier_period.replace("-", "–"),
                ),
                Patch(
                    facecolor=mcolors.to_rgba(grey, LATER_ALPHA),
                    edgecolor=grey,
                    label=later_period,
                ),
                Patch(facecolor="white", edgecolor=grey, label="All cases"),
                Patch(
                    facecolor="white",
                    edgecolor=grey,
                    hatch="///",
                    label="Outbreak cases",
                ),
            ],
            loc="lower right",
            bbox_to_anchor=(1.0, 1.05),
            edgecolor="black",
            fontsize=22,
            ncol=2,
        )


def plot_2026_totals(ax, observed, statistical, mechanistic):
    """Panel b: the cases of 2026 in all places together, observed so far (the
    star) and as the two models predict them (draws conditioned on the imports
    of 2025)."""
    ax.scatter(0, observed, color="k", marker="*", s=200, zorder=10)
    plot_violin(ax, statistical, 1, STATISTICAL_COLOR, 0.6)
    plot_violin(ax, mechanistic, 2, MECHANISTIC_COLOR, 0.6)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([OBSERVED_2026_LABEL, "Statistical", "Mechanistic"])
    ax.set_ylabel("2026 nationwide cases (N)")
    ax.set_yscale("log")
    # a tick at every power of ten that the plotted values reach
    low = min(
        observed,
        np.percentile(statistical, 2.5),
        np.percentile(mechanistic, 2.5),
    )
    high = max(
        observed,
        np.percentile(statistical, 97.5),
        np.percentile(mechanistic, 97.5),
    )
    ticks = [
        10**k
        for k in range(
            int(np.floor(np.log10(max(low, 1)))),
            int(np.ceil(np.log10(high))) + 1,
        )
    ]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:,}" for t in ticks])


def plot_2026_risk_percentiles(ax, cases, percentiles):
    """Panel c: the risk percentile of the places by their cases of 2026 so far
    (none, 1 to 9, 10 or more), for each ranking in `percentiles` ({name: the
    percentile of every place})."""
    colors = {
        "Statistical": STATISTICAL_COLOR,
        "Mechanistic": MECHANISTIC_COLOR,
        "Vaccine coverage only": COVERAGE_COLOR,
    }
    buckets = case_bucket(cases)
    offsets = np.linspace(-0.28, 0.28, len(colors))
    for bucket in range(len(BUCKET_LABELS)):
        for (name, color), offset in zip(colors.items(), offsets):
            values = np.asarray(percentiles[name])[buckets == bucket]
            if values.size > 0:
                plot_violin(ax, values, bucket + offset, color, 0.26)
    ax.set_xticks(range(len(BUCKET_LABELS)))
    ax.set_xticklabels(BUCKET_LABELS)
    ax.set_xlabel("Observed cases (N)")
    ax.set_ylabel("Predicted county\noutbreak size percentile")
    ax.set_ylim([0, 100])
    ax.legend(
        handles=[
            Patch(facecolor=color, alpha=0.5, label=name)
            for name, color in colors.items()
        ],
        loc=(0.0, 1.05),
        edgecolor="black",
        fontsize=18,
        ncol=len(colors),
    )


def plot_future_scenario_cases(
    future_years,
    plot_scenarios,
    scenario_colors,
    all_future_samples,
    importation_samples,
    importation_count_samples,
    fig,
    ax,
    scenario_labels=None,
):
    # the main figure's labels, unless the caller passes its own
    if scenario_labels is None:
        scenario_labels = {
            "no_change": "No changes in vaccination",
            "baseline": "Vaccination trends continue",
            "targeted_worsening": "Vaccinations universally decline",
            "universal_worsening": "Vaccinations universally decline by 50% more",
            "targeted_improvement": "Universal 95% coverage by 2030",
            "universal_improvement": "1% improvement year-over-year",
        }

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

            # no scenario bounds here (see SCENARIO BOUNDS at the top of the file)
            combined_raw = (
                raw * importation_samples + importation_count_samples
            )
            combined = jnp.sum(combined_raw, axis=1)

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
    print("CUMULATIVE CASES BY SCENARIO (2027-2030)")
    print("=" * 60)
    for scenario in plot_scenarios:
        cumulative_samples = None
        for year in future_years:
            raw = all_future_samples[(scenario, year)]

            # no scenario bounds here (see SCENARIO BOUNDS at the top of the file)
            combined_raw = (
                raw * importation_samples + importation_count_samples
            )
            combined = jnp.sum(combined_raw, axis=1)

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

    # geopandas cannot plot an empty frame (every county may have a value)
    if bool(no_data_mask.any()):
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
        norm = mcolors.LogNorm(vmin=counties_with_data["value"].min(), vmax=1)  # type: ignore
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
            # midpoint of each bucket (the bar spaces buckets evenly and
            # interpolates within them), so every label is centered on its
            # color block
            cbar.set_ticks([0.75, 3.75, 7.5, 12.5, 17.5, 60])
            cbar.set_ticklabels(
                ["< 2.5%", "2.5%-5%", "5%-10%", "10%-15%", "15%-20%", "≥20%"]
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


def create_supplemental_future_projections_figure(
    future_years,
    scenarios,
    scenario_colors,
    all_future_samples,
    importation_samples,
    importation_count_samples,
    output_path,
):
    """
    Create a standalone figure showing alternative future projection scenarios.

    Parameters
    ----------
    future_years : list
        Years to plot (e.g., [2027, 2028, 2029, 2030])
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
        fig=fig,
        ax=ax,
        # supplemental figure: label the scenarios as the tables do
        scenario_labels=VACCINATION_SCENARIO_LABELS,
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
        f"Supplemental figure saved to {output_path}.png and {output_path}.pdf"
    )


def plot_excess_cases_vs_no_change(
    future_years,
    plot_scenarios,
    scenario_colors,
    all_future_samples,
    importation_samples,
    importation_count_samples,
    fig,
    ax,
):
    """
    Plot excess cases relative to 'no_change' scenario using paired sample differences.

    Positive values indicate more cases than no_change scenario.
    Negative values indicate fewer cases than no_change scenario.
    """
    scenario_labels = {
        scenario: VACCINATION_SCENARIO_LABELS[scenario]
        for scenario in [
            "baseline",
            "targeted_worsening",
            "universal_worsening",
            "targeted_improvement",
            "universal_improvement",
        ]
    }

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

            # no scenario bounds here (see SCENARIO BOUNDS at the top of the file)
            combined_scenario = (
                raw_scenario * importation_samples + importation_count_samples
            )
            total_scenario = jnp.sum(combined_scenario, axis=1)

            # Get no_change samples
            raw_no_change = all_future_samples[("no_change", year)]
            combined_no_change = (
                raw_no_change * importation_samples + importation_count_samples
            )
            total_no_change = jnp.sum(combined_no_change, axis=1)

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
    print("CUMULATIVE EXCESS CASES VS NO CHANGE (2027-2030)")
    print("=" * 60)
    for scenario in plot_scenarios:
        cumulative_excess = None
        for year in future_years:
            raw_scenario = all_future_samples[(scenario, year)]

            # no scenario bounds here (see SCENARIO BOUNDS at the top of the file)
            combined_scenario = (
                raw_scenario * importation_samples + importation_count_samples
            )
            total_scenario = jnp.sum(combined_scenario, axis=1)

            raw_no_change = all_future_samples[("no_change", year)]
            combined_no_change = (
                raw_no_change * importation_samples + importation_count_samples
            )
            total_no_change = jnp.sum(combined_no_change, axis=1)

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
    output_path,
):
    """
    Create a supplemental figure showing excess cases relative to no_change scenario.

    Parameters
    ----------
    future_years : list
        Years to plot (e.g., [2027, 2028, 2029, 2030])
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
    output_path,
):
    """
    Create a table with risk percentiles for each vaccination scenario by
    county and year (one row per county, year and scenario) and save it.

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
    output_path : str
        Path to save CSV
    """
    results_df = scenario_tables.vaccination_scenario_effects(
        vaccination_scenarios,
        future_years,
        future_base_df,
        all_future_samples,
        importation_samples,
        importation_count_samples,
        VACCINATION_SCENARIO_LABELS,
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
    output_path,
):
    """
    Create state-level vaccination scenario effects table (one row per state,
    year and scenario) and save it. The parameters are those of
    create_vaccination_scenario_effects_table.
    """
    results_df = scenario_tables.state_vaccination_scenario_effects(
        vaccination_scenarios,
        future_years,
        future_base_df,
        all_future_samples,
        importation_samples,
        importation_count_samples,
        VACCINATION_SCENARIO_LABELS,
    )
    results_df.write_csv(output_path)
    print(
        f"\nState-level vaccination scenario effects table saved to {output_path}"
    )
    print(f"Total rows: {len(results_df)}")

    return results_df


def plot_case_concentration_curves(
    validation_panels,
    plot_scenarios,
    scenario_colors,
    all_future_samples,
    importation_samples,
    importation_count_samples,
    output_path,
):
    """
    Case-concentration (Lorenz-style) curves in two rows. In every panel the x
    axis is the risk percentile (0 for the lowest place, 100 for the highest;
    outbreak_size.ranking) and the y axis the cumulative % of the cases in the
    places at or below it: a ranking that sends the cases to the highest
    percentiles stays low until the right edge, the diagonal is a ranking that
    tells nothing.

    Row 1, validation: one panel for each of `validation_panels`, a list of
    (title, {ranking: curve over concentration.GRID}). Each curve is a ranking
    of the places (the statistical and the mechanistic model, the vaccine
    coverage, the population and the imports alone, and the observed cases, the
    best ranking there is) against the observed cases.
    Row 2, projections: the scenarios of `plot_scenarios` in 2027, in 2030 and
    cumulatively over 2027-2030, the places ranked by their projected cases
    (importations included) and the cases of a place its mean over the draws
    (concentration.scenario_curve). The scenarios are drawn as they come, with
    no bounds (see SCENARIO BOUNDS at the top of the file).
    """
    scenario_labels = {
        scenario: VACCINATION_SCENARIO_LABELS[scenario]
        for scenario in [
            "no_change",
            "baseline",
            "targeted_worsening",
            "targeted_improvement",
        ]
    }
    ranking_styles = {
        STATISTICAL: (STATISTICAL_COLOR, "-"),
        MECHANISTIC: (MECHANISTIC_COLOR, "-"),
        COVERAGE: (COVERAGE_COLOR, "-"),
        POPULATION: (POPULATION_COLOR, "-"),
        IMPORTS: (IMPORTS_COLOR, "-"),
        OBSERVED: ("black", "--"),
    }
    panel_defs = [
        ("2027", [2027]),
        ("2030", [2030]),
        ("Cumulative 2027-2030", [2027, 2028, 2029, 2030]),
    ]

    def scenario_combined_samples(scenario, year):
        raw = all_future_samples[(scenario, year)]
        # no scenario bounds here (see SCENARIO BOUNDS at the top of the file)
        return raw * importation_samples + importation_count_samples

    fig = plt.figure(figsize=(24, 16))
    grid = fig.add_gridspec(2, 6)
    top = [fig.add_subplot(grid[0, 0:3]), fig.add_subplot(grid[0, 3:6])]
    bottom = [fig.add_subplot(grid[1, i : i + 2]) for i in (0, 2, 4)]
    for row in (top, bottom):
        for ax in row[1:]:
            ax.sharey(row[0])
    for row in (top, bottom):
        for ax in row:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            ax.tick_params(axis="x", direction="out")
            ax.tick_params(axis="y", direction="out")
            for spine in ax.spines.values():
                spine.set_position(("outward", 5))
            ax.set_axisbelow(True)
            ax.plot(
                GRID, GRID, color="lightgray", linewidth=1.5, linestyle=":"
            )
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_xlabel("County risk percentile")
        row[0].set_ylabel("Cumulative % of cases")
        for ax in row[1:]:
            plt.setp(ax.get_yticklabels(), visible=False)

    for ax, (title, curves) in zip(top, validation_panels):
        for name in RANKINGS:
            color, linestyle = ranking_styles[name]
            ax.plot(
                GRID,
                curves[name],
                color=color,
                linestyle=linestyle,
                linewidth=2.5,
                label=name,
            )
        ax.set_title(title)
    top[0].legend(loc="upper left", edgecolor="black", fontsize=18)

    for ax, (title, years) in zip(bottom, panel_defs):
        for scenario in plot_scenarios:
            combined = None
            for year in years:
                cy = scenario_combined_samples(scenario, year)
                combined = cy if combined is None else combined + cy
            ax.plot(
                GRID,
                scenario_curve(combined),
                color=scenario_colors[scenario],
                linewidth=2.5,
                label=scenario_labels[scenario],
            )
        ax.set_title(title)
    bottom[0].legend(loc="upper left", edgecolor="black", fontsize=18)

    fig.tight_layout()
    fig.savefig(f"{output_path}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{output_path}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(
        f"Supplemental case-concentration figure saved to "
        f"{output_path}.png and {output_path}.pdf"
    )


def validation_year_rankings(year, observed, import_df, headline):
    """The risk percentile of the places of a validation year under each
    ranking of the case-concentration figure (concentration.rankings), from the
    draws of the headline models trained on the years before `year`, and their
    observed cases."""
    places, keep, _ = validation_year_rows(
        year, output_dir, import_df, observed, params["seed"], list(COVARIATES)
    )
    in_rows = int(places["cases"].sum())
    in_files = int(observed.filter(pl.col("year") == year)["cases"].sum())
    print(
        f"{year}: {places.height} places; cases of the places {in_rows}, "
        f"cases of the files {in_files}"
    )
    if in_rows != in_files:
        print(
            "WARNING: cases of the files that are on no place of the "
            f"validation rows: {in_files - in_rows}"
        )
    design = headline.statistical
    size_statistical = np.asarray(
        keep_rows(
            load_samples(
                os.path.join(
                    output_dir,
                    validation_file_name(
                        training_years(design.start_year, year),
                        design.pooled,
                        design.spatial,
                        design.covariates,
                        GLOBALS_PRIOR_SD,
                    ),
                )
            ),
            keep,
            f"{year} fold draws",
        ),
        dtype=np.float64,
    )
    mechanistic_file = load_samples(
        os.path.join(
            output_dir,
            "predictive_final_outbreak_size_predictions_"
            f"{training_years(headline.mechanistic_start_year, year)}.npz",
        )
    )
    size_mechanistic = np.asarray(
        keep_rows(mechanistic_file["cases"], keep, f"{year} mechanistic"),
        dtype=np.float64,
    )
    coverage = np.asarray(
        keep_rows(mechanistic_file["coverage"], keep, f"{year} coverage")[0],
        dtype=float,
    )
    cases = places["cases"].to_numpy()
    return (
        rankings(
            size_statistical,
            size_mechanistic,
            np.vstack(places["importation_probability_samples"].to_numpy()).T,
            np.vstack(places["expected_importations_samples"].to_numpy()).T,
            coverage,
            places["pop"].to_numpy(),
            cases,
        ),
        cases,
    )


if __name__ == "__main__":
    # the inputs that only the 2026 panels read: stop here, before the large
    # files are loaded, if one is missing, and say which script makes it
    headline = read_headline_models(
        os.path.join(output_dir, HEADLINE_FILE_NAME), fig_dir=fig_dir
    )
    design = headline.statistical
    HEADLINE_MODEL = design.name
    MECHANISTIC_MODEL = headline.mechanistic
    fold_years = training_years(design.start_year, LATER_YEAR)
    fold_file = os.path.join(
        output_dir,
        validation_file_name(
            fold_years,
            design.pooled,
            design.spatial,
            design.covariates,
            GLOBALS_PRIOR_SD,
        ),
    )
    mechanistic_fold_years = training_years(
        headline.mechanistic_start_year, LATER_YEAR
    )
    mechanistic_fold_file = os.path.join(
        output_dir,
        "predictive_final_outbreak_size_predictions_"
        f"{mechanistic_fold_years}.npz",
    )
    with_2026 = "00_explore_data.py and 01_assemble_analysis_dataset.py"
    inputs_2026 = {
        HEADLINE_2026_TABLE: "table_headline_2026_validation.py",
        fold_file: "06_rolling_validation.py --headline-2026",
        mechanistic_fold_file: "06_rolling_validation.py --headline-2026",
        os.path.join(data_dir, "analysis_dataset_with_2026.csv"): with_2026,
        os.path.join(data_dir, "missing_counties_with_2026.csv"): with_2026,
    }
    absent = {p: by for p, by in inputs_2026.items() if not os.path.exists(p)}
    if absent:
        raise FileNotFoundError(
            "the panels of 2026 need files that are not there yet:\n"
            + "\n".join(f"  {p} (made by {by})" for p, by in absent.items())
        )

    # panels: a (performance), b and c (2026), d and e (maps), f (scenarios)
    fig = plt.figure(figsize=(20, 30))
    grid = fig.add_gridspec(4, 2, height_ratios=[0.9, 1, 1, 0.9])
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])
    ax_d = fig.add_subplot(grid[2, 0])
    ax_e = fig.add_subplot(grid[2, 1])
    ax_f = fig.add_subplot(grid[3, :])

    for temp in [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f]:
        temp.spines["top"].set_visible(False)
        temp.spines["right"].set_visible(False)
        temp.get_xaxis().tick_bottom()
        temp.get_yaxis().tick_left()
        temp.tick_params(axis="x", direction="out")
        temp.tick_params(axis="y", direction="out")
        for spine in temp.spines.values():
            spine.set_position(("outward", 5))
        temp.set_axisbelow(True)

    s1 = pl.read_csv(
        os.path.join(
            fig_dir, "supplemental_data_s1_county_level_prediction_scoring.csv"
        )
    )

    # panel a: the performance of the two models in S1 (the earlier years) and
    # in the later year (the headline table of that year)
    performance = performance_values(
        s1,
        pl.read_csv(HEADLINE_2026_TABLE),
        {"Statistical": HEADLINE_MODEL, "Mechanistic": MECHANISTIC_MODEL},
        PERFORMANCE_LEVEL,
        EARLIER_WINDOW,
        LATER_YEAR,
    )
    plot_performance_bars(ax_a, performance, EARLIER_WINDOW, str(LATER_YEAR))

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
        probability_samples = pickle.load(f)
    with open(
        os.path.join(output_dir, "empirical_importation_count_samples.pkl"),
        "rb",
    ) as f:
        count_samples = pickle.load(f)
    import_df = import_df.with_columns(
        importation_probability_samples=pl.Series(probability_samples),
        expected_importations_samples=pl.Series(count_samples),
    )
    # --- panels b and c: 2026 so far. The headline model and the mechanistic
    # model, trained on 2019-2025, against the cases of 2026 (to September)
    with open(os.path.join(output_dir, "import_scaler_2025.pkl"), "rb") as f:
        import_scaler_2025 = pickle.load(f)
    analysis_2026 = observed_county_years(
        pl.read_csv(
            os.path.join(data_dir, "analysis_dataset_with_2026.csv"),
            schema_overrides={"fips": pl.String},
        ),
        pl.read_csv(
            os.path.join(data_dir, "missing_counties_with_2026.csv"),
            schema_overrides={"fips": pl.String},
        ),
    )
    places_2026, keep_2026, _ = validation_year_rows(
        LATER_YEAR,
        output_dir,
        import_df,
        analysis_2026,
        params["seed"],
        list(COVARIATES),
    )
    observed_2026 = int(
        analysis_2026.filter(pl.col("year") == LATER_YEAR)["cases"].sum()
    )
    print(
        f"{LATER_YEAR}: {places_2026.height} places; cases of the places "
        f"{int(places_2026['cases'].sum())}, cases of the files {observed_2026}"
    )

    # the outbreak size draws of the places (the rows of the 2026 projection
    # dataset without the twins and the region members)
    fold_samples = load_samples(fold_file)
    size_statistical = np.asarray(
        keep_rows(fold_samples, keep_2026, "2026 fold draws"), dtype=np.float64
    )
    mechanistic_file = load_samples(mechanistic_fold_file)
    size_mechanistic = np.asarray(
        keep_rows(
            mechanistic_file["cases"], keep_2026, "2026 mechanistic draws"
        ),
        dtype=np.float64,
    )
    coverage_2026 = np.asarray(
        keep_rows(mechanistic_file["coverage"], keep_2026, "2026 coverage")[0],
        dtype=float,
    )
    expected_imports = np.vstack(
        places_2026["expected_importations_samples"].to_numpy()
    ).T
    import_probability = np.vstack(
        places_2026["importation_probability_samples"].to_numpy()
    ).T
    assert (
        size_statistical.shape
        == size_mechanistic.shape
        == expected_imports.shape
    ), "the draws of the 2026 places do not line up"

    # panel b: the imports of 2026 conditioned on those of 2025 (the scaler of
    # the projections), the same imports drawn for both models
    totals = {
        name: conditioned_national_totals(
            size,
            expected_imports,
            import_scaler_2025,
            np.random.default_rng([params["seed"], LATER_YEAR, 3]),
        )
        for name, size in (
            ("Statistical", size_statistical),
            ("Mechanistic", size_mechanistic),
        )
    }
    for name, draws in totals.items():
        lower, median, upper = np.percentile(draws, [2.5, 50, 97.5])
        print(
            f"{LATER_YEAR} cases, {name}: median {median:,.0f}, 95% interval "
            f"[{lower:,.0f}, {upper:,.0f}]; observed so far {observed_2026:,} "
            f"({100 * (draws <= observed_2026).mean():.1f}th percentile)"
        )
    plot_2026_totals(
        ax_b, observed_2026, totals["Statistical"], totals["Mechanistic"]
    )

    # panel c: the rank of every place against its cases of 2026 so far
    percentiles_2026 = {
        "Statistical": risk_percentile(
            size_statistical * import_probability + expected_imports
        ),
        "Mechanistic": risk_percentile(
            size_mechanistic * import_probability + expected_imports
        ),
        "Vaccine coverage only": percentile_of_scores(-coverage_2026),
    }
    plot_2026_risk_percentiles(
        ax_c, places_2026["cases"].to_numpy(), percentiles_2026
    )
    # the rankings of all the places of 2026 against its cases, for the
    # case-concentration curves of the later year
    curves_2026 = validation_curves(
        rankings(
            size_statistical,
            size_mechanistic,
            import_probability,
            expected_imports,
            coverage_2026,
            places_2026["pop"].to_numpy(),
            places_2026["cases"].to_numpy(),
        ),
        places_2026["cases"].to_numpy(),
    )
    del (
        size_statistical,
        size_mechanistic,
        expected_imports,
        import_probability,
    )

    df = pl.read_csv(
        os.path.join(output_dir, "assembled_projection_results.csv"),
        schema_overrides={"fips": pl.String},
    )

    # panel d: the probability of an import (the median over the draws),
    # conditioned on the imports of 2025 like the projections of panel f
    map_fips_code_data(
        df["fips"],
        pl.Series(
            scale_importation_probability(
                df["importation_probability_median"].to_numpy(),
                import_scaler_2025,
            )
        ),
        fig,
        ax_d,
        cbar_label="Probability of ≥1 international importation",
        cbar_type="probability",
        point_data=None,
    )

    future_years_plot = [2027, 2028, 2029, 2030]
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

    # the counties of the projections: the baseline scenario's 2027 rows (05
    # refuses a scenario that loses a county, so all scenarios share them)
    future_base = pl.read_csv(
        os.path.join(output_dir, "projection_dataset_baseline_2027.csv"),
        schema_overrides={"fips": pl.String},
    )

    # the importation draws of the counties of the projections (the projection
    # rows of the validation year whose importation posterior serves every year
    # after 2025), conditioned on the level of 2025 once, here: every
    # table and figure below uses these two arrays
    importation_probability, importation_counts = future_importation_draws(
        import_df, future_base
    )
    importation_samples_scaled = scale_importation_probability(
        jnp.array(importation_probability), import_scaler_2025
    )
    importation_count_samples_scaled = (
        jnp.array(importation_counts) * import_scaler_2025
    )
    del importation_probability, importation_counts

    # no scenario bounds in the figure (see SCENARIO BOUNDS at the top of the
    # file): no 2026-fold floor for no_change is built here any more
    # the draws of every scenario and year are loaded where they are used, one
    # at a time (together they are 5 GB or more; the figures and tables below
    # need one or two at once)
    all_future_samples = LazyDraws(
        {
            (scenario, future_year): os.path.join(
                output_dir,
                f"risk_projections_{scenario}_{future_year}_globals_prior_sd={GLOBALS_PRIOR_SD}.npz",
            )
            for scenario in scenarios_ordered
            for future_year in future_years_plot
        },
        n_columns=future_base.height,
    )

    # the validation years, each ranked within itself, averaged with the same
    # weight for each (as the table of the scoring averages its years)
    years_curves = [
        validation_curves(
            *validation_year_rankings(year, analysis_2026, import_df, headline)
        )
        for year in BEST_MODEL_YEARS
    ]
    plot_case_concentration_curves(
        validation_panels=[
            (
                f"Validation {EARLIER_WINDOW} (the years averaged)",
                {
                    name: mean_curve([curves[name] for curves in years_curves])
                    for name in RANKINGS
                },
            ),
            (f"Validation {LATER_YEAR} (to September)", curves_2026),
        ],
        plot_scenarios=[
            "no_change",
            "baseline",
            "targeted_improvement",
            "targeted_worsening",
        ],
        scenario_colors=scenario_colors,
        all_future_samples=all_future_samples,
        importation_samples=importation_samples_scaled,
        importation_count_samples=importation_count_samples_scaled,
        output_path=os.path.join(
            fig_dir, "figure_supplemental_case_concentration_curves"
        ),
    )

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
        importation_samples=importation_samples_scaled,
        importation_count_samples=importation_count_samples_scaled,
        fig=fig,
        ax=ax_f,
    )

    map_fips_code_data(
        df["fips"],
        df["unconditioned_count_rank"],
        fig,
        ax_e,
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

    panel_letter(ax_a, "a", y=1.3)
    panel_letter(ax_b, "b", x=-0.12)
    panel_letter(ax_c, "c", x=-0.12, y=1.2)
    panel_letter(ax_d, "d")
    panel_letter(ax_e, "e")
    panel_letter(ax_f, "f", x=-0.025, y=1.1)

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
        importation_samples=importation_samples_scaled,
        importation_count_samples=importation_count_samples_scaled,
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
        importation_samples=importation_samples_scaled,
        importation_count_samples=importation_count_samples_scaled,
        output_path=os.path.join(
            fig_dir, "figure_supplemental_future_projections"
        ),
    )

    vaccination_scenario_effects = create_vaccination_scenario_effects_table(
        vaccination_scenarios=scenarios_ordered,
        future_years=future_years_plot,
        future_base_df=future_base,
        all_future_samples=all_future_samples,
        importation_samples=importation_samples_scaled,
        importation_count_samples=importation_count_samples_scaled,
        output_path=os.path.join(
            fig_dir,
            "supplemental_data_s3a_vaccination_scenario_effects_by_county.csv",
        ),
    )

    state_vaccination_scenario_effects = create_state_vaccination_scenario_effects_table(
        vaccination_scenarios=scenarios_ordered,
        future_years=future_years_plot,
        future_base_df=future_base,
        all_future_samples=all_future_samples,
        importation_samples=importation_samples_scaled,
        importation_count_samples=importation_count_samples_scaled,
        output_path=os.path.join(
            fig_dir,
            "supplemental_data_s3b_vaccination_scenario_effects_by_state.csv",
        ),
    )
