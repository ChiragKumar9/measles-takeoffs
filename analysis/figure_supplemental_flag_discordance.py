"""
Where the statistical and the mechanistic model flag different places (outbreak
cases of the places with a case, the highest 10% of their scores flagged, 2022-2026):
for the places with 5+ outbreak cases that at least one of the models flagged, the
percentile of each model's score among the places with a case (a; the flagged places
are those at or above the 90th percentile, inside the top bucket), how often the two models disagree in each year (b), and how the
places only one of them flagged differ in their sociodemographic characteristics (c).
Places with 5+ outbreak cases that neither model flagged are left out.

Reads outputs/flag_discordance_places.csv (2022-2025) and
outputs/flag_discordance_places_2026.csv (the cases of 2026 to date), both written by
table_flag_discordance_places.py: the places with 5+ outbreak cases that the headline
statistical or mechanistic model (chosen with 06b_choose_headline_models.py) flagged,
with their covariates and the percentile of each model's score. The scores of the other places and of the other
models are not saved. Panel c shows only the covariates on which
the places only one model flagged differ by at least MIN_GAP percentile points;
all of them are in outputs/flag_discordance_covariate_percentiles.csv.
Writes figures/figure_supplemental_flag_discordance.{png,pdf}.
"""

import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import rc
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation, blended_transform_factory

from outbreak_size.flag_discordance import (
    COVARIATES,
    GROUPS,
    LEVEL,
    MECHANISTIC_SCORE,
    OUTCOME,
    PLACES_FILE_NAME,
    PLACES_FILE_NAME_2026,
    SHARE,
    STATISTICAL_SCORE,
    covariate_gaps,
    different_covariates,
    discordance_rates,
    group_counts,
    group_medians,
    group_percentiles,
)
from outbreak_size.model_colors import MECHANISTIC_COLOR, STATISTICAL_COLOR

fig_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)

font = {"family": "Nimbus Roman", "size": 28}
rc("font", **font)

STEM = "flag_discordance"
# panel c shows a covariate if the medians of the places only one model flagged
# differ by at least this many percentile points
MIN_GAP = 10
COVARIATE_FILE = f"{STEM}_covariate_percentiles.csv"
# statistical blue and mechanistic orange, the colours of the two models in
# every figure (outbreak_size.model_colors); grey for the places both flag and
# for neither
COLORS = {
    "Both": "#525252",
    "Statistical only": STATISTICAL_COLOR,
    "Mechanistic only": MECHANISTIC_COLOR,
    "Neither": "#bdbdbd",
}
# the groups compared on their characteristics: the two groups the models disagree
# on, and the places both flag as the reference
COMPARED_GROUPS = ["Statistical only", "Mechanistic only", "Both"]
# the groups drawn: the places flagged by at least one model
SHOWN_GROUPS = [g for g in GROUPS if g != "Neither"]
# the five buckets of a place's score percentile in panel a, with the colours and labels of the
# percentile map of figure_validation_and_risk_projections.py
PERCENTILE_BOUNDARIES = [0, 20, 40, 60, 80, 100]
PERCENTILE_CMAP = mcolors.ListedColormap(
    ["#fcbba1", "#fc9272", "#fb6a4a", "#de2d26", "#a50f15"]
)
PERCENTILE_LABELS = ["Bottom 20%", "20%-40%", "40%-60%", "60%-80%", "Top 20%"]


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis="x", direction="out")
    ax.tick_params(axis="y", direction="out")
    for spine in ax.spines.values():
        spine.set_position(("outward", 5))
    ax.set_axisbelow(True)


def panel_letter(ax, letter, x=-0.05):
    return ax.text(
        x,
        1.08,
        letter,
        transform=ax.transAxes,
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )


def make_figure(grouped, out_path, covariate_path):
    """Draws the figure from the flagged places with 5+ outbreak cases
    (flag_discordance.discordance_places: their group, the percentile of each
    model's score and of each covariate) and saves it as out_path.png and
    out_path.pdf; the percentiles of every covariate go to covariate_path."""
    n = grouped.height
    print(
        f"{n} flagged places with {LEVEL}+ outbreak cases; groups:",
        grouped["group"].value_counts().rows(),
    )
    # the places only one model flagged, each with the percentile of both
    # models' scores among the places with a case of the year
    pct = [
        f"{STATISTICAL_SCORE} percentile",
        f"{MECHANISTIC_SCORE} percentile",
    ]
    for group, only, other in [
        ("Statistical only", "statistical", "mechanistic"),
        ("Mechanistic only", "mechanistic", "statistical"),
    ]:
        print(
            f"places with {LEVEL}+ outbreak cases that only the {only} model "
            f"flagged (not the {other} model); percentiles of the statistical "
            "and mechanistic scores:"
        )
        for row in (
            grouped.filter(pl.col("group") == group)
            .sort(
                "year", OUTCOME, descending=[False, True], maintain_order=True
            )
            .iter_rows(named=True)
        ):
            print(
                f"  {row['county']}, {row['state']}, {row['year']}: "
                f"{row[OUTCOME]} outbreak cases; "
                f"statistical {row[pct[0]]:.0f}, mechanistic {row[pct[1]]:.0f}"
            )
    # how often the models disagree, among the places with enough cases that
    # at least one of them flagged
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(
            f"discordance among the {LEVEL}+ case places flagged (top "
            f"{SHARE:.0%}) by at least one model; each share's denominator is "
            "the places the named model flagged:"
        )
        print(
            discordance_rates(group_counts(grouped)).with_columns(
                pl.col("^Of them.*$").mul(100).round(1)
            )
        )

    fig = plt.figure(figsize=(30, 36))
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[0.55, 0.55, 1.6],
        height_ratios=[1, 0.95],
        wspace=0.35,
        hspace=0.38,
    )
    ax_dots = fig.add_subplot(grid[0, 0])
    ax_cases = fig.add_subplot(grid[0, 1], sharey=ax_dots)
    # the space under b is left blank
    right = grid[0, 2].subgridspec(2, 1, hspace=0.3)
    ax_b = fig.add_subplot(right[0])
    # the characteristics have the whole row under the rest
    ax_c = fig.add_subplot(grid[1, :])
    for ax in (ax_dots, ax_cases, ax_b, ax_c):
        style(ax)

    # a: the places with 5+ outbreak cases flagged by a model, coloured by the percentile
    # of the model's score among the places with a case of the year
    y = np.arange(n)
    norm = mcolors.BoundaryNorm(
        PERCENTILE_BOUNDARIES, ncolors=len(PERCENTILE_LABELS)
    )
    cells = np.column_stack(
        [
            grouped[f"{score} percentile"].to_numpy()
            for score in (STATISTICAL_SCORE, MECHANISTIC_SCORE)
        ]
    )
    ax_dots.pcolormesh(
        [-0.5, 0.5, 1.5],
        np.arange(n + 1) - 0.5,
        cells,
        cmap=PERCENTILE_CMAP,
        norm=norm,
        edgecolors="white",
        linewidth=3,
    )
    ax_dots.set_xticks([0, 1])
    ax_dots.set_xticklabels(
        ["Statistical", "Mechanistic"], rotation=30, ha="right"
    )
    ax_dots.set_xlim(-0.5, 1.5)
    ax_dots.set_yticks([])
    ax_dots.set_ylim(n - 0.5, -0.5)
    ax_dots.set_ylabel(
        f"Places with {LEVEL}+ outbreak cases flagged by at least one model"
    )
    # the colour bar under the columns, as the percentile maps of the projections figure
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=PERCENTILE_CMAP, norm=norm),
        cax=ax_dots.inset_axes((0.0, -0.14, 1.0, 0.015)),
        orientation="horizontal",
    )
    colorbar.set_ticks([10, 30, 50, 70, 90])
    colorbar.set_ticklabels(PERCENTILE_LABELS)
    colorbar.ax.tick_params(labelsize=22, rotation=-60, length=0)
    # each label a touch to the right of its bucket's centre
    nudge = ScaledTranslation(0.08, 0, fig.dpi_scale_trans)
    for label in colorbar.ax.get_xticklabels():
        label.set_transform(label.get_transform() + nudge)
    ax_cases.barh(y, grouped[OUTCOME].to_numpy(), color="#9e9e9e", height=0.7)
    ax_cases.set_xscale("log")
    top = int(np.ceil(np.log10(grouped[OUTCOME].max())))
    ax_cases.set_xticks([10**k for k in range(1, top + 1)])
    ax_cases.set_xticklabels([f"{10**k:,}" for k in range(1, top + 1)])
    ax_cases.set_xlabel("Outbreak cases")
    ax_cases.set_xlim(left=4)
    plt.setp(ax_cases.get_yticklabels(), visible=False)
    ax_cases.tick_params(axis="y", length=0)
    ax_cases.grid(axis="x", color="#eeeeee", linewidth=1)
    letter_a = panel_letter(ax_dots, "a", x=-0.08)

    # b: how many places each group holds, by year and in all the years
    counts = group_counts(grouped)
    labels = counts["Year"].unique(maintain_order=True).to_list()
    bottoms = np.zeros(len(labels))
    for group in SHOWN_GROUPS:
        places_in_group = counts.filter(pl.col("Group") == group)[
            "Places"
        ].to_numpy()
        ax_b.bar(
            labels,
            places_in_group,
            bottom=bottoms,
            color=COLORS[group],
            edgecolor="white",
            linewidth=2,
        )
        bottoms += places_in_group
    ax_b.set_ylabel(f"Flagged (top {SHARE:.0%})")
    ax_b.set_yscale("symlog", linthresh=5)
    ticks = [
        t for t in (0, 1, 2, 5, 10, 20, 40, 80) if t <= bottoms.max() * 1.5
    ]
    ax_b.set_yticks(ticks)
    ax_b.set_yticklabels([str(t) for t in ticks])
    ax_b.set_ylim(0, bottoms.max() * 1.1)
    ax_b.legend(
        handles=[Patch(facecolor=COLORS[g], label=g) for g in SHOWN_GROUPS],
        loc=(0.0, 1.05),
        edgecolor="black",
        fontsize=22,
        ncol=len(SHOWN_GROUPS),
    )
    panel_letter(ax_b, "b")

    # c: percentile (among all places of the year) of the sociodemographic
    # characteristics of the places in each group: median and interquartile range.
    # Only the covariates on which the two groups the models disagree on differ
    # are drawn; the percentiles of all of them are saved
    summary = group_percentiles(grouped, list(COVARIATES), COMPARED_GROUPS)
    shown = different_covariates(summary, MIN_GAP)
    summary.with_columns(
        (pl.col("Covariate").is_in(shown)).alias(
            f"Shown (medians at least {MIN_GAP} points apart)"
        )
    ).write_csv(covariate_path)
    print(f"covariates shown in panel c: {shown}")
    print(
        f"no difference of {MIN_GAP}+ points: {[c for c in COVARIATES if c not in shown]}"
    )
    # the places only the statistical and only the mechanistic model flagged:
    # medians of every characteristic as percentiles among all places of the
    # year (the panel's scale) and in its own units (fractions: 0.10 = 10
    # percentage points), and whether the percentile gap is over MIN_GAP points
    gaps = covariate_gaps(
        summary,
        group_medians(grouped, list(COVARIATES), COMPARED_GROUPS),
        MIN_GAP,
    ).with_columns(pl.col("Covariate").replace(COVARIATES))
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
        print(
            "medians of the places only the statistical / only the "
            "mechanistic model flagged:"
        )
        print(gaps.with_columns(pl.col(pl.Float64).round(3)))
    # in percentage points, for the characteristics that are percentages or
    # fractions (the rest, a size or a log, have no percentage points)
    to_points = {
        COVARIATES[c]: 100.0 if c.startswith("fraction") else 1.0
        for c in COVARIATES
        if c.startswith("fraction")
        or c in ("estimated_mmr_cov", "nonmed_exempt_pct", "med_exempt_pct")
    }
    points = {
        row["Covariate"]: round(
            row["Gap in own units"] * to_points[row["Covariate"]], 1
        )
        for row in gaps.iter_rows(named=True)
        if row["Covariate"] in to_points
    }
    print(
        "gap in percentage points between the two groups (statistical only "
        f"minus mechanistic only): {points}; more than {MIN_GAP} points: "
        f"{[c for c, v in points.items() if abs(v) > MIN_GAP]}"
    )
    over = gaps.filter(pl.col(f"Gap over {MIN_GAP} points"))
    print(
        f"characteristics whose percentile medians differ by more than "
        f"{MIN_GAP} points between the two groups: "
        f"{over['Covariate'].to_list()}"
    )
    offsets = dict(
        zip(COMPARED_GROUPS, np.linspace(-0.25, 0.25, len(COMPARED_GROUPS)))
    )
    for j, covariate in enumerate(shown):
        for group in COMPARED_GROUPS:
            row = summary.filter(
                (pl.col("Covariate") == covariate) & (pl.col("Group") == group)
            ).row(0, named=True)
            if row["Places"] == 0:
                continue
            ypos = j + offsets[group]
            ax_c.plot(
                [row["Q1"], row["Q3"]],
                [ypos, ypos],
                color=COLORS[group],
                linewidth=5,
                solid_capstyle="butt",
                zorder=3,
            )
            ax_c.scatter(
                [row["Median"]],
                [ypos],
                s=260,
                color=COLORS[group],
                edgecolor="white",
                linewidth=2,
                zorder=4,
            )
    ax_c.axvline(50, color="#9e9e9e", linewidth=1.5, linestyle=":", zorder=1)
    ax_c.set_yticks(range(len(shown)))
    ax_c.set_yticklabels([COVARIATES[c] for c in shown], fontsize=36)
    ax_c.tick_params(axis="x", labelsize=32)
    ax_c.set_ylim(max(len(shown), 1) - 0.4, -0.6)
    ax_c.set_xlim(0, 100)
    ax_c.set_xlabel("Percentile", fontsize=34)
    ax_c.grid(axis="x", color="#eeeeee", linewidth=1)
    ax_c.legend(
        handles=[Patch(facecolor=COLORS[g], label=g) for g in COMPARED_GROUPS],
        loc=(0.0, 1.03),
        edgecolor="black",
        fontsize=30,
        ncol=len(COMPARED_GROUPS),
    )

    # c is as wide as the figure: its labels start where the y label of a starts, and its
    # letter is under the letter of a
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()  # type: ignore
    a_left = ax_dots.yaxis.label.get_window_extent(renderer).x0
    c_left = min(
        label.get_window_extent(renderer).x0
        for label in ax_c.get_yticklabels()
    )
    shift = (a_left - c_left) / fig.bbox.width
    position = ax_c.get_position()
    ax_c.set_position(
        (
            position.x0 + shift,
            position.y0,
            position.width - shift,
            position.height,
        )
    )
    ax_c.text(
        letter_a.get_window_extent(renderer).x1 / fig.bbox.width,
        1.08,
        "c",
        transform=blended_transform_factory(fig.transFigure, ax_c.transAxes),
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )

    fig.savefig(f"{out_path}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_path}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # 2022-2025 and 2026, from the one script (most cases first; the years of
    # equal cases keep their order)
    parts = []
    for name, script in (
        (PLACES_FILE_NAME, "table_flag_discordance_places.py"),
        (PLACES_FILE_NAME_2026, "table_flag_discordance_places.py"),
    ):
        path = os.path.join(output_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} is missing: run {script}")
        parts.append(pl.read_csv(path, schema_overrides={"fips": pl.String}))
    grouped = pl.concat(parts).sort(
        OUTCOME, descending=True, maintain_order=True
    )
    make_figure(
        grouped,
        os.path.join(fig_dir, f"figure_supplemental_{STEM}"),
        os.path.join(output_dir, COVARIATE_FILE),
    )
