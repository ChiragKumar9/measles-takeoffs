"""
The kindergarten cohort used for the doses of the interventions (outbreak_size.doses)
is a fifth of the population aged 5-9: children of kindergarten age are those aged five,
one of the five single years of age of the group. This figure compares that cohort with
counts of kindergartners (each point: a count on the vertical axis, the cohort of the
same place and year on the horizontal axis; on the line the two are equal):

a. the kindergarten population of each state and school year of the CDC (the school
   years starting 2020-2024), against the sum of the cohorts of the state's counties;
b. the kindergarten enrollment of each county and year of the two states that give it
   (California and Minnesota), against the cohort of the county.

The cohort of a school year (or year) is that of the ACS 5-year estimates of the year
that the school year starts in (the Census answers saved in outputs/census_cache.sqlite,
which has every county for the years 2020-2024). Writes
figures/figure_supplemental_kindergarten_cohort.{png,pdf}.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from addfips import AddFIPS
from matplotlib import rc

from outbreak_size.doses import census_answer, cohort_from_answer
from outbreak_size.utils import CENSUS_CACHE_PATH, CensusCache

repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(repo, "data")
output_dir = os.path.join(repo, "outputs")
fig_dir = os.path.join(repo, "figures")

font = {"family": "Nimbus Roman", "size": 26}
rc("font", **font)

# the ACS years with an answer for every county in the cache
FIRST_YEAR, LAST_YEAR = 2020, 2024
STATE_COLOR = "#444444"
COUNTY_COLORS = {"California": "#444444", "Minnesota": "#b5651d"}
COUNTY_MARKERS = {"California": "o", "Minnesota": "^"}


def cohorts_of_state_years(cdc, counties, cache):
    """The CDC state kindergarten populations with the sum of the cohorts of
    the state's counties in the year the school year starts."""
    by_state = {}
    for fips in counties["fips"]:
        by_state.setdefault(fips[:2], []).append(fips)
    rows = []
    for row in cdc.iter_rows(named=True):
        total = 0.0
        for fips in by_state[row["state_fips"]]:
            answer = census_answer(fips, row["start"], cache)
            assert answer is not None, (fips, row["start"])
            total += cohort_from_answer(answer)
        rows.append({"counted": row["counted"], "cohort": total})
    return pl.DataFrame(rows)


def ratio_text(frame):
    """n, median and the quartiles of counted / cohort."""
    ratio = (frame["counted"] / frame["cohort"]).to_numpy()
    low, high = np.percentile(ratio, [25, 75])
    return f"n = {len(ratio)}\nmedian {np.median(ratio):.2f} (IQR {low:.2f}-{high:.2f})"


def style(ax, lo, hi, label_x, label_y):
    ax.plot([lo, hi], [lo, hi], color="k", lw=1.5, zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(alpha=0.2)


if __name__ == "__main__":
    cache = CensusCache(CENSUS_CACHE_PATH)
    counties = pl.read_csv(
        os.path.join(output_dir, "projection_dataset_baseline_2027.csv"),
        schema_overrides={"fips": pl.String},
    ).select("fips")

    # a. the CDC kindergarten population of each state and school year
    cdc = pl.read_csv(
        os.path.join(
            data_dir,
            "Vaccination_Coverage_and_Exemptions_among_Kindergartners_20260526.csv",
        ),
        infer_schema_length=10000,
        null_values=["NA", ""],
    )
    add_fips = AddFIPS()
    cdc = (
        cdc.filter(
            (pl.col("Vaccine/Exemption") == "MMR")
            & (pl.col("Geography Type") == "States")
            & pl.col("Dose").is_null()
        )
        .with_columns(
            counted=pl.col("Population Size")
            .cast(pl.String)
            .str.replace_all(",", "")
            .cast(pl.Float64, strict=False),
            start=pl.col("School Year").str.slice(0, 4).cast(pl.Int64),
        )
        .unique(["Geography", "School Year"])
    )
    cdc = cdc.with_columns(
        state_fips=pl.Series(
            [add_fips.get_state_fips(s) for s in cdc["Geography"]]
        )
    ).filter(
        pl.col("counted").is_not_null()
        & pl.col("state_fips").is_not_null()
        & pl.col("start").is_between(FIRST_YEAR, LAST_YEAR)
    )
    states = cohorts_of_state_years(cdc, counties, cache)

    # b. the direct county enrollment of California and Minnesota
    stanford = pl.read_csv(
        os.path.join(data_dir, "stanford_output.csv"),
        infer_schema_length=10000,
        null_values=["NA", ""],
        schema_overrides={"FIPS": pl.String},
    ).filter(
        pl.col("State").is_in(list(COUNTY_COLORS))
        & pl.col("Year").is_between(FIRST_YEAR, LAST_YEAR)
        & pl.col("FIPS").is_not_null()
    )
    stanford = stanford.with_columns(
        counted=pl.col("Enrollment").cast(pl.Float64, strict=False)
    ).filter(pl.col("counted") > 0)
    cohorts = []
    for row in stanford.iter_rows(named=True):
        answer = census_answer(row["FIPS"], row["Year"], cache)
        cohorts.append(None if answer is None else cohort_from_answer(answer))
    stanford = stanford.with_columns(cohort=pl.Series(cohorts)).drop_nulls(
        "cohort"
    )

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(21, 10))
    ax_a.scatter(
        states["cohort"],
        states["counted"],
        s=70,
        c=STATE_COLOR,
        alpha=0.7,
        lw=0,
    )
    style(
        ax_a,
        3e3,
        1e6,
        "Cohort: population aged 5-9, divided by 5",
        "Kindergartners counted (CDC)",
    )
    ax_a.set_title("States, school years 2020-21 to 2024-25")
    ax_a.text(
        0.04, 0.95, ratio_text(states), transform=ax_a.transAxes, va="top"
    )
    for name in COUNTY_COLORS:
        sub = stanford.filter(pl.col("State") == name)
        ax_b.scatter(
            sub["cohort"],
            sub["counted"],
            s=70,
            c=COUNTY_COLORS[name],
            marker=COUNTY_MARKERS[name],
            alpha=0.6,
            lw=0,
            label=f"{name}: {ratio_text(sub).splitlines()[1]}",
        )
    style(
        ax_b,
        30,
        3e4,
        "Cohort: population aged 5-9, divided by 5",
        "Kindergarten enrollment counted (state data)",
    )
    ax_b.set_title("Counties of California and Minnesota")
    ax_b.legend(loc="upper left", frameon=False, fontsize=22)
    for ax, letter in ((ax_a, "a"), (ax_b, "b")):
        ax.text(
            -0.14,
            1.1,
            letter,
            transform=ax.transAxes,
            fontsize=34,
            fontweight="bold",
            va="top",
        )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(
            os.path.join(
                fig_dir, f"figure_supplemental_kindergarten_cohort.{ext}"
            ),
            dpi=300,
            bbox_inches="tight",
        )
    print("states", states.height, "counties", stanford.height)
    print("a:", ratio_text(states).replace("\n", "; "))
    for name in COUNTY_COLORS:
        print(
            "b",
            name,
            ratio_text(stanford.filter(pl.col("State") == name)).replace(
                "\n", "; "
            ),
        )
