"""
Supplemental figure: the panels of the imputed-data figure
(figure_supplemental_imputed_inferential_analyses.py: a, the coverage by
outbreak size; b, the coverage effect; c, the spatial correlation) for the
fits of 02 on subsets of the county-years of analysis_dataset.csv, one row per
subset: the county-years with a recorded number of imports, and those whose
coverage is the MMR coverage (not the all-vaccine coverage). The fits are All
covariates.

The % change per 1pp of b is averaged over all the county-years of
analysis_dataset.csv, as in the main figure, so that the rows are comparable.
"""

import importlib.util
import os

from numpyro import enable_x64

enable_x64()

import polars as pl

from outbreak_size import adjacency
from outbreak_size.samples_io import load_samples

root = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(root, "data")
output_dir = os.path.join(root, "outputs")
fig_dir = os.path.join(root, "figures")

# the plotting functions of the imputed-data figure (importing it runs only its
# definitions)
spec = importlib.util.spec_from_file_location(
    "imputed_figure",
    os.path.join(
        root, "analysis", "figure_supplemental_imputed_inferential_analyses.py"
    ),
)
imputed_figure = importlib.util.module_from_spec(spec)  # type: ignore
spec.loader.exec_module(imputed_figure)  # type: ignore

YEAR_BUCKETS = [
    [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    [2021, 2022, 2023, 2024, 2025],
]
# (the tag of the files of 02, the label of the row, the rows of the subset),
# as defined by SAMPLE_SUBSETS in 02_fit_hierarchical_poisson.py
SUBSETS = {
    "imports_known": ("Known imports", pl.col("imports_known")),
    "mmr_coverage_only": (
        "MMR coverage only",
        pl.col("coverage_source") == "mmr",
    ),
}
FIT = "samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_{years}_{tag}.npz"


def check_buckets(label, rows):
    """
    Print the county-years of the subset by period and outbreak size, and stop
    if a cell is empty (the violin of an empty cell cannot be drawn).
    """
    counts = (
        rows.with_columns(
            period=pl.when(pl.col("year") >= 2021)
            .then(pl.lit("2021-2025"))
            .otherwise(pl.lit("2011-2020")),
            bucket=pl.col("cases_outbreak")
            .cut([0, 9], labels=["0", "1-9", ">=10"], left_closed=False)
            .cast(pl.String),
        )
        .group_by("period", "bucket")
        .len()
        .sort("period", "bucket")
    )
    print(f"\n{label}: {rows.height} county-years")
    print(counts)
    assert counts.height == 6, (
        f"{label}: a period x outbreak size cell (0, 1-9, >=10) is empty, "
        "so the coverage distribution cannot be drawn"
    )


def print_places(label, rows):
    """
    Print the places of the subset in each period: the county-years of each
    place, their years and their outbreak cases.
    """
    for period, in_period in [
        ("2011-2020", pl.col("year") < 2021),
        ("2021-2025", pl.col("year") >= 2021),
    ]:
        period_rows = rows.filter(in_period)
        places = (
            period_rows.group_by("state", "county", "fips")
            .agg(
                county_years=pl.len(),
                years=pl.col("year").sort().cast(pl.String).str.join(","),
                outbreak_cases=pl.col("cases_outbreak").sum(),
            )
            .sort("state", "county", "fips")
        )
        print(
            f"\n{label}, {period}: {places.height} places, "
            f"{period_rows.height} county-years"
        )
        with pl.Config(tbl_rows=-1, tbl_width_chars=160, fmt_str_lengths=60):
            print(places)


def check_fit_rows(rows, years, tag):
    """
    The rows rebuilt here are those the fit used: as many as the cases of its
    predictive file (the SD that turns the coefficient into % per 1pp is that
    of these rows).
    """
    predictive = load_samples(
        os.path.join(
            output_dir,
            "predictive_"
            + FIT.removeprefix("samples_").format(years=years, tag=tag),
        )
    )
    n_rows = rows.filter(pl.col("year").is_in(years)).height
    assert predictive["cases"].shape[1] == n_rows, (
        f"{tag}, {years}: the fit has {predictive['cases'].shape[1]} rows, "
        f"the rebuilt subset {n_rows}"
    )


if __name__ == "__main__":
    df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        schema_overrides={"fips": pl.String},
    )
    absent = [
        c for c in ("imports_known", "coverage_source") if c not in df.columns
    ]
    if absent:
        print(
            f"skipped: analysis_dataset.csv has no {absent} column (rerun 00 and 01), so the subsets cannot be built."
        )
        raise SystemExit(0)
    adjacency_data = pl.read_csv(
        os.path.join(data_dir, "county_adjacencies.txt"),
        separator="|",
        schema_overrides={
            "County GEOID": pl.String,
            "Neighbor GEOID": pl.String,
        },
    )

    fig, ax = imputed_figure.setup_plot(len(SUBSETS), 3)  # type: ignore
    letters = iter("abcdefghi")
    for row, (tag, (label, in_subset)) in enumerate(SUBSETS.items()):
        rows = df.filter(in_subset)
        check_buckets(label, rows)
        print_places(label, rows)
        for years in YEAR_BUCKETS:
            check_fit_rows(rows, years, tag)

        for column in range(3):
            ax[row, column].text(  # type: ignore
                -0.25,
                1.08,
                next(letters),
                transform=ax[row, column].transAxes,  # type: ignore
                fontsize=24,
                fontweight="bold",
                va="top",
                ha="right",
            )

        # a: the coverage by outbreak size, with the MSR-blocked test of each
        # period on the counties of the subset
        imputed_figure.plot_coverage_distribution(
            rows,
            95,
            fig,
            ax[row, 0],  # type: ignore
            vaccination_metric="estimated_mmr_cov",
            adjacency_matrix_pre=adjacency.assemble_adjacency_matrix(
                adjacency_data,
                rows.filter(pl.col("year") < 2021)["fips"]
                .unique(maintain_order=True)
                .to_list(),
            ),
            adjacency_matrix_post=adjacency.assemble_adjacency_matrix(
                adjacency_data,
                rows.filter(pl.col("year") >= 2021)["fips"]
                .unique(maintain_order=True)
                .to_list(),
            ),
        )
        # b: the coverage effect of the fits of the subset, per 1pp over all
        # the county-years
        imputed_figure.plot_coverage_various_models(
            rows,
            fig,
            ax[row, 1],  # type: ignore
            year_buckets=YEAR_BUCKETS,
            model_name="All covariates",
            file_tag=f"_{tag}",
            reference_rows=df,
        )
        # c: the spatial correlation of the 2021-2025 fit
        imputed_figure.plot_car_effect_correlation_simple(
            load_samples(
                os.path.join(
                    output_dir,
                    FIT.format(years=YEAR_BUCKETS[1], tag=tag),
                )
            ),
            fig,
            ax[row, 2],  # type: ignore
        )

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(
            os.path.join(fig_dir, f"figure_supplemental_sample_subsets.{ext}"),
            dpi=300,
            bbox_inches="tight",
        )
