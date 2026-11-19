"""
Calibration check of the Spearman block permutation test of
outbreak_size.msr.spearman_msr_blocked, which gives the p-values of the
coverage-predictability figures.

The outcome is the real one (the cases of the analysis dataset, with its
spatial pattern, ties, heavy tail and counties with several years). The
covariate is simulated, independent of the outcome, and as smooth as the real
coverage: its county-level Moran's I and its share of variance between
counties are those of the real coverage. A calibrated test rejects about 5% of
the simulations at the 0.05 level (and 1% at 0.01). The plain Spearman test,
which ignores the spatial and repeated structure, is shown for reference.

With --sweep the covariate is also made smoother and smoother (Moran's I of
its county means is printed for each level), to show the regime where the
test is calibrated and where it is not: the test holds its level when the
covariate is about as smooth as the real coverage and loses it when the
covariate is far smoother than any real one.

usage: uv run analysis/check_msr_calibration.py [--simulations N]
       [--permutations N] [--workers N] [--sweep]
The defaults take a few minutes; 400 simulations of 999 permutations give a
standard error of about 1 point.
"""

import argparse
import concurrent.futures as cf
import os

os.environ.setdefault("TQDM_DISABLE", "1")  # no progress bar per simulation

import numpy as np
import polars as pl
from scipy import stats

from outbreak_size import adjacency
from outbreak_size.msr import spearman_msr_blocked
from outbreak_size.msr_calibration import (
    between_county_share,
    calibration_verdict,
    independent_covariate,
    morans_i,
    rejection_rate,
    smoothness_matching,
)

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

PERIODS = {
    "2011-2020": lambda df: df.filter(pl.col("year") < 2021),
    "2021-2025": lambda df: df.filter(pl.col("year") >= 2021),
    "all years": lambda df: df,
}


def one_simulation(args):
    """
    The p-values of the plain and of the block permutation test, and the
    Moran's I of the simulated covariate's county means.
    """
    sim, n_permutations, period = args
    rng = np.random.default_rng(sim)
    x = independent_covariate(
        period["adjacency"],
        period["county_indices"],
        period["smoothness"],
        period["between_share"],
        rng,
    )
    plain_p = stats.spearmanr(x, period["y"])[1]
    _, block_p, _ = spearman_msr_blocked(
        x,
        period["y"],
        period["adjacency"],
        period["county_indices"],
        n_permutations=n_permutations,
        seed=sim,
    )
    county_means = np.bincount(period["county_indices"], weights=x) / (
        np.bincount(period["county_indices"])
    )
    return plain_p, block_p, morans_i(county_means, period["adjacency"])


def period_inputs(rows, adjacency_data):
    """The real outcome and the smoothness of the covariate for one period."""
    fips = sorted(rows["fips"].unique().to_list())
    matrix = adjacency.assemble_adjacency_matrix(adjacency_data, fips)
    position = {fips_: i for i, fips_ in enumerate(fips)}
    county_indices = np.array([position[f] for f in rows["fips"].to_list()])
    coverage = rows["estimated_mmr_cov"].to_numpy()
    county_means = np.array(
        [coverage[county_indices == i].mean() for i in range(len(fips))]
    )
    target = morans_i(county_means, matrix)
    outcome = rows["cases_outbreak"].to_numpy().astype(float)
    return dict(
        outcome_i=morans_i(
            np.array(
                [outcome[county_indices == i].mean() for i in range(len(fips))]
            ),
            matrix,
        ),
        adjacency=matrix,
        county_indices=county_indices,
        y=rows["cases_outbreak"].to_numpy().astype(float),
        between_share=between_county_share(coverage, county_indices),
        target_i=target,
        smoothness=smoothness_matching(matrix, target),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n")[0]
    )
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--permutations", type=int, default=299)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="also run covariates of smoothness 0, 0.3, 0.6 and 0.9",
    )
    args = parser.parse_args()

    df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        schema_overrides={"fips": pl.String},
    )
    adjacency_data = pl.read_csv(
        os.path.join(data_dir, "county_adjacencies.txt"),
        separator="|",
        schema_overrides={
            "County GEOID": pl.String,
            "Neighbor GEOID": pl.String,
        },
    )
    print(
        f"{args.simulations} simulations of {args.permutations} permutations; "
        "share of simulations rejected (standard error if calibrated)"
    )
    for name, select in PERIODS.items():
        period = period_inputs(select(df), adjacency_data)
        print(
            f"{name}: real coverage Moran's I {period['target_i']:.2f}, "
            f"between-county share {period['between_share']:.2f}; real outcome "
            f"Moran's I {period['outcome_i']:.2f}"
        )
        levels = [("matched to the real coverage", period["smoothness"])]
        if args.sweep:
            levels += [(f"{s}", s) for s in (0.0, 0.3, 0.6, 0.9)]
        for label, smoothness in levels:
            with cf.ProcessPoolExecutor(args.workers) as pool:
                results = np.array(
                    list(
                        pool.map(
                            one_simulation,
                            [
                                (
                                    sim,
                                    args.permutations,
                                    dict(period, smoothness=smoothness),
                                )
                                for sim in range(args.simulations)
                            ],
                        )
                    )
                )
            text = []
            for level in (0.05, 0.01):
                plain, _ = rejection_rate(results[:, 0], level)
                block, se = rejection_rate(results[:, 1], level)
                text.append(
                    f"at {level}: plain {plain:.3f}, block permutation "
                    f"{block:.3f} ({calibration_verdict(block, level, se)}, "
                    f"standard error {se:.3f})"
                )
            print(
                f"   covariate smoothness {label} (Moran's I of its county "
                f"means {results[:, 2].mean():.2f}): " + "; ".join(text)
            )
