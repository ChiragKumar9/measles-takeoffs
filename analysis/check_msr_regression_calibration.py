"""
Calibration check of the logistic regression test of
outbreak_size.msr_logistic.logistic_msr_test on the real regressors of panel d
of the predicting-outbreaks figure: the test of the SD term (the CV term for
the CV model) of each model of its supplemental table (HOMOGENEITY_MODELS of
the figure script; the counties of 2025 with the outbreak size, the controls
or the state dummies of the model).

For each test the 0/1 outcome (whether a county is part of a multi-county
outbreak) is rebuilt so that the tested predictor has no effect: the real fit
of the logistic model without that predictor, plus spatially smooth noise on
the logit scale (as smooth as the real residuals: their county-level Moran's I
is matched; the noise scale is matched to the between-county variance of the
real residuals), then a Bernoulli draw. The real predictors are kept. A
calibrated test rejects about 5% of the simulations at the 0.05 level (and 1%
at 0.01). Two Wald tests are shown for reference: the ordinary one, which
ignores the spatial structure, and one with standard errors clustered by
county (the real test clusters by outbreak; the simulated outcomes have no
outbreaks, only county-level noise).

usage: uv run analysis/check_msr_regression_calibration.py
       [--simulations N] [--permutations N] [--workers N] [--models 1,2]
--models takes the numbers of the models in the supplemental table, from 1
(default: all of them). The old SD x period table of eleven models took about
an hour with the defaults; 200 simulations give a standard error of about 1.5
points at the 0.05 level.
"""

import argparse
import concurrent.futures as cf
import importlib.util
import multiprocessing
import os

os.environ.setdefault("TQDM_DISABLE", "1")  # no progress bar per simulation

import numpy as np
import polars as pl
from scipy import stats

from outbreak_size import adjacency
from outbreak_size.msr_calibration import (
    calibration_verdict,
    logistic_null_outcome,
    match_noise_scale,
    morans_i,
    rejection_rate,
    residual_smoothness,
)
from outbreak_size.msr_logistic import logistic_fit, logistic_msr_test

root = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(root, "data")


def one_simulation(args):
    """The p values of the permutation test and of the two Wald tests."""
    sim, n_permutations, case = args
    rng = np.random.default_rng(sim)
    for _ in range(20):  # redraw an outcome with fewer than 5 ones or zeros
        y = logistic_null_outcome(
            case["adjacency"],
            case["county_indices"],
            case["logit"],
            case["noise_scale"],
            case["smoothness"],
            rng,
        )
        if 5 <= y.sum() <= len(y) - 5:
            break
    result = logistic_msr_test(
        case["X"],
        y,  # type: ignore
        case["county_indices"],
        case["adjacency"],
        tested=case["tested"],
        n_permutations=n_permutations,
        seed=sim,
        clusters=case["county_indices"],
    )
    return (
        result["p_value"],
        2 * stats.norm.sf(abs(result["z"])),
        result["p_clustered"],
    )


def main():
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n")[0]
    )
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--permutations", type=int, default=299)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--models",
        default=None,
        help=(
            "numbers of the models of the supplemental table, from 1, comma "
            "separated (default: all)"
        ),
    )
    args = parser.parse_args()

    # the figure script builds the county-years and the regressors of the
    # models; its tests are run here on the outcomes built from them
    spec = importlib.util.spec_from_file_location(
        "predicting_outbreaks",
        os.path.join(
            root, "analysis", "figure_predicting_county_level_outbreaks.py"
        ),
    )
    figure = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(figure)  # type: ignore
    if args.models is not None:
        keep = [int(number) for number in args.models.split(",")]
        for number, model in enumerate(list(figure.HOMOGENEITY_MODELS), 1):
            if number not in keep:
                del figure.HOMOGENEITY_MODELS[model]
    names = list(figure.HOMOGENEITY_MODELS)

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
    # every county with outbreak cases (regions left out) and the adjacency of
    # each period, as the main block of the figure script builds them
    df_spread = figure.outbreak_cases_all_counties(
        pl.read_csv(
            os.path.join(data_dir, "raw_merged_outbreaks_vaccinations.csv"),
            schema_overrides={"fips": pl.String},
        )
    )
    matrices = [
        adjacency.assemble_adjacency_matrix(
            adjacency_data,
            df_spread.filter(in_period)["fips"]
            .unique(maintain_order=True)
            .to_list(),
        )
        for in_period in (pl.col("year") < 2021, pl.col("year") >= 2021)
    ]

    calls = []

    def calibrate(
        X,
        y,
        county_indices,
        adjacency_matrix,
        *,
        tested,
        n_permutations,
        seed,
        clusters=None,
    ):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        county_indices = np.asarray(county_indices)
        adjacency_float = np.asarray(adjacency_matrix, dtype=float)
        scale = X.std(axis=0)
        scale[scale == 0] = 1.0
        design = np.column_stack(
            [np.ones(len(y)), (X - X.mean(axis=0)) / scale]
        )
        # the model without the tested predictor, and what it leaves
        reduced = np.delete(design, tested + 1, axis=1)
        beta, _, _ = logistic_fit(reduced, y)
        logit = reduced @ beta
        residual = y - 1 / (1 + np.exp(-logit))
        smoothness = residual_smoothness(
            adjacency_float, residual, county_indices
        )
        noise_scale = match_noise_scale(
            adjacency_float, county_indices, logit, y, smoothness
        )
        case = dict(
            adjacency=adjacency_float,
            county_indices=county_indices,
            X=X,
            tested=tested,
            logit=logit,
            smoothness=smoothness,
            noise_scale=noise_scale,
        )
        with cf.ProcessPoolExecutor(
            args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:  # forked workers hang when polars is loaded
            p_values = np.array(
                list(
                    pool.map(
                        one_simulation,
                        [
                            (sim, args.permutations, case)
                            for sim in range(args.simulations)
                        ],
                    )
                )
            )
        text = []
        for level in (0.05, 0.01):
            rate, se = rejection_rate(p_values[:, 0], level)
            text.append(
                f"at {level}: {rate:.3f} "
                f"({calibration_verdict(rate, level, se)}, standard error "
                f"{se:.3f})"
            )
        ordinary, _ = rejection_rate(p_values[:, 1], 0.05)
        clustered, _ = rejection_rate(
            p_values[:, 2][~np.isnan(p_values[:, 2])], 0.05
        )
        county_means = np.bincount(
            county_indices, weights=residual, minlength=len(adjacency_float)
        ) / np.bincount(county_indices, minlength=len(adjacency_float))
        print(
            f"{names[len(calls)]}: {len(y)} counties, "
            f"{int(y.sum())} multi-county, "
            f"residual Moran's I {morans_i(county_means, adjacency_float):.2f} "
            f"(smoothness {smoothness:.2f}, noise scale {noise_scale}"
            + (", the largest tried" if noise_scale == 5.0 else "")
            + ") | "
            "permutation test "
            + "; ".join(text)
            + f" | ordinary Wald test at 0.05: {ordinary:.3f}"
            f" | Wald test clustered by county at 0.05: {clustered:.3f}",
            flush=True,
        )
        calls.append(tested)
        # the figure code needs a result to go on with, not a precise one
        return logistic_msr_test(
            X,
            y,
            county_indices,
            adjacency_matrix,
            tested=tested,
            n_permutations=19,
            seed=seed,
            clusters=clusters,
        )

    setattr(figure, "logistic_msr_test", calibrate)
    print(
        f"{args.simulations} simulations of {args.permutations} permutations "
        "per test; share of simulations rejected (standard error if "
        "calibrated)"
    )
    samples = figure.homogeneity_samples(
        df, df_spread, matrices[0], matrices[1]
    )
    # 19 permutations: calibrate returns a 19-permutation test to go on with
    figure.create_homogeneity_table(samples, adjacency_data, n_permutations=19)


if __name__ == "__main__":
    main()
