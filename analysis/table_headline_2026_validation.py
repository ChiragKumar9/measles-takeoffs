"""
The headline models on 2026: a prospective validation of the model of script 07
(the headline statistical model, chosen with 06b_choose_headline_models.py),
trained on its window of 2019-2025 by `06_rolling_validation.py
--headline-2026`, and of the headline mechanistic model of its window.

The model is scored as in table_model_performance_validation.py (the same
places, the same imports drawn for the places, the same scores), against the
cases of 2026 (to September/October: a part year), next to the comparators
that need no model fit of 2026: the headline mechanistic model, the rankings by
coverage, population and prior cases, and the import model alone.
There is no gap from the best of a pool (a pool has one model here), and the
scores of 2026 are not part of S1 or S2, which stay on 2021-2025. Every score
has the lower and upper end of a 95% interval from the bootstrap of the places
that S1 uses (by state, within groups of places that reached the same level of
cases; outbreak_size.summary_scores): it counts the sampling of the places of
the year, as one year has no other year to vary (`--replicates` for fewer than
1000, for a quick look).

The table is written to figures/table_supplemental_headline_2026_validation.csv.
It is printed twice: for all the places and without Hudspeth County, Texas
(its cases are printed; no source for their setting is cited here), which is one
place with a large share of the cases of the year.

Run after 06b_choose_headline_models.py, 06_rolling_validation.py
--headline-2026 and after 00, 01 (the cases of 2026 are in their _with_2026
files), 05 and 08.
"""

import argparse
import gc
import os
import pickle

import numpy as np
import polars as pl
from numpyro import enable_x64

enable_x64()
import jax.numpy as jnp
import yaml

from outbreak_size.county_scores import CONDITIONS, ScoreBook
from outbreak_size.flag_discordance import COVARIATES
from outbreak_size.headline_models import (
    HEADLINE_FILE_NAME,
    read_headline_models,
)
from outbreak_size.importation import expected_case_draws, total_case_draws
from outbreak_size.mechanistic import ranking_order
from outbreak_size.model_designs import (
    statistical_model_name,
    training_years,
)
from outbreak_size.model_inputs import COVARIATE_SCALES
from outbreak_size.model_selection import (
    COVERAGE_RANK_MODEL_NAME,
    IMPORT_MODEL_NAME,
    POPULATION_RANK_MODEL_NAME,
    PRIOR_CASES_MODEL_NAME,
)
from outbreak_size.place_scores import place_score_table, place_value_table
from outbreak_size.poisson import LIKELIHOODS
from outbreak_size.prior_cases import place_key, prior_cases_score
from outbreak_size.samples_io import load_samples
from outbreak_size.summary_scores import (
    check_means,
    single_year_summary,
    window_column,
)
from outbreak_size.validation import (
    keep_rows,
    validation_file_name,
    variant_file_suffix,
)
from outbreak_size.year_rows import validation_year_rows

data_dir = "data"
output_dir = "outputs"
fig_dir = "figures"

YEAR = 2026
# Hudspeth County, Texas: one place with a large share of the cases of 2026
HUDSPETH_FIPS = "48229"

with open("params.yml", "r") as f:
    params = yaml.safe_load(f)

GLOBALS_PRIOR_SD = params["validation_globals_prior_sd"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--likelihood",
        choices=LIKELIHOODS,
        default=params["validation_likelihood"],
    )
    parser.add_argument(
        "--covariate-scale",
        choices=COVARIATE_SCALES,
        default=params["covariate_scale"],
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=1000,
        help="bootstrap replicates of the intervals (fewer for a quick look)",
    )
    args = parser.parse_args()
    VARIANT_SUFFIX = variant_file_suffix(args.likelihood, args.covariate_scale)

    # the headline models, as the user chose them (06b_choose_headline_models.py)
    # and the years each is trained on before 2026
    headline = read_headline_models(
        os.path.join(output_dir, HEADLINE_FILE_NAME), fig_dir=fig_dir
    )
    design = headline.statistical
    years = training_years(design.start_year, YEAR)
    assert years[-1] == YEAR - 1
    print(f"Headline model: {design.name}, trained on {years[0]}-{years[-1]}")
    model_name = statistical_model_name(
        design.covariates, design.spatial, design.pooled, design.start_year
    )
    mechanistic_years = training_years(headline.mechanistic_start_year, YEAR)
    assert mechanistic_years[-1] == YEAR - 1
    print(
        f"Headline mechanistic model: {headline.mechanistic}, trained on "
        f"{mechanistic_years[0]}-{mechanistic_years[-1]}"
    )

    # the observed cases: the cases of 2011-2026 and the counties without
    # vaccine data (2021 on), keyed like the places of the validation year
    analysis = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset_with_2026.csv"),
        schema_overrides={"fips": pl.String},
    )
    missing_all = pl.read_csv(
        os.path.join(data_dir, "missing_counties_with_2026.csv"),
        schema_overrides={"fips": pl.String},
    )
    case_history = pl.concat(
        [
            frame.select("state", "county", "fips", "year", "cases")
            for frame in (analysis, missing_all)
        ]
    ).with_columns(
        place=place_key(pl.col("fips"), pl.col("state"), pl.col("county"))
    )
    analysis = pl.concat(
        [analysis, missing_all.filter(pl.col("year") >= 2021)],
        how="diagonal_relaxed",
    ).with_columns(
        state_county_year=pl.when(pl.col("fips").str.len_chars() == 5)
        .then(pl.concat_str([pl.col("fips"), pl.col("year")], separator="_"))
        .otherwise(
            pl.concat_str(
                [pl.col("state"), pl.col("county"), pl.col("year")],
                separator="_",
            )
        )
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

    # the places of 2026 and the imports drawn for them (as for every year
    # of the table script)
    places, keep, import_draws = validation_year_rows(
        YEAR, output_dir, import_df, analysis, params["seed"], list(COVARIATES)
    )
    observed_in_rows = int(places["cases"].sum())
    observed_in_files = int(
        analysis.filter(pl.col("year") == YEAR)["cases"].sum()
    )
    print(
        f"{YEAR}: {places.height} places; cases of the places {observed_in_rows}, "
        f"cases of the files {observed_in_files}"
    )
    if observed_in_rows != observed_in_files:
        print(
            "WARNING: cases of the files that are on no place of the "
            f"validation rows: {observed_in_files - observed_in_rows}"
        )

    true = np.array([int(t) for t in places["cases"].to_numpy()])
    true_outbreak = np.array(
        [int(t) for t in places["cases_outbreak"].to_numpy()]
    )
    introduced = true >= 1
    # the "any window" outcomes of one year are the year's own: a place that
    # had this many cases in the year
    window_all = {level: true >= level for level in (1, 5, 10, 20)}

    # --- the draws of the headline model and the comparators ---
    samples = keep_rows(
        load_samples(
            os.path.join(
                output_dir,
                validation_file_name(
                    years,
                    design.pooled,
                    design.spatial,
                    design.covariates + VARIANT_SUFFIX,
                    GLOBALS_PRIOR_SD,
                ),
            )
        ),
        keep,
        "2026 fold draws",
    )
    draws_all = total_case_draws(samples, import_draws)
    # the expected imports and the probability of an import of the places in
    # each draw (the rates the imports above were drawn from): the risk score
    # and the forecast of the cases including imports are the expected cases,
    # not the mean of the simulated cases
    expected_imports = np.vstack(
        places["expected_importations_samples"].to_numpy()
    ).T.astype(np.float32)
    import_probability = np.vstack(
        places["importation_probability_samples"].to_numpy()
    ).T.astype(np.float32)
    expected_cases = expected_case_draws(
        samples, import_probability, expected_imports
    ).mean(axis=0)

    mechanistic_file = load_samples(
        os.path.join(
            output_dir,
            f"predictive_final_outbreak_size_predictions_{mechanistic_years}.npz",
        )
    )
    mechanistic_samples = keep_rows(
        mechanistic_file["cases"], keep, "mechanistic draws"
    )
    mechanistic_mean = keep_rows(
        mechanistic_file["mean"], keep, "mechanistic mean"
    )[0]
    mechanistic_coverage = keep_rows(
        mechanistic_file["coverage"], keep, "mechanistic coverage"
    )[0]
    mechanistic_score = -np.asarray(
        ranking_order(
            np.asarray(mechanistic_mean), np.asarray(mechanistic_coverage)
        ),
        dtype=float,
    )
    mechanistic_draws_all = total_case_draws(mechanistic_samples, import_draws)
    mechanistic_expected_cases = expected_case_draws(
        mechanistic_samples, import_probability, expected_imports
    ).mean(axis=0)

    prior_cases = prior_cases_score(
        case_history,
        places.select(
            place_key(pl.col("fips"), pl.col("state"), pl.col("county")).alias(
                "place"
            )
        )["place"],
        YEAR,
    )
    coverage_score = -np.asarray(mechanistic_coverage, dtype=float)
    population_score = places["pop"].to_numpy().astype(float)
    import_score = np.asarray(expected_imports, dtype=float).mean(axis=0)
    gc.collect()

    def score(place_mask):
        """The table of the models on the places of `place_mask`."""
        scores = ScoreBook()
        # the scores of every place, for the intervals of the table
        place_scores = {}
        mask_introduced = introduced[place_mask]

        def add(
            name,
            score_all,
            score_outbreak,
            draws_all=None,
            draws_outbreak=None,
            forecast_all=None,
        ):
            score_all = np.asarray(score_all)[place_mask]
            score_outbreak = np.asarray(score_outbreak)[place_mask]
            place_scores[(name, YEAR)] = (
                np.asarray(score_all, dtype=float),
                np.asarray(score_outbreak, dtype=float),
            )
            scores.add(
                name,
                YEAR,
                true[place_mask],
                true_outbreak[place_mask][mask_introduced],
                score_all,
                score_outbreak[mask_introduced],
                None
                if draws_all is None
                else np.asarray(draws_all)[:, place_mask],
                None
                if draws_outbreak is None
                else np.asarray(draws_outbreak)[:, place_mask][
                    :, mask_introduced
                ],
                windows_all={k: v[place_mask] for k, v in window_all.items()},
                forecast_all=None
                if forecast_all is None
                else np.asarray(forecast_all)[place_mask],
            )

        add(
            model_name,
            expected_cases,
            jnp.mean(samples, axis=0),
            draws_all,
            samples,
            forecast_all=expected_cases,
        )
        add(
            headline.mechanistic,
            mechanistic_expected_cases,
            mechanistic_score,
            mechanistic_draws_all,
            mechanistic_samples,
            forecast_all=mechanistic_expected_cases,
        )
        for rank_name, rank_score in (
            (COVERAGE_RANK_MODEL_NAME, coverage_score),
            (POPULATION_RANK_MODEL_NAME, population_score),
            (PRIOR_CASES_MODEL_NAME, prior_cases),
        ):
            add(rank_name, rank_score, rank_score)
        add(
            IMPORT_MODEL_NAME,
            import_score,
            import_score,
            draws_all=import_draws,
            forecast_all=import_score,
        )
        # the row of the average over the years is the year itself here
        yearly = scores.table().filter(
            pl.col("Evaluation year").cast(pl.String) != "2021-2025"
        )
        # the scores of the year with the interval of the bootstrap of the
        # places, as S1 has it for 2021-2025 (summary_scores)
        window_columns = [window_column(level) for level in window_all]
        kept = places.filter(pl.Series(place_mask)).with_columns(
            [
                pl.Series(window_column(level), v[place_mask])
                for level, v in window_all.items()
            ]
        )
        place_table = place_score_table(
            {YEAR: kept},
            place_scores,
            CONDITIONS,
            [*COVARIATES, *window_columns],
        )
        value_table = place_value_table(
            scores.place_values, {YEAR: mask_introduced}, CONDITIONS
        )
        table = single_year_summary(
            YEAR, place_table, value_table, args.replicates, params["seed"]
        ).with_columns(pl.col(pl.Float64).round(4))
        # the means of the summary are the scores of the year itself
        checked, worst = check_means(table, yearly, {str(YEAR): [YEAR]})
        print(
            f"the means equal the scores of the year ({checked} values "
            f"checked, largest difference {worst:.5f})"
        )
        return table

    everywhere = np.ones(places.height, dtype=bool)
    table_all = score(everywhere)
    print(f"\nScores on all {places.height} places of {YEAR}:")

    def means(table):
        return table.select(
            c for c in table.columns if not c.endswith((", lower", ", upper"))
        )

    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
        print(means(table_all))

    without_hudspeth = (places["fips"] != HUDSPETH_FIPS).to_numpy()
    print(
        f"\nWithout Hudspeth County, TX ({int((~without_hudspeth).sum())} place; "
        f"{int(true[~without_hudspeth].sum())} of the {int(true.sum())} cases):"
    )
    table_without = score(without_hudspeth)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
        print(means(table_without))

    out_path = os.path.join(
        fig_dir, "table_supplemental_headline_2026_validation.csv"
    )
    pl.concat(
        [
            table_all.with_columns(places=pl.lit("all places")),
            table_without.with_columns(places=pl.lit("without Hudspeth")),
        ],
        how="diagonal_relaxed",
    ).write_csv(out_path)
    print(f"\nSaved {out_path}")
