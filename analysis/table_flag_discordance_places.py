"""
The places of the flag discordance figure (figure_supplemental_flag_discordance.py):
where the headline statistical model and the headline mechanistic model (chosen with
06b_choose_headline_models.py) flag different places. Of the places with a case, the
highest 10% of each model's score of the outbreak cases (no imports) in the year are
flagged; the places with 5+ outbreak cases that at least one model flagged are
written, each with its group (both, statistical only, mechanistic only), the percentile
of each model's score among the places with a case, and its covariates in their own
units and as percentiles among all the places of the year
(outbreak_size.flag_discordance.discordance_places). The scores of the other places
and the performance of the models are not saved.

Two files are written to outputs/:
- flag_discordance_places.csv: the validation years 2022-2025, from the rolling
  validation of 06_rolling_validation.py (the models trained on the years before each);
- flag_discordance_places_2026.csv: 2026 so far, from the fits of
  `06_rolling_validation.py --headline-2026` and the cases of the _with_2026 files of
  00 and 01. `--no-2026` writes only the first file, for a run before the data of
  2026 are there.

Run after 06b_choose_headline_models.py and the rolling validation (and, for 2026,
`06_rolling_validation.py --headline-2026`).
"""

import argparse
import os
import pickle

import numpy as np
import polars as pl
import yaml

from outbreak_size.flag_discordance import (
    COVARIATES,
    MECHANISTIC_SCORE,
    PLACES_FILE_NAME,
    PLACES_FILE_NAME_2026,
    STATISTICAL_SCORE,
    discordance_places,
)
from outbreak_size.headline_models import (
    HEADLINE_FILE_NAME,
    read_headline_models,
)
from outbreak_size.mechanistic import ranking_order
from outbreak_size.model_designs import training_years
from outbreak_size.model_selection import BEST_MODEL_YEARS
from outbreak_size.samples_io import load_samples
from outbreak_size.validation import (
    keep_rows,
    validation_file_name,
    variant_file_suffix,
)
from outbreak_size.year_rows import observed_county_years, validation_year_rows

data_dir = "data"
output_dir = "outputs"

LATER_YEAR = 2026

with open("params.yml", "r") as f:
    params = yaml.safe_load(f)

GLOBALS_PRIOR_SD = params["validation_globals_prior_sd"]
VARIANT_SUFFIX = variant_file_suffix(
    params["validation_likelihood"], params["covariate_scale"]
)


def scored_places(year, observed, import_df, headline):
    """The places of `year` with the columns discordance_places needs: the
    cases, the outbreak cases, the covariates and the score of each headline
    model for the outbreak cases (no imports): the mean of the outbreak sizes
    of the statistical model, and the ranking of the mechanistic model by its
    mean, ties broken by lower coverage (as in the table script)."""
    places, keep, _ = validation_year_rows(
        year, output_dir, import_df, observed, params["seed"], list(COVARIATES)
    )
    design = headline.statistical
    samples = keep_rows(
        load_samples(
            os.path.join(
                output_dir,
                validation_file_name(
                    training_years(design.start_year, year),
                    design.pooled,
                    design.spatial,
                    design.covariates + VARIANT_SUFFIX,
                    GLOBALS_PRIOR_SD,
                ),
            )
        ),
        keep,
        f"{year} fold draws",
    )
    mechanistic_years = training_years(headline.mechanistic_start_year, year)
    mechanistic_file = load_samples(
        os.path.join(
            output_dir,
            f"predictive_final_outbreak_size_predictions_{mechanistic_years}.npz",
        )
    )
    mechanistic_score = -np.asarray(
        ranking_order(
            np.asarray(
                keep_rows(mechanistic_file["mean"], keep, f"{year} mean")[0]
            ),
            np.asarray(
                keep_rows(
                    mechanistic_file["coverage"], keep, f"{year} coverage"
                )[0]
            ),
        ),
        dtype=float,
    )
    return places.select(
        pl.lit(year, dtype=pl.Int64).alias("year"),
        "fips",
        "state",
        "county",
        pl.col("cases").cast(pl.Int64),
        # the table script builds the outbreak cases with a string replace
        pl.col("cases_outbreak").cast(pl.Int64),
        *COVARIATES,
    ).with_columns(
        pl.Series(STATISTICAL_SCORE, np.asarray(samples).mean(axis=0)),
        pl.Series(MECHANISTIC_SCORE, mechanistic_score),
    )


def write_places(frame, first_year, file_name, label):
    flagged = discordance_places(frame, first_year)
    path = os.path.join(output_dir, file_name)
    flagged.write_csv(path)
    print(
        f"wrote {path}: {flagged.height} places of {label} with 5+ outbreak "
        f"cases that a model flagged; groups "
        f"{flagged['group'].value_counts().rows()}; places with no value, by "
        f"covariate: { {c: int(frame[c].null_count()) for c in COVARIATES} }"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-2026",
        action="store_true",
        help="write only the validation years (the data of 2026 are not there)",
    )
    args = parser.parse_args()

    headline = read_headline_models(
        os.path.join(output_dir, HEADLINE_FILE_NAME), fig_dir="figures"
    )
    print(
        f"Statistical model: {headline.statistical.name}\n"
        f"Mechanistic model: {headline.mechanistic}"
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
    import_df = pl.read_csv(
        os.path.join(output_dir, "empirical_importation_probability.csv"),
        schema_overrides={"fips": pl.String},
    ).with_columns(
        importation_probability_samples=pl.Series(probability_samples),
        expected_importations_samples=pl.Series(count_samples),
    )

    def observed(suffix):
        return observed_county_years(
            pl.read_csv(
                os.path.join(data_dir, f"analysis_dataset{suffix}.csv"),
                schema_overrides={"fips": pl.String},
            ),
            pl.read_csv(
                os.path.join(data_dir, f"missing_counties{suffix}.csv"),
                schema_overrides={"fips": pl.String},
            ),
        )

    validation_observed = observed("")
    write_places(
        pl.concat(
            [
                scored_places(year, validation_observed, import_df, headline)
                for year in BEST_MODEL_YEARS
            ]
        ),
        min(BEST_MODEL_YEARS),
        PLACES_FILE_NAME,
        f"{BEST_MODEL_YEARS[0]}-{BEST_MODEL_YEARS[-1]}",
    )
    if not args.no_2026:
        write_places(
            scored_places(
                LATER_YEAR, observed("_with_2026"), import_df, headline
            ),
            LATER_YEAR,
            PLACES_FILE_NAME_2026,
            str(LATER_YEAR),
        )
