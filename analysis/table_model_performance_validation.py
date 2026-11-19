"""
The table of county-level prediction scoring: every model of the rolling
validation, scored in each validation year (2021-2025) from its predictions of
the places (counties, and regions made of several counties), in two
conditions: all the cases including the imported ones (every place), and the
outbreak cases only (the places that had at least one case of any kind that
year). A model's forecast of a place is the mean of its predictive draws, the
expected number of cases; with imports each draw is the simulated imports plus
the outbreak that follows an import (see outbreak_size.county_scores for the
metrics). The models:

- the statistical models: three sets of covariates, with and without the
  spatial term, with and without pooling of the states, trained since 2011,
  since 2019 or on the last two years (36 models), all fitted with one
  likelihood and one scale of the vaccination covariates, the negative
  binomial and the natural scale (log of the susceptible fraction) unless the
  run says otherwise;
- the mechanistic models, with outbreaks from one import and with the
  outbreaks of several chains, trained since 2011, since 2019 or on the last
  two years (6 models);
- rankings of the places that predict no cases: their vaccine coverage, their
  population, and the cases they had in earlier years (all the reported cases
  from 2011 up to the year before, see outbreak_size.prior_cases);
- the import model alone, which predicts the imported cases and no outbreak.

Run it as `uv run analysis/table_model_performance_validation.py`, with
`--likelihood poisson` and `--covariate-scale raw` to score the fits of
another variant of 06_rolling_validation.py: the names of the files then end
with that variant (the same suffix as the files of 06).

Reads the draws of 06_rolling_validation.py from outputs/ and writes to
figures/:
- table_supplemental_outcome_counts.csv: the places scored, the places with
  each number of cases and the places flagged, in each year and condition
  (they do not depend on the model);
- supplemental_data_s1_county_level_prediction_scoring.csv: every model's scores
  averaged over 2021-2025 and over 2022-2025, each with the lower and upper end
  of a 95% interval from a bootstrap of the places within each year, by state
  (outbreak_size.summary_scores). A year with no place of the level of cases a
  score is about is left out of that score's mean. The means are checked
  against the yearly scores, and 1000
  bootstrap replicates take about 25 minutes (`--replicates` for fewer);
- supplemental_data_s2_model_consistency_across_years.csv: how close each
  model is to the best model of its pool in every year, and its mean gap over
  2022-2025 (the years the headline model is chosen on).
"""

import argparse
import gc
import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl
from numpyro import enable_x64

enable_x64()
import os

import jax.numpy as jnp
import yaml
from tqdm.auto import tqdm

from outbreak_size.county_scores import CONDITIONS, ScoreBook, count_columns
from outbreak_size.flag_discordance import COVARIATES
from outbreak_size.importation import expected_case_draws, total_case_draws
from outbreak_size.mechanistic import ranking_order
from outbreak_size.model_designs import (
    COVARIATE_SETS,
    START_YEARS,
    statistical_model_name,
    trained_since,
    training_years,
)
from outbreak_size.model_inputs import COVARIATE_SCALES
from outbreak_size.model_selection import (
    COVERAGE_RANK_MODEL_NAME,
    IMPORT_MODEL_NAME,
    POPULATION_RANK_MODEL_NAME,
    PRIOR_CASES_MODEL_NAME,
    consistency_table,
)
from outbreak_size.place_scores import place_score_table, place_value_table
from outbreak_size.poisson import LIKELIHOODS
from outbreak_size.prior_cases import place_key, prior_cases_score
from outbreak_size.samples_io import load_samples
from outbreak_size.summary_scores import (
    check_means,
    summarize_task,
    summary_table,
    value_groups,
    window_column,
    year_places,
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

# read analysis params yaml
with open("params.yml", "r") as f:
    params = yaml.safe_load(f)

GLOBALS_PRIOR_SD = params["validation_globals_prior_sd"]

VALIDATION_YEARS = [2021, 2022, 2023, 2024, 2025]
# how many models each pool passes forward from a year to the next (the
# models tied with the last one are passed forward too)
PASS_FORWARD = {"Statistical": 5, "Mechanistic": 2}
# the windows of years of S1, and how its intervals are made
WINDOWS = {"2021-2025": VALIDATION_YEARS, "2022-2025": VALIDATION_YEARS[1:]}
DEFAULT_REPLICATES = 1000
WORKERS = 6


if __name__ == "__main__":
    # the variant of the fits to score: the likelihood and the scale of the
    # vaccination covariates (as in 06_rolling_validation.py). A variant other
    # than the negative binomial with the natural scale puts its suffix at the
    # end of the names of the files it writes
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
        default=DEFAULT_REPLICATES,
        help="bootstrap replicates of the intervals of S1 (fewer for a quick look)",
    )
    args = parser.parse_args()
    LIKELIHOOD = args.likelihood
    COVARIATE_SCALE = args.covariate_scale
    VARIANT_SUFFIX = variant_file_suffix(LIKELIHOOD, COVARIATE_SCALE)

    # for each model for each year, we need to calculate:
    # classification metrics: sensitivity, specificity, roc auc score
    # regression metrics: RMSE, MAE, rho at >= 0, >= 3, >= 10 case thresholds
    # we need to calculate classification metrics with and without importations

    # let's first read the real outbreak size data and associated import
    # probabilities
    analysis = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        schema_overrides={"fips": pl.String},
    )
    missing_all = pl.read_csv(
        os.path.join(data_dir, "missing_counties.csv"),
        schema_overrides={"fips": pl.String},
    )
    missing = missing_all.filter(pl.col("year") >= 2021)
    # the cases of every county-year with a case, in every year (the missing
    # counties of all the years: their earlier cases count for the prior-cases
    # ranking), keyed like the places of the validation years
    case_history = pl.concat(
        [
            frame.select("state", "county", "fips", "year", "cases")
            for frame in (analysis, missing_all)
        ]
    ).with_columns(
        place=place_key(pl.col("fips"), pl.col("state"), pl.col("county"))
    )
    analysis = pl.concat([analysis, missing], how="diagonal_relaxed")
    # a place is keyed by fips and year; county names are not unique within a
    # state (Virginia's Fairfax is both a county and a city). Rows without a
    # county fips (some missing-data places) are keyed by name
    analysis = analysis.with_columns(
        state_county_year=pl.when(pl.col("fips").str.len_chars() == 5)
        .then(pl.concat_str([pl.col("fips"), pl.col("year")], separator="_"))
        .otherwise(
            pl.concat_str(
                [pl.col("state"), pl.col("county"), pl.col("year")],
                separator="_",
            )
        )
    )

    # --- descriptive: prior-case history of recent 10+-case counties ---
    def _counties(df, years, threshold=1):
        return set(
            df.filter(
                pl.col("year").is_in(list(years))
                & (pl.col("cases") >= threshold)
            )
            .select(["state", "county"])
            .unique()
            .iter_rows()
        )

    for window_start in [2021, 2023]:
        outbreak = _counties(analysis, range(window_start, 2026), threshold=10)
        ever = _counties(analysis, range(2011, window_start))
        since19 = _counties(analysis, range(2019, window_start))
        n = len(outbreak)
        print(f"Counties with 10+ total cases in {window_start}-2025: {n}")
        print(
            f"  had any cases 2011-{window_start - 1} (ever prior): "
            f"{len(outbreak & ever)}/{n} = {100 * len(outbreak & ever) / n:.1f}%"
        )
        print(
            f"  had any cases 2019-{window_start - 1}:              "
            f"{len(outbreak & since19)}/{n} = {100 * len(outbreak & since19) / n:.1f}%"
        )

    import_df = pl.read_csv(
        os.path.join(output_dir, "empirical_importation_probability.csv"),
        schema_overrides={"fips": pl.String},
    )
    with open(
        os.path.join(
            output_dir,
            "empirical_importation_probability_samples.pkl",
        ),
        "rb",
    ) as f:
        samples = pickle.load(f)
    with open(
        os.path.join(
            output_dir,
            "empirical_importation_count_samples.pkl",
        ),
        "rb",
    ) as f:
        count_samples = pickle.load(f)
    import_df = import_df.with_columns(
        importation_probability_samples=pl.Series(samples),
        expected_importations_samples=pl.Series(count_samples),
    )
    validation_years = [2021, 2022, 2023, 2024, 2025]
    all_counties = []
    # which rows of each year's projection dataset are kept (one row per
    # county-year), to apply to the samples
    row_keep = []
    # the imported cases drawn for each year (the same imports are used for
    # every model of a year)
    all_import_draws = []
    for year in validation_years:
        temp, keep, import_draws = validation_year_rows(
            year,
            output_dir,
            import_df,
            analysis,
            params["seed"],
            list(COVARIATES),
        )
        row_keep.append(keep)
        all_import_draws.append(import_draws)
        all_counties.append(temp)
    # the expected imports and the probability of an import of every place in
    # each draw (the rates the imports above were drawn from), for the
    # expected cases of the models: the risk score of the cases including
    # imports and their forecast are exact, not the mean of the simulated cases
    all_expected_imports = [
        np.vstack(
            counties["expected_importations_samples"].to_numpy()
        ).T.astype(np.float32)
        for counties in all_counties
    ]
    all_import_probability = [
        np.vstack(
            counties["importation_probability_samples"].to_numpy()
        ).T.astype(np.float32)
        for counties in all_counties
    ]

    # whether each place of each validation year had at least this many cases
    # (including the imported ones) in the year or in any later validation
    # year, the outcome of the "any window" PPV and NPV of the cases including
    # imports (the outbreak cases have none). The places are matched by fips,
    # the first row of a fips
    window_levels = (1, 5, 10, 20)
    first_row_of_fips = []
    for counties in all_counties:
        assert counties["fips"].null_count() == 0, "a place has no fips"
        rows = {}
        for fips, cases in zip(counties["fips"], counties["cases"]):
            rows.setdefault(fips, int(cases))
        first_row_of_fips.append(rows)
    window_labels = []
    for idx, counties in enumerate(all_counties):
        window_labels.append(
            {
                level: np.array(
                    [
                        any(
                            first_row_of_fips[later].get(fips, 0) >= level
                            for later in range(idx, len(all_counties))
                        )
                        for fips in counties["fips"]
                    ]
                )
                for level in window_levels
            }
        )

    # the scores of every model, for the table of county-level prediction
    # scoring (see outbreak_size.county_scores)
    scores = ScoreBook()
    place_scores = {}

    for start_year in tqdm(START_YEARS, desc="Training year variants"):
        for idx, validation_year in enumerate(
            tqdm(VALIDATION_YEARS, desc="Evaluation year")
        ):
            years = training_years(start_year, validation_year)

            true = all_counties[idx]["cases"].to_numpy()
            true = np.array([int(t) for t in true])
            true_outbreak = all_counties[idx]["cases_outbreak"].to_numpy()
            true_outbreak = np.array([int(t) for t in true_outbreak])
            window_all = window_labels[idx]

            # The outbreak cases are scored only on the places that had at
            # least one case of any kind that year (an introduction), against
            # their outbreak cases. The outbreak model predicts the size of an
            # outbreak given an introduction (it is fitted to the places with a
            # recorded case), so a place with no case at all has no outbreak to
            # predict. Every model, and the rankings without draws, go
            # through add_scores, so they all score the same places.
            introduced = true >= 1

            def add_scores(
                model_name,
                score_all,
                score_outbreak,
                draws_all=None,
                draws_outbreak=None,
                forecast_all=None,
            ):
                # the scores of every place, for the place file (the outbreak
                # scores of all the places, before the places with a case are
                # kept)
                place_scores[(model_name, validation_year)] = (
                    np.asarray(score_all, dtype=float),
                    np.asarray(score_outbreak, dtype=float),
                )
                scores.add(
                    model_name,
                    validation_year,
                    true,
                    true_outbreak[introduced],
                    score_all,
                    np.asarray(score_outbreak)[introduced],
                    draws_all,
                    None
                    if draws_outbreak is None
                    else np.asarray(draws_outbreak)[:, introduced],
                    windows_all=window_all,
                    forecast_all=forecast_all,
                )

            def score_statistical_fit(model_variant, car, pool):
                """Scores a statistical model from the draws of its fit (of
                the likelihood and the covariate scale of the run)."""
                samples = load_samples(
                    os.path.join(
                        output_dir,
                        validation_file_name(
                            years,
                            pool,
                            car,
                            model_variant + VARIANT_SUFFIX,
                            GLOBALS_PRIOR_SD,
                        ),
                    )
                )
                samples = keep_rows(samples, row_keep[idx], "validation draws")
                # draws of all the cases of the places: the imported cases and,
                # when there is an import, the outbreak that follows
                draws_all = total_case_draws(samples, all_import_draws[idx])
                # the risk score and the forecast of the cases including
                # imports: the expected cases (the draws are for the scores of
                # the distribution)
                expected = expected_case_draws(
                    samples,
                    all_import_probability[idx],
                    all_expected_imports[idx],
                ).mean(axis=0)
                add_scores(
                    statistical_model_name(
                        model_variant, car, pool, start_year
                    ),
                    expected,
                    jnp.mean(samples, axis=0),
                    draws_all,
                    samples,
                    forecast_all=expected,
                )

            for car in [True, False]:
                for pool in [True, False]:
                    for model_variant in COVARIATE_SETS:
                        score_statistical_fit(model_variant, car, pool)

            # --- the mechanistic model ---
            mechanistic_file = load_samples(
                os.path.join(
                    output_dir,
                    f"predictive_final_outbreak_size_predictions_{years}.npz",
                )
            )
            samples = keep_rows(
                mechanistic_file["cases"], row_keep[idx], "validation draws"
            )
            # The mechanistic model ranks the places (without imports) by the
            # mean of its Poisson draws, which is not limited by the
            # population, ties broken by lower coverage. The average of the
            # 8000 draws is not used for that ranking: places with the same
            # mean would be ordered by the noise of their draws.
            # Each saved array has one row (the model has a single mean per
            # place), kept by keep_rows like the draws
            mechanistic_mean = keep_rows(
                mechanistic_file["mean"], row_keep[idx], "validation mean"
            )[0]
            mechanistic_coverage = keep_rows(
                mechanistic_file["coverage"],
                row_keep[idx],
                "validation coverage",
            )[0]
            # higher score = higher risk (ranking_order puts 0 at the top)
            local_ranking_score = -np.asarray(
                ranking_order(
                    np.asarray(mechanistic_mean),
                    np.asarray(mechanistic_coverage),
                ),
                dtype=float,
            )

            draws_all = total_case_draws(samples, all_import_draws[idx])
            expected = expected_case_draws(
                samples, all_import_probability[idx], all_expected_imports[idx]
            ).mean(axis=0)
            add_scores(
                "Mechanistic: MMR coverage" + trained_since(start_year),
                expected,
                local_ranking_score,
                draws_all,
                samples,
                forecast_all=expected,
            )

            # the sensitivity analysis with the outbreaks of several chains:
            # every import starts its own independent chain, so after N
            # imports the outbreak cases are the sum of N independent Poisson
            # outbreaks with the mean of the place (limited to the
            # population, as for one chain), which is a Poisson with N times
            # that mean. The cases of the place are the imported cases plus
            # these outbreak cases (no import, no cases), and the total over
            # the chains is limited to the population of the place. The
            # imports are the same simulated imports as for every other model
            # of the year
            mechanistic_mean_limited = keep_rows(
                mechanistic_file["mean_limited_to_population"],
                row_keep[idx],
                "validation mean limited to the population",
            )[0]
            chains_rng = np.random.default_rng(
                [params["seed"], validation_year, 2]
            )
            draws_multi = np.minimum(
                all_import_draws[idx]
                + chains_rng.poisson(
                    all_import_draws[idx]
                    * np.asarray(mechanistic_mean_limited)
                ),
                all_counties[idx]["pop"].to_numpy(),
            )
            # the expected cases of this model: after N imports the outbreak
            # cases have the mean N * m (m: the mean of the place limited to
            # the population), so with the imported cases the expected cases
            # are (1 + m) times the expected imports, limited to the
            # population. The limit is applied to the expectation, which is
            # exact as long as the limit of the population is not reached
            # in the draws: it is checked
            multi_pop = all_counties[idx]["pop"].to_numpy()
            multi_uncapped = all_expected_imports[idx] * (
                1 + np.asarray(mechanistic_mean_limited, dtype=float)
            )
            share_capped = float((multi_uncapped > multi_pop).mean())
            assert share_capped < 1e-3, (
                f"{validation_year}: the expected cases of the multi-chain "
                f"model reach the population in {100 * share_capped:.3f}% of "
                "the draws, so their mean is not the mean of the limited cases"
            )
            expected_multi = np.minimum(multi_uncapped, multi_pop).mean(axis=0)
            add_scores(
                "Mechanistic with multi-chain outbreaks: MMR coverage"
                + trained_since(start_year),
                expected_multi,
                # without imports this model is the mechanistic model above
                local_ranking_score,
                draws_multi,
                samples,
                forecast_all=expected_multi,
            )

            # --- the two rankings that predict no cases ---
            # Each ranks the places by one number: the vaccine coverage
            # (lower coverage, higher risk) and the population (more people,
            # higher risk). They need no model and no imports, so they are the
            # same for every training window and are scored once. The
            # coverage is the one the mechanistic model used for the places of
            # this year, so the mechanistic model without imports ranks the
            # places exactly as this ranking does.
            if start_year == 2011:
                coverage_score = -np.asarray(mechanistic_coverage, dtype=float)
                population_score = (
                    all_counties[idx]["pop"].to_numpy().astype(float)
                )
                # the flags are the places with the highest 10% and 20% of
                # the scores; the share of places with each outcome should be
                # below that, so that a flag can find every one of them
                for condition_name, condition_cases in (
                    ("all cases", true),
                    (
                        "outbreak cases, places with a case",
                        true_outbreak[introduced],
                    ),
                ):
                    shares = {
                        level: float(np.mean(condition_cases >= level))
                        for level in (1, 5, 10, 20)
                    }
                    print(
                        f"{validation_year}, {condition_name}: share of the "
                        f"{len(condition_cases)} places with 1+/5+/10+/20+ "
                        "cases: "
                        + " / ".join(
                            f"{100 * v:.2f}%" for v in shares.values()
                        )
                        + f"; all below 10%: {max(shares.values()) < 0.10}"
                    )
                # the cases each place had in the years before this one (most
                # places had none and tie)
                prior_cases = prior_cases_score(
                    case_history,
                    all_counties[idx].select(
                        place_key(
                            pl.col("fips"), pl.col("state"), pl.col("county")
                        ).alias("place")
                    )["place"],
                    validation_year,
                )
                print(
                    f"{validation_year}: {int((prior_cases > 0).sum())} of "
                    f"{len(prior_cases)} places had cases in "
                    f"2011-{validation_year - 1}"
                )
                for rank_name, rank_score in (
                    (COVERAGE_RANK_MODEL_NAME, coverage_score),
                    (POPULATION_RANK_MODEL_NAME, population_score),
                    (PRIOR_CASES_MODEL_NAME, prior_cases),
                ):
                    add_scores(rank_name, rank_score, rank_score)

                # --- the import model alone ---
                # It predicts the imported cases and no outbreak. For all the
                # cases its draws are the simulated imports (the same ones as
                # for every other model of the year): a forecast and a
                # predictive distribution of the cases of a place. For the
                # outbreak cases it predicts none, so it only ranks the
                # places by their expected imports (no draws). It shows what
                # the outbreak models add beyond the risk of an import.
                import_score = np.asarray(
                    all_expected_imports[idx], dtype=float
                ).mean(axis=0)
                add_scores(
                    IMPORT_MODEL_NAME,
                    import_score,
                    import_score,
                    draws_all=all_import_draws[idx],
                    forecast_all=import_score,
                )

            # memory hygiene: this loop loads many large sample files per
            # validation year; free them before the next year
            gc.collect()

    # the scores of every model, for each year and for the average of the
    # years (see outbreak_size.county_scores for the metrics). The selection
    # below reads the table as written, rounded to 4 decimals, so that it can
    # be reproduced from the file
    scoring_table = scores.table().with_columns(pl.col(pl.Float64).round(4))

    # the places counted do not depend on the model: they have their own file
    counts = count_columns()
    outcome_counts = scoring_table.filter(
        pl.col("Evaluation year") != "2021-2025"
    ).select("Evaluation year", "Condition", *counts)
    outcome_counts = outcome_counts.unique(maintain_order=True)
    assert outcome_counts.height == 2 * len(VALIDATION_YEARS), (
        "the places counted differ between models of the same year"
    )
    outcome_counts.write_csv(
        os.path.join(fig_dir, "table_supplemental_outcome_counts.csv")
    )
    # the yearly scores (not saved): the input of the selection tables below and
    # the reference that the means of S1 are checked against
    print(f"scored {scoring_table['Model'].n_unique()} models")

    # the one-year-ahead selection of the models, each pool on its own, and
    # how close every model is to the best of its pool in each year; the
    # mechanistic models are a pool of their own, and the two rankings and the
    # import model alone are comparators
    models = scoring_table["Model"].unique(maintain_order=True).to_list()
    pools = {
        "Statistical": [m for m in models if m.startswith("Statistical:")],
        "Mechanistic": [m for m in models if m.startswith("Mechanistic")],
    }
    comparators = {
        "Vaccine coverage ranking": COVERAGE_RANK_MODEL_NAME,
        "Population size ranking": POPULATION_RANK_MODEL_NAME,
        "Import model alone": IMPORT_MODEL_NAME,
        "Prior cases ranking": PRIOR_CASES_MODEL_NAME,
    }
    # S2 as written to its file (rounded to 4 decimals), which
    # 06b_choose_headline_models.py shows when the headline models are chosen
    consistency = consistency_table(
        scoring_table, pools, PASS_FORWARD, comparators
    ).with_columns(pl.col(pl.Float64).round(4))
    consistency.write_csv(
        os.path.join(
            fig_dir,
            f"supplemental_data_s2_model_consistency_across_years{VARIANT_SUFFIX}.csv",
        )
    )

    print(
        f"wrote the selection tables of {len(pools['Statistical'])} "
        f"statistical and {len(pools['Mechanistic'])} mechanistic models"
    )

    # the scores of every model for every place and year (in memory: S1 is made
    # from them)
    # the outcomes of the "any window" metrics go with the places: whether each had
    # that many cases in its year or a later one
    window_columns = [window_column(level) for level in window_levels]
    places_by_year = {
        year: counties.with_columns(
            [
                pl.Series(window_column(level), window_labels[idx][level])
                for level in window_levels
            ]
        )
        for idx, (year, counties) in enumerate(
            zip(VALIDATION_YEARS, all_counties)
        )
    }
    place_table = place_score_table(
        places_by_year,
        place_scores,
        CONDITIONS,
        [*COVARIATES, *window_columns],
    )
    print(
        "places with no value, by covariate: "
        f"{ {c: int(place_table[c].null_count()) for c in COVARIATES} }"
    )
    # the forecast and the scores of the distribution of every place (the places of
    # the outbreak condition are those with a case)
    value_table = place_value_table(
        scores.place_values,
        {
            year: (counties["cases"].cast(pl.Int64) >= 1).to_numpy()
            for year, counties in zip(VALIDATION_YEARS, all_counties)
        },
        CONDITIONS,
    )

    # S1: the scores of every model averaged over the windows of years, with the
    # intervals of the bootstrap (written last: script 07 reads it). Each model and condition is one task of a pool of
    # processes, started fresh (spawn): JAX is multithreaded and must not be forked
    groups = value_groups(value_table)
    prefix = f"{CONDITIONS[0]} | "
    scored_models = [
        c[len(prefix) :] for c in place_table.columns if c.startswith(prefix)
    ]
    tasks = [
        (
            model,
            condition,
            index,
            {
                year: year_places(place_table, groups, model, condition, year)
                for year in VALIDATION_YEARS
            },
            WINDOWS,
            args.replicates,
            params["seed"],
        )
        for model in scored_models
        for index, condition in enumerate(CONDITIONS)
    ]
    print(
        f"S1: {len(scored_models)} models, {args.replicates} replicates, {WORKERS} processes"
    )
    with ProcessPoolExecutor(
        max_workers=WORKERS, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        summaries = dict(pool.map(summarize_task, tasks))
    s1_table = summary_table(summaries).with_columns(
        pl.col(pl.Float64).round(4)
    )
    checked, worst = check_means(s1_table, scoring_table, WINDOWS)
    print(
        f"the means of S1 equal those of the yearly scores ({checked} values checked, largest difference {worst:.5f})"
    )
    s1_path = os.path.join(
        fig_dir,
        f"supplemental_data_s1_county_level_prediction_scoring{VARIANT_SUFFIX}.csv",
    )
    s1_table.write_csv(s1_path)
    print(
        f"wrote {s1_table.height} rows and {s1_table.width} columns to {s1_path}"
    )
