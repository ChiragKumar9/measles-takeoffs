import pickle

import numpy as np
import polars as pl
from numpyro import enable_x64

enable_x64()
import os

import jax.numpy as jnp
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm.auto import tqdm

data_dir = "data"
output_dir = "outputs"
fig_dir = "figures"


def performance_table(
    samples,
    true,
    mode,
    classification_threshold=None,
    regression_percentile=None,
    regression_threshold=None,
):
    if mode == "classification":
        fpr, tpr, _ = roc_curve(
            true >= classification_threshold, samples / samples.max()
        )
        if np.isnan(fpr[0]) or np.isnan(tpr[0]):
            sensitivity = np.nan
            specificity = np.nan
            roc_auc = np.nan
        else:
            # select a specificity such that sensitivity is >= 80%
            sensitivity = min(tpr[tpr >= 0.80])
            specificity = 1.0 - min(fpr[tpr >= 0.80])

            roc_auc = roc_auc_score(
                true >= classification_threshold, samples / samples.max()
            )

        # make a dataframe
        performance_summary = {
            "Sensitivity (%)": sensitivity * 100,
            "Specificity (%)": specificity * 100,
            "AUROC (%)": roc_auc * 100,
        }

        performance_summary = pl.DataFrame(performance_summary)

    elif mode == "regression":
        # calculate regression metrics
        # subset
        mask = true > 0
        if regression_threshold is None:
            assert regression_percentile is not None, (
                "Can't provide both regression percentile and threshold"
            )
            threshold = jnp.percentile(
                true[mask],
                regression_percentile,  # type: ignore
            )
            mask = jnp.logical_and(true >= threshold, mask)
        else:
            assert regression_percentile is None, (
                "Can't provide both regression percentile and threshold"
            )
            mask = jnp.logical_and(true >= regression_threshold, mask)
        true_subset = true[mask]
        samples_subset = samples[mask]
        # calculate metrics
        rmse = jnp.sqrt(jnp.mean((samples_subset - true_subset) ** 2))
        mae = jnp.median(jnp.abs(samples_subset - true_subset))
        absolute_difference = abs(samples.sum() - true.sum())
        mfe = jnp.median(
            jnp.maximum(
                samples_subset / true_subset, true_subset / samples_subset
            )
        )
        rho2 = stats.spearmanr(true_subset, samples_subset)[0] ** 2  # type: ignore

        performance_summary = {
            "RMSE (cases)": rmse,
            "MAE (cases)": mae,
            "Absolute difference (cases)": absolute_difference,
            "Median fold error": mfe,
            "Spearman rho^2 (%)": rho2 * 100,
        }

        performance_summary = pl.DataFrame(performance_summary)

    elif mode == "selection":
        # run both classification and regression
        classification = performance_table(
            samples, true, "classification", classification_threshold=10
        )
        regression = performance_table(
            samples,
            true,
            "regression",
            regression_percentile=80,
        )

        # concate
        performance_summary = pl.concat(
            [classification, regression], how="horizontal"
        )

    else:
        assert False, "mode not found"

    return performance_summary


def aggregate(df, metric_columns):
    # create a weighted average value across all years
    # weight by log of total cases observed in that year
    df_agg = (
        df.group_by(["Model"], maintain_order=True)
        .agg(
            [
                *[
                    (pl.col(col).filter(pl.col(col).is_not_nan())).mean()
                    for col in metric_columns
                ],
                *[
                    pl.col(col)
                    .filter(pl.col(col).is_not_nan())
                    .min()
                    .alias(f"{col}_lower")
                    for col in metric_columns
                ],
                *[
                    pl.col(col)
                    .filter(pl.col(col).is_not_nan())
                    .max()
                    .alias(f"{col}_upper")
                    for col in metric_columns
                ],
            ]
        )
        .with_columns(pl.lit("2021-2025").alias("Evaluation year"))
    )

    # add this to the end
    df = pl.concat([df, df_agg], how="diagonal_relaxed")
    return df


def beautify(df, column_stub):
    # for each column in column stub, concatenate its upper and lower values
    # in one row and round
    df = df.with_columns(
        [
            *[pl.col(col).round(2) for col in column_stub],
            *[pl.col(f"{col}_lower").round(2) for col in column_stub],
            *[pl.col(f"{col}_upper").round(2) for col in column_stub],
        ]
    )
    # join
    df = df.with_columns(
        [
            pl.format(
                "{} [{}, {}]", col, f"{col}_lower", f"{col}_upper"
            ).alias(col)
            for col in column_stub
        ]
    )
    # drop the lower and upper
    df = df.drop([f"{col}_lower" for col in column_stub])
    df = df.drop([f"{col}_upper" for col in column_stub])

    return df


def other_tables(df, samples_unconditioned, samples, true, true_outbreak):
    # this function accepts the model selection table, which has classification
    # metrics calculated for the 10+
    # for the main table, we need to add classification for 5+ and 20+ with imports
    # for the supplemental table, we need to add classification for 5, 10, and 20
    # without imports
    # for the the supplemental regression table, we need to add regression
    # metrics for the top 80%, top 50%, and top 20%

    # first let's relabel the existing columns
    df = df.rename(
        {
            "AUROC (%)": "AUROC, 10+ cases (%)",
            "Sensitivity (%)": "Sensitivity, 10+ cases (%)",
            "Specificity (%)": "Specificity, 10+ cases (%)",
            "RMSE (cases)": "RMSE, top 20% (cases)",
            "MAE (cases)": "MAE, top 20% (cases)",
            "Median fold error": "Median fold error, top 20%",
            "Spearman rho^2 (%)": "Spearman rho^2, top 20% (%)",
        }
    )

    classification5_with_imports = performance_table(
        samples_unconditioned,
        true,
        mode="classification",
        classification_threshold=5,
    )
    classification5_with_imports = classification5_with_imports.rename(
        {
            "Sensitivity (%)": "Sensitivity, 5+ cases (%)",
            "Specificity (%)": "Specificity, 5+ cases (%)",
            "AUROC (%)": "AUROC, 5+ cases (%)",
        }
    )

    classification20_with_imports = performance_table(
        samples_unconditioned,
        true,
        mode="classification",
        classification_threshold=20,
    )
    classification20_with_imports = classification20_with_imports.rename(
        {
            "Sensitivity (%)": "Sensitivity, 20+ cases (%)",
            "Specificity (%)": "Specificity, 20+ cases (%)",
            "AUROC (%)": "AUROC, 20+ cases (%)",
        }
    )

    classification5_wo_imports = performance_table(
        samples,
        true_outbreak,
        mode="classification",
        classification_threshold=5,
    )
    classification5_wo_imports = classification5_wo_imports.rename(
        {
            "Sensitivity (%)": "Sensitivity, 5+ outbreak cases (%)",
            "Specificity (%)": "Specificity, 5+ outbreak cases (%)",
            "AUROC (%)": "AUROC, 5+ outbreak cases (%)",
        }
    )

    classification10_wo_imports = performance_table(
        samples,
        true_outbreak,
        mode="classification",
        classification_threshold=10,
    )
    classification10_wo_imports = classification10_wo_imports.rename(
        {
            "Sensitivity (%)": "Sensitivity, 10+ outbreak cases (%)",
            "Specificity (%)": "Specificity, 10+ outbreak cases (%)",
            "AUROC (%)": "AUROC, 10+ outbreak cases (%)",
        }
    )

    classification20_wo_imports = performance_table(
        samples,
        true_outbreak,
        mode="classification",
        classification_threshold=20,
    )
    classification20_wo_imports = classification20_wo_imports.rename(
        {
            "Sensitivity (%)": "Sensitivity, 20+ outbreak cases (%)",
            "Specificity (%)": "Specificity, 20+ outbreak cases (%)",
            "AUROC (%)": "AUROC, 20+ outbreak cases (%)",
        }
    )

    regression_50 = performance_table(
        samples_unconditioned,
        true,
        mode="regression",
        regression_percentile=50,
    ).drop("Absolute difference (cases)")
    regression_50 = regression_50.rename(
        {
            "RMSE (cases)": "RMSE, top 50% (cases)",
            "MAE (cases)": "MAE, top 50% (cases)",
            "Median fold error": "Median fold error, top 50%",
            "Spearman rho^2 (%)": "Spearman rho^2, top 50% (%)",
        }
    )

    regression_80 = performance_table(
        samples_unconditioned,
        true,
        mode="regression",
        regression_percentile=80,
    ).drop("Absolute difference (cases)")
    regression_80 = regression_80.rename(
        {
            "RMSE (cases)": "RMSE, top 80% (cases)",
            "MAE (cases)": "MAE, top 80% (cases)",
            "Median fold error": "Median fold error, top 80%",
            "Spearman rho^2 (%)": "Spearman rho^2, top 80% (%)",
        }
    )

    # bring this all together in one table
    df = pl.concat(
        [
            df,
            classification5_with_imports,
            classification20_with_imports,
            classification5_wo_imports,
            classification10_wo_imports,
            classification20_wo_imports,
            regression_50,
            regression_80,
        ],
        how="horizontal",
    )

    return df


def find_best_model(df, delta=35, subset=None, exclude=None):
    """
    Find best model: lowest absolute difference, then highest AUROC within delta

    Parameters
    ----------
    df : pl.DataFrame
        Model performance dataframe
    subset : str, optional
        Filter models containing this string
    exclude : str, optional
        Filter out models containing this string

    Returns
    -------
    str
        Name of the winning model
    """
    if subset is not None:
        df = df.filter(pl.col("Model").str.contains(subset))

    if exclude is not None:
        df = df.filter(~pl.col("Model").str.contains(exclude))

    best = df["Absolute difference (cases)"].min()

    winner = df.filter(
        pl.col("Absolute difference (cases)") <= best + delta
    ).filter(
        pl.col("AUROC, 10+ cases (%)") == pl.col("AUROC, 10+ cases (%)").max()
    )["Model"][0]

    return winner


if __name__ == "__main__":
    # for each model for each year, we need to calculate:
    # classification metrics: sensitivity, specificity, roc auc score
    # regression metrics: RMSE, MAE, R2 or rho^2 (preferably the latter)
    # we need to calculate regression metrics over the top 80%, the top 50%,
    # and the top 20%

    # we need to calculate classification metrics with and without importations

    # we need one table that gets at some model selection metric -- we use
    # AUROC10 + rho^2 20% for model selection

    # we need to read in every model, calculate the mean posterior, pass that
    # through the function to get out the metrics for model selection
    # from the model selection table, we can subset to the other tables we need
    # for the spatial models, we need to do our CAR term prior fitting

    # let's first read the real outbreak size data and associated import
    # probabilities
    # let's get the actual case counts from the train data
    analysis = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        schema_overrides={"fips": pl.String},
    )

    import_df = pl.read_csv(
        os.path.join(output_dir, "empirical_importation_probability.csv"),
        # make sure we read fips codes as strings
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
    import_df = import_df.with_columns(
        importation_probability_samples=pl.Series(samples)
    )
    # each place can only occur once per year
    # we want to drop duplicates, deferring to projection rather than train
    # type values
    # to do this, sort by type
    import_df = import_df.sort("type").unique(
        subset=["state", "county", "year"], keep="last"
    )

    validation_years = [2021, 2022, 2023, 2024, 2025]
    all_counties = []
    all_imports = []
    for year in validation_years:
        temp = pl.read_csv(
            os.path.join(output_dir, f"projection_dataset_{year}.csv"),
            schema_overrides={"fips": pl.String},
        )
        temp = temp.with_columns(validation_year=pl.lit(year))

        # the only things we need from the df are the county, state, year, and cases
        temp = temp.select(
            [
                "county",
                "state",
                "year",
                "validation_year",
                "cases_outbreak",
                "type",
            ]
        )

        temp = temp.join(
            analysis[["year", "state", "county", "cases"]],
            on=["year", "state", "county"],
            how="left",
            maintain_order="left",
        )
        # fill null cases with 0 because they were in projection only
        temp = temp.with_columns(
            cases=pl.when(pl.col("type") == "projection")
            .then(pl.lit(0))
            .otherwise(pl.col("cases"))
        )

        # we need to merge this up with the import data
        temp = temp.join(
            import_df.select(
                [
                    "county",  # rather than fips because we have some regions
                    "state",
                    "year",
                    "importation_probability_samples",
                ]
            ),
            on=["county", "state", "year"],
            how="left",
            maintain_order="left",
        )

        # it helps to have the importation samples separately for easy
        # multiplication
        temp_import = jnp.array(
            np.vstack(temp["importation_probability_samples"].to_numpy())  # type: ignore
        ).T
        all_imports.append(temp_import)

        all_counties.append(temp)

    car_scale_factors = {}  # Store scale factors keyed by (start_year, pool, model_variant)

    for start_year in [2011, 2019, -2]:
        for pool in [True, False]:
            for model_variant in [
                "MMR coverage",
                "MMR coverage + exemptions",
                "All covariates",
            ]:
                observed = []
                predicted = []

                for idx, validation_year in enumerate(
                    [2021, 2022, 2023, 2024, 2025]
                ):
                    if start_year < 0:
                        years = list(
                            range(
                                validation_year + start_year, validation_year
                            )
                        )
                    else:
                        years = list(range(start_year, validation_year))

                    cases = all_counties[idx]["cases"].sum()

                    # Only process CAR models for scaling calculation
                    with open(
                        os.path.join(
                            output_dir,
                            f"samples_partial_pooled_poisson_validation_testing_{years}_pool={pool}_CAR=True_{model_variant}.pkl",
                        ),
                        "rb",
                    ) as f:
                        samples = pickle.load(f)

                    samples_with_imports = samples * all_imports[idx]
                    samples_with_imports = jnp.sum(
                        samples_with_imports, axis=1
                    )
                    predicted_median = jnp.median(samples_with_imports)

                    observed.append(cases)
                    predicted.append(predicted_median)

                # Calculate scale factor for this model configuration across all years
                scale = np.mean(np.array(predicted) / np.array(observed))
                car_scale_factors[(start_year, pool, model_variant)] = scale

    # we need to read in all of our different models and calculate the statistics
    # on them
    # we need the model selection metrics
    model_performance = []
    for start_year in tqdm([2011, 2019, -2], desc="Training year variants"):
        for idx, validation_year in enumerate(
            tqdm([2021, 2022, 2023, 2024, 2025], desc="Evaluation year")
        ):
            # assemble the full year list
            if start_year < 0:
                years = list(
                    range(validation_year + start_year, validation_year)
                )
            else:
                years = list(range(start_year, validation_year))

            cases = all_counties[idx]["cases"].sum()
            log_total_cases = cases
            true = all_counties[idx]["cases"].to_numpy()
            true_outbreak = all_counties[idx]["cases_outbreak"].to_numpy()

            for car in [True, False]:
                for pool in [True, False]:
                    for model_variant in [
                        "MMR coverage",
                        "MMR coverage + exemptions",
                        "All covariates",
                    ]:
                        with open(
                            os.path.join(
                                output_dir,
                                f"samples_partial_pooled_poisson_validation_testing_{years}_pool={pool}_CAR={car}_{model_variant}.pkl",
                            ),
                            "rb",
                        ) as f:
                            samples = pickle.load(
                                f,
                            )

                        # if car, we need to scale to account for the improper
                        # CAR prior
                        if car:
                            scale = car_scale_factors[
                                (start_year, pool, model_variant)
                            ]
                            samples = samples / scale

                        # calculate the mean
                        # pass through the performance table function
                        # under various modes

                        samples_unconditioned = samples * all_imports[idx]

                        samples_unconditioned = jnp.mean(
                            samples_unconditioned, axis=0
                        )
                        samples = jnp.mean(samples, axis=0)

                        df = performance_table(
                            samples_unconditioned, true, mode="selection"
                        )

                        # for the first df we get out from the fcn, we need to
                        # editorialize
                        model_name = f"Statistical: {model_variant}"
                        if car:
                            model_name += " + spatial"
                        if pool:
                            model_name += (
                                " + state-specific vaccine coefficients"
                            )
                        if start_year == -2:
                            model_name += ", trained on last two years"
                        elif start_year == 2011:
                            model_name += ", trained since 2011"
                        elif start_year == 2019:
                            model_name += ", trained since 2019"
                        else:
                            assert False, "Invalid training start year"

                        df = df.with_columns(
                            pl.lit(validation_year).alias("Evaluation year"),
                            Model=pl.lit(model_name),
                        )

                        # let's now calculate the other stuff we need
                        df = other_tables(
                            df,
                            samples_unconditioned,
                            samples,
                            true,
                            true_outbreak,
                        )

                        model_performance.append(df)

            # need to also read the mechanistic model for this training variant
            with open(
                os.path.join(
                    output_dir,
                    f"predictive_final_outbreak_size_predictions_{years}.pkl",
                ),
                "rb",
            ) as f:
                samples = pickle.load(
                    f,
                )["cases"]

            samples_unconditioned = samples * all_imports[idx]

            samples_unconditioned = jnp.mean(samples_unconditioned, axis=0)
            samples = jnp.mean(samples, axis=0)

            df = performance_table(
                samples_unconditioned, true, mode="selection"
            )
            model_name = "Mechanistic: MMR coverage"
            if start_year == -2:
                model_name += ", trained on last two years"
            elif start_year == 2011:
                model_name += ", trained since 2011"
            elif start_year == 2019:
                model_name += ", trained since 2019"
            else:
                assert False, "Invalid training start year"

            df = df.with_columns(
                pl.lit(validation_year).alias("Evaluation year"),
                Model=pl.lit(model_name),
            )
            # other stuff we need
            df = other_tables(
                df, samples_unconditioned, samples, true, true_outbreak
            )
            model_performance.append(df)

    # let's do some clean up of the tables and export
    model_performance = pl.concat(model_performance)
    model_performance = aggregate(
        model_performance,
        [
            col
            for col in model_performance.columns
            if col != "Model" and col != "Evaluation year"
        ],
    )
    model_performance = model_performance.filter(
        pl.col("Evaluation year") == "2021-2025"
    ).drop("Evaluation year")

    # model selection only requires certain columns
    model_selection_columns = [
        "Absolute difference (cases)",
        "AUROC, 10+ cases (%)",
        "MAE, top 50% (cases)",
    ]
    model_selection = model_performance.select(
        [
            *[
                "Model",
            ],
            *[
                f"{col}{suffix}"
                for col in model_selection_columns
                for suffix in ["", "_lower", "_upper"]
            ],
        ]
    )
    model_selection = model_selection.sort(
        by=model_selection_columns, descending=False
    )
    model_selection = beautify(model_selection, model_selection_columns)
    # save
    model_selection.write_csv(
        os.path.join(fig_dir, "table_supplemental_model_selection_metrics.csv")
    )

    # we want the best performing mechanistic model
    # we want the best performing statistical model
    # we want the best performing statistical model w/out spatial
    # we want the best performing statistical model with all covariates
    # we want the best performing statistical model with all vaccination metrics
    main_models = [
        find_best_model(model_performance, subset="Mechanistic"),
        find_best_model(model_performance, subset="Statistical"),
        "Statistical: All covariates + spatial + state-specific vaccine coefficients, trained since 2019",
        find_best_model(
            model_performance, subset="Statistical", exclude="spatial"
        ),
        find_best_model(model_performance, subset="exemptions"),
        find_best_model(model_performance, subset="All covariates"),
    ]

    # main table only requires certain columns
    main_table_columns = [
        # 'Absolute difference (cases)',
        "Sensitivity, 10+ cases (%)",
        "Specificity, 10+ cases (%)",
        "AUROC, 10+ cases (%)",
        "Sensitivity, 20+ cases (%)",
        "Specificity, 20+ cases (%)",
        "AUROC, 20+ cases (%)",
        "Median fold error, top 20%",
        "Spearman rho^2, top 20% (%)",
        "RMSE, top 20% (cases)",
        "MAE, top 20% (cases)",
    ]
    main_table = model_performance.select(
        [
            *[
                "Model",
            ],
            *[
                f"{col}{suffix}"
                for col in main_table_columns
                for suffix in ["", "_lower", "_upper"]
            ],
        ]
    )
    # select set models
    main_table = main_table.filter(pl.col("Model").is_in(main_models))
    # order the models in a particular way we want
    main_table = main_table.with_columns(
        pl.col("Model").cast(pl.Enum(main_models))
    ).sort(by="Model")

    main_table = beautify(main_table, main_table_columns)
    # export
    main_table.write_csv(
        os.path.join(fig_dir, "table_validation_performance_metrics.csv")
    )
    # supplemental classification table with imports only requires certain columns
    classification_with_imports_columns = [
        "Sensitivity, 5+ cases (%)",
        "Specificity, 5+ cases (%)",
        "AUROC, 5+ cases (%)",
        "Sensitivity, 10+ cases (%)",
        "Specificity, 10+ cases (%)",
        "AUROC, 10+ cases (%)",
        "Sensitivity, 20+ cases (%)",
        "Specificity, 20+ cases (%)",
        "AUROC, 20+ cases (%)",
    ]
    classification_with_imports = model_performance.select(
        [
            *["Model"],
            *[
                f"{col}{suffix}"
                for col in classification_with_imports_columns
                for suffix in ["", "_lower", "_upper"]
            ],
        ]
    )
    # beautify
    classification_with_imports = beautify(
        classification_with_imports, classification_with_imports_columns
    )
    # export
    classification_with_imports.write_csv(
        os.path.join(
            fig_dir,
            "table_supplemental_classification_metrics_with_importations.csv",
        )
    )

    # supplemental classification table w/out imports only requires certain columns
    classification_wo_imports_columns = [
        "Sensitivity, 5+ outbreak cases (%)",
        "Specificity, 5+ outbreak cases (%)",
        "AUROC, 5+ outbreak cases (%)",
        "Sensitivity, 10+ outbreak cases (%)",
        "Specificity, 10+ outbreak cases (%)",
        "AUROC, 10+ outbreak cases (%)",
        "Sensitivity, 20+ outbreak cases (%)",
        "Specificity, 20+ outbreak cases (%)",
        "AUROC, 20+ outbreak cases (%)",
    ]
    classification_wo_imports = model_performance.select(
        [
            *["Model"],
            *[
                f"{col}{suffix}"
                for col in classification_wo_imports_columns
                for suffix in ["", "_lower", "_upper"]
            ],
        ]
    )
    # beautify
    classification_wo_imports = beautify(
        classification_wo_imports, classification_wo_imports_columns
    )
    # export
    classification_wo_imports.write_csv(
        os.path.join(
            fig_dir,
            "table_supplemental_classification_metrics_without_importations.csv",
        )
    )

    # supplemental regression table only requires certain columns
    regression_columns = [
        "Median fold error, top 20%",
        "RMSE, top 20% (cases)",
        "MAE, top 20% (cases)",
        "Spearman rho^2, top 20% (%)",
        "Median fold error, top 50%",
        "RMSE, top 50% (cases)",
        "MAE, top 50% (cases)",
        "Spearman rho^2, top 50% (%)",
        "Median fold error, top 80%",
        "RMSE, top 80% (cases)",
        "MAE, top 80% (cases)",
        "Spearman rho^2, top 80% (%)",
    ]
    regression = model_performance.select(
        [
            *["Model"],
            *[
                f"{col}{suffix}"
                for col in regression_columns
                for suffix in ["", "_lower", "_upper"]
            ],
        ]
    )
    # beautify
    regression = beautify(regression, regression_columns)
    # export
    regression.write_csv(
        os.path.join(fig_dir, "table_supplemental_regression_metrics.csv")
    )
