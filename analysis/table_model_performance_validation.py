import os
import pickle

import numpy as np
from numpyro import enable_x64
from tqdm.auto import tqdm

enable_x64()

import jax.numpy as jnp
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)
fig_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")


def performance_table(
    df,
    models,
    subset_models,
    threshold,
    with_importation,
    include_classification,
    classification_only,
    model_selection,
):
    models = (
        {k: v for k, v in models.items() if k in subset_models}
        if subset_models is not None
        else models
    )

    projection_years = [2021, 2022, 2023, 2024, 2025]
    performance_all = []
    for idx, year in enumerate(projection_years):
        performance = pl.DataFrame()
        if model_selection is not None:
            performance = performance.with_columns(
                Model=pl.Series(models.keys()),
                R2=pl.lit(None),
                R2_lower=pl.lit(None),
                R2_upper=pl.lit(None),
                adj_R2=pl.lit(None),
                adj_R2_lower=pl.lit(None),
                adj_R2_upper=pl.lit(None),
                MAE=pl.lit(None),
                MAE_lower=pl.lit(None),
                MAE_upper=pl.lit(None),
                AUROC=pl.lit(None),
                Year=pl.lit(year),
                n=pl.lit(None),
                n_full=pl.lit(None),
                k=pl.Series(model_selection[:, idx]),
            )
        else:
            performance = performance.with_columns(
                Model=pl.Series(models.keys()),
                R2=pl.lit(None),
                R2_lower=pl.lit(None),
                R2_upper=pl.lit(None),
                # spearman_rho = pl.lit(None),
                # spearman_rho_lower = pl.lit(None),
                # spearman_rho_upper = pl.lit(None),
                # kendall_tau = pl.lit(None),
                # kendall_tau_lower = pl.lit(None),
                # kendall_tau_upper = pl.lit(None),
                RMSE=pl.lit(None),
                RMSE_lower=pl.lit(None),
                RMSE_upper=pl.lit(None),
                MAE=pl.lit(None),
                MAE_lower=pl.lit(None),
                MAE_upper=pl.lit(None),
                Year=pl.lit(year),
                AUROC=pl.lit(None),
                average_precision=pl.lit(None),
                n=pl.lit(None),
                n_full=pl.lit(None),
            ).with_columns(
                pl.lit(None).alias("Sensitivity, 10+"),
                pl.lit(None).alias("Sensitivity, 10+_lower"),
                pl.lit(None).alias("Sensitivity, 10+_upper"),
                pl.lit(None).alias("Specificity, 10+"),
                pl.lit(None).alias("Specificity, 10+_lower"),
                pl.lit(None).alias("Specificity, 10+_upper"),
                pl.lit(None).alias("Sensitivity, 2+"),
                pl.lit(None).alias("Sensitivity, 2+_lower"),
                pl.lit(None).alias("Sensitivity, 2+_upper"),
                pl.lit(None).alias("Specificity, 2+"),
                pl.lit(None).alias("Specificity, 2+_lower"),
                pl.lit(None).alias("Specificity, 2+_upper"),
                pl.lit(None).alias("Sensitivity, 5+"),
                pl.lit(None).alias("Sensitivity, 5+_lower"),
                pl.lit(None).alias("Sensitivity, 5+_upper"),
                pl.lit(None).alias("Specificity, 5+"),
                pl.lit(None).alias("Specificity, 5+_lower"),
                pl.lit(None).alias("Specificity, 5+_upper"),
                pl.lit(None).alias("Sensitivity, 20+"),
                pl.lit(None).alias("Sensitivity, 20+_lower"),
                pl.lit(None).alias("Sensitivity, 20+_upper"),
                pl.lit(None).alias("Specificity, 20+"),
                pl.lit(None).alias("Specificity, 20+_lower"),
                pl.lit(None).alias("Specificity, 20+_upper"),
            )

        if with_importation is None:
            outcome = "cases_outbreak"
        else:
            outcome = "cases"
            if model_selection is None:
                performance = performance.with_columns(
                    fold_error=pl.lit(None),
                    fold_error_lower=pl.lit(None),
                    fold_error_upper=pl.lit(None),
                )
        year_mask = (df["year"] == year).to_numpy()
        threshold_val = np.percentile(
            df.filter(pl.col("year") == year)[outcome].to_numpy(),
            100 - threshold,
        )
        mask = (
            df.filter(pl.col("year") == year)[outcome].to_numpy()
            >= threshold_val
        )
        true_full = df.filter(pl.col("year") == year)[outcome].to_numpy()
        true = true_full[mask]
        n_full = true_full.shape[0]
        n = true.shape[0]
        true = np.array(true, dtype=np.float64)
        true_full = np.array(true_full, dtype=np.float64)

        for model in tqdm(models.keys()):
            # calculate R2, RMSE, MAE, Sensitivity for 10+ cases, Specificity for 10+ cases
            samples_full = models[model][idx]
            samples_full = np.array(samples_full, dtype=np.float64)
            samples = samples_full[:, mask].copy()

            if with_importation is not None:
                samples *= np.array(
                    with_importation[:, year_mask][:, mask], dtype=np.float64
                )
                samples_full *= np.array(
                    with_importation[:, year_mask], dtype=np.float64
                )

            if not classification_only:
                R2 = 100 * np.array(
                    [
                        np.corrcoef(true, c)[0, 1] ** 2
                        for c in np.sort(samples, axis=0)
                    ]
                )
                R2_median = np.nanmedian(R2)
                R2_median = round(float(R2_median), 2)
                R2_lower = round(float(np.nanpercentile(R2, 2.5)), 2)
                R2_upper = round(float(np.nanpercentile(R2, 97.5)), 2)

                # let us make samples have the same levels as true
                samples = samples - np.mean(samples, axis=1, keepdims=True)
                samples = (
                    samples
                    / np.std(samples, axis=1, keepdims=True)
                    * np.std(true)
                )
                samples = samples + np.mean(true)
                samples = np.sort(samples, axis=0)

                rmse = np.sqrt(np.mean((true - samples) ** 2, axis=1))
                rmse_median = round(float(np.nanmedian(rmse)), 2)
                rmse_lower = round(float(np.nanpercentile(rmse, 2.5)), 2)
                rmse_upper = round(float(np.nanpercentile(rmse, 97.5)), 2)

                if with_importation is not None and model_selection is None:
                    fold_error = np.nanmedian(
                        np.exp(np.abs(np.log(samples / true))), axis=1
                    )
                    fold_error_median = round(
                        float(np.nanmedian(fold_error)), 2
                    )
                    fold_error_lower = round(
                        float(np.nanpercentile(fold_error, 2.5)), 2
                    )
                    fold_error_upper = round(
                        float(np.nanpercentile(fold_error, 97.5)), 2
                    )

                mae = np.nanmedian(np.abs(true - samples), axis=1)
                mae_median = round(float(np.nanmedian(mae)), 2)
                mae_lower = round(float(np.nanpercentile(mae, 2.5)), 2)
                mae_upper = round(float(np.nanpercentile(mae, 97.5)), 2)

            if model_selection is not None or include_classification:
                y_true = true_full >= 10
                y_true = y_true.astype(np.int64)
                samples_full = np.sort(samples_full, axis=0)
                # we want to calculate the sensitivity, specificity, auroc, and auprc
                # we need to figure out the best cutoff point for each draw from the posterior
                for i in range(samples_full.shape[0]):
                    temp = samples_full[i, :]
                    precision, recall, thresholds = precision_recall_curve(
                        y_true, temp
                    )
                    # f1_scores = 2 * (precision * recall) / (precision + recall)
                    # f1_scores = np.nan_to_num(f1_scores, nan=-np.inf)
                    distances = np.sqrt(
                        (1 - precision) ** 2 + (1 - recall) ** 2
                    )
                    optimal_idx = np.argmin(distances)
                    # fpr, tpr, thresholds = roc_curve(y_true, temp)
                    # youdens_j = tpr - fpr
                    # optimal_idx = np.argmax(youdens_j)
                    if optimal_idx >= len(thresholds):
                        # this means every value is negative
                        if np.max(temp) != 0:
                            temp = temp / np.max(temp) * 10 - 1e-6
                    else:
                        optimal_threshold = thresholds[optimal_idx]
                        if optimal_threshold <= 0:
                            # we want to classify every value as positive
                            temp = temp + 10
                        elif optimal_threshold == np.inf:
                            # we want to classify every value as negative
                            if np.max(temp) != 0:
                                temp = temp / np.max(temp) * 10 - 1e-6
                        else:
                            temp = temp / optimal_threshold
                            temp = temp * 10
                    samples_full[i, :] = temp

                for threshold in [2, 5, 10, 20]:
                    cases_geq = samples_full >= threshold
                    # sensitivity asks for how many times is true >= 10 that cases is also
                    sensitivity = 100 * cases_geq[
                        :, true_full >= threshold
                    ].mean(axis=1)
                    sensitivity_median = np.nanmedian(sensitivity)
                    sensitivity_median = round(float(sensitivity_median), 2)
                    sensitivity_lower = round(
                        float(np.nanpercentile(sensitivity, 2.5)), 2
                    )
                    sensitivity_upper = round(
                        float(np.nanpercentile(sensitivity, 97.5)), 2
                    )
                    # specificity asks for how many times is true < 10 that cases is also
                    cases_le = samples_full < threshold
                    specificity = 100 * (
                        1 - cases_le[:, true_full < threshold]
                    ).mean(axis=1)
                    specificity_median = np.nanmedian(specificity)
                    specificity_median = round(float(specificity_median), 2)
                    specificity_lower = round(
                        float(np.nanpercentile(specificity, 2.5)), 2
                    )
                    specificity_upper = round(
                        float(np.nanpercentile(specificity, 97.5)), 2
                    )

                    # set the values
                    if model_selection is None:
                        performance = performance.with_columns(
                            pl.when(pl.col("Model") == model)
                            .then(pl.lit(sensitivity_median))
                            .otherwise(pl.col(f"Sensitivity, {threshold}+"))
                            .alias(f"Sensitivity, {threshold}+"),
                            pl.when(pl.col("Model") == model)
                            .then(pl.lit(sensitivity_lower))
                            .otherwise(
                                pl.col(f"Sensitivity, {threshold}+_lower")
                            )
                            .alias(f"Sensitivity, {threshold}+_lower"),
                            pl.when(pl.col("Model") == model)
                            .then(pl.lit(sensitivity_upper))
                            .otherwise(
                                pl.col(f"Sensitivity, {threshold}+_upper")
                            )
                            .alias(f"Sensitivity, {threshold}+_upper"),
                            pl.when(pl.col("Model") == model)
                            .then(pl.lit(specificity_median))
                            .otherwise(pl.col(f"Specificity, {threshold}+"))
                            .alias(f"Specificity, {threshold}+"),
                            pl.when(pl.col("Model") == model)
                            .then(pl.lit(specificity_lower))
                            .otherwise(
                                pl.col(f"Specificity, {threshold}+_lower")
                            )
                            .alias(f"Specificity, {threshold}+_lower"),
                            pl.when(pl.col("Model") == model)
                            .then(pl.lit(specificity_upper))
                            .otherwise(
                                pl.col(f"Specificity, {threshold}+_upper")
                            )
                            .alias(f"Specificity, {threshold}+_upper"),
                        )
                auc = round(
                    100
                    * roc_auc_score(
                        y_true, np.mean(samples_full >= 10, axis=0)
                    ),
                    2,
                )
                average_precision = round(
                    100
                    * average_precision_score(
                        y_true, np.mean(samples_full >= 10, axis=0)
                    ),
                    2,
                )

                # set the values
                if model_selection is None:
                    performance = performance.with_columns(
                        pl.when(pl.col("Model") == model)
                        .then(pl.lit(auc))
                        .otherwise(pl.col("AUROC"))
                        .alias("AUROC"),
                        pl.when(pl.col("Model") == model)
                        .then(pl.lit(average_precision))
                        .otherwise(pl.col("average_precision"))
                        .alias("average_precision"),
                    )
                else:
                    performance = performance.with_columns(
                        pl.when(pl.col("Model") == model)
                        .then(pl.lit(auc))
                        .otherwise(pl.col("AUROC"))
                        .alias("AUROC"),
                    )
            if not classification_only:
                performance = performance.with_columns(
                    R2=pl.when(pl.col("Model") == model)
                    .then(pl.lit(R2_median))  # type: ignore
                    .otherwise(pl.col("R2")),
                    R2_lower=pl.when(pl.col("Model") == model)
                    .then(pl.lit(R2_lower))  # type: ignore
                    .otherwise(pl.col("R2_lower")),
                    R2_upper=pl.when(pl.col("Model") == model)
                    .then(pl.lit(R2_upper))  # type: ignore
                    .otherwise(pl.col("R2_upper")),
                    # spearman_rho = pl.when(pl.col("Model") == model).then(pl.lit(spearman_rho_median)).otherwise(pl.col("spearman_rho")),
                    # spearman_rho_lower = pl.when(pl.col("Model") == model).then(pl.lit(spearman_rho_lower)).otherwise(pl.col("spearman_rho_lower")),
                    # spearman_rho_upper = pl.when(pl.col("Model") == model).then(pl.lit(spearman_rho_upper)).otherwise(pl.col("spearman_rho_upper")),
                    # kendall_tau = pl.when(pl.col("Model") == model).then(pl.lit(kendall_tau_median)).otherwise(pl.col("kendall_tau")),
                    # kendall_tau_lower = pl.when(pl.col("Model") == model).then(pl.lit(kendall_tau_lower)).otherwise(pl.col("kendall_tau_lower")),
                    # kendall_tau_upper = pl.when(pl.col("Model") == model).then(pl.lit(kendall_tau_upper)).otherwise(pl.col("kendall_tau_upper")),
                    MAE=pl.when(pl.col("Model") == model)
                    .then(pl.lit(mae_median))  # type: ignore
                    .otherwise(pl.col("MAE")),
                    MAE_lower=pl.when(pl.col("Model") == model)
                    .then(pl.lit(mae_lower))  # type: ignore
                    .otherwise(pl.col("MAE_lower")),
                    MAE_upper=pl.when(pl.col("Model") == model)
                    .then(pl.lit(mae_upper))  # type: ignore
                    .otherwise(pl.col("MAE_upper")),
                )

                if model_selection is None:
                    performance = performance.with_columns(
                        RMSE=pl.when(pl.col("Model") == model)
                        .then(pl.lit(rmse_median))  # type: ignore
                        .otherwise(pl.col("RMSE")),
                        RMSE_lower=pl.when(pl.col("Model") == model)
                        .then(pl.lit(rmse_lower))  # type: ignore
                        .otherwise(pl.col("RMSE_lower")),
                        RMSE_upper=pl.when(pl.col("Model") == model)
                        .then(pl.lit(rmse_upper))  # type: ignore
                        .otherwise(pl.col("RMSE_upper")),
                    )

                if with_importation is not None and model_selection is None:
                    performance = performance.with_columns(
                        pl.when(pl.col("Model") == model)
                        .then(pl.lit(fold_error_median))  # type: ignore
                        .otherwise(pl.col("fold_error"))
                        .alias("fold_error"),
                        pl.when(pl.col("Model") == model)
                        .then(pl.lit(fold_error_lower))  # type: ignore
                        .otherwise(pl.col("fold_error_lower"))
                        .alias("fold_error_lower"),
                        pl.when(pl.col("Model") == model)
                        .then(pl.lit(fold_error_upper))  # type: ignore
                        .otherwise(pl.col("fold_error_upper"))
                        .alias("fold_error_upper"),
                    )

            performance = performance.with_columns(
                pl.when(pl.col("Model") == model)
                .then(pl.lit(n))
                .otherwise(pl.col("n"))
                .alias("n"),
                pl.when(pl.col("Model") == model)
                .then(pl.lit(n_full))
                .otherwise(pl.col("n_full"))
                .alias("n_full"),
            )

        performance_all.append(performance)

    performance = pl.concat(performance_all).sort(["Model", "Year"])

    if not include_classification:
        performance = performance.drop(
            [
                x
                for x in performance.columns
                if "Sensitivity" in x
                or "Specificity" in x
                or "AUROC" in x
                or "average_precision" in x
            ]
        )

    if classification_only:
        performance = performance.select(
            ["Model", "Year", "n", "n_full"]
            + [
                x
                for x in performance.columns
                if "Sensitivity" in x
                or "Specificity" in x
                or "AUROC" in x
                or "average_precision" in x
            ]
        )
    if model_selection is not None:
        # calculate adjusted R2
        performance = performance.with_columns(
            adj_R2=(
                1
                - (1 - pl.col("R2") / 100)
                * (pl.col("n") - 1)
                / (pl.col("n") - pl.col("k") - 1)
            )
            * 100,
            adj_R2_lower=(
                1
                - (1 - pl.col("R2_lower") / 100)
                * (pl.col("n") - 1)
                / (pl.col("n") - pl.col("k") - 1)
            )
            * 100,
            adj_R2_upper=(
                1
                - (1 - pl.col("R2_upper") / 100)
                * (pl.col("n") - 1)
                / (pl.col("n") - pl.col("k") - 1)
            )
            * 100,
        )

        performance_grouped = (
            performance.group_by("Model")
            .agg(
                R2=(
                    pl.col("R2").filter(pl.col("n") > 2)
                    * pl.col("n").filter(pl.col("n") > 2)
                    / pl.col("n").filter(pl.col("n") > 2).sum()
                ).sum(),
                R2_lower=(
                    pl.col("R2_lower").filter(pl.col("n") > 2)
                    * pl.col("n").filter(pl.col("n") > 2)
                    / pl.col("n").filter(pl.col("n") > 2).sum()
                ).sum(),
                R2_upper=(
                    pl.col("R2_upper").filter(pl.col("n") > 2)
                    * pl.col("n").filter(pl.col("n") > 2)
                    / pl.col("n").filter(pl.col("n") > 2).sum()
                ).sum(),
                MAE=(pl.col("MAE") * pl.col("n") / pl.sum("n")).sum(),
                MAE_lower=(
                    pl.col("MAE_lower") * pl.col("n") / pl.sum("n")
                ).sum(),
                MAE_upper=(
                    pl.col("MAE_upper") * pl.col("n") / pl.sum("n")
                ).sum(),
                AUROC=(
                    pl.col("AUROC") * pl.col("n_full") / pl.sum("n_full")
                ).sum(),
            )
            .with_columns(Year=pl.lit("Overall"))
        )
        # we're actually going to drop the non-necessary columns
        performance = performance.drop(
            [
                "adj_R2",
                "adj_R2_lower",
                "adj_R2_upper",
                "n",
                "n_full",
                "k",
            ],
            strict=False,
        )
        performance = pl.concat(
            [performance, performance_grouped], how="diagonal_relaxed"
        )
        performance = performance[
            [
                "Model",
                "Year",
                "R2",
                "R2_lower",
                "R2_upper",
                "MAE",
                "MAE_lower",
                "MAE_upper",
                "AUROC",
            ]
        ]
        performance = performance.sort(
            ["Year", "AUROC", "R2"], descending=[False, True, True]
        )
        performance = performance.with_columns(
            # round everythin that is numeric to two decimal places
            [
                pl.col(x).round(2)
                for x in performance.columns
                if x != "Model" and x != "Year"
            ]
        )

    else:
        n_full_metrics = ["AUROC", "average_precision"]
        agg = []
        for metric in performance.columns:
            if (
                metric == "Model"
                or metric == "Year"
                or metric == "n"
                or metric == "n_full"
                or metric == "k"
            ):
                continue
            else:
                if (
                    metric == "R2"
                    or metric == "R2_lower"
                    or metric == "R2_upper"
                ):
                    agg.append(
                        (
                            pl.col(metric).filter(pl.col("n") > 2)
                            * pl.col("n").filter(pl.col("n") > 2)
                            / pl.col("n").filter(pl.col("n") > 2).sum()
                        ).sum()
                    )
                else:
                    if (
                        metric in n_full_metrics
                        or "Sensitivity" in metric
                        or "Specificity" in metric
                    ):
                        n = "n_full"
                    else:
                        n = "n"
                    agg.append((pl.col(metric) * pl.col(n) / pl.sum(n)).sum())

        performance_grouped = performance.group_by("Model").agg(agg)
        performance_grouped = performance_grouped.with_columns(
            Year=pl.lit("Overall")
        )
        performance = pl.concat(
            [performance, performance_grouped], how="diagonal_relaxed"
        )

        performance = performance.with_columns(
            # round everythin that is numeric to two decimal places
            [
                pl.col(x).round(2)
                for x in performance.columns
                if x != "Model" and x != "Year"
            ]
        )

    columns_metrics = [
        x
        for x in performance.columns
        if "lower" not in x
        and "upper" not in x
        and x
        not in [
            "Model",
            "AUROC",
            "average_precision",
            "Year",
            "n",
            "n_full",
            "k",
        ]
    ]

    beautified = performance.with_columns(
        # let's concatenate the lower and upper bounds into one column
        [
            (
                pl.col(x).cast(pl.String)
                + " ["
                + pl.col(f"{x}_lower").cast(pl.String)
                + ", "
                + pl.col(f"{x}_upper").cast(pl.String)
                + "]"
            ).alias(x)
            for x in columns_metrics
        ]
    )
    beautified = beautified.drop(
        [f"{x}_lower" for x in columns_metrics]
        + [f"{x}_upper" for x in columns_metrics]
    )

    # beautify performance
    beautified = beautified.rename(
        {
            "Sensitivity, 10+": "Sensitivity, 10+ cases (%)",
            "Specificity, 10+": "Specificity, 10+ cases (%)",
            "Sensitivity, 2+": "Sensitivity, 2+ cases (%)",
            "Specificity, 2+": "Specificity, 2+ cases (%)",
            "Sensitivity, 5+": "Sensitivity, 5+ cases (%)",
            "Specificity, 5+": "Specificity, 5+ cases (%)",
            "Sensitivity, 20+": "Sensitivity, 20+ cases (%)",
            "Specificity, 20+": "Specificity, 20+ cases (%)",
            "RMSE": "RMSE (cases)",
            "MAE": "MAE (cases)",
            "R2": "R² (%)",
            "AUROC": "AU-ROC, 10+ cases (%)",
            "average_precision": "Average precision, 10+ cases (%)",
        },
        strict=False,
    )
    if (
        with_importation is not None
        and model_selection is None
        and not classification_only
    ):
        beautified = beautified.rename(
            {
                "fold_error": "Fold error",
            }
        )
    if model_selection is not None:
        beautified = beautified.rename(
            {
                "adj_R2": "Adjusted R² (%)",
            },
            strict=False,
        )

    beautified = beautified.drop(["n", "n_full", "k"], strict=False)

    return performance, beautified


if __name__ == "__main__":
    df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        schema_overrides={"fips": pl.String},
    )

    # we need to read the following models
    # read the mechanistic model separately because it is stored differently
    models = {}
    model_selection = np.zeros((39, 5))
    mechanistic = []
    for idx, year in enumerate(
        [
            [2019, 2020],
            [2019, 2020, 2021],
            [2019, 2020, 2021, 2022],
            [2019, 2020, 2021, 2022, 2023],
            [2019, 2020, 2021, 2022, 2023, 2024],
        ]
    ):
        with open(
            os.path.join(
                output_dir, f"final_outbreak_size_predictions_{year}.pkl"
            ),
            "rb",
        ) as f:
            m = pickle.load(
                f,
            )
        # this regression has two free parameters -- the slope and intercept
        model_selection[0, idx] = 2

        mechanistic.append(m["cases"])
    models["Mechanistic: MMR coverage, starting 2019"] = mechanistic

    mechanistic = []
    for idx, year in enumerate(
        [
            [2019, 2020],
            [2020, 2021],
            [2021, 2022],
            [2022, 2023],
            [2023, 2024],
        ]
    ):
        with open(
            os.path.join(
                output_dir, f"final_outbreak_size_predictions_{year}.pkl"
            ),
            "rb",
        ) as f:
            m = pickle.load(
                f,
            )
        # this regression has two free parameters -- the slope and intercept
        model_selection[0, idx] = 2

        mechanistic.append(m["cases"])
    models["Mechanistic: MMR coverage, last two years"] = mechanistic

    mechanistic = []
    for idx, year in enumerate(
        [
            [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
            [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
            [
                2011,
                2012,
                2013,
                2014,
                2015,
                2016,
                2017,
                2018,
                2019,
                2020,
                2021,
                2022,
            ],
            [
                2011,
                2012,
                2013,
                2014,
                2015,
                2016,
                2017,
                2018,
                2019,
                2020,
                2021,
                2022,
                2023,
            ],
            [
                2011,
                2012,
                2013,
                2014,
                2015,
                2016,
                2017,
                2018,
                2019,
                2020,
                2021,
                2022,
                2023,
                2024,
            ],
        ]
    ):
        with open(
            os.path.join(
                output_dir, f"final_outbreak_size_predictions_{year}.pkl"
            ),
            "rb",
        ) as f:
            m = pickle.load(
                f,
            )
        # this regression has two free parameters -- the slope and intercept
        model_selection[0, idx] = 2

        mechanistic.append(m["cases"])
    models["Mechanistic: MMR coverage, starting 2011"] = mechanistic

    # we now have to read the different poisson models
    # the model options are covariate versions, with/without pooling, with and without car
    # for the covariate versions, if car is true, we add the number of counties for that year
    model_counter = 3
    for car in [True, False]:
        for pool in [True, False]:
            for model in [
                "MMR coverage",
                "MMR coverage + exemptions",
                "All covariates",
            ]:
                predictions = []
                for idx, year in enumerate(
                    [
                        [2019, 2020],
                        [2019, 2020, 2021],
                        [2019, 2020, 2021, 2022],
                        [2019, 2020, 2021, 2022, 2023],
                        [2019, 2020, 2021, 2022, 2023, 2024],
                    ]
                ):
                    k = 0
                    with open(
                        os.path.join(
                            output_dir,
                            f"samples_partial_pooled_poisson_validation_testing_{year}_pool={pool}_CAR={car}_{model}.pkl",
                        ),
                        "rb",
                    ) as file:
                        p = pickle.load(file)
                    predictions.append(p)
                    if model == "MMR coverage":
                        k += 1  # 1 variable
                    elif model == "MMR coverage + exemptions":
                        k += 3  # 3 variables
                    else:
                        k += 12  # 12 variables
                    if pool:
                        n_states = (
                            df.filter(pl.col("year").is_in(year))
                            .select(pl.col("state"))
                            .n_unique()
                        )
                        if model == "MMR coverage":
                            k += 1  # 1 variance parameter
                            k += n_states  # each state has their own shifts
                        else:
                            k += 3
                            k += n_states * 3
                    if car:
                        k += (
                            df.filter(pl.col("year").is_in(year))
                            .select(pl.col("county"))
                            .n_unique()
                        )

                    model_selection[model_counter, idx] = k

                additional_info = ", starting 2019"
                if pool:
                    additional_info += ", partial pooled"
                if car:
                    additional_info += ", spatial"
                models[f"Statistical: {model}{additional_info}"] = predictions
                model_counter += 1

    for car in [True, False]:
        for pool in [True, False]:
            for model in [
                "MMR coverage",
                "MMR coverage + exemptions",
                "All covariates",
            ]:
                predictions = []
                for idx, year in enumerate(
                    [
                        [2019, 2020],
                        [2020, 2021],
                        [2021, 2022],
                        [2022, 2023],
                        [2023, 2024],
                    ]
                ):
                    k = 0
                    with open(
                        os.path.join(
                            output_dir,
                            f"samples_partial_pooled_poisson_validation_testing_{year}_pool={pool}_CAR={car}_{model}.pkl",
                        ),
                        "rb",
                    ) as file:
                        p = pickle.load(file)
                    predictions.append(p)
                    if model == "MMR coverage":
                        k += 1  # 1 variable
                    elif model == "MMR coverage + exemptions":
                        k += 3  # 3 variables
                    else:
                        k += 12  # 12 variables
                    if pool:
                        n_states = (
                            df.filter(pl.col("year").is_in(year))
                            .select(pl.col("state"))
                            .n_unique()
                        )
                        if model == "MMR coverage":
                            k += 1  # 1 variance parameter
                            k += n_states  # each state has their own shifts
                        else:
                            k += 3
                            k += n_states * 3
                    if car:
                        k += (
                            df.filter(pl.col("year").is_in(year))
                            .select(pl.col("county"))
                            .n_unique()
                        )

                    model_selection[model_counter, idx] = k

                additional_info = ", last two years"
                if pool:
                    additional_info += ", partial pooled"
                if car:
                    additional_info += ", spatial"
                models[f"Statistical: {model}{additional_info}"] = predictions
                model_counter += 1

    for car in [True, False]:
        for pool in [True, False]:
            for model in [
                "MMR coverage",
                "MMR coverage + exemptions",
                "All covariates",
            ]:
                predictions = []
                for idx, year in enumerate(
                    [
                        [
                            2011,
                            2012,
                            2013,
                            2014,
                            2015,
                            2016,
                            2017,
                            2018,
                            2019,
                            2020,
                        ],
                        [
                            2011,
                            2012,
                            2013,
                            2014,
                            2015,
                            2016,
                            2017,
                            2018,
                            2019,
                            2020,
                            2021,
                        ],
                        [
                            2011,
                            2012,
                            2013,
                            2014,
                            2015,
                            2016,
                            2017,
                            2018,
                            2019,
                            2020,
                            2021,
                            2022,
                        ],
                        [
                            2011,
                            2012,
                            2013,
                            2014,
                            2015,
                            2016,
                            2017,
                            2018,
                            2019,
                            2020,
                            2021,
                            2022,
                            2023,
                        ],
                        [
                            2011,
                            2012,
                            2013,
                            2014,
                            2015,
                            2016,
                            2017,
                            2018,
                            2019,
                            2020,
                            2021,
                            2022,
                            2023,
                            2024,
                        ],
                    ]
                ):
                    k = 0
                    with open(
                        os.path.join(
                            output_dir,
                            f"samples_partial_pooled_poisson_validation_testing_{year}_pool={pool}_CAR={car}_{model}.pkl",
                        ),
                        "rb",
                    ) as file:
                        p = pickle.load(file)
                    predictions.append(p)
                    if model == "MMR coverage":
                        k += 1  # 1 variable
                    elif model == "MMR coverage + exemptions":
                        k += 3  # 3 variables
                    else:
                        k += 12  # 12 variables
                    if pool:
                        n_states = (
                            df.filter(pl.col("year").is_in(year))
                            .select(pl.col("state"))
                            .n_unique()
                        )
                        if model == "MMR coverage":
                            k += 1  # 1 variance parameter
                            k += n_states  # each state has their own shifts
                        else:
                            k += 3
                            k += n_states * 3
                    if car:
                        k += (
                            df.filter(pl.col("year").is_in(year))
                            .select(pl.col("county"))
                            .n_unique()
                        )

                    model_selection[model_counter, idx] = k

                additional_info = ", starting 2011"
                if pool:
                    additional_info += ", partial pooled"
                if car:
                    additional_info += ", spatial"
                models[f"Statistical: {model}{additional_info}"] = predictions
                model_counter += 1

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
    import_df = import_df.filter(pl.col("type") == "train")

    # join up with df
    df = df.join(
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
    )
    importation_samples = jnp.array(
        np.vstack(df["importation_probability_samples"].to_numpy())  # type: ignore
    ).T

    _, model_selection_results = performance_table(
        df,
        models,
        subset_models=None,
        threshold=20,
        with_importation=importation_samples,
        include_classification=True,
        classification_only=False,
        model_selection=model_selection,
    )

    model_selection_results.write_csv(
        os.path.join(fig_dir, "table_supplemental_model_selection_metrics.csv")
    )

    _, classification_metrics_with_importation = performance_table(
        df,
        models,
        subset_models=None,
        threshold=20,
        with_importation=importation_samples,
        include_classification=True,
        classification_only=True,
        model_selection=None,
    )

    classification_metrics_with_importation.write_csv(
        os.path.join(
            fig_dir,
            "table_supplemental_classification_metrics_with_importation.csv",
        )
    )

    _, classification_metrics_no_importation = performance_table(
        df,
        models,
        subset_models=None,
        threshold=20,
        with_importation=None,
        include_classification=True,
        classification_only=True,
        model_selection=None,
    )

    classification_metrics_no_importation.write_csv(
        os.path.join(
            fig_dir,
            "table_supplemental_classification_metrics_no_importation.csv",
        )
    )

    subset_models = [
        "Mechanistic: MMR coverage, last two years",
        "Statistical: MMR coverage, starting 2019, partial pooled, spatial",
        "Statistical: MMR coverage + exemptions, last two years, spatial",
        "Statistical: All covariates, starting 2019, partial pooled, spatial",
    ]

    _, main_text_table = performance_table(
        df,
        models,
        subset_models=subset_models,
        threshold=20,
        with_importation=importation_samples,
        include_classification=True,
        classification_only=False,
        model_selection=None,
    )

    main_text_table = main_text_table[
        [
            "Model",
            "Year",
            "R² (%)",
            "MAE (cases)",
            "RMSE (cases)",
            "Fold error",
            "Sensitivity, 10+ cases (%)",
            "Specificity, 10+ cases (%)",
            "AU-ROC, 10+ cases (%)",
            "Average precision, 10+ cases (%)",
        ]
    ]
    main_text_table = main_text_table.filter(pl.col("Year") == "Overall")
    main_text_table = main_text_table.drop("Year")
    main_text_table = main_text_table.sort("Model")

    main_text_table.write_csv(
        os.path.join(fig_dir, "table_validation_performance_metrics.csv")
    )

    # table of with/without importation performance metrics
    # and at 20%, 50%, and 80% thresholds
    thresholds = [20, 50, 80]
    importations = [None, importation_samples]
    for threshold in thresholds:
        for importation in importations:
            if importation is None:
                vars_ordered = [
                    "Model",
                    "Year",
                    "R² (%)",
                    "MAE (cases)",
                    "RMSE (cases)",
                ]
                suffix = "no_importation"
            else:
                vars_ordered = [
                    "Model",
                    "Year",
                    "R² (%)",
                    "MAE (cases)",
                    "RMSE (cases)",
                    "Fold error",
                ]
                suffix = "with_importation"
            _, temp_table = performance_table(
                df,
                models,
                subset_models=None,
                threshold=threshold,
                with_importation=importation,
                include_classification=False,
                classification_only=False,
                model_selection=None,
            )
            temp_table = temp_table[vars_ordered]
            temp_table.write_csv(
                os.path.join(
                    fig_dir,
                    f"table_supplemental_performance_metrics_{suffix}_threshold_{threshold}.csv",
                )
            )

    # total case counts table for the supplemental
    df_counts = df.filter(pl.col("year").is_in([2021, 2022, 2023, 2024, 2025]))
    df_counts = (
        df_counts.group_by("year")
        .agg(
            total_cases=pl.sum("cases"),
            total_cases_outbreak=pl.sum("cases_outbreak"),
            n=pl.len(),
            total_cases_geq_10=(
                pl.when(pl.col("cases") >= 10).then(1).otherwise(0)
            ).sum(),
            total_cases_outbreak_geq_10=(
                pl.when(pl.col("cases_outbreak") >= 10).then(1).otherwise(0)
            ).sum(),
        )
        .sort("year")
        .rename(
            {
                "total_cases": "Total cases (with importation)",
                "total_cases_outbreak": "Total outbreak cases (without importation)",
                "total_cases_geq_10": "Total counties with ≥10 cases (with importation)",
                "total_cases_outbreak_geq_10": "Total counties with ≥10 outbreak cases (without importation)",
                "n": "Number of counties",
            }
        )
        .write_csv(
            os.path.join(
                fig_dir,
                "table_supplemental_total_case_counts_validation_time_frame.csv",
            )
        )
    )
