import pickle

import numpy as np
import polars as pl
from numpyro import enable_x64

enable_x64()
import os

import jax.numpy as jnp
from scipy import stats
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
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
    raw_samples=None,  # full posterior matrix [n_draws, n_locations] for absolute difference and coverage
    true_labels_override=None,  # override for multi-year outbreak detection (PPV only)
):
    if mode == "classification":
        # Choose optimization method
        OPTIMIZE_METHOD = "closest_corner"  # 'f1', 'youden', 'fbeta', 'markedness', 'mcc', 'closest_corner'
        BETA = 2.0

        # Use standard single-year labels for ROC analysis
        # convert true to int if it is not already
        true = np.array([int(t) for t in true])
        y_true = (true >= classification_threshold).astype(int)

        fpr, tpr, thresholds_roc = roc_curve(y_true, samples)

        if OPTIMIZE_METHOD == "youden":
            # ==================== YOUDEN'S J (INFORMEDNESS) ====================
            if np.isnan(fpr[0]) or np.isnan(tpr[0]):
                sensitivity = np.nan
                specificity = np.nan
                roc_auc = np.nan
            else:
                J = tpr - fpr  # Sensitivity + Specificity - 1
                idx = np.argmax(J)
                sensitivity = tpr[idx]
                specificity = 1.0 - fpr[idx]

                roc_auc = roc_auc_score(y_true, samples)

        elif OPTIMIZE_METHOD == "closest_corner":
            # ==================== CLOSEST TO (0,1) CORNER ====================
            # Minimize distance to perfect classifier at (FPR=0, TPR=1)
            if np.isnan(fpr[0]) or np.isnan(tpr[0]):
                sensitivity = np.nan
                specificity = np.nan
                roc_auc = np.nan
            else:
                # Distance = sqrt((FPR - 0)^2 + (TPR - 1)^2)
                # Equivalently: sqrt(FPR^2 + (1 - TPR)^2)
                distances = np.sqrt(fpr**2 + (1 - tpr) ** 2)
                idx = np.argmin(distances)  # Minimize distance

                sensitivity = tpr[idx]
                specificity = 1.0 - fpr[idx]

                roc_auc = roc_auc_score(y_true, samples)

        elif OPTIMIZE_METHOD == "mcc":
            # ==================== MATTHEWS CORRELATION COEFFICIENT ====================
            if np.isnan(fpr[0]) or np.isnan(tpr[0]):
                sensitivity = np.nan
                specificity = np.nan
                roc_auc = np.nan
            else:
                # Calculate MCC for each threshold
                mcc_scores = []
                for thresh in thresholds_roc:
                    y_pred = (samples >= thresh).astype(int)

                    tp = np.sum((y_true == 1) & (y_pred == 1))
                    fp = np.sum((y_true == 0) & (y_pred == 1))
                    tn = np.sum((y_true == 0) & (y_pred == 0))
                    fn = np.sum((y_true == 1) & (y_pred == 0))

                    # MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
                    numerator = tp * tn - fp * fn
                    denominator = np.sqrt(
                        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
                    )

                    mcc = numerator / denominator if denominator > 0 else 0
                    mcc_scores.append(mcc)

                mcc_scores = np.array(mcc_scores)
                idx = np.argmax(mcc_scores)

                sensitivity = tpr[idx]
                specificity = 1.0 - fpr[idx]

                roc_auc = roc_auc_score(y_true, samples)

        elif OPTIMIZE_METHOD == "markedness":
            # ==================== MARKEDNESS (PPV + NPV - 1) ====================
            if np.isnan(fpr[0]) or np.isnan(tpr[0]):
                sensitivity = np.nan
                specificity = np.nan
                roc_auc = np.nan
            else:
                # Calculate Markedness for each threshold
                markedness_scores = []
                for thresh in thresholds_roc:
                    y_pred = (samples >= thresh).astype(int)

                    tp = np.sum((y_true == 1) & (y_pred == 1))
                    fp = np.sum((y_true == 0) & (y_pred == 1))
                    tn = np.sum((y_true == 0) & (y_pred == 0))
                    fn = np.sum((y_true == 1) & (y_pred == 0))

                    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

                    markedness = ppv + npv - 1
                    markedness_scores.append(markedness)

                markedness_scores = np.array(markedness_scores)
                idx = np.argmax(markedness_scores)

                sensitivity = tpr[idx]
                specificity = 1.0 - fpr[idx]

                roc_auc = roc_auc_score(y_true, samples)

        elif OPTIMIZE_METHOD == "f1":
            # ==================== F1-SCORE METHOD ====================
            precision, recall, thresholds_pr = precision_recall_curve(
                y_true, samples
            )

            if np.isnan(precision[0]) or np.isnan(recall[0]):
                sensitivity = np.nan
                specificity = np.nan
                roc_auc = np.nan
            else:
                f1_scores = np.where(
                    (precision + recall) > 0,
                    2 * (precision * recall) / (precision + recall),
                    0,
                )

                idx = np.argmax(f1_scores)
                optimal_threshold = (
                    thresholds_pr[idx]
                    if idx < len(thresholds_pr)
                    else thresholds_pr[-1]
                )

                sensitivity = recall[idx]

                y_pred = (samples >= optimal_threshold).astype(int)
                tn = np.sum((y_true == 0) & (y_pred == 0))
                fp = np.sum((y_true == 0) & (y_pred == 1))
                specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

                roc_auc = roc_auc_score(y_true, samples)

        elif OPTIMIZE_METHOD == "fbeta":
            # ==================== F-BETA SCORE METHOD ====================
            precision, recall, thresholds_pr = precision_recall_curve(
                y_true, samples
            )

            if np.isnan(precision[0]) or np.isnan(recall[0]):
                sensitivity = np.nan
                specificity = np.nan
                roc_auc = np.nan
            else:
                fbeta_scores = np.where(
                    (precision + recall) > 0,
                    (1 + BETA**2)
                    * (precision * recall)
                    / (BETA**2 * precision + recall),
                    0,
                )

                idx = np.argmax(fbeta_scores)
                optimal_threshold = (
                    thresholds_pr[idx]
                    if idx < len(thresholds_pr)
                    else thresholds_pr[-1]
                )

                sensitivity = recall[idx]

                y_pred = (samples >= optimal_threshold).astype(int)
                tn = np.sum((y_true == 0) & (y_pred == 0))
                fp = np.sum((y_true == 0) & (y_pred == 1))
                specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

                roc_auc = roc_auc_score(y_true, samples)

        # ==================== ENFORCE MINIMUM SENSITIVITY CONSTRAINT ====================
        MIN_SENSITIVITY = 0.0

        if not np.isnan(sensitivity) and sensitivity < MIN_SENSITIVITY:  # type: ignore
            # Current threshold doesn't meet minimum sensitivity requirement
            # Find threshold that achieves at least MIN_SENSITIVITY with best specificity

            valid_indices = tpr >= MIN_SENSITIVITY

            if np.any(valid_indices):
                # Among thresholds meeting minimum sensitivity, pick one with best specificity
                valid_specificities = 1.0 - fpr[valid_indices]
                best_valid_idx = np.argmax(valid_specificities)
                idx = np.where(valid_indices)[0][best_valid_idx]

                sensitivity = tpr[idx]
                specificity = 1.0 - fpr[idx]
            else:
                # No threshold achieves MIN_SENSITIVITY, use highest available sensitivity
                idx = np.argmax(tpr)
                sensitivity = tpr[idx]
                specificity = 1.0 - fpr[idx]

        # Calculate PPV and NPV using the optimal threshold
        optimal_threshold = (
            thresholds_roc[idx] if not np.isnan(sensitivity) else np.nan
        )  # type: ignore

        if np.isnan(sensitivity) or np.isnan(specificity):  # type: ignore
            ppv = np.nan
            npv = np.nan
        else:
            y_pred = (samples >= optimal_threshold).astype(int)

            # For PPV and NPV: use multi-year labels if provided, otherwise use single-year
            if true_labels_override is not None:
                y_true_for_ppv_npv = true_labels_override.astype(int)
            else:
                y_true_for_ppv_npv = y_true

            # PPV calculation with multi-year outbreak definition
            tp = np.sum((y_true_for_ppv_npv == 1) & (y_pred == 1))
            fp = np.sum((y_true_for_ppv_npv == 0) & (y_pred == 1))
            ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan

            # NPV calculation with multi-year outbreak definition
            tn = np.sum((y_true_for_ppv_npv == 0) & (y_pred == 0))
            fn = np.sum((y_true_for_ppv_npv == 1) & (y_pred == 0))
            npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

        # make a dataframe
        performance_summary = {
            "Sensitivity (%)": sensitivity * 100,  # type: ignore
            "Specificity (%)": specificity * 100,  # type: ignore
            "AUROC (%)": roc_auc * 100,  # type: ignore
            "PPV (%)": ppv * 100,
            "NPV (%)": npv * 100,
        }

        performance_summary = pl.DataFrame(performance_summary)

    elif mode == "regression":
        # Build mask from threshold or percentile
        if regression_threshold is not None:
            assert regression_percentile is None, (
                "Can't provide both regression percentile and threshold"
            )
            true = np.array([int(t) for t in true])
            mask = true >= regression_threshold
        else:
            assert regression_percentile is not None, (
                "Must provide either regression_percentile or regression_threshold"
            )
            mask = true > 0
            threshold = jnp.percentile(true[mask], regression_percentile)
            mask = jnp.logical_and(true >= threshold, mask)

        # If no locations meet the cutoff for this year, return all-NaN row
        if int(mask.sum()) == 0:
            performance_summary = pl.DataFrame(
                {
                    "RMSE (cases)": [float("nan")],
                    "MAE (cases)": [float("nan")],
                    "Absolute difference (cases)": [
                        abs(jnp.mean(raw_samples.sum(axis=1)) - true.sum())
                        if raw_samples is not None
                        else float(abs(samples.sum() - true.sum()))
                    ],
                    "Median fold error": [float("nan")],
                    "Spearman rho": [float("nan")],
                    "R² (%)": [float("nan")],
                }
            )
            return performance_summary

        true_subset = true[mask]
        samples_subset = samples[mask]

        # calculate metrics
        rmse = jnp.sqrt(jnp.mean((samples_subset - true_subset) ** 2))
        mae = jnp.median(jnp.abs(samples_subset - true_subset))

        # absolute difference: |mean of posterior nationwide total - observed
        # nationwide total|, using the full posterior matrix
        if raw_samples is not None:
            nationwide_draws = raw_samples.sum(axis=1)
            predicted_mean = jnp.mean(nationwide_draws)
        else:
            predicted_mean = samples.sum()
        absolute_difference = abs(predicted_mean - true.sum())

        mfe = jnp.median(
            jnp.maximum(
                samples_subset / true_subset, true_subset / samples_subset
            )
        )
        rho = stats.spearmanr(true_subset, samples_subset)[0]  # type: ignore

        if len(true_subset) >= 2:
            pearson_r = stats.pearsonr(true_subset, samples_subset)[0]  # type: ignore
            r2 = pearson_r**2 * 100  # type: ignore
        else:
            r2 = float("nan")

        performance_summary = {
            "RMSE (cases)": rmse,
            "MAE (cases)": mae,
            "Absolute difference (cases)": absolute_difference,
            "Median fold error": mfe,
            "Spearman rho": rho,
            "R² (%)": r2,
        }

        performance_summary = pl.DataFrame(performance_summary)

    elif mode == "selection":
        # run both classification and regression
        classification = performance_table(
            samples,
            true,
            "classification",
            classification_threshold=10,
            true_labels_override=true_labels_override,
        )
        regression = performance_table(
            samples,
            true,
            "regression",
            regression_threshold=3,
            raw_samples=raw_samples,
        )

        # concatenate
        performance_summary = pl.concat(
            [classification, regression], how="horizontal"
        )

    else:
        assert False, "mode not found"

    return performance_summary


def coverage_table(raw_samples, true):
    """
    Calculate whether the observed nationwide total (true.sum()) falls
    within the 95% credible interval of the posterior nationwide total.

    Parameters
    ----------
    raw_samples : jnp.array, shape [n_draws, n_locations]
        Full posterior samples (unconditioned, i.e. including importations)
        before averaging across draws.
    true : np.array, shape [n_locations]
        Observed case counts.

    Returns
    -------
    pl.DataFrame with one column: "Coverage (%)" = 100.0 or 0.0
    """
    nationwide_draws = raw_samples.sum(axis=1)  # [n_draws]
    lower = jnp.percentile(nationwide_draws, 2.5)
    upper = jnp.percentile(nationwide_draws, 97.5)
    observed_total = true.sum()
    covered = float(lower <= observed_total <= upper)
    return pl.DataFrame({"Coverage (%)": [covered * 100.0]})


def aggregate(df, metric_columns):
    # create a weighted average value across all years
    df_agg = (
        df.group_by(["Model"], maintain_order=True)
        .agg(
            [
                *[
                    (pl.col(col).filter(pl.col(col).is_not_nan())).mean()
                    for col in metric_columns
                    if col != "Absolute difference (cases)"
                    and col != "Coverage (%)"
                ],
                *[
                    pl.col(col)
                    .filter(pl.col(col).is_not_nan())
                    .min()
                    .alias(f"{col}_lower")
                    for col in metric_columns
                    if col != "Absolute difference (cases)"
                    and col != "Coverage (%)"
                ],
                *[
                    pl.col(col)
                    .filter(pl.col(col).is_not_nan())
                    .max()
                    .alias(f"{col}_upper")
                    for col in metric_columns
                    if col != "Absolute difference (cases)"
                    and col != "Coverage (%)"
                ],
                # Absolute difference: sum across years, no uncertainty interval
                pl.col("Absolute difference (cases)")
                .filter(pl.col("Absolute difference (cases)").is_not_nan())
                .sum(),
                # Coverage: mean across years (fraction of years covered), no interval
                pl.col("Coverage (%)")
                .filter(pl.col("Coverage (%)").is_not_nan())
                .mean(),
            ]
        )
        .with_columns(pl.lit("2021-2025").alias("Evaluation year"))
    )

    # add this to the end
    df = pl.concat([df, df_agg], how="diagonal_relaxed")
    return df


def beautify(df, column_stub, no_interval_columns=None):
    """
    Format columns as "value [lower, upper]".

    Parameters
    ----------
    no_interval_columns : list of str, optional
        Columns that should be rounded and displayed as a plain number,
        with no [lower, upper] bracketing (e.g. "Absolute difference (cases)"
        and "Coverage (%)" in the model selection table).
    """
    if no_interval_columns is None:
        no_interval_columns = []

    interval_columns = [c for c in column_stub if c not in no_interval_columns]

    df = df.with_columns(
        [
            *[pl.col(col).round(1) for col in interval_columns],
            *[pl.col(f"{col}_lower").round(1) for col in interval_columns],
            *[pl.col(f"{col}_upper").round(1) for col in interval_columns],
            *[pl.col(col).round(1) for col in no_interval_columns],
        ]
    )
    # join interval columns
    df = df.with_columns(
        [
            pl.format(
                "{} [{}, {}]", col, f"{col}_lower", f"{col}_upper"
            ).alias(col)
            for col in interval_columns
        ]
    )
    # drop the lower and upper for interval columns
    df = df.drop([f"{col}_lower" for col in interval_columns])
    df = df.drop([f"{col}_upper" for col in interval_columns])

    return df


def other_tables(
    df,
    samples_unconditioned,
    samples,
    true,
    true_outbreak,
    true_labels_override_5=None,
    true_labels_override_10=None,
    true_labels_override_20=None,
):
    # this function accepts the model selection table, which has classification
    # metrics calculated for the 10+ threshold and regression for 3+.
    # For the main table we need classification for 10+ and regression for 3+
    # (both with and without imports).
    # For the supplemental tables we need classification for 5/10/20 and
    # regression for >= 0 / >= 3 / >= 10, both with and without imports.

    # first let's relabel the existing columns from mode="selection"
    df = df.rename(
        {
            "AUROC (%)": "AUROC, 10+ cases (%)",
            "Sensitivity (%)": "Sensitivity, 10+ cases (%)",
            "Specificity (%)": "Specificity, 10+ cases (%)",
            "PPV (%)": "PPV, 10+ cases (%)",
            "NPV (%)": "NPV, 10+ cases (%)",
            "RMSE (cases)": "RMSE, 3+ cases (cases)",
            "MAE (cases)": "MAE, 3+ cases (cases)",
            "Median fold error": "Median fold error, 3+ cases",
            "Spearman rho": "Spearman rho, 3+ cases",
            "R² (%)": "R², 3+ cases (%)",
        }
    )

    # --- classification: with imports, 5+ and 20+ ---
    classification5_with_imports = performance_table(
        samples_unconditioned,
        true,
        mode="classification",
        classification_threshold=5,
        true_labels_override=true_labels_override_5,
    )
    classification5_with_imports = classification5_with_imports.rename(
        {
            "Sensitivity (%)": "Sensitivity, 5+ cases (%)",
            "Specificity (%)": "Specificity, 5+ cases (%)",
            "AUROC (%)": "AUROC, 5+ cases (%)",
            "PPV (%)": "PPV, 5+ cases (%)",
            "NPV (%)": "NPV, 5+ cases (%)",
        }
    )

    classification20_with_imports = performance_table(
        samples_unconditioned,
        true,
        mode="classification",
        classification_threshold=20,
        true_labels_override=true_labels_override_20,
    )
    classification20_with_imports = classification20_with_imports.rename(
        {
            "Sensitivity (%)": "Sensitivity, 20+ cases (%)",
            "Specificity (%)": "Specificity, 20+ cases (%)",
            "AUROC (%)": "AUROC, 20+ cases (%)",
            "PPV (%)": "PPV, 20+ cases (%)",
            "NPV (%)": "NPV, 20+ cases (%)",
        }
    )

    # --- classification: without imports, 5+, 10+, 20+ ---
    classification5_wo_imports = performance_table(
        samples,
        true_outbreak,
        mode="classification",
        classification_threshold=5,
        true_labels_override=true_labels_override_5,
    )
    classification5_wo_imports = classification5_wo_imports.rename(
        {
            "Sensitivity (%)": "Sensitivity, 5+ outbreak cases (%)",
            "Specificity (%)": "Specificity, 5+ outbreak cases (%)",
            "AUROC (%)": "AUROC, 5+ outbreak cases (%)",
            "PPV (%)": "PPV, 5+ outbreak cases (%)",
            "NPV (%)": "NPV, 5+ outbreak cases (%)",
        }
    )

    classification10_wo_imports = performance_table(
        samples,
        true_outbreak,
        mode="classification",
        classification_threshold=10,
        true_labels_override=true_labels_override_10,
    )
    classification10_wo_imports = classification10_wo_imports.rename(
        {
            "Sensitivity (%)": "Sensitivity, 10+ outbreak cases (%)",
            "Specificity (%)": "Specificity, 10+ outbreak cases (%)",
            "AUROC (%)": "AUROC, 10+ outbreak cases (%)",
            "PPV (%)": "PPV, 10+ outbreak cases (%)",
            "NPV (%)": "NPV, 10+ outbreak cases (%)",
        }
    )

    classification20_wo_imports = performance_table(
        samples,
        true_outbreak,
        mode="classification",
        classification_threshold=20,
        true_labels_override=true_labels_override_20,
    )
    classification20_wo_imports = classification20_wo_imports.rename(
        {
            "Sensitivity (%)": "Sensitivity, 20+ outbreak cases (%)",
            "Specificity (%)": "Specificity, 20+ outbreak cases (%)",
            "AUROC (%)": "AUROC, 20+ outbreak cases (%)",
            "PPV (%)": "PPV, 20+ outbreak cases (%)",
            "NPV (%)": "NPV, 20+ outbreak cases (%)",
        }
    )

    # --- regression: with imports, >= 0 / >= 3 (already in df) / >= 10 ---
    regression_0_with_imports = performance_table(
        samples_unconditioned,
        true,
        mode="regression",
        regression_threshold=0,
    ).drop("Absolute difference (cases)")
    regression_0_with_imports = regression_0_with_imports.rename(
        {
            "RMSE (cases)": "RMSE, 0+ cases (cases)",
            "MAE (cases)": "MAE, 0+ cases (cases)",
            "Median fold error": "Median fold error, 0+ cases",
            "Spearman rho": "Spearman rho, 0+ cases",
            "R² (%)": "R², 0+ cases (%)",
        }
    )

    # >= 3 with imports is already present in df (from mode="selection"), skip

    regression_10_with_imports = performance_table(
        samples_unconditioned,
        true,
        mode="regression",
        regression_threshold=10,
    ).drop("Absolute difference (cases)")
    regression_10_with_imports = regression_10_with_imports.rename(
        {
            "RMSE (cases)": "RMSE, 10+ cases (cases)",
            "MAE (cases)": "MAE, 10+ cases (cases)",
            "Median fold error": "Median fold error, 10+ cases",
            "Spearman rho": "Spearman rho, 10+ cases",
            "R² (%)": "R², 10+ cases (%)",
        }
    )

    # --- regression: without imports, >= 0 / >= 3 / >= 10 ---
    regression_0_wo_imports = performance_table(
        samples,
        true_outbreak,
        mode="regression",
        regression_threshold=0,
    ).drop("Absolute difference (cases)")
    regression_0_wo_imports = regression_0_wo_imports.rename(
        {
            "RMSE (cases)": "RMSE, 0+ outbreak cases (cases)",
            "MAE (cases)": "MAE, 0+ outbreak cases (cases)",
            "Median fold error": "Median fold error, 0+ outbreak cases",
            "Spearman rho": "Spearman rho, 0+ outbreak cases",
            "R² (%)": "R², 0+ outbreak cases (%)",
        }
    )

    regression_3_wo_imports = performance_table(
        samples,
        true_outbreak,
        mode="regression",
        regression_threshold=3,
    ).drop("Absolute difference (cases)")
    regression_3_wo_imports = regression_3_wo_imports.rename(
        {
            "RMSE (cases)": "RMSE, 3+ outbreak cases (cases)",
            "MAE (cases)": "MAE, 3+ outbreak cases (cases)",
            "Median fold error": "Median fold error, 3+ outbreak cases",
            "Spearman rho": "Spearman rho, 3+ outbreak cases",
            "R² (%)": "R², 3+ outbreak cases (%)",
        }
    )

    regression_10_wo_imports = performance_table(
        samples,
        true_outbreak,
        mode="regression",
        regression_threshold=10,
    ).drop("Absolute difference (cases)")
    regression_10_wo_imports = regression_10_wo_imports.rename(
        {
            "RMSE (cases)": "RMSE, 10+ outbreak cases (cases)",
            "MAE (cases)": "MAE, 10+ outbreak cases (cases)",
            "Median fold error": "Median fold error, 10+ outbreak cases",
            "Spearman rho": "Spearman rho, 10+ outbreak cases",
            "R² (%)": "R², 10+ outbreak cases (%)",
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
            regression_0_with_imports,
            regression_10_with_imports,
            regression_0_wo_imports,
            regression_3_wo_imports,
            regression_10_wo_imports,
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
    # regression metrics: RMSE, MAE, rho at >= 0, >= 3, >= 10 case thresholds
    # we need to calculate classification metrics with and without importations

    # let's first read the real outbreak size data and associated import
    # probabilities
    analysis = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        schema_overrides={"fips": pl.String},
    )
    missing = (
        pl.read_csv(os.path.join(data_dir, "missing_counties.csv"))
        .filter(pl.col("year") >= 2021)
        .with_columns(cases_outbreak=pl.col("cases"))
    )
    analysis = pl.concat([analysis, missing], how="diagonal_relaxed")
    analysis = analysis.with_columns(
        state_county_year=pl.concat_str(
            [pl.col("state"), pl.col("county"), pl.col("year")], separator="_"
        )
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
    import_df = import_df.sort("type").unique(
        subset=["state", "county", "year"], keep="last"
    )

    validation_years = [2021, 2022, 2023, 2024, 2025]
    all_counties = []
    all_imports = []
    all_import_counts = []
    for year in validation_years:
        temp = pl.read_csv(
            os.path.join(output_dir, f"projection_dataset_{year}.csv"),
            schema_overrides={"fips": pl.String},
        )
        temp = temp.with_columns(validation_year=pl.lit(year))

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
            import_df.select(
                [
                    "county",
                    "state",
                    "year",
                    "importation_probability_samples",
                    "expected_importations_samples",
                ]
            ),
            on=["county", "state", "year"],
            how="left",
            maintain_order="left",
        )

        temp = temp.with_columns(
            state_county_year=pl.concat_str(
                [pl.col("state"), pl.col("county"), pl.col("validation_year")],
                separator="_",
            )
        )
        temp = temp.with_columns(
            type=pl.when(
                pl.col("state_county_year").is_in(
                    analysis["state_county_year"]
                )
            )
            .then(pl.lit("train"))
            .otherwise(pl.col("type"))
        )
        # year of these places also needs to be updated to be the current year
        temp = temp.with_columns(
            year=pl.when(pl.col("type") == "train")
            .then(pl.col("validation_year"))
            .otherwise(pl.col("year"))
        )
        # update cases outbreak as well
        temp = temp.with_columns(
            cases_outbreak=pl.when(pl.col("type") == "train")
            # get the particular cases outbreak for this state county year from the train data
            .then(
                pl.col("state_county_year").replace(
                    dict(
                        zip(
                            analysis["state_county_year"],
                            analysis["cases_outbreak"],
                        )
                    )
                )
            )
            .otherwise(pl.lit(0))
        )

        temp = temp.join(
            analysis[["year", "state", "county", "cases"]],
            on=["year", "state", "county"],
            how="left",
            maintain_order="left",
        )

        temp = temp.with_columns(
            cases=pl.when(pl.col("type") == "projection")
            .then(pl.lit(0))
            .otherwise(pl.col("cases"))
        )

        temp_import = jnp.array(
            np.vstack(temp["importation_probability_samples"].to_numpy())  # type: ignore
        ).T
        all_imports.append(temp_import)

        temp_import_counts = jnp.array(
            np.vstack(temp["expected_importations_samples"].to_numpy())  # type: ignore
        ).T
        all_import_counts.append(temp_import_counts)

        all_counties.append(temp)

    # Create lookup dictionaries for multi-year outbreak detection
    # For each (county, state, validation_year), check if outbreak occurred in any future year
    county_outbreak_lookup_5 = {}
    county_outbreak_lookup_10 = {}
    county_outbreak_lookup_20 = {}

    for idx, validation_year in enumerate(validation_years):
        counties_this_year = all_counties[idx]

        for i in range(len(counties_this_year)):
            county = counties_this_year["county"][i]
            state = counties_this_year["state"][i]
            key = (county, state, validation_year)

            # Initialize as False
            county_outbreak_lookup_5[key] = False
            county_outbreak_lookup_10[key] = False
            county_outbreak_lookup_20[key] = False

            # Check all future years (including current year)
            for future_idx in range(idx, len(validation_years)):
                future_counties = all_counties[future_idx]

                # Find matching county in future year
                future_match = future_counties.filter(
                    (pl.col("county") == county) & (pl.col("state") == state)
                )

                if len(future_match) > 0:
                    cases = future_match["cases"][0]

                    if cases >= 5:
                        county_outbreak_lookup_5[key] = True
                    if cases >= 10:
                        county_outbreak_lookup_10[key] = True
                    if cases >= 20:
                        county_outbreak_lookup_20[key] = True

    car_scale_factors = {}

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

                    if validation_year != 2025:
                        cases = all_counties[idx]["cases"].sum()
                    else:
                        cases = (
                            all_counties[idx]
                            .filter(pl.col("state") != "South Carolina")[
                                "cases"
                            ]
                            .sum()
                            + 333
                        )  # Add 2025 case count for SC

                    with open(
                        os.path.join(
                            output_dir,
                            f"samples_partial_pooled_poisson_validation_testing_{years}_pool={pool}_CAR=True_{model_variant}.pkl",
                        ),
                        "rb",
                    ) as f:
                        samples = pickle.load(f)

                    samples_with_imports = (
                        samples * all_imports[idx] + all_import_counts[idx]
                    )
                    samples_with_imports = jnp.sum(
                        samples_with_imports, axis=1
                    )
                    predicted_median = jnp.median(samples_with_imports)

                    observed.append(cases)
                    predicted.append(predicted_median)

                scale = np.mean(np.array(predicted) / np.array(observed))
                car_scale_factors[(start_year, pool, model_variant)] = scale

    model_performance = []
    true_geq_10 = []
    for start_year in tqdm([2011, 2019, -2], desc="Training year variants"):
        for idx, validation_year in enumerate(
            tqdm([2021, 2022, 2023, 2024, 2025], desc="Evaluation year")
        ):
            if start_year < 0:
                years = list(
                    range(validation_year + start_year, validation_year)
                )
            else:
                years = list(range(start_year, validation_year))

            true = all_counties[idx]["cases"].to_numpy()
            true_outbreak = all_counties[idx]["cases_outbreak"].to_numpy()

            # Create override labels for this year
            counties_df = all_counties[idx]
            true_labels_5 = np.zeros(len(true), dtype=int)
            true_labels_10 = np.zeros(len(true), dtype=int)
            true_labels_20 = np.zeros(len(true), dtype=int)

            for i in range(len(counties_df)):
                county = counties_df["county"][i]
                state = counties_df["state"][i]
                key = (county, state, validation_year)

                if county_outbreak_lookup_5.get(key, False):
                    true_labels_5[i] = 1
                if county_outbreak_lookup_10.get(key, False):
                    true_labels_10[i] = 1
                if county_outbreak_lookup_20.get(key, False):
                    true_labels_20[i] = 1

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
                            samples = pickle.load(f)

                        raw_samples_unconditioned = (
                            samples * all_imports[idx] + all_import_counts[idx]
                        )  # shape: [n_draws, n_locations]

                        if car:
                            scale = car_scale_factors[
                                (start_year, pool, model_variant)
                            ]
                            raw_samples_unconditioned = (
                                raw_samples_unconditioned / scale
                            )
                            samples = samples / scale

                        samples_unconditioned = jnp.mean(
                            raw_samples_unconditioned, axis=0
                        )
                        samples = jnp.mean(samples, axis=0)

                        df = performance_table(
                            samples_unconditioned,
                            true,
                            mode="selection",
                            raw_samples=raw_samples_unconditioned,
                            true_labels_override=true_labels_10,
                        )

                        cov = coverage_table(raw_samples_unconditioned, true)
                        df = pl.concat([df, cov], how="horizontal")

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

                        df = other_tables(
                            df,
                            samples_unconditioned,
                            samples,
                            true,
                            true_outbreak,
                            true_labels_override_5=true_labels_5,
                            true_labels_override_10=true_labels_10,
                            true_labels_override_20=true_labels_20,
                        )

                        model_performance.append(df)

            with open(
                os.path.join(
                    output_dir,
                    f"predictive_final_outbreak_size_predictions_{years}.pkl",
                ),
                "rb",
            ) as f:
                samples = pickle.load(f)["cases"]

            raw_samples_unconditioned = (
                samples * all_imports[idx] + all_import_counts[idx]
            )  # shape: [n_draws, n_locations]

            samples_unconditioned = jnp.mean(raw_samples_unconditioned, axis=0)
            samples = jnp.mean(samples, axis=0)

            df = performance_table(
                samples_unconditioned,
                true,
                mode="selection",
                raw_samples=raw_samples_unconditioned,
                true_labels_override=true_labels_10,
            )

            cov = coverage_table(raw_samples_unconditioned, true)
            df = pl.concat([df, cov], how="horizontal")

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
            df = other_tables(
                df,
                samples_unconditioned,
                samples,
                true,
                true_outbreak,
                true_labels_override_5=true_labels_5,
                true_labels_override_10=true_labels_10,
                true_labels_override_20=true_labels_20,
            )
            true_geq_10.append(((true >= 10).sum(), len(true)))
            model_performance.append(df)

            # --- sensitivity analysis: multiplicative imports only ---
            with open(
                os.path.join(
                    output_dir,
                    f"predictive_final_outbreak_size_predictions_{years}.pkl",
                ),
                "rb",
            ) as f:
                samples_multi = pickle.load(f)["cases"]

            raw_samples_unconditioned_multi = (
                samples_multi * all_import_counts[idx]
            )  # shape: [n_draws, n_locations]

            samples_unconditioned_multi = jnp.mean(
                raw_samples_unconditioned_multi, axis=0
            )
            samples_multi_mean = jnp.mean(samples_multi, axis=0)

            df_multi = performance_table(
                samples_unconditioned_multi,
                true,
                mode="selection",
                raw_samples=raw_samples_unconditioned_multi,
                true_labels_override=true_labels_10,
            )

            cov_multi = coverage_table(raw_samples_unconditioned_multi, true)
            df_multi = pl.concat([df_multi, cov_multi], how="horizontal")

            model_name_multi = (
                "Mechanistic with multi-chain outbreaks: MMR coverage"
            )
            if start_year == -2:
                model_name_multi += ", trained on last two years"
            elif start_year == 2011:
                model_name_multi += ", trained since 2011"
            elif start_year == 2019:
                model_name_multi += ", trained since 2019"
            else:
                assert False, "Invalid training start year"

            df_multi = df_multi.with_columns(
                pl.lit(validation_year).alias("Evaluation year"),
                Model=pl.lit(model_name_multi),
            )
            df_multi = other_tables(
                df_multi,
                samples_unconditioned_multi,
                samples_multi_mean,
                true,
                true_outbreak,
                true_labels_override_5=true_labels_5,
                true_labels_override_10=true_labels_10,
                true_labels_override_20=true_labels_20,
            )
            model_performance.append(df_multi)

    # --- supplemental table: yearly R² and sample sizes ---
    yearly_r2_and_counts = []

    for start_year in tqdm([2011, 2019, -2], desc="Training year variants"):
        for idx, validation_year in enumerate(
            tqdm(
                [2021, 2022, 2023, 2024, 2025],
                desc="Evaluation year",
                leave=False,
            )
        ):
            if start_year < 0:
                years = list(
                    range(validation_year + start_year, validation_year)
                )
            else:
                years = list(range(start_year, validation_year))

            true = all_counties[idx]["cases"].to_numpy()
            true_outbreak = all_counties[idx]["cases_outbreak"].to_numpy()

            # Count county-years with 3+ cases
            true = np.array([int(t) for t in true])
            mask_with_imports = true >= 3
            true_outbreak = np.array([int(t) for t in true_outbreak])
            mask_wo_imports = true_outbreak >= 3
            count_with_imports = int(mask_with_imports.sum())
            count_wo_imports = int(mask_wo_imports.sum())

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
                            samples = pickle.load(f)

                        raw_samples_unconditioned = (
                            samples * all_imports[idx] + all_import_counts[idx]
                        )

                        if car:
                            scale = car_scale_factors[
                                (start_year, pool, model_variant)
                            ]
                            raw_samples_unconditioned = (
                                raw_samples_unconditioned / scale
                            )
                            samples = samples / scale

                        samples_unconditioned = jnp.mean(
                            raw_samples_unconditioned, axis=0
                        )
                        samples_mean = jnp.mean(samples, axis=0)

                        # Calculate R² for 3+ cases (with imports)
                        if count_with_imports >= 2:
                            true_subset_with = true[mask_with_imports]
                            samples_subset_with = samples_unconditioned[
                                mask_with_imports
                            ]
                            pearson_r_with = stats.pearsonr(
                                true_subset_with, samples_subset_with
                            )[0]
                            r2_with = pearson_r_with**2 * 100  # type: ignore
                        else:
                            r2_with = float("nan")

                        # Calculate R² for 3+ outbreak cases (without imports)
                        if count_wo_imports >= 2:
                            true_subset_wo = true_outbreak[mask_wo_imports]
                            samples_subset_wo = samples_mean[mask_wo_imports]
                            pearson_r_wo = stats.pearsonr(
                                true_subset_wo, samples_subset_wo
                            )[0]
                            r2_wo = pearson_r_wo**2 * 100  # type: ignore
                        else:
                            r2_wo = float("nan")

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

                        yearly_r2_and_counts.append(
                            {
                                "Model": model_name,
                                "Evaluation year": validation_year,
                                "County-years with 3+ cases": count_with_imports,
                                "R², 3+ cases (%)": r2_with,
                                "County-years with 3+ outbreak cases": count_wo_imports,
                                "R², 3+ outbreak cases (%)": r2_wo,
                            }
                        )

            # Mechanistic model
            with open(
                os.path.join(
                    output_dir,
                    f"predictive_final_outbreak_size_predictions_{years}.pkl",
                ),
                "rb",
            ) as f:
                samples = pickle.load(f)["cases"]

            raw_samples_unconditioned = (
                samples * all_imports[idx] + all_import_counts[idx]
            )

            samples_unconditioned = jnp.mean(raw_samples_unconditioned, axis=0)
            samples_mean = jnp.mean(samples, axis=0)

            # Calculate R² for 3+ cases (with imports)
            if count_with_imports >= 2:
                true_subset_with = true[mask_with_imports]
                samples_subset_with = samples_unconditioned[mask_with_imports]
                pearson_r_with = stats.pearsonr(
                    true_subset_with, samples_subset_with
                )[0]
                r2_with = pearson_r_with**2 * 100  # type: ignore
            else:
                r2_with = float("nan")

            # Calculate R² for 3+ outbreak cases (without imports)
            if count_wo_imports >= 2:
                true_subset_wo = true_outbreak[mask_wo_imports]
                samples_subset_wo = samples_mean[mask_wo_imports]
                pearson_r_wo = stats.pearsonr(
                    true_subset_wo, samples_subset_wo
                )[0]
                r2_wo = pearson_r_wo**2 * 100  # type: ignore
            else:
                r2_wo = float("nan")

            model_name = "Mechanistic: MMR coverage"
            if start_year == -2:
                model_name += ", trained on last two years"
            elif start_year == 2011:
                model_name += ", trained since 2011"
            elif start_year == 2019:
                model_name += ", trained since 2019"

            yearly_r2_and_counts.append(
                {
                    "Model": model_name,
                    "Evaluation year": validation_year,
                    "County-years with 3+ cases": count_with_imports,
                    "R², 3+ cases (%)": r2_with,
                    "County-years with 3+ outbreak cases": count_wo_imports,
                    "R², 3+ outbreak cases (%)": r2_wo,
                }
            )

            # --- sensitivity analysis: multiplicative imports only ---
            with open(
                os.path.join(
                    output_dir,
                    f"predictive_final_outbreak_size_predictions_{years}.pkl",
                ),
                "rb",
            ) as f:
                samples_multi = pickle.load(f)["cases"]

            raw_samples_unconditioned_multi = (
                samples_multi * all_import_counts[idx]
            )

            samples_unconditioned_multi = jnp.mean(
                raw_samples_unconditioned_multi, axis=0
            )
            samples_multi_mean = jnp.mean(samples_multi, axis=0)

            # Calculate R² for 3+ cases (with imports)
            if count_with_imports >= 2:
                true_subset_with = true[mask_with_imports]
                samples_subset_with = samples_unconditioned_multi[
                    mask_with_imports
                ]
                pearson_r_with = stats.pearsonr(
                    true_subset_with, samples_subset_with
                )[0]
                r2_with = pearson_r_with**2 * 100  # type: ignore
            else:
                r2_with = float("nan")

            # Calculate R² for 3+ outbreak cases (without imports)
            if count_wo_imports >= 2:
                true_subset_wo = true_outbreak[mask_wo_imports]
                samples_subset_wo = samples_multi_mean[mask_wo_imports]
                pearson_r_wo = stats.pearsonr(
                    true_subset_wo, samples_subset_wo
                )[0]
                r2_wo = pearson_r_wo**2 * 100  # type: ignore
            else:
                r2_wo = float("nan")

            model_name_multi = (
                "Mechanistic with multi-chain outbreaks: MMR coverage"
            )
            if start_year == -2:
                model_name_multi += ", trained on last two years"
            elif start_year == 2011:
                model_name_multi += ", trained since 2011"
            elif start_year == 2019:
                model_name_multi += ", trained since 2019"

            yearly_r2_and_counts.append(
                {
                    "Model": model_name_multi,
                    "Evaluation year": validation_year,
                    "County-years with 3+ cases": count_with_imports,
                    "R², 3+ cases (%)": r2_with,
                    "County-years with 3+ outbreak cases": count_wo_imports,
                    "R², 3+ outbreak cases (%)": r2_wo,
                }
            )

    # Create DataFrame
    yearly_r2_table = pl.DataFrame(yearly_r2_and_counts)

    # Round R² values
    yearly_r2_table = yearly_r2_table.with_columns(
        pl.col("R², 3+ cases (%)").round(1),
        pl.col("R², 3+ outbreak cases (%)").round(1),
    )

    # Sort by model and year
    yearly_r2_table = yearly_r2_table.sort(["Model", "Evaluation year"])

    # Write to CSV
    yearly_r2_table.write_csv(
        os.path.join(
            fig_dir, "table_supplemental_yearly_r2_and_sample_sizes.csv"
        )
    )

    # clean up and export
    model_performance = pl.concat(model_performance)

    all_metric_columns = [
        col
        for col in model_performance.columns
        if col != "Model" and col != "Evaluation year"
    ]

    model_performance = aggregate(model_performance, all_metric_columns)
    model_performance = model_performance.filter(
        pl.col("Evaluation year") == "2021-2025"
    ).drop("Evaluation year")

    # --- model selection table ---
    model_selection_no_interval = [
        "Absolute difference (cases)",
        "Coverage (%)",
    ]

    model_selection = model_performance.select(
        [
            "Model",
            "Absolute difference (cases)",
            "Coverage (%)",
            *[
                f"{col}{suffix}"
                for col in ["AUROC, 10+ cases (%)", "MAE, 3+ cases (cases)"]
                for suffix in ["", "_lower", "_upper"]
            ],
        ]
    )
    model_selection = model_selection.sort(
        by=["Coverage (%)", "Absolute difference (cases)"],
        descending=[True, False],
    )
    model_selection = beautify(
        model_selection,
        [
            "AUROC, 10+ cases (%)",
            "MAE, 3+ cases (cases)",
            "Absolute difference (cases)",
            "Coverage (%)",
        ],
        no_interval_columns=model_selection_no_interval,
    )
    model_selection.write_csv(
        os.path.join(
            fig_dir, "supplemental_data_s1_model_selection_metrics.csv"
        )
    )

    # --- pick main models ---
    main_models = [
        "Statistical: All covariates + spatial + state-specific vaccine coefficients, trained since 2019",
        find_best_model(model_performance, subset="Mechanistic: MMR"),
        find_best_model(
            model_performance, subset="Statistical", exclude="spatial"
        ),
        # "Statistical: MMR coverage, trained on last two years",
    ]

    # shared column names used for both with- and without-imports rows
    shared_columns = [
        "Sensitivity, 10+ (%)",
        "Specificity, 10+ (%)",
        "PPV, 10+ (%)",
        "NPV, 10+ (%)",
        "RMSE, 3+ cases (cases)",
        "R², 3+ cases (%)",
    ]

    # --- with importations row ---
    with_imports_columns = [
        "Sensitivity, 10+ cases (%)",
        "Specificity, 10+ cases (%)",
        "PPV, 10+ cases (%)",
        "NPV, 10+ cases (%)",
        "RMSE, 3+ cases (cases)",
        "R², 3+ cases (%)",
    ]

    with_imports = model_performance.select(
        ["Model"]
        + [
            f"{col}{suffix}"
            for col in with_imports_columns
            for suffix in ["", "_lower", "_upper"]
        ]
    )
    with_imports = with_imports.filter(pl.col("Model").is_in(main_models))

    rename_map = {
        orig: shared
        for orig, shared in zip(with_imports_columns, shared_columns)
        if orig != shared
    }
    rename_map.update(
        {
            f"{orig}{suf}": f"{shared}{suf}"
            for orig, shared in zip(with_imports_columns, shared_columns)
            for suf in ["_lower", "_upper"]
            if orig != shared
        }
    )
    with_imports = with_imports.rename(rename_map)
    with_imports = with_imports.with_columns(
        pl.lit("All cases (including imports)").alias("Condition")
    )

    # --- without importations (outbreak only) row ---
    wo_imports_columns = [
        "Sensitivity, 10+ outbreak cases (%)",
        "Specificity, 10+ outbreak cases (%)",
        "PPV, 10+ outbreak cases (%)",
        "NPV, 10+ outbreak cases (%)",
        "RMSE, 3+ outbreak cases (cases)",
        "R², 3+ outbreak cases (%)",
    ]

    wo_imports = model_performance.select(
        ["Model"]
        + [
            f"{col}{suffix}"
            for col in wo_imports_columns
            for suffix in ["", "_lower", "_upper"]
        ]
    )
    wo_imports = wo_imports.filter(pl.col("Model").is_in(main_models))

    rename_map = {
        orig: shared
        for orig, shared in zip(wo_imports_columns, shared_columns)
        if orig != shared
    }
    rename_map.update(
        {
            f"{orig}{suf}": f"{shared}{suf}"
            for orig, shared in zip(wo_imports_columns, shared_columns)
            for suf in ["_lower", "_upper"]
            if orig != shared
        }
    )
    wo_imports = wo_imports.rename(rename_map)
    wo_imports = wo_imports.with_columns(
        pl.lit("Outbreak cases").alias("Condition")
    )

    # interleave: Model A (with imports), Model A (outbreak), Model B ...
    main_table = pl.concat([with_imports, wo_imports], how="vertical")

    model_order = {m: i for i, m in enumerate(main_models)}
    condition_order = {"All cases (including imports)": 0, "Outbreak cases": 1}
    main_table = (
        main_table.with_columns(
            pl.col("Model")
            .replace_strict(model_order, return_dtype=pl.Int32)
            .alias("_model_order"),
            pl.col("Condition")
            .replace_strict(condition_order, return_dtype=pl.Int32)
            .alias("_cond_order"),
        )
        .sort(["_model_order", "_cond_order"])
        .drop(["_model_order", "_cond_order"])
    )

    main_table = main_table.select(
        ["Model", "Condition"]
        + [
            f"{col}{suffix}"
            for col in shared_columns
            for suffix in ["", "_lower", "_upper"]
        ]
    )

    main_table = beautify(main_table, shared_columns)
    main_table.write_csv(
        os.path.join(fig_dir, "table_validation_performance_metrics.csv")
    )

    # --- supplemental classification table: with imports ---
    classification_with_imports_columns = [
        "Sensitivity, 5+ cases (%)",
        "Specificity, 5+ cases (%)",
        "AUROC, 5+ cases (%)",
        "PPV, 5+ cases (%)",
        "NPV, 5+ cases (%)",
        "Sensitivity, 10+ cases (%)",
        "Specificity, 10+ cases (%)",
        "AUROC, 10+ cases (%)",
        "PPV, 10+ cases (%)",
        "NPV, 10+ cases (%)",
        "Sensitivity, 20+ cases (%)",
        "Specificity, 20+ cases (%)",
        "AUROC, 20+ cases (%)",
        "PPV, 20+ cases (%)",
        "NPV, 20+ cases (%)",
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
    classification_with_imports = beautify(
        classification_with_imports, classification_with_imports_columns
    )
    classification_with_imports.write_csv(
        os.path.join(
            fig_dir,
            "supplemental_data_s2a_classification_metrics_with_importations.csv",
        )
    )

    # --- supplemental classification table: without imports ---
    classification_wo_imports_columns = [
        "Sensitivity, 5+ outbreak cases (%)",
        "Specificity, 5+ outbreak cases (%)",
        "AUROC, 5+ outbreak cases (%)",
        "PPV, 5+ outbreak cases (%)",
        "NPV, 5+ outbreak cases (%)",
        "Sensitivity, 10+ outbreak cases (%)",
        "Specificity, 10+ outbreak cases (%)",
        "AUROC, 10+ outbreak cases (%)",
        "PPV, 10+ outbreak cases (%)",
        "NPV, 10+ outbreak cases (%)",
        "Sensitivity, 20+ outbreak cases (%)",
        "Specificity, 20+ outbreak cases (%)",
        "AUROC, 20+ outbreak cases (%)",
        "PPV, 20+ outbreak cases (%)",
        "NPV, 20+ outbreak cases (%)",
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
    classification_wo_imports = beautify(
        classification_wo_imports, classification_wo_imports_columns
    )
    classification_wo_imports.write_csv(
        os.path.join(
            fig_dir,
            "supplemental_data_s2b_classification_metrics_without_importations.csv",
        )
    )

    # --- supplemental regression table: with imports ---
    regression_with_imports_columns = [
        "RMSE, 0+ cases (cases)",
        "MAE, 0+ cases (cases)",
        "Spearman rho, 0+ cases",
        "R², 0+ cases (%)",
        "RMSE, 3+ cases (cases)",
        "MAE, 3+ cases (cases)",
        "Median fold error, 3+ cases",
        "Spearman rho, 3+ cases",
        "R², 3+ cases (%)",
        "RMSE, 10+ cases (cases)",
        "MAE, 10+ cases (cases)",
        "Median fold error, 10+ cases",
        "Spearman rho, 10+ cases",
        "R², 10+ cases (%)",
    ]
    regression_with_imports = model_performance.select(
        [
            *["Model"],
            *[
                f"{col}{suffix}"
                for col in regression_with_imports_columns
                for suffix in ["", "_lower", "_upper"]
            ],
        ]
    )
    regression_with_imports = beautify(
        regression_with_imports, regression_with_imports_columns
    )
    regression_with_imports.write_csv(
        os.path.join(
            fig_dir,
            "supplemental_data_s3a_regression_metrics_with_importations.csv",
        )
    )

    # --- supplemental regression table: without imports ---
    regression_wo_imports_columns = [
        "RMSE, 0+ outbreak cases (cases)",
        "MAE, 0+ outbreak cases (cases)",
        "R², 0+ outbreak cases (%)",
        "Spearman rho, 0+ outbreak cases",
        "RMSE, 3+ outbreak cases (cases)",
        "MAE, 3+ outbreak cases (cases)",
        "Median fold error, 3+ outbreak cases",
        "Spearman rho, 3+ outbreak cases",
        "R², 3+ outbreak cases (%)",
        "RMSE, 10+ outbreak cases (cases)",
        "MAE, 10+ outbreak cases (cases)",
        "Median fold error, 10+ outbreak cases",
        "Spearman rho, 10+ outbreak cases",
        "R², 10+ outbreak cases (%)",
    ]
    regression_wo_imports = model_performance.select(
        [
            *["Model"],
            *[
                f"{col}{suffix}"
                for col in regression_wo_imports_columns
                for suffix in ["", "_lower", "_upper"]
            ],
        ]
    )
    regression_wo_imports = beautify(
        regression_wo_imports, regression_wo_imports_columns
    )
    regression_wo_imports.write_csv(
        os.path.join(
            fig_dir,
            "supplemental_data_s3b_regression_metrics_without_importations.csv",
        )
    )

    print(true_geq_10)
    r = [a / b if b != 0 else 0 for a, b in true_geq_10]
    print(r)
    print(np.mean(r))
    print(min(r))
    print(max(r))
    total_true_geq_10 = sum(a for a, b in true_geq_10)
    total_count = sum(b for a, b in true_geq_10)
    overall_proportion = (
        total_true_geq_10 / total_count if total_count != 0 else 0
    )
    print(overall_proportion)
