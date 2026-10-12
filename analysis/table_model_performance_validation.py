import gc
import pickle

import numpy as np
import polars as pl
from numpyro import enable_x64

enable_x64()
import os

import jax.numpy as jnp
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from tqdm.auto import tqdm

from outbreak_size import scoring

data_dir = "data"
output_dir = "outputs"
fig_dir = "figures"

GLOBALS_PRIOR_SD = 1


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

        if np.isnan(fpr[0]) or np.isnan(tpr[0]):
            auprc = np.nan
        else:
            auprc = average_precision_score(y_true, samples)

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
            thresholds_roc[idx] if not np.isnan(sensitivity) else np.nan  # type: ignore
        )

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
            "AUPRC (%)": auprc * 100,
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


# per-year mean absolute error column -> the relative MAE column built from it
RELATIVE_MAE_COLUMNS = {
    "Mean absolute error, 0+ cases (cases)": "Relative MAE, 0+ cases",
    "Mean absolute error, 0+ outbreak cases (cases)": "Relative MAE, 0+ outbreak cases",
}


def add_relative_mae(df):
    """
    Relative MAE: each model's per-year mean absolute error divided by that of
    the mechanistic MMR-coverage model trained over the same window, in the
    same evaluation year (< 1 is better than the mechanistic model). Models
    with no matching mechanistic model (the naive baselines) get null.
    """
    df = df.with_columns(
        _window=pl.col("Model").str.extract(r"(, trained .*)$", 1)
    )
    reference = df.filter(
        pl.col("Model")
        == pl.lit("Mechanistic: MMR coverage") + pl.col("_window")
    ).select(
        "_window",
        "Evaluation year",
        *[pl.col(col).alias(f"_ref {col}") for col in RELATIVE_MAE_COLUMNS],
    )
    df = df.join(
        reference,
        on=["_window", "Evaluation year"],
        how="left",
        maintain_order="left",
    )
    df = df.with_columns(
        [
            (pl.col(col) / pl.col(f"_ref {col}")).alias(relative_col)
            for col, relative_col in RELATIVE_MAE_COLUMNS.items()
        ]
    )
    return df.drop("_window", *[f"_ref {col}" for col in RELATIVE_MAE_COLUMNS])


def aggregate(df, metric_columns):
    # create a weighted average value across all years (the geometric mean
    # for the relative MAE, which is a ratio)
    df_agg = (
        df.group_by(["Model"], maintain_order=True)
        .agg(
            [
                *[
                    (
                        pl.col(col)
                        .filter(pl.col(col).is_not_nan())
                        .log()
                        .mean()
                        .exp()
                        if col in RELATIVE_MAE_COLUMNS.values()
                        else (
                            pl.col(col).filter(pl.col(col).is_not_nan())
                        ).mean()
                    )
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


def beautify(
    df, column_stub, no_interval_columns=None, decimals_by_column=None
):
    """
    Format columns as "value [lower, upper]".

    Parameters
    ----------
    no_interval_columns : list of str, optional
        Columns that should be rounded and displayed as a plain number,
        with no [lower, upper] bracketing (e.g. "Absolute difference (cases)"
        and "Coverage (%)" in the model selection table).
    decimals_by_column : dict, optional
        Decimal places for specific interval columns (default 1).
    """
    if no_interval_columns is None:
        no_interval_columns = []
    if decimals_by_column is None:
        decimals_by_column = {}

    interval_columns = [c for c in column_stub if c not in no_interval_columns]

    df = df.with_columns(
        [
            *[
                pl.col(col).round(decimals_by_column.get(col, 1))
                for col in interval_columns
            ],
            *[
                pl.col(f"{col}_lower").round(decimals_by_column.get(col, 1))
                for col in interval_columns
            ],
            *[
                pl.col(f"{col}_upper").round(decimals_by_column.get(col, 1))
                for col in interval_columns
            ],
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
    raw_samples_unconditioned=None,  # predictive draws [n_draws, n_locations] with imports
    raw_samples=None,  # predictive draws [n_draws, n_locations] without imports
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
            "AUPRC (%)": "AUPRC, 10+ cases (%)",
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
            "AUPRC (%)": "AUPRC, 5+ cases (%)",
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
            "AUPRC (%)": "AUPRC, 20+ cases (%)",
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
            "AUPRC (%)": "AUPRC, 5+ outbreak cases (%)",
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
            "AUPRC (%)": "AUPRC, 10+ outbreak cases (%)",
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
            "AUPRC (%)": "AUPRC, 20+ outbreak cases (%)",
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

    # --- 0+ WIS and CRPS (need the predictive draws; NaN for models without
    # draws) and mean absolute error of the draw-mean forecast (which the
    # relative MAE is built from, see add_relative_mae) ---
    def draw_scores_and_mean_absolute_error(draws, forecast, truth, label):
        truth = np.asarray(truth)
        wis = crps = float("nan")
        if draws is not None:
            draws = np.asarray(draws)
            wis = float(np.mean(scoring.weighted_interval_score(truth, draws)))
            crps = float(np.mean(scoring.crps(truth, draws)))
        return pl.DataFrame(
            {
                f"WIS, 0+ {label}cases (cases)": [wis],
                f"CRPS, 0+ {label}cases (cases)": [crps],
                f"Mean absolute error, 0+ {label}cases (cases)": [
                    float(np.mean(np.abs(np.asarray(forecast) - truth)))
                ],
            }
        )

    scores_with_imports = draw_scores_and_mean_absolute_error(
        raw_samples_unconditioned, samples_unconditioned, true, ""
    )
    scores_wo_imports = draw_scores_and_mean_absolute_error(
        raw_samples, samples, true_outbreak, "outbreak "
    )

    # bring this all together in one table
    df = pl.concat(
        [
            df,
            scores_with_imports,
            scores_wo_imports,
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


def at_risk_flags(score, y_true):
    """
    Boolean "at-risk" call for each county, at the same closest-corner ROC
    operating point used inside performance_table to derive sensitivity/
    specificity. Used to compare which counties each model actually flags.

    Parameters
    ----------
    score : array-like
        Per-county predicted score (posterior mean count, or a naive score).
    y_true : array-like of 0/1
        Binary outbreak label.

    Returns
    -------
    np.ndarray of bool
    """
    fpr, tpr, thresholds_roc = roc_curve(y_true, score)
    if np.isnan(fpr[0]) or np.isnan(tpr[0]):
        # degenerate (single-class) year: no operating point, flag nobody
        return np.zeros(len(score), dtype=bool)
    idx = np.argmin(np.sqrt(fpr**2 + (1 - tpr) ** 2))
    threshold = thresholds_roc[idx]
    return np.asarray(score) >= threshold


AGREEMENT_METRIC_NAMES = [
    "AUROC",
    "AUPRC",
    "Sensitivity",
    "Specificity",
    "PPV",
    "NPV",
]


def _pooled_classification_metrics(scores, roc_label, ppv_label, auprc_label):
    """
    Pooled (micro-averaged) AUROC/AUPRC/Sensitivity/Specificity/PPV/NPV over
    all rows in `scores`, using the same closest-corner ROC operating point
    and PPV/NPV label logic as `performance_table` (mode="classification").

    `roc_label` (single-year, 10+) is used for AUROC/Sens/Spec; `ppv_label`
    (multi-year forward-looking, 10+) is used for PPV/NPV, matching
    `true_labels_override` in `performance_table`; `auprc_label` (multi-year,
    10+, currently identical to `ppv_label`) is used for AUPRC.
    """
    scores = np.asarray(scores, dtype=float)
    roc_label = np.asarray(roc_label).astype(int)
    ppv_label = np.asarray(ppv_label).astype(int)
    auprc_label = np.asarray(auprc_label).astype(int)

    if len(np.unique(roc_label)) < 2:
        return {k: np.nan for k in AGREEMENT_METRIC_NAMES}

    fpr, tpr, thresholds_roc = roc_curve(roc_label, scores)
    auroc = roc_auc_score(roc_label, scores)
    auprc = average_precision_score(
        auprc_label, scores
    )  # multi-year 10+ label (AUPRC only)

    idx = np.argmin(np.sqrt(fpr**2 + (1 - tpr) ** 2))
    sensitivity = tpr[idx]
    specificity = 1.0 - fpr[idx]
    threshold = thresholds_roc[idx]

    y_pred = (scores >= threshold).astype(int)
    tp = np.sum((ppv_label == 1) & (y_pred == 1))
    fp = np.sum((ppv_label == 0) & (y_pred == 1))
    tn = np.sum((ppv_label == 0) & (y_pred == 0))
    fn = np.sum((ppv_label == 1) & (y_pred == 0))
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    return {
        "AUROC": auroc * 100,
        "AUPRC": auprc * 100,
        "Sensitivity": sensitivity * 100,
        "Specificity": specificity * 100,
        "PPV": ppv * 100,
        "NPV": npv * 100,
    }


def agreement_table_bootstrap(
    pooled_capture,
    models,
    condition_key,
    threshold,
    n_boot,
    seed,
    cluster=True,
):
    """
    Bootstrap for the agreement table's classification metrics. Concatenates
    each model's captured per-validation-year arrays into one pooled
    county-year array, then resamples with replacement -- if `cluster` (the
    default), whole *counties* are resampled together (keeping each county's
    full multi-year history intact, so every model is evaluated on the
    identical resample each iteration and pairwise differences share the
    resampling noise); if not, individual county-year rows are resampled
    independently (treats each row as iid, matching the model's own
    per-county-year likelihood).

    Returns
    -------
    point : {model: {metric: value}}          pooled (unresampled) estimate
    ci    : {model: {metric: (lower, upper)}} 2.5/97.5 percentiles
    diffs : {(model_a, model_b): {metric: {"mean_diff", "lower", "upper", "p_gt_0"}}}
        for every pair where model_a or model_b is `models[0]` (the naive
        reference model is expected to be first in `models`)
    """
    pooled = {}
    for m in models:
        entries = pooled_capture[(m, condition_key, threshold)]
        pooled[m] = {
            "score": np.concatenate([e["score"] for e in entries]),
            "roc_label": np.concatenate([e["roc_label"] for e in entries]),
            "ppv_label": np.concatenate([e["ppv_label"] for e in entries]),
            "auprc_label": np.concatenate([e["auprc_label"] for e in entries]),
            "cid": np.concatenate([e["cid"] for e in entries]),
        }

    ref_cid = pooled[models[0]]["cid"]
    for m in models:
        assert np.array_equal(pooled[m]["cid"], ref_cid), (
            f"pooled county-year row order mismatch for {m!r} vs "
            f"{models[0]!r} in condition {condition_key!r}; capture "
            "sites must run over the same (start_year, validation_year) grid"
        )

    point = {
        m: _pooled_classification_metrics(
            pooled[m]["score"],
            pooled[m]["roc_label"],
            pooled[m]["ppv_label"],
            pooled[m]["auprc_label"],
        )
        for m in models
    }

    if cluster:
        unique_counties, inverse = np.unique(ref_cid, return_inverse=True)
        county_to_indices = [
            np.where(inverse == i)[0] for i in range(len(unique_counties))
        ]
        n_units = len(unique_counties)
    else:
        n_units = len(ref_cid)

    rng = np.random.default_rng(seed)
    boot_values = {m: {k: [] for k in AGREEMENT_METRIC_NAMES} for m in models}

    for _ in range(n_boot):
        sampled = rng.integers(0, n_units, size=n_units)
        resample_idx = (
            np.concatenate([county_to_indices[i] for i in sampled])  # type: ignore
            if cluster
            else sampled
        )
        for m in models:
            res = _pooled_classification_metrics(
                pooled[m]["score"][resample_idx],
                pooled[m]["roc_label"][resample_idx],
                pooled[m]["ppv_label"][resample_idx],
                pooled[m]["auprc_label"][resample_idx],
            )
            for k in AGREEMENT_METRIC_NAMES:
                boot_values[m][k].append(res[k])

    ci = {
        m: {
            k: (
                float(np.nanpercentile(boot_values[m][k], 2.5)),
                float(np.nanpercentile(boot_values[m][k], 97.5)),
            )
            for k in AGREEMENT_METRIC_NAMES
        }
        for m in models
    }

    ref_model = models[0]
    diffs = {}
    for m in models:
        if m == ref_model:
            continue
        diffs[(ref_model, m)] = {}
        for k in AGREEMENT_METRIC_NAMES:
            d = np.array(boot_values[ref_model][k]) - np.array(
                boot_values[m][k]
            )
            diffs[(ref_model, m)][k] = {
                "mean_diff": float(np.nanmean(d)),
                "lower": float(np.nanpercentile(d, 2.5)),
                "upper": float(np.nanpercentile(d, 97.5)),
                "p_gt_0": float(np.nanmean(d > 0)),
            }

    return point, ci, diffs


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
    # local (outbreak-cases-only, imports excluded) counterparts of the 5/10
    # lookups above, for the R01 agreement table's Outbreak-cases panel --
    # kept separate from county_outbreak_lookup_5/10 (which stay with-imports
    # and continue to feed other_tables()/the supplemental PPV/NPV tables
    # unchanged) so this AUPRC-label fix is isolated to that one figure.
    county_outbreak_lookup_5_local = {}
    county_outbreak_lookup_10_local = {}

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
            county_outbreak_lookup_5_local[key] = False
            county_outbreak_lookup_10_local[key] = False

            # Check all future years (including current year)
            for future_idx in range(idx, len(validation_years)):
                future_counties = all_counties[future_idx]

                # Find matching county in future year
                future_match = future_counties.filter(
                    (pl.col("county") == county) & (pl.col("state") == state)
                )

                if len(future_match) > 0:
                    cases = future_match["cases"][0]
                    cases_outbreak = int(future_match["cases_outbreak"][0])

                    if cases >= 5:
                        county_outbreak_lookup_5[key] = True
                    if cases >= 10:
                        county_outbreak_lookup_10[key] = True
                    if cases >= 20:
                        county_outbreak_lookup_20[key] = True
                    if cases_outbreak >= 5:
                        county_outbreak_lookup_5_local[key] = True
                    if cases_outbreak >= 10:
                        county_outbreak_lookup_10_local[key] = True

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
                            f"samples_partial_pooled_poisson_validation_testing_{years}_pool={pool}_CAR=True_{model_variant}_globals_prior_sd={GLOBALS_PRIOR_SD}.pkl",
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

    # --- targeted CAR scale factors for the prior-cases augmented models
    # (fit by 06b_prior_cases_rolling_validation.py) ---
    # only the (start_year, model_variant) combos that script actually
    # fits; NOT folded into the loop above, which would try to load
    # pool=False / start=-2 pickles that don't exist for these variants
    PRIOR_CASES_BASE_MODELS = [
        "All covariates",
        "MMR coverage",
        "MMR coverage + exemptions",
    ]
    # (start_year, model_variant, pool) combos needing a CAR scale factor --
    # every CAR=True prior-cases fit that was actually run. "in-window" was
    # skipped for pool=False (see 06b_prior_cases_rolling_validation.py):
    # it was the worst window def for every base/architecture already fit.
    PRIOR_CASES_COMBOS = [
        (start_year, f"{base} + prior cases {suffix}", True)
        for base in PRIOR_CASES_BASE_MODELS
        for start_year, suffix in [
            (2011, "since 2011"),
            (2019, "since 2011"),
            (2019, "in-window"),
        ]
    ] + [
        (start_year, f"{base} + prior cases since 2011", False)
        for base in PRIOR_CASES_BASE_MODELS
        for start_year in [2011, 2019]
    ]
    for start_year, model_variant, pool in PRIOR_CASES_COMBOS:
        observed = []
        predicted = []
        for idx, validation_year in enumerate([2021, 2022, 2023, 2024, 2025]):
            years = list(range(start_year, validation_year))
            if validation_year != 2025:
                cases = all_counties[idx]["cases"].sum()
            else:
                cases = (
                    all_counties[idx]
                    .filter(pl.col("state") != "South Carolina")["cases"]
                    .sum()
                    + 333
                )  # Add 2025 case count for SC

            with open(
                os.path.join(
                    output_dir,
                    f"samples_partial_pooled_poisson_validation_testing_{years}_pool={pool}_CAR=True_{model_variant}_globals_prior_sd={GLOBALS_PRIOR_SD}.pkl",
                ),
                "rb",
            ) as f:
                samples = pickle.load(f)

            samples_with_imports = (
                samples * all_imports[idx] + all_import_counts[idx]
            )
            samples_with_imports = jnp.sum(samples_with_imports, axis=1)
            predicted_median = jnp.median(samples_with_imports)

            observed.append(cases)
            predicted.append(predicted_median)

        scale = np.mean(np.array(predicted) / np.array(observed))
        car_scale_factors[(start_year, pool, model_variant)] = scale

    model_performance = []
    # kept separate from `model_performance` so this naive coverage-ranker
    # model feeds only the R01 agreement table, not the other supplemental
    # tables (which select from `model_performance` with no model filter)
    coverage_naive_perf = []
    # kept separate from `model_performance` for the same reason: these
    # "All covariates + prior cases" statistical models (from
    # 06b_prior_cases_rolling_validation.py) should feed only the R01
    # agreement table, not the other supplemental tables
    prior_cases_perf = []
    true_geq_10 = []
    # per-county "at-risk" calls captured for the model-agreement table:
    # (model_name, validation_year, "total"|"local") -> bool array, aligned
    # to all_counties[idx] row order
    STAT_HEADLINE_MODEL_NAME = "Statistical: All covariates + spatial + state-specific vaccine coefficients, trained since 2019"
    agreement_flags = {}
    # per-county-year score + labels captured for the pooled (micro) metrics
    # and county-clustered bootstrap in the model-agreement table:
    # (model_name, "total"|"local") -> list of dicts with score/roc_label/
    # ppv_label/auprc_label/cid arrays, one dict per validation year, aligned
    # to all_counties[idx] row order
    pooled_capture = {}

    # model-independent per-validation-year raw counts + cid, for ad hoc
    # threshold/label re-analysis of `pooled_capture` outside this script
    truth_by_year = []

    def _capture(
        model_name,
        condition_key,
        threshold,
        score,
        roc_label,
        ppv_label,
        auprc_label,
        cid,
    ):
        pooled_capture.setdefault(
            (model_name, condition_key, threshold), []
        ).append(
            {
                "score": np.asarray(score),
                "roc_label": np.asarray(roc_label),
                "ppv_label": np.asarray(ppv_label),
                "auprc_label": np.asarray(auprc_label),
                "cid": np.asarray(cid),
            }
        )

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
            true = np.array([int(t) for t in true])
            true_outbreak = all_counties[idx]["cases_outbreak"].to_numpy()
            true_outbreak = np.array([int(t) for t in true_outbreak])

            # Create override labels for this year
            counties_df = all_counties[idx]
            true_labels_5 = np.zeros(len(true), dtype=int)
            true_labels_10 = np.zeros(len(true), dtype=int)
            true_labels_20 = np.zeros(len(true), dtype=int)
            true_labels_5_local = np.zeros(len(true), dtype=int)
            true_labels_10_local = np.zeros(len(true), dtype=int)

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
                if county_outbreak_lookup_5_local.get(key, False):
                    true_labels_5_local[i] = 1
                if county_outbreak_lookup_10_local.get(key, False):
                    true_labels_10_local[i] = 1

            # county id for the pooled-metrics county-clustered bootstrap
            cid = (
                counties_df["state"].cast(pl.Utf8)
                + "_"
                + counties_df["county"].cast(pl.Utf8)
            ).to_numpy()

            # raw counts/cid don't depend on start_year; capture once
            if start_year == 2011:
                truth_by_year.append(
                    {
                        "year": validation_year,
                        "cid": cid,
                        "cases": true,
                        "cases_outbreak": true_outbreak,
                    }
                )

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
                                f"samples_partial_pooled_poisson_validation_testing_{years}_pool={pool}_CAR={car}_{model_variant}_globals_prior_sd={GLOBALS_PRIOR_SD}.pkl",
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
                        raw_samples = samples
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
                            raw_samples_unconditioned=raw_samples_unconditioned,
                            raw_samples=raw_samples,
                        )

                        model_performance.append(df)

                        # captured for every statistical variant (not just
                        # the headline) so the R01 figure/table can pick the
                        # highest-AUPRC statistical model dynamically
                        agreement_flags[
                            (model_name, validation_year, "total")
                        ] = at_risk_flags(samples_unconditioned, true >= 10)
                        agreement_flags[
                            (model_name, validation_year, "local")
                        ] = at_risk_flags(samples, true_outbreak >= 10)
                        _capture(
                            model_name,
                            "total",
                            10,
                            samples_unconditioned,
                            true >= 10,
                            true_labels_10,
                            true_labels_10,
                            cid,
                        )
                        _capture(
                            model_name,
                            "local",
                            10,
                            samples,
                            true_outbreak >= 10,
                            true_labels_10_local,
                            true_labels_10_local,
                            cid,
                        )
                        _capture(
                            model_name,
                            "total",
                            5,
                            samples_unconditioned,
                            true >= 5,
                            true_labels_5,
                            true_labels_5,
                            cid,
                        )
                        _capture(
                            model_name,
                            "local",
                            5,
                            samples,
                            true_outbreak >= 5,
                            true_labels_5_local,
                            true_labels_5_local,
                            cid,
                        )

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
            raw_samples = samples
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
                raw_samples_unconditioned=raw_samples_unconditioned,
                raw_samples=raw_samples,
            )
            true_geq_10.append(((true >= 10).sum(), len(true)))
            model_performance.append(df)

            # capture at-risk calls for every trained-since variant; the
            # winning mechanistic model isn't known until after aggregation
            agreement_flags[(model_name, validation_year, "total")] = (
                at_risk_flags(samples_unconditioned, true >= 10)
            )
            agreement_flags[(model_name, validation_year, "local")] = (
                at_risk_flags(samples, true_outbreak >= 10)
            )
            _capture(
                model_name,
                "total",
                10,
                samples_unconditioned,
                true >= 10,
                true_labels_10,
                true_labels_10,
                cid,
            )
            _capture(
                model_name,
                "local",
                10,
                samples,
                true_outbreak >= 10,
                true_labels_10_local,
                true_labels_10_local,
                cid,
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
                raw_samples_unconditioned=raw_samples_unconditioned_multi,
                raw_samples=samples_multi,
            )
            model_performance.append(df_multi)

            # --- naive surveillance baseline models ---
            # score each county purely from its own case history over the
            # same lookback window (`years`) used to train the other models
            history = analysis.filter(pl.col("year").is_in(years))
            history_agg = history.group_by(
                ["state", "county"], maintain_order=True
            ).agg(
                pl.col("cases").sum().alias("sum_cases"),
            )
            county_keys = all_counties[idx].select(["state", "county"])
            history_aligned = county_keys.join(
                history_agg,
                on=["state", "county"],
                how="left",
                maintain_order="left",
            ).fill_null(0)

            sum_cases = history_aligned["sum_cases"].to_numpy()

            naive_variants = [
                (
                    sum_cases.astype(float),
                    "Naive surveillance: prior case count",
                ),
            ]

            for naive_score, naive_model_name in naive_variants:
                df_naive = performance_table(
                    naive_score,
                    true,
                    mode="selection",
                    true_labels_override=true_labels_10,
                )

                if start_year == -2:
                    naive_model_name += ", last two years"
                elif start_year == 2011:
                    naive_model_name += ", surveillance since 2011"
                elif start_year == 2019:
                    naive_model_name += ", surveillance since 2019"
                else:
                    assert False, "Invalid training start year"

                df_naive = df_naive.with_columns(
                    pl.lit(validation_year).alias("Evaluation year"),
                    Model=pl.lit(naive_model_name),
                )
                df_naive = other_tables(
                    df_naive,
                    naive_score,
                    naive_score,
                    true,
                    true_outbreak,
                    true_labels_override_5=true_labels_5,
                    true_labels_override_10=true_labels_10,
                    true_labels_override_20=true_labels_20,
                )
                model_performance.append(df_naive)

                agreement_flags[
                    (naive_model_name, validation_year, "total")
                ] = at_risk_flags(naive_score, true >= 10)
                agreement_flags[
                    (naive_model_name, validation_year, "local")
                ] = at_risk_flags(naive_score, true_outbreak >= 10)
                _capture(
                    naive_model_name,
                    "total",
                    10,
                    naive_score,
                    true >= 10,
                    true_labels_10,
                    true_labels_10,
                    cid,
                )
                _capture(
                    naive_model_name,
                    "local",
                    10,
                    naive_score,
                    true_outbreak >= 10,
                    true_labels_10_local,
                    true_labels_10_local,
                    cid,
                )
                _capture(
                    naive_model_name,
                    "total",
                    5,
                    naive_score,
                    true >= 5,
                    true_labels_5,
                    true_labels_5,
                    cid,
                )
                _capture(
                    naive_model_name,
                    "local",
                    5,
                    naive_score,
                    true_outbreak >= 5,
                    true_labels_5_local,
                    true_labels_5_local,
                    cid,
                )

            # --- statistical models augmented with a prior-cases covariate
            # (fit by 06b_prior_cases_rolling_validation.py) ---
            # routed to a SEPARATE frame (`prior_cases_perf`), mirroring
            # `coverage_naive_perf`, so they feed only the R01 agreement table
            # CAR=True/pool=False was only fit for the "since 2011" window
            # def (see 06b_prior_cases_rolling_validation.py: "in-window"
            # was uniformly the worst window def for the other two
            # architectures, so it was skipped here to save compute)
            if start_year == 2011:
                prior_cases_variants = [
                    (f"{base} + prior cases since 2011", car, pool)
                    for base in PRIOR_CASES_BASE_MODELS
                    for car, pool in [
                        (True, True),
                        (False, False),
                        (True, False),
                    ]
                ]
            elif start_year == 2019:
                prior_cases_variants = [
                    (f"{base} + prior cases since 2011", car, pool)
                    for base in PRIOR_CASES_BASE_MODELS
                    for car, pool in [
                        (True, True),
                        (False, False),
                        (True, False),
                    ]
                ] + [
                    (f"{base} + prior cases in-window", car, pool)
                    for base in PRIOR_CASES_BASE_MODELS
                    for car, pool in [(True, True), (False, False)]
                ]
            else:
                # start_year == -2 (last-two-years) has no prior-cases fits
                prior_cases_variants = []

            for model_variant, car, pool in prior_cases_variants:
                with open(
                    os.path.join(
                        output_dir,
                        f"samples_partial_pooled_poisson_validation_testing_{years}_pool={pool}_CAR={car}_{model_variant}_globals_prior_sd={GLOBALS_PRIOR_SD}.pkl",
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
                    model_name += " + state-specific vaccine coefficients"
                if start_year == 2011:
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
                prior_cases_perf.append(df)

                agreement_flags[(model_name, validation_year, "total")] = (
                    at_risk_flags(samples_unconditioned, true >= 10)
                )
                agreement_flags[(model_name, validation_year, "local")] = (
                    at_risk_flags(samples, true_outbreak >= 10)
                )
                _capture(
                    model_name,
                    "total",
                    10,
                    samples_unconditioned,
                    true >= 10,
                    true_labels_10,
                    true_labels_10,
                    cid,
                )
                _capture(
                    model_name,
                    "local",
                    10,
                    samples,
                    true_outbreak >= 10,
                    true_labels_10_local,
                    true_labels_10_local,
                    cid,
                )
                _capture(
                    model_name,
                    "total",
                    5,
                    samples_unconditioned,
                    true >= 5,
                    true_labels_5,
                    true_labels_5,
                    cid,
                )
                _capture(
                    model_name,
                    "local",
                    5,
                    samples,
                    true_outbreak >= 5,
                    true_labels_5_local,
                    true_labels_5_local,
                    cid,
                )

            # --- naive MMR-coverage baseline model (prior-year coverage) ---
            # rank counties purely by their MMR vaccination coverage in the
            # year before the outbreak; lower coverage = higher predicted
            # risk. No importation model (mirrors the surveillance baseline).
            # This score is training-horizon-independent, so compute it once
            # (on the first `start_year`). It is routed to a SEPARATE frame
            # (`coverage_naive_perf`) so it feeds only the R01 agreement table,
            # never `model_performance` (which feeds the other supplements).
            if start_year == 2011:
                # `analysis` (analysis_dataset.csv) only carries the ~362
                # counties with reported case data, not the full county
                # universe. `projection_dataset_{validation_year}.csv` is the
                # same file `all_counties` was built from and has complete
                # per-county coverage back to 2018, so source from there.
                proj = pl.read_csv(
                    os.path.join(
                        output_dir, f"projection_dataset_{validation_year}.csv"
                    ),
                    schema_overrides={"fips": pl.String},
                )
                cov_prior = (
                    proj.filter(pl.col("year") == validation_year - 1)
                    .group_by(["state", "county"], maintain_order=True)
                    .agg(pl.col("estimated_mmr_cov").mean().alias("cov"))
                )
                cov = (
                    all_counties[idx]
                    .select(["state", "county"])
                    .join(
                        cov_prior,
                        on=["state", "county"],
                        how="left",
                        maintain_order="left",
                    )["cov"]
                    .to_numpy()
                    .astype(float)
                )
                # carry-forward fallback for counties missing year-(Y-1)
                # coverage: most recent available coverage before Y, else the
                # prior-year median (neutral rank).
                missing = np.isnan(cov)
                still_missing = np.zeros_like(missing)
                if missing.any():
                    recent = (
                        proj.filter(pl.col("year") < validation_year)
                        .sort("year")
                        .group_by(["state", "county"], maintain_order=True)
                        .agg(pl.col("estimated_mmr_cov").last().alias("cov"))
                    )
                    recent_cov = (
                        all_counties[idx]
                        .select(["state", "county"])
                        .join(
                            recent,
                            on=["state", "county"],
                            how="left",
                            maintain_order="left",
                        )["cov"]
                        .to_numpy()
                        .astype(float)
                    )
                    cov[missing] = recent_cov[missing]
                    still_missing = np.isnan(cov)
                    if still_missing.any():
                        cov[still_missing] = np.nanmedian(cov)
                print(
                    f"[coverage naive] {validation_year}: "
                    f"{int(missing.sum())} counties missing {validation_year - 1} "
                    f"coverage, {int(still_missing.sum())} needed median fill"
                )
                # higher score = lower coverage = higher risk
                cov_score = -cov
                cov_model_name = "Naive coverage: prior-year MMR coverage"

                df_cov = performance_table(
                    cov_score,
                    true,
                    mode="selection",
                    true_labels_override=true_labels_10,
                )
                df_cov = df_cov.with_columns(
                    pl.lit(validation_year).alias("Evaluation year"),
                    Model=pl.lit(cov_model_name),
                )
                df_cov = other_tables(
                    df_cov,
                    cov_score,
                    cov_score,
                    true,
                    true_outbreak,
                    true_labels_override_5=true_labels_5,
                    true_labels_override_10=true_labels_10,
                    true_labels_override_20=true_labels_20,
                )
                coverage_naive_perf.append(df_cov)

                agreement_flags[(cov_model_name, validation_year, "total")] = (
                    at_risk_flags(cov_score, true >= 10)
                )
                agreement_flags[(cov_model_name, validation_year, "local")] = (
                    at_risk_flags(cov_score, true_outbreak >= 10)
                )
                _capture(
                    cov_model_name,
                    "total",
                    10,
                    cov_score,
                    true >= 10,
                    true_labels_10,
                    true_labels_10,
                    cid,
                )
                _capture(
                    cov_model_name,
                    "local",
                    10,
                    cov_score,
                    true_outbreak >= 10,
                    true_labels_10_local,
                    true_labels_10_local,
                    cid,
                )
                _capture(
                    cov_model_name,
                    "total",
                    5,
                    cov_score,
                    true >= 5,
                    true_labels_5,
                    true_labels_5,
                    cid,
                )
                _capture(
                    cov_model_name,
                    "local",
                    5,
                    cov_score,
                    true_outbreak >= 5,
                    true_labels_5_local,
                    true_labels_5_local,
                    cid,
                )

            # memory hygiene: this loop loads ~15 large sample pickles per
            # validation year across the car/pool/model_variant + naive +
            # prior-cases blocks above; free them before the next year
            gc.collect()

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
                                f"samples_partial_pooled_poisson_validation_testing_{years}_pool={pool}_CAR={car}_{model_variant}_globals_prior_sd={GLOBALS_PRIOR_SD}.pkl",
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

            # memory hygiene: same rationale as the first Training-year-
            # variants loop above
            gc.collect()

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
    model_performance = pl.concat(model_performance, how="diagonal_relaxed")
    model_performance = add_relative_mae(model_performance)

    all_metric_columns = [
        col
        for col in model_performance.columns
        if col != "Model" and col != "Evaluation year"
    ]

    model_performance = aggregate(model_performance, all_metric_columns)
    model_performance = model_performance.filter(
        pl.col("Evaluation year") == "2021-2025"
    ).drop("Evaluation year")

    # aggregated separately: this naive coverage-ranker model must not enter
    # model_performance (see note at its definition above)
    coverage_naive_perf = pl.concat(
        coverage_naive_perf, how="diagonal_relaxed"
    )
    # `aggregate()` unconditionally references "Coverage (%)" (populated by
    # `coverage_table()` for statistical/mechanistic models); this naive model
    # never goes through that function, so add it as null to match.
    if "Coverage (%)" not in coverage_naive_perf.columns:
        coverage_naive_perf = coverage_naive_perf.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("Coverage (%)")
        )
    coverage_naive_metric_columns = [
        col
        for col in coverage_naive_perf.columns
        if col != "Model" and col != "Evaluation year"
    ]
    coverage_naive_perf = aggregate(
        coverage_naive_perf, coverage_naive_metric_columns
    )
    coverage_naive_perf = coverage_naive_perf.filter(
        pl.col("Evaluation year") == "2021-2025"
    ).drop("Evaluation year")

    # aggregated separately: these prior-cases statistical models must not
    # enter model_performance (see note at their definition above). They go
    # through the same performance_table/coverage_table path as the other
    # statistical models, so no "Coverage (%)" null-patch is needed.
    prior_cases_perf = pl.concat(prior_cases_perf, how="diagonal_relaxed")
    prior_cases_metric_columns = [
        col
        for col in prior_cases_perf.columns
        if col != "Model" and col != "Evaluation year"
    ]
    prior_cases_perf = aggregate(prior_cases_perf, prior_cases_metric_columns)
    prior_cases_perf = prior_cases_perf.filter(
        pl.col("Evaluation year") == "2021-2025"
    ).drop("Evaluation year")

    # --- model selection table ---
    model_selection_no_interval = [
        # "Absolute difference (cases)",
        "Coverage (%)",
    ]

    model_selection = model_performance.filter(
        ~pl.col("Model").str.starts_with("Naive")
    ).select(
        [
            "Model",
            # "Absolute difference (cases)",
            "Coverage (%)",
            *[
                f"{col}{suffix}"
                for col in [
                    "RMSE, 0+ outbreak cases (cases)",
                    "AUROC, 10+ cases (%)",
                ]
                for suffix in ["", "_lower", "_upper"]
            ],
        ]
    )
    model_selection = model_selection.sort(
        by=["Coverage (%)", "RMSE, 0+ outbreak cases (cases)"],
        descending=[True, False],
    )
    model_selection = beautify(
        model_selection,
        [
            "Coverage (%)",
            "RMSE, 0+ outbreak cases (cases)",
            "AUROC, 10+ cases (%)",
            # "Absolute difference (cases)",
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
        STAT_HEADLINE_MODEL_NAME,
        find_best_model(model_performance, subset="Mechanistic: MMR"),
        # find_best_model(
        #     model_performance, subset="Statistical", exclude="spatial"
        # ),
        "Statistical: MMR coverage + exemptions + state-specific vaccine coefficients, trained since 2011",
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

    # --- model agreement + classification table ---
    # Compares the at-risk county lists (each model's own closest-corner ROC
    # operating point) produced by the naive prior-case-count variant x 3
    # training horizons against the headline mechanistic and statistical
    # models.
    naive_base_names = [
        "Naive surveillance: prior case count",
    ]
    naive_suffixes = [
        ", surveillance since 2011",
        ", surveillance since 2019",
        ", last two years",
    ]
    naive_model_names = [
        base + suffix for base in naive_base_names for suffix in naive_suffixes
    ]
    COVERAGE_NAIVE_MODEL_NAME = "Naive coverage: prior-year MMR coverage"

    # dynamically pick the statistical variant with the highest pooled
    # All-cases AUPRC, among "Statistical:" models only
    stat_model_names = [
        name
        for (name, cond_key, threshold) in pooled_capture
        if cond_key == "total"
        and threshold == 10
        and name.startswith("Statistical:")
    ]
    stat_auprc_model_name = max(
        stat_model_names,
        key=lambda name: average_precision_score(
            np.concatenate(
                [c["auprc_label"] for c in pooled_capture[(name, "total", 10)]]
            ),
            np.concatenate(
                [c["score"] for c in pooled_capture[(name, "total", 10)]]
            ),
        ),
    )
    # same mechanical highest-pooled-All-cases-AUPRC rule, at threshold 5
    stat_model_names_t5 = [
        name
        for (name, cond_key, threshold) in pooled_capture
        if cond_key == "total"
        and threshold == 5
        and name.startswith("Statistical:")
    ]
    stat_auprc_model_name_t5 = max(
        stat_model_names_t5,
        key=lambda name: average_precision_score(
            np.concatenate(
                [c["auprc_label"] for c in pooled_capture[(name, "total", 5)]]
            ),
            np.concatenate(
                [c["score"] for c in pooled_capture[(name, "total", 5)]]
            ),
        ),
    )

    # ad hoc dump for offline threshold/label re-analysis of pooled_capture;
    # set POOLED_DUMP=<path> to enable. No effect on any figures/outputs table.
    if os.environ.get("POOLED_DUMP"):
        with open(os.environ["POOLED_DUMP"], "wb") as f:
            pickle.dump((pooled_capture, truth_by_year, validation_years), f)

    # raw pooled (score, roc_label, cid), All-cases/5+ only, for the 3 models
    # figure_r01_validation_comparison.py plots -- so it can recompute its own
    # flags/operating points/bootstrap CIs directly against the 5+ label,
    # instead of `agreement_flags` below, which is only ever built against
    # the 10+ label (that dict is for the main table's t=10 columns).
    r01_pooled_t5 = {
        name: {
            "score": np.concatenate(
                [c["score"] for c in pooled_capture[(name, "total", 5)]]
            ),
            "roc_label": np.concatenate(
                [c["roc_label"] for c in pooled_capture[(name, "total", 5)]]
            ),
            "cid": np.concatenate(
                [c["cid"] for c in pooled_capture[(name, "total", 5)]]
            ),
        }
        for name in [
            naive_model_names[0],
            COVERAGE_NAIVE_MODEL_NAME,
            stat_auprc_model_name_t5,
        ]
    }

    # per-county "at-risk" flags + true case counts, for figure_r01's
    # top-20%-outbreak / pairwise-discordance summary (report_model_stats.py
    # style block at the bottom of figure_r01_validation_comparison.py).
    # Small (a few MB) so this is written unconditionally, unlike POOLED_DUMP.
    with open(
        os.path.join(output_dir, "agreement_flags_and_truth.pkl"), "wb"
    ) as f:
        pickle.dump((agreement_flags, truth_by_year, r01_pooled_t5), f)

    mech_headline_model = main_models[1]
    MECH_2019_MODEL_NAME = "Mechanistic: MMR coverage, trained since 2019"
    # prior-cases augmented statistical models (fit by
    # 06b_prior_cases_rolling_validation.py); shown explicitly alongside the
    # naive models even though `stat_auprc_model_name` above may already
    # have picked one of them as the table's best "Statistical:" model
    PRIOR_CASES_MODEL_NAMES = (
        [
            f"Statistical: {base} + prior cases {suffix} + spatial"
            " + state-specific vaccine coefficients, trained since"
            f" {start_year}"
            for base in PRIOR_CASES_BASE_MODELS
            for start_year, suffix in [
                (2011, "since 2011"),
                (2019, "since 2011"),
                (2019, "in-window"),
            ]
        ]
        + [
            # CAR=False/pool=False ("vanilla") architecture -- the baseline
            # (non-prior-cases) versions of these had much higher AUPRC than
            # their CAR=True/pool=True siblings
            f"Statistical: {base} + prior cases {suffix}, trained since {start_year}"
            for base in PRIOR_CASES_BASE_MODELS
            for start_year, suffix in [
                (2011, "since 2011"),
                (2019, "since 2011"),
                (2019, "in-window"),
            ]
        ]
        + [
            # CAR=True/pool=False -- the baseline sweep found this architecture
            # beats CAR=True/pool=True on average, "since 2011" window def only
            # (see 06b_prior_cases_rolling_validation.py)
            f"Statistical: {base} + prior cases since 2011 + spatial, trained since {start_year}"
            for base in PRIOR_CASES_BASE_MODELS
            for start_year in [2011, 2019]
        ]
    )
    agreement_table_models = naive_model_names + [
        COVERAGE_NAIVE_MODEL_NAME,
        STAT_HEADLINE_MODEL_NAME,
        stat_auprc_model_name,
        mech_headline_model,
        MECH_2019_MODEL_NAME,
        *PRIOR_CASES_MODEL_NAMES,
    ]
    # drop duplicates, preserve order (mech_headline_model may already be the
    # trained-since-2019 mechanistic; stat_auprc_model_name may already be
    # the headline statistical model, or one of the PRIOR_CASES_MODEL_NAMES)
    agreement_table_models = list(dict.fromkeys(agreement_table_models))

    # best-performing naive model by AUROC (with imports, 10+ cases), same
    # metric find_best_model uses to rank mechanistic/statistical variants
    naive_ref_name = model_performance.filter(
        pl.col("Model").is_in(naive_model_names)
    ).sort(["AUROC, 10+ cases (%)", "Model"], descending=[True, False])[
        "Model"
    ][0]
    reference_models = {
        "% agreement vs Naive": naive_ref_name,
        "% agreement vs Mechanistic": mech_headline_model,
        "% agreement vs Statistical": STAT_HEADLINE_MODEL_NAME,
    }

    agreement_shared_columns = [
        "AUROC, 10+ (%)",
        "AUPRC, 10+ (%)",
        "Sensitivity, 10+ (%)",
        "Specificity, 10+ (%)",
        "PPV, 10+ (%)",
        "NPV, 10+ (%)",
    ]
    agreement_with_imports_columns = [
        "AUROC, 10+ cases (%)",
        "AUPRC, 10+ cases (%)",
        "Sensitivity, 10+ cases (%)",
        "Specificity, 10+ cases (%)",
        "PPV, 10+ cases (%)",
        "NPV, 10+ cases (%)",
    ]
    agreement_wo_imports_columns = [
        "AUROC, 10+ outbreak cases (%)",
        "AUPRC, 10+ outbreak cases (%)",
        "Sensitivity, 10+ outbreak cases (%)",
        "Specificity, 10+ outbreak cases (%)",
        "PPV, 10+ outbreak cases (%)",
        "NPV, 10+ outbreak cases (%)",
    ]

    # coverage_naive_perf and prior_cases_perf are concatenated only here, so
    # they appear in the R01 agreement table but never in the other
    # supplemental tables above (which all select from `model_performance`
    # alone)
    _agreement_source = pl.concat(
        [model_performance, coverage_naive_perf, prior_cases_perf],
        how="diagonal_relaxed",
    )

    def _agreement_metrics(columns, condition_label):
        sub = _agreement_source.select(  # type: ignore
            ["Model"]
            + [
                f"{col}{suffix}"
                for col in columns
                for suffix in ["", "_lower", "_upper"]
            ]
        )
        sub = sub.filter(pl.col("Model").is_in(agreement_table_models))
        rename_map = {
            orig: shared
            for orig, shared in zip(columns, agreement_shared_columns)
            if orig != shared
        }
        rename_map.update(
            {
                f"{orig}{suf}": f"{shared}{suf}"
                for orig, shared in zip(columns, agreement_shared_columns)
                for suf in ["_lower", "_upper"]
                if orig != shared
            }
        )
        sub = sub.rename(rename_map)
        return sub.with_columns(pl.lit(condition_label).alias("Condition"))

    agreement_table = pl.concat(
        [
            _agreement_metrics(
                agreement_with_imports_columns, "All cases (including imports)"
            ),
            _agreement_metrics(agreement_wo_imports_columns, "Outbreak cases"),
        ],
        how="vertical",
    )

    # pairwise % agreement: pool each model's per-county at-risk flags across
    # the 5 validation years, then compare to each reference model
    condition_to_flag_key = {
        "All cases (including imports)": "total",
        "Outbreak cases": "local",
    }

    def _pooled_flags(model_name, condition_label):
        key = condition_to_flag_key[condition_label]
        arrs = [
            agreement_flags[(model_name, vy, key)]
            for vy in [2021, 2022, 2023, 2024, 2025]
            if (model_name, vy, key) in agreement_flags
        ]
        return np.concatenate(arrs) if arrs else None

    agreement_rows = []
    for model_name in agreement_table_models:
        for condition_label in condition_to_flag_key:
            model_flags = _pooled_flags(model_name, condition_label)
            row = {"Model": model_name, "Condition": condition_label}
            for ref_col, ref_model in reference_models.items():
                ref_flags = _pooled_flags(ref_model, condition_label)
                if model_flags is None or ref_flags is None:
                    row[ref_col] = float("nan")  # type: ignore
                else:
                    row[ref_col] = 100.0 * float(  # type: ignore
                        np.mean(model_flags == ref_flags)
                    )
            agreement_rows.append(row)
    agreement_pct = pl.DataFrame(agreement_rows)

    agreement_table = agreement_table.join(
        agreement_pct, on=["Model", "Condition"], how="left"
    )

    # --- replace the macro (per-year-mean, min/max-bracket) classification
    # metrics with pooled (micro-averaged) point estimates + a
    # county-clustered bootstrap CI. Also emit a macro-vs-micro comparison
    # (to check for the naive model's growing-lookback-window confound) and
    # a paired-difference bootstrap of naive vs every other model in this
    # table (to test whether an apparent AUPRC advantage is real or a
    # small-sample macro-averaging artifact).
    N_BOOT = 2000
    BOOTSTRAP_SEED = 0
    metric_to_shared_col = dict(
        zip(AGREEMENT_METRIC_NAMES, agreement_shared_columns)
    )
    override_cols = [
        f"{col}{suffix}"
        for col in agreement_shared_columns
        for suffix in ["", "_lower", "_upper"]
    ]

    macro_check = agreement_table.select(
        ["Model", "Condition"] + override_cols
    )
    macro_check.write_csv(
        os.path.join(
            fig_dir, "table_model_agreement_classification_macro_check.csv"
        )
    )

    # naive reference model first, so `diffs` keys are naive-minus-other
    bootstrap_models = [naive_ref_name] + [
        m for m in agreement_table_models if m != naive_ref_name
    ]

    pooled_rows = []
    diff_rows = []
    for cond_idx, (condition_label, flag_key) in enumerate(
        condition_to_flag_key.items()
    ):
        # distinct seed per condition so the "total"/"local" bootstraps draw
        # independent county resamples rather than perfectly correlated ones
        point, ci, diffs = agreement_table_bootstrap(
            pooled_capture,
            bootstrap_models,
            flag_key,
            10,
            n_boot=N_BOOT,
            seed=BOOTSTRAP_SEED + cond_idx,
            cluster=False,
        )
        for m in bootstrap_models:
            row = {"Model": m, "Condition": condition_label}
            for metric, shared_col in metric_to_shared_col.items():
                row[shared_col] = point[m][metric]
                row[f"{shared_col}_lower"] = ci[m][metric][0]
                row[f"{shared_col}_upper"] = ci[m][metric][1]
            pooled_rows.append(row)

        for (model_a, model_b), metrics in diffs.items():
            for metric, shared_col in metric_to_shared_col.items():
                d = metrics[metric]
                diff_rows.append(
                    {
                        "Condition": condition_label,
                        "Model A (naive reference)": model_a,
                        "Model B": model_b,
                        "Metric": shared_col,
                        "Mean difference (A-B)": d["mean_diff"],
                        "CI lower": d["lower"],
                        "CI upper": d["upper"],
                        "P(A>B)": d["p_gt_0"],
                    }
                )

    pooled_df = pl.DataFrame(pooled_rows)
    agreement_table = agreement_table.drop(override_cols).join(
        pooled_df, on=["Model", "Condition"], how="left"
    )
    pl.DataFrame(diff_rows).write_csv(
        os.path.join(
            fig_dir, "table_agreement_paired_bootstrap_differences.csv"
        )
    )

    agreement_model_order = {
        m: i for i, m in enumerate(agreement_table_models)
    }
    agreement_condition_order = {
        "All cases (including imports)": 0,
        "Outbreak cases": 1,
    }
    agreement_table = (
        agreement_table.with_columns(
            pl.col("Model")
            .replace_strict(agreement_model_order, return_dtype=pl.Int32)
            .alias("_model_order"),
            pl.col("Condition")
            .replace_strict(agreement_condition_order, return_dtype=pl.Int32)
            .alias("_cond_order"),
        )
        .sort(["_model_order", "_cond_order"])
        .drop(["_model_order", "_cond_order"])
    )

    agreement_table = agreement_table.select(
        ["Model", "Condition"]
        + [
            f"{col}{suffix}"
            for col in agreement_shared_columns
            for suffix in ["", "_lower", "_upper"]
        ]
        + list(reference_models.keys())
    )

    agreement_table = beautify(
        agreement_table,
        agreement_shared_columns + list(reference_models.keys()),
        no_interval_columns=list(reference_models.keys()),
    )
    agreement_table.write_csv(
        os.path.join(fig_dir, "table_model_agreement_and_classification.csv")
    )

    # --- minimal t=5 comparison table (surveillance / coverage / headline
    # statistical model only), for the R01 figure's t=5 panels. Not folded
    # into agreement_table above because the mechanistic-model and percent-
    # agreement columns there were never captured at threshold 5 -- this
    # table only carries what figure_r01_validation_comparison.py needs.
    t5_models = [
        naive_model_names[0],
        COVERAGE_NAIVE_MODEL_NAME,
        stat_auprc_model_name_t5,
    ]
    t5_shared_columns = [
        "AUROC, 5+ (%)",
        "AUPRC, 5+ (%)",
        "Sensitivity, 5+ (%)",
        "Specificity, 5+ (%)",
    ]
    t5_metric_to_col = dict(
        zip(
            ["AUROC", "AUPRC", "Sensitivity", "Specificity"], t5_shared_columns
        )
    )
    t5_rows = []
    for cond_idx, (condition_label, flag_key) in enumerate(
        condition_to_flag_key.items()
    ):
        point, ci, _ = agreement_table_bootstrap(
            pooled_capture,
            t5_models,
            flag_key,
            5,
            n_boot=N_BOOT,
            seed=BOOTSTRAP_SEED + cond_idx,
            cluster=False,
        )
        for m in t5_models:
            row = {"Model": m, "Condition": condition_label}
            for metric, col in t5_metric_to_col.items():
                row[col] = point[m][metric]  # type: ignore
                row[f"{col}_lower"] = ci[m][metric][0]  # type: ignore
                row[f"{col}_upper"] = ci[m][metric][1]  # type: ignore
            t5_rows.append(row)
    t5_table = beautify(pl.DataFrame(t5_rows), t5_shared_columns)
    t5_table.write_csv(
        os.path.join(
            fig_dir, "table_model_agreement_and_classification_t5.csv"
        )
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
        "WIS, 0+ cases (cases)",
        "CRPS, 0+ cases (cases)",
        "Relative MAE, 0+ cases",
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
        regression_with_imports,
        regression_with_imports_columns,
        decimals_by_column={"Relative MAE, 0+ cases": 2},
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
        "WIS, 0+ outbreak cases (cases)",
        "CRPS, 0+ outbreak cases (cases)",
        "Relative MAE, 0+ outbreak cases",
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
        regression_wo_imports,
        regression_wo_imports_columns,
        decimals_by_column={"Relative MAE, 0+ outbreak cases": 2},
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
