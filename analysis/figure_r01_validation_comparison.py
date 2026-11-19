import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import rc
from matplotlib.patches import Patch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from outbreak_size.headline_models import (
    HEADLINE_FILE_NAME,
    read_headline_models,
)
from outbreak_size.model_colors import MECHANISTIC_COLOR, STATISTICAL_COLOR

fig_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)

font = {"family": "Nimbus Roman", "size": 28}
rc("font", **font)


def setup_plot(nrows, ncols, figsize=(16, 8)):
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis="x", direction="out")
    ax.tick_params(axis="y", direction="out")
    # offset the spines
    for spine in ax.spines.values():
        spine.set_position(("outward", 5))
    # put the grid behind
    ax.set_axisbelow(True)
    return fig, ax


def parse_point_ci(s):
    """'78.7 [70.5, 85.8]' -> (point, lower_error, upper_error) for yerr."""
    point, lower, upper = (
        float(x)
        for x in re.match(r"([\d.]+) \[([\d.]+), ([\d.]+)\]", s).groups()  # type: ignore
    )
    return point, point - lower, upper - point


# (Model string in table_model_agreement_and_classification.csv, legend label, color)
# colors are the first three categorical slots of the dataviz skill's validated
# palette, chosen because they are explicitly validated as CVD-safe for all
# pairs among exactly 3 series. The statistical model is picked dynamically
# (see `pick_models` below) since it's whichever "Statistical:" variant has
# the highest All-cases AUPRC in the table.
SURVEILLANCE_MODEL = (
    "Naive surveillance: prior case count, surveillance since 2011"
)
COVERAGE_MODEL = "Naive coverage: prior-year MMR coverage"
MODEL_COLORS = {
    SURVEILLANCE_MODEL: ("Surveillance", "#1baf7a"),
    COVERAGE_MODEL: ("MMR coverage", MECHANISTIC_COLOR),
}

# row-level (cluster=False) bootstrap, matching agreement_table_bootstrap's own
# call for the All-cases ("total") condition in table_model_performance_validation.py
N_BOOT = 2000
BOOTSTRAP_SEED = 0


def _operating_point(score, roc_label, rule):
    """A point on the (score, roc_label) ROC curve: 'closest_corner' is the
    same pick the table itself uses for its Sensitivity/Specificity columns;
    'balanced' instead picks the point minimizing |sensitivity - specificity|."""
    fpr, tpr, thr = roc_curve(roc_label, score)
    if rule == "balanced":
        idx = np.argmin(np.abs(tpr - (1 - fpr)))
    else:
        idx = np.argmin(np.sqrt(fpr**2 + (1 - tpr) ** 2))
    return tpr[idx] * 100, (1 - fpr[idx]) * 100, thr[idx]


def sens_spec_and_flags(score, roc_label, rule):
    """Point + bootstrap CI for sensitivity/specificity at the chosen
    operating point, plus the resulting per-county flag array."""
    sens, spec, thr = _operating_point(score, roc_label, rule)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(score)
    sens_boot = np.empty(N_BOOT)
    spec_boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        sens_boot[i], spec_boot[i], _ = _operating_point(
            score[idx], roc_label[idx], rule
        )
    metrics = {
        "Sensitivity": (
            sens,
            np.percentile(sens_boot, 2.5),
            np.percentile(sens_boot, 97.5),
        ),
        "Specificity": (
            spec,
            np.percentile(spec_boot, 2.5),
            np.percentile(spec_boot, 97.5),
        ),
    }
    return metrics, score >= thr


def pick_models(df, threshold):
    """Fixed Surveillance/MMR-coverage rows + whichever 'Statistical:' model
    has the highest All-cases AUPRC in the agreement table (at t=10 this is
    the table's only dynamic pick; at t=5 the table carries only a single,
    manually-chosen 'Statistical:' row, so this just returns it)."""
    stat_auprc = parse_point_ci
    best_row = (
        df.filter(
            pl.col("Model").str.starts_with("Statistical:")
            & (pl.col("Condition") == ALL_CASES)
        )
        .with_columns(
            pl.col(f"AUPRC, {threshold}+ (%)")
            .map_elements(lambda s: stat_auprc(s)[0], return_dtype=pl.Float64)
            .alias("_auprc_point")
        )
        .sort("_auprc_point", descending=True, maintain_order=True)
    )
    stat_model = best_row["Model"][0]
    return [
        (SURVEILLANCE_MODEL, *MODEL_COLORS[SURVEILLANCE_MODEL]),
        (COVERAGE_MODEL, *MODEL_COLORS[COVERAGE_MODEL]),
        (stat_model, "Multivariable", STATISTICAL_COLOR),
    ]


METRIC_LABELS = ["AUPRC", "AUC", "Sensitivity", "Specificity"]


def metric_cols(threshold):
    return [
        f"{name}, {threshold}+ (%)"
        for name in ["AUPRC", "AUROC", "Sensitivity", "Specificity"]
    ]


ALL_CASES = "All cases (including imports)"
OUTBREAK_CASES = "Outbreak cases"


def plot_comparison(
    df,
    condition,
    models,
    threshold,
    coverage_ops=None,
    ylabel="Validation performance, top 20% outbreaks (%)",
):
    """`coverage_ops`: the {"Sensitivity": (point, lo, hi), "Specificity": ...}
    dict from `sens_spec_and_flags` for the MMR-coverage model (balanced
    operating point), applied only when `condition == ALL_CASES`."""
    fig, ax = setup_plot(1, 1, figsize=(12, 9))

    cols = metric_cols(threshold)
    n_bins = len(cols)
    n_bars_per_group = len(models)
    bar_width = 0.4
    group_width = bar_width * n_bars_per_group
    group_spacing = 0.3
    total_group_width = group_width + group_spacing
    x_positions = np.arange(n_bins) * total_group_width

    for bar_idx, (model_name, label, color) in enumerate(models):
        row = df.filter(
            (pl.col("Model") == model_name)
            & (pl.col("Condition") == condition)
        )
        x_pos = x_positions + bar_idx * bar_width

        for m_idx, col in enumerate(cols):
            x = x_pos[m_idx]
            metric_name = METRIC_LABELS[m_idx]

            if (
                model_name == COVERAGE_MODEL
                and condition == ALL_CASES
                and coverage_ops is not None
                and metric_name in coverage_ops
            ):
                point, lo_ci, hi_ci = coverage_ops[metric_name]
                lo, hi = point - lo_ci, hi_ci - point
            else:
                point, lo, hi = parse_point_ci(row[col].item())

            ax.bar(x, point, width=bar_width, color=color)
            ax.errorbar(
                x,
                point,
                yerr=[[lo], [hi]],
                fmt="none",
                ecolor="black",
                capsize=5,
                elinewidth=1.5,
                alpha=0.8,
            )

    ax.set_xticks(x_positions + group_width / 2 - bar_width / 2)
    ax.set_xticklabels(METRIC_LABELS)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 100)

    model_handles = [
        Patch(facecolor=color, label=label) for _, label, color in models
    ]
    ax.legend(
        handles=model_handles,
        loc="upper left",
        edgecolor="black",
        fontsize=font["size"] - 4,  # type: ignore
    )

    return fig, ax


THRESHOLD_TABLES = {
    5: "table_model_agreement_and_classification_t5.csv",
}


def report_model_comparison_stats(df, models, threshold):
    """Print AUPRC/AUROC (from `df`) and Sensitivity/Specificity with
    bootstrap CIs (recomputed directly against the 5+ label -- see
    `sens_spec_and_flags`), the % of each model's flagged counties that were
    actually in the top 20% of that year's case counts, the % of counties
    with >=1 / >=3 true cases each model flags, and flag discordance between
    Surveillance/Multivariable and MMR coverage/Multivariable restricted to
    top-20%-outbreak counties (the only discordance that matters, since
    restricting to the true outbreaks avoids diluting it with the trivial
    true-negative majority). Flags/case counts are pooled across all 5
    validation years (same pooling the agreement table already uses for its
    "% agreement" columns). Returns the MMR-coverage Sensitivity/Specificity
    dict so `plot_comparison` can reuse it instead of recomputing."""
    with open(
        os.path.join(output_dir, "agreement_flags_and_truth.pkl"), "rb"
    ) as f:
        _, truth_by_year, r01_pooled_t5 = pickle.load(f)

    validation_years = [d["year"] for d in truth_by_year]
    pooled_cases = np.concatenate([d["cases"] for d in truth_by_year])
    # most validation counties have 0 cases, so the 80th percentile of all
    # counties is 0 and every flag trivially counts as "top 20%"; take the
    # percentile among counties with any cases instead
    top20_cutoff = np.quantile(pooled_cases[pooled_cases > 0], 0.8)

    print(
        f"\nt={threshold}: top-20% outbreak cutoff (pooled across "
        f"{validation_years}): {top20_cutoff:.1f}+ cases"
    )
    flags = {}
    coverage_ops = None
    for model_name, label, _color in models:
        row = df.filter(
            (pl.col("Model") == model_name)
            & (pl.col("Condition") == ALL_CASES)
        )
        pooled = r01_pooled_t5[model_name]
        rule = "balanced" if model_name == COVERAGE_MODEL else "closest_corner"
        sens_spec, model_flags = sens_spec_and_flags(
            pooled["score"], pooled["roc_label"], rule
        )
        if model_name == COVERAGE_MODEL:
            coverage_ops = sens_spec

        metric_strs = []
        for metric_name in ["AUPRC", "AUROC"]:
            point, lo_err, hi_err = parse_point_ci(
                row[f"{metric_name}, {threshold}+ (%)"].item()
            )
            lo, hi = point - lo_err, point + hi_err
            metric_strs.append(
                f"{metric_name}={point:5.1f} [{lo:.1f},{hi:.1f}]"
            )
        for metric_name in ["Sensitivity", "Specificity"]:
            point, lo, hi = sens_spec[metric_name]
            metric_strs.append(
                f"{metric_name}={point:5.1f} [{lo:.1f},{hi:.1f}]"
            )
        print(f"  {label:<14} " + "  ".join(metric_strs))

        flags[label] = model_flags
        n_flagged = int(model_flags.sum())
        pct_top20 = (
            100 * np.mean(pooled_cases[model_flags] >= top20_cutoff)
            if n_flagged
            else float("nan")
        )
        # recall: of counties that actually had >= c cases, % the model flagged
        recall_1 = 100 * np.mean(model_flags[pooled_cases >= 1])
        recall_3 = 100 * np.mean(model_flags[pooled_cases >= 3])
        print(
            f"  {'':<14} flagged={n_flagged:<5} of which top-20%={pct_top20:5.1f}%  "
            f"found 1+ cases={recall_1:5.1f}%  found 3+ cases={recall_3:5.1f}%"
        )

    outbreak_mask = pooled_cases >= top20_cutoff
    n_outbreak = int(outbreak_mask.sum())
    print(
        f"  pairwise county-level discordance, restricted to top-20% outbreak "
        f"counties (>= {top20_cutoff:.1f} true cases, n={n_outbreak}):"
    )
    for a, b in [
        ("Surveillance", "Multivariable"),
        ("MMR coverage", "Multivariable"),
    ]:
        fa = flags[a][outbreak_mask]
        fb = flags[b][outbreak_mask]
        both = int(np.sum((fa == 1) & (fb == 1)))
        a_only = int(np.sum((fa == 1) & (fb == 0)))
        b_only = int(np.sum((fa == 0) & (fb == 1)))
        neither = int(np.sum((fa == 0) & (fb == 0)))
        disc = a_only + b_only
        print(
            f"    {a} vs {b}: {a}=1,{b}=1: {both}   {a}=1,{b}=0: {a_only}   "
            f"{a}=0,{b}=1: {b_only}   {a}=0,{b}=0: {neither}"
        )
        print(
            f"      discordant = ({a}-only + {b}-only) / n = "
            f"({a_only} + {b_only}) / {n_outbreak} = {100 * disc / n_outbreak:.1f}%"
        )

    return coverage_ops


def any_later_label(truth_by_year, threshold):
    """1 if the county has >= `threshold` cases in this validation year or any
    later one: the table's multi-year label, which it uses for AUPRC only."""
    years = [d["year"] for d in truth_by_year]
    by_year = [dict(zip(d["cid"], d["cases"])) for d in truth_by_year]
    return np.array(
        [
            int(
                any(
                    by_year[j].get(c, 0) >= threshold
                    for j in range(i, len(years))
                )
            )
            for i, d in enumerate(truth_by_year)
            for c in d["cid"]
        ]
    )


def table_from_pooled_dump(path, threshold):
    """All-cases table in the layout of `table_model_agreement_and_classification_t5.csv`
    ('point [lo, hi]' strings) relabeled at any `threshold`, from a POOLED_DUMP
    pickle (set POOLED_DUMP=<path> when running table_model_performance_validation.py).
    AUROC/Sensitivity/Specificity use the same-year label, AUPRC the any-later
    label; the statistical model is the one with the highest pooled AUPRC.
    Row-level bootstrap with draws shared across models, as in the table."""
    with open(path, "rb") as f:
        pooled_capture, truth_by_year, _ = pickle.load(f)
    cid = np.concatenate([d["cid"] for d in truth_by_year])
    cases = np.concatenate([d["cases"] for d in truth_by_year])
    roc_label = (cases >= threshold).astype(int)
    auprc_label = any_later_label(truth_by_year, threshold)

    # scores don't depend on the label threshold the dump was captured at
    scores = {}
    for (name, cond, thr), entries in pooled_capture.items():
        if cond == "total" and thr == 5:
            assert np.array_equal(
                np.concatenate([e["cid"] for e in entries]), cid
            )
            scores[name] = np.concatenate(
                [e["score"] for e in entries]
            ).astype(float)
    stat_model = max(  # type: ignore
        (n for n in scores if n.startswith("Statistical:")),
        key=lambda n: average_precision_score(auprc_label, scores[n]),  # type: ignore
    )
    models = [
        (SURVEILLANCE_MODEL, *MODEL_COLORS[SURVEILLANCE_MODEL]),
        (COVERAGE_MODEL, *MODEL_COLORS[COVERAGE_MODEL]),
        (stat_model, "Multivariable", STATISTICAL_COLOR),
    ]

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = [rng.integers(0, len(cid), size=len(cid)) for _ in range(N_BOOT)]

    def fmt(point, lo, hi):
        return f"{point:.1f} [{lo:.1f}, {hi:.1f}]"

    rows = []
    for model_name, _label, _color in models:
        s = scores[model_name]
        rule = "balanced" if model_name == COVERAGE_MODEL else "closest_corner"
        sens_spec, _ = sens_spec_and_flags(s, roc_label, rule)
        auroc = [roc_auc_score(roc_label[i], s[i]) * 100 for i in draws]
        auprc = [
            average_precision_score(auprc_label[i], s[i]) * 100 for i in draws
        ]
        cols = metric_cols(threshold)
        rows.append(
            {
                "Model": model_name,
                "Condition": ALL_CASES,
                cols[0]: fmt(
                    average_precision_score(auprc_label, s) * 100,
                    *np.percentile(
                        np.asarray(auprc, dtype=float), [2.5, 97.5]
                    ),
                ),
                cols[1]: fmt(
                    roc_auc_score(roc_label, s) * 100,
                    *np.percentile(
                        np.asarray(auroc, dtype=float), [2.5, 97.5]
                    ),
                ),
                cols[2]: fmt(*sens_spec["Sensitivity"]),
                cols[3]: fmt(*sens_spec["Specificity"]),
            }
        )
    return pl.DataFrame(rows), models


MACRO_TABLE = "table_model_agreement_classification_macro_check.csv"


def macro_table_as_strings(path, threshold):
    """The main-table framework (per-year metrics averaged over validation
    years; `_lower`/`_upper` are the min/max across years, not bootstrap CIs),
    reformatted to the 'point [lo, hi]' strings `plot_comparison` parses."""
    raw = pl.read_csv(path)
    out = raw.select("Model", "Condition")
    for col in metric_cols(threshold):
        out = out.with_columns(
            pl.format(
                "{} [{}, {}]",
                pl.lit(raw[col].round(1)),
                pl.lit(raw[f"{col}_lower"].round(1)),
                pl.lit(raw[f"{col}_upper"].round(1)),
            ).alias(col)
        )
    return out


if __name__ == "__main__":
    # the headline model is the statistical model chosen with
    # 06b_choose_headline_models.py (the model of 07 and of the projections)
    HEADLINE_MODEL = read_headline_models(
        os.path.join(output_dir, HEADLINE_FILE_NAME), fig_dir=fig_dir
    ).statistical.name

    # main-table framework (10+, per-year average) with the headline model
    macro_df = macro_table_as_strings(os.path.join(fig_dir, MACRO_TABLE), 10)
    # the table of the comparison holds the models it was made with (no script
    # of the pipeline writes it): the headline model must be one of them
    if HEADLINE_MODEL not in macro_df["Model"].to_list():
        raise ValueError(
            f"{MACRO_TABLE} has no scores of the headline model "
            f"{HEADLINE_MODEL}: the comparison needs a table that has it"
        )
    headline_models = [
        (SURVEILLANCE_MODEL, *MODEL_COLORS[SURVEILLANCE_MODEL]),
        (COVERAGE_MODEL, *MODEL_COLORS[COVERAGE_MODEL]),
        (HEADLINE_MODEL, "Multivariable", STATISTICAL_COLOR),
    ]
    for condition, suffix in [
        (ALL_CASES, "all_cases"),
        (OUTBREAK_CASES, "outbreak_cases"),
    ]:
        print(f"\nt=10 macro, {condition}:")
        for model_name, label, _ in headline_models:
            row = macro_df.filter(
                (pl.col("Model") == model_name)
                & (pl.col("Condition") == condition)
            )
            print(
                f"  {label:<14}"
                + "  ".join(
                    f"{m}={row[c].item()}"
                    for m, c in zip(METRIC_LABELS, metric_cols(10))
                )
            )
        fig, ax = plot_comparison(macro_df, condition, headline_models, 10)
        fig.savefig(
            os.path.join(
                fig_dir,
                f"figure_r01_validation_comparison_{suffix}_t10_macro_headline.png",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # 3+ cases, All cases: AUROC same-year label, AUPRC any-later label
    # (needs a POOLED_DUMP pickle, see `table_from_pooled_dump`)
    if os.environ.get("POOLED_DUMP"):
        df3, models3 = table_from_pooled_dump(os.environ["POOLED_DUMP"], 3)
        print(f"\nt=3, {ALL_CASES}: statistical model = {models3[2][0]}")
        for model_name, label, _ in models3:
            row = df3.filter(pl.col("Model") == model_name)
            print(
                f"  {label:<14}"
                + "  ".join(
                    f"{m}={row[c].item()}"
                    for m, c in zip(METRIC_LABELS, metric_cols(3))
                )
            )
        fig, ax = plot_comparison(
            df3,
            ALL_CASES,
            models3,
            3,
            ylabel="Validation performance, outbreaks (%)",
        )
        fig.savefig(
            os.path.join(
                fig_dir, "figure_r01_validation_comparison_all_cases_t3.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    for threshold, table_name in THRESHOLD_TABLES.items():
        df = pl.read_csv(os.path.join(fig_dir, table_name))
        models = pick_models(df, threshold)
        print(
            f"t={threshold}: statistical model selected for the R01 figure: {models[2][0]}"
        )
        coverage_ops = report_model_comparison_stats(df, models, threshold)

        for condition, suffix in [
            (ALL_CASES, "all_cases"),
            (OUTBREAK_CASES, "outbreak_cases"),
        ]:
            fig, ax = plot_comparison(
                df,
                condition,
                models,
                threshold,
                coverage_ops=coverage_ops if condition == ALL_CASES else None,
            )
            fig.savefig(
                os.path.join(
                    fig_dir,
                    f"figure_r01_validation_comparison_{suffix}_t{threshold}.png",
                ),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)
