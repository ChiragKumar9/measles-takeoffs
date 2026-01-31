import os
import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import polars as pl
from matplotlib import rc
from numpyro import enable_x64
from scipy.stats import rankdata
from sklearn.metrics import precision_recall_curve, roc_curve

enable_x64()

font = {"family": "Nimbus Roman", "size": 28}
rc("font", **font)

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)
fig_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")

df = pl.read_csv(
    os.path.join(data_dir, "analysis_dataset.csv"),
    # make sure we read fips codes as strings
    schema_overrides={"fips": pl.String},
)

# we need to read the following models
# read the mechanistic model separately because it is stored differently
models = {}
mechanistic = []
for idx, year in enumerate([2021, 2022, 2023, 2024, 2025]):
    with open(
        os.path.join(
            output_dir, f"final_outbreak_size_predictions_{year}.pkl"
        ),
        "rb",
    ) as f:
        m = pickle.load(
            f,
        )

    mechanistic.append(m["cases"])
models["Mechanistic: MMR coverage"] = mechanistic

# we now have to read the different poisson models
# the model options are covariate versions, with/without pooling, with and without car
# for the covariate versions, if car is true, we add the number of counties for that year
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

            additional_info = ""
            if pool:
                additional_info += ", partial pooled"
            if car:
                additional_info += ", spatial"
            models[f"Statistical: {model}{additional_info}"] = predictions


def setup_plot(nrows, ncols, figsize=(10, 7.5)):
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows * ncols == 1:
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
    for temp in ax.flatten():  # type: ignore
        temp.spines["top"].set_visible(False)
        temp.spines["right"].set_visible(False)
        temp.get_xaxis().tick_bottom()
        temp.get_yaxis().tick_left()
        temp.tick_params(axis="x", direction="out")
        temp.tick_params(axis="y", direction="out")
        # offset the spines
        for spine in temp.spines.values():
            spine.set_position(("outward", 5))
        # put the grid behind
        temp.set_axisbelow(True)
    return fig, ax


def plot_predictions(
    df,
    samples,
    naive_model,
    fig,
    ax,
    label_naive="Mechanistic: MMR coverage",
    label_statistical="Statistical: All covariates",
    show_r2=False,
):
    median = jnp.median(samples, axis=0)
    # median = stats.poisson.ppf(0.5, samples).flatten()

    lower = jnp.quantile(samples, 0.025, axis=0)
    upper = jnp.quantile(samples, 0.975, axis=0)

    # lower = stats.poisson.ppf(0.025, median).flatten()
    # upper = stats.poisson.ppf(0.975, median).flatten()

    naive_median = jnp.median(naive_model, axis=0)
    # naive_median = stats.poisson.ppf(0.5, naive_model).flatten()
    naive_lower = jnp.quantile(naive_model, 0.025, axis=0)
    naive_upper = jnp.quantile(naive_model, 0.975, axis=0)
    # naive_lower = stats.poisson.ppf(0.025, naive_median).flatten()
    # naive_upper = stats.poisson.ppf(0.975, naive_median).flatten()

    offset = 0.1
    ax.errorbar(
        df["cases_outbreak"] - offset,
        naive_median,
        yerr=[
            naive_median - naive_lower,
            naive_upper - naive_median,
        ],
        fmt="o",
        ecolor="red",
        c="red",
        capsize=0,
        alpha=0.4,
        label=label_naive,
    )

    ax.errorbar(
        df["cases_outbreak"] + offset,
        median,
        yerr=[median - lower, upper - median],
        fmt="o",
        ecolor="green",
        c="green",
        capsize=0,
        alpha=0.8,
        label=label_statistical,
    )

    ax.set_xlabel("Reported cases (N)")
    ax.set_ylabel("Predicted cases (N)")
    ax.set_xscale("symlog")
    ax.set_yscale("symlog")
    ax.plot(
        [0, 1.05 * max(df["cases_outbreak"].max(), upper.max())],  # type: ignore
        [0, 1.05 * max(df["cases_outbreak"].max(), upper.max())],  # type: ignore
        ls="--",
        c="k",
    )
    ax.set_xlim(-0.3, 1.05 * max(df["cases_outbreak"].max(), upper.max()))  # type: ignore
    ax.set_ylim(-0.3, 1.05 * max(df["cases_outbreak"].max(), upper.max()))  # type: ignore

    ax.set_yticks([0, 1, 10, 100], labels=["0", "1", "10", "100"])
    ax.set_xticks([0, 1, 10, 100], labels=["0", "1", "10", "100"])

    ax.legend(loc=(0.0, 0.995))

    # Manually set minor tick locations on the x-axis
    minor_x_ticks = [
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        200,
        300,
        400,
    ]
    ax.xaxis.set_minor_locator(tck.FixedLocator(minor_x_ticks))

    # Manually set minor tick locations on the y-axis
    minor_y_ticks = [
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        200,
        300,
        400,
    ]
    ax.yaxis.set_minor_locator(tck.FixedLocator(minor_y_ticks))

    if show_r2:
        # add a text annotation of the R2
        R2 = (
            jnp.corrcoef(
                df["cases_outbreak"].to_jax(),
                np.median(samples_full_scaled, axis=0),
            )[0, 1]
            ** 2
        )
        R2 = R2 * 100  # as a percentage
        ax.text(
            0.035 * max(df["cases_outbreak"].max(), upper.max()),
            1.4 * max(df["cases_outbreak"].max(), upper.max()),
            f"$R^2$ = {R2:.2f}%",
            fontsize=26,
            color="green",
        )

        # add a text annotation of the R2
        R2 = (
            jnp.corrcoef(
                df["cases_outbreak"].to_jax(),
                np.median(naive_full_scaled, axis=0),
            )[0, 1]
            ** 2
        )
        if np.isnan(R2):
            R2 = 0.0
        R2 = R2 * 100  # as a percentage
        ax.text(
            0.035 * max(df["cases_outbreak"].max(), upper.max()),
            3 * max(df["cases_outbreak"].max(), upper.max()),
            f"$R^2$ = {R2:.2f}%",
            fontsize=26,
            color="red",
        )
    ax.set_ylim(-0.05)

    return fig, ax


df_validation = df.filter(pl.col("year") == 2025)
# post hoc scaling
naive_model = models["Mechanistic: MMR coverage"][-1]
samples_full = models["Statistical: All covariates"][-1]

true_full = df_validation["cases_outbreak"].to_list()

samples_full = np.array(samples_full, dtype=np.float64)
naive_full = np.array(naive_model, dtype=np.float64)
true_full = np.array(true_full, dtype=np.float64)

samples_full_scaled = samples_full.copy()
naive_full_scaled = naive_full.copy()

y_true = true_full >= 10

samples_full_scaled = np.sort(samples_full_scaled, axis=0)

samples_full_scaled_median = np.median(samples_full_scaled, axis=0)
max_val = np.max(samples_full_scaled_median)
precision, recall, thresholds = precision_recall_curve(
    y_true, samples_full_scaled_median / max_val
)
f1_scores = 2 * (precision * recall) / (precision + recall)
f1_scores = np.nan_to_num(f1_scores, nan=-np.inf)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold = optimal_threshold * max_val
samples_full_scaled = samples_full_scaled / optimal_threshold * 10

# for i in range(samples_full.shape[0]):
#     temp = samples_full[i, :]
#     max_val = np.max(temp)
#     if max_val == 0:
#         max_val = 1 # to mitigate divide by zero error
#     temp = temp / max_val
#     precision, recall, thresholds = precision_recall_curve(y_true, temp)
#     f1_scores = 2 * (precision * recall) / (precision + recall)
#     f1_scores = np.nan_to_num(f1_scores, nan=-np.inf)
#     # fpr, tpr, thresholds = roc_curve(y_true, temp)
#     # youdens_j = tpr - fpr
#     optimal_idx = np.argmax(f1_scores)
#     if optimal_idx >= len(thresholds):
#         temp = 0 * temp
#     else:
#         optimal_threshold = thresholds[optimal_idx]
#         if optimal_threshold == 0:
#             temp = temp + 10
#         else:
#             # temp = temp * range_
#             temp = temp / optimal_threshold
#             temp = temp * 10
#             # temp = temp * max_val
#     samples_full_scaled[i, :] = temp

# for i in range(naive_full.shape[0]):
#     temp = naive_full[i, :]
#     max_val = np.max(temp)
#     range_ = np.max(temp) - np.min(temp)
#     if max_val == 0:
#         max_val = 1 # to mitigate divide by zero error
#     temp = temp / max_val
#     precision, recall, thresholds = precision_recall_curve(y_true, temp)
#     f1_scores = 2 * (precision * recall) / (precision + recall)
#     f1_scores = np.nan_to_num(f1_scores, nan=-np.inf)
#     # fpr, tpr, thresholds = roc_curve(y_true, temp)
#     # youdens_j = tpr - fpr
#     optimal_idx = np.argmax(f1_scores)
#     if optimal_idx >= len(thresholds):
#         temp = 0 * temp
#     else:
#         optimal_threshold = thresholds[optimal_idx]
#         if optimal_threshold == 0:
#             temp = 0 * temp + 10
#         else:
#             # temp = temp * range_
#             temp = temp / optimal_threshold
#             temp = temp * 10
#             # temp = temp * max_val
#     naive_full_scaled[i, :] = temp

fig, ax = setup_plot(1, 1, figsize=(10, 7.5))

plot_predictions(
    df_validation,
    samples_full_scaled,
    naive_full_scaled,
    fig,
    ax,
    label_naive="Naive mechanistic model (vaccine coverage only)",
    label_statistical="Statistical model",
    show_r2=True,
)

plt.show()

# fig.savefig(
#     os.path.join(fig_dir, "figure_r01_performance_validation.png"),
#     bbox_inches="tight",
# )
