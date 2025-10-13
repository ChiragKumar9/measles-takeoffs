import argparse
import os
import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml
from matplotlib import rc
from scipy.stats import beta

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input_stub",
    help="Input stub of the pickle file (e.g., 'within_county_Re_varies')",
    type=str,
    default="within_county_coverage_varies",
)

args = parser.parse_args()
input_stub = args.input_stub

plt.rcParams["figure.figsize"] = [9, 9]
font = {"family": "Nimbus Roman", "size": 28}
rc("font", **font)

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)

# read analysis params yaml
with open("params.yml", "r") as f:
    params = yaml.safe_load(f)

if __name__ == "__main__":
    # read data
    df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
    )

    with open(
        os.path.join(
            output_dir,
            f"samples_{input_stub}.pkl",
        ),
        "rb",
    ) as file:
        posterior_samples = pickle.load(file)

    with open(
        os.path.join(
            output_dir,
            f"predictive_{input_stub}.pkl",
        ),
        "rb",
    ) as file:
        predictive_samples = pickle.load(file)

    estimated_cases = predictive_samples["cases"]
    alpha = 0.95
    median = jnp.median(estimated_cases, axis=0)
    lower = jnp.quantile(estimated_cases, (1 - alpha) / 2, axis=0)
    upper = jnp.quantile(estimated_cases, 1 - ((1 - alpha) / 2), axis=0)

    # plot
    fig, ax = plt.subplots(1, 1)
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
    ax.errorbar(
        df["cases_outbreak"],
        median,
        yerr=[median - lower, upper - median],
        fmt="o",
        ecolor="blue",
        c="blue",
        capsize=0,
        alpha=0.4,
    )
    ax.set_xlabel("Observed cases")
    ax.set_ylabel("Fitted values")
    ax.set_xscale("symlog")
    ax.set_yscale("symlog")
    ax.plot(
        [0, 1.05 * max(df["cases_outbreak"].max(), median.max())],  # type: ignore
        [0, 1.05 * max(df["cases_outbreak"].max(), median.max())],  # type: ignore
        ls="--",
        c="black",
    )
    ax.set_xlim(-0.3, 1.1 * max(df["cases_outbreak"].max(), median.max()))  # type: ignore
    ax.set_ylim(-0.3, 1.1 * max(df["cases_outbreak"].max(), median.max()))  # type: ignore

    plt.show()
    # Report R2
    print(jnp.polyfit(median, df["cases_outbreak"].to_jax(), 1))
    print(jnp.corrcoef(df["cases_outbreak"].to_jax(), median)[0, 1] ** 2)
    print(
        jnp.corrcoef(
            jnp.log10(df["cases_outbreak"].to_jax() + 1), jnp.log10(median + 1)
        )[0, 1]
        ** 2
    )
    # Report out the percent of samples within the 95% credible interval
    within_interval = jnp.mean(
        (df["cases_outbreak"].to_jax() >= lower)
        & (df["cases_outbreak"].to_jax() <= upper)
    )
    print(
        f"Percent of observations within 95% credible interval: {within_interval * 100:.1f}%"
    )

    # Validation
    # read in outbreak clusters data
    df_clusters = pl.read_csv(
        os.path.join(data_dir, "outbreaks_in_clusters.csv"),
        schema_overrides={"fips": pl.String},
    )

    # merge with main df on fips and year
    df = df.join(df_clusters, on=["fips", "year"], how="left")
    # plot inferred reported coverage and inferred coverage and assess correlation

    # inferred coverage
    inferred_coverage = posterior_samples["inferred_coverage"] * 100
    median = jnp.median(inferred_coverage, axis=0)
    lower = jnp.quantile(inferred_coverage, (1 - alpha) / 2, axis=0)
    upper = jnp.quantile(inferred_coverage, 1 - ((1 - alpha) / 2), axis=0)

    fig, ax = plt.subplots(1, 1)
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
    ax.errorbar(
        df["cluster_coverage"],
        median,
        yerr=[median - lower, upper - median],
        fmt="^",
        ecolor="red",
        c="red",
        capsize=0,
        alpha=1.0,
    )

    ax.set_xlabel("Reported outbreak cluster coverage (%)")
    ax.set_ylabel("Estimated outbreak cluster coverage (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend()

    plt.show()
    # calculate correlation
    valid = df["cluster_coverage"].is_not_null()
    print(
        jnp.corrcoef(
            df["cluster_coverage"].to_jax()[valid.to_jax()],
            median[valid.to_jax()],
        )[0, 1]
        ** 2
    )
    # omit the 77% coverage point
    valid = valid & (df["cluster_coverage"] != 77)
    print(
        jnp.corrcoef(
            df["cluster_coverage"].to_jax()[valid.to_jax()],
            median[valid.to_jax()],
        )[0, 1]
        ** 2
    )

    # Plot within-county coverage vs overall coverage

    # Underlying beta distributions
    phi = predictive_samples["concentration"]
    beta_alpha = 1.0 + df["estimated_mmr_cov"].to_jax() / 100 * phi
    beta_beta = 1.0 + (1 - df["estimated_mmr_cov"].to_jax() / 100) * phi
    beta_frozen = beta(beta_alpha, beta_beta)
    underlying_median = jnp.median(beta_frozen.median(), axis=0) * 100
    underlying_lower = (
        jnp.median(beta_frozen.ppf((1 - alpha) / 2), axis=0) * 100
    )
    underlying_upper = (
        jnp.median(beta_frozen.ppf(1 - ((1 - alpha) / 2)), axis=0) * 100
    )

    fig, ax = plt.subplots(1, 1)
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
    ax.errorbar(
        df["estimated_mmr_cov"],
        underlying_median,
        yerr=[
            underlying_median - underlying_lower,
            underlying_upper - underlying_median,
        ],
        fmt="o",
        ecolor="blue",
        c="blue",
        capsize=0,
        alpha=0.2,
        label="Estimated sampling coverage",
    )
    ax.errorbar(
        df["estimated_mmr_cov"],
        median,
        yerr=[median - lower, upper - median],
        fmt="^",
        ecolor="red",
        c="red",
        capsize=0,
        alpha=1.0,
        label="Estimated outbreak cluster coverage",
    )
    ax.plot(
        [70, 100],
        [70, 100],
        ls="--",
        c="black",
    )
    ax.set_xlabel("Reported county-level coverage (%)")
    ax.set_ylabel("Estimated within-county coverage (%)")
    ax.set_xlim(70, 100)
    ax.set_ylim(0, 100)
    ax.legend()

    plt.show()

    # plot the two against each other
    fig, ax = plt.subplots(1, 1)
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
    ax.scatter(underlying_median, median, c="blue")
    ax.errorbar(
        underlying_median,
        median,
        xerr=[
            underlying_median - underlying_lower,
            underlying_upper - underlying_median,
        ],
        yerr=[median - lower, upper - median],
        fmt="o",
        ecolor="blue",
        c="blue",
        capsize=0,
        alpha=0.2,
    )

    ax.plot(
        [0, 100],
        [0, 100],
        ls="--",
        c="black",
    )
    ax.set_xlabel("Estimated sampling coverage (%)")
    ax.set_ylabel("Estimated outbreak cluster coverage (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend()

    plt.show()

    # histogram of inferred coverage based on whether the county had an outbreak
    fig, ax = plt.subplots(1, 1)
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
    ax.hist(
        inferred_coverage[:, df["cases_outbreak"].to_jax() > 0].flatten(),
        bins=100,
        color="red",
        alpha=0.5,
        density=True,
        label="Counties with outbreaks",
    )
    ax.hist(
        inferred_coverage[:, df["cases_outbreak"].to_jax() == 0].flatten(),
        bins=100,
        color="blue",
        alpha=0.5,
        density=True,
        label="Counties without outbreaks",
    )
    ax.set_xlabel("Estimated outbreak cluster coverage (%)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 100)
    ax.legend()
    plt.show()

    # report out some statistics
    print(
        "Statistics for inferred coverage among counties without an outbreak:"
    )
    no_outbreak = inferred_coverage[
        :, df["cases_outbreak"].to_jax() == 0
    ].flatten()
    no_outbreak_median = jnp.median(no_outbreak)
    no_outbreak_lower = jnp.quantile(no_outbreak, (1 - alpha) / 2)
    no_outbreak_upper = jnp.quantile(no_outbreak, 1 - ((1 - alpha) / 2))
    print(
        f"  Median: {no_outbreak_median:.1f} ({no_outbreak_lower:.1f}, {no_outbreak_upper:.1f})"
    )

    # variance in outbreak cluster coverage across repeated counties
    df = df.with_columns(median_inferred_outbreak_coverage=np.array(median))
    df_grouped = df.group_by("fips").agg(
        std_inferred_coverage=pl.col("median_inferred_outbreak_coverage").std()
    )
    # plot
    fig, ax = plt.subplots(1, 1)
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
    ax.hist(
        df_grouped["std_inferred_coverage"],
        bins=20,
        color="blue",
        alpha=0.5,
        density=True,
    )
    ax.set_xlabel(
        "Standard deviation in estimated outbreak cluster coverage (%)"
    )
    ax.set_ylabel("Density")
    plt.show()

    # inferred coverage and outbreak size
    fig, ax = plt.subplots(1, 1)
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
    ax.errorbar(
        median,
        df["cases_outbreak"],
        xerr=[median - lower, upper - median],
        fmt="^",
        ecolor="red",
        c="red",
        capsize=0,
        alpha=1.0,
    )
    ax.set_yscale("symlog")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Estimated outbreak cluster coverage (%)")
    ax.set_ylabel("Reported cases")
    plt.show()

    # variance in coverage and outbreak size
    fig, ax = plt.subplots(1, 1)
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
    ax.errorbar(
        underlying_median,
        df["cases_outbreak"],
        xerr=[
            underlying_median - underlying_lower,
            underlying_upper - underlying_median,
        ],
        fmt="o",
        ecolor="blue",
        c="blue",
        capsize=0,
        alpha=1.0,
    )
    ax.set_yscale("symlog")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Estimated sampling coverage (%)")
    ax.set_ylabel("Reported cases")

    plt.show()

    # Report effect of coefficients on variance
    if "Re" in input_stub:
        coefs_to_report = ["log_baseline_concentration", "regression_betas"]
    elif "coverage" in input_stub:
        coefs_to_report = [
            "log_baseline_concentration",
            "log_variance_coverage_effect",
            "log_variance_exemption_effect",
            "log_variance_nonmed_exempt_effect",
            "log_variance_population_effect",
        ]
    else:
        assert False, "Don't know what coefficients to grab from this model."
    for coef in coefs_to_report:
        samples = posterior_samples[coef]
        median = jnp.median(samples, axis=0)
        lower = jnp.quantile(samples, (1 - alpha) / 2, axis=0)
        upper = jnp.quantile(samples, 1 - ((1 - alpha) / 2), axis=0)
        print(f"Coefficient: {coef}")
        if median.shape == ():
            print(f"  {median:.3f} ({lower:.3f}, {upper:.3f})")
        else:
            for i in range(median.shape[0]):
                print(
                    f"  Covariate {i}: {median[i]:.3f} ({lower[i]:.3f}, {upper[i]:.3f})"
                )

    # report regression coefficients -- in particular, effect of vaccinations
    if "coverage" in input_stub:
        for coef in [
            "inferred_coverage_effect",
            "baseline",
            "regression_betas",
        ]:
            samples = posterior_samples[coef]
            median = jnp.median(samples, axis=0)
            lower = jnp.quantile(samples, (1 - alpha) / 2, axis=0)
            upper = jnp.quantile(samples, 1 - ((1 - alpha) / 2), axis=0)
            print(f"Coefficient: {coef}")
            if median.shape == ():
                print(f"  {median:.3f} ({lower:.3f}, {upper:.3f})")
            else:
                for i in range(median.shape[0]):
                    print(
                        f"  Covariate {i}: {median[i]:.3f} ({lower[i]:.3f}, {upper[i]:.3f})"
                    )

    # report R0 distribution
    if "Re" in input_stub:
        r0_samples = posterior_samples["r0"]
        median = jnp.median(r0_samples)
        lower = jnp.quantile(r0_samples, (1 - alpha) / 2)
        upper = jnp.quantile(r0_samples, 1 - ((1 - alpha) / 2))
        print(f"R0: {median:.3f} ({lower:.3f}, {upper:.3f})")

        # Plot histogram
        fig, ax = plt.subplots(1, 1)
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
        ax.hist(r0_samples, bins=100, color="blue", alpha=1.0, density=True)
        ax.set_xlabel(r"$R_0$")
        ax.set_ylabel("Density")
        plt.show()
