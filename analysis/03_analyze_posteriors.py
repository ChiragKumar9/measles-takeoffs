import os
import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt
import polars as pl
import yaml
from jax.random import PRNGKey
from matplotlib import rc
from numpyro.infer import Predictive

from outbreak_size import poisson

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
    df = pl.read_csv(
        os.path.join(data_dir, "outbreaks_vaccinations_cleaned.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
    )

    with open(
        os.path.join(
            output_dir,
            "samples_partial_pooled_poisson_just_controls.pkl",
        ),
        "rb",
    ) as file:
        posterior_samples = pickle.load(file)

    # Generate posterior predictive samples
    # make log of reasonably log scaled covariates
    df = df.with_columns(
        log_population=pl.col("pop_5_9").log(),
        log_income=pl.col("median_household_income").log(),
    )

    # make new variables
    df = df.with_columns(
        fraction_nonmed_exempt=pl.col("nonmed_exempt_pct")
        / (pl.col("nonmed_exempt_pct") + pl.col("med_exempt_pct")),
    )
    df = df.with_columns(
        fraction_nonmed_exempt=pl.col("fraction_nonmed_exempt").fill_nan(0.0)
    )

    # make year dummy variables
    df = df.sort("year", descending=True)
    df = df.to_dummies(columns="year", separator="_", drop_first=True)
    years = [y for y in df.columns if y.startswith("year_")]

    # set up model
    iv = df[
        ["estimated_mmr_cov", "total_exempt_pct", "fraction_nonmed_exempt"]
    ].to_jax()
    model = poisson.PartialPooledPoisson(
        iv=iv,
        covariates=df.select(
            [
                "fraction_public",
                "pop_density_5_9",
                "fraction_white",
                "fraction_graduate_degree",
                "log_population",
                "log_income",
                *years,
            ]
        ).to_jax(),
        counties=df["fips"],
    )

    predictive = Predictive(model.fit, posterior_samples)
    predictive_samples = predictive(PRNGKey(params["seed"]), cases=None)

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
        [0, max(df["cases_outbreak"].max(), median.max())],  # type: ignore
        [0, max(df["cases_outbreak"].max(), median.max())],  # type: ignore
        ls="--",
        c="black",
    )

    plt.show()

    # marginal effects of vaccine coverage based on whether there was an outbreak
    iv_effect = (
        posterior_samples["iv_global"][:, jnp.newaxis]
        + posterior_samples["iv_variabilities"][:, jnp.newaxis]
        * posterior_samples["iv_county_offsets"]
    )
    coverage_effect = iv_effect[:, :, 0]

    df_grouped = df.group_by("fips", maintain_order=True)

    county_ever_outbreak = df_grouped.agg(pl.max("cases_outbreak"))
    outbreak_ever = county_ever_outbreak["cases_outbreak"] >= 3
    outbreak_never = county_ever_outbreak["cases_outbreak"] == 0

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

    ax.hist(
        jnp.exp(coverage_effect[:, outbreak_never.to_jax()].flatten()),
        bins=1000,
        density=True,
        label="Counties without outbreaks",
        color="blue",
    )

    ax.hist(
        jnp.exp(coverage_effect[:, outbreak_ever.to_jax()].flatten()),
        bins=1000,
        density=True,
        label="Counties with outbreaks",
        color="red",
    )
    ax.set_xlabel("Marginal effect of increasing vaccination coverage")
    ax.legend()
    plt.show()

    mean_vaccination_coverage = df_grouped.agg(pl.mean("estimated_mmr_cov"))
    mean_exemptions = df_grouped.agg(pl.mean("nonmed_exempt_pct"))
    fraction_public_school = df_grouped.agg(pl.mean("fraction_public"))

    # vaccination coverage vs marginal effect
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

    median_vacc_effect = jnp.exp(jnp.median(coverage_effect, axis=0))
    lower = jnp.exp(jnp.quantile(coverage_effect, (1 - alpha) / 2, axis=0))
    upper = jnp.exp(
        jnp.quantile(coverage_effect, 1 - ((1 - alpha) / 2), axis=0)
    )

    ax.errorbar(
        mean_vaccination_coverage["estimated_mmr_cov"].filter(outbreak_never),
        median_vacc_effect[outbreak_never.to_jax()],
        yerr=[
            median_vacc_effect[outbreak_never.to_jax()]
            - lower[outbreak_never.to_jax()],
            upper[outbreak_never.to_jax()]
            - median_vacc_effect[outbreak_never.to_jax()],
        ],
        fmt="o",
        ecolor="blue",
        c="blue",
        capsize=0,
        alpha=0.4,
        label="Counties without an outbreak",
    )

    ax.errorbar(
        mean_vaccination_coverage["estimated_mmr_cov"].filter(outbreak_ever),
        median_vacc_effect[outbreak_ever.to_jax()],
        yerr=[
            median_vacc_effect[outbreak_ever.to_jax()]
            - lower[outbreak_ever.to_jax()],
            upper[outbreak_ever.to_jax()]
            - median_vacc_effect[outbreak_ever.to_jax()],
        ],
        fmt="o",
        ecolor="red",
        c="red",
        capsize=0,
        alpha=0.4,
        label="Counties with an outbreak",
    )

    ax.set_ylabel("Marginal effect of increasing vaccination coverage")
    ax.set_xlabel("Mean vaccination coverage (%)")
    ax.legend()
    plt.show()

    # vaccination coverage vs marginal effect
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

    ax.errorbar(
        mean_exemptions["nonmed_exempt_pct"].filter(outbreak_never),
        median_vacc_effect[outbreak_never.to_jax()],
        yerr=[
            median_vacc_effect[outbreak_never.to_jax()]
            - lower[outbreak_never.to_jax()],
            upper[outbreak_never.to_jax()]
            - median_vacc_effect[outbreak_never.to_jax()],
        ],
        fmt="o",
        ecolor="blue",
        c="blue",
        capsize=0,
        alpha=0.4,
        label="Counties without an outbreak",
    )

    ax.errorbar(
        mean_exemptions["nonmed_exempt_pct"].filter(outbreak_ever),
        median_vacc_effect[outbreak_ever.to_jax()],
        yerr=[
            median_vacc_effect[outbreak_ever.to_jax()]
            - lower[outbreak_ever.to_jax()],
            upper[outbreak_ever.to_jax()]
            - median_vacc_effect[outbreak_ever.to_jax()],
        ],
        fmt="o",
        ecolor="red",
        c="red",
        capsize=0,
        alpha=0.4,
        label="Counties with an outbreak",
    )

    ax.set_ylabel("Marginal effect of increasing vaccination coverage")
    ax.set_xlabel("Mean nonmedical exemptions (%)")
    ax.legend()
    plt.show()

    # public school fraction coverage vs marginal effect
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

    ax.errorbar(
        fraction_public_school["fraction_public"].filter(outbreak_never),
        median_vacc_effect[outbreak_never.to_jax()],
        yerr=[
            median_vacc_effect[outbreak_never.to_jax()]
            - lower[outbreak_never.to_jax()],
            upper[outbreak_never.to_jax()]
            - median_vacc_effect[outbreak_never.to_jax()],
        ],
        fmt="o",
        ecolor="blue",
        c="blue",
        capsize=0,
        alpha=0.4,
        label="Counties without an outbreak",
    )

    ax.errorbar(
        fraction_public_school["fraction_public"].filter(outbreak_ever),
        median_vacc_effect[outbreak_ever.to_jax()],
        yerr=[
            median_vacc_effect[outbreak_ever.to_jax()]
            - lower[outbreak_ever.to_jax()],
            upper[outbreak_ever.to_jax()]
            - median_vacc_effect[outbreak_ever.to_jax()],
        ],
        fmt="o",
        ecolor="red",
        c="red",
        capsize=0,
        alpha=0.4,
        label="Counties with an outbreak",
    )

    ax.set_ylabel("Marginal effect of increasing vaccination coverage")
    ax.set_xlabel("Fraction in public school")
    ax.legend()
    plt.show()

    if iv.shape[1] >= 2:
        exemption_effect = iv_effect[:, :, 1]

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

        ax.hist(
            jnp.exp(exemption_effect[:, outbreak_never.to_jax()].flatten()),
            bins=1000,
            density=True,
            label="Counties without outbreaks",
            color="blue",
        )

        ax.hist(
            jnp.exp(exemption_effect[:, outbreak_ever.to_jax()].flatten()),
            bins=1000,
            density=True,
            label="Counties with outbreaks",
            color="red",
        )

        # add a line at 1
        ax.axvline(1, ls="--", c="black")
        ax.set_ylabel("Density")
        ax.set_xlabel("Marginal effect of increasing vaccination exemptions")
        ax.legend()
        plt.show()

        median_exemption_effect = jnp.exp(jnp.median(exemption_effect, axis=0))
        lower_exemptions = jnp.exp(
            jnp.quantile(exemption_effect, (1 - alpha) / 2, axis=0)
        )
        upper_exemptions = jnp.exp(
            jnp.quantile(exemption_effect, 1 - ((1 - alpha) / 2), axis=0)
        )

        # vaccination coverage vs marginal effect
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

        ax.errorbar(
            mean_exemptions["nonmed_exempt_pct"].filter(outbreak_never),
            median_exemption_effect[outbreak_never.to_jax()],
            yerr=[
                median_exemption_effect[outbreak_never.to_jax()]
                - lower_exemptions[outbreak_never.to_jax()],
                upper_exemptions[outbreak_never.to_jax()]
                - median_exemption_effect[outbreak_never.to_jax()],
            ],
            fmt="o",
            ecolor="blue",
            c="blue",
            capsize=0,
            alpha=0.4,
            label="Counties without an outbreak",
        )

        ax.errorbar(
            mean_exemptions["nonmed_exempt_pct"].filter(outbreak_ever),
            median_exemption_effect[outbreak_ever.to_jax()],
            yerr=[
                median_exemption_effect[outbreak_ever.to_jax()]
                - lower_exemptions[outbreak_ever.to_jax()],
                upper_exemptions[outbreak_ever.to_jax()]
                - median_exemption_effect[outbreak_ever.to_jax()],
            ],
            fmt="o",
            ecolor="red",
            c="red",
            capsize=0,
            alpha=0.4,
            label="Counties with an outbreak",
        )

        ax.set_ylabel("Marginal effect of increasing vaccination exemptions")
        ax.set_xlabel("Mean nonmedical exemptions (%)")
        ax.legend()
        plt.show()

        # plot marginal effects against each other
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

        ax.errorbar(
            y=median_exemption_effect[outbreak_never.to_jax()],
            x=median_vacc_effect[outbreak_never.to_jax()],
            yerr=[
                median_exemption_effect[outbreak_never.to_jax()]
                - lower_exemptions[outbreak_never.to_jax()],
                upper_exemptions[outbreak_never.to_jax()]
                - median_exemption_effect[outbreak_never.to_jax()],
            ],
            xerr=[
                median_vacc_effect[outbreak_never.to_jax()]
                - lower[outbreak_never.to_jax()],
                upper[outbreak_never.to_jax()]
                - median_vacc_effect[outbreak_never.to_jax()],
            ],
            fmt="o",
            ecolor="blue",
            c="blue",
            capsize=0,
            alpha=0.4,
            label="Counties without an outbreak",
        )

        ax.errorbar(
            y=median_exemption_effect[outbreak_ever.to_jax()],
            x=median_vacc_effect[outbreak_ever.to_jax()],
            yerr=[
                median_exemption_effect[outbreak_ever.to_jax()]
                - lower_exemptions[outbreak_ever.to_jax()],
                upper_exemptions[outbreak_ever.to_jax()]
                - median_exemption_effect[outbreak_ever.to_jax()],
            ],
            xerr=[
                median_vacc_effect[outbreak_ever.to_jax()]
                - lower[outbreak_ever.to_jax()],
                upper[outbreak_ever.to_jax()]
                - median_vacc_effect[outbreak_ever.to_jax()],
            ],
            fmt="o",
            ecolor="red",
            c="red",
            capsize=0,
            alpha=0.4,
            label="Counties with an outbreak",
        )

        ax.set_ylabel("Marginal effect of increasing vaccination exemptions")
        ax.set_xlabel("Marginal effect of increasing vaccination coverage")
        ax.legend()
        ax.set_yscale("log")
        plt.show()

        # how do the exemption effects relate to public school fraction and mobility
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

        ax.errorbar(
            fraction_public_school["fraction_public"].filter(outbreak_never),
            median_exemption_effect[outbreak_never.to_jax()],
            yerr=[
                median_exemption_effect[outbreak_never.to_jax()]
                - lower_exemptions[outbreak_never.to_jax()],
                upper_exemptions[outbreak_never.to_jax()]
                - median_exemption_effect[outbreak_never.to_jax()],
            ],
            fmt="o",
            ecolor="blue",
            c="blue",
            capsize=0,
            alpha=0.4,
            label="Counties without an outbreak",
        )

        ax.errorbar(
            fraction_public_school["fraction_public"].filter(outbreak_ever),
            median_exemption_effect[outbreak_ever.to_jax()],
            yerr=[
                median_exemption_effect[outbreak_ever.to_jax()]
                - lower_exemptions[outbreak_ever.to_jax()],
                upper_exemptions[outbreak_ever.to_jax()]
                - median_exemption_effect[outbreak_ever.to_jax()],
            ],
            fmt="o",
            ecolor="red",
            c="red",
            capsize=0,
            alpha=0.4,
            label="Counties with an outbreak",
        )

        ax.set_ylabel("Marginal effect of increasing vaccination exemptions")
        ax.set_xlabel("Fraction in public school")
        ax.legend()
        plt.show()
