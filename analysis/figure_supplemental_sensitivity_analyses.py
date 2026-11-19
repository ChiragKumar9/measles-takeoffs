import os

from numpyro import enable_x64

enable_x64()

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml
from matplotlib import rc

from outbreak_size.coverage_effect import (
    covariate_sd,
    effect_label,
    marginal_percent_change_per_pp,
    percent_change_per_pp,
)
from outbreak_size.model_colors import (
    COVARIATE_SET_COLORS,
    EARLIER_PERIOD_COLOR,
    LATER_PERIOD_COLOR,
    POPULATION_COLOR,
)
from outbreak_size.outbreak_clusters import (
    CLUSTER_NAMES,
    CLUSTER_TAGS,
    cluster_seeds_2025,
    drop_places_of_year,
    outbreak_cases_all_counties,
    places_in_cluster,
    region_members,
)
from outbreak_size.samples_io import load_samples

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)
fig_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")

font = {"family": "Nimbus Roman", "size": 28}
rc("font", **font)

# read analysis params yaml
with open("params.yml", "r") as f:
    params = yaml.safe_load(f)
# the scale of the vaccination covariates of the fits (see params.yml)
COVARIATE_SCALE = params["covariate_scale"]
MMR_EFFECTIVENESS = params.get("mmr_effectiveness")

# (the years of a fit, its colour, its label)
PERIODS = [
    (
        [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
        EARLIER_PERIOD_COLOR,
        "2011-2020",
    ),
    ([2021, 2022, 2023, 2024, 2025], LATER_PERIOD_COLOR, "2021-2025"),
]
FIT_PREFIX = "partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_"
# the fits of 02 that estimate the effectiveness of the MMR vaccine
FIT_EFFECTIVENESS_TAG = "_fit_population_immunity"


def setup_plot(nrows, ncols):
    fig, ax = plt.subplots(nrows, ncols, figsize=(30, 7.5 * nrows))
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


def plot_immunity_effects_on_vaccine_coverage(df, fig, ax):
    df = df.sort("year", maintain_order=True)
    # the % change per 1pp is averaged over the county-years of all the years
    reference_rows = df

    df = df.with_columns(
        previous_outbreak_size=pl.cum_sum("cases_outbreak").over(
            ["county", "state"]
        )
    ).with_columns(
        # want to remove the current year's outbreak size
        previous_outbreak_size=pl.col("previous_outbreak_size")
        - pl.col("cases_outbreak")
    )

    df = df.filter(pl.col("year") >= 2021)

    medians = []
    lowers = []
    uppers = []

    posterior_samples = load_samples(
        os.path.join(
            output_dir,
            "samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_[2021, 2022, 2023, 2024, 2025].npz",
        )
    )
    samples_of_interest = posterior_samples["iv_global"][:, 0]
    # per SD of the covariate -> % change per 1pp (see coverage_effect)
    samples_of_interest = percent_change_per_pp(
        samples_of_interest,
        df,
        "estimated_mmr_cov",
        reference_rows,
        COVARIATE_SCALE,
        MMR_EFFECTIVENESS,
    )
    print(f"Baseline p-value: {jnp.mean(samples_of_interest > 0):.3f}")
    medians.append(jnp.median(samples_of_interest))
    lowers.append(jnp.percentile(samples_of_interest, 2.5))
    uppers.append(jnp.percentile(samples_of_interest, 97.5))

    posterior_samples = load_samples(
        os.path.join(
            output_dir,
            "samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_[2021, 2022, 2023, 2024, 2025]_no_previous_outbreak.npz",
        )
    )
    samples_of_interest = posterior_samples["iv_global"][:, 0]
    # per SD of the covariate (of the rows of that fit) -> % change per 1pp
    samples_of_interest = percent_change_per_pp(
        samples_of_interest,
        df.filter(pl.col("previous_outbreak_size") == 0),
        "estimated_mmr_cov",
        reference_rows,
        COVARIATE_SCALE,
        MMR_EFFECTIVENESS,
    )
    print(f"No prior cases p-value: {jnp.mean(samples_of_interest > 0):.3f}")
    medians.append(jnp.median(samples_of_interest))
    lowers.append(jnp.percentile(samples_of_interest, 2.5))
    uppers.append(jnp.percentile(samples_of_interest, 97.5))

    posterior_samples = load_samples(
        os.path.join(
            output_dir,
            "samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_[2021, 2022, 2023, 2024, 2025]_previous_outbreak_size.npz",
        )
    )
    samples_of_interest = posterior_samples["iv_global"][:, 0]
    # per SD of the covariate -> % change per 1pp (see coverage_effect)
    samples_of_interest = percent_change_per_pp(
        samples_of_interest,
        df,
        "estimated_mmr_cov",
        reference_rows,
        COVARIATE_SCALE,
        MMR_EFFECTIVENESS,
    )
    print(
        f"Covariate for previous cases p-value: {jnp.mean(samples_of_interest > 0):.3f}"
    )
    medians.append(jnp.median(samples_of_interest))
    lowers.append(jnp.percentile(samples_of_interest, 2.5))
    uppers.append(jnp.percentile(samples_of_interest, 97.5))

    x = np.arange(3)
    ax.errorbar(
        x,
        medians,
        yerr=[
            np.array(medians) - np.array(lowers),
            np.array(uppers) - np.array(medians),
        ],
        fmt="o",
        color="black",
        capsize=0,
    )
    ax.axhline(y=0, color="black", linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "Baseline",
            "No prior\ncases",
            "Covariate for\nprevious cases",
        ]
    )
    ax.set_ylabel(effect_label())


def plot_previous_outbreak_size_effects(df, fig, ax):
    df = df.sort("year", maintain_order=True)

    df = df.with_columns(
        previous_outbreak_size=pl.cum_sum("cases_outbreak").over(
            ["county", "state"]
        )
    ).with_columns(
        # want to remove the current year's outbreak size
        previous_outbreak_size=pl.col("previous_outbreak_size")
        - pl.col("cases_outbreak")
    )

    df = df.filter(pl.col("year") >= 2021)
    covariate = df["previous_outbreak_size"].drop_nulls().to_numpy()
    std = jnp.std(covariate)
    posterior_samples = load_samples(
        os.path.join(
            output_dir,
            "samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_[2021, 2022, 2023, 2024, 2025]_previous_outbreak_size.npz",
        )
    )
    # regression_betas follow the model's regression covariates (the variables
    # after the three pooled coverage/exemption ivs), then the year dummies and
    # the state fixed effects, so the last coefficient is a state effect. The
    # model's variables after the ivs are the nine below and then the previous
    # outbreak size
    regression_covariates = [
        "fraction_under_5",
        "fraction_school_aged",
        "avg_house_size",
        "std_house_size",
        "fraction_public",
        "fraction_white",
        "fraction_graduate_degree",
        "log_population_density",
        "log_income",
        "previous_outbreak_size",
    ]
    previous_outbreak_index = regression_covariates.index(
        "previous_outbreak_size"
    )
    samples_of_interest = posterior_samples["regression_betas"][
        :, previous_outbreak_index
    ]
    samples_of_interest = samples_of_interest / std
    # make a percent change
    samples_of_interest = jnp.exp(samples_of_interest) * 100 - 100
    print(
        f"Covariate for previous cases p-value: {jnp.mean(jnp.abs(samples_of_interest) > 0)}"
    )
    # plot the effect
    ax.hist(
        samples_of_interest,
        bins=100,
        density=True,
        color="gray",
        alpha=0.7,
    )

    ax.axvline(
        x=jnp.percentile(samples_of_interest, 2.5),
        color="red",
    )
    ax.axvline(
        x=jnp.percentile(samples_of_interest, 97.5),
        color="red",
    )
    ax.axvline(
        x=jnp.median(samples_of_interest),
        color="black",
    )
    ax.axvline(x=0, color="blue", linestyle="--")

    ax.set_xlabel(
        "% change in outbreak size per\n1pp increase in previous outbreak size"
    )
    ax.set_ylabel("Density")
    ax.set_xticks([-5, 0, 5])


def plot_medical_exemption_coefficients(df, fig, ax):
    # the % change per 1pp is averaged over the county-years of all the years
    reference_rows = df
    df_old = df.filter(pl.col("year") <= 2020)
    df_new = df.filter(pl.col("year") >= 2021)
    # we want to plot the baseline me coefficient, the coefficient
    # when we subset to <10% and <5%
    medians_old = []
    lowers_old = []
    uppers_old = []
    medians_new = []
    lowers_new = []
    uppers_new = []
    for idx, years in enumerate(
        [
            [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
            [2021, 2022, 2023, 2024, 2025],
        ]
    ):
        if idx == 0:
            df = df_old
            medians = medians_old
            lowers = lowers_old
            uppers = uppers_old
        else:
            df = df_new
            medians = medians_new
            lowers = lowers_new
            uppers = uppers_new
        for me_cutoff in [None, 10.0, 5.0]:
            if me_cutoff is not None:
                additional_info = f"_me_cutoff={me_cutoff}"
                fit_rows = df.filter(pl.col("med_exempt_pct") <= me_cutoff)
            else:
                additional_info = ""
                fit_rows = df
            posterior_samples = load_samples(
                os.path.join(
                    output_dir,
                    f"samples_partial_pooled_poisson_state_fixed_effect=True_pool=True_five_year=False_CAR=True_All covariates_{years}{additional_info}.npz",
                )
            )
            samples_of_interest = posterior_samples["iv_global"][:, -1]
            samples_of_interest = percent_change_per_pp(
                samples_of_interest,
                fit_rows,
                "med_exempt_pct",
                reference_rows,
                COVARIATE_SCALE,
                MMR_EFFECTIVENESS,
            )
            medians.append(jnp.median(samples_of_interest))
            lowers.append(jnp.percentile(samples_of_interest, 2.5))
            uppers.append(jnp.percentile(samples_of_interest, 97.5))

    x = np.arange(3)
    ax.errorbar(
        x - 0.1,  # type: ignore
        medians_old,
        yerr=[
            np.array(medians_old) - np.array(lowers_old),
            np.array(uppers_old) - np.array(medians_old),
        ],
        fmt="o",
        label="2011-2020",
        color=EARLIER_PERIOD_COLOR,
        capsize=0,
    )
    ax.errorbar(
        x + 0.1,
        medians_new,
        yerr=[
            np.array(medians_new) - np.array(lowers_new),
            np.array(uppers_new) - np.array(medians_new),
        ],
        fmt="o",
        label="2021-2025",
        color=LATER_PERIOD_COLOR,
        capsize=0,
    )
    ax.axhline(y=0, color="black", linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(["All data", "ME ≤10%", "ME ≤5%"])
    ax.set_ylabel(
        effect_label("exemption").replace("exemptions", "medical exemptions")
    )
    ax.legend(loc=(0.03, 0.78), edgecolor="black")


def check_fit_rows(fit_rows, years, tag):
    """
    The rows rebuilt here are those the fit used: as many as the cases of its
    predictive file (the SD that turns a coefficient into % per 1pp is that of
    these rows).
    """
    predictive = load_samples(
        os.path.join(output_dir, f"predictive_{FIT_PREFIX}{years}{tag}.npz")
    )
    assert predictive["cases"].shape[1] == fit_rows.height, (
        f"{years}{tag}: the fit has {predictive['cases'].shape[1]} rows, "
        f"the rebuilt rows {fit_rows.height}"
    )


def plot_neighbors_significance(df, fig, ax):
    # the % change per 1pp is averaged over the county-years of all the years
    reference_rows = df
    # regression_betas follow the model's regression covariates (the variables
    # after the three pooled coverage/exemption ivs), then the year dummies and
    # the state fixed effects, so the last coefficient is a state effect. For
    # the all-covariates model the variables after the ivs are the nine below
    # and then the neighbors' coverage
    regression_covariates = [
        "fraction_under_5",
        "fraction_school_aged",
        "avg_house_size",
        "std_house_size",
        "fraction_public",
        "fraction_white",
        "fraction_graduate_degree",
        "log_population_density",
        "log_income",
        "neighbors_coverage",
    ]
    neighbors_index = regression_covariates.index("neighbors_coverage")
    for years, color, label in PERIODS:
        # the fit has the county-years of the period with the neighbors' coverage
        fit_rows = df.filter(
            pl.col("year").is_in(years)
            & pl.col("neighbors_coverage").is_not_null()
        )
        check_fit_rows(fit_rows, years, "_use_neighbors")
        posterior_samples = load_samples(
            os.path.join(
                output_dir,
                f"samples_{FIT_PREFIX}{years}_use_neighbors.npz",
            )
        )
        samples_of_interest = posterior_samples["regression_betas"][
            :, neighbors_index
        ]
        samples_of_interest = percent_change_per_pp(
            samples_of_interest,
            fit_rows,
            "neighbors_coverage",
            reference_rows,
            COVARIATE_SCALE,
            MMR_EFFECTIVENESS,
        )
        low, median, high = np.percentile(samples_of_interest, [2.5, 50, 97.5])
        print(
            f"Neighbors' coverage effect {label}: {median:.3f} [{low:.3f}, {high:.3f}]"
        )

        # plot the effect, the median (solid) and 95% interval (dotted) of the
        # period in its colour
        ax.hist(
            samples_of_interest,
            bins=100,
            density=True,
            color=color,
            alpha=0.5,
            label=label,
        )
        ax.axvline(x=median, color=color)
        ax.axvline(x=low, color=color, linestyle=":")
        ax.axvline(x=high, color=color, linestyle=":")
    ax.axvline(x=0, color="black", linestyle="--")

    ax.set_xlabel(
        effect_label().replace(
            "MMR vaccine coverage", "MMR coverage\nin neighboring counties"
        )
    )
    ax.set_ylabel("Density")
    ax.legend(edgecolor="black")


def plot_overdispersion_posterior(fig, ax):
    """
    The posterior of the overdispersion of the negative binomial of the
    baseline All covariates fit of each period: 1 / the concentration the
    model samples, so that variance = mean + overdispersion * mean**2 (0 is a
    Poisson, and the larger it is, the more overdispersed the outbreak sizes
    are).
    """
    overdispersions = {}
    for years, _, label in PERIODS:
        concentration = np.asarray(
            load_samples(
                os.path.join(output_dir, f"samples_{FIT_PREFIX}{years}.npz")
            )["nb_concentration"]
        )
        overdispersions[label] = 1 / concentration
        lower, median, upper = np.percentile(
            overdispersions[label], [2.5, 50, 97.5]
        )
        print(
            f"Overdispersion (1 / concentration) {label}: {median:.2f} "
            f"[{lower:.2f}, {upper:.2f}] (concentration "
            f"{np.median(concentration):.2f})"
        )
    # the same bins for both periods, up to the 99.5th percentile of the larger
    high = max(np.percentile(o, 99.5) for o in overdispersions.values())
    for (_, color, label), draws in zip(PERIODS, overdispersions.values()):
        ax.hist(
            draws,
            bins=np.linspace(0, high, 41),
            density=True,
            color=color,
            alpha=0.5,
            label=label,
        )
    ax.set_xlim(0, high)
    ax.set_xlabel(
        "Overdispersion of outbreak size\n(1 / negative binomial "
        "concentration;\n0 = Poisson, larger = more overdispersed)"
    )
    ax.set_ylabel("Density")
    ax.legend(edgecolor="black", loc="upper left")


def plot_effectiveness_coverage_effect(df, fig, ax):
    """
    The effect of the coverage in each period (panel c of the main figure of
    the changing predictors), from the baseline fit with the effectiveness
    fixed and from the fit that estimates it.
    """
    # units: the elasticity is d log(outbreak size) / d log(susceptible
    # fraction), the effectiveness a fraction, the coverage a percent, and the
    # result a % change in outbreak size per 1pp of coverage. Toy case: at 90%
    # coverage and effectiveness 0.97 the susceptible fraction goes from 0.127
    # to 0.1173 with 1pp more, so an elasticity of 1 gives (0.1173 / 0.127 - 1)
    # = -7.6378%, with the effectiveness as a number or as one per draw
    toy = -100 * (1 - 0.1173 / 0.127)
    for effectiveness in (0.97, np.array([[0.97]])):
        np.testing.assert_allclose(
            marginal_percent_change_per_pp(
                np.array([1.0]), np.array([90.0]), "natural", effectiveness
            ),
            [toy],
        )

    fitted_samples = {
        tuple(years): load_samples(
            os.path.join(
                output_dir,
                f"samples_{FIT_PREFIX}{years}{FIT_EFFECTIVENESS_TAG}.npz",
            )
        )
        for years, _, _ in PERIODS
    }
    # the estimated effectiveness of each period, printed here and in the panel
    estimated_effectiveness = []
    for years, _, label in PERIODS:
        lower, median, upper = np.percentile(
            fitted_samples[tuple(years)]["mmr_effectiveness"], [2.5, 50, 97.5]
        )
        estimated_effectiveness.append(
            f"{label}: {median:.2f} [{lower:.2f}, {upper:.2f}]"
        )
        print(
            f"MMR effectiveness {label}: {median:.3f} [{lower:.3f}, {upper:.3f}]"
        )
    # The % change per 1pp is averaged over the county-years of all the years,
    # but 1pp more coverage leaves a positive susceptible fraction only if
    # effectiveness * (coverage + 1) < 100, which a draw of the highest
    # effectiveness does not for the counties near 100% coverage (their power
    # is then undefined). Both series of this panel therefore average over the
    # county-years where it holds for every draw (the same ones, so that they
    # compare), unlike panel c of the main figure, which has all of them.
    max_effectiveness = max(
        float(np.max(s["mmr_effectiveness"])) for s in fitted_samples.values()
    )
    reference_rows = df.filter(
        (pl.col("estimated_mmr_cov") + 1) * max_effectiveness < 100
    )
    print(
        f"Coverage effect with the effectiveness estimated: averaged over "
        f"{reference_rows.height} of the {df.height} county-years (the others "
        f"are within 1pp of 100% susceptibility at the largest effectiveness "
        f"drawn, {max_effectiveness:.3f})"
    )
    reference_coverage = reference_rows["estimated_mmr_cov"].to_numpy()
    effects = {"fixed": [], "fitted": []}
    for years, _, label in PERIODS:
        rows = df.filter(pl.col("year").is_in(years))
        baseline = load_samples(
            os.path.join(output_dir, f"samples_{FIT_PREFIX}{years}.npz")
        )
        effects["fixed"].append(
            percent_change_per_pp(
                baseline["iv_global"][:, 0],
                rows,
                "estimated_mmr_cov",
                reference_rows,
                COVARIATE_SCALE,
                MMR_EFFECTIVENESS,
            )
        )
        fitted = fitted_samples[tuple(years)]
        # the coefficient is per SD of the log susceptible fraction at the
        # fixed effectiveness (what the fit standardizes with), so the
        # elasticity is the coefficient / that SD; the % change per 1pp then
        # uses the effectiveness of each draw
        elasticity = np.asarray(fitted["iv_global"][:, 0]) / covariate_sd(
            rows, "estimated_mmr_cov", "natural", MMR_EFFECTIVENESS
        )
        effects["fitted"].append(
            marginal_percent_change_per_pp(
                elasticity,
                reference_coverage,
                "natural",
                np.asarray(fitted["mmr_effectiveness"])[:, None],
            )
        )
        for name, draws in effects.items():
            low, median, high = np.percentile(draws[-1], [2.5, 50, 97.5])
            print(
                f"Coverage effect, effectiveness {name}, {label}: {median:.3f} [{low:.3f}, {high:.3f}]"
            )

    ax.axhline(y=0, color="k", linestyle="--")
    for (name, label, color), offset in zip(
        [
            (
                "fixed",
                f"Effectiveness fixed at {MMR_EFFECTIVENESS}",
                COVARIATE_SET_COLORS["All covariates"],
            ),
            ("fitted", "Effectiveness estimated", POPULATION_COLOR),
        ],
        [-0.035, 0.035],
    ):
        summaries = np.array(
            [np.percentile(d, [2.5, 50, 97.5]) for d in effects[name]]
        )
        ax.errorbar(
            [0 + offset, 1 + offset],
            summaries[:, 1],
            yerr=[
                summaries[:, 1] - summaries[:, 0],
                summaries[:, 2] - summaries[:, 1],
            ],
            fmt="o",
            label=label,
            color=color,
            linewidth=3,
            markersize=10,
        )
    ax.set_xticks([0, 1])
    ax.set_xticklabels([label for _, _, label in PERIODS])
    ax.set_xlim([-0.18, 1.18])
    # leave the top 30% of the axes to the legend, so it covers no estimate
    low, high = ax.get_ylim()
    ax.set_ylim(low, low + (high - low) / 0.7)
    ax.set_xlabel("Year")
    ax.set_ylabel(effect_label())
    ax.legend(loc="upper center", edgecolor="black")
    # the effectiveness estimated in each period (median and 95% interval),
    # below the points, between the two periods
    ax.text(
        0.5,
        0.03,
        "Estimated effectiveness\n" + "\n".join(estimated_effectiveness),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=20,
    )


def plot_leave_one_cluster_out(df, leave_out, fig, ax):
    """
    The effect of the coverage in 2021-2025 (All covariates) in the fit on all
    the county-years and in the fits (02) that leave out the places of one
    cluster of 2025 outbreaks each. `leave_out` is {cluster name: the places
    left out of 2025}. The % change per 1pp is averaged over all the
    county-years of `df`, as in the other panels.
    """
    years = [2021, 2022, 2023, 2024, 2025]
    period_rows = df.filter(pl.col("year").is_in(years))
    # (label, the tag of the fit's files, the rows the fit was made on)
    fits = [("All places", "", period_rows)] + [
        (
            f"Without\n{name}",
            f"_leave_out_{CLUSTER_TAGS[name]}",
            drop_places_of_year(period_rows, places, 2025),
        )
        for name, places in leave_out.items()
    ]
    summaries = []
    for label, tag, fit_rows in fits:
        fit_name = f"{FIT_PREFIX}{years}{tag}"
        # the rows rebuilt here are those the fit used (the SD that turns the
        # coefficient into % per 1pp is that of these rows)
        predictive = load_samples(
            os.path.join(output_dir, f"predictive_{fit_name}.npz")
        )
        assert predictive["cases"].shape[1] == fit_rows.height, (
            f"{fit_name}: the fit has {predictive['cases'].shape[1]} rows, "
            f"the rebuilt rows {fit_rows.height}"
        )
        posterior_samples = load_samples(
            os.path.join(output_dir, f"samples_{fit_name}.npz")
        )
        effects = percent_change_per_pp(
            posterior_samples["iv_global"][:, 0],
            fit_rows,
            "estimated_mmr_cov",
            df,
            COVARIATE_SCALE,
            MMR_EFFECTIVENESS,
        )
        low, median, high = np.percentile(effects, [2.5, 50, 97.5])
        summaries.append([low, median, high])
        print(
            f"{label.replace(chr(10), ' ')}: {fit_rows.height} county-years, "
            f"{fit_rows['cases_outbreak'].sum()} outbreak cases, effect "
            f"{median:.3f} [{low:.3f}, {high:.3f}]"
        )
    summaries = np.array(summaries)

    # dashed line at 0
    ax.axhline(y=0, color="k", linestyle="--")
    ax.errorbar(
        np.arange(len(fits)),
        summaries[:, 1],
        yerr=[
            summaries[:, 1] - summaries[:, 0],
            summaries[:, 2] - summaries[:, 1],
        ],
        fmt="o",
        color=COVARIATE_SET_COLORS["All covariates"],
        linewidth=3,
        markersize=10,
    )
    ax.set_xticks(np.arange(len(fits)))
    # tilted so that the four labels do not overlap in this narrow panel
    ax.set_xticklabels(
        [label for label, _, _ in fits],
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_xlim(-0.4, len(fits) - 0.6)
    ax.set_ylabel(effect_label())


if __name__ == "__main__":
    df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        # make sure we read fips codes as strings
        schema_overrides={"fips": pl.String},
    )

    fig, ax = setup_plot(2, 3)  # type: ignore

    # performance
    ax[0, 0].text(  # type: ignore
        -0.25,
        1.08,
        "a",
        transform=ax[0, 0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_immunity_effects_on_vaccine_coverage(
        df,
        fig,
        ax[0, 0],  # type: ignore
    )

    # ax[0, 1].text(  # type: ignore
    #     -0.25,
    #     1.08,
    #     "b",
    #     transform=ax[0, 1].transAxes,  # type: ignore
    #     fontsize=24,
    #     fontweight="bold",
    #     va="top",
    #     ha="right",
    # )
    # plot_previous_outbreak_size_effects(
    #     df,
    #     fig,
    #     ax[0, 1],  # type: ignore
    # )

    ax[0, 1].text(  # type: ignore
        -0.25,
        1.08,
        "b",
        transform=ax[0, 1].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_medical_exemption_coefficients(
        df,
        fig,
        ax[0, 1],  # type: ignore
    )

    ax[0, 2].text(  # type: ignore
        -0.25,
        1.08,
        "c",
        transform=ax[0, 2].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_neighbors_significance(
        df,
        fig,
        ax[0, 2],  # type: ignore
    )

    ax[1, 0].text(  # type: ignore
        -0.25,
        1.08,
        "d",
        transform=ax[1, 0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_overdispersion_posterior(
        fig,
        ax[1, 0],  # type: ignore
    )

    # the fits that estimate the MMR effectiveness exist only on the natural
    # scale (02)
    if COVARIATE_SCALE == "natural":
        ax[1, 1].text(  # type: ignore
            -0.25,
            1.08,
            "e",
            transform=ax[1, 1].transAxes,  # type: ignore
            fontsize=24,
            fontweight="bold",
            va="top",
            ha="right",
        )
        plot_effectiveness_coverage_effect(
            df,
            fig,
            ax[1, 1],  # type: ignore
        )
    else:
        print(
            "skipped panel e: the fits that estimate the MMR effectiveness need covariate_scale: natural in params.yml."
        )
        ax[1, 1].set_visible(False)  # type: ignore

    # the places of each cluster of 2025 outbreaks, left out of the fit one
    # cluster at a time (the leave-one-cluster-out fits of 02; the clusters
    # are found from the cases of all the counties, also those without
    # vaccine data; outbreak_size.outbreak_clusters)
    adjacency_data = pl.read_csv(
        os.path.join(data_dir, "county_adjacencies.txt"),
        separator="|",
        schema_overrides={
            "County GEOID": pl.String,
            "Neighbor GEOID": pl.String,
        },
    )
    cluster_regions = region_members()
    cluster_places = outbreak_cases_all_counties(
        pl.read_csv(
            os.path.join(data_dir, "raw_merged_outbreaks_vaccinations.csv"),
            schema_overrides={"fips": pl.String},
        ),
        cluster_regions,
    )
    cluster_seeds = cluster_seeds_2025(adjacency_data, cluster_regions)
    leave_out = {
        name: places_in_cluster(
            cluster_places,
            adjacency_data,
            cluster_regions,
            2025,
            cluster_seeds[name],
        )
        for name in CLUSTER_NAMES
    }
    ax[1, 2].text(  # type: ignore
        -0.25,
        1.08,
        "f",
        transform=ax[1, 2].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_leave_one_cluster_out(
        df,
        leave_out,
        fig,
        ax[1, 2],  # type: ignore
    )
    fig.tight_layout()

    # save
    fig.savefig(
        os.path.join(
            fig_dir,
            "figure_supplemental_sensitivity_analyses_vaccine_coverage_effect.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(
            fig_dir,
            "figure_supplemental_sensitivity_analyses_vaccine_coverage_effect.pdf",
        ),
        dpi=300,
        bbox_inches="tight",
    )
