import os
import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import rc

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)
fig_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")

font = {"family": "Nimbus Roman", "size": 28}
rc("font", **font)


def setup_plot(nrows, ncols):
    fig, ax = plt.subplots(nrows, ncols, figsize=(20, 30))
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


def plot_time_varying_vaccine_effect(
    sample_dict_wh, df, years_in_order, rolling_average, fig, ax
):
    std_wh = df["estimated_mmr_cov"].to_numpy().std()
    # get the relevant part of the sample for each sample
    for idx, model in enumerate(sample_dict_wh.keys()):
        mu = sample_dict_wh[model]["mu"]
        samples = sample_dict_wh[model]["effects_over_time_and_group"][:, :, 0]
        samples_wh = samples / std_wh
        # exponentiate and multiply by 100
        samples_wh = jnp.exp(samples_wh) * 100
        # to express as percent change
        samples_wh = samples_wh - 100

        # collect the values by years
        median_effect = np.array([])
        lower_effect = np.array([])
        upper_effect = np.array([])
        years_all_out = range(
            int(min(years_in_order)), int(max(years_in_order)) + 1
        )
        for year in years_all_out:
            mask = df["year"].is_in(year).to_jax()
            samples = samples_wh[:, mask]
            # average weighted by cases
            samples = (
                samples * mu[:, mask] / mu[:, mask].sum(axis=1, keepdims=True)
            )
            samples = samples.sum(axis=1)
            median_effect = np.append(
                median_effect, float(jnp.median(samples, axis=0))
            )
            lower_effect = np.append(
                lower_effect, float(jnp.quantile(samples, 0.025, axis=0))
            )
            upper_effect = np.append(
                upper_effect, float(jnp.quantile(samples, 0.975, axis=0))
            )

        if rolling_average:
            median_effect_expanded_cumsum = jnp.cumsum(median_effect)

            median_effect_expanded = jnp.concatenate(
                [
                    median_effect[:1],
                    median_effect_expanded_cumsum[1:2] / 2,
                    median_effect_expanded_cumsum[2:3] / 3,
                    (
                        median_effect_expanded_cumsum[3:]
                        - median_effect_expanded_cumsum[:-3]
                    )
                    / 3,
                ]
            )

            lower_effect_expanded_cumsum = jnp.cumsum(lower_effect)

            lower_effect_expanded = jnp.concatenate(
                [
                    lower_effect[:1],
                    lower_effect_expanded_cumsum[1:2] / 2,
                    lower_effect_expanded_cumsum[2:3] / 3,
                    (
                        lower_effect_expanded_cumsum[3:]
                        - lower_effect_expanded_cumsum[:-3]
                    )
                    / 3,
                ]
            )

            upper_effect_expanded_cumsum = jnp.cumsum(upper_effect)

            upper_effect_expanded = jnp.concatenate(
                [
                    upper_effect[:1],
                    upper_effect_expanded_cumsum[1:2] / 2,
                    upper_effect_expanded_cumsum[2:3] / 3,
                    (
                        upper_effect_expanded_cumsum[3:]
                        - upper_effect_expanded_cumsum[:-3]
                    )
                    / 3,
                ]
            )
        else:
            median_effect_expanded = median_effect
            lower_effect_expanded = lower_effect
            upper_effect_expanded = upper_effect

        ax.fill_between(
            years_all_out,
            lower_effect_expanded,
            upper_effect_expanded,
            color=f"C{idx}",
            alpha=0.3,
        )
        ax.plot(
            years_all_out, median_effect_expanded, color=f"C{idx}", label=model
        )

    # horizontal line at 0
    ax.axhline(0, ls="--", color="black")

    ax.legend(loc=(0.34, 0.7))

    ax.set_xlabel("Year")
    ax.set_ylabel(
        "% change in incidence rate per 1%\nincrease in vaccine coverage"
    )

    return fig, ax


def plot_time_variance_group_variance_distributions(
    rw_sigma, state_sigma, fig, ax
):
    ax.hist(
        state_sigma,
        color="green",
        alpha=0.5,
        density=True,
        bins=30,
        label="State-to-state",
    )
    ax.hist(
        rw_sigma,
        color="orange",
        alpha=0.5,
        density=True,
        bins=30,
        label="Year-to-year",
    )
    ax.legend(loc=(0.0, 0.75))
    ax.set_xlabel("Standard deviation in vaccine effect")
    ax.set_ylabel("Density")
    ax.set_xlim(0)


def plot_time_variance_vs_group_variance(rw_sigma, state_sigma, fig, ax):
    ax.scatter(state_sigma, rw_sigma, c="k", marker="o", alpha=0.2)
    ax.set_xlim(0)
    ax.set_xlabel("State-to-state standard deviation\nin vaccine effect")
    ax.set_ylabel("Year-to-year standard deviation\nin vaccine effect")
    # R2
    r2 = jnp.corrcoef(state_sigma, rw_sigma)[0, 1] ** 2
    ax.text(
        0.65,
        0.99,
        f"$R^2$ = {r2:.2f}",
        transform=ax.transAxes,
        fontsize=32,
        verticalalignment="top",
    )


if __name__ == "__main__":
    fig, ax = setup_plot(4, 2)  # type: ignore

    df = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        schema_overrides={"fips": pl.String},
    )

    years_in_order = list(
        [2011, 2012.5, 2014.5, 2016.5, 2018.5, 2020.5, 2022.5, 2025]
    )

    sample_dict_wh = {}

    # time and group varying
    with open(
        os.path.join(
            output_dir,
            "predictive_random_walk_poisson_CAR_time_and_group_varying_2025_mega_regions=True.pkl",
        ),
        "rb",
    ) as file:
        samples = pickle.load(file)
    sample_dict_wh["Time and state varying"] = samples

    # controls are time varying only
    with open(
        os.path.join(
            output_dir,
            "predictive_random_walk_poisson_CAR_time_varying_only_2025_mega_regions=True.pkl",
        ),
        "rb",
    ) as file:
        samples = pickle.load(file)
    sample_dict_wh["Time varying only"] = samples

    # controls are group varying only
    with open(
        os.path.join(
            output_dir,
            "predictive_random_walk_poisson_CAR_group_varying_only_2025_mega_regions=True.pkl",
        ),
        "rb",
    ) as file:
        samples = pickle.load(file)
    sample_dict_wh["State varying only"] = samples

    # controls are constant
    with open(
        os.path.join(
            output_dir,
            "predictive_random_walk_poisson_CAR_constant_2025_mega_regions=True.pkl",
        ),
        "rb",
    ) as file:
        samples = pickle.load(file)
    sample_dict_wh["Constant"] = samples

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
    plot_time_varying_vaccine_effect(
        sample_dict_wh,
        df,
        years_in_order,
        True,
        fig,
        ax[0, 0],  # type: ignore
    )

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
    plot_time_varying_vaccine_effect(
        sample_dict_wh,
        df,
        years_in_order,
        False,
        fig,
        ax[0, 1],  # type: ignore
    )

    with open(
        os.path.join(
            output_dir,
            "samples_random_walk_poisson_CAR_time_and_group_varying_2025_mega_regions=True.pkl",
        ),
        "rb",
    ) as file:
        samples = pickle.load(file)

    ax[1, 0].text(  # type: ignore
        -0.25,
        1.08,
        "c",
        transform=ax[1, 0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_time_variance_group_variance_distributions(
        samples["rw_drift"][:, 0] / (df["estimated_mmr_cov"].to_numpy().std()),
        samples["time_and_group_varying_group_variabilities"][:, 0]
        / (df["estimated_mmr_cov"].to_numpy().std()),
        fig,
        ax[1, 0],  # type: ignore
    )

    ax[1, 1].text(  # type: ignore
        -0.25,
        1.08,
        "d",
        transform=ax[1, 1].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_time_variance_vs_group_variance(
        samples["rw_drift"][:, 0] / (df["estimated_mmr_cov"].to_numpy().std()),
        samples["time_and_group_varying_group_variabilities"][:, 0]
        / (df["estimated_mmr_cov"].to_numpy().std()),
        fig,
        ax[1, 1],  # type: ignore
    )

    # From here on we test alternative data sets
    # the first row is what happens if we drop 2016/2017
    sample_dict_wh = {}

    # time and group varying
    with open(
        os.path.join(
            output_dir,
            "predictive_random_walk_poisson_CAR_time_and_group_varying_2025_mega_regions=True_exclude_16_17.pkl",
        ),
        "rb",
    ) as file:
        samples = pickle.load(file)
    sample_dict_wh["Time and state varying"] = samples

    # controls are time varying only
    with open(
        os.path.join(
            output_dir,
            "predictive_random_walk_poisson_CAR_time_varying_only_2025_mega_regions=True_exclude_16_17.pkl",
        ),
        "rb",
    ) as file:
        samples = pickle.load(file)
    sample_dict_wh["Time varying only"] = samples

    # controls are group varying only
    with open(
        os.path.join(
            output_dir,
            "predictive_random_walk_poisson_CAR_group_varying_only_2025_mega_regions=True_exclude_16_17.pkl",
        ),
        "rb",
    ) as file:
        samples = pickle.load(file)
    sample_dict_wh["State varying only"] = samples

    # controls are constant
    with open(
        os.path.join(
            output_dir,
            "predictive_random_walk_poisson_CAR_constant_2025_mega_regions=True_exclude_16_17.pkl",
        ),
        "rb",
    ) as file:
        samples = pickle.load(file)
    sample_dict_wh["Constant"] = samples

    exclude_16_17 = pl.read_csv(
        os.path.join(output_dir, "exclude_16_17.csv"),
    )
    df = df.with_columns(
        year_county_state=pl.col("year").cast(pl.String)
        + pl.col("county")
        + pl.col("state")
    )

    ax[2, 0].text(  # type: ignore
        -0.25,
        1.08,
        "e",
        transform=ax[2, 0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_time_varying_vaccine_effect(
        sample_dict_wh,
        df.filter(
            ~pl.col("year_county_state").is_in(
                exclude_16_17["year_county_state"].to_list()
            )
        ),
        years_in_order,
        True,
        fig,
        ax[2, 0],  # type: ignore
    )

    ax[2, 1].text(  # type: ignore
        -0.25,
        1.08,
        "f",
        transform=ax[2, 1].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_time_varying_vaccine_effect(
        sample_dict_wh,
        df.filter(
            ~pl.col("year_county_state").is_in(
                exclude_16_17["year_county_state"].to_list()
            )
        ),
        years_in_order,
        False,
        fig,
        ax[2, 1],  # type: ignore
    )

    # the second row is what happens if we drop the 2025 mega regions
    sample_dict_wh = {}

    # time and group varying
    with open(
        os.path.join(
            output_dir,
            "predictive_random_walk_poisson_CAR_time_and_group_varying_2025_mega_regions=False.pkl",
        ),
        "rb",
    ) as file:
        samples = pickle.load(file)
    sample_dict_wh["Time and state varying"] = samples

    # controls are time varying only
    with open(
        os.path.join(
            output_dir,
            "predictive_random_walk_poisson_CAR_time_varying_only_2025_mega_regions=False.pkl",
        ),
        "rb",
    ) as file:
        samples = pickle.load(file)
    sample_dict_wh["Time varying only"] = samples

    # controls are group varying only
    with open(
        os.path.join(
            output_dir,
            "predictive_random_walk_poisson_CAR_group_varying_only_2025_mega_regions=False.pkl",
        ),
        "rb",
    ) as file:
        samples = pickle.load(file)
    sample_dict_wh["State varying only"] = samples

    # controls are constant
    with open(
        os.path.join(
            output_dir,
            "predictive_random_walk_poisson_CAR_constant_2025_mega_regions=False.pkl",
        ),
        "rb",
    ) as file:
        samples = pickle.load(file)
    sample_dict_wh["Constant"] = samples

    ax[3, 0].text(  # type: ignore
        -0.25,
        1.08,
        "g",
        transform=ax[3, 0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_time_varying_vaccine_effect(
        sample_dict_wh,
        df.filter(~pl.col("mega_region_2025")),
        years_in_order,
        True,
        fig,
        ax[3, 0],  # type: ignore
    )

    ax[3, 1].text(  # type: ignore
        -0.25,
        1.08,
        "h",
        transform=ax[3, 1].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_time_varying_vaccine_effect(
        sample_dict_wh,
        df.filter(~pl.col("mega_region_2025")),
        years_in_order,
        False,
        fig,
        ax[3, 1],  # type: ignore
    )

    fig.tight_layout()

    # save
    fig.savefig(
        os.path.join(
            fig_dir,
            "figure_supplemental_alternative_time_varying_specifications.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(
            fig_dir,
            "figure_supplemental_alternative_time_varying_specifications.pdf",
        ),
        dpi=300,
        bbox_inches="tight",
    )
