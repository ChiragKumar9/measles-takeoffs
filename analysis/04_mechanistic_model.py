import os
import pickle

import numpy as np
import polars as pl
import yaml
from scipy import stats

from outbreak_size.mechanistic import (
    fit_mechanistic,
    limit_to_population,
    poisson_log_likelihood,
    theory_sizes,
)

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs"
)

# read analysis params yaml
with open("params.yml", "r") as f:
    params = yaml.safe_load(f)

MMR_EFFECTIVENESS = params["mmr_effectiveness"]
MECHANISTIC_R0 = params["mechanistic_r0"]

# The mechanistic model is in outbreak_size.mechanistic; its documentation
# explains the model and how it is fitted. Here it is fitted to all the places
# of a set of years at once, with a year effect and a state effect, and its
# predictions are those of the places it was fitted to. Each set of years is
# fitted with the coverage of the year before the outbreak and with the
# five-year average coverage.
if __name__ == "__main__":
    # load dataset
    df_all = pl.read_csv(
        os.path.join(data_dir, "analysis_dataset.csv"),
        schema_overrides={"fips": pl.String},
    )

    for years in [
        [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
        [2021, 2022, 2023, 2024, 2025],
    ]:
        for five_year in [False, True]:
            df = df_all.filter(pl.col("year").is_in(years))

            coverage_col = (
                "estimated_mmr_cov_5yr" if five_year else "estimated_mmr_cov"
            )
            coverage = df[coverage_col].to_numpy()
            population = df["pop"].to_numpy()
            observed_outbreaks = df["cases_outbreak"].to_numpy()

            # Which places have a lower coverage (Re >= 1; outbreaks can
            # infect much of the population) and the size outbreak theory
            # gives each place
            lower_coverage, size = theory_sizes(
                coverage, population, MECHANISTIC_R0, MMR_EFFECTIVENESS
            )

            fit = fit_mechanistic(
                lower_coverage,
                size,
                observed_outbreaks,
                df["year"].to_numpy(),
                df["state"].to_numpy(),
            )
            search = fit["search"]
            print(
                f"{years}, five_year={five_year}: threshold size "
                f"{np.exp(fit['log_threshold']):.2f} cases (that of the "
                f"average state and year), slope {fit['slope']:.3f}, "
                f"{fit['n_parameters']} parameters, log-likelihood "
                f"{fit['log_likelihood']:.1f}"
            )
            print(
                "  states left out of the fit (no outbreak cases in any of "
                f"their places) and predicted at 0: {fit['dropped_states']}"
            )
            print(
                "  search for the threshold size: the fit at the lowest and "
                f"highest bound is {search['drop_at_lowest_bound']:.1f} and "
                f"{search['drop_at_highest_bound']:.1f} log-likelihood units "
                f"below the best; {len(search['other_peaks'])} other peak(s) "
                "of the likelihood near the best"
            )

            # The mean of every place, limited to the population (a place
            # cannot have more cases than people): the predictive
            # distribution of the cases of a place is Poisson(mu)
            mu = limit_to_population(fit["mean"], population)
            median = stats.poisson.ppf(0.5, mu)
            lower = stats.poisson.ppf(0.025, mu)
            upper = stats.poisson.ppf(0.975, mu)

            # pickle results. The Poisson mean of every place is kept for
            # WAIC; llf and k are the log-likelihood of the observed cases
            # under those means and the number of fitted parameters (see the
            # documentation of the model for how they are counted)
            with open(
                os.path.join(
                    output_dir,
                    f"final_outbreak_size_predictions_{years}_five_year={five_year}.pkl",
                ),
                "wb",
            ) as f:
                pickle.dump(
                    {
                        "median": median,
                        "lower": lower,
                        "upper": upper,
                        # which places have a lower coverage, and which a
                        # higher one (Re < 1: outbreaks fizzle out)
                        "supercritical_mask": lower_coverage,
                        "subcritical_mask": ~lower_coverage,
                        "mu": mu,
                        "llf": poisson_log_likelihood(observed_outbreaks, mu),
                        "k": fit["n_parameters"],
                        # the coverage the places were fitted with, and the
                        # mean not limited by the population
                        "coverage": coverage,
                        "mean_not_limited_by_population": fit["mean"],
                        "log_threshold": fit["log_threshold"],
                        "slope": fit["slope"],
                    },
                    f,
                )
