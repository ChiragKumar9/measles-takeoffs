import numpy as np
from scipy import stats


def importation_distribution_by_county(
    state_total_imports,
    number_of_years,
    county_populations,
    prior_alpha=1.0,
    prior_beta=1.0,
    log=False,
):
    """Calculate importation probabilities by county using a Bayesian approach.
    - Calculate the yearly average number of imports for the state.
        In a naive model, this would be state_total_imports / number_of_years.
        In a Bayesian model where we say that in the absence of any data, there
        is a prior on the probability of having at least one importation,
        this becomes (state_total_imports + prior_alpha) /
        (number_of_years + prior_beta). prior_alpha and prior_beta specify the
        parameters of a Gamma prior on the importation rate. When they are both
        set to 1, this is equivalent to saying that the probability of having
        at least one importation in a year is uniform (meaning we know nothing
        about the importation rate).
        This mixture of a Poisson likelihood and Gamma prior results in a
        Negative Binomial posterior for the number of importations by state.
    - Distribute the state-level importation rate to counties proportional to
      their population (or log population if chosen)."""

    posterior_alpha = state_total_imports + prior_alpha
    posterior_beta = number_of_years + prior_beta

    if log:
        county_populations = np.log(county_populations)

    pop_weights = county_populations / np.sum(county_populations)

    # by the rules of multiplying a gamma distribution by a constant
    county_level_posterior_alpha = posterior_alpha * pop_weights

    # we return the posterior parameters for each county
    return county_level_posterior_alpha, posterior_beta


def importation_probabilities_by_county(county_level_alpha, beta):
    importation_rate_samples = stats.gamma.rvs(
        a=county_level_alpha,
        scale=1.0 / beta,
        size=(10000, len(county_level_alpha)),
    )

    number_of_import_samples = stats.poisson.rvs(importation_rate_samples)

    prob_at_least_one_import = np.mean(number_of_import_samples >= 1, axis=0)

    return number_of_import_samples, prob_at_least_one_import


if __name__ == "__main__":
    # example usage
    state_total_imports = 10
    number_of_years = 5
    county_populations = np.array([1000, 2000, 3000])

    county_alphas, beta = importation_distribution_by_county(
        state_total_imports,
        number_of_years,
        county_populations,
        prior_alpha=1.0,
        prior_beta=1.0,
        log=False,
    )

    samples, probs = importation_probabilities_by_county(county_alphas, beta)

    print("Importation probabilities by county:", probs)
    print("Importation samples by county:", samples)
