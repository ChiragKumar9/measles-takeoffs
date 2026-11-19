import numpy as np
import pytest
import statsmodels.api as sm

from outbreak_size.msr_logistic import logistic_fit, logistic_msr_test


def chain_adjacency(n):
    """n counties in a line, each bordering the next."""
    matrix = np.zeros((n, n))
    for i in range(n - 1):
        matrix[i, i + 1] = matrix[i + 1, i] = 1
    return matrix


def make_data(effect, n_counties=40, per_county=3, seed=0):
    """Two predictors, the first with a logit effect `effect` on y; each county has per_county observations."""
    rng = np.random.default_rng(seed)
    n = n_counties * per_county
    X = rng.normal(size=(n, 2)) * [1.0, 5.0] + [0.0, 20.0]
    eta = -0.3 + effect * X[:, 0] + 0.1 * (X[:, 1] - 20.0) / 5.0
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    return X, y, np.repeat(np.arange(n_counties), per_county)


def test_logistic_fit_matches_statsmodels():
    X, y, _ = make_data(0.8)
    design = np.column_stack([np.ones(len(y)), X])
    beta, cov, converged = logistic_fit(design, y, ridge=0.0)
    reference = sm.GLM(y, design, family=sm.families.Binomial()).fit()
    assert converged
    np.testing.assert_allclose(beta, reference.params, atol=1e-6)
    np.testing.assert_allclose(np.sqrt(np.diag(cov)), reference.bse, atol=1e-6)


def test_score_equation_holds_at_the_fit():
    # at the maximum of the likelihood the residuals are orthogonal to every column of the design
    X, y, _ = make_data(0.8)
    design = np.column_stack([np.ones(len(y)), X])
    beta, _, _ = logistic_fit(design, y, ridge=0.0)
    p = 1 / (1 + np.exp(-design @ beta))
    np.testing.assert_allclose(design.T @ (y - p), 0.0, atol=1e-6)


def test_coefficients_are_in_the_units_of_X():
    X, y, county = make_data(0.8)
    adjacency = chain_adjacency(county.max() + 1)
    reference = sm.GLM(
        y, sm.add_constant(X), family=sm.families.Binomial()
    ).fit()
    out = logistic_msr_test(
        X, y, county, adjacency, tested=0, n_permutations=9
    )
    np.testing.assert_allclose(out["coef"], reference.params, atol=1e-4)
    np.testing.assert_allclose(out["se"][1:], reference.bse[1:], atol=1e-4)
    assert out["z"] == pytest.approx(reference.tvalues[1], abs=1e-3)


def test_a_strong_effect_is_significant_and_no_effect_is_not():
    adjacency = chain_adjacency(40)
    X, y, county = make_data(1.5)
    strong = logistic_msr_test(
        X, y, county, adjacency, tested=0, n_permutations=199
    )
    assert strong["p_value"] <= 0.02
    assert strong["z"] > 3
    X, y, county = make_data(0.0, seed=3)
    none = logistic_msr_test(
        X, y, county, adjacency, tested=0, n_permutations=199
    )
    assert none["p_value"] > 0.05


def test_same_seed_same_result():
    X, y, county = make_data(0.5)
    adjacency = chain_adjacency(county.max() + 1)
    a = logistic_msr_test(
        X, y, county, adjacency, tested=1, n_permutations=49, seed=5
    )
    b = logistic_msr_test(
        X, y, county, adjacency, tested=1, n_permutations=49, seed=5
    )
    assert a["p_value"] == b["p_value"]
    np.testing.assert_array_equal(a["z_null"], b["z_null"])


def test_inputs_are_checked():
    X, y, county = make_data(0.5)
    adjacency = chain_adjacency(county.max() + 1)
    with pytest.raises(ValueError, match="0 or 1"):
        logistic_msr_test(
            X, y + 1, county, adjacency, tested=0, n_permutations=9
        )
    with pytest.raises(ValueError, match="column of X"):
        logistic_msr_test(X, y, county, adjacency, tested=2, n_permutations=9)
    with pytest.raises(ValueError, match="same length"):
        logistic_msr_test(
            X[:-1], y, county, adjacency, tested=0, n_permutations=9
        )


def test_covariance_r2_and_clustered_p_match_statsmodels():
    X, y, county = make_data(0.8)
    adjacency = chain_adjacency(county.max() + 1)
    clusters = county // 2  # two neighbouring counties in a cluster
    out = logistic_msr_test(
        X, y, county, adjacency, tested=0, n_permutations=9, clusters=clusters
    )
    fit = sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial()).fit()
    # the covariance of the coefficients of the columns of X, per unit of X
    np.testing.assert_allclose(
        out["cov"], np.asarray(fit.cov_params())[1:, 1:], atol=1e-5
    )
    assert out["mcfadden_r2"] == pytest.approx(
        1 - fit.llf / fit.llnull, abs=1e-6
    )
    clustered = sm.GLM(
        y, sm.add_constant(X), family=sm.families.Binomial()
    ).fit(cov_type="cluster", cov_kwds={"groups": clusters})
    assert out["p_clustered"] == pytest.approx(clustered.pvalues[1], abs=1e-4)
    # without clusters the clustered p is empty
    assert np.isnan(
        logistic_msr_test(X, y, county, adjacency, tested=0, n_permutations=9)[
            "p_clustered"
        ]
    )
