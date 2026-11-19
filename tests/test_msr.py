"""Moran spectral randomization (outbreak_size.msr)."""

import numpy as np
import pytest
from scipy import stats

from outbreak_size.msr import (
    _block_permuted_outcome,
    _count_counties,
    _partial_null_outcome,
    calculate_moran_eigenvectors_adjacency,
    msr_county_permutation,
    normal_scores,
    regression_msr_multiple_blocked,
    spearman_msr_blocked,
    spearman_msr_interval,
)


def path_adjacency(n):
    adjacency = np.zeros((n, n))
    for i in range(n - 1):
        adjacency[i, i + 1] = adjacency[i + 1, i] = 1
    return adjacency


def morans_i(values, adjacency):
    z = values - values.mean()
    return len(values) / adjacency.sum() * (z @ adjacency @ z) / (z @ z)


def test_eigenvectors_are_orthonormal_and_sorted_by_eigenvalue():
    eigenvectors, eigenvalues = calculate_moran_eigenvectors_adjacency(
        path_adjacency(20), n_vectors=20
    )
    assert eigenvectors.shape == (20, 20)
    assert eigenvectors.T @ eigenvectors == pytest.approx(np.eye(20), abs=1e-8)
    assert (np.diff(eigenvalues) <= 1e-12).all()


def test_positive_eigenvalues_are_kept_by_default():
    _, eigenvalues = calculate_moran_eigenvectors_adjacency(path_adjacency(20))
    assert len(eigenvalues) > 0
    assert (eigenvalues > 0).all()


def test_an_asymmetric_matrix_is_refused():
    adjacency = path_adjacency(6)
    adjacency[0, 1] = 0
    with pytest.raises(ValueError, match="symmetric"):
        calculate_moran_eigenvectors_adjacency(adjacency)


def setup_field(n=40, seed=0):
    adjacency = path_adjacency(n)
    rng = np.random.default_rng(seed)
    county_values = np.linspace(0, 10, n) + rng.normal(0, 0.05, n)
    eigenvectors, _ = calculate_moran_eigenvectors_adjacency(
        adjacency, n_vectors=n
    )
    coefficients = eigenvectors.T @ (county_values - county_values.mean())
    return adjacency, county_values, eigenvectors, coefficients


def test_the_permutation_is_a_permutation_of_the_counties():
    _, values, eigenvectors, coefficients = setup_field()
    np.random.seed(1)
    for _ in range(20):
        perm = msr_county_permutation(eigenvectors, coefficients, values)
        assert sorted(perm) == list(range(len(values)))


def test_the_same_seed_gives_the_same_permutation():
    _, values, eigenvectors, coefficients = setup_field()
    np.random.seed(7)
    first = msr_county_permutation(eigenvectors, coefficients, values)
    np.random.seed(7)
    second = msr_county_permutation(eigenvectors, coefficients, values)
    assert (first == second).all()


def test_permuted_values_keep_more_spatial_structure_than_a_random_shuffle():
    adjacency, values, eigenvectors, coefficients = setup_field()
    np.random.seed(3)
    msr_i = np.mean(
        [
            morans_i(
                values[
                    msr_county_permutation(eigenvectors, coefficients, values)
                ],
                adjacency,
            )
            for _ in range(200)
        ]
    )
    rng = np.random.default_rng(3)
    shuffle_i = np.mean(
        [morans_i(rng.permutation(values), adjacency) for _ in range(200)]
    )
    assert shuffle_i < 0.2
    assert msr_i > shuffle_i + 0.3


def test_tied_county_values_are_ranked_in_a_random_order():
    # most outcomes are 0: the tied counties must not always keep their order
    n = 30
    adjacency = path_adjacency(n)
    values = np.zeros(n)
    values[5] = 4.0
    eigenvectors, _ = calculate_moran_eigenvectors_adjacency(
        adjacency, n_vectors=n
    )
    coefficients = eigenvectors.T @ (values - values.mean())
    np.random.seed(0)
    perms = {
        tuple(msr_county_permutation(eigenvectors, coefficients, values))
        for _ in range(30)
    }
    assert len(perms) > 1


# --- block permutation of counties with several observations each ------------


def grid_adjacency(side=6):
    """Rook adjacency of a side x side grid of counties."""
    n = side * side
    adjacency = np.zeros((n, n))
    for i in range(side):
        for j in range(side):
            k = i * side + j
            if i + 1 < side:
                adjacency[k, k + side] = adjacency[k + side, k] = 1
            if j + 1 < side:
                adjacency[k, k + 1] = adjacency[k + 1, k] = 1
    return adjacency


def panel(seed=0):
    """A 6 x 6 grid of counties observed 3 times each (4 times for the first
    six, so the panel is unbalanced), an outcome with no spatial structure and
    two covariates: one that is the outcome plus a little noise and one that
    is unrelated."""
    rng = np.random.default_rng(seed)
    counts = np.where(np.arange(36) < 6, 4, 3)
    county_indices = np.repeat(np.arange(36), counts)
    y = rng.normal(size=len(county_indices))
    x_related = y + 0.1 * rng.normal(size=len(county_indices))
    x_unrelated = rng.normal(size=len(county_indices))
    return grid_adjacency(), county_indices, y, x_related, x_unrelated


def test_count_counties_needs_indices_that_number_the_adjacency_rows():
    adjacency = grid_adjacency(2)  # 4 counties
    assert _count_counties(np.array([0, 0, 1, 2, 3, 3]), adjacency) == 4
    with pytest.raises(ValueError, match="doesn't match"):
        _count_counties(np.array([0, 1, 2]), adjacency)
    with pytest.raises(ValueError, match="0-indexed"):
        _count_counties(np.array([1, 2, 3, 4]), adjacency)  # right count, gap


def test_block_permutation_moves_whole_blocks_of_equal_size():
    """With blocks of the same size, county i takes county county_perm[i]'s
    observations in order, so the identity changes nothing and a swap of two
    counties swaps their blocks."""
    y = np.arange(9, dtype=float)
    masks = [np.arange(9) // 3 == i for i in range(3)]
    assert np.array_equal(_block_permuted_outcome(y, masks, [0, 1, 2]), y)
    swapped = _block_permuted_outcome(y, masks, [2, 1, 0])
    assert swapped.tolist() == [6, 7, 8, 3, 4, 5, 0, 1, 2]


def test_block_permutation_samples_when_the_blocks_differ_in_size():
    """A county with fewer observations gets some of the donor's values, with
    no repeats; one with more gets draws (with repeats allowed) of them. Either
    way every value comes from the donor county."""
    y = np.array([10.0, 11.0, 12.0, 20.0, 21.0])
    masks = [np.arange(5) < 3, np.arange(5) >= 3]  # sizes 3 and 2
    for seed in range(30):  # a repeat is possible, so one seed proves nothing
        np.random.seed(seed)
        out = _block_permuted_outcome(y, masks, [1, 0])
        # county 0 (3 values) takes from county 1 (2 values): with replacement
        assert set(out[:3]) <= {20.0, 21.0}
        # county 1 (2 values) takes from county 0 (3 values): without repeats
        assert set(out[3:]) <= {10.0, 11.0, 12.0}
        assert out[3] != out[4]


def test_spearman_blocked_finds_a_related_covariate_and_not_an_unrelated_one():
    adjacency, counties, y, x_related, x_unrelated = panel()
    n_permutations = 199
    rho, p, null = spearman_msr_blocked(
        x_related, y, adjacency, counties, n_permutations, seed=1
    )
    assert rho == pytest.approx(stats.spearmanr(x_related, y)[0])
    assert rho > 0.95  # type: ignore
    assert null.shape == (n_permutations,)
    # no permutation is as extreme: the smallest p-value there can be
    assert p == pytest.approx(1 / (n_permutations + 1))
    _, p_unrelated, _ = spearman_msr_blocked(
        x_unrelated, y, adjacency, counties, n_permutations, seed=1
    )
    assert p_unrelated > 0.05


def test_spearman_blocked_p_value_counts_the_extreme_permutations():
    """p = (permutations at least as extreme in absolute value + 1) divided by
    (permutations + 1)."""
    adjacency, counties, y, _, x_unrelated = panel()
    rho, p, null = spearman_msr_blocked(
        x_unrelated, y, adjacency, counties, 199, seed=2
    )
    expected = (np.sum(np.abs(null) >= abs(rho)) + 1) / 200  # type: ignore
    assert p == pytest.approx(expected)


def test_spearman_blocked_is_reproducible_and_depends_on_the_seed():
    adjacency, counties, y, _, x_unrelated = panel()
    args = (x_unrelated, y, adjacency, counties, 99)
    first = spearman_msr_blocked(*args, seed=3)
    again = spearman_msr_blocked(*args, seed=3)
    other = spearman_msr_blocked(*args, seed=4)
    assert np.array_equal(first[2], again[2])
    assert not np.array_equal(first[2], other[2])


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6])
def test_spearman_interval_excludes_0_exactly_when_p_is_at_most_05(seed):
    rng = np.random.default_rng(seed)
    n_permutations = 199
    null = rng.normal(0, 0.1, n_permutations)
    for rho in (0.05, 0.12, 0.2, -0.15, -0.3):
        p = (np.sum(np.abs(null) >= abs(rho)) + 1) / (n_permutations + 1)
        low, high, critical = spearman_msr_interval(rho, null)
        assert (low > 0 or high < 0) == (p <= 0.05)
        assert (low, high) == (rho - critical, rho + critical)


def test_spearman_interval_needs_at_least_19_permutations():
    with pytest.raises(ValueError, match="19 permutations"):
        spearman_msr_interval(0.1, np.zeros(18))
    spearman_msr_interval(0.1, np.linspace(-0.2, 0.2, 19))


def test_spearman_blocked_rejects_inconsistent_inputs():
    adjacency, counties, y, x, _ = panel()
    with pytest.raises(ValueError, match="same length"):
        spearman_msr_blocked(x[:-1], y, adjacency, counties, 9)
    with pytest.raises(ValueError, match="doesn't match"):
        spearman_msr_blocked(x, y, grid_adjacency(5), counties, 9)


def least_squares_t(design, y, j):
    """The t statistic of coefficient j, by the textbook formula."""
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    resid = y - design @ coef
    sigma2 = resid @ resid / (len(y) - design.shape[1])
    return coef[j] / np.sqrt(sigma2 * np.linalg.inv(design.T @ design)[j, j])


def test_regression_blocked_recovers_the_least_squares_fit():
    adjacency, counties, y, x_related, x_unrelated = panel()
    X = np.column_stack([x_related, x_unrelated])
    out = regression_msr_multiple_blocked(
        X, y, counties, adjacency, tested=(1, 2), n_permutations=199, seed=1
    )
    design = np.column_stack([np.ones(len(y)), X])
    expected = np.linalg.lstsq(design, y, rcond=None)[0]
    assert out["coef"] == pytest.approx(expected)
    residuals = y - design @ expected
    assert out["r_squared"] == pytest.approx(
        1 - residuals @ residuals / ((y - y.mean()) @ (y - y.mean()))
    )
    assert out["t"][1] == pytest.approx(least_squares_t(design, y, 1))
    assert out["t"][2] == pytest.approx(least_squares_t(design, y, 2))


def test_regression_blocked_tests_only_the_listed_coefficients():
    adjacency, counties, y, x_related, x_unrelated = panel()
    X = np.column_stack([x_related, x_unrelated])
    out = regression_msr_multiple_blocked(
        X, y, counties, adjacency, tested=(2,), n_permutations=49, seed=1
    )
    assert list(out["p_values"]) == [2]
    assert list(out["t"]) == [2]
    assert out["t_null"][2].shape == (49,)


def test_regression_blocked_finds_a_related_predictor_and_not_an_unrelated_one():
    adjacency, counties, y, x_related, x_unrelated = panel()
    X = np.column_stack([x_related, x_unrelated])
    out = regression_msr_multiple_blocked(
        X, y, counties, adjacency, tested=(1, 2), n_permutations=199, seed=1
    )
    assert out["p_values"][1] == pytest.approx(1 / 200)
    assert out["p_values"][2] > 0.05


def partial_panel(effect_of_second):
    """The grid panel with an outcome that the first predictor explains almost
    completely (3 per unit) and a second one with its own small effect."""
    rng = np.random.default_rng(3)
    adjacency, counties, _, _, _ = panel()
    n = len(counties)
    first = rng.normal(size=n)
    second = rng.normal(size=n)
    y = 3 * first + effect_of_second * second + 0.1 * rng.normal(size=n)
    return adjacency, counties, y, np.column_stack([first, second])


def test_a_coefficient_is_tested_given_the_other_predictors():
    # the first predictor explains nearly all of the outcome, so the second
    # one's small effect is clear next to what is left; a test that shuffled
    # the whole outcome would compare it with the spread of the outcome
    adjacency, counties, y, X = partial_panel(effect_of_second=0.5)
    out = regression_msr_multiple_blocked(
        X, y, counties, adjacency, tested=(2,), n_permutations=199, seed=1
    )
    assert out["p_values"][2] == pytest.approx(1 / 200)


def test_a_predictor_with_no_effect_is_not_found_next_to_a_strong_one():
    adjacency, counties, y, X = partial_panel(effect_of_second=0.0)
    out = regression_msr_multiple_blocked(
        X, y, counties, adjacency, tested=(1, 2), n_permutations=199, seed=1
    )
    assert out["p_values"][1] == pytest.approx(1 / 200)
    assert out["p_values"][2] > 0.05


def test_the_null_t_statistics_are_centered_and_on_the_scale_of_a_t():
    adjacency, counties, y, X = partial_panel(effect_of_second=0.0)
    null = regression_msr_multiple_blocked(
        X, y, counties, adjacency, tested=(2,), n_permutations=199, seed=1
    )["t_null"][2]
    assert abs(null.mean()) < 0.3
    assert 0.7 < null.std() < 1.4


def test_regression_blocked_accepts_a_single_predictor():
    adjacency, counties, y, x_related, _ = panel()
    one = regression_msr_multiple_blocked(
        x_related,
        y,
        counties,
        adjacency,
        tested=(1,),
        n_permutations=49,
        seed=5,
    )
    as_column = regression_msr_multiple_blocked(
        x_related[:, None],
        y,
        counties,
        adjacency,
        tested=(1,),
        n_permutations=49,
        seed=5,
    )
    assert len(one["coef"]) == 2
    assert one["p_values"] == as_column["p_values"]


def test_regression_blocked_is_reproducible_and_depends_on_the_seed():
    adjacency, counties, y, X = partial_panel(effect_of_second=0.0)

    def null(seed):
        return regression_msr_multiple_blocked(
            X,
            y,
            counties,
            adjacency,
            tested=(2,),
            n_permutations=49,
            seed=seed,
        )["t_null"][2]

    assert np.array_equal(null(4), null(4))
    assert not np.array_equal(null(4), null(5))


@pytest.mark.parametrize("tested", [(0,), (3,), (1, 4)])
def test_regression_blocked_refuses_coefficients_it_cannot_test(tested):
    adjacency, counties, y, X = partial_panel(effect_of_second=0.0)
    with pytest.raises(ValueError, match="cannot be tested"):
        regression_msr_multiple_blocked(
            X, y, counties, adjacency, tested=tested, n_permutations=9
        )


def test_the_null_outcome_is_the_fit_plus_permuted_residual_blocks():
    adjacency, counties, y, x_related, _ = panel()
    rng = np.random.default_rng(8)
    fitted = 5.0 + rng.normal(size=len(y))  # not zero, not the outcome
    residual = rng.normal(size=len(y))
    masks = [counties == i for i in range(36)]
    eigenvectors, _ = calculate_moran_eigenvectors_adjacency(
        adjacency, n_vectors=36
    )
    np.random.seed(0)
    null = _partial_null_outcome(eigenvectors, fitted, residual, masks)
    moved = null - fitted
    # every county's block of residuals is one of the residual values (the
    # outcome and the fit do not enter it), and the counties are different
    # from the original assignment
    assert set(np.round(moved, 12)) <= set(np.round(residual, 12))
    assert not np.array_equal(moved, residual)


def test_a_negative_effect_is_found_too():
    adjacency, counties, y, x_related, _ = panel()
    out = regression_msr_multiple_blocked(
        -x_related,
        y,
        counties,
        adjacency,
        tested=(1,),
        n_permutations=199,
        seed=1,
    )
    assert out["t"][1] < 0
    assert out["p_values"][1] == pytest.approx(1 / 200)


def test_the_null_outcome_keeps_the_spatial_pattern_of_the_residual():
    # the residual is a smooth field along a path of counties, the fit is
    # random: the permuted residual stays smooth, as the residual is
    n = 40
    adjacency = path_adjacency(n)
    residual = np.sin(np.linspace(0, 3 * np.pi, n))
    fitted = np.random.default_rng(2).normal(size=n)
    masks = [np.arange(n) == i for i in range(n)]
    eigenvectors, _ = calculate_moran_eigenvectors_adjacency(
        adjacency, n_vectors=n
    )
    np.random.seed(0)
    smoothness = [
        morans_i(
            _partial_null_outcome(eigenvectors, fitted, residual, masks)
            - fitted,
            adjacency,
        )
        for _ in range(30)
    ]
    assert np.mean(smoothness) > 0.5  # a random shuffle gives about 0


def test_the_p_value_counts_the_null_statistics_as_extreme_as_the_observed():
    # four counties have few distinct permutations, so some null statistics
    # equal the observed one exactly, and those count
    rng = np.random.default_rng(6)
    out = regression_msr_multiple_blocked(
        rng.normal(size=4),
        rng.normal(size=4),
        np.arange(4),
        path_adjacency(4),
        tested=(1,),
        n_permutations=999,
        seed=2,
    )
    null, observed = out["t_null"][1], out["t"][1]
    assert np.any(np.abs(null) == abs(observed))
    expected = (np.sum(np.abs(null) >= abs(observed)) + 1) / 1000
    assert out["p_values"][1] == pytest.approx(expected)


# --- swaps only between counties with the same number of observations ---------


def test_counties_only_swap_within_their_number_of_observations():
    _, values, eigenvectors, coefficients = setup_field()
    n = len(values)
    lengths = np.where(np.arange(n) % 3 == 0, 2, 1)
    np.random.seed(2)
    for _ in range(20):
        perm = msr_county_permutation(
            eigenvectors, coefficients, values, lengths
        )
        assert sorted(perm) == list(range(n))
        assert (lengths[perm] == lengths).all()


def test_a_county_alone_in_its_group_keeps_its_block():
    _, values, eigenvectors, coefficients = setup_field()
    lengths = np.ones(len(values), dtype=int)
    lengths[7] = 5
    np.random.seed(2)
    for _ in range(10):
        perm = msr_county_permutation(
            eigenvectors, coefficients, values, lengths
        )
        assert perm[7] == 7


def test_within_a_group_the_ranks_of_the_surrogate_pick_the_observed_ranks():
    # an identity basis makes the surrogate the coefficients with random
    # signs, which the test reproduces from the same seed
    n = 8
    eigenvectors = np.eye(n)
    coefficients = np.array([5.0, 1.0, 4.0, 2.0, 3.0, 6.0, 0.5, 7.0])
    values = np.arange(n, dtype=float)[::-1] * 10  # 70, 60, ..., 0
    lengths = np.array([1, 2, 1, 2, 1, 2, 1, 2])
    np.random.seed(11)
    signs = np.random.choice([-1.0, 1.0], size=n)
    surrogate = coefficients * signs
    np.random.seed(11)
    perm = msr_county_permutation(eigenvectors, coefficients, values, lengths)
    for length in (1, 2):
        group = np.where(lengths == length)[0]
        by_surrogate = group[np.argsort(surrogate[group])]
        by_observed = group[np.argsort(values[group])]
        assert (perm[by_surrogate] == by_observed).all()


def test_the_null_outcome_is_a_permutation_of_the_outcome():
    adjacency, counties, y, _, _ = panel()
    masks = [counties == i for i in range(36)]
    lengths = np.array([mask.sum() for mask in masks])
    eigenvectors, _ = calculate_moran_eigenvectors_adjacency(
        adjacency, n_vectors=36
    )
    county_values = np.array([y[mask].mean() for mask in masks])
    scores = normal_scores(county_values)
    coefficients = eigenvectors.T @ (scores - scores.mean())
    np.random.seed(0)
    for _ in range(10):
        perm = msr_county_permutation(
            eigenvectors, coefficients, county_values, lengths
        )
        permuted = _block_permuted_outcome(y, masks, perm)
        assert np.array_equal(np.sort(permuted), np.sort(y))


def test_normal_scores_are_ordered_symmetric_and_equal_for_ties():
    scores = normal_scores(np.array([0.0, 0.0, 0.0, 3.0, 10.0]))
    assert scores[0] == scores[1] == scores[2]
    assert scores[2] < scores[3] < scores[4]
    symmetric = normal_scores(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert symmetric == pytest.approx(-symmetric[::-1])
    assert symmetric[2] == pytest.approx(0.0)
    assert symmetric[0] == pytest.approx(stats.norm.ppf(0.1))


def test_spearman_blocked_null_draws_are_permutations_of_the_outcome(
    monkeypatch,
):
    # two counties with one observation and two with two: the only outcomes
    # a null draw can have are the four stratified swaps, and so only four
    # null correlations (a resampled block would give others)
    counties = np.array([0, 1, 2, 2, 3, 3])
    y = np.array([5.0, 1.0, 2.0, 9.0, 7.0, 3.0])
    x = np.array([4.0, 2.0, 8.0, 1.0, 6.0, 3.0])
    rho_null = spearman_msr_blocked(
        x, y, path_adjacency(4), counties, n_permutations=300, seed=1
    )[2]
    achievable = set()
    for singles in ([0, 1], [1, 0]):
        for doubles in ([2, 3], [3, 2]):
            owner = [singles[0], singles[1], doubles[0], doubles[1]]
            blocks = {0: [5.0], 1: [1.0], 2: [2.0, 9.0], 3: [7.0, 3.0]}
            permuted = np.concatenate([blocks[owner[i]] for i in range(4)])
            achievable.add(
                round(float(np.asarray(stats.spearmanr(x, permuted))[0]), 9)
            )
    assert {round(r, 9) for r in rho_null} <= achievable


def test_spearman_blocked_swaps_by_the_normal_scores_spectrum_in_groups(
    monkeypatch,
):
    import outbreak_size.msr as msr

    seen = {}
    real = msr.msr_county_permutation

    def spy(eigenvectors, coefficients, county_values, block_lengths=None):
        seen["coefficients"] = coefficients
        seen["county_values"] = county_values
        seen["block_lengths"] = block_lengths
        return real(eigenvectors, coefficients, county_values, block_lengths)

    monkeypatch.setattr(msr, "msr_county_permutation", spy)
    adjacency, counties, y, x, _ = panel()
    spearman_msr_blocked(x, y, adjacency, counties, n_permutations=2, seed=1)
    masks = [counties == i for i in range(36)]
    means = np.array([y[mask].mean() for mask in masks])
    eigenvectors, _ = calculate_moran_eigenvectors_adjacency(
        adjacency, n_vectors=36
    )
    scores = normal_scores(means)
    assert seen["coefficients"] == pytest.approx(
        eigenvectors.T @ (scores - scores.mean())
    )
    assert seen["county_values"] == pytest.approx(means)
    assert list(seen["block_lengths"]) == [mask.sum() for mask in masks]


def test_tied_counties_are_ordered_at_random_whatever_the_surrogate():
    # a zero spectrum makes every surrogate value equal, and all the observed
    # values are tied: the permutation then comes from the random tie-break
    n = 10
    np.random.seed(0)
    perms = {
        tuple(
            msr_county_permutation(
                np.eye(n), np.zeros(n), np.zeros(n), np.ones(n)
            )
        )
        for _ in range(20)
    }
    assert len(perms) > 1
