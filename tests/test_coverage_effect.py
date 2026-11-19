import numpy as np
import polars as pl
import pytest

from outbreak_size.coverage_effect import (
    cases_change_per_pp,
    covariate_sd,
    effect_label,
    factor_per_pp,
    log_change_per_step,
    log_effect_per_pp,
    marginal_percent_change_per_pp,
    percent_change,
    percent_change_per_pp,
    posterior_contrast,
)

# effectiveness 0.5 and coverage 0% and 50% give susceptible fractions 1 and
# 0.75, so the covariate (their log) is 0 and log(0.75) = -0.287682, whose
# population SD is half their difference, 0.143841 (the sample SD would be
# 0.203422)
ROWS = pl.DataFrame({"estimated_mmr_cov": [0.0, 50.0]})
SD_NATURAL = abs(np.log(0.75)) / 2


def test_sd_of_the_natural_column_is_the_population_sd():
    assert covariate_sd(
        ROWS, "estimated_mmr_cov", "natural", 0.5
    ) == pytest.approx(SD_NATURAL)


def test_sd_of_the_raw_column_is_the_coverage_in_percent():
    assert covariate_sd(ROWS, "estimated_mmr_cov", "raw") == pytest.approx(
        25.0
    )


def test_exemptions_are_standardized_as_log_one_plus_percent():
    # 0% and 3% exemptions: log(1) = 0 and log(4); population SD log(4) / 2
    rows = pl.DataFrame({"nonmed_exempt_pct": [0.0, 3.0, None]})
    assert covariate_sd(
        rows, "nonmed_exempt_pct", "natural", 0.5
    ) == pytest.approx(np.log(4) / 2)


def test_elasticity_of_two_gives_two_point_zero_one_percent_per_one_percent():
    # a coefficient of 2 SD is an elasticity of 2: 1.01 ** 2 - 1 = 2.01%
    assert percent_change(
        2 * SD_NATURAL, SD_NATURAL, "natural"
    ) == pytest.approx(2.01)
    assert np.exp(log_change_per_step(2.0, 1.0, "natural")) == pytest.approx(
        1.01**2
    )
    assert np.exp(log_change_per_step(2.0, 1.0, "raw")) == pytest.approx(
        np.exp(2.0)
    )
    assert percent_change(0.0, 1.0, "natural") == 0.0
    assert percent_change(-1.0, 1.0, "natural") < 0


def test_one_percentage_point_of_coverage_at_fifty_percent():
    # effectiveness 0.5: susceptible fraction 1 - 0.25 = 0.75 at 50% coverage
    # and 1 - 0.255 = 0.745 at 51%; with an elasticity of 2 the outbreak size
    # is multiplied by (0.745 / 0.75) ** 2 = 0.986711
    factor = factor_per_pp([2.0], [50.0], "natural", 0.5)
    assert factor.shape == (1, 1)
    assert factor[0, 0] == pytest.approx((0.745 / 0.75) ** 2)
    assert factor[0, 0] == pytest.approx(0.986711, abs=1e-6)
    # on the raw scale a log effect of -0.1 per point is the same everywhere
    raw = factor_per_pp([-0.1], [50.0, 90.0], "raw")
    assert raw[0].tolist() == pytest.approx([np.exp(-0.1)] * 2)


def test_one_percentage_point_of_exemptions():
    # exemptions at 0% and 3%: 1 + x goes from 1 to 2 and from 4 to 5
    factor = factor_per_pp([1.5], [0.0, 3.0], "natural", 0.97, "exemption")
    assert factor[0].tolist() == pytest.approx([2**1.5, (5 / 4) ** 1.5])


def test_one_elasticity_gives_a_different_effect_per_point_at_different_coverage():
    # the same elasticity (1) at the mean coverage of 2011-2020 (94.4%) and of
    # 2021-2025 (91.0%), effectiveness 0.97: (1 - 0.97 * 0.954) / (1 - 0.97 *
    # 0.944) = 0.88496 and (1 - 0.97 * 0.92) / (1 - 0.97 * 0.91) = 0.91731
    high = marginal_percent_change_per_pp([1.0], [94.4], "natural", 0.97)[0]
    low = marginal_percent_change_per_pp([1.0], [91.0], "natural", 0.97)[0]
    assert high == pytest.approx(-11.504, abs=0.001)
    assert low == pytest.approx(-8.269, abs=0.001)


def test_a_point_of_coverage_matters_more_where_coverage_is_high():
    low, high = factor_per_pp([2.0], [50.0, 95.0], "natural", 0.97)[0]
    assert high < low < 1


def test_average_over_county_years_and_in_cases():
    # two county-years at 50% and 90% coverage, effectiveness 0.5, elasticity 1
    # factors 0.745 / 0.75 and (1 - 0.455) / (1 - 0.45) = 0.545 / 0.55
    f1, f2 = 0.745 / 0.75, 0.545 / 0.55
    assert marginal_percent_change_per_pp([1.0], [50.0, 90.0], "natural", 0.5)[
        0
    ] == pytest.approx(100 * ((f1 - 1) + (f2 - 1)) / 2)
    # 10 and 30 cases: the average change per outbreak, in cases
    assert cases_change_per_pp([1.0], [10, 30], [50.0, 90.0], "natural", 0.5)[
        0
    ] == pytest.approx((10 * (f1 - 1) + 30 * (f2 - 1)) / 2)


def test_percent_per_pp_standardizes_with_the_fit_rows_and_averages_over_the_reference():
    # fit on the rows at 0% and 50% (effectiveness 0.5): a coefficient of
    # 2 * SD is an elasticity of 2; reference rows at 50% and 90%
    reference = pl.DataFrame({"estimated_mmr_cov": [50.0, 90.0]})
    f1, f2 = (0.745 / 0.75) ** 2, (0.545 / 0.55) ** 2
    out = percent_change_per_pp(
        [2 * SD_NATURAL], ROWS, "estimated_mmr_cov", reference, "natural", 0.5
    )
    assert out[0] == pytest.approx(100 * ((f1 - 1) + (f2 - 1)) / 2)
    # a coefficient per group keeps its shape; the groups are separate columns
    two = percent_change_per_pp(
        np.array([[2 * SD_NATURAL, 0.0]]),
        ROWS,
        "estimated_mmr_cov",
        reference,
        "natural",
        0.5,
    )
    assert two.shape == (1, 2)
    assert two[0, 0] == pytest.approx(out[0])
    assert two[0, 1] == pytest.approx(0.0)
    # raw scale: exp(0.1) - 1 per point whatever the reference
    raw = percent_change_per_pp(
        [0.1 * 25.0], ROWS, "estimated_mmr_cov", reference, "raw"
    )
    assert raw[0] == pytest.approx(100 * (np.exp(0.1) - 1))


def test_log_effect_per_pp_at_the_mean_coverage():
    # effectiveness 0.5, reference mean coverage 50%: d log(1 - 0.5 c / 100) /
    # dc = -0.005 / 0.75 per point, so an elasticity of 3 is 3 * 0.005 / 0.75
    # (the size, not the sign)
    reference = pl.DataFrame({"estimated_mmr_cov": [40.0, 60.0]})
    out = log_effect_per_pp(
        3 * SD_NATURAL, ROWS, "estimated_mmr_cov", reference, "natural", 0.5
    )
    assert out == pytest.approx(3 * 0.005 / 0.75)
    assert log_effect_per_pp(
        0.5 * 25.0, ROWS, "estimated_mmr_cov", reference, "raw"
    ) == pytest.approx(0.5)


def test_posterior_contrast_counts_all_pairs_with_ties_half():
    out = posterior_contrast([0, 1, 2, 3], [2, 3, 4, 5])
    # 16 pairs: 13 with after above, 2 ties (half each), none below
    assert out["P(after > before)"] == pytest.approx(14 / 16)
    assert out["two-sided p"] == pytest.approx(2 * 2 / 16)
    # paired by position the difference is 2 in every pair
    assert out["difference median"] == 2.0
    same = posterior_contrast(np.arange(100), np.arange(100))
    assert same["P(after > before)"] == pytest.approx(0.5)
    assert same["two-sided p"] == pytest.approx(1.0)


def test_unknown_scale_is_an_error():
    with pytest.raises(ValueError):
        percent_change(1.0, 1.0, "linear")


def test_labels_are_per_percentage_point():
    assert effect_label() == (
        "% change in outbreak size per 1pp\nincrease in MMR vaccine coverage"
    )
    assert "exemptions" in effect_label("exemption")
