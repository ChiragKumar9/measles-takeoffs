"""Convergence check and the refit with the non-centred CAR (outbreak_size.convergence)."""

import os
from concurrent.futures import ThreadPoolExecutor

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import polars as pl
from numpyro.infer import init_to_median

from outbreak_size import convergence
from outbreak_size.convergence import fit_with_fallback


class Normal:
    """A model of data y ~ Normal(mu, 1) with mu ~ Normal(0, 10), plus a
    deterministic site (as the real models have)."""

    def __init__(self, non_centered):
        self.non_centered = non_centered

    def fit(self, y):
        mu = numpyro.sample("mu", dist.Normal(0, 10))
        numpyro.deterministic("twice_mu", 2 * mu)
        numpyro.sample("y", dist.Normal(mu, 1), obs=y)


Y = jnp.array([1.0, 2.0, 3.0])
SMALL = dict(num_warmup=100, num_samples=100, num_chains=2)


class Funnel:
    """Neal's funnel, which a centred sampler cannot explore (divergent
    transitions); the non-centred form samples the same posterior without."""

    def __init__(self, non_centered):
        self.non_centered = non_centered

    def fit(self, y):
        v = numpyro.sample("v", dist.Normal(0, 3))
        if self.non_centered:
            raw = numpyro.sample("raw", dist.Normal(0, 1))
            numpyro.deterministic("x", raw * jnp.exp(v / 2))
        else:
            numpyro.sample("x", dist.Normal(0, jnp.exp(v / 2)))


def fit(
    tmp_path,
    rhat_max,
    non_centered_car=False,
    spatial=True,
    model_class=Normal,
    **checks,
):
    built = []

    def build_model(non_centered):
        built.append(non_centered)
        return model_class(non_centered)

    path = str(tmp_path / "convergence.csv")
    model, mcmc = fit_with_fallback(
        build_model,
        dict(y=Y),
        seed=1,
        non_centered_car=non_centered_car,
        label="toy",
        init_strategy=init_to_median(),
        target_accept_prob=0.8,
        record_path=path,
        rhat_max=rhat_max,
        spatial=spatial,
        **checks,
        **SMALL,
    )
    return built, model, mcmc, pl.read_csv(path)


def test_the_summary_covers_the_sampled_parameters_only(tmp_path):
    _, _, mcmc, _ = fit(tmp_path, rhat_max=10.0)
    summary = convergence.convergence_summary(mcmc)
    assert summary["worst_site"] == "mu"  # not the deterministic site
    assert 0.99 < summary["max_rhat"] < 1.2
    assert summary["min_neff"] > 20
    assert summary["divergences"] == 0


def test_a_fit_below_the_threshold_is_kept_as_it_is(tmp_path):
    built, model, _, record = fit(tmp_path, rhat_max=10.0)
    assert built == [False]
    assert not model.non_centered
    assert record["non_centered_car"].to_list() == [False]
    assert record["centred_max_rhat"].null_count() == 1


def test_a_fit_above_the_threshold_is_refit_non_centred(tmp_path):
    built, model, _, record = fit(tmp_path, rhat_max=0.5)  # always exceeded
    assert built == [False, True]
    assert model.non_centered
    assert record["non_centered_car"].to_list() == [True]
    # the centred fit's diagnostics are kept next to those of the fit kept
    assert record["centred_max_rhat"][0] > 0.5
    assert record["centred_min_neff"][0] > 0


def test_a_fit_already_non_centred_is_not_refit(tmp_path):
    built, _, _, record = fit(tmp_path, rhat_max=0.5, non_centered_car=True)
    assert built == [True]
    assert record["non_centered_car"].to_list() == [True]


def test_a_model_without_a_spatial_term_is_never_refit(tmp_path):
    built, _, _, record = fit(
        tmp_path, rhat_max=0.5, spatial=False
    )  # always exceeded
    assert built == [False]
    assert record["non_centered_car"].to_list() == [False]
    assert record["centred_max_rhat"].null_count() == 1


def test_records_written_at_the_same_time_all_survive(tmp_path):
    # the two lanes of the rolling validation record to one file
    path = str(tmp_path / "record.csv")
    labels = [f"fit {i}" for i in range(24)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(
            pool.map(
                lambda label: convergence.record_convergence(
                    path, dict(label=label, max_rhat=1.0)
                ),
                labels,
            )
        )
    assert sorted(pl.read_csv(path)["label"].to_list()) == sorted(labels)
    assert not os.path.exists(path + ".part")


def test_a_record_replaces_the_row_of_the_same_label(tmp_path):
    path = str(tmp_path / "record.csv")
    row = dict(label="a", max_rhat=1.2)
    convergence.record_convergence(path, row)
    convergence.record_convergence(path, dict(label="b", max_rhat=1.0))
    convergence.record_convergence(path, dict(label="a", max_rhat=1.05))
    record = pl.read_csv(path).sort("label")
    assert record["label"].to_list() == ["a", "b"]
    assert record["max_rhat"].to_list() == [1.05, 1.0]


def test_the_summary_reports_tree_depth_saturation_and_bfmi(tmp_path):
    _, _, mcmc, record = fit(tmp_path, rhat_max=10.0)
    summary = convergence.convergence_summary(mcmc)
    assert summary["treedepth_saturation"] == 0.0  # an easy toy model
    assert 0.3 < summary["min_bfmi"] < 3  # a healthy chain is about 1
    assert (
        "min_bfmi" in record.columns
        and "treedepth_saturation" in record.columns
    )


def test_the_default_checks_r_hat_only_as_in_scripts_06_and_07(tmp_path):
    # an effective sample size that no fit of 200 draws can reach, and a
    # divergence limit, are not checked unless the caller asks for them
    built, _, _, record = fit(tmp_path, rhat_max=10.0, model_class=Funnel)
    assert built == [False]
    assert record["refit_reason"].null_count() == 1


def test_a_low_effective_sample_size_triggers_the_refit_when_asked(tmp_path):
    built, model, _, record = fit(tmp_path, rhat_max=10.0, neff_min=1e9)
    assert built == [False, True]
    assert model.non_centered
    assert "effective sample size" in record["refit_reason"][0]
    assert "r_hat" not in record["refit_reason"][0]  # r_hat was fine


def test_a_sufficient_effective_sample_size_does_not_trigger_it(tmp_path):
    built, _, _, record = fit(tmp_path, rhat_max=10.0, neff_min=1.0)
    assert built == [False]
    assert record["refit_reason"].null_count() == 1


def test_a_divergent_transition_triggers_the_refit_when_asked(tmp_path):
    built, model, mcmc, record = fit(
        tmp_path,
        rhat_max=10.0,
        model_class=Funnel,
        max_divergences=0,
    )
    assert built == [False, True]
    assert model.non_centered
    assert record["centred_divergences"][0] > 0
    assert "divergent transitions" in record["refit_reason"][0]
    # the non-centred form of the same posterior has fewer or no divergences
    assert record["divergences"][0] < record["centred_divergences"][0]


def test_the_reason_lists_every_check_that_fails(tmp_path):
    _, _, _, record = fit(
        tmp_path, rhat_max=0.5, neff_min=1e9, model_class=Normal
    )
    reason = record["refit_reason"][0]
    assert "r_hat" in reason and "effective sample size" in reason


def test_a_model_without_a_spatial_term_is_never_refit_on_neff_either(
    tmp_path,
):
    built, _, _, record = fit(
        tmp_path, rhat_max=10.0, spatial=False, neff_min=1e9
    )
    assert built == [False]
    assert record["refit_reason"].null_count() == 1


def test_convergence_problems_names_each_failed_check():
    ok = dict(max_rhat=1.005, worst_site="a", min_neff=900.0, divergences=0)
    assert (
        convergence.convergence_problems(ok, neff_min=400, max_divergences=0)
        == []
    )
    bad = dict(max_rhat=1.02, worst_site="b", min_neff=300.0, divergences=3)
    assert (
        len(
            convergence.convergence_problems(
                bad, neff_min=400, max_divergences=0
            )
        )
        == 3
    )
    # the limits that are None are not checked
    assert len(convergence.convergence_problems(bad)) == 1
    # r_hat exactly at the limit is acceptable
    edge = dict(max_rhat=1.01, worst_site="c", min_neff=400.0, divergences=0)
    assert (
        convergence.convergence_problems(edge, neff_min=400, max_divergences=0)
        == []
    )
