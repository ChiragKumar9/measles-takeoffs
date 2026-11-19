"""
Convergence of an MCMC fit, and a refit with the non-centred CAR field when the
centred fit has not converged (see car_noncentered.py and the
`non_centered_car` option of the models). The refit is triggered by an r_hat
above RHAT_MAX; a caller can also ask for it on divergent transitions and on an
effective sample size below a floor (`max_divergences`, `neff_min`; script 02
does, with MAX_DIVERGENCES and NEFF_MIN). By default only r_hat counts, as in
scripts 03, 06 and 07.
"""

import fcntl
import os

import numpy as np
import polars as pl
from jax.random import PRNGKey
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS

# a fit whose largest r_hat is above this is refit with the non-centred CAR
RHAT_MAX = 1.01
# the stricter checks of script 02 (a caller passes them to fit_with_fallback;
# the default is to check r_hat only): more divergent transitions than this ...
MAX_DIVERGENCES = 0
# ... or an effective sample size of some parameter below this (about 100
# independent draws per chain with four chains)
NEFF_MIN = 400


def convergence_problems(
    summary_: dict,
    rhat_max: float = RHAT_MAX,
    neff_min: float | None = None,
    max_divergences: int | None = None,
) -> list[str]:
    """What is wrong with a fit, from its `convergence_summary`: the checks of
    r_hat, divergences and effective sample size that it fails (an empty list
    for a fit that passes them). A check whose limit is None is not made."""
    problems = []
    if summary_["max_rhat"] > rhat_max:
        problems.append(
            f"r_hat {summary_['max_rhat']:.3f} ({summary_['worst_site']}) "
            f"is above {rhat_max}"
        )
    if (
        max_divergences is not None
        and summary_["divergences"] > max_divergences
    ):
        problems.append(f"{summary_['divergences']} divergent transitions")
    if neff_min is not None and summary_["min_neff"] < neff_min:
        problems.append(
            f"effective sample size {summary_['min_neff']:.0f} is below "
            f"{neff_min:.0f}"
        )
    return problems


def convergence_summary(mcmc) -> dict:
    """
    The largest r_hat and the smallest effective sample size over the sampled
    parameters of a fit (its deterministic sites left out, as numpyro's own
    print_summary does), the parameter with the largest r_hat, the number
    of divergent transitions, and the share of transitions at the maximum
    tree depth and the smallest BFMI (see below).
    """
    draws = mcmc.get_samples(group_by_chain=True)
    sampled = mcmc.last_state.z
    stats = summary(
        {site: v for site, v in draws.items() if site in sampled},
        group_by_chain=True,
    )
    max_rhat, worst_site, min_neff = -np.inf, "", np.inf
    for site, stat in stats.items():
        rhat = np.nanmax(np.asarray(stat["r_hat"]))
        if rhat > max_rhat:
            max_rhat, worst_site = float(rhat), site
        min_neff = min(min_neff, float(np.nanmin(np.asarray(stat["n_eff"]))))
    summary_ = dict(
        max_rhat=max_rhat,
        min_neff=min_neff,
        worst_site=worst_site,
        divergences=int(np.sum(mcmc.get_extra_fields()["diverging"])),
    )
    # the two other checks of NUTS (only when the fit was run with the extra
    # fields `num_steps` and `energy`, as fit_with_fallback does): the share
    # of transitions that stopped at the maximum tree depth (the sampler was
    # cut short), and the smallest energy fraction of missing information of
    # a chain (below about 0.3 the momentum resampling explores the energy
    # distribution too slowly)
    fields = mcmc.get_extra_fields(group_by_chain=True)
    if "num_steps" in fields and "energy" in fields:
        depth = getattr(mcmc.sampler, "_max_tree_depth", 10)
        depth = max(depth) if isinstance(depth, tuple) else depth
        energy = np.asarray(fields["energy"], dtype=float)  # chains x draws
        summary_["treedepth_saturation"] = float(
            np.mean(np.asarray(fields["num_steps"]) >= 2**depth - 1)
        )
        summary_["min_bfmi"] = float(
            np.min(
                np.mean(np.diff(energy, axis=1) ** 2, axis=1)
                / np.var(energy, axis=1)
            )
        )
    return summary_


def record_convergence(path: str, row: dict) -> None:
    """
    Add `row` to the csv at `path`, replacing the row of the same label.

    Safe when several processes record to the same file (the two lanes of the
    rolling validation): the file is locked while it is read, edited and
    replaced, and it is replaced as a whole, so a reader never sees half of it
    and no row of another process is lost.
    """
    with open(path + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        new = pl.DataFrame([row])
        if os.path.exists(path):
            old = pl.read_csv(path, schema_overrides={"label": pl.String})
            new = pl.concat(
                [old.filter(pl.col("label") != row["label"]), new],
                how="diagonal_relaxed",
            )
        new.write_csv(path + ".part")
        os.replace(path + ".part", path)


def fit_with_fallback(
    build_model,
    run_kwargs: dict,
    *,
    seed: int,
    non_centered_car: bool,
    label: str,
    init_strategy,
    target_accept_prob: float,
    record_path: str,
    rhat_max: float = RHAT_MAX,
    neff_min: float | None = None,
    max_divergences: int | None = None,
    spatial: bool = True,
    num_warmup: int = 2000,
    num_samples: int = 2000,
    num_chains: int = 4,
):
    """
    Fit a model, and refit it with the non-centred CAR field if the fit has
    a parameter with r_hat above `rhat_max` (and, when they are given, more than
    `max_divergences` divergent transitions or a parameter with an effective
    sample size below `neff_min`), and was not already non-centred (a model with
    no spatial term, `spatial=False`, is never refit: it has no CAR field).

    build_model(non_centered_car) returns the model (an object with a `fit`
    method); run_kwargs are the arguments of that method's data. Returns the
    model and the MCMC of the fit kept (the non-centred one after a refit,
    even if it also fails a check), and writes the diagnostics of the fit kept,
    and of the centred fit and why it was refit if it was, to `record_path`.
    """

    def fit(non_centered):
        model = build_model(non_centered)
        mcmc = MCMC(
            NUTS(
                model.fit,
                init_strategy=init_strategy,
                target_accept_prob=target_accept_prob,
            ),
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
        )
        mcmc.run(
            PRNGKey(seed),
            extra_fields=("num_steps", "energy"),
            **run_kwargs,
        )
        return model, mcmc

    model, mcmc = fit(non_centered_car)
    kept = convergence_summary(mcmc)
    first = None
    reasons = convergence_problems(kept, rhat_max, neff_min, max_divergences)
    # a model with no spatial term has no CAR field to refit
    if spatial and not non_centered_car and reasons:
        first = kept
        print(
            f"{label}: {'; '.join(reasons)}; "
            "refitting with the non-centred CAR field"
        )
        model, mcmc = fit(True)
        kept = convergence_summary(mcmc)
    print(
        f"{label}: kept the {'non-centred' if non_centered_car or first else 'centred'} fit, "
        f"max r_hat {kept['max_rhat']:.3f} ({kept['worst_site']}), "
        f"min n_eff {kept['min_neff']:.0f}, divergences {kept['divergences']}"
    )
    row = dict(
        label=label,
        non_centered_car=non_centered_car or first is not None,
        **kept,
        centred_max_rhat=None if first is None else first["max_rhat"],
        centred_min_neff=None if first is None else first["min_neff"],
        centred_divergences=None if first is None else first["divergences"],
        refit_reason=None if first is None else "; ".join(reasons),
    )
    record_convergence(record_path, row)
    return model, mcmc
