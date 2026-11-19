"""The panels a, b and c of analysis/figure_validation_and_risk_projections.py
(the plotting functions, on synthetic numbers)."""

import importlib.util
import os
import sys

import numpy as np
import polars as pl
import pytest

for module in ("matplotlib", "geopandas", "jax"):
    pytest.importorskip(module)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.container as mcontainer  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from outbreak_size.headline_panels import PERFORMANCE_METRICS  # noqa: E402
from outbreak_size.model_colors import (  # noqa: E402
    MECHANISTIC_COLOR,
    STATISTICAL_COLOR,
)

ROOT = os.path.dirname(os.path.dirname(__file__))
SCRIPT = os.path.join(
    ROOT, "analysis", "figure_validation_and_risk_projections.py"
)


@pytest.fixture(scope="module")
def figure():
    """The script as a module (its main program does not run)."""
    import jax

    cwd, argv = os.getcwd(), sys.argv
    os.chdir(ROOT)  # the script opens params.yml from the working directory
    sys.argv = ["x"]
    x64 = jax.config.jax_enable_x64
    try:
        spec = importlib.util.spec_from_file_location("validation_fig", SCRIPT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        os.chdir(cwd)
        sys.argv = argv
        jax.config.update("jax_enable_x64", x64)
    return module


def count_error_bars(ax):
    return sum(
        isinstance(c, mcontainer.ErrorbarContainer) for c in ax.containers
    )


def count_violins(ax):
    return sum("PolyCollection" in type(c).__name__ for c in ax.collections)


def performance(later_has_intervals):
    rows = []
    for metric in PERFORMANCE_METRICS:
        for model in ("Statistical", "Mechanistic"):
            for condition in ("All cases", "Outbreak cases"):
                for period in ("2022-2025", "2026"):
                    has = period == "2022-2025" or later_has_intervals
                    rows.append(
                        {
                            "metric": metric,
                            "model": model,
                            "condition": condition,
                            "period": period,
                            "value": 50.0,
                            "lower": 40.0 if has else np.nan,
                            "upper": 60.0 if has else np.nan,
                        }
                    )
    return pl.DataFrame(rows)


def test_each_metric_has_eight_bars_the_models_alternating(figure):
    fig, ax = plt.subplots()
    figure.plot_performance_bars(ax, performance(False), "2022-2025", "2026")
    # an all-cases bar is one patch, an outbreak-case bar two (its hatch and
    # its outline): 4 + 4 * 2 patches in each of the 6 metrics
    assert len(ax.patches) == 6 * (4 + 4 * 2)
    # the bars that fill their patch alternate statistical, mechanistic
    filled = [p for p in ax.patches if p.get_fill()]
    assert len(filled) == 6 * 8
    colors = [
        mcolors.to_hex(p.get_edgecolor())
        for p in ax.patches
        if not p.get_hatch()
    ]
    expected = [
        mcolors.to_hex(c) for c in (STATISTICAL_COLOR, MECHANISTIC_COLOR)
    ]
    # the first metric: all cases, earlier then later period, models alternating
    first = [mcolors.to_hex(p.get_edgecolor()) for p in ax.patches[:4]]
    assert first == expected * 2
    assert len(colors) > 0
    # the bars of the later period are lighter than those of the earlier one
    alphas = [p.get_facecolor()[3] for p in ax.patches[:4]]
    assert alphas[0] > alphas[2] and alphas[1] > alphas[3]
    # the outbreak cases are hatched
    assert sum(1 for p in ax.patches if p.get_hatch()) == 6 * 4
    assert list(ax.get_xticklabels())[0].get_text() == "AUC"
    assert [t.get_text() for t in ax.get_xticklabels()] == list(
        PERFORMANCE_METRICS
    )
    assert ax.get_ylim() == (0, 100)
    # two legends, each in a box
    assert ax.get_legend() is not None and len(ax.artists) == 1
    plt.close(fig)


def test_a_period_without_intervals_has_no_error_bars(figure):
    fig, ax = plt.subplots()
    figure.plot_performance_bars(ax, performance(False), "2022-2025", "2026")
    # one error bar for each bar of the earlier period only
    assert count_error_bars(ax) == 6 * 4
    fig2, ax2 = plt.subplots()
    figure.plot_performance_bars(ax2, performance(True), "2022-2025", "2026")
    assert count_error_bars(ax2) == 6 * 8
    plt.close(fig)
    plt.close(fig2)


def test_a_violin_needs_three_different_values(figure):
    fig, ax = plt.subplots()
    figure.plot_violin(ax, [5.0, 5.0], 0, "k", 0.5)  # no density
    assert count_violins(ax) == 0 and len(ax.collections) == 1  # the median
    figure.plot_violin(ax, np.arange(100.0), 1, "k", 0.5)
    assert count_violins(ax) == 1  # and now a violin
    plt.close(fig)


def test_the_totals_panel_has_a_tick_at_each_power_of_ten_it_reaches(figure):
    fig, ax = plt.subplots()
    rng = np.random.default_rng(0)
    figure.plot_2026_totals(
        ax,
        2664,
        rng.lognormal(7, 0.5, 500),  # about 1,100 (95% from ~400 to ~3,000)
        rng.lognormal(6.3, 0.1, 500),
    )
    ticks = [t.get_text() for t in ax.get_yticklabels()]
    assert ticks[:1] == ["100"] and ticks[-1] == "10,000"
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        "Observed\n(September, 2026)",
        "Statistical",
        "Mechanistic",
    ]
    assert ax.get_yscale() == "log"
    plt.close(fig)


def test_the_risk_panel_has_three_violins_in_each_bucket_and_skips_empty_ones(
    figure,
):
    rng = np.random.default_rng(1)
    cases = np.array([0] * 300 + [3] * 40 + [12] * 8)
    percentiles = {
        name: rng.uniform(0, 100, len(cases))
        for name in ("Statistical", "Mechanistic", "Vaccine coverage only")
    }
    fig, ax = plt.subplots()
    figure.plot_2026_risk_percentiles(ax, cases, percentiles)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["0", "1–9", "≥10"]
    assert count_violins(ax) == 9  # 3 buckets x 3 rankings
    plt.close(fig)
    # a partial year may have no place with 10 or more cases
    fig, ax = plt.subplots()
    figure.plot_2026_risk_percentiles(
        ax, cases[:340], {k: v[:340] for k, v in percentiles.items()}
    )
    assert count_violins(ax) == 6
    plt.close(fig)


# --- the case-concentration figure -----------------------------------------


def synthetic_concentration_inputs(places=40, draws=30):
    import jax.numpy as jnp

    from outbreak_size.concentration import rankings, validation_curves

    rng = np.random.default_rng(5)
    rate = rng.gamma(0.5, 0.4, size=(draws, places))
    cases = rng.poisson(rate.mean(axis=0) * 3)

    def year_curves():
        return validation_curves(
            rankings(
                rng.gamma(2.0, 5.0, size=(draws, places)),
                rng.gamma(2.0, 5.0, size=(draws, places)),
                1 - np.exp(-rate),
                rate,
                rng.uniform(80, 99, places),
                rng.integers(1_000, 100_000, places).astype(float),
                cases,
            ),
            cases,
        )

    scenarios = [
        "no_change",
        "baseline",
        "targeted_improvement",
        "targeted_worsening",
    ]
    samples = {
        (s, y): jnp.array(rng.gamma(2.0, 3.0, size=(draws, places)))
        for s in scenarios
        for y in (2027, 2028, 2029, 2030)
    }
    colors = dict(zip(scenarios, ["gray", "brown", "green", "red"]))
    return year_curves, scenarios, samples, colors, rate


def test_the_concentration_figure_has_two_validation_and_three_scenario_panels(
    figure, tmp_path, monkeypatch
):
    import jax.numpy as jnp

    from outbreak_size.concentration import RANKINGS, mean_curve

    year_curves, scenarios, samples, colors, rate = (
        synthetic_concentration_inputs()
    )
    first, second = year_curves(), year_curves()
    panels = [
        (
            "averaged",
            {n: mean_curve([first[n], second[n]]) for n in RANKINGS},
        ),
        ("2026", year_curves()),
    ]
    monkeypatch.setattr(figure.plt, "close", lambda fig: None)
    output = str(tmp_path / "concentration")
    figure.plot_case_concentration_curves(
        panels,
        scenarios,
        colors,
        samples,
        jnp.array(1 - np.exp(-rate)),
        jnp.array(rate),
        output,
    )
    axes = plt.gcf().axes
    assert len(axes) == 5
    assert os.path.exists(output + ".png") and os.path.exists(output + ".pdf")
    # the two validation panels draw the six rankings and the diagonal, with one
    # boxed legend (in the first) that names them in order
    for ax in axes[:2]:
        assert len(ax.lines) == len(RANKINGS) + 1
    legend = axes[0].get_legend()
    assert [t.get_text() for t in legend.get_texts()] == RANKINGS
    assert axes[1].get_legend() is None
    colors_by_name = {
        line.get_label(): line.get_color() for line in axes[0].lines
    }
    assert colors_by_name["Statistical model"] == STATISTICAL_COLOR
    assert colors_by_name["Mechanistic model"] == MECHANISTIC_COLOR
    assert colors_by_name["Vaccine coverage alone"] == "gray"
    # the three scenario panels draw the four scenarios and the diagonal, with
    # a boxed legend of their own in the first
    for ax in axes[2:]:
        assert len(ax.lines) == len(scenarios) + 1
    labels = [t.get_text() for t in axes[2].get_legend().get_texts()]
    assert len(labels) == len(scenarios)
    assert axes[3].get_legend() is None
    assert axes[0].get_ylim() == axes[1].get_ylim() == (0.0, 100.0)
    # every curve is a cumulative share: it never goes down and ends at 100
    for ax in axes:
        for line in ax.lines:
            if line.get_label().startswith("_"):
                continue
            y = np.asarray(line.get_ydata())
            assert (np.diff(y) >= -1e-9).all()
            assert y[-1] == pytest.approx(100.0)
    plt.close("all")
