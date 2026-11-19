"""region_members() of analysis/figure_predicting_county_level_outbreaks.py."""

import importlib.util
import os
import sys

import pytest

# the figure script needs the plotting stack; skip where it is not installed
for module in ("matplotlib", "geopandas", "jax", "addfips"):
    pytest.importorskip(module)

SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "analysis",
    "figure_predicting_county_level_outbreaks.py",
)


def load_script():
    import jax

    # the script calls enable_x64() when it is loaded: put the setting back,
    # or every test after this one runs in double precision (the short MCMC
    # run of test_poisson_recovery then misses its tolerance)
    argv, sys.argv = sys.argv, ["x"]
    x64 = jax.config.jax_enable_x64
    try:
        spec = importlib.util.spec_from_file_location("outbreak_fig", SCRIPT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.argv = argv
        jax.config.update("jax_enable_x64", x64)
    return module


def test_the_georgia_places_come_from_regions_py_and_district_1_is_region_1():
    members = load_script().region_members()
    # both Georgia places are keys of regions.py now, with their old lists
    assert members["Fulton, Gwinett"] == ["13121", "13135"]
    assert members["Forsyth, Fulton, Paulding"] == ["13117", "13121", "13223"]
    assert "Fulton and Gwynett" not in members
    # "District 1" is "Region 1, Louisiana" (the case data's name for it)
    assert members["District 1"] == members["Region 1, Louisiana"]
