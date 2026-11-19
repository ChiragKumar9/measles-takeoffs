"""Saving and loading the draws as compressed .npz files
(outbreak_size.samples_io)."""

import json
import os
import pathlib
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from outbreak_size import samples_io
from outbreak_size.samples_io import (
    LazyDraws,
    load_samples,
    save_samples,
    smallest_integer_dtype,
)


def roundtrip(tmp_path, obj):
    path = str(tmp_path / "x.npz")
    save_samples(path, obj)
    return load_samples(path)


def stored_dtype(tmp_path, member="a0"):
    """The dtype an array has inside the file (before it is restored)."""
    with np.load(tmp_path / "x.npz", allow_pickle=False) as archive:
        return archive[member].dtype


def same_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and a.tobytes() == b.tobytes()
    )


# --- the integer types on disk -------------------------------------------------


@pytest.mark.parametrize(
    "values, expected",
    [
        ([0, 255], np.uint8),
        ([0, 256], np.uint16),
        ([0, 65535], np.uint16),
        ([0, 65536], np.uint32),
        ([0, 2**32 - 1], np.uint32),
        ([0, 2**32], np.uint64),
        ([-1, 127], np.int8),
        ([-128, 127], np.int8),
        ([-129, 0], np.int16),
        ([-1, 128], np.int16),
        ([-(2**31), 2**31 - 1], np.int32),
        ([-1, 2**31], np.int64),
        ([0, 2**63 - 1], np.uint64),
    ],
)
def test_the_smallest_integer_type_that_holds_every_value(values, expected):
    assert smallest_integer_dtype(np.array(values, dtype=np.int64)) == expected


def test_integers_are_stored_small_and_restored_to_their_dtype(tmp_path):
    counts = np.array([[0, 3, 65535], [7, 0, 1]], dtype=np.int64)
    out = roundtrip(tmp_path, counts)
    assert stored_dtype(tmp_path) == np.uint16
    assert same_bits(out, counts)
    assert out.dtype == np.int64


def test_values_beyond_int32_are_exact(tmp_path):
    counts = np.array([2**40 + 7, 5, 2**31, -(2**33)], dtype=np.int64)
    out = roundtrip(tmp_path, counts)
    assert stored_dtype(tmp_path) == np.int64
    assert same_bits(out, counts)


def test_a_uint64_value_beyond_int64_is_exact(tmp_path):
    counts = np.array([2**63 + 5, 1], dtype=np.uint64)
    out = roundtrip(tmp_path, counts)
    assert same_bits(out, counts)


def test_an_empty_integer_array_keeps_its_dtype_and_shape(tmp_path):
    counts = np.zeros((0, 4), dtype=np.int64)
    assert same_bits(roundtrip(tmp_path, counts), counts)


def test_floats_and_booleans_come_back_bit_for_bit(tmp_path):
    floats = np.array(
        [0.1234567890123, -0.0, np.nan, np.inf, -np.inf, 5e-324, 1e300]
    )
    assert same_bits(roundtrip(tmp_path, floats), floats)
    single = np.array([0.1, -0.0, np.nan, np.inf, 3e38], dtype=np.float32)
    assert same_bits(roundtrip(tmp_path, single), single)
    flags = np.array([True, False, True])
    assert same_bits(roundtrip(tmp_path, flags), flags)


def test_the_shape_and_memory_order_do_not_matter(tmp_path):
    a = np.asfortranarray(np.arange(24, dtype=np.int64).reshape(2, 3, 4))
    assert same_bits(roundtrip(tmp_path, a), np.ascontiguousarray(a))
    scalar = np.array(7, dtype=np.int64)
    out = roundtrip(tmp_path, scalar)
    assert out.shape == () and out.dtype == np.int64 and out == 7


# --- dicts ---------------------------------------------------------------------


def test_a_dict_comes_back_in_order_with_awkward_keys(tmp_path):
    obj = {
        "year_1/raw_spatial_effect": np.arange(3.0),
        "baseline rate": np.arange(4, dtype=np.int64),
        "a0": np.ones(2),  # the name the file uses inside for the first array
        "__manifest__": np.zeros(1),
    }
    out = roundtrip(tmp_path, obj)
    assert list(out) == list(obj)
    for key in obj:
        assert same_bits(out[key], obj[key])


def test_lists_of_names_flags_and_missing_values_come_back_exactly(tmp_path):
    obj = {
        "predictions": np.arange(6, dtype=np.int64).reshape(2, 3),
        "fips": ["01001", "01003", "ünï"],
        "intervention_applied": [True, False, True],
        "county": ["A", None, "C"],
        "n": [1, 2, 3],
    }
    out = roundtrip(tmp_path, obj)
    assert out["fips"] == obj["fips"]
    assert out["intervention_applied"] == [True, False, True]
    assert all(type(v) is bool for v in out["intervention_applied"])
    assert out["county"] == ["A", None, "C"]
    assert out["n"] == [1, 2, 3]
    assert all(type(v) is int for v in out["n"])
    assert list(out) == list(obj)


def test_an_empty_dict_and_an_empty_list_round_trip(tmp_path):
    assert roundtrip(tmp_path, {}) == {}
    assert roundtrip(tmp_path, {"names": []}) == {"names": []}


# --- jax or numpy --------------------------------------------------------------


def test_jax_arrays_load_as_jax_and_numpy_arrays_as_numpy(tmp_path):
    obj = {
        "from_jax": jnp.arange(5, dtype=jnp.int32),
        "from_numpy": np.arange(5, dtype=np.int32),
    }
    out = roundtrip(tmp_path, obj)
    assert isinstance(out["from_jax"], jax.Array)
    assert isinstance(out["from_numpy"], np.ndarray)
    assert not isinstance(out["from_numpy"], jax.Array)
    assert same_bits(out["from_jax"], obj["from_jax"])
    assert same_bits(out["from_numpy"], obj["from_numpy"])
    top = roundtrip(tmp_path, jnp.arange(3.0))
    assert isinstance(top, jax.Array)


# --- 64-bit values and the process that loads them -----------------------------

WRITE = """
import pickle, sys
from numpyro import enable_x64
enable_x64()
import jax.numpy as jnp
from outbreak_size.samples_io import save_samples
d = {
    "cases": jnp.asarray([2**40 + 7, 5, 0]),
    "mu": jnp.asarray([0.1234567890123, 1.0, 3e15]),
}
assert d["cases"].dtype == jnp.int64 and d["mu"].dtype == jnp.float64
with open(sys.argv[1], "wb") as f:
    pickle.dump(d, f)
save_samples(sys.argv[2], d)
"""

READ = """
import pickle, sys
{enable}
from outbreak_size.samples_io import load_samples
with open(sys.argv[1], "rb") as f:
    from_pickle = pickle.load(f)
from_npz = load_samples(sys.argv[2])
for name, d in (("pickle", from_pickle), ("npz", from_npz)):
    for k in d:
        print(name, k, d[k].dtype, [repr(float(v)) for v in d[k]])
"""


# the child processes must find the package even when the tests run in an
# environment that has only the test dependencies (as in CI)
SRC = str(pathlib.Path(samples_io.__file__).resolve().parents[1])


def run(code, *args):
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([SRC, os.environ.get("PYTHONPATH", "")]),
    }
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    ).stdout.splitlines()


def test_a_process_without_x64_gets_what_unpickling_gives_it(tmp_path):
    pkl, npz = str(tmp_path / "d.pkl"), str(tmp_path / "d.npz")
    run(WRITE, pkl, npz)
    lines = run(READ.format(enable=""), pkl, npz)
    pickled = [line.split(" ", 1)[1] for line in lines if line[:6] == "pickle"]
    loaded = [line.split(" ", 1)[1] for line in lines if line[:3] == "npz"]
    assert len(pickled) == len(loaded) == 2
    assert pickled == loaded
    assert "int32" in pickled[0] and "float32" in pickled[1]  # the 32-bit load


def test_a_process_with_x64_gets_the_64_bit_values_back(tmp_path):
    pkl, npz = str(tmp_path / "d.pkl"), str(tmp_path / "d.npz")
    run(WRITE, pkl, npz)
    enable = "from numpyro import enable_x64; enable_x64()"
    lines = run(READ.format(enable=enable), pkl, npz)
    pickled = [line.split(" ", 1)[1] for line in lines if line[:6] == "pickle"]
    loaded = [line.split(" ", 1)[1] for line in lines if line[:3] == "npz"]
    assert pickled == loaded
    assert "int64" in loaded[0] and "float64" in loaded[1]
    assert "1099511627783.0" in loaded[0]  # 2**40 + 7 survived


# --- what is refused, and files ------------------------------------------------


@pytest.mark.parametrize(
    "obj",
    [
        pl.DataFrame({"a": [1]}),
        (np.arange(3), np.arange(3)),
        np.float64(1.0),
        1.5,
        {"x": np.array(["a", "b"])},  # a string array
        {"x": np.array([1, None], dtype=object)},
        {"x": ["a", 1.5]},  # a float in a list
        {"x": ["a", np.str_("b")]},  # not a plain str
        {"x": (1, 2)},
        {1: np.arange(3)},
        ["a", "b"],
    ],
)
def test_other_objects_are_refused_and_no_file_is_left(tmp_path, obj):
    path = str(tmp_path / "x.npz")
    with pytest.raises(TypeError):
        save_samples(path, obj)
    assert os.listdir(tmp_path) == []


def test_the_path_must_end_in_npz(tmp_path):
    with pytest.raises(ValueError, match="npz"):
        save_samples(str(tmp_path / "x.pkl"), np.arange(3))
    assert os.listdir(tmp_path) == []


def test_a_failed_save_leaves_an_existing_file_untouched(tmp_path):
    path = str(tmp_path / "x.npz")
    save_samples(path, {"a": np.arange(3, dtype=np.int64)})
    with pytest.raises(TypeError):
        save_samples(path, {"a": np.arange(4), "b": (1, 2)})
    assert os.listdir(tmp_path) == ["x.npz"]
    assert same_bits(load_samples(path)["a"], np.arange(3, dtype=np.int64))


def test_a_save_replaces_a_file_and_leaves_no_partial_file(tmp_path):
    path = str(tmp_path / "x.npz")
    save_samples(path, np.arange(3))
    save_samples(path, np.arange(5))
    assert os.listdir(tmp_path) == ["x.npz"]
    assert load_samples(path).tolist() == [0, 1, 2, 3, 4]


def test_the_file_is_much_smaller_than_the_pickled_counts(tmp_path):
    """Mostly zero counts, as the draws are: int64 pickles take 8 bytes per
    value, the file about a bit per value."""
    rng = np.random.default_rng(0)
    counts = (rng.random((200, 500)) < 0.05) * rng.integers(1, 20, (200, 500))
    path = tmp_path / "x.npz"
    save_samples(str(path), counts.astype(np.int64))
    assert os.path.getsize(path) < counts.size  # under one byte per value


def test_a_write_that_fails_midway_leaves_no_partial_file(
    tmp_path, monkeypatch
):
    path = str(tmp_path / "x.npz")
    save_samples(path, np.arange(3))

    def fail_after_writing(f, **members):
        f.write(b"half a file")
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez_compressed", fail_after_writing)
    with pytest.raises(OSError, match="disk full"):
        save_samples(path, np.arange(5))
    assert os.listdir(tmp_path) == ["x.npz"]
    assert load_samples(path).tolist() == [0, 1, 2]


def test_a_file_holding_a_pickled_object_is_not_loaded(tmp_path):
    path = str(tmp_path / "x.npz")
    manifest = {
        "kind": "array",
        "type": "array",
        "member": "a0",
        "dtype": "|O",
        "origin": "numpy",
    }
    np.savez(
        path,
        __manifest__=np.array(json.dumps(manifest)),
        a0=np.array([{"x": 1}], dtype=object),
    )
    with pytest.raises(ValueError, match="allow_pickle"):
        load_samples(path)


def test_the_file_loads_without_pickle(tmp_path):
    path = tmp_path / "x.npz"
    save_samples(str(path), {"a": np.arange(3), "b": ["x"]})
    with np.load(path, allow_pickle=False) as archive:
        assert sorted(archive.files) == ["__manifest__", "a0"]


# --- LazyDraws ------------------------------------------------------------------


def lazy_files(tmp_path):
    """Two files of draws (3 draws, 4 counties) and a mapping over them."""
    paths = {}
    for scenario in ("baseline", "no_change"):
        for year in (2027, 2028):
            path = str(tmp_path / f"{scenario}_{year}.npz")
            save_samples(
                path, np.full((3, 4), year + len(scenario), dtype=np.int64)
            )
            paths[(scenario, year)] = path
    return paths


def test_lazy_draws_load_what_was_saved_when_a_key_is_looked_up(tmp_path):
    draws = LazyDraws(lazy_files(tmp_path), n_columns=4)
    assert draws[("baseline", 2027)].tolist() == [[2035] * 4] * 3
    assert draws[("no_change", 2028)].dtype == np.int64


def test_lazy_draws_keep_nothing_so_a_lookup_reads_the_file_again(tmp_path):
    paths = lazy_files(tmp_path)
    draws = LazyDraws(paths)
    first = draws[("baseline", 2027)]
    save_samples(paths[("baseline", 2027)], np.zeros((3, 4), dtype=np.int64))
    # a kept array would still be the first one
    assert draws[("baseline", 2027)].sum() == 0
    assert first.sum() > 0


def test_lazy_draws_are_a_mapping_over_the_keys_of_the_grid(tmp_path):
    paths = lazy_files(tmp_path)
    draws = LazyDraws(paths)
    assert len(draws) == 4
    assert list(draws) == list(paths)
    assert ("baseline", 2028) in draws and ("baseline", 2030) not in draws
    # what the excess-cases figure asks of the draws
    assert any(k[0] == "no_change" for k in draws.keys())


def test_lazy_draws_refuse_missing_files_when_they_are_made(tmp_path):
    paths = lazy_files(tmp_path)
    paths[("baseline", 2030)] = str(tmp_path / "not_there.npz")
    with pytest.raises(FileNotFoundError, match="1 files.*not_there"):
        LazyDraws(paths)


def test_lazy_draws_refuse_an_array_with_another_number_of_counties(tmp_path):
    draws = LazyDraws(lazy_files(tmp_path), n_columns=5)
    with pytest.raises(ValueError, match="Shape mismatch.*4 columns"):
        draws[("baseline", 2027)]


def test_lazy_draws_of_an_unknown_key_raise_a_key_error(tmp_path):
    with pytest.raises(KeyError):
        LazyDraws(lazy_files(tmp_path))[("baseline", 1999)]
