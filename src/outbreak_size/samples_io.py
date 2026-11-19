"""
Save and load the draws the pipeline writes to outputs/: an array, or a dict
of arrays (and lists of strings, booleans or integers, such as the county
names stored beside the predictions), as a compressed .npz file.

An .npz is far smaller than the pickle of the same draws because the count
draws (almost all of the bytes) are stored in the smallest integer type that
holds them, and then compressed. Nothing is lost: the original dtype of every
array is stored and restored, so what is loaded equals what was saved bit for
bit. Loading an .npz also never runs code, as unpickling can.

Arrays that were jax arrays load as jax arrays (through jnp.asarray, as
unpickling does, so a process without numpyro.enable_x64() gets the same
32-bit arrays from either); numpy arrays load as numpy arrays. Any other
object (polars frames, tuples, scalars) is not stored: keep it as a pickle.
"""

import json
import os
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

MANIFEST = "__manifest__"
UNSIGNED = (np.uint8, np.uint16, np.uint32, np.uint64)
SIGNED = (np.int8, np.int16, np.int32, np.int64)
LIST_ITEM_TYPES = (str, bool, int, type(None))


def smallest_integer_dtype(a):
    """The smallest integer dtype that holds every value of the integer
    array a (an unsigned one when there is no negative value)."""
    if a.size == 0:
        return a.dtype
    low, high = int(a.min()), int(a.max())
    for dtype in UNSIGNED if low >= 0 else SIGNED:
        info = np.iinfo(dtype)
        if info.min <= low and high <= info.max:
            return np.dtype(dtype)
    return a.dtype


def encode(value, members):
    """The manifest entry of one value; its array, if any, is added to
    members under a numbered name (key names may contain '/' and spaces,
    which are not safe as file names)."""
    if isinstance(value, list):
        if not all(type(v) in LIST_ITEM_TYPES for v in value):
            raise TypeError(
                "a list can only hold strings, booleans, integers and None"
            )
        return {"type": "list", "values": value}
    if isinstance(value, jax.Array):
        origin = "jax"
    elif isinstance(value, np.ndarray):
        origin = "numpy"
    else:
        raise TypeError(
            f"cannot store a {type(value).__name__}: keep it as a pickle"
        )
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise TypeError(f"cannot store an array of dtype {array.dtype}")
    stored = array
    if array.dtype.kind in "iu":
        stored = array.astype(smallest_integer_dtype(array))
    name = f"a{len(members)}"
    members[name] = stored
    return {
        "type": "array",
        "member": name,
        "dtype": array.dtype.str,
        "origin": origin,
    }


def decode(entry, archive):
    """The value of a manifest entry."""
    if entry["type"] == "list":
        return list(entry["values"])
    array = archive[entry["member"]].astype(
        np.dtype(entry["dtype"]), copy=False
    )
    return jnp.asarray(array) if entry["origin"] == "jax" else array


def save_samples(path, obj):
    """
    Write obj (an array, or a dict with string keys whose values are arrays
    or lists) to the .npz file path. The file appears only when it is
    complete: it is written under another name and renamed.
    """
    if not path.endswith(".npz"):
        raise ValueError(f"the path must end in .npz: {path}")
    members = {}
    if isinstance(obj, dict):
        entries = []
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError(f"keys must be strings, not {key!r}")
            entries.append({"key": key, **encode(value, members)})
        manifest = {"kind": "dict", "entries": entries}
    elif isinstance(obj, list):
        raise TypeError("save an array or a dict, not a list")
    else:
        manifest = {"kind": "array", **encode(obj, members)}
    members[MANIFEST] = np.array(json.dumps(manifest))

    partial = path + ".part"
    try:
        with open(partial, "wb") as f:
            np.savez_compressed(f, **members)
        os.replace(partial, path)
    except BaseException:
        if os.path.exists(partial):
            os.remove(partial)
        raise


def load_samples(path) -> Any:
    """What save_samples was given: the array, or the dict with its keys in
    the order they were saved."""
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive[MANIFEST]))
        if manifest["kind"] == "array":
            return decode(manifest, archive)
        return {
            entry["key"]: decode(entry, archive)
            for entry in manifest["entries"]
        }


class LazyDraws(Mapping):
    """
    A read-only mapping from a key to the draws in an .npz file, loaded when
    the key is looked up and not kept: what one lookup returns is freed when
    the caller is done with it. The draws of all the scenarios and years
    together do not fit in memory (each array is 100 to 200 MB), but a figure
    only needs one or two at a time.

    `paths` maps each key to the path of its file; every file must exist when
    the mapping is made. `n_columns`, if given, is the number of columns
    (counties) each array must have: a lookup of an array with another number
    raises a ValueError.
    """

    def __init__(self, paths, n_columns=None):
        missing = [p for p in paths.values() if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} files of the draws are missing, "
                f"the first is {missing[0]}"
            )
        self._paths = dict(paths)
        self._n_columns = n_columns

    def __getitem__(self, key):
        draws = load_samples(self._paths[key])
        if self._n_columns is not None and draws.shape[1] != self._n_columns:
            raise ValueError(
                f"Shape mismatch for {key}: {self._paths[key]} has "
                f"{draws.shape[1]} columns, expected {self._n_columns}"
            )
        return draws

    def __iter__(self):
        return iter(self._paths)

    def __len__(self):
        return len(self._paths)
