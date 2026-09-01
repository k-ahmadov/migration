"""Read and write run data (HDF5, legacy pickle)."""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import h5py
import numpy as np

from fracinj.types import Field, Parameters, Vector, parameter_names


@dataclass
class RunData:
    t: Vector  # (n_time,)
    x_vert: Vector  # (n_vert,)
    w: Field  # (n_time, n_vert)
    x_sc: Vector  # (n_sc,)
    sn: Field  # (n_time, n_sc)
    p: Field  # (n_time, n_sc)
    params: Parameters
    tau: Field | None = None  # (n_time, n_sc)


# name -> HDF5 path, for the canonical layout written by ``save_run``.
FIELD_PATHS = {
    "t": "coordinates/t",
    "x_vert": "coordinates/x_vertices",
    "x_sc": "coordinates/x_subcontacts",
    "w": "fields/aperture",
    "sn": "fields/stress_normal",
    "p": "fields/fluid_pressure",
    "tau": "fields/stress_shear",
}
OPTIONAL_FIELDS = {"tau"}

# Legacy half-space runs used different coordinate names and no shear stress.
HALFSPACE_PATHS = {
    "t": "coordinates/t",
    "x_vert": "coordinates/x_vertices",
    "x_sc": "coordinates/x_subcontacts",
    "w": "fields/aperture",
    "sn": "fields/stress_normal",
    "p": "fields/fluid_pressure",
}


def sort_fields(x: Vector, field: Field) -> tuple[Vector, Field]:
    """Return ``x`` sorted ascending and ``field`` reordered to match."""
    idx = np.argsort(x)
    return x[idx], field[:, idx]


def _params_from_mapping(raw: dict) -> Parameters:
    known = parameter_names()
    return Parameters(**{k: v for k, v in raw.items() if k in known})


def _read_group(f: h5py.File, field_paths: dict[str, str]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, path in field_paths.items():
        if name in OPTIONAL_FIELDS and path not in f:
            arrays[name] = None  # type: ignore[assignment]
        else:
            arrays[name] = cast(h5py.Dataset, f[path])[()]
    return arrays


def read_hdf5(filepath: Path, field_paths: dict[str, str] = FIELD_PATHS) -> RunData:
    """Read a run written in the canonical layout (see ``FIELD_PATHS``)."""
    with h5py.File(str(filepath), "r") as f:
        arrays = _read_group(f, field_paths)
        raw = {k: cast(h5py.Dataset, f[f"parameters/{k}"])[()] for k in f["parameters"]}
    return RunData(**arrays, params=_params_from_mapping(raw))


def read_halfspace(filepath: Path) -> RunData:
    """Read a legacy half-space run (``x_fvm`` / ``x_elastic`` coordinates)."""
    return read_hdf5(filepath, HALFSPACE_PATHS)


def read_pickle(filepath: Path) -> RunData:
    """Read a legacy pickled result dict."""
    with open(filepath, "rb") as f:
        results = pickle.load(f)
    return RunData(
        t=results["w"][1:, 0],
        x_vert=results["w"][0, 1:],
        w=results["w"][1:, 1:],
        p=results["P"][1:, 1:],
        x_sc=results["sn"][0, 1:],
        sn=results["sn"][1:, 1:],
        params=_params_from_mapping(results["parameters"]),
    )


def save_run(
    filepath: Path,
    *,
    t: Vector,
    x_vert: Vector,
    w: Field,
    p: Field,
    params: Parameters,
    x_sc: Vector | None = None,
    sn: Field | None = None,
    tau: Field | None = None,
) -> None:
    """Write a run in the canonical layout read back by ``read_hdf5``."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    gzip = dict(compression="gzip", compression_opts=4)

    with h5py.File(str(filepath), "w") as f:
        coords = f.create_group("coordinates")
        coords.create_dataset("t", data=t).attrs["unit"] = "s"
        coords.create_dataset("x_vertices", data=x_vert).attrs["unit"] = "m"
        if x_sc is not None:
            coords.create_dataset("x_subcontacts", data=x_sc).attrs["unit"] = "m"

        fields = f.create_group("fields")
        fields.create_dataset("aperture", data=w, **gzip).attrs["unit"] = "m"
        fields.create_dataset("fluid_pressure", data=p, **gzip).attrs["unit"] = "Pa"
        if sn is not None:
            fields.create_dataset("stress_normal", data=sn, **gzip).attrs["unit"] = "Pa"
        if tau is not None:
            fields.create_dataset("stress_shear", data=tau, **gzip).attrs["unit"] = "Pa"

        g = f.create_group("parameters")
        for name in parameter_names():
            value = getattr(params, name)
            if value == 0.0:
                continue  # unset driver / bound
            dset = g.create_dataset(name, data=value)
            dset.attrs["unit"] = params.unit(name)
            dset.attrs["description"] = params.description(name)
