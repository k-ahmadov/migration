import dataclasses
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import h5py
import numpy as np

from mypackages.types import Field, Parameters, Time, XPositions


@dataclass
class RunData:
    t: Time  # shape: (n_time, )
    x_vert: XPositions  # shape: (n_vert, )
    w: Field  # shape: (n_time, n_vert)
    x_sc: XPositions  # shape: (n_sc, )
    sn: Field  # shape: (n_time, n_sc)
    tau: Field | None  # shape: (n_time, n_sc)
    p: Field  # shape: (n_time, n_sc)
    params: Parameters


FIELD_PATHS = {
    "x_vert": "coordinates/x_vertices",
    "t": "coordinates/t",
    "w": "fields/aperture",
    "x_sc": "coordinates/x_subcontacts",
    "sn": "fields/stress_normal",
    "tau": "fields/stress_shear",
    "p": "fields/fluid_pressure",
}


def sort_fields(
    x: XPositions,
    field: Field,
) -> tuple[XPositions, Field]:
    idx = np.argsort(x)
    return x[idx], field[:, idx]


def read_run(filepath: Path) -> RunData:
    with h5py.File(str(filepath), "r") as f:
        arrays = {k: cast(h5py.Dataset, f[path])[()] for k, path in FIELD_PATHS.items()}
        params = {
            k: cast(h5py.Dataset, f[f"parameters/{k}"])[()]
            for k in cast(h5py.Group, f["parameters"]).keys()
        }
        known = {f.name for f in dataclasses.fields(Parameters)}
        params = Parameters(**{k: v for k, v in params.items() if k in known})

    return RunData(**arrays, params=params)


def read_pickle(filepath: Path) -> RunData:
    with open(filepath, "rb") as f:
        results = pickle.load(f)
        arrays = {
            "t": results["w"][1:, 0],
            "x_vert": results["w"][0, 1:],
            "w": results["w"][1:, 1:],
            "p": results["P"][1:, 1:],
            "x_sc": results["sn"][0, 1:],
            "sn": results["sn"][1:, 1:],
        }
        params_raw = results["parameters"]

        known = {f.name for f in dataclasses.fields(Parameters)}
        params = Parameters(**{k: v for k, v in params_raw.items() if k in known})
    return RunData(**arrays, tau=None, params=params)


def read_fvm(filepath: Path) -> RunData:
    with h5py.File(str(filepath), "r") as f:
        x_el = cast(h5py.Dataset, f["coordinates/x_elastic"])[:]
        x_fvm = cast(h5py.Dataset, f["coordinates/x_fvm"])[:]
        t = cast(h5py.Dataset, f["coordinates/t"])[:]
        w_tx = cast(h5py.Dataset, f["fields/aperture"])[:]
        sn_tx = cast(h5py.Dataset, f["fields/stress_normal"])[:]
        p_tx = cast(h5py.Dataset, f["fields/fluid_pressure"])[:]
        params = {
            k: cast(h5py.Dataset, f[f"parameters/{k}"])[()]
            for k in cast(h5py.Group, f["parameters"]).keys()
        }
        known = {f.name for f in dataclasses.fields(Parameters)}
        params = Parameters(**{k: v for k, v in params.items() if k in known})
    return RunData(
        t=t, x_vert=x_fvm, w=w_tx, x_sc=x_el, sn=sn_tx, p=p_tx, params=params, tau=None
    )
