"""Read and write run data (HDF5, legacy pickle)."""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import h5py
import numpy as np
import pandas as pd

from fracinj import paths
from fracinj.types import Field, Parameters, Vector, parameter_names

FIELD_DATA = {
    "soultz": paths.DATA / "SSFS1993-Catalogue_Bourouis.csv",
    "basel": paths.DATA / "supp2_compilation_existing_catalogs.dat",
}


@dataclass
class RunData:
    t: Vector  # (n_time,)
    x_vert: Vector  # (n_vert,)
    w: Field  # (n_time, n_vert)
    x_sc: Vector  # (n_sc,)
    x_fz: Vector
    sn: Field  # (n_time, n_sc)
    p: Field  # (n_time, n_sc)
    q: Field
    params: Parameters
    tau: Field | None = None  # (n_time, n_sc)


# name -> HDF5 path, for the canonical layout written by ``save_run``.
FIELD_PATHS = {
    "t": "coordinates/t",
    "x_vert": "coordinates/x_vertices",
    "x_sc": "coordinates/x_subcontacts",
    "x_fz": "coordinates/x_flowzones",
    "w": "fields/aperture",
    "sn": "fields/stress_normal",
    "p": "fields/fluid_pressure",
    "tau": "fields/stress_shear",
    "q": "fields/flow_rate"
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
    gzip = {"compression": "gzip", "compression_opts": 4}

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


SOULTZ_MAX_DISTANCE = 1100


def _load_soultz(filepath: str | Path) -> tuple[np.ndarray, np.ndarray]:
    # load data file
    df = pd.read_csv(filepath, sep=";")
    # formatting string for date and time format
    fmt = "%d/%m/%Y %H:%M:%S.%f"
    # import time
    timestamps = pd.to_datetime(
        df["Date (local)"] + " " + df["Time (local)"], format=fmt
    )
    # cut data to injection period
    date_mask = timestamps < timestamps.iloc[0] + pd.Timedelta(days=18)
    df = df.loc[date_mask]
    timestamps = timestamps[date_mask]
    # import time in seconds
    t = (timestamps - timestamps.iloc[0]).dt.total_seconds()
    # import coordinates of events
    x, y, z = df["East (m)"], df["North (m)"], df["Depth (m)"]
    # median of first 10 events
    x0, y0, z0 = x.iloc[:10].median(), y.iloc[:10].median(), z.iloc[:10].median()
    # euclidian distance
    distance = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
    keep = distance <= SOULTZ_MAX_DISTANCE
    return t.values[keep], distance.values[keep]


def _load_basel(filepath: str | Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(
        filepath,
        comment="#",
        sep=r"\s+",
        names=[
            "SourceDateTime",
            "LSrc",
            "Lat",
            "Lon",
            "Dep",
            "X",
            "Y",
            "Z",
            "Mwx",
            "MwGEL",
            "MwSED",
            "MLSED",
            "ID",
            "TpID",
            "GELID",
            "SEDID",
        ],
    )
    df.replace(
        {"-.--": float("nan"), "-.-": float("nan"), r"\-{3,}": float("nan")},
        inplace=True,
        regex=True,
    )
    df["SourceDateTime"] = pd.to_datetime(df["SourceDateTime"])
    for col in ["Lat", "Lon", "Dep", "X", "Y", "Z", "Mwx", "MwGEL", "MwSED", "MLSED"]:
        df[col] = pd.to_numeric(df[col])
    # injection started at 2006-12-2T18:00, source https://doi.org/10.5194/npg-23-107-2016
    date_mask = (df["SourceDateTime"] > pd.Timestamp(2006, 12, 2, 18)) & (
        df["SourceDateTime"] < pd.Timestamp(2006, 12, 9)
    )
    df = df.loc[date_mask].reset_index(drop=True)
    timestamps = df["SourceDateTime"]
    t = (timestamps - timestamps.iloc[0]).dt.total_seconds()
    x, y, z = df["X"], df["Y"], df["Z"]
    x0, y0, z0 = x.iloc[:10].median(), y.iloc[:10].median(), z.iloc[:10].median()
    distance = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
    return t.values, distance.values


def load(field: Literal["soultz", "basel"]):
    if field == "soultz":
        load_func = _load_soultz
    elif field == "basel":
        load_func = _load_basel
    else:
        raise ValueError(f"there is no data for the provided field {field!r}")
    return load_func(FIELD_DATA[field])
