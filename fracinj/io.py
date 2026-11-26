"""Read and write run data (HDF5, legacy pickle)."""

import pickle
from dataclasses import dataclass, replace
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
    """One run. Aperture is on ``x_vert``; pressure and stresses on ``x_sc``;
    flow rate on ``x_fz``. Check with ``validate``; the readers do."""

    t: Vector  # (n_time,)
    x_vert: Vector  # (n_vert,)
    w: Field  # (n_time, n_vert)
    x_sc: Vector  # (n_sc,)
    sn: Field  # (n_time, n_sc)
    p: Field  # (n_time, n_sc)
    params: Parameters
    q: Field | None = None
    x_fz: Vector | None = None
    tau: Field | None = None  # (n_time, n_sc)


# field -> name of the RunData grid it lives on
FIELD_GRIDS = {"w": "x_vert", "p": "x_sc", "sn": "x_sc", "tau": "x_sc", "q": "x_fz"}


def validate(run: RunData) -> RunData:
    """Raise ``ValueError`` unless ``t`` is strictly increasing and every field
    has shape ``(len(t), len(its grid))``. Returns ``run`` for chaining."""
    t = run.t
    if t.ndim != 1:
        raise ValueError(f"t must be 1-D, got shape {t.shape}")
    bad = np.flatnonzero(np.diff(t) <= 0)
    if bad.size:
        i = bad[0]
        raise ValueError(
            f"t is not strictly increasing at index {i} ({t[i]:.6g} -> {t[i + 1]:.6g}); "
            f"{bad.size} such step(s). A 3DEC restart appending to an old file?"
        )
    for name, grid_name in FIELD_GRIDS.items():
        field, grid = getattr(run, name), getattr(run, grid_name)
        if field is None:
            continue
        if grid is None:
            raise ValueError(f"{name} is present but its grid {grid_name} is not")
        if grid.ndim != 1:
            raise ValueError(f"{grid_name} must be 1-D, got shape {grid.shape}")
        expected = (t.size, grid.size)
        if field.shape != expected:
            hint = ""
            if name == "p" and field.shape[-1] == run.x_vert.size:
                hint = "; p is on the aperture grid: a surrogate run? use read_halfspace"
            raise ValueError(
                f"{name} has shape {field.shape}, expected (len(t), len({grid_name})) "
                f"= {expected}{hint}"
            )
    return run


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
OPTIONAL_FIELDS = {"tau", "x_fz", "q"}

# Pure-Python surrogate runs (scripts/run_simulation.py): aperture and pressure
# on the FVM grid, normal stress on the elastic grid, no shear stress. Older
# files name those two grids x_fvm / x_elastic (and the oldest call the
# pressure field "pressure").
HALFSPACE_PATHS = {
    "t": "coordinates/t",
    "x_vert": "coordinates/x_vertices",
    "x_sc": "coordinates/x_subcontacts",
    "w": "fields/aperture",
    "sn": "fields/stress_normal",
    "p": "fields/fluid_pressure",
}
LEGACY_HALFSPACE_PATHS = {
    **HALFSPACE_PATHS,
    "x_vert": "coordinates/x_fvm",
    "x_sc": "coordinates/x_elastic",
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


def _put_on_pressure_grid(run: RunData, x_p: Vector) -> RunData:
    """Sort every field along x and linearly interpolate ``sn`` onto the
    pressure grid ``x_p``, which becomes ``x_sc``. Outside the stress grid's
    range ``np.interp`` holds the end values."""
    x_vert, w = sort_fields(run.x_vert, run.w)
    x, p = sort_fields(x_p, run.p)
    x_sn, sn_old = sort_fields(run.x_sc, run.sn)
    sn = np.stack([np.interp(x, x_sn, row) for row in sn_old])
    return replace(run, x_vert=x_vert, w=w, x_sc=x, p=p, sn=sn)


def read_hdf5(
    filepath: Path, field_paths: dict[str, str] = FIELD_PATHS, check: bool = True
) -> RunData:
    """Read a run written in the canonical layout (see ``FIELD_PATHS``).

    For pure-Python surrogate runs written before the FFT stress solve, use
    ``read_halfspace``: their pressure is on the aperture grid, which fails the
    shape check here.
    """
    with h5py.File(str(filepath), "r") as f:
        arrays = _read_group(f, field_paths)
        raw = {k: cast(h5py.Dataset, f[f"parameters/{k}"])[()] for k in f["parameters"]}
    run = RunData(**arrays, params=_params_from_mapping(raw))
    return validate(run) if check else run


def read_halfspace(filepath: Path) -> RunData:
    """Read a pure-Python surrogate run, in the current or the legacy layout.

    These runs keep pressure on the FVM grid (``x_vert``) and stress on a
    separate elastic grid. To match the 3DEC convention (``p`` and ``sn`` both
    on ``x_sc``), ``sn`` is interpolated onto the FVM grid and ``x_sc = x_vert``.
    """
    with h5py.File(str(filepath), "r") as f:
        field_paths = LEGACY_HALFSPACE_PATHS if "coordinates/x_fvm" in f else HALFSPACE_PATHS
        if "fields/pressure" in f:
            field_paths = {**field_paths, "p": "fields/pressure"}
    run = read_hdf5(filepath, field_paths, check=False)
    return validate(_put_on_pressure_grid(run, run.x_vert))


def read_pickle(filepath: Path) -> RunData:
    """Read a legacy pickled 3DEC result dict.

    Each array's first row is x and first column is t. Pressure there is on the
    vertex grid, so ``sn`` is interpolated onto it, as in ``read_halfspace``.
    """
    with open(filepath, "rb") as f:
        results = pickle.load(f)
    run = RunData(
        t=results["w"][1:, 0],
        x_vert=results["w"][0, 1:],
        w=results["w"][1:, 1:],
        p=results["P"][1:, 1:],
        x_sc=results["sn"][0, 1:],
        sn=results["sn"][1:, 1:],
        params=_params_from_mapping(results["parameters"]),
    )
    return validate(_put_on_pressure_grid(run, results["P"][0, 1:]))


def save_run(
    filepath: Path,
    *,
    t: Vector,
    x_vert: Vector,
    w: Field,
    p: Field,
    x_sc: Vector,
    sn: Field,
    params: Parameters,
    tau: Field | None = None,
) -> None:
    """Write a run in the canonical layout read back by ``read_hdf5``."""
    validate(RunData(t=t, x_vert=x_vert, w=w, x_sc=x_sc, sn=sn, p=p, params=params, tau=tau))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    gzip = {"compression": "gzip", "compression_opts": 4}

    with h5py.File(str(filepath), "w") as f:
        coords = f.create_group("coordinates")
        coords.create_dataset("t", data=t).attrs["unit"] = "s"
        coords.create_dataset("x_vertices", data=x_vert).attrs["unit"] = "m"
        coords.create_dataset("x_subcontacts", data=x_sc).attrs["unit"] = "m"

        fields = f.create_group("fields")
        fields.create_dataset("aperture", data=w, **gzip).attrs["unit"] = "m"
        fields.create_dataset("fluid_pressure", data=p, **gzip).attrs["unit"] = "Pa"
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
    """Soultz 1993 events: (seconds since the first event, distance in m).

    Times come from the UTC columns, converted to local time (Europe/Paris:
    CEST = UTC+2, CET = UTC+1 from 1993-09-26), to match the local-time
    hydraulic record. The local columns are not used: two events (rows
    12218-12219) carry a local date one day late (17/09 instead of 16/09).
    """
    # load data file
    df = pd.read_csv(filepath, sep=";")
    # formatting string for date and time format
    fmt = "%d/%m/%Y %H:%M:%S.%f"
    # import time (UTC -> local, naive)
    timestamps = (
        pd.to_datetime(df["Date (UTC)"] + " " + df["Time (UTC)"], format=fmt)
        .dt.tz_localize("UTC")
        .dt.tz_convert("Europe/Paris")
        .dt.tz_localize(None)
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


# Basel-1 stimulation started 2006-12-02 18:00 local time (CET = UTC+1), i.e.
# 17:00 UTC (https://doi.org/10.5194/npg-23-107-2016; the digitized hydraulic
# record of Haring et al. 2008 gives 17:59:30 local, within 1 min).
BASEL_INJECTION_START_UTC = pd.Timestamp(2006, 12, 2, 17)


def _load_basel(filepath: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Basel 2006 events: (seconds since injection start, distance in m).

    Catalogue times are UTC. Events from the injection start to 2006-12-09 are
    kept (this drops 4 pre-stimulation events of 2006-11-25); distance is from
    the median location of the first 10 kept events.
    """
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
    date_mask = (df["SourceDateTime"] > BASEL_INJECTION_START_UTC) & (
        df["SourceDateTime"] < pd.Timestamp(2006, 12, 9)
    )
    df = df.loc[date_mask].reset_index(drop=True)
    t = (df["SourceDateTime"] - BASEL_INJECTION_START_UTC).dt.total_seconds()
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
