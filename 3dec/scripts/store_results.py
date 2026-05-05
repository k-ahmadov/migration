from pathlib import Path
from typing import Callable, cast

import h5py
import itasca as it  # pyright: ignore[reportMissingImports]
import numpy as np

# Type aliases for clarity
FindFunc = Callable[[int], object | None]
FieldFunc = Callable[[object], float]


# ---------------------------------------------------------------------------
# HDF5 mesh ID loading
# ---------------------------------------------------------------------------


class MeshIDs:
    """Stores element IDs loaded from the coordinates group of an HDF5 file."""

    vertices: np.ndarray
    subcontacts: np.ndarray
    flowzones: np.ndarray

    def __init__(self, filepath: Path | str) -> None:
        with h5py.File(filepath, "r") as f:
            coords = cast(h5py.Group, f["coordinates"])
            self.vertices = cast(h5py.Dataset, coords["id_vertices"])[()]
            self.subcontacts = cast(h5py.Dataset, coords["id_subcontacts"])[()]
            self.flowzones = cast(h5py.Dataset, coords["id_flowzones"])[()]


# ---------------------------------------------------------------------------
# Field extraction from ITASCA model
# ---------------------------------------------------------------------------


def _extract_field(
    element_ids: np.ndarray,
    find_func: FindFunc,
    field_func: FieldFunc,
    element_label: str,
) -> np.ndarray:
    """
    Walk *element_ids*, look each one up with *find_func*, and collect the
    scalar returned by *field_func* into a float64 array.
    """
    values: list[float] = []
    for elem_id in element_ids:
        obj = find_func(elem_id)
        if obj is None:
            raise ValueError(
                f"ID {elem_id} not found in model for element type '{element_label}'"
            )
        values.append(field_func(obj))
    return np.asarray(values, dtype=np.float64)


class Fields:
    """Physical field profiles extracted from the current ITASCA model state."""

    sn_eff: np.ndarray  # effective normal stress
    fluid_pressure: np.ndarray
    sn: np.ndarray  # total normal stress  (sn_eff + fluid_pressure)
    tau: np.ndarray  # shear stress
    w: np.ndarray  # hydraulic aperture
    q: np.ndarray  # discharge (x-component)
    v: np.ndarray  # velocity  (x-component)

    def __init__(self, mesh_ids: MeshIDs) -> None:
        sc = it.block.subcontact
        fp = it.flowplane

        def restore(
            element_type: str,
            find_func: FindFunc,
            field_func: FieldFunc,
        ) -> np.ndarray:
            return _extract_field(
                getattr(mesh_ids, element_type),
                find_func,
                field_func,
                element_label=element_type,
            )

        self.sn_eff = restore("subcontacts", sc.find, sc.Subcontact.stress_norm)
        self.fluid_pressure = restore("subcontacts", sc.find, sc.Subcontact.pp)
        self.sn = self.sn_eff + self.fluid_pressure
        self.tau = restore("subcontacts", sc.find, sc.Subcontact.stress_shear)
        self.w = restore(
            "vertices", fp.vertex.find, fp.vertex.Vertex.aperture_hydraulic
        )
        self.q = restore("flowzones", fp.zone.find, fp.zone.Zone.discharge_x)
        self.v = restore("flowzones", fp.zone.find, fp.zone.Zone.velocity_x)


# ---------------------------------------------------------------------------
# HDF5 helpers — extendable datasets
# ---------------------------------------------------------------------------


def _require_profile_1d(
    group: h5py.Group,
    name: str,
    n: int,
    dtype: np.dtype = np.float64,
) -> h5py.Dataset:
    """
    Return an extendable dataset shaped ``(nt, n)`` inside *group*, creating
    it on first call.  Raises if an existing dataset has a mismatched width.
    """
    if name in group:
        ds = group[name]
        if ds.shape[1] != n:
            raise ValueError(f"'{name}': existing width {ds.shape[1]} != new width {n}")
        return ds

    return group.create_dataset(
        name,
        shape=(0, n),
        maxshape=(None, n),
        dtype=dtype,
        compression="gzip",
        compression_opts=4,
    )


def _require_scalar_series(
    group: h5py.Group,
    name: str,
    dtype: np.dtype = np.float64,
) -> h5py.Dataset:
    """Return an extendable 1-D dataset shaped ``(nt,)`` inside *group*."""
    if name in group:
        return group[name]
    return group.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype)


# ---------------------------------------------------------------------------
# Result appending
# ---------------------------------------------------------------------------

#: Maps HDF5 dataset names to the corresponding Fields attributes.
_FIELD_MAP: dict[str, str] = {
    "fluid_pressure": "fluid_pressure",
    "stress_normal": "sn",
    "stress_shear": "tau",
    "aperture": "w",
    "flow_rate": "q",
    "fluid_velocity": "v",
}


def append_results(filepath: Path | str, fields: Fields, t: np.float64) -> None:
    """
    Append one time-step of results to *filepath*.

    The function is idempotent for a given *t*: if that timestamp already
    exists in the file the call is a no-op.  X-coordinate arrays are assumed
    to be stored elsewhere in the file.
    """
    with h5py.File(filepath, "a") as f:
        g_coords = f.require_group("coordinates")
        ds_t = cast(h5py.Dataset, _require_scalar_series(g_coords, "t"))

        # Skip duplicate timestamps
        if ds_t.shape[0] > 0 and np.any(np.isclose(ds_t[()], t)):
            return

        # Append timestamp
        k = ds_t.shape[0]
        ds_t.resize((k + 1,))
        ds_t[k] = t

        # Append each field profile as a new row
        g_fields = f.require_group("fields")
        for dataset_name, attr_name in _FIELD_MAP.items():
            arr: np.ndarray = getattr(fields, attr_name)
            ds = cast(
                h5py.Dataset, _require_profile_1d(g_fields, dataset_name, len(arr))
            )
            ds.resize((k + 1, ds.shape[1]))
            ds[k, :] = arr


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    out_file = Path(it.fish.get("filepath"))

    mesh_ids = MeshIDs(out_file)
    fields = Fields(mesh_ids)
    t = np.float64(it.fish.get("t"))

    append_results(out_file, fields, t)


if __name__ == "__main__":
    main()
