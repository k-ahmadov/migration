import h5py
import itasca as it
import numpy as np
from scripts.helpers import apply_function_to_iterable
from pathlib import Path


class Fields:
    def __init__(self):
        # --- Subcontact fields ---
        self.sn_eff = apply_function_to_iterable(
            func=it.block.subcontact.Subcontact.stress_norm,
            iterable=it.block.subcontact.list(),
            count=it.block.subcontact.count(),
        )

        self.pore_pressure = apply_function_to_iterable(
            func=it.block.subcontact.Subcontact.pp,
            iterable=it.block.subcontact.list(),
            count=it.block.subcontact.count(),
        )

        self.sn = self.sn_eff + self.pore_pressure

        self.tau = apply_function_to_iterable(
            func=it.block.subcontact.Subcontact.stress_shear,
            iterable=it.block.subcontact.list(),
            count=it.block.subcontact.count(),
        )

        # --- Flowplane vertex fields ---
        self.w = apply_function_to_iterable(
            func=it.flowplane.vertex.Vertex.aperture_hydraulic,
            iterable=it.flowplane.vertex.list(),
            count=it.flowplane.vertex.count(),
        )

        # --- Flowplane zone fields ---
        self.q = apply_function_to_iterable(
            func=it.flowplane.zone.Zone.discharge_x,
            iterable=it.flowplane.zone.list(),
            count=it.flowplane.zone.count(),
        )

        self.v = apply_function_to_iterable(
            func=it.flowplane.zone.Zone.velocity_x,
            iterable=it.flowplane.zone.list(),
            count=it.flowplane.zone.count(),
        )


def _require_profile_1d(group, name: str, n: int, dtype=np.float64):
    """
    Create (or get) an extendable dataset shaped (nt, n) for 1D profiles.
    nt is not an input, can be arbitrary value
    """
    if name in group:
        ds = group[name]
        if ds.shape[1] != n:
            raise ValueError(f"{name}: existing n={ds.shape[1]} but new n={n}")
        return ds

    return group.create_dataset(
        name,
        shape=(0, n),
        maxshape=(None, n),
        dtype=dtype,
        compression="gzip",
        compression_opts=4,
    )


def _require_scalar(group, name: str, dtype=np.float64):
    """
    create (or get) an extendable dataset shaped (nt,) for time.
    """
    if name in group:
        return group[name]

    return group.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype)


def append_results_hdf5(filepath: str | Path, fields: Fields, t: float):
    """
    Call this every given time interval. It appends results of one timestep (time + profiles).
    Assumes x-coordinate arrays are already stored elsewhere in the file.
    """
    with h5py.File(filepath, "a") as f:
        g_coords = f.require_group("coordinates")
        ds_t = _require_scalar(g_coords, "t")

        g_fields = f.require_group("fields")
        ds_p = _require_profile_1d(
            g_fields, "fluid_pressure", len(fields.pore_pressure)
        )
        ds_sn = _require_profile_1d(g_fields, "stress_normal", len(fields.sn))
        ds_tau = _require_profile_1d(g_fields, "stress_shear", len(fields.tau))
        ds_w = _require_profile_1d(g_fields, "aperture", len(fields.w))
        ds_q = _require_profile_1d(g_fields, "flow_rate", len(fields.q))
        ds_v = _require_profile_1d(g_fields, "fluid_velocity", len(fields.v))

        k = ds_t.shape[0]

        # resize to add one row every callback
        ds_t.resize((k + 1,))
        ds_t[k] = t

        # assign obtained profile to new row
        for ds, arr in [
            (ds_p, fields.pore_pressure),
            (ds_sn, fields.sn),
            (ds_tau, fields.tau),
            (ds_w, fields.w),
            (ds_q, fields.q),
            (ds_v, fields.v),
        ]:
            ds.resize((k + 1, ds.shape[1]))
            ds[k, :] = arr


def main():
    out_dir = Path.cwd().parent / "results" / "3dec" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "run-q-1e-03.hdf5"
    fields = Fields()
    t = np.float64(it.fish.get("fluid_time"))
    append_results_hdf5(out_file, fields, t)


if __name__ == "__main__":
    main()
