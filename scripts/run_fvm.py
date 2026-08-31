from pathlib import Path

import h5py
import numpy as np
from run_simulation import run_fvm_code


class Param:
    def __init__(self, value: float, unit: str, description: str) -> None:
        self.value = value
        self.unit = unit
        self.description = description


def save_results_hdf5(
    filepath: Path,
    x_fvm: np.ndarray,
    t: np.ndarray,
    p_tx: np.ndarray,
    w_tx: np.ndarray,
    parameters: dict[str, Param],
) -> None:
    # ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(filepath), "w") as f:
        # Coordinates
        g_coords = f.create_group("coordinates")
        ds_xp = g_coords.create_dataset("x_fvm", data=x_fvm)
        ds_t = g_coords.create_dataset("t", data=t)
        ds_xp.attrs["unit"] = "m"
        ds_t.attrs["unit"] = "s"

        # Fields
        g_fields = f.create_group("fields")
        ds_p = g_fields.create_dataset(
            "pressure", data=p_tx, compression="gzip", compression_opts=4
        )
        ds_w = g_fields.create_dataset(
            "aperture", data=w_tx, compression="gzip", compression_opts=4
        )
        ds_p.attrs["unit"] = "Pa"
        ds_w.attrs["unit"] = "m"

        # Parameters
        g_params = f.create_group("parameters")
        for name, p in parameters.items():
            dset = g_params.create_dataset(name, data=p.value)
            dset.attrs["unit"] = p.unit
            dset.attrs["description"] = p.description


def main():
    parameters: dict[str, Param] = {
        "k_n": Param(50e9, "Pa/m", "Normal stiffness"),
        "L": Param(1000.0, "m", "Fracture length"),
        "mu": Param(1e-3, "Pa.s", "Fluid viscosity"),
        "w_i": Param(1e-4, "m", "Initial aperture"),
        "T": Param(10000.0, "s", "Duration"),
        "q_0": Param(1e-5, "m^2/s", "Applied injection rate"),
        "Nx_p": Param(10000, "-", "Number of spatial cells for fvm code"),
        "Nt": Param(20000, "-", "Number of time steps"),
    }

    L = float(parameters["L"].value)
    k_n = float(parameters["k_n"].value)
    mu = float(parameters["mu"].value)
    w_i = float(parameters["w_i"].value)
    q_0 = float(parameters["q_0"].value)
    T = float(parameters["T"].value)
    Nx_p = int(parameters["Nx_p"].value)
    Nt = int(parameters["Nt"].value)

    FVM_result = run_fvm_code(
        L=L, k_n=k_n, mu=mu, w_i=w_i, T=T, q=q_0, Nx=Nx_p, Nt=Nt
    )

    out_dir = Path.cwd() / "results" / "fvm" / "runs"
    out_file = out_dir / f"run-L-{L:.0f}.hdf5"
    # store stress results
    save_results_hdf5(
        filepath=out_file,
        x_fvm=FVM_result.x,
        t=FVM_result.t,
        p_tx=FVM_result.p,
        w_tx=FVM_result.w,
        parameters=parameters,
    )


if __name__ == "__main__":
    main()
