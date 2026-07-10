from pathlib import Path
from typing import Any, Collection, cast

import h5py
import numpy as np

import mysolvers.aperture_solver as aperture_solver
import mysolvers.elastic_solution as elastic_solution
from mypackages import physics, types


class Param:
    def __init__(self, value: float, unit: str, description: str) -> None:
        self.value = value
        self.unit = unit
        self.description = description


class FVMResults:
    def __init__(
        self,
        x: np.ndarray,
        t: np.ndarray,
        p: np.ndarray,
        w: np.ndarray,
    ) -> None:
        if x.ndim != 1 or t.ndim != 1:
            raise ValueError("x and t must be 1D arrays")
        if p.ndim != 2 or w.ndim != 2:
            raise ValueError("p_tx and w_tx must be 2D arrays")
        Nx = x.size
        Nt = t.size
        if p.shape != (Nt, Nx):
            raise ValueError(f"p_tx must have shape ({Nt}, {Nx})")
        if w.shape != (Nt, Nx):
            raise ValueError(f"w_tx must have shape ({Nt}, {Nx})")
        self.x = x
        self.t = t
        self.p = p
        self.w = w


class ElasticResults:
    def __init__(
        self,
        x: np.ndarray,
        t: np.ndarray,
        sn: np.ndarray,
    ) -> None:
        if x.ndim != 1 or t.ndim != 1:
            raise ValueError("x and t must be 1D arrays")
        if sn.ndim != 2:
            raise ValueError("sn_tx must be a 2D array")
        Nx = x.size
        Nt = t.size
        if sn.shape != (Nt, Nx):
            raise ValueError(f"sn_tx must have shape ({Nt}, {Nx})")
        self.x = x
        self.t = t
        self.sn = sn


def run_fvm_code(
    *,
    L: float,
    k_n: float,
    mu: float,
    w_i: float,
    T: float,
    q_0: float,
    Nx: int,
    Nt: int,
) -> FVMResults:
    if Nx <= 0 or Nt <= 0:
        raise ValueError("Nx and Nt must be positive integers.")
    if T < 0:
        raise ValueError("T must be non-negative.")

    w_char, t_char = physics.dimensionalize(
        types.Parameters(L=L, mu=mu, k_n=k_n, q_0=q_0)
    )

    w_hat_tx = aperture_solver.solve_dimless_nonlinear_diffusion_n3_constant_flux(
        Nx=Nx,
        Nt=Nt,
        ui_hat=w_i / w_char,
        T_hat=T / t_char,
    )

    # dimensionalize + pressure
    w_tx = w_hat_tx * w_char
    p_tx = k_n * (w_tx - w_i)

    # grids (cell centers in x, uniform time steps)
    dx = L / Nx
    x = (np.arange(Nx, dtype=np.float64) + 0.5) * dx
    # dt = T / Nt
    # t = np.arange(Nt, dtype=np.float64) * dt
    t = np.linspace(0, T, Nt, dtype=np.float64)

    return FVMResults(x=x, t=t, p=p_tx, w=w_tx)


class InterpRHS:
    def __init__(self, x: np.ndarray, L: float) -> None:
        self.x = x
        self.L = L
        self.y: np.ndarray | None = None  # set per step

    def set_y(self, y: np.ndarray) -> None:
        self.y = y

    def __call__(self, s: np.ndarray) -> np.ndarray:
        assert self.y is not None
        return np.interp(s * self.L, self.x, self.y)


def run_elastic_solution(
    *,
    E: float,
    nu: float,
    k_n: float,
    L: float,
    sn_char: float,
    Nx_sn: int,
    T: float,
    x_fvm: np.ndarray,
    p_tx: np.ndarray,
) -> ElasticResults:
    Nt = int(p_tx.shape[0])
    if Nt <= 0:
        raise ValueError("p_tx must have at least one time step (Nt > 0).")
    if x_fvm.ndim != 1 or p_tx.ndim != 2 or p_tx.shape[1] != x_fvm.size:
        raise ValueError("Expected p_tx shape (Nt, Nx) with Nx == x_fvm.size.")

    # material + nondimensional parameter
    E_eff = E / (1.0 - nu**2)
    lam = (4.0 / np.pi) * k_n * L / E_eff

    # mirrored grid (avoid double-counting x=0)
    x_mirror = np.concatenate((-x_fvm[:0:-1], x_fvm))
    left_n = x_fvm.size - 1

    rhs = InterpRHS(x=x_mirror, L=L)
    p_mirror = np.empty_like(x_mirror, dtype=np.float64)

    def set_mirrored_pressure(p_x: np.ndarray) -> None:
        # left side: reversed excluding p_x[0]; right side: original
        p_mirror[:left_n] = p_x[:0:-1]
        p_mirror[left_n:] = p_x
        rhs.set_y(p_mirror / sn_char)

    # do first step once to get x grid, then fill the output array
    set_mirrored_pressure(p_tx[0])
    x_hat, sn_hat, _ = elastic_solution.FIE_log_sing(lam, rhs, Nx_sn)

    x_out = x_hat * L
    sn_tx = np.empty((Nt, sn_hat.size), dtype=np.float64)
    sn_tx[0] = sn_hat * sn_char

    for idx_t in range(1, Nt):
        set_mirrored_pressure(p_tx[idx_t])
        _, sn_hat, _ = elastic_solution.FIE_log_sing(lam, rhs, Nx_sn)
        sn_tx[idx_t] = sn_hat * sn_char

    # grids
    dt = T / Nt
    t = np.arange(Nt, dtype=np.float64) * dt

    # keep x >= 0 half
    mask = x_out >= 0
    return ElasticResults(x=x_out[mask], t=t, sn=sn_tx[:, mask])


def run_simulation(parameters: dict, out_filepath: Path):
    # find characteristic duration for the simulation
    T = physics.dimensionalize(
        types.Parameters(
            k_n=parameters["k_n"].value,
            mu=parameters["mu"].value,
            q_0=parameters["q_0"].value,
            L=parameters["L"].value,
        )
    )[1]
    parameters["T"] = Param(T, "s", "Duration")
    print(f"Simulation duration: {T}")

    FVM_result = run_fvm_code(
        L=parameters["L"].value,
        k_n=parameters["k_n"].value,
        mu=parameters["mu"].value,
        w_i=parameters["w_i"].value,
        T=parameters["T"].value,
        q_0=parameters["q_0"].value,
        Nx=int(parameters["Nx_p"].value),
        Nt=int(parameters["Nt"].value),
    )
    print("FVM simulation finished")

    Elastic_result = run_elastic_solution(
        E=parameters["E"].value,
        nu=parameters["nu"].value,
        k_n=parameters["k_n"].value,
        L=parameters["L"].value,
        sn_char=parameters["sn_char"].value,
        Nx_sn=int(parameters["Nx_sn"].value),
        T=int(parameters["T"].value),
        x_fvm=FVM_result.x,
        p_tx=FVM_result.p,
    )
    print("Elastic half-space solution finished")

    save_results_hdf5(
        filepath=out_filepath,
        x_fvm=FVM_result.x,
        t=FVM_result.t,
        p_tx=FVM_result.p,
        w_tx=FVM_result.w,
        x_elastic=Elastic_result.x,
        sn_tx=Elastic_result.sn,
        parameters=parameters,
    )
    print(f"Results stored in {out_filepath}")


def save_results_hdf5(
    filepath: Path,
    x_fvm: np.ndarray,
    t: np.ndarray,
    p_tx: np.ndarray,
    w_tx: np.ndarray,
    x_elastic: np.ndarray,
    sn_tx: np.ndarray,
    parameters: dict[str, Param],
) -> None:
    # ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(filepath), "w") as f:
        # Coordinates
        g_coords = f.create_group("coordinates")
        ds_xp = g_coords.create_dataset("x_fvm", data=x_fvm)
        ds_xs = g_coords.create_dataset("x_elastic", data=x_elastic)
        ds_t = g_coords.create_dataset("t", data=t)
        ds_xp.attrs["unit"] = "m"
        ds_xs.attrs["unit"] = "m"
        ds_t.attrs["unit"] = "s"

        # Fields
        g_fields = f.create_group("fields")
        ds_p = g_fields.create_dataset(
            "fluid_pressure", data=p_tx, compression="gzip", compression_opts=4
        )
        ds_w = g_fields.create_dataset(
            "aperture", data=w_tx, compression="gzip", compression_opts=4
        )
        ds_sn = g_fields.create_dataset(
            "stress_normal", data=sn_tx, compression="gzip", compression_opts=4
        )
        ds_p.attrs["unit"] = "Pa"
        ds_sn.attrs["unit"] = "Pa"
        ds_w.attrs["unit"] = "m"

        # Parameters
        g_params = f.create_group("parameters")
        for name, p in parameters.items():
            dset = g_params.create_dataset(name, data=cast(Collection[Any], p.value))
            dset.attrs["unit"] = p.unit
            dset.attrs["description"] = p.description


def run_multiple_simuls(parameters: dict, out_dirpath: Path):
    k_n_min = 10e9
    k_n_max = 10000e9
    k_n_values = np.linspace(k_n_min, k_n_max, 10)
    for k_n in k_n_values:
        parameters["k_n"] = Param(k_n, "GPa/m", "Normal Stiffness")
        out_file = out_dirpath / f"run-kn-{parameters['k_n'].value:.1e}.hdf5"
        run_simulation(parameters, out_file)


def main() -> None:
    parameters: dict[str, Param] = {
        "k_n": Param(200e9, "Pa/m", "Normal stiffness"),
        "L": Param(100.0, "m", "Fracture length"),
        "mu": Param(1e-3, "Pa.s", "Fluid viscosity"),
        "w_i": Param(5e-5, "m", "Initial aperture"),
        "q_0": Param(1e-4, "m^2/s", "Applied injection rate"),
        "E": Param(60e9, "Pa", "Young's modulus"),
        "nu": Param(0.25, "-", "Poisson's ratio"),
        "sn_char": Param(1e6, "Pa", "Characteristic stress"),
        "Nx_p": Param(256, "-", "Number of spatial cells for fvm code"),
        "Nx_sn": Param(512, "-", "Number of spatial cells for elastic solution"),
        "Nt": Param(500, "-", "Number of time steps"),
    }

    # out_dir = Path.cwd() / "results" / "fvm-elastic" / "wi-1e-05"
    # out_file = out_dir / f"run-q-{parameters['q_0'].value:.0e}.hdf5"
    # run_simulation(parameters, out_file)

    out_dir = Path.cwd() / "results" / "fvm-elastic" / "for-V-T" / "mixed"
    run_multiple_simuls(parameters, out_dir)


if __name__ == "__main__":
    main()
