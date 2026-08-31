from pathlib import Path
from typing import Any, Collection, cast

import h5py
import numpy as np

import mysolvers.aperture_solver as aperture_solver
import mysolvers.elastic_solution as elastic_solution
from mypackages import physics
from mypackages.typesdefs import Parameters


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
        Nt = t.size - 1
        if p.shape != (Nt, Nx):
            raise ValueError(f"p_tx must have shape ({Nt}, {Nx}), but it is {p.shape}")
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


BC_RATE_PARAM_KEY = {
    True: "q_0",   # constant rate
    False: "m_q",  # linearly ramped rate
}


def _rate_param_key(left_bc_constant_rate: bool) -> str:
    return BC_RATE_PARAM_KEY[left_bc_constant_rate]


def run_fvm_code(
    *,
    L: float,
    k_n: float,
    mu: float,
    w_i: float,
    T: float,
    rate: float,
    Nx: int,
    Nt: int,
    left_bc_constant_rate: bool = True,
) -> FVMResults:
    if Nx <= 0 or Nt <= 0:
        raise ValueError("Nx and Nt must be positive integers.")
    if T < 0:
        raise ValueError("T must be non-negative.")

    w_char, t_char = physics.dimensionalize(
        Parameters(L=L, mu=mu, k_n=k_n, _rate_param_key(left_bc_constant_rate)=rate), left_bc_constant_rate=left_bc_constant_rate
    )

    # TODO: run a simulation with linearly increasing injection rate
    w_hat_tx = aperture_solver.solve_diffusion(
        num_nodes=Nx,
        num_steps=Nt,
        w_initial=w_i / w_char,
        t_final=T / t_char,
        k_func=lambda w: w**3,
        left_bc_constant_rate=left_bc_constant_rate,
    )

    # dimensionalize + pressure
    w_tx = w_hat_tx * w_char
    p_tx = k_n * (w_tx - w_i)

    # grids (node (vertex)-centered)
    x = np.linspace(0, L, Nx)
    t = np.linspace(T / (Nt - 1), T, Nt)

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
    q_0: float,
    mu: float,
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
    a = k_n / (12 * mu)
    sn_char = E_eff * (q_0 / (L**3 * a)) ** (1 / 4)

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


def run_simulation(
    parameters: dict[str, Param],
    out_filepath: Path,
    left_bc_constant_rate: bool = True,
) -> None:
    rate_key = _rate_param_key(left_bc_constant_rate)
    if rate_key not in parameters:
        raise KeyError(
            f"parameters must contain '{rate_key}' when "
            f"left_bc_constant_rate={left_bc_constant_rate}"
        )
    rate = parameters[rate_key].value

    # find characteristic duration for the simulation
    _, T = _dimensionalize(
        L=parameters["L"].value,
        mu=parameters["mu"].value,
        k_n=parameters["k_n"].value,
        rate=rate,
        left_bc_constant_rate=left_bc_constant_rate,
    )

    parameters["T"] = Param(T, "s", "Duration")
    print(f"Simulation duration: {T}")

    FVM_result = run_fvm_code(
        L=parameters["L"].value,
        k_n=parameters["k_n"].value,
        mu=parameters["mu"].value,
        w_i=parameters["w_i"].value,
        T=parameters["T"].value,
        rate=rate,
        Nx=int(parameters["Nx_p"].value),
        Nt=int(parameters["Nt"].value),
        left_bc_constant_rate=left_bc_constant_rate,
    )
    print("FVM simulation finished")

    # NOTE: run_elastic_solution's characteristic stress scale (sn_char) is
    # currently derived assuming a constant rate q_0. If you run the ramp-rate
    # mode, double check whether sn_char should instead depend on m_q here.
    Elastic_result = run_elastic_solution(
        E=parameters["E"].value,
        nu=parameters["nu"].value,
        k_n=parameters["k_n"].value,
        L=parameters["L"].value,
        q_0=rate,
        mu=parameters["mu"].value,
        Nx_sn=int(parameters["Nx_sn"].value),
        T=parameters["T"].value,
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


def run_multiple_simuls_q0(parameters: dict, out_dirpath: Path):
    q_0_min = 1e-4
    q_0_max = 1e-10
    q_0_values = np.geomspace(q_0_min, q_0_max, 7)
    for q_0 in q_0_values:
        print(f"running q_0={q_0:.0e}")
        parameters["q_0"] = Param(q_0, "m^2/s", "Applied injection rate")
        out_file = out_dirpath / f"q0-{parameters['q_0'].value:.0e}.hdf5"
        run_simulation(parameters, out_file)


def make_default_parameters(left_bc_constant_rate: bool = True) -> dict[str, Param]:
    """Build the default parameter dict, using q_0 or m_q depending on the
    chosen left boundary condition."""
    parameters: dict[str, Param] = {
        "k_n": Param(200e9, "Pa/m", "Normal stiffness"),
        "L": Param(100.0, "m", "Fracture length"),
        "mu": Param(1e-3, "Pa.s", "Fluid viscosity"),
        "w_i": Param(1e-5, "m", "Initial aperture"),
        "E": Param(60e9, "Pa", "Young's modulus"),
        "nu": Param(0.25, "-", "Poisson's ratio"),
        "Nx_p": Param(1024, "-", "Number of spatial cells for fvm code"),
        "Nx_sn": Param(512, "-", "Number of spatial cells for elastic solution"),
        "Nt": Param(500, "-", "Number of time steps"),
    }

    if left_bc_constant_rate:
        parameters["q_0"] = Param(5e-7, "m^2/s", "Applied injection rate")
    else:
        parameters["m_q"] = Param(5e-7, "m^2/s^2", "Injection rate ramp slope")

    return parameters


def main() -> None:
    left_bc_constant_rate = True
    parameters = make_default_parameters(left_bc_constant_rate)

    multi_simul = False
    rate_key = _rate_param_key(left_bc_constant_rate)

    if multi_simul:
        out_dir = Path.cwd() / "results" / "halfspace" / "wi-1e-05"
        run_multiple_simuls_q0(parameters, out_dir)

    out_dir = Path.cwd() / "results" / "halfspace" / "wi-1e-05"
    out_file = out_dir / f"{rate_key}-{parameters[rate_key].value:.0e}.hdf5"
    run_simulation(parameters, out_file, left_bc_constant_rate=left_bc_constant_rate)


if __name__ == "__main__":
    main()
