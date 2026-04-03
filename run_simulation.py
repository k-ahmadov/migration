import numpy as np
import h5py

from pathlib import Path

import mysolvers.aperture_solver as aperture_solver
import mysolvers.elastic_solution as elastic_solution


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
        p_tx: np.ndarray,
        w_tx: np.ndarray,
    ) -> None:
        if x.ndim != 1 or t.ndim != 1:
            raise ValueError("x and t must be 1D arrays")
        if p_tx.ndim != 2 or w_tx.ndim != 2:
            raise ValueError("p_tx and w_tx must be 2D arrays")
        Nx = x.size
        Nt = t.size
        if p_tx.shape != (Nt, Nx):
            raise ValueError(f"p_tx must have shape ({Nt}, {Nx})")
        if w_tx.shape != (Nt, Nx):
            raise ValueError(f"w_tx must have shape ({Nt}, {Nx})")
        self.x = x
        self.t = t
        self.p_tx = p_tx
        self.w_tx = w_tx


class ElasticResults:
    def __init__(
        self,
        x: np.ndarray,
        t: np.ndarray,
        sn_tx: np.ndarray,
    ) -> None:
        if x.ndim != 1 or t.ndim != 1:
            raise ValueError("x and t must be 1D arrays")
        if sn_tx.ndim != 2:
            raise ValueError("sn_tx must be a 2D array")
        Nx = x.size
        Nt = t.size
        if sn_tx.shape != (Nt, Nx):
            raise ValueError(f"sn_tx must have shape ({Nt}, {Nx})")
        self.x = x
        self.t = t
        self.sn_tx = sn_tx


def _dimensionalize(
    *, L: float, k_n: float, mu: float, q_0: float
) -> tuple[float, float]:
    """Return characteristic aperture (w_char) and time (t_char)."""
    if L <= 0 or mu <= 0:
        raise ValueError("L and mu must be positive.")
    coefficient_a = k_n / (12.0 * mu)
    w_char = (L * q_0 / coefficient_a) ** 0.25
    t_char = (L * L) / (coefficient_a * (w_char**3))
    return w_char, t_char


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

    w_char, t_char = _dimensionalize(L=L, k_n=k_n, mu=mu, q_0=q_0)

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
    dt = T / Nt
    t = np.arange(Nt, dtype=np.float64) * dt

    return FVMResults(x=x, t=t, p_tx=p_tx, w_tx=w_tx)


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
    return ElasticResults(x=x_out[mask], t=t, sn_tx=sn_tx[:, mask])


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

    with h5py.File(filepath, "w") as f:
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
            "pressure", data=p_tx, compression="gzip", compression_opts=4
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
            dset = g_params.create_dataset(name, data=p.value)
            dset.attrs["unit"] = p.unit
            dset.attrs["description"] = p.description


def main() -> None:
    parameters: dict[str, Param] = {
        "k_n": Param(50e9, "Pa/m", "Normal stiffness"),
        "L": Param(100.0, "m", "Fracture length"),
        "mu": Param(1e-3, "Pa.s", "Fluid viscosity"),
        "w_i": Param(1e-4, "m", "Initial aperture"),
        "T": Param(50.0, "s", "Duration"),
        "q_0": Param(1e-3, "m^2/s", "Applied injection rate"),
        "E": Param(60e9, "Pa/m", "Young's modulus"),
        "nu": Param(0.25, "-", "Poisson's ratio"),
        "sn_char": Param(1e6, "Pa", "Characteristic stress"),
        "Nx_p": Param(1000, "-", "Number of spatial cells for fvm code"),
        "Nx_sn": Param(512, "-", "Number of spatial cells for elastic solution"),
        "Nt": Param(400, "-", "Number of time steps"),
    }

    L = float(parameters["L"].value)
    k_n = float(parameters["k_n"].value)
    mu = float(parameters["mu"].value)
    w_i = float(parameters["w_i"].value)
    E = float(parameters["E"].value)
    nu = float(parameters["nu"].value)
    sn_char = float(parameters["sn_char"].value)
    T = float(parameters["T"].value)
    Nx_p = int(parameters["Nx_p"].value)
    Nx_sn = int(parameters["Nx_sn"].value)
    Nt = int(parameters["Nt"].value)
    q_0 = float(parameters["q_0"].value)

    FVM_result = run_fvm_code(
        L=L, k_n=k_n, mu=mu, w_i=w_i, T=T, q_0=q_0, Nx=Nx_p, Nt=Nt
    )
    print("FVM simulation finished")

    Elastic_result = run_elastic_solution(
        E=E,
        nu=nu,
        k_n=k_n,
        L=L,
        sn_char=sn_char,
        Nx_sn=Nx_sn,
        T=T,
        x_fvm=FVM_result.x,
        p_tx=FVM_result.p_tx,
    )
    print("Elastic half-space solution finished")

    out_dir = Path.cwd() / "results" / "fvm-elastic" / "runs"
    out_file = out_dir / f"run-L-{q_0:.0e}.hdf5"
    save_results_hdf5(
        filepath=out_file,
        x_fvm=FVM_result.x,
        t=FVM_result.t,
        p_tx=FVM_result.p_tx,
        w_tx=FVM_result.w_tx,
        x_elastic=Elastic_result.x,
        sn_tx=Elastic_result.sn_tx,
        parameters=parameters,
    )
    print(f"Results stored in {out_file}")

if __name__ == "__main__":
    main()
