"""Run the FVM aperture solve + elastic half-space solve and store the result.

python scripts/run_simulation.py            # single run, constant rate
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from fracinj import paths, physics
from fracinj.io import save_run
from fracinj.solvers import aperture, elastic
from fracinj.types import Field, Parameters, Vector


class RateBC(str, Enum):
    """Left boundary condition; the value is the driving Parameters field."""

    CONST = "q_0"
    RAMP = "m_q"

    @property
    def dimensionalize_key(self) -> physics.RateBC:
        return "const_rate" if self is RateBC.CONST else "ramp_rate"


@dataclass(frozen=True)
class FVMResult:
    x: Vector
    t: Vector
    p: Field
    w: Field


@dataclass(frozen=True)
class ElasticResult:
    x: Vector
    t: Vector
    sn: Field


def run_fvm(params: Parameters, *, bc: RateBC = RateBC.CONST) -> FVMResult:
    nx, nt = int(params.Nx_p), int(params.Nt)
    if nx <= 0 or nt <= 0 or params.T < 0:
        raise ValueError("need Nx_p > 0, Nt > 0, T >= 0")

    scales = physics.dimensionalize(params, bc.dimensionalize_key)
    w_hat = aperture.solve_diffusion(
        num_nodes=nx,
        num_steps=nt,
        w_initial=params.w_i / scales.w,
        t_final=params.T / scales.t,
        k_func=lambda w: w**3,
        left_bc_constant_rate=(bc is RateBC.CONST),
    )
    w = w_hat * scales.w
    return FVMResult(
        x=np.linspace(0, params.L, nx),
        t=np.linspace(params.T / (nt - 1), params.T, nt),
        p=params.k_n * (w - params.w_i),
        w=w,
    )


class _InterpRHS:
    """Stateful forcing term for ``elastic.FIE_log_sing`` (pressure on [-1, 1])."""

    def __init__(self, x_mirror: Vector, L: float) -> None:
        self.x_mirror = x_mirror
        self.L = L
        self.y: Vector | None = None

    def __call__(self, s: np.ndarray) -> np.ndarray:
        assert self.y is not None
        return np.interp(s * self.L, self.x_mirror, self.y)


def run_elastic(
    params: Parameters, x_fvm: Vector, p_tx: Field, bc: RateBC = RateBC.CONST
) -> ElasticResult:
    nt = int(p_tx.shape[0])
    if x_fvm.ndim != 1 or p_tx.ndim != 2 or p_tx.shape[1] != x_fvm.size:
        raise ValueError("expected p_tx of shape (Nt, Nx) with Nx == x_fvm.size")

    E_plane = physics.plane_strain_modulus(params)
    lam = (4.0 / np.pi) * params.k_n * params.L / E_plane
    sn_char = physics.dimensionalize(params, bc=bc.dimensionalize_key).sn

    x_mirror = np.concatenate((-x_fvm[:0:-1], x_fvm))  # avoid double-counting x=0
    rhs = _InterpRHS(x_mirror, params.L)

    def set_pressure(p_x: np.ndarray) -> None:
        rhs.y = np.concatenate((p_x[:0:-1], p_x)) / sn_char

    set_pressure(p_tx[0])
    x_hat, sn_hat, _ = elastic.FIE_log_sing(lam, rhs, int(params.Nx_sn))
    x_out = x_hat * params.L

    sn_tx = np.empty((nt, sn_hat.size))
    sn_tx[0] = sn_hat * sn_char
    for i in range(1, nt):
        set_pressure(p_tx[i])
        _, sn_hat, _ = elastic.FIE_log_sing(lam, rhs, int(params.Nx_sn))
        sn_tx[i] = sn_hat * sn_char

    t = np.arange(nt, dtype=np.float64) * (params.T / nt)
    keep = x_out >= 0
    return ElasticResult(x=x_out[keep], t=t, sn=sn_tx[:, keep])


def run_simulation(
    params: Parameters, out_filepath: Path, *, bc: RateBC = RateBC.CONST
) -> None:
    if getattr(params, bc.value) == 0.0:
        raise ValueError(f"Parameters.{bc.value} must be set for bc={bc.name}")

    params.T = physics.dimensionalize(params, bc.dimensionalize_key).t
    print(f"Simulation duration: {params.T:.3g} s")

    fvm = run_fvm(params, bc=bc)
    print("FVM simulation finished")

    el = run_elastic(params, fvm.x, fvm.p, bc=bc)
    print("Elastic half-space solution finished")

    save_run(
        out_filepath,
        t=fvm.t,
        x_vert=fvm.x,
        w=fvm.w,
        p=fvm.p,
        x_sc=el.x,
        sn=el.sn,
        params=params,
    )
    print(f"Results stored in {out_filepath}")


def sweep(
    params: Parameters,
    field: str,
    values: np.ndarray,
    out_dir: Path,
    *,
    bc: RateBC = RateBC.CONST,
) -> None:
    """Run one simulation per value of ``params.<field>``."""
    for value in values:
        setattr(params, field, float(value))
        print(f"running {field}={value:.1e}")
        run_simulation(params, out_dir / f"{field}-{value:.0e}.hdf5", bc=bc)


def default_parameters(bc: RateBC = RateBC.CONST) -> Parameters:
    params = Parameters(
        k_n=50e9,
        L=100.0,
        mu=1e-3,
        w_i=1e-4,
        E=60e9,
        nu=0.25,
        Nx_p=1024,
        Nx_sn=512,
        Nt=500,
    )
    setattr(params, bc.value, 1e-6)
    return params


def main() -> None:
    bc = RateBC.RAMP
    params = default_parameters(bc)
    out_dir = paths.results_dir("halfspace", "linear")

    out_file = out_dir / f"{bc.value}-{getattr(params, bc.value):.0e}.hdf5"
    run_simulation(params, out_file, bc=bc)


if __name__ == "__main__":
    main()
